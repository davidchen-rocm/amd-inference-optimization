#!/usr/bin/env python3
"""Prepare, explicitly apply, or roll back a guarded immutable frontend release.

The default prepare command is read-only against Kubernetes. No credentials are
copied into the release directory. Supply an existing kubeconfig on the command line.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
ASSET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.(?:js|css|html|svg|json)\Z")
NGINX_ANCHOR = "        location ^~ /api/ {"
TRAINING_ERROR = json.dumps(
    {
        "error": {
            "code": "training_unavailable",
            "message": "The training connection is unavailable. Please retry.",
            "retryable": True,
        }
    },
    separators=(",", ":"),
)
TRAINING_BLOCK = """        # BEGIN MACFIT TRAINING PROXY
        location ^~ /api/training/ {
            proxy_pass http://macfit-training-bridge:8080;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header Authorization $http_authorization;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            client_max_body_size 3m;
            client_body_timeout 20s;
            proxy_connect_timeout 3s;
            proxy_read_timeout 180s;
            proxy_send_timeout 20s;
            proxy_buffering off;
            proxy_intercept_errors on;
            error_page 502 504 = @training_unavailable;
            add_header Cache-Control "no-store" always;
            add_header X-Content-Type-Options nosniff always;
            add_header Referrer-Policy strict-origin-when-cross-origin always;
            add_header X-Frame-Options SAMEORIGIN always;
        }
        location @training_unavailable {
            default_type application/json;
            add_header Cache-Control "no-store" always;
            add_header X-Content-Type-Options nosniff always;
            add_header Referrer-Policy strict-origin-when-cross-origin always;
            add_header X-Frame-Options SAMEORIGIN always;
            return 503 '__TRAINING_ERROR__';
        }
        # END MACFIT TRAINING PROXY
""".replace("__TRAINING_ERROR__", TRAINING_ERROR)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def volume_map(deployment: dict) -> dict:
    volumes = deployment["spec"]["template"]["spec"]["volumes"]
    if len({volume["name"] for volume in volumes}) != len(volumes):
        raise ValueError("Duplicate Deployment volume names")
    return {volume["name"]: (index, volume) for index, volume in enumerate(volumes)}


def with_training_proxy(nginx: str) -> str:
    if TRAINING_BLOCK in nginx:
        if nginx.count(TRAINING_BLOCK) != 1:
            raise ValueError("Duplicate training proxy blocks")
        return nginx
    if "/api/training/" in nginx or "@training_unavailable" in nginx:
        raise ValueError("An existing training proxy needs explicit review")
    if nginx.count(NGINX_ANCHOR) != 1:
        raise ValueError("Expected a single existing catalog API location")
    return nginx.replace(NGINX_ANCHOR, TRAINING_BLOCK + NGINX_ANCHOR, 1)


def read_source(source: Path) -> dict:
    data = {}
    for path in sorted(source.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Source must contain only regular, flat public asset files")
        if path.name != "nginx.conf" and not ASSET_NAME.fullmatch(path.name):
            raise ValueError("Unexpected non-public frontend file")
        data[path.name] = path.read_text(encoding="utf-8")
    for required in ("index.html", "app.js", "training-client.js", "nginx.conf"):
        if required not in data:
            raise ValueError("Required frontend source asset is missing")
    if "MACFIT_FIREBASE_CONFIG_REQUIRED" in data.get("firebase-config.js", ""):
        raise ValueError(
            "Replace the Firebase configuration template before deployment, "
            "or pass --source pointing to an already configured private working copy"
        )
    return data


def build_plan(deployment: dict, previous: dict, source_data: dict) -> dict:
    volumes = volume_map(deployment)
    old_name = volumes["content"][1]["configMap"]["name"]
    if volumes["nginx"][1]["configMap"]["name"] != old_name:
        raise ValueError("content and nginx must share the current content ConfigMap")
    if previous["metadata"]["name"] != old_name or previous.get("immutable") is not True:
        raise ValueError("The live content ConfigMap must be the expected immutable object")
    old_data = previous["data"]
    candidate_nginx = with_training_proxy(old_data["nginx.conf"])
    # A configured private working copy may still carry the original catalog-only
    # nginx file after the first release. Only remove our exact managed block when
    # comparing; every other live rule must still match byte for byte.
    prior_without_managed_proxy = old_data["nginx.conf"].replace(TRAINING_BLOCK, "")
    if source_data["nginx.conf"] not in (
        old_data["nginx.conf"],
        candidate_nginx,
        prior_without_managed_proxy,
    ):
        raise ValueError("Local nginx differs from live rules; review before preparing")
    data = {**old_data, **source_data, "nginx.conf": candidate_nginx}
    # Preserve unknown live keys; never publish missing files as silent deletions.
    if len(json.dumps(data, ensure_ascii=False).encode()) > 950 * 1024:
        raise ValueError("Frontend ConfigMap is too close to the Kubernetes 1 MiB limit")
    name = "mac-fit-content-" + digest(data)[:12]
    namespace = deployment["metadata"]["namespace"]
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "immutable": True,
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": previous["metadata"].get("labels", {}),
        },
        "data": data,
    }
    patch = [
        {"op": "test", "path": "/metadata/uid", "value": deployment["metadata"]["uid"]},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": deployment["metadata"]["resourceVersion"],
        },
    ]
    before, after = {}, {}
    for key in ("content", "nginx"):
        index, volume = volumes[key]
        path = f"/spec/template/spec/volumes/{index}/configMap"
        old = volume["configMap"]
        new = copy.deepcopy(old)
        new["name"] = name
        if key == "content":
            if not isinstance(new.get("items"), list):
                raise ValueError("Explicit content items are required to keep nginx.conf private")
            present = {item["key"] for item in new["items"]}
            destinations = {item["path"] for item in new["items"]}
            if len(present) != len(new["items"]) or len(destinations) != len(new["items"]):
                raise ValueError("Duplicate content item keys or destinations")
            if "nginx.conf" in present:
                raise ValueError("nginx.conf must not be mounted in the public content directory")
            for filename in sorted(source_data):
                if filename == "nginx.conf" or filename in present:
                    continue
                if filename in destinations:
                    raise ValueError("A new frontend asset would overwrite an existing mount path")
                new["items"].append({"key": filename, "path": filename})
        before[path], after[path] = old, new
        patch.extend(
            [
                {"op": "test", "path": path, "value": old},
                {"op": "replace", "path": path, "value": new},
            ]
        )
    state = {
        "schema": "macfit-frontend-release-v1",
        "namespace": namespace,
        "deployment": deployment["metadata"]["name"],
        "deployment_uid": deployment["metadata"]["uid"],
        "deployment_resource_version": deployment["metadata"]["resourceVersion"],
        "previous": old_name,
        "current": name,
        "previous_configmap_uid": previous["metadata"]["uid"],
        "previous_configmap_resource_version": previous["metadata"]["resourceVersion"],
        "previous_data_sha256": digest(old_data),
        "new_data_sha256": digest(data),
        "before": before,
        "after": after,
        "source_files": sorted(source_data),
        "changed": [key for key in data if data[key] != old_data.get(key)],
    }
    return {"configmap": configmap, "patch": patch, "state": state}


def write_json(path: Path, data) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


class Kubernetes:
    def __init__(self, executable: str, kubeconfig: Path, namespace: str):
        self.command = [executable, "--kubeconfig", str(kubeconfig), "-n", namespace]

    def run(self, *args: str) -> str:
        result = subprocess.run(
            self.command + list(args), text=True, capture_output=True, timeout=240
        )
        if result.returncode:
            # kubectl errors can contain request content. Keep console output bounded and private.
            raise RuntimeError(
                "kubectl operation failed; inspect the command against the saved release"
            )
        return result.stdout

    def get(self, kind: str, name: str) -> dict:
        return json.loads(self.run("get", kind, name, "-o", "json", "--request-timeout=20s"))


def prepare(client: Kubernetes, deployment_name: str, source: Path, release: Path) -> dict:
    deployment = client.get("deployment", deployment_name)
    current = volume_map(deployment)["content"][1]["configMap"]["name"]
    previous = client.get("configmap", current)
    plan = build_plan(deployment, previous, read_source(source))
    release.mkdir(mode=0o700, parents=True, exist_ok=False)
    for filename, value in (
        ("deployment.before.json", deployment),
        ("configmap.before.json", previous),
        ("configmap.new.json", plan["configmap"]),
        ("patch.json", plan["patch"]),
        ("state.json", plan["state"]),
    ):
        write_json(release / filename, value)
    return {
        "release": str(release),
        "previous": plan["state"]["previous"],
        "current": plan["state"]["current"],
        "changed": plan["state"]["changed"],
        "files": len(plan["configmap"]["data"]),
        "deployment_resource_version": plan["state"]["deployment_resource_version"],
        "cluster_mutated": False,
    }


def load_plan(release: Path) -> dict:
    def read(name):
        return json.loads((release / name).read_text(encoding="utf-8"))

    state, configmap = read("state.json"), read("configmap.new.json")
    if state.get("schema") != "macfit-frontend-release-v1":
        raise ValueError("Unsupported frontend release snapshot")
    source_data = {key: configmap["data"][key] for key in state["source_files"]}
    rebuilt = build_plan(read("deployment.before.json"), read("configmap.before.json"), source_data)
    if (
        rebuilt["configmap"] != configmap
        or rebuilt["patch"] != read("patch.json")
        or rebuilt["state"] != state
    ):
        raise ValueError("Prepared frontend release was changed; prepare a new release")
    return rebuilt


def guard_current(client: Kubernetes, state: dict) -> dict:
    deployment = client.get("deployment", state["deployment"])
    previous = client.get("configmap", state["previous"])
    if (
        deployment["metadata"]["uid"] != state["deployment_uid"]
        or deployment["metadata"]["resourceVersion"] != state["deployment_resource_version"]
    ):
        raise ValueError("Deployment changed since preparation; prepare and review again")
    if (
        previous["metadata"]["uid"] != state["previous_configmap_uid"]
        or previous["metadata"]["resourceVersion"] != state["previous_configmap_resource_version"]
        or digest(previous["data"]) != state["previous_data_sha256"]
        or previous.get("immutable") is not True
    ):
        raise ValueError("Previous content ConfigMap changed since preparation")
    for path, expected in state["before"].items():
        if (
            deployment["spec"]["template"]["spec"]["volumes"][int(path.split("/")[-2])]["configMap"]
            != expected
        ):
            raise ValueError("A content volume changed since preparation")
    return deployment


def apply(client: Kubernetes, release: Path) -> dict:
    plan = load_plan(release)
    state = plan["state"]
    guard_current(client, state)
    existing = json.loads(
        client.run("get", "configmap", state["current"], "--ignore-not-found", "-o", "json")
        or "null"
    )
    if existing is None:
        client.run(
            "create", "--dry-run=server", "-f", str(release / "configmap.new.json"), "-o", "name"
        )
        client.run("create", "-f", str(release / "configmap.new.json"))
    elif existing.get("immutable") is not True or existing.get("data") != plan["configmap"]["data"]:
        raise ValueError("Release ConfigMap name already exists with different content")
    client.run(
        "patch",
        "deployment",
        state["deployment"],
        "--type=json",
        "--patch-file",
        str(release / "patch.json"),
    )
    client.run("rollout", "status", "deployment/" + state["deployment"], "--timeout=180s")
    deployment = client.get("deployment", state["deployment"])
    if any(
        volume_map(deployment)[name][1]["configMap"]["name"] != state["current"]
        for name in ("content", "nginx")
    ):
        raise ValueError("The Deployment no longer points to the reviewed release")
    return {"current": state["current"], "rollout": "complete", "http_verification": "required"}


def rollback(client: Kubernetes, release: Path) -> dict:
    state = load_plan(release)["state"]
    deployment = client.get("deployment", state["deployment"])
    previous = client.get("configmap", state["previous"])
    if (
        deployment["metadata"]["uid"] != state["deployment_uid"]
        or previous.get("immutable") is not True
        or digest(previous["data"]) != state["previous_data_sha256"]
    ):
        raise ValueError("Rollback target no longer matches the captured deployment")
    patch = [
        {"op": "test", "path": "/metadata/uid", "value": state["deployment_uid"]},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": deployment["metadata"]["resourceVersion"],
        },
    ]
    for path, expected in state["after"].items():
        patch.extend(
            [
                {"op": "test", "path": path, "value": expected},
                {"op": "replace", "path": path, "value": state["before"][path]},
            ]
        )
    client.run("patch", "deployment", state["deployment"], "--type=json", "-p", json.dumps(patch))
    client.run("rollout", "status", "deployment/" + state["deployment"], "--timeout=180s")
    return {"current": state["previous"], "rollback": "complete", "http_verification": "required"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--kubeconfig", required=True, type=Path)
    parser.add_argument("--namespace", default="mac-fit")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare", help="Read live resources and write an offline release only"
    )
    prep.add_argument("--deployment", default="mac-fit")
    prep.add_argument("--source", type=Path, default=REPOSITORY / "web/macfit/src")
    prep.add_argument(
        "--release",
        type=Path,
        default=REPOSITORY
        / "artifacts"
        / ("frontend-release-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")),
    )
    for command in ("apply", "rollback"):
        action = commands.add_parser(command)
        action.add_argument("release", type=Path)
    args = parser.parse_args(argv)
    client = Kubernetes(args.kubectl, args.kubeconfig, args.namespace)
    if args.command == "prepare":
        result = prepare(client, args.deployment, args.source, args.release.absolute())
    else:
        result = globals()[args.command](client, args.release.absolute())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
