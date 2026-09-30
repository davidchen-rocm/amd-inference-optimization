#!/usr/bin/env python3
"""Build finalizer-only resources from the exact already-reviewed base manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def render(base: dict, scripts: Path | None = None) -> dict:
    scripts = scripts or Path(__file__).resolve().parent
    periodic = next(
        row
        for row in base["items"]
        if row["kind"] == "CronJob" and row["metadata"]["name"] == "macfit-training-backup"
    )
    namespace = periodic["metadata"]["namespace"]
    name = "macfit-training-finalizer"
    content = {
        filename: (scripts / filename).read_text()
        for filename in ("finalizer.py", "finalizer_fetch.sh")
    }
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
    config_name = name + "-" + digest

    def metadata(resource):
        return {"name": resource, "namespace": namespace}

    template = copy.deepcopy(periodic["spec"]["jobTemplate"])
    job = template["spec"]
    job.update(backoffLimit=0, activeDeadlineSeconds=600)
    pod_template = job["template"]
    pod_template["metadata"] = {"labels": {"app": name}}
    pod = pod_template["spec"]
    pod.update(serviceAccountName=name, automountServiceAccountToken=False)
    pod["volumes"].extend(
        [
            {"name": "final-scripts", "configMap": {"name": config_name, "defaultMode": 0o444}},
            {"name": "final-state", "emptyDir": {"sizeLimit": "1Mi"}},
            {
                "name": "final-api",
                "projected": {
                    "defaultMode": 0o440,
                    "sources": [
                        {"serviceAccountToken": {"path": "token", "expirationSeconds": 600}},
                        {
                            "configMap": {
                                "name": "kube-root-ca.crt",
                                "items": [{"key": "ca.crt", "path": "ca.crt"}],
                            }
                        },
                    ],
                },
            },
        ]
    )
    mounts = [
        {"name": "final-scripts", "mountPath": "/opt/finalizer", "readOnly": True},
        {"name": "final-state", "mountPath": "/final-state"},
    ]
    uid_env = {"name": "POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}}
    main = pod["containers"][0]
    recorder = {
        "name": "record-final-start",
        "image": main["image"],
        "imagePullPolicy": main["imagePullPolicy"],
        "securityContext": copy.deepcopy(main["securityContext"]),
        "resources": copy.deepcopy(main["resources"]),
        "command": [
            "python",
            "-c",
            "import time; from pathlib import Path; "
            "Path('/final-state/started-at').write_text(str(time.time()))",
        ],
        "volumeMounts": [{"name": "final-state", "mountPath": "/final-state"}],
    }
    fetch = next(item for item in pod["initContainers"] if item["name"] == "fetch-private-snapshot")
    fetch["command"] = ["/bin/sh", "/opt/finalizer/finalizer_fetch.sh"]
    fetch["volumeMounts"].extend(copy.deepcopy(mounts))
    fetch["env"].append(uid_env)
    pod["initContainers"].insert(0, recorder)
    main["name"] = "verify-and-stop-schedules"
    main["command"] = ["python", "/opt/finalizer/finalizer.py"]
    main["volumeMounts"].extend(
        [*mounts, {"name": "final-api", "mountPath": "/run/finalizer-api", "readOnly": True}]
    )
    main["env"].extend(
        [
            uid_env,
            {"name": "POD_NAMESPACE", "value": namespace},
            {"name": "FINAL_PLANNED_AT", "value": "2026-10-01T03:55:00Z"},
        ]
    )
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": metadata(config_name),
                "immutable": True,
                "data": content,
            },
            {
                "apiVersion": "v1",
                "kind": "ServiceAccount",
                "metadata": metadata(name),
                "automountServiceAccountToken": False,
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "Role",
                "metadata": metadata(name),
                "rules": [
                    {
                        "apiGroups": ["batch"],
                        "resources": ["cronjobs"],
                        "resourceNames": ["macfit-training-backup", name],
                        "verbs": ["get", "patch"],
                    }
                ],
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "RoleBinding",
                "metadata": metadata(name),
                "subjects": [{"kind": "ServiceAccount", "name": name, "namespace": namespace}],
                "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": name},
            },
            {
                "apiVersion": "batch/v1",
                "kind": "CronJob",
                "metadata": {
                    **metadata(name),
                    "annotations": {"macfit.training/one-shot-year": "2026"},
                },
                "spec": {
                    "schedule": "55 23 30 9 *",
                    "timeZone": "America/Detroit",
                    "suspend": False,
                    "concurrencyPolicy": "Forbid",
                    "startingDeadlineSeconds": 180,
                    "successfulJobsHistoryLimit": 1,
                    "failedJobsHistoryLimit": 3,
                    "jobTemplate": template,
                },
            },
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    document = render(json.loads(args.base_manifest.read_text()))
    text = json.dumps(document, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")
