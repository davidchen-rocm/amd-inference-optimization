"""Offline deployment planning and concurrency guards; no Kubernetes mutation."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "frontend_deployment", REPOSITORY / "deploy/macfit-training/deploy_frontend_training.py"
)
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


@pytest.fixture
def configured_source(tmp_path):
    source = tmp_path / "configured-source"
    shutil.copytree(REPOSITORY / "web/macfit/src", source)
    (source / "firebase-config.js").write_text(
        "export const firebaseConfig = {projectId: 'synthetic-test-project'};\n"
        "export const FIREBASE_SDK_VERSION = 'test';\n"
    )
    return source


@pytest.fixture
def inputs(configured_source):
    source = deploy.read_source(configured_source)
    previous_data = dict(source)
    previous_data.pop("training-client.js")
    previous_data["nginx.conf"] = source["nginx.conf"].replace(deploy.TRAINING_BLOCK, "")
    previous_data["private-maintenance-note.txt"] = "Preserve this existing unmounted key."
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "immutable": True,
        "metadata": {
            "name": "mac-fit-content-old",
            "uid": "cm-uid",
            "resourceVersion": "123",
            "namespace": "mac-fit",
            "labels": {"app": "mac-fit"},
        },
        "data": previous_data,
    }
    deployment = {
        "metadata": {
            "name": "mac-fit",
            "namespace": "mac-fit",
            "uid": "deploy-uid",
            "resourceVersion": "456",
        },
        "spec": {
            "template": {
                "spec": {
                    "securityContext": {"runAsNonRoot": True},
                    "containers": [{"name": "frontend", "image": "unchanged@sha256:example"}],
                    "volumes": [
                        {
                            "name": "content",
                            "configMap": {
                                "name": configmap["metadata"]["name"],
                                "defaultMode": 420,
                                "items": [
                                    {"key": key, "path": key}
                                    for key in previous_data
                                    if key not in ("nginx.conf", "private-maintenance-note.txt")
                                ],
                            },
                        },
                        {
                            "name": "nginx",
                            "configMap": {
                                "name": configmap["metadata"]["name"],
                                "items": [{"key": "nginx.conf", "path": "nginx.conf"}],
                            },
                        },
                        {"name": "tmp", "emptyDir": {}},
                    ],
                }
            }
        },
    }
    return deployment, configmap, source


def test_training_proxy_preserves_all_original_rules_and_is_idempotent(inputs):
    _, old, _ = inputs
    nginx = old["data"]["nginx.conf"]
    modified = deploy.with_training_proxy(nginx)
    assert modified.replace(deploy.TRAINING_BLOCK, "") == nginx
    assert deploy.with_training_proxy(modified) == modified
    assert modified.index("location ^~ /api/training/ {") < modified.index("location ^~ /api/ {")
    assert "proxy_pass http://macfit-training-bridge:8080;" in modified
    assert "proxy_pass http://macfit-api:8080;" in modified
    assert "client_max_body_size 3m;" in modified


def test_plan_adds_training_module_without_exposing_existing_unmounted_keys(inputs):
    original, old, source = inputs
    preserved = copy.deepcopy(original)
    plan = deploy.build_plan(original, old, source)
    new = plan["configmap"]
    assert original == preserved
    assert new["immutable"] is True and "resourceVersion" not in new["metadata"]
    assert (
        new["data"]["private-maintenance-note.txt"] == old["data"]["private-maintenance-note.txt"]
    )
    items = plan["state"]["after"]["/spec/template/spec/volumes/0/configMap"]["items"]
    assert {"key": "training-client.js", "path": "training-client.js"} in items
    assert not {"nginx.conf", "private-maintenance-note.txt"} & {item["key"] for item in items}
    assert [op["path"] for op in plan["patch"] if op["op"] == "replace"] == [
        "/spec/template/spec/volumes/0/configMap",
        "/spec/template/spec/volumes/1/configMap",
    ]
    assert plan["patch"][1] == {"op": "test", "path": "/metadata/resourceVersion", "value": "456"}


def test_plan_refuses_unreviewed_nginx_change(inputs):
    original, old, source = inputs
    source["nginx.conf"] += "# unrelated change\n"
    with pytest.raises(ValueError, match="differs from live"):
        deploy.build_plan(original, old, source)


def test_followup_release_accepts_original_private_nginx_but_keeps_live_training_proxy(inputs):
    original, old, source = inputs
    source["nginx.conf"] = old["data"]["nginx.conf"]
    old["data"]["nginx.conf"] = deploy.with_training_proxy(old["data"]["nginx.conf"])
    plan = deploy.build_plan(original, old, source)
    assert plan["configmap"]["data"]["nginx.conf"] == old["data"]["nginx.conf"]


class FakeKubernetes:
    def __init__(self, deployment, configmap):
        self.deployment, self.configmap = deployment, configmap
        self.calls = []

    def get(self, kind, name):
        self.calls.append(("get", kind, name))
        return copy.deepcopy(self.deployment if kind == "deployment" else self.configmap)

    def run(self, *args):
        self.calls.append(args)
        raise AssertionError("No mutation should be attempted")


def test_prepare_only_reads_live_state_and_keeps_complete_rollback_snapshot(
    tmp_path, inputs, configured_source
):
    original, old, _ = inputs
    client = FakeKubernetes(original, old)
    target = tmp_path / "release"
    receipt = deploy.prepare(client, "mac-fit", configured_source, target)
    assert receipt["cluster_mutated"] is False
    assert all(call[0] == "get" for call in client.calls)
    assert json.loads((target / "deployment.before.json").read_text()) == original
    assert json.loads((target / "configmap.before.json").read_text()) == old
    assert deploy.load_plan(target)["state"]["current"] == receipt["current"]


@pytest.mark.parametrize("change", ["deployment", "content-uid", "content-data"])
def test_apply_checks_concurrency_before_any_mutation(tmp_path, inputs, change, configured_source):
    original, old, _ = inputs
    client = FakeKubernetes(original, old)
    target = tmp_path / "release"
    deploy.prepare(client, "mac-fit", configured_source, target)
    if change == "deployment":
        client.deployment["metadata"]["resourceVersion"] = "newer"
    elif change == "content-uid":
        client.configmap["metadata"]["uid"] = "recreated"
    else:
        client.configmap["data"]["app.js"] = "changed"
    with pytest.raises(ValueError, match="changed since preparation"):
        deploy.apply(client, target)
    assert all(call[0] == "get" for call in client.calls)


def test_saved_plan_tampering_is_rejected_before_live_mutation(tmp_path, inputs, configured_source):
    original, old, _ = inputs
    client = FakeKubernetes(original, old)
    target = tmp_path / "release"
    deploy.prepare(client, "mac-fit", configured_source, target)
    altered = json.loads((target / "patch.json").read_text())
    altered.pop(0)
    (target / "patch.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="was changed"):
        deploy.apply(client, target)
    assert all(call[0] == "get" for call in client.calls)


def test_unconfigured_repository_template_cannot_be_prepared_for_deployment():
    with pytest.raises(ValueError, match="Replace the Firebase configuration template"):
        deploy.read_source(REPOSITORY / "web/macfit/src")
