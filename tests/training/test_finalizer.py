from __future__ import annotations

import copy
import importlib.util
import io
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy/macfit-training"


def script(name):
    spec = importlib.util.spec_from_file_location(name, DEPLOY / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pull = script("backup_pull")
finalizer = script("finalizer")
render_base = script("render_cluster")
render_final = script("render_finalizer")
NAMES = ("macfit-training-backup", "macfit-training-finalizer")


def test_finalizer_reuses_reviewed_template_and_projects_token_only_into_main():
    base = render_base.render(ROOT)
    original = copy.deepcopy(base)
    document = render_final.render(base)
    assert base == original
    resources = {item["kind"]: item for item in document["items"]}
    assert resources["Role"]["rules"] == [
        {
            "apiGroups": ["batch"],
            "resources": ["cronjobs"],
            "resourceNames": list(NAMES),
            "verbs": ["get", "patch"],
        }
    ]
    cron = resources["CronJob"]["spec"]
    assert cron["schedule"] == "55 23 30 9 *" and cron["timeZone"] == "America/Detroit"
    job = cron["jobTemplate"]["spec"]
    assert job["backoffLimit"] == 0 and job["activeDeadlineSeconds"] == 600
    pod = job["template"]["spec"]
    assert pod["serviceAccountName"] == NAMES[1] and pod["automountServiceAccountToken"] is False
    main = pod["containers"][0]
    assert any(mount["name"] == "final-api" for mount in main["volumeMounts"])
    assert all(
        not any(mount["name"] == "final-api" for mount in container["volumeMounts"])
        for container in pod["initContainers"]
    )
    base_pod = next(item for item in base["items"] if item["kind"] == "CronJob")["spec"][
        "jobTemplate"
    ]["spec"]["template"]["spec"]
    assert pod["volumes"][: len(base_pod["volumes"])] == base_pod["volumes"]
    assert pod["initContainers"][1] == base_pod["initContainers"][0]
    fetch = next(
        container
        for container in pod["initContainers"]
        if container["name"] == "fetch-private-snapshot"
    )
    baseline_fetch = base_pod["initContainers"][1]
    assert next(value for value in fetch["env"] if value["name"] == "MACFIT_GPU_HOST") == next(
        value for value in baseline_fetch["env"] if value["name"] == "MACFIT_GPU_HOST"
    )


@pytest.fixture
def snapshot(tmp_path):
    started = datetime(2026, 10, 1, 3, 55, tzinfo=UTC).timestamp()
    version = "20261001T035510.000000Z-1234abcd"
    receipt = {
        "schema": pull.PUBLISH_SCHEMA,
        "verified": True,
        "archive_only": True,
        "sqlite_quick_check": "ok",
        "version": version,
        "jobs": 1,
        "source_created_at": datetime.fromtimestamp(started + 5, UTC).isoformat(),
    }
    directory = tmp_path / "versions" / version
    (directory / "data").mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps(receipt))
    with sqlite3.connect(directory / "data/jobs.sqlite3") as connection:
        connection.execute("CREATE TABLE jobs(status TEXT,error TEXT,worker_pid INTEGER)")
        connection.execute("INSERT INTO jobs VALUES('succeeded',NULL,NULL)")
    return tmp_path, started, receipt, directory


class FakeCluster:
    names = NAMES

    def __init__(self):
        self.calls = []

    def suspend(self, name):
        self.calls.append(name)


def test_fresh_terminal_backup_success_requires_both_scoped_suspensions(snapshot):
    root, started, receipt, _ = snapshot
    client = FakeCluster()

    def verified(*_args):
        assert client.calls == list(
            NAMES
        )  # Stop new schedules before potentially long verification.
        return receipt

    assert (
        finalizer.finalize(
            root,
            root / "incoming",
            started,
            True,
            client,
            verifier=verified,
            clock=lambda: started + 30,
            pause=lambda _: None,
        )
        == 0
    )
    report = json.loads((root / "finalization.json").read_text())
    assert report["verified"] is True and report["suspended_cronjobs"] == list(NAMES)


@pytest.mark.parametrize("failure", ["stale", "interrupted", "transfer"])
def test_failed_final_backup_pauses_schedules_but_never_claims_success(snapshot, failure):
    root, started, receipt, directory = snapshot
    if failure == "stale":
        receipt["source_created_at"] = datetime.fromtimestamp(started - 120, UTC).isoformat()
        (directory / "metadata.json").write_text(json.dumps(receipt))
    if failure == "interrupted":
        with sqlite3.connect(directory / "data/jobs.sqlite3") as connection:
            connection.execute(
                "UPDATE jobs SET status='failed',error=?",
                (json.dumps({"code": "archive_interrupted"}),),
            )
    original = (directory / "data/jobs.sqlite3").read_bytes()
    client = FakeCluster()
    result = finalizer.finalize(
        root,
        root / "incoming",
        started,
        failure != "transfer",
        client,
        verifier=lambda *_: receipt,
        clock=lambda: started + 30,
        pause=lambda _: None,
    )
    assert result == 1 and client.calls == list(NAMES)
    report = json.loads((root / "finalization.json").read_text())
    assert report["verified"] is False and report["version"] is None
    assert (directory / "data/jobs.sqlite3").read_bytes() == original


def test_api_failure_is_bounded_and_requires_operator_action(snapshot):
    root, started, receipt, _ = snapshot

    class Denied(FakeCluster):
        def suspend(self, name):
            self.calls.append(name)
            raise OSError("not available")

    client = Denied()
    assert (
        finalizer.finalize(
            root,
            root / "incoming",
            started,
            True,
            client,
            verifier=lambda *_: receipt,
            clock=lambda: started + 30,
            pause=lambda _: None,
        )
        == 2
    )
    assert client.calls == [NAMES[0]] * 3 + [NAMES[1]] * 3


def test_cluster_client_uses_tls_projected_credentials_and_only_patch_suspend(
    tmp_path, monkeypatch
):
    (tmp_path / "token").write_text("synthetic-service-account-token")
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "198.51.100.1")
    context = object()
    monkeypatch.setattr(finalizer.ssl, "create_default_context", lambda **kw: context)
    calls = []

    def fake_api(request, **options):
        calls.append(request.full_url)
        assert options["context"] is context and options["timeout"] == 5
        assert request.method == "PATCH" and json.loads(request.data) == {"spec": {"suspend": True}}
        assert request.headers["Authorization"] == "Bearer synthetic-service-account-token"
        response = io.BytesIO(
            json.dumps({"metadata": {"name": NAMES[0]}, "spec": {"suspend": True}}).encode()
        )
        response.status = 200
        return response

    client = finalizer.ClusterClient("mac-fit", NAMES, credentials=tmp_path, opener=fake_api)
    client.suspend(NAMES[0])
    with pytest.raises(ValueError, match="outside"):
        client.suspend("unrelated-cron")
    assert len(calls) == 1 and calls[0].startswith(
        "https://198.51.100.1:443/apis/batch/v1/namespaces/mac-fit/"
    )
