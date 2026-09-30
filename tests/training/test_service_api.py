from __future__ import annotations

import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from macfit_training.service.api import create_app
from macfit_training.service.auth import AuthenticationError, Identity
from macfit_training.service.settings import Settings

SECRET = "synthetic-gateway-secret-for-tests-only"


class TestVerifier:
    __test__ = False

    def verify(self, token):
        mapping = {"test-alice": "alice", "test-bob": "bob", "test-charlie": "charlie"}
        if token not in mapping:
            raise AuthenticationError("invalid")
        return Identity(mapping[token])


class QueueOnlySupervisor:
    def __init__(self, *_args, **_kwargs):
        self.healthy = False

    def start(self):
        self.healthy = True

    def stop(self):
        self.healthy = False

    def notify(self):
        pass


def headers(owner="alice"):
    return {"X-Training-Gateway": SECRET, "Authorization": "Bearer test-" + owner}


def submission(**changes):
    value = {
        "request_id": str(uuid.uuid4()),
        "project_id": str(uuid.uuid4()),
        "kind": "generation",
        "input": {
            "model_id": "qwen3-0-6b",
            "task": "answers",
            "goal": "Answer factual shop questions.",
            "language": "en",
            "source": "",
            "purpose": "preview",
            "seeds": [],
            "target_count": 3,
        },
    }
    value.update(changes)
    return value


@pytest.fixture
def service(tmp_path):
    settings = Settings(tmp_path, SECRET, min_free_bytes=0, max_queue_jobs=2)
    ready = [True]
    app = create_app(
        settings,
        verifier=TestVerifier(),
        readiness=lambda: ready[0],
        supervisor_factory=QueueOnlySupervisor,
    )
    with TestClient(app) as client:
        yield client, app.state.store, ready


def test_gateway_and_user_authentication_are_separate(service):
    client, _, _ = service
    assert client.get("/api/training/capabilities").status_code == 403
    result = client.get("/api/training/capabilities", headers={"X-Training-Gateway": SECRET})
    assert result.status_code == 200 and result.json()["auth"]["required"] is True
    assert (
        client.post(
            "/api/training/jobs", json=submission(), headers={"X-Training-Gateway": SECRET}
        ).status_code
        == 401
    )
    assert (
        client.post("/api/training/jobs", json=submission(), headers=headers("forged")).status_code
        == 401
    )
    forged = submission(owner_uid="bob")
    assert client.post("/api/training/jobs", json=forged, headers=headers()).status_code == 422


@pytest.mark.parametrize("from_environment", [False, True])
def test_archive_gateway_normalizes_configured_secret_but_keeps_authentication(
    tmp_path, monkeypatch, from_environment
):
    stored_secret = " \t" + SECRET + "\r\n"
    if from_environment:
        monkeypatch.setenv("MACFIT_TRAINING_DATA", str(tmp_path))
        monkeypatch.setenv("MACFIT_TRAINING_GATEWAY_SECRET", stored_secret)
        monkeypatch.setenv("MACFIT_ARCHIVE_ONLY", "1")
        monkeypatch.delenv("MACFIT_GPU_DEADLINE", raising=False)
        monkeypatch.delenv("MACFIT_STOP_ACCEPTING_AT", raising=False)
        settings = Settings.from_env()
    else:
        settings = Settings(tmp_path, stored_secret, archive_only=True)
    assert settings.gateway_secret == SECRET
    app = create_app(settings, verifier=TestVerifier())
    with TestClient(app) as client:
        response = client.get(
            "/api/training/capabilities", headers={"X-Training-Gateway": SECRET}
        )
        assert response.status_code == 200
        assert response.json()["available"] is False
        for wrong_secret in (SECRET[:-1] + "X", SECRET + " "):
            assert client.get(
                "/api/training/capabilities", headers={"X-Training-Gateway": wrong_secret}
            ).status_code == 403
        assert client.get("/api/training/capabilities").status_code == 403
        assert client.get(
            "/api/training/jobs", headers={"X-Training-Gateway": SECRET}
        ).status_code == 401
        assert client.get("/api/training/jobs", headers=headers()).json() == {"items": []}


@pytest.mark.parametrize("secret", [" " * 32, "x" * 31 + "\n"])
def test_gateway_secret_length_is_checked_after_normalization(tmp_path, secret):
    with pytest.raises(ValueError, match="at least 32 bytes"):
        Settings(tmp_path, secret)


def test_job_is_immutable_idempotent_and_owner_scoped(service):
    client, store, ready = service
    request = submission()
    first = client.post("/api/training/jobs", json=request, headers=headers())
    assert first.status_code == 202
    job = first.json()
    assert "owner_uid" not in job and "worker_pid" not in job
    saved = json.loads((store.directory(job["id"]) / "input.json").read_text())
    assert len(saved["base_model"]["revision"]) == 40
    again = client.post("/api/training/jobs", json=request, headers=headers())
    assert again.status_code == 200 and again.json()["id"] == job["id"]
    different = json.loads(json.dumps(request))
    different["input"]["goal"] = "A different set of instructions."
    assert client.post("/api/training/jobs", json=different, headers=headers()).status_code == 409
    ready[0] = False
    assert client.post("/api/training/jobs", json=request, headers=headers()).status_code == 200
    for suffix in ["", "/artifacts/guess"]:
        assert (
            client.get(
                "/api/training/jobs/" + job["id"] + suffix, headers=headers("bob")
            ).status_code
            == 404
        )
    assert (
        client.post(
            "/api/training/jobs/" + job["id"] + "/cancel", headers=headers("bob")
        ).status_code
        == 404
    )
    assert client.get(
        "/api/training/jobs", params={"request_id": request["request_id"]}, headers=headers("bob")
    ).json() == {"items": []}
    assert (
        client.get(
            "/api/training/jobs", params={"request_id": request["request_id"]}, headers=headers()
        ).json()["items"][0]["id"]
        == job["id"]
    )
    assert json.loads((store.directory(job["id"]) / "input.json").read_text()) == saved


def test_owner_active_quota_global_queue_and_cancellation(service):
    client, store, _ = service
    first = client.post("/api/training/jobs", json=submission(), headers=headers()).json()
    assert (
        client.post("/api/training/jobs", json=submission(), headers=headers()).json()["error"][
            "code"
        ]
        == "owner_busy"
    )
    assert (
        client.post("/api/training/jobs", json=submission(), headers=headers("bob")).status_code
        == 202
    )
    assert (
        client.post("/api/training/jobs", json=submission(), headers=headers("charlie")).status_code
        == 429
    )
    cancelled = client.post(
        "/api/training/jobs/" + first["id"] + "/cancel", headers=headers()
    ).json()
    assert cancelled["status"] == "cancelled"
    assert store.get(first["id"])["worker_pid"] is None
    assert (
        client.post("/api/training/jobs", json=submission(), headers=headers()).status_code == 202
    )


def test_concurrent_duplicate_submissions_create_exactly_one_durable_job(service):
    client, store, _ = service
    request = submission()
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(
                lambda _: client.post("/api/training/jobs", json=request, headers=headers()),
                range(4),
            )
        )
    assert sorted(r.status_code for r in responses) == [200, 200, 200, 202]
    assert len({r.json()["id"] for r in responses}) == 1
    assert len(store.list("alice")) == 1
    assert len(list(store.jobs_dir.iterdir())) == 1


def test_body_limits_and_public_registry_validation(service):
    client, _, _ = service
    assert (
        client.post(
            "/api/training/jobs", content=b"x" * (3 * 1024 * 1024 + 1), headers=headers()
        ).status_code
        == 413
    )
    invalid = submission()
    invalid["input"]["model_id"] = "../arbitrary-model"
    assert client.post("/api/training/jobs", json=invalid, headers=headers()).status_code == 422
    invalid = submission()
    invalid["input"]["command"] = "arbitrary python"
    assert client.post("/api/training/jobs", json=invalid, headers=headers()).status_code == 422
    assert client.get("/api/training/jobs?owner_uid=bob", headers=headers()).status_code == 422


def publish_fixture(store, job_id, payload=b"valid adapter"):
    job = store.claim()
    assert job["id"] == job_id
    directory = store.directory(job_id) / "artifacts"
    directory.mkdir()
    (directory / "adapter.tar.gz").write_bytes(payload)
    descriptor = {
        "id": "adapter",
        "type": "lora_adapter",
        "name": "adapter.tar.gz",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    store.finish(job_id, "succeeded", result={"method": "lora_sft"}, artifacts=[descriptor])
    return directory / "adapter.tar.gz"


def test_download_checks_owner_regular_file_and_content_hash(service, tmp_path):
    client, store, _ = service
    job = client.post("/api/training/jobs", json=submission(), headers=headers()).json()
    path = publish_fixture(store, job["id"])
    url = "/api/training/jobs/" + job["id"] + "/artifacts/adapter"
    assert client.get(url, headers=headers("bob")).status_code == 404
    response = client.get(url, headers=headers())
    assert response.status_code == 200 and response.content == b"valid adapter"
    assert "attachment" in response.headers["content-disposition"]
    path.write_bytes(b"corrupted")
    assert client.get(url, headers=headers()).status_code == 409
    outside = tmp_path / "unrelated"
    outside.write_bytes(b"valid adapter")
    path.unlink()
    path.symlink_to(outside)
    assert client.get(url, headers=headers()).status_code == 409
    assert outside.read_bytes() == b"valid adapter"


def test_gpu_deadline_closes_admission_but_keeps_saved_jobs_and_downloads(tmp_path, monkeypatch):
    import time

    now = time.time()
    settings = Settings(
        tmp_path, SECRET, min_free_bytes=0, gpu_deadline=now + 7200, stop_accepting_at=now + 3600
    )
    app = create_app(
        settings,
        verifier=TestVerifier(),
        readiness=lambda: True,
        supervisor_factory=QueueOnlySupervisor,
    )
    with TestClient(app) as client:
        request = submission()
        response = client.post("/api/training/jobs", json=request, headers=headers())
        assert response.status_code == 202
        job_id = response.json()["id"]
        publish_fixture(app.state.store, job_id)
        monkeypatch.setattr("macfit_training.service.settings.time.time", lambda: now + 7300)
        capability = client.get("/api/training/capabilities", headers=headers()).json()
        assert capability["available"] is False
        assert (
            client.post("/api/training/jobs", json=submission(), headers=headers()).json()["error"][
                "code"
            ]
            == "gpu_window_closed"
        )
        assert client.post("/api/training/jobs", json=request, headers=headers()).status_code == 200
        assert (
            client.get("/api/training/jobs/" + job_id, headers=headers()).json()["status"]
            == "succeeded"
        )
        assert (
            client.get(
                "/api/training/jobs/" + job_id + "/artifacts/adapter", headers=headers()
            ).content
            == b"valid adapter"
        )


def test_archive_mode_cannot_be_overridden_by_healthy_gpu(tmp_path):
    (tmp_path / "archive-mode.json").write_text('{"archive_only":true}')
    settings = Settings(tmp_path, SECRET, min_free_bytes=0)
    app = create_app(
        settings,
        verifier=TestVerifier(),
        readiness=lambda: True,
        supervisor_factory=QueueOnlySupervisor,
    )
    with TestClient(app) as client:
        assert (
            client.get("/api/training/capabilities", headers=headers()).json()["available"] is False
        )
        assert (
            client.post("/api/training/jobs", json=submission(), headers=headers()).status_code
            == 503
        )
        assert client.get("/api/training/jobs", headers=headers()).json() == {"items": []}
