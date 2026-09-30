from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tarfile
import time
import uuid

import pytest

from macfit_training.service.backup import export_archive, restore_archive
from macfit_training.service.settings import Settings
from macfit_training.service.store import JobStore, ServiceError


def setup_store(tmp_path, **changes):
    settings = Settings(
        tmp_path / "data", "synthetic-gateway-secret-for-tests", min_free_bytes=0, **changes
    )
    store = JobStore(settings)

    def submit(owner="alice"):
        return store.create(
            owner,
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            "generation",
            {"base_model": {"id": "test"}, "generation_config": {"walltime_seconds": 3600}},
        )[0]

    return store, submit


def test_online_backup_preserves_verified_artifacts_and_turns_unfinished_jobs_into_archive(
    tmp_path,
):
    store, submit = setup_store(tmp_path)
    completed = submit()
    store.claim()
    directory = store.directory(completed["id"])
    artifacts = directory / "artifacts"
    artifacts.mkdir()
    payload = b"actual test artifact"
    (artifacts / "generation.json").write_bytes(payload)
    descriptor = {
        "id": "generation-json",
        "type": "dataset",
        "name": "generation.json",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (directory / "result.json").write_text(
        json.dumps({"method": "model_generation", "artifacts": [descriptor]})
    )
    provenance = {
        "model-snapshot.json": {
            "repo_id": "Qwen/Qwen3-0.6B",
            "revision": "a" * 40,
            "files": [{"path": "config.json", "sha256": "b" * 64, "size_bytes": 123}],
        },
        "runtime-environment.json": {
            "python_version": "3.12.3",
            "torch_version": "2.10.0",
            "device": "AMD MI300X",
        },
        "training-source.json": {"files": {"trainer.py": "c" * 64}},
    }
    for filename, contents in provenance.items():
        (directory / filename).write_text(json.dumps(contents))
    (directory / "gpu.env").write_text("synthetic credential fixture must stay excluded")
    (directory / "worker.log").write_text("private log fixture must stay excluded")
    store.finish(
        completed["id"], "succeeded", result={"method": "model_generation"}, artifacts=[descriptor]
    )
    pending = submit("bob")
    store.claim()
    pending_dir = store.directory(pending["id"])
    (pending_dir / "events.jsonl").write_text('{"stage":"training"}\n')
    (pending_dir / "result.json").write_text('{"never":"treat this as completed"}')
    (pending_dir / "worker.log").write_text("private logs are excluded")
    (store.root / "auth.env").write_text("synthetic secret excluded")
    stream = io.BytesIO()
    export_archive(store.root, stream)
    archive = tmp_path / "backup.tar.gz"
    archive.write_bytes(stream.getvalue())
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert not any("auth.env" in name or "worker.log" in name for name in names)
        assert f"jobs/{pending['id']}/result.json" not in names
    restored = tmp_path / "restored"
    restore_archive(archive, restored)
    with sqlite3.connect(restored / "jobs.sqlite3") as connection:
        assert (
            connection.execute("SELECT status FROM jobs WHERE id=?", (pending["id"],)).fetchone()[0]
            == "failed"
        )
        assert (
            connection.execute("SELECT status FROM jobs WHERE id=?", (completed["id"],)).fetchone()[
                0
            ]
            == "succeeded"
        )
    assert (
        restored / "jobs" / completed["id"] / "artifacts" / "generation.json"
    ).read_bytes() == payload
    for filename, contents in provenance.items():
        assert json.loads((restored / "jobs" / completed["id"] / filename).read_text()) == contents
    assert not (restored / "jobs" / completed["id"] / "gpu.env").exists()
    assert not (restored / "jobs" / completed["id"] / "worker.log").exists()
    assert store.get(pending["id"])["status"] == "running"
    assert (restored / "archive-mode.json").is_file()


def test_export_rejects_artifact_symlink(tmp_path):
    store, submit = setup_store(tmp_path)
    job = submit()
    store.claim()
    external = tmp_path / "external"
    external.write_bytes(b"data")
    artifacts = store.directory(job["id"]) / "artifacts"
    artifacts.mkdir()
    (artifacts / "data.json").symlink_to(external)
    descriptor = {
        "id": "data",
        "type": "dataset",
        "name": "data.json",
        "size_bytes": 4,
        "sha256": hashlib.sha256(b"data").hexdigest(),
    }
    store.finish(job["id"], "succeeded", result={}, artifacts=[descriptor])
    with pytest.raises((ValueError, OSError)):
        export_archive(store.root, io.BytesIO())


def test_restore_rejects_path_traversal_without_publishing(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escaped")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="Unsafe"):
        restore_archive(archive, destination)
    assert not destination.exists() and not (tmp_path / "escaped").exists()


def test_queue_checks_deadline_and_keeps_existing_data_readable(tmp_path, monkeypatch):
    now = time.time()
    store, submit = setup_store(tmp_path, gpu_deadline=now + 7500, stop_accepting_at=now + 5000)
    first = submit()
    second = submit("bob")
    with pytest.raises(ServiceError) as error:
        submit("charlie")
    assert error.value.code == "gpu_window_closed"
    monkeypatch.setattr("macfit_training.service.settings.time.time", lambda: now + 4000)
    assert store.claim() is None
    assert store.get(first["id"])["error"]["code"] == "gpu_window_closed"
    assert store.get(second["id"])["status"] == "failed"
    assert (store.directory(first["id"]) / "input.json").is_file()


def test_backup_includes_allowlisted_source_and_public_evidence_only(tmp_path):
    store, submit = setup_store(tmp_path)
    submit()
    source = tmp_path / "source-repository"
    (source / "src" / "macfit_training").mkdir(parents=True)
    (source / "src" / "private").mkdir()
    (source / "src" / "macfit_training" / "worker.py").write_text('print("example source")\n')
    (source / "src" / "private" / "secret.py").write_text("excluded private data")
    (source / "pyproject.toml").write_text('[project]\nname="sample"\n')
    (source / ".env").write_text("excluded credentials")
    (source / "unrelated.json").write_text("{}")
    evidence = store.root / "evidence"
    evidence.mkdir()
    (evidence / "runtime.json").write_text('{"torch":"2.10.0","gpu":"MI300X"}')
    stream = io.BytesIO()
    export_archive(store.root, stream, source_dir=source)
    archive = tmp_path / "complete.tar.gz"
    archive.write_bytes(stream.getvalue())
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert "source/src/macfit_training/worker.py" in names
        assert "source/pyproject.toml" in names
        assert "evidence/runtime.json" in names
        assert not any("private" in name or ".env" in name or "unrelated" in name for name in names)
    destination = tmp_path / "restored"
    restore_archive(archive, destination)
    assert (destination / "source" / "src" / "macfit_training" / "worker.py").is_file()
    assert Settings(destination, "synthetic-gateway-secret-for-tests").archive_only is True
    with pytest.raises(ValueError):
        restore_archive(archive, destination)


def test_runtime_evidence_rejects_private_material(tmp_path):
    store, submit = setup_store(tmp_path)
    submit()
    evidence = tmp_path / "runtime.json"
    evidence.write_text('{"gateway_secret":"synthetic value"}')
    stream = io.BytesIO()
    with pytest.raises(ValueError, match="Sensitive"):
        export_archive(store.root, stream, evidence_files=[evidence])
    assert stream.getvalue() == b""


def test_restore_rejects_tampered_content_even_when_tar_is_valid(tmp_path):
    store, submit = setup_store(tmp_path)
    submit()
    stream = io.BytesIO()
    export_archive(store.root, stream)
    original = io.BytesIO(stream.getvalue())
    corrupt = tmp_path / "corrupt.tar.gz"
    with tarfile.open(fileobj=original) as source, tarfile.open(corrupt, "w:gz") as target:
        for member in source:
            contents = source.extractfile(member).read()
            if member.name.endswith("/input.json"):
                contents = contents.replace(b"test", b"fake")
            target.addfile(member, io.BytesIO(contents))
    with pytest.raises(ValueError, match="integrity"):
        restore_archive(corrupt, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_complete_web_and_training_test_sources_survive_backup_restore(tmp_path):
    store, submit = setup_store(tmp_path)
    submit()
    source = tmp_path / "repository"
    included = [
        "web/macfit/src/index.html",
        "web/macfit/src/styles.css",
        "web/macfit/src/app.js",
        "web/macfit/tests/training-client.test.mjs",
        "web/macfit/src/favicon.svg",
        "web/macfit/src/app.webmanifest",
        "web/macfit/src/nginx.conf",
        "web/macfit/package.json",
        "web/macfit/README.md",
        "tests/training/test_service_api.py",
        "tests/training/fixtures/request.json",
    ]
    excluded = [
        "web/macfit/node_modules/dependency/index.js",
        "web/macfit/.git/config.json",
        "web/macfit/.env",
        "web/macfit/.private/key.json",
        "web/macfit/private/token.json",
        "web/macfit/credentials.json",
        "web/macfit/secrets/gateway.json",
        "web/macfit/serviceAccountKey.json",
        "web/macfit/src/other.conf",
        "web/another-app/app.js",
        "tests/other/test_private.py",
        "tests/training/__pycache__/test_file.py",
        "src/arbitrary-static.js",
    ]
    for name in included + excluded:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("public fixture content" if name in included else "excluded fixture")
    stream = io.BytesIO()
    export_archive(store.root, stream, source_dir=source)
    archive = tmp_path / "full-source.tar.gz"
    archive.write_bytes(stream.getvalue())
    with tarfile.open(archive) as tar:
        actual = {
            name.removeprefix("source/") for name in tar.getnames() if name.startswith("source/")
        }
    assert actual == set(included)
    restored = tmp_path / "restored-full-source"
    restore_archive(archive, restored)
    for name in included:
        assert (restored / "source" / name).read_text() == "public fixture content"


def test_realistic_rocm_runtime_fingerprint_over_four_mib_roundtrips(tmp_path):
    store, submit = setup_store(tmp_path)
    job = submit()
    # The real ROCm host emitted a 7,135,387-byte runtime report, beyond the old 4 MiB cap.
    metadata = {"package_file_fingerprints": "x" * 7_135_000}
    raw = json.dumps(metadata).encode()
    assert len(raw) > 4 * 1024**2
    runtime = store.directory(job["id"]) / "runtime-environment.json"
    runtime.write_bytes(raw)
    stream = io.BytesIO()
    export_archive(store.root, stream)
    archive = tmp_path / "runtime-backup.tar.gz"
    archive.write_bytes(stream.getvalue())
    restored = tmp_path / "restored-runtime"
    restore_archive(archive, restored)
    assert (restored / "jobs" / job["id"] / "runtime-environment.json").read_bytes() == raw
