"""CPU-only regression tests for publication into the OpenShift backup PVC."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
import sqlite3
import tarfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from macfit_training.service.backup import export_archive
from macfit_training.service.settings import Settings
from macfit_training.service.store import JobStore

SCRIPT = Path(__file__).resolve().parents[2] / "deploy/macfit-training/backup_pull.py"
SPEC = importlib.util.spec_from_file_location("macfit_backup_pull", SCRIPT)
backup_pull = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup_pull)


def valid_archive(directory: Path) -> Path:
    directory.mkdir()
    store = JobStore(Settings(directory / "source-data", "synthetic-gateway-secret-for-tests"))
    store.create(
        "synthetic-owner",
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        "generation",
        {"base_model": {"id": "test"}, "generation_config": {"walltime_seconds": 3600}},
    )
    source = directory / "source"
    (source / "src").mkdir(parents=True)
    (source / "src/example.py").write_text("# source preserved as evidence\n")
    path = directory / "snapshot.tgz"
    with path.open("wb") as stream:
        export_archive(store.root, stream, source_dir=source)
    return path


def rewrite_archive(
    source: Path, target: Path, *, corrupt_database=False, change_manifest=False, created_at=None
):
    with tarfile.open(source) as archive:
        members = {member.name: archive.extractfile(member).read() for member in archive}
    manifest = json.loads(members.pop("manifest.json"))
    if created_at is not None:
        manifest["created_at"] = created_at
    if corrupt_database:
        members["jobs.sqlite3"] = b"not a database"
        for item in manifest["files"]:
            if item["path"] == "jobs.sqlite3":
                item["size_bytes"] = len(members["jobs.sqlite3"])
                item["sha256"] = hashlib.sha256(members["jobs.sqlite3"]).hexdigest()
    if change_manifest:
        manifest["files"][0]["sha256"] = "0" * 64
    members["manifest.json"] = json.dumps(manifest).encode()
    with tarfile.open(target, "w:gz") as archive:
        for name, contents in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))


@pytest.fixture
def published(tmp_path):
    archive = valid_archive(tmp_path / "input")
    root = tmp_path / "pvc"
    metadata = backup_pull.publish_archive(archive, root)
    return archive, root, metadata


def assert_previous_snapshot(root, metadata):
    assert os.readlink(root / "latest") == "versions/" + metadata["version"]
    assert json.loads((root / "latest/metadata.json").read_text()) == metadata
    assert (root / "latest/data/jobs.sqlite3").is_file()
    assert not list((root / "versions").glob("*.pending"))
    assert not list((root / "versions").glob(".*.pending"))


def test_publish_retains_archive_and_verified_archive_only_data(published):
    archive, root, metadata = published
    assert metadata["verified"] is True and metadata["sqlite_quick_check"] == "ok"
    assert metadata["jobs"] == 1
    assert (root / "latest/archive.tgz").read_bytes() == archive.read_bytes()
    assert metadata["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert (root / "latest/data/source/src/example.py").is_file()
    assert json.loads((root / "latest/data/archive-mode.json").read_text())["archive_only"] is True
    with sqlite3.connect(root / "latest/data/jobs.sqlite3") as connection:
        assert connection.execute("SELECT status FROM jobs").fetchone()[0] == "failed"
    assert_previous_snapshot(root, metadata)


@pytest.mark.parametrize(
    "failure",
    ["hash", "database", "truncated-gzip", "entry-limit", "total-limit", "compressed-limit"],
)
def test_failed_backup_does_not_replace_previous_snapshot(
    tmp_path, published, monkeypatch, failure
):
    archive, root, metadata = published
    broken = tmp_path / "broken.tgz"
    if failure == "hash":
        rewrite_archive(archive, broken, change_manifest=True)
    elif failure == "database":
        rewrite_archive(archive, broken, corrupt_database=True)
    elif failure == "truncated-gzip":
        broken.write_bytes(archive.read_bytes()[:-5])
    else:
        broken.write_bytes(archive.read_bytes())
        field = {
            "entry-limit": "MAX_ENTRY_BYTES",
            "total-limit": "MAX_UNPACKED_BYTES",
            "compressed-limit": "MAX_ARCHIVE_BYTES",
        }[failure]
        monkeypatch.setattr(backup_pull, field, 1)
    with pytest.raises((ValueError, EOFError, sqlite3.Error, tarfile.TarError)):
        backup_pull.publish_archive(broken, root)
    assert_previous_snapshot(root, metadata)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../outside", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("jobs.sqlite3", tarfile.SYMTYPE),
        ("jobs.sqlite3", tarfile.LNKTYPE),
        ("jobs.sqlite3", tarfile.CHRTYPE),
        ("jobs.sqlite3", tarfile.BLKTYPE),
        ("jobs.sqlite3", tarfile.FIFOTYPE),
        ("jobs.sqlite3", tarfile.DIRTYPE),
    ],
)
def test_unsafe_tar_never_changes_latest(tmp_path, published, name, kind):
    _archive, root, metadata = published
    broken = tmp_path / "unsafe.tgz"
    with tarfile.open(broken, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "../../outside"
        member.size = 1 if kind == tarfile.REGTYPE else 0
        archive.addfile(member, io.BytesIO(b"x") if member.size else None)
    with pytest.raises(ValueError, match="Unsafe"):
        backup_pull.publish_archive(broken, root)
    assert_previous_snapshot(root, metadata)
    assert not (tmp_path / "outside").exists()


def test_quick_check_failure_does_not_publish_restored_directory(published, monkeypatch):
    archive, root, metadata = published

    def failed_check(*_args):
        raise ValueError("quick_check failed")

    monkeypatch.setattr(backup_pull, "quick_check", failed_check)
    with pytest.raises(ValueError, match="quick_check"):
        backup_pull.publish_archive(archive, root)
    assert_previous_snapshot(root, metadata)
    assert len(backup_pull.verified_versions(root)) == 1


def test_retains_latest_four_verified_snapshots(published):
    archive, root, first = published
    published_versions = [first["version"]]
    for _ in range(5):
        metadata = backup_pull.publish_archive(archive, root)
        published_versions.append(metadata["version"])
    assert [item.name for item in backup_pull.verified_versions(root)] == published_versions[-4:]
    assert_previous_snapshot(root, metadata)


def test_full_disk_keeps_last_successful_backup(published, monkeypatch):
    archive, root, metadata = published
    monkeypatch.setattr(backup_pull.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="Insufficient"):
        backup_pull.publish_archive(archive, root)
    assert_previous_snapshot(root, metadata)


def test_older_periodic_archive_cannot_replace_newer_final_snapshot(tmp_path, published):
    archive, root, metadata = published
    older = tmp_path / "older-periodic.tgz"
    instant = datetime.fromisoformat(metadata["source_created_at"].replace("Z", "+00:00"))
    rewrite_archive(archive, older, created_at=(instant - timedelta(minutes=5)).isoformat())
    with pytest.raises(ValueError, match="older source snapshot"):
        backup_pull.publish_archive(older, root)
    assert_previous_snapshot(root, metadata)
    assert len(backup_pull.verified_versions(root)) == 1


def test_atomic_pointer_publication_failure_keeps_previous_version(published, monkeypatch):
    archive, root, metadata = published
    original_replace = backup_pull.os.replace

    def failed_pointer(source, target):
        if target == root / "latest":
            raise OSError("synthetic failed pointer publication")
        return original_replace(source, target)

    monkeypatch.setattr(backup_pull.os, "replace", failed_pointer)
    with pytest.raises(OSError, match="pointer publication"):
        backup_pull.publish_archive(archive, root)
    assert_previous_snapshot(root, metadata)
    assert len(backup_pull.verified_versions(root)) == 1


def test_abandoned_stage_is_removed_but_latest_survives_failed_next_pull(published):
    archive, root, metadata = published
    abandoned = root / "versions/.20000101T000000.000000Z-12345678.pending"
    abandoned.mkdir()
    (abandoned / "partial").write_bytes(b"partial archive")
    archive.write_bytes(b"transfer failed")
    with pytest.raises(tarfile.TarError):
        backup_pull.publish_archive(archive, root)
    assert not abandoned.exists()
    assert_previous_snapshot(root, metadata)


def test_first_invalid_backup_creates_no_latest(tmp_path):
    path = tmp_path / "broken.tgz"
    path.write_bytes(b"invalid tar")
    root = tmp_path / "pvc"
    with pytest.raises(tarfile.TarError):
        backup_pull.publish_archive(path, root)
    assert not (root / "latest").exists()
    assert not backup_pull.verified_versions(root)


def test_hidden_archive_after_tar_end_is_rejected(tmp_path, published):
    archive, root, metadata = published
    hidden = io.BytesIO()
    with tarfile.open(fileobj=hidden, mode="w") as second:
        item = tarfile.TarInfo("../../hidden")
        item.size = 1
        second.addfile(item, io.BytesIO(b"x"))
    path = tmp_path / "hidden.tgz"
    path.write_bytes(gzip.compress(gzip.decompress(archive.read_bytes()) + hidden.getvalue()))
    with pytest.raises(ValueError, match="after the end"):
        backup_pull.publish_archive(path, root)
    assert_previous_snapshot(root, metadata)


def test_cli_can_raise_limits_within_service_restore_ceiling(monkeypatch, capsys):
    recorded = {}
    for name in (
        "MAX_ARCHIVE_BYTES",
        "MAX_UNPACKED_BYTES",
        "MAX_ENTRY_BYTES",
        "FREE_RESERVE_BYTES",
    ):
        monkeypatch.setattr(backup_pull, name, getattr(backup_pull, name))

    def capture(*_args, **_kwargs):
        recorded.update(
            archive=backup_pull.MAX_ARCHIVE_BYTES,
            unpacked=backup_pull.MAX_UNPACKED_BYTES,
            entry=backup_pull.MAX_ENTRY_BYTES,
        )
        return {"verified": True}

    monkeypatch.setattr(backup_pull, "publish_archive", capture)
    assert (
        backup_pull.main(
            [
                "verify",
                "--max-archive-gib",
                "24",
                "--max-unpacked-gib",
                "24",
                "--max-entry-gib",
                "8",
            ]
        )
        == 0
    )
    assert recorded == {
        "archive": 24 * backup_pull.GIB,
        "unpacked": 24 * backup_pull.GIB,
        "entry": 8 * backup_pull.GIB,
    }
    assert json.loads(capsys.readouterr().out) == {"verified": True}
    with pytest.raises(SystemExit) as error:
        backup_pull.main(["verify", "--max-unpacked-gib", "33"])
    assert error.value.code == 2
