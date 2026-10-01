#!/usr/bin/env python3
"""Verify a pulled private archive and atomically publish it on the backup PVC.

Only the standard library and the CPU-only service.backup module are required.
This process neither connects to SSH nor reads authentication credentials.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import tarfile
import uuid
import zlib
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from macfit_training.service.backup import restore_archive, safe_relative

GIB = 1024**3
MAX_ARCHIVE_BYTES = 5 * GIB
MAX_UNPACKED_BYTES = 20 * GIB
MAX_ENTRY_BYTES = 2 * GIB
MAX_MANIFEST_BYTES = 8 * 1024**2
MAX_MEMBERS = 20000
FREE_RESERVE_BYTES = GIB
VERSION_NAME = re.compile(r"\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{8}\Z")
PENDING_NAME = re.compile(r"\.\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{8}\.pending\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PUBLISH_SCHEMA = "macfit-pvc-backup-v1"


def real_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Backup directory must not be a symbolic link")


@contextmanager
def regular_file(path: Path, maximum: int):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
            raise ValueError("Archive must be a bounded regular file with no links")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            yield stream
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@contextmanager
def archive_lock(root: Path):
    descriptor = os.open(root / ".backup.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Invalid backup lock")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def checked_json(raw: bytes):
    def invalid_constant(_value):
        raise ValueError("Non-finite JSON is not allowed")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=invalid_constant, object_pairs_hook=unique_object)


def inspect_archive(path: Path) -> dict:
    """Stream every member, checking the manifest and resource limits before extraction."""
    seen, manifest, total, count = {}, None, 0, 0
    with regular_file(path, MAX_ARCHIVE_BYTES) as raw:
        compressed_size = os.fstat(raw.fileno()).st_size
        with tarfile.open(fileobj=raw, mode="r|gz") as archive:
            for member in archive:
                count += 1
                parts = PurePosixPath(member.name).parts
                if (
                    not member.isfile()
                    or member.issparse()
                    or member.name.startswith("/")
                    or ".." in parts
                    or str(PurePosixPath(member.name)) != member.name
                    or count > MAX_MEMBERS
                    or member.name in seen
                    or (member.name == "manifest.json" and manifest is not None)
                    or (member.name != "manifest.json" and not safe_relative(member.name))
                ):
                    raise ValueError("Unsafe or duplicate archive member")
                maximum = MAX_MANIFEST_BYTES if member.name == "manifest.json" else MAX_ENTRY_BYTES
                total += member.size
                if member.size < 0 or member.size > maximum or total > MAX_UNPACKED_BYTES:
                    raise ValueError("Archive exceeds the unpacked storage budget")
                digest, size, chunks = hashlib.sha256(), 0, []
                with archive.extractfile(member) as source:
                    while chunk := source.read(1024 * 1024):
                        size += len(chunk)
                        if size > maximum or size > member.size:
                            raise ValueError("Archive member exceeded its declared size")
                        digest.update(chunk)
                        if member.name == "manifest.json":
                            chunks.append(chunk)
                if size != member.size:
                    raise ValueError("Archive member was truncated")
                if member.name == "manifest.json":
                    manifest = checked_json(b"".join(chunks))
                else:
                    seen[member.name] = {"size_bytes": size, "sha256": digest.hexdigest()}
            end_of_members = archive.offset
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "macfit-private-backup-v1"
        or manifest.get("restore_mode") != "archive_only"
        or not isinstance(manifest.get("files"), list)
        or type(manifest.get("jobs")) is not int
        or not 0 <= manifest["jobs"] <= 1000
        or not isinstance(manifest.get("created_at"), str)
    ):
        raise ValueError("Invalid backup manifest")
    created_at = datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
    if created_at.tzinfo is None:
        raise ValueError("Backup creation time must include its timezone")
    expected = {}
    for item in manifest["files"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "size_bytes", "sha256"}
            or not isinstance(item["path"], str)
            or not safe_relative(item["path"])
            or item["path"] in expected
            or type(item["size_bytes"]) is not int
            or not 0 <= item["size_bytes"] <= MAX_ENTRY_BYTES
            or not isinstance(item["sha256"], str)
            or not SHA256.fullmatch(item["sha256"])
        ):
            raise ValueError("Invalid manifest file declaration")
        expected[item["path"]] = {"size_bytes": item["size_bytes"], "sha256": item["sha256"]}
    if expected != seen or "jobs.sqlite3" not in seen:
        raise ValueError("Archive manifest integrity check failed")
    # tar readers can stop at end-of-tar before reading the gzip CRC/trailer. Consume
    # the complete gzip stream as well, so a truncated transfer cannot be published.
    with regular_file(path, MAX_ARCHIVE_BYTES) as raw, gzip.GzipFile(fileobj=raw) as stream:
        expanded = 0
        while chunk := stream.read(1024 * 1024):
            # Only zero tar padding may follow the last parsed member. Reject a
            # second hidden tar archive or nonzero data after the end marker.
            padding_offset = max(0, end_of_members - expanded)
            if padding_offset < len(chunk) and any(chunk[padding_offset:]):
                raise ValueError("Unexpected data after the end of the tar archive")
            expanded += len(chunk)
            if expanded > MAX_UNPACKED_BYTES + MAX_MEMBERS * 4096 + 10240:
                raise ValueError("Archive padding exceeds the unpacked storage budget")
    return {
        "manifest": manifest,
        "compressed_size_bytes": compressed_size,
        "unpacked_size_bytes": total,
        "file_count": len(seen),
    }


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def current_version(root: Path) -> str | None:
    latest = root / "latest"
    if not latest.is_symlink():
        if latest.exists():
            raise ValueError("latest must be a managed relative symbolic link")
        return None
    target = os.readlink(latest)
    parts = PurePosixPath(target).parts
    if len(parts) != 2 or parts[0] != "versions" or not VERSION_NAME.fullmatch(parts[1]):
        raise ValueError("latest has an unexpected target")
    if not (root / target).is_dir() or (root / target).is_symlink():
        raise ValueError("latest target is unavailable")
    return parts[1]


def verified_versions(root: Path) -> list[Path]:
    result = []
    for directory in (root / "versions").iterdir():
        if (
            not VERSION_NAME.fullmatch(directory.name)
            or directory.is_symlink()
            or not directory.is_dir()
        ):
            continue
        try:
            with regular_file(directory / "metadata.json", 64 * 1024) as stream:
                metadata = checked_json(stream.read(64 * 1024 + 1))
            if (
                metadata.get("schema") == PUBLISH_SCHEMA
                and metadata.get("verified") is True
                and metadata.get("version") == directory.name
            ):
                result.append(directory)
        except (OSError, ValueError, AttributeError):
            continue
    return sorted(result, key=lambda directory: directory.name)


def guard_source_time(root: Path, previous: str | None, incoming: dict) -> None:
    """An older periodic pull must not replace a newer final archive."""
    if previous is None:
        return
    with regular_file(root / "versions" / previous / "metadata.json", 64 * 1024) as stream:
        metadata = checked_json(stream.read(64 * 1024 + 1))
    previous_time = datetime.fromisoformat(metadata["source_created_at"].replace("Z", "+00:00"))
    incoming_time = datetime.fromisoformat(incoming["created_at"].replace("Z", "+00:00"))
    if previous_time.tzinfo is None or incoming_time.tzinfo is None:
        raise ValueError("Snapshot creation times must include a timezone")
    if incoming_time < previous_time:
        raise ValueError("An older source snapshot cannot replace the latest successful backup")


def ensure_space(root: Path, required: int, protected: str | None) -> None:
    """Prefer four copies, but discard older verified copies before filling the PVC."""
    for directory in verified_versions(root):
        if shutil.disk_usage(root).free >= required + FREE_RESERVE_BYTES:
            return
        if directory.name != protected:
            shutil.rmtree(directory)
            sync_directory(root / "versions")
    if shutil.disk_usage(root).free < required + FREE_RESERVE_BYTES:
        raise ValueError(
            "Insufficient backup PVC space; the latest successful snapshot is retained"
        )


def quick_check(data: Path, expected_jobs: int) -> None:
    with sqlite3.connect((data / "jobs.sqlite3").as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("Restored SQLite quick_check failed")
        if connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] != expected_jobs:
            raise ValueError("Restored SQLite job count does not match the manifest")


def sync_tree(root: Path) -> None:
    for directory, _children, filenames in os.walk(root, topdown=False):
        for name in filenames:
            with regular_file(Path(directory) / name, MAX_ENTRY_BYTES) as stream:
                os.fsync(stream.fileno())
        sync_directory(Path(directory))


def publish_archive(archive: Path, root: Path, *, keep: int = 4) -> dict:
    if not 1 <= keep <= 4:
        raise ValueError("Retention must be between one and four snapshots")
    archive, root = archive.absolute(), root.absolute()
    real_directory(root)
    real_directory(root / "versions")
    with archive_lock(root):
        previous = current_version(root)
        snapshots = verified_versions(root)
        # Never choose a deletion candidate when there is no known-good latest pointer.
        if previous is None and snapshots:
            raise ValueError("Existing snapshots require recovery of the latest pointer first")
        if previous is not None and not any(item.name == previous for item in snapshots):
            raise ValueError("The latest snapshot metadata must be recovered before pruning")
        # A terminated pod can leave a private staging directory; only this script's
        # exact staging names are eligible, and the lock excludes any live verifier.
        for directory in (root / "versions").iterdir():
            if (
                PENDING_NAME.fullmatch(directory.name)
                and directory.is_dir()
                and not directory.is_symlink()
            ):
                shutil.rmtree(directory)
        checked = inspect_archive(archive)
        guard_source_time(root, previous, checked["manifest"])
        version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:8]
        staging = root / "versions" / ("." + version + ".pending")
        final = root / "versions" / version
        latest_temp = root / (".latest-" + uuid.uuid4().hex)
        ensure_space(
            root, checked["compressed_size_bytes"] + checked["unpacked_size_bytes"], previous
        )
        staging.mkdir(mode=0o700)
        published = False
        try:
            digest, size = hashlib.sha256(), 0
            with (
                regular_file(archive, MAX_ARCHIVE_BYTES) as source,
                (staging / "archive.tgz").open("xb") as target,
            ):
                os.chmod(staging / "archive.tgz", 0o600)
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise ValueError("Incoming archive grew beyond its storage budget")
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            # Revalidate the exact saved bytes; the incoming file may have changed during copying.
            saved = inspect_archive(staging / "archive.tgz")
            if saved != checked:
                raise ValueError("Incoming archive changed while being saved")
            # This destination must not exist: restore_archive performs a verified atomic restore.
            restore_archive(staging / "archive.tgz", staging / "data")
            quick_check(staging / "data", saved["manifest"]["jobs"])
            metadata = {
                "schema": PUBLISH_SCHEMA,
                "verified": True,
                "archive_only": True,
                "version": version,
                "verified_at": datetime.now(UTC).isoformat(),
                "source_created_at": saved["manifest"]["created_at"],
                "compressed_size_bytes": size,
                "unpacked_size_bytes": saved["unpacked_size_bytes"],
                "archive_sha256": digest.hexdigest(),
                "file_count": saved["file_count"],
                "jobs": saved["manifest"]["jobs"],
                "sqlite_quick_check": "ok",
            }
            write_json(staging / "metadata.json", metadata)
            write_json(staging / "manifest.json", saved["manifest"])
            sync_tree(staging / "data")
            sync_directory(staging)
            os.replace(staging, final)
            sync_directory(root / "versions")
            latest_temp.symlink_to("versions/" + version)
            os.replace(latest_temp, root / "latest")
            published = True
            sync_directory(root)
        finally:
            latest_temp.unlink(missing_ok=True)
            if staging.exists():
                shutil.rmtree(staging)
            if not published and final.exists():
                shutil.rmtree(final)
        # Retention happens after publication and never removes the version latest points to.
        old = [item for item in verified_versions(root) if item.name != version]
        for directory in old[: max(0, len(old) - keep + 1)]:
            shutil.rmtree(directory)
        sync_directory(root / "versions")
        return metadata


def main(argv=None) -> int:
    global MAX_ARCHIVE_BYTES, MAX_UNPACKED_BYTES, MAX_ENTRY_BYTES, FREE_RESERVE_BYTES
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="Verify, restore and publish an incoming snapshot")
    verify.add_argument("--archive", type=Path, default=Path("/archive/incoming/snapshot.tgz"))
    verify.add_argument("--archive-root", type=Path, default=Path("/archive"))
    verify.add_argument("--keep", type=int, default=4)
    verify.add_argument("--max-archive-gib", type=int, default=5)
    verify.add_argument("--max-unpacked-gib", type=int, default=20)
    verify.add_argument("--max-entry-gib", type=int, default=2)
    verify.add_argument("--free-reserve-gib", type=int, default=1)
    args = parser.parse_args(argv)
    if not (
        1 <= args.max_entry_gib <= args.max_unpacked_gib <= 32
        and 1 <= args.max_archive_gib <= 32
        and 1 <= args.free_reserve_gib <= 100
    ):
        parser.error(
            "Require 1 <= entry <= unpacked <= 32 GiB, archive 1..32 GiB, reserve 1..100 GiB"
        )
    MAX_ARCHIVE_BYTES = args.max_archive_gib * GIB
    MAX_UNPACKED_BYTES = args.max_unpacked_gib * GIB
    MAX_ENTRY_BYTES = args.max_entry_gib * GIB
    FREE_RESERVE_BYTES = args.free_reserve_gib * GIB
    try:
        metadata = publish_archive(args.archive, args.archive_root, keep=args.keep)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        EOFError,
        RecursionError,
        sqlite3.Error,
        tarfile.TarError,
        zlib.error,
    ):
        # Archive content and exception strings can include user data. Do not print either.
        print(
            "Backup verification/publication failed; "
            "inspect the retained latest snapshot and PVC capacity.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
