"""Private, credential-free SQLite/job exports and validated CPU archive restoration.

Export to a temporary destination and publish it only after this command exits zero:
  python -m macfit_training.service.backup export --data-dir /srv/macfit-training/data
Restore requires a destination that does not exist:
  python -m macfit_training.service.backup restore --archive backup.tar.gz --data-dir /data
Serve a restored archive with MACFIT_ARCHIVE_ONLY=1 and newly supplied gateway credentials.
No training dependencies or GPU are needed for this module.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sqlite3
import sys
import tarfile
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from .outputs import SAFE_NAME, open_regular

MAX_ARCHIVE_BYTES = 32 * 1024**3
MAX_MEMBERS = 20000
MAX_EVIDENCE_FILES = 64
SOURCE_ROOTS = (
    "src",
    "docs/training",
    "examples/training",
    "deploy/macfit-training",
    "web/macfit",
    "tests/training",
)
SOURCE_FILES = {"pyproject.toml", "README", "README.md", "web/macfit/src/nginx.conf"}
SOURCE_SUFFIXES = {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".sh", ".txt", ".csv"}
WEB_SUFFIXES = {".html", ".css", ".js", ".mjs", ".svg", ".webmanifest"}
FORBIDDEN_COMPONENTS = {
    "private",
    "node_modules",
    "__pycache__",
    "venv",
    "credentials",
    "secrets",
    "secret",
}
SENSITIVE_SOURCE_NAMES = {
    "credentials.json",
    "credentials.yaml",
    "credentials.yml",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "service-account.json",
    "service_account.json",
    "serviceaccountkey.json",
    "kubeconfig",
    "kubeconfig.yaml",
    "kubeconfig.yml",
    "id_rsa",
    "id_ed25519",
}
METADATA_LIMITS = {
    "input.json": 3 * 1024**2 + 65536,
    "result.json": 8 * 1024**2,
    "error.json": 16384,
    "events.jsonl": 4 * 1024**2,
    "model-snapshot.json": 8 * 1024**2,
    # ROCm package/file fingerprints on the deployed host already exceed 7 MiB.
    "runtime-environment.json": 16 * 1024**2,
    "training-source.json": 1024**2,
}


def safe_relative(name: str) -> bool:
    parts = PurePosixPath(name).parts
    if not parts or str(PurePosixPath(name)) != name or any(part in {".", ".."} for part in parts):
        return False
    if parts[0] == "source":
        relative = PurePosixPath(*parts[1:])
        return source_allowed(relative)
    if parts[0] == "evidence":
        return (
            len(parts) == 2 and bool(SAFE_NAME.fullmatch(parts[1])) and parts[1].endswith(".json")
        )
    if name == "jobs.sqlite3":
        return True
    if len(parts) not in (3, 4) or parts[0] != "jobs" or str(PurePosixPath(name)) != name:
        return False
    try:
        if str(uuid.UUID(parts[1])) != parts[1]:
            return False
    except ValueError:
        return False
    return (len(parts) == 3 and parts[2] in METADATA_LIMITS) or (
        len(parts) == 4 and parts[2] == "artifacts" and bool(SAFE_NAME.fullmatch(parts[3]))
    )


def source_allowed(relative: PurePosixPath) -> bool:
    if any(part.startswith(".") or part.lower() in FORBIDDEN_COMPONENTS for part in relative.parts):
        return False
    if relative.name.lower() in SENSITIVE_SOURCE_NAMES:
        return False
    name = str(relative)
    return name in SOURCE_FILES or (
        any(name.startswith(root + "/") for root in SOURCE_ROOTS)
        and (
            relative.suffix in SOURCE_SUFFIXES
            or (name.startswith("web/macfit/") and relative.suffix in WEB_SUFFIXES)
        )
    )


def evidence_is_public(value):
    sensitive = {
        "token",
        "secret",
        "password",
        "authorization",
        "access_key",
        "private_key",
        "gateway_secret",
        "id_token",
        "refresh_token",
        "kubeconfig",
        "ssh_key",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower().replace("-", "_") in sensitive:
                raise ValueError("Sensitive fields must not appear in runtime evidence")
            evidence_is_public(item)
    elif isinstance(value, list):
        for item in value:
            evidence_is_public(item)
    elif isinstance(value, str) and "PRIVATE KEY-----" in value:
        raise ValueError("Private keys must not appear in runtime evidence")


def _copy_regular(source: Path, target: Path, maximum: int, *, prefix: bool = False):
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open_regular(source, maximum) as incoming, target.open("xb") as outgoing:
        # Snapshot exactly the initial length. Running events may append afterward.
        remaining = os.fstat(incoming.fileno()).st_size
        digest, size = hashlib.sha256(), 0
        while remaining:
            chunk = incoming.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Backup input changed while reading")
            outgoing.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            remaining -= len(chunk)
        if not prefix and incoming.read(1):
            raise ValueError("Backup input grew while reading")
    target.chmod(0o600)
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def export_archive(
    data_dir: Path,
    output,
    *,
    source_dir: Path | None = None,
    evidence_files: list[Path] | None = None,
):
    data_dir = Path(data_dir).absolute()
    if data_dir.is_symlink() or (data_dir / "jobs").is_symlink():
        raise ValueError("Backup directories must not be symbolic links")
    database = data_dir / "jobs.sqlite3"
    with open_regular(database, 8 * 1024**3):
        pass
    # A separate SQLite connection takes an online consistent snapshot across API writers.
    with tempfile.TemporaryDirectory(prefix="macfit-backup-") as temporary:
        root = Path(temporary)
        snapshot = root / "jobs.sqlite3"
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(snapshot) as destination:
                source.backup(destination, pages=128, sleep=0.05)
        files = []
        with sqlite3.connect(snapshot) as connection:
            connection.row_factory = sqlite3.Row
            jobs = [dict(row) for row in connection.execute("SELECT * FROM jobs ORDER BY id")]
            if len(jobs) > 1000:
                raise ValueError("Backup job limit exceeded")
            # Restoration never replays a queued or interrupted GPU operation.
            connection.execute(
                "UPDATE jobs SET status='failed',stage='failed',error=?,result=NULL,artifacts='[]',"
                "artifact_bytes=0,worker_pid=NULL,worker_start_ticks=NULL "
                "WHERE status IN ('queued','running','cancelling')",
                (
                    json.dumps(
                        {
                            "code": "archive_interrupted",
                            "message": "This job was unfinished when this archive was saved. "
                            "Its inputs are preserved.",
                            "retryable": True,
                        }
                    ),
                ),
            )
            connection.commit()
            connection.execute("PRAGMA journal_mode=DELETE")
        total = snapshot.stat().st_size
        for job in jobs:
            job_id = job["id"]
            if str(uuid.UUID(job_id)) != job_id:
                raise ValueError("Invalid backup job identity")
            directory = data_dir / "jobs" / job_id
            if directory.is_symlink():
                raise ValueError("Backup job directory is a symbolic link")
            for filename, maximum in METADATA_LIMITS.items():
                # A just-published result for a snapshot-running job is not completed evidence.
                if filename == "result.json" and job["status"] != "succeeded":
                    continue
                source = directory / filename
                if filename != "input.json" and not source.exists() and not source.is_symlink():
                    continue
                name = f"jobs/{job_id}/{filename}"
                info = _copy_regular(
                    source, root / name, maximum, prefix=filename == "events.jsonl"
                )
                files.append({"path": name, **info})
                total += info["size_bytes"]
                if filename == "input.json":
                    job_input = json.loads((root / name).read_bytes())
                    identity = json.dumps(
                        {"project_id": job["project_id"], "kind": job["kind"], "input": job_input},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                    if hashlib.sha256(identity).hexdigest() != job["input_hash"]:
                        raise ValueError("Immutable job input failed its backup integrity check")
            if job["status"] == "succeeded":
                declared = json.loads(job["artifacts"])
                if not isinstance(declared, list) or not 1 <= len(declared) <= 16:
                    raise ValueError("Invalid completed job artifacts")
                for artifact in declared:
                    filename = artifact["name"]
                    if not isinstance(filename, str) or not SAFE_NAME.fullmatch(filename):
                        raise ValueError("Invalid artifact filename")
                    name = f"jobs/{job_id}/artifacts/{filename}"
                    info = _copy_regular(directory / "artifacts" / filename, root / name, 1024**3)
                    if (
                        info["size_bytes"] != artifact["size_bytes"]
                        or info["sha256"] != artifact["sha256"]
                    ):
                        raise ValueError("Completed artifact failed its backup integrity check")
                    files.append({"path": name, **info})
                    total += info["size_bytes"]
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("Backup storage limit exceeded")
        if source_dir is not None:
            source_dir = Path(source_dir).absolute()
            if source_dir.is_symlink() or not source_dir.is_dir():
                raise ValueError("Invalid source directory")
            source_total = 0
            for parent, directories, filenames in os.walk(source_dir, followlinks=False):
                relative_parent = Path(parent).relative_to(source_dir)
                directories[:] = [
                    name
                    for name in directories
                    if not name.startswith(".")
                    and name.lower() not in FORBIDDEN_COMPONENTS
                    and any(
                        str(relative_parent / name) == allowed
                        or allowed.startswith(str(relative_parent / name) + "/")
                        or str(relative_parent / name).startswith(allowed + "/")
                        for allowed in SOURCE_ROOTS
                    )
                ]
                for name in directories:
                    if (Path(parent) / name).is_symlink():
                        raise ValueError("Source directories must not be symbolic links")
                for filename in filenames:
                    relative = relative_parent / filename
                    if not source_allowed(PurePosixPath(relative.as_posix())):
                        continue
                    name = "source/" + relative.as_posix()
                    info = _copy_regular(Path(parent) / filename, root / name, 16 * 1024**2)
                    source_total += info["size_bytes"]
                    if source_total > 64 * 1024**2 or len(files) >= MAX_MEMBERS - 2:
                        raise ValueError("Source backup size limit exceeded")
                    files.append({"path": name, **info})
                    total += info["size_bytes"]
        evidence_dir = data_dir / "evidence"
        if evidence_dir.is_symlink():
            raise ValueError("Evidence directories must not be symbolic links")
        selected_evidence = list(evidence_files or []) + sorted(evidence_dir.glob("*.json"))
        if len(selected_evidence) > MAX_EVIDENCE_FILES:
            raise ValueError("Too many runtime evidence files")
        for source in selected_evidence:
            source = Path(source)
            if not SAFE_NAME.fullmatch(source.name) or source.suffix != ".json":
                raise ValueError("Runtime evidence must be a named JSON file")
            name = "evidence/" + source.name
            info = _copy_regular(source, root / name, 1024**2)
            evidence_is_public(json.loads((root / name).read_bytes()))
            files.append({"path": name, **info})
            total += info["size_bytes"]
        if total > MAX_ARCHIVE_BYTES:
            raise ValueError("Backup storage limit exceeded")
        with snapshot.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        files.insert(
            0, {"path": "jobs.sqlite3", "size_bytes": snapshot.stat().st_size, "sha256": digest}
        )
        manifest = {
            "schema": "macfit-private-backup-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "restore_mode": "archive_only",
            "jobs": len(jobs),
            "files": files,
        }
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        with tarfile.open(fileobj=output, mode="w|gz", format=tarfile.PAX_FORMAT) as archive:
            for entry in files:
                info = tarfile.TarInfo(entry["path"])
                info.size, info.mode = entry["size_bytes"], 0o600
                with (root / entry["path"]).open("rb") as handle:
                    archive.addfile(info, handle)
            info = tarfile.TarInfo("manifest.json")
            info.size, info.mode = len(encoded), 0o600
            archive.addfile(info, io.BytesIO(encoded))


def restore_archive(archive_path: Path, data_dir: Path):
    data_dir = Path(data_dir).absolute()
    if data_dir.is_symlink() or data_dir.exists():
        raise ValueError("Restore requires a destination directory that does not exist")
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".macfit-restore-", dir=data_dir.parent) as temporary:
        root = Path(temporary)
        seen, total, manifest = {}, 0, None
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or member.name in seen or len(seen) >= MAX_MEMBERS:
                    raise ValueError("Invalid or duplicate archive member")
                if member.name != "manifest.json" and not safe_relative(member.name):
                    raise ValueError("Unsafe archive member path")
                total += member.size
                if member.size < 0 or total > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive size limit exceeded")
                incoming = archive.extractfile(member)
                if member.name == "manifest.json":
                    if member.size > 8 * 1024**2:
                        raise ValueError("Manifest size limit exceeded")
                    manifest = json.loads(incoming.read())
                    seen[member.name] = None
                    continue
                target = root / member.name
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                digest, size = hashlib.sha256(), 0
                with target.open("xb") as output:
                    while chunk := incoming.read(1024 * 1024):
                        size += len(chunk)
                        digest.update(chunk)
                        output.write(chunk)
                target.chmod(0o600)
                seen[member.name] = {"size_bytes": size, "sha256": digest.hexdigest()}
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != "macfit-private-backup-v1"
            or manifest.get("restore_mode") != "archive_only"
        ):
            raise ValueError("Missing or unsupported backup manifest")
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise ValueError("Invalid backup manifest")
        expected = {}
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"path", "size_bytes", "sha256"}
                or entry["path"] in expected
            ):
                raise ValueError("Invalid backup manifest entry")
            expected[entry["path"]] = {"size_bytes": entry["size_bytes"], "sha256": entry["sha256"]}
        seen.pop("manifest.json", None)
        if seen != expected or "jobs.sqlite3" not in seen:
            raise ValueError("Backup manifest integrity check failed")
        with sqlite3.connect((root / "jobs.sqlite3").as_uri() + "?mode=ro", uri=True) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Restored database failed its integrity check")
            if connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running','cancelling')"
            ).fetchone()[0]:
                raise ValueError("Backup contains runnable GPU jobs")
        (root / "archive-mode.json").write_text(
            json.dumps({"archive_only": True, "created_at": manifest["created_at"]})
        )
        os.replace(root, data_dir)
        data_dir.chmod(0o700)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--data-dir", type=Path, required=True)
    export.add_argument("--source-dir", type=Path)
    export.add_argument("--evidence-file", type=Path, action="append", default=[])
    restore = sub.add_parser("restore")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            export_archive(
                args.data_dir,
                sys.stdout.buffer,
                source_dir=args.source_dir,
                evidence_files=args.evidence_file,
            )
        else:
            restore_archive(args.archive, args.data_dir)
    except (OSError, ValueError, KeyError, sqlite3.Error, tarfile.TarError):
        print(
            "MacFit backup operation failed validation; the destination was not published.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
