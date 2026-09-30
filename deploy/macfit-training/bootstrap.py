"""Unpack pinned source or populate persistent CPU-only dependencies, without root."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath


def source_files(raw: bytes) -> dict[str, bytes]:
    total, seen = 0, set()
    files = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            total += member.file_size
            if (
                member.is_dir()
                or name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] != "src"
                or name.suffix != ".py"
                or str(name) != member.filename
                or member.filename in seen
                or stat.S_ISLNK(member.external_attr >> 16)
                or total > 24 * 1024**2
                or len(seen) >= 1024
            ):
                raise ValueError("Unsafe or oversized source bundle")
            seen.add(member.filename)
            files[member.filename] = archive.read(member)
    if not files:
        raise ValueError("The source bundle is empty")
    return files


def verify_source(destination: Path, manifest: dict) -> None:
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("Published source must be a real directory")
    found = set()
    for path in destination.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Published source contains unsupported entries")
        if path.is_file():
            found.add(path.relative_to(destination).as_posix())
    if found != {*manifest["files"], ".source-manifest.json"}:
        raise ValueError("Published source inventory is incomplete or changed")
    marker = destination / ".source-manifest.json"
    if json.loads(marker.read_text()) != manifest:
        raise ValueError("Published source digest marker does not match")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Published source bytes changed after verification")


def source(bundle: Path, destination: Path, expected: str) -> None:
    raw = bundle.read_bytes()
    if len(raw) > 1024**2 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Source bundle digest or size is invalid")
    files = source_files(raw)
    manifest = {
        "bundle_sha256": expected,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = destination.parent / ("." + destination.name + ".source.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pending_pattern = re.compile(
            re.escape("." + destination.name + ".pending-") + "[a-f0-9]{32}\\Z"
        )
        for previous in destination.parent.iterdir():
            if (
                pending_pattern.fullmatch(previous.name)
                and previous.is_dir()
                and not previous.is_symlink()
            ):
                shutil.rmtree(previous)
        if destination.exists() or destination.is_symlink():
            verify_source(destination, manifest)
            return
        staging = destination.parent / ("." + destination.name + ".pending-" + uuid.uuid4().hex)
        staging.mkdir(mode=0o700)
        try:
            for name, contents in files.items():
                target = staging / name
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(contents)
                    stream.flush()
                    os.fsync(stream.fileno())
            with (staging / ".source-manifest.json").open("x") as stream:
                json.dump(manifest, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            verify_source(staging, manifest)
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def vendor(requirements: Path, root: Path) -> None:
    specification = requirements.read_bytes()
    identity = hashlib.sha256(
        specification + f"{sys.version_info[:2]}:{platform.machine()}".encode()
    ).hexdigest()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("Vendor directory must not be a symlink")
    with (root / ".vendor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        final = root / identity
        if not final.exists():
            temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--no-cache-dir",
                        "--only-binary=:all:",
                        "--no-deps",
                        "--target",
                        str(temporary),
                        "--requirement",
                        str(requirements),
                    ],
                    check=True,
                    timeout=900,
                )
                # The lockfile includes transitive CPU dependencies; import before publish.
                environment = {**os.environ, "PYTHONPATH": str(temporary)}
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import fastapi,uvicorn,jwt,cryptography,mcp,yaml; "
                        "import sys; assert 'torch' not in sys.modules",
                    ],
                    check=True,
                    env=environment,
                    timeout=60,
                )
                (temporary / "vendor-manifest.json").write_text(
                    json.dumps(
                        {
                            "identity": identity,
                            "python": platform.python_version(),
                            "requirements_sha256": hashlib.sha256(specification).hexdigest(),
                        }
                    )
                )
                temporary.rename(final)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        manifest = final / "vendor-manifest.json"
        if (
            final.is_symlink()
            or manifest.is_symlink()
            or json.loads(manifest.read_text())["identity"] != identity
        ):
            raise ValueError("Persistent vendor installation failed identity verification")
        temporary_link = root / ".current-pending"
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(identity)
        os.replace(temporary_link, root / "current")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    unpack = commands.add_parser("source")
    unpack.add_argument("--bundle", type=Path, required=True)
    unpack.add_argument("--destination", type=Path, required=True)
    unpack.add_argument("--sha256", required=True)
    packages = commands.add_parser("vendor")
    packages.add_argument("--requirements", type=Path, required=True)
    packages.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "source":
        source(args.bundle, args.destination, args.sha256)
    else:
        vendor(args.requirements, args.root)


if __name__ == "__main__":
    main()
