"""Bounded reads and verified artifact handles; never follow worker-created links."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from .settings import Settings
from .store import ServiceError

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


def open_regular(path: Path, maximum: int):
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
            raise ValueError("Output is not a bounded private regular file")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def read_json(path: Path, maximum: int) -> dict:
    with open_regular(path, maximum) as handle:
        content = handle.read(maximum + 1)
    if len(content) > maximum:
        raise ValueError("Output exceeded its size limit")
    value = json.loads(
        content, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON"))
    )
    if not isinstance(value, dict):
        raise ValueError("Worker output must be an object")
    return value


def hash_handle(handle, maximum: int = 1024**3) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
        if size > maximum:
            raise ValueError("Output exceeded its size limit")
    return size, digest.hexdigest()


def verify_outputs(directory: Path, settings: Settings) -> tuple[dict, list[dict]]:
    result = read_json(directory / "result.json", settings.max_result_bytes)
    declared = result.pop("artifacts", None)
    if not isinstance(declared, list) or not 1 <= len(declared) <= settings.max_artifacts:
        raise ValueError("A successful worker must declare its output artifacts")
    artifact_dir = directory / "artifacts"
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ValueError("Invalid artifact directory")
    actual_names = {entry.name for entry in artifact_dir.iterdir()}
    if len(actual_names) > settings.max_artifacts:
        raise ValueError("Too many artifact files")
    checked, names, identifiers, total = [], set(), set(), 0
    for item in declared:
        if not isinstance(item, dict):
            raise ValueError("Invalid artifact declaration")
        name, artifact_id, kind = item.get("name"), item.get("id"), item.get("type")
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
            raise ValueError("Invalid artifact filename")
        if not isinstance(artifact_id, str) or not SAFE_ID.fullmatch(artifact_id):
            raise ValueError("Invalid artifact identity")
        if not isinstance(kind, str) or not SAFE_ID.fullmatch(kind):
            raise ValueError("Invalid artifact type")
        if name in names or artifact_id in identifiers:
            raise ValueError("Duplicate artifact declaration")
        with open_regular(artifact_dir / name, settings.max_artifact_bytes) as handle:
            size, digest = hash_handle(handle, settings.max_artifact_bytes)
        if (
            type(item.get("size_bytes")) is not int
            or item["size_bytes"] != size
            or item.get("sha256") != digest
        ):
            raise ValueError("Artifact integrity check failed")
        total += size
        if total > settings.max_job_artifact_bytes:
            raise ValueError("Artifacts exceeded the job storage limit")
        checked.append(
            {"id": artifact_id, "type": kind, "name": name, "size_bytes": size, "sha256": digest}
        )
        names.add(name)
        identifiers.add(artifact_id)
    if names != actual_names:
        raise ValueError("Unexpected or incomplete artifact files")
    return result, checked


def artifact_handle(directory: Path, artifact: dict, settings: Settings):
    try:
        name = artifact["name"]
        if not SAFE_NAME.fullmatch(name):
            raise ValueError("Invalid artifact name")
        handle = open_regular(directory / "artifacts" / name, settings.max_artifact_bytes)
        try:
            size, digest = hash_handle(handle, settings.max_artifact_bytes)
            if size != artifact["size_bytes"] or digest != artifact["sha256"]:
                raise ValueError("Artifact integrity check failed")
            handle.seek(0)
            return handle
        except Exception:
            handle.close()
            raise
    except (OSError, ValueError, KeyError) as error:
        raise ServiceError(
            409,
            "artifact_unavailable",
            "This download failed its integrity check. Please contact the site owner.",
        ) from error
