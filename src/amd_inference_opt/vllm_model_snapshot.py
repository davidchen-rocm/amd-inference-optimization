"""Portable, hash-bound provenance for a local Hugging Face model snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import StrictModel, utc_now

SNAPSHOT_MANIFEST_SCHEMA = "gpuopt.vllm-model-snapshot.v1"


class VLLMModelSnapshotError(RuntimeError):
    """A local model directory cannot be represented by stable evidence."""


def _sha256_fd(descriptor: int) -> tuple[str, int, os.stat_result]:
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(descriptor, "rb", closefd=True) as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        final_stat = os.fstat(stream.fileno())
    return digest.hexdigest(), size, final_stat


def _valid_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("digest must be a lowercase SHA-256 value")
    return value


class VLLMSnapshotFile(StrictModel):
    relative_path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    storage: Literal["regular", "symlink"]

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        portable = PurePosixPath(value)
        if portable.is_absolute() or ".." in portable.parts or value in {"", "."}:
            raise ValueError("snapshot paths must be safe and relative")
        return portable.as_posix()

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _valid_sha256(value)


class VLLMModelSnapshotManifest(StrictModel):
    schema_name: Literal["gpuopt.vllm-model-snapshot.v1"] = Field(
        default=SNAPSHOT_MANIFEST_SCHEMA,
        alias="schema",
    )
    model_id: str
    revision: str
    tokenizer_revision: str
    root: Path
    file_count: int = Field(ge=1)
    total_bytes: int = Field(ge=1)
    files: list[VLLMSnapshotFile] = Field(min_length=1)
    snapshot_digest: str
    captured_at: str = Field(default_factory=lambda: utc_now().isoformat())

    @field_validator("model_id", "revision", "tokenizer_revision")
    @classmethod
    def validate_coordinate(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\x00" in normalized or "\n" in normalized:
            raise ValueError("snapshot coordinates must be non-empty single-line text")
        return normalized

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("snapshot root must be absolute")
        return value

    @field_validator("snapshot_digest")
    @classmethod
    def validate_snapshot_digest(cls, value: str) -> str:
        return _valid_sha256(value)

    @model_validator(mode="after")
    def validate_inventory(self) -> VLLMModelSnapshotManifest:
        paths = [item.relative_path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("snapshot files must be sorted and unique")
        if self.file_count != len(self.files):
            raise ValueError("snapshot file_count does not match files")
        if self.total_bytes != sum(item.size_bytes for item in self.files):
            raise ValueError("snapshot total_bytes does not match files")
        expected = snapshot_identity_sha256(
            model_id=self.model_id,
            revision=self.revision,
            tokenizer_revision=self.tokenizer_revision,
            files=self.files,
        )
        if self.snapshot_digest != expected:
            raise ValueError("snapshot_digest is inconsistent with the file inventory")
        return self


def snapshot_identity_sha256(
    *,
    model_id: str,
    revision: str,
    tokenizer_revision: str,
    files: list[VLLMSnapshotFile],
) -> str:
    document = {
        "schema": "gpuopt.vllm-model-snapshot-identity.v1",
        "model_id": model_id,
        "revision": revision,
        "tokenizer_revision": tokenizer_revision,
        "files": [
            {
                "relative_path": item.relative_path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in files
        ],
    }
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclass(frozen=True)
class _FileBinding:
    lexical: Path
    lexical_identity: tuple[int, int, int, int, int]
    target_path: Path
    target_identity: tuple[int, int, int, int, int]
    link_value: str | None


def _stable_file(*, lexical: Path) -> tuple[VLLMSnapshotFile, _FileBinding]:
    try:
        lexical_before = lexical.lstat()
        is_link = stat.S_ISLNK(lexical_before.st_mode)
        if is_link:
            link_value = os.readlink(lexical)
            declared_target = Path(link_value)
            target_path = (
                declared_target
                if declared_target.is_absolute()
                else lexical.parent / declared_target
            )
            descriptor = os.open(
                target_path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
        else:
            link_value = None
            target_path = lexical
            descriptor = os.open(
                lexical,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
        target_before = os.fstat(descriptor)
        if not stat.S_ISREG(target_before.st_mode):
            os.close(descriptor)
            raise VLLMModelSnapshotError(
                f"model snapshot contains a non-regular file: {lexical}"
            )
        digest, size, target_after_fd = _sha256_fd(descriptor)
        # The descriptor is closed by _sha256_fd; bind the final pathname back
        # to the exact inode that supplied the hashed bytes.
        target_after = target_path.stat()
        lexical_after = lexical.lstat()
    except OSError as error:
        raise VLLMModelSnapshotError(f"cannot hash model file: {lexical}") from error
    if (
        _stat_identity(target_before) != _stat_identity(target_after)
        or _stat_identity(target_before) != _stat_identity(target_after_fd)
        or _stat_identity(lexical_before) != _stat_identity(lexical_after)
        or size != target_after.st_size
        or (is_link and os.readlink(lexical) != link_value)
    ):
        raise VLLMModelSnapshotError(f"model file changed while hashing: {lexical}")
    return (
        VLLMSnapshotFile(
            relative_path="placeholder",
            sha256=digest,
            size_bytes=size,
            storage="symlink" if is_link else "regular",
        ),
        _FileBinding(
            lexical=lexical,
            lexical_identity=_stat_identity(lexical_after),
            target_path=target_path,
            target_identity=_stat_identity(target_after),
            link_value=link_value,
        ),
    )


def _verify_file_binding(binding: _FileBinding) -> None:
    try:
        lexical = binding.lexical.lstat()
        target = binding.target_path.stat()
        link_value = os.readlink(binding.lexical) if binding.link_value is not None else None
    except OSError as error:
        raise VLLMModelSnapshotError(
            f"model file changed after hashing: {binding.lexical}"
        ) from error
    if (
        _stat_identity(lexical) != binding.lexical_identity
        or _stat_identity(target) != binding.target_identity
        or link_value != binding.link_value
    ):
        raise VLLMModelSnapshotError(
            f"model file changed after hashing: {binding.lexical}"
        )


def _walk_error(error: OSError) -> None:
    raise error


def _tree_inventory(
    root: Path,
) -> tuple[list[str], dict[str, tuple[int, int, int, int, int]]]:
    entries: list[str] = []
    directories: dict[str, tuple[int, int, int, int, int]] = {}
    try:
        for directory, directory_names, file_names in os.walk(
            root,
            topdown=True,
            followlinks=False,
            onerror=_walk_error,
        ):
            base = Path(directory)
            relative_directory = base.relative_to(root).as_posix()
            directory_stat = base.stat()
            directories[relative_directory] = _stat_identity(directory_stat)
            for name in directory_names:
                child = base / name
                relative = child.relative_to(root).as_posix()
                if child.is_symlink():
                    raise VLLMModelSnapshotError(
                        "directory symlinks are not allowed in model snapshots: "
                        f"{child}"
                    )
                entries.append(f"d:{relative}")
            for name in file_names:
                lexical = base / name
                relative = lexical.relative_to(root).as_posix()
                mode = lexical.lstat().st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                    raise VLLMModelSnapshotError(
                        f"model snapshot contains a non-regular file: {relative}"
                    )
                entries.append(f"f:{relative}")
    except OSError as error:
        raise VLLMModelSnapshotError(f"cannot enumerate model snapshot: {root}") from error
    entries.sort()
    return entries, directories


def capture_vllm_model_snapshot(
    root: str | Path,
    *,
    model_id: str,
    revision: str,
    tokenizer_revision: str,
) -> VLLMModelSnapshotManifest:
    """Hash every model file without following directory symlinks."""

    lexical_root = Path(root).expanduser()
    if lexical_root.is_symlink():
        raise VLLMModelSnapshotError("model snapshot root must not be a symlink")
    selected_root = lexical_root.resolve()
    if not selected_root.is_dir():
        raise VLLMModelSnapshotError("model snapshot root must be a real directory")
    initial_entries, initial_directories = _tree_inventory(selected_root)
    files: list[VLLMSnapshotFile] = []
    bindings: list[_FileBinding] = []
    for entry in initial_entries:
        if not entry.startswith("f:"):
            continue
        relative = entry[2:]
        evidence, binding = _stable_file(lexical=selected_root / relative)
        files.append(evidence.model_copy(update={"relative_path": relative}))
        bindings.append(binding)
    final_entries, final_directories = _tree_inventory(selected_root)
    if initial_entries != final_entries or initial_directories != final_directories:
        raise VLLMModelSnapshotError(
            "model snapshot directory tree changed while it was being hashed"
        )
    for binding in bindings:
        _verify_file_binding(binding)
    files.sort(key=lambda item: item.relative_path)
    if not files:
        raise VLLMModelSnapshotError("model snapshot contains no files")
    digest = snapshot_identity_sha256(
        model_id=model_id,
        revision=revision,
        tokenizer_revision=tokenizer_revision,
        files=files,
    )
    return VLLMModelSnapshotManifest(
        model_id=model_id,
        revision=revision,
        tokenizer_revision=tokenizer_revision,
        root=selected_root,
        file_count=len(files),
        total_bytes=sum(item.size_bytes for item in files),
        files=files,
        snapshot_digest=digest,
    )


def verify_vllm_model_snapshot(manifest: VLLMModelSnapshotManifest) -> None:
    """Re-capture and require byte-for-byte identity before a paid GPU run."""

    observed = capture_vllm_model_snapshot(
        manifest.root,
        model_id=manifest.model_id,
        revision=manifest.revision,
        tokenizer_revision=manifest.tokenizer_revision,
    )
    if observed.snapshot_digest != manifest.snapshot_digest:
        raise VLLMModelSnapshotError(
            "local model snapshot no longer matches its immutable manifest"
        )


__all__ = [
    "SNAPSHOT_MANIFEST_SCHEMA",
    "VLLMModelSnapshotError",
    "VLLMModelSnapshotManifest",
    "VLLMSnapshotFile",
    "capture_vllm_model_snapshot",
    "snapshot_identity_sha256",
    "verify_vllm_model_snapshot",
]
