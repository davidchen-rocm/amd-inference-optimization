"""Filesystem-backed experiment store with atomic structured writes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from pydantic import BaseModel

from .models import ArtifactRef, OptimizationTask, WorkflowRecord

ModelT = TypeVar("ModelT", bound=BaseModel)


class StoreError(RuntimeError):
    """Raised for unsafe paths, missing records, and integrity failures."""


class ExperimentStore:
    """Persist task records and immutable evidence under one task directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_task_id(task_id: str) -> None:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not task_id or any(char not in allowed for char in task_id):
            raise StoreError("unsafe task id")

    def task_dir(self, task_id: str) -> Path:
        self._validate_task_id(task_id)
        return self.root / task_id

    def _resolve(self, task_id: str, relative_path: str | Path) -> tuple[Path, str]:
        base = self.task_dir(task_id).resolve()
        portable = PurePosixPath(str(relative_path).replace(os.sep, "/"))
        if portable.is_absolute() or ".." in portable.parts or str(portable) in {"", "."}:
            raise StoreError("path must be task-relative and cannot contain '..'")
        destination = (base / Path(*portable.parts)).resolve(strict=False)
        if not destination.is_relative_to(base):
            raise StoreError("path escapes task directory")
        return destination, str(portable)

    def create_task(self, task: OptimizationTask) -> Path:
        task_dir = self.task_dir(task.id)
        try:
            task_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise StoreError(f"task already exists: {task.id}") from exc
        for directory in (
            "state",
            "events",
            "artifacts",
            "experiments",
            "workspaces",
            "reports",
        ):
            (task_dir / directory).mkdir()
        # This file is intentionally created in place and never atomically replaced.
        # flock locks an inode, so using save_bytes/_atomic_write for it would silently
        # split concurrent callers across different locks.
        lock_fd = os.open(task_dir / ".task.lock", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(lock_fd)
        self.save_json(task.id, "task.json", task, producer="workflow")
        return task_dir

    @contextmanager
    def task_lock(self, task_id: str) -> Iterator[None]:
        """Take the task's stable-inode advisory lock.

        Existing stores are upgraded lazily by creating only the lock file; no
        persisted task, workflow, experiment, or artifact is rewritten.
        """

        directory = self.task_dir(task_id)
        if not directory.is_dir():
            raise StoreError(f"task not found: {task_id}")
        lock_path = directory / ".task.lock"
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise StoreError(f"cannot open task lock: {task_id}") from exc
        try:
            opened = os.fstat(descriptor)
            current = lock_path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != current.st_dev
                or opened.st_ino != current.st_ino
            ):
                raise StoreError("task lock inode changed while being opened")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            # Recheck after waiting: an externally replaced path must never be
            # treated as the same lock domain.
            current = lock_path.stat(follow_symlinks=False)
            if opened.st_dev != current.st_dev or opened.st_ino != current.st_ino:
                raise StoreError("task lock inode was replaced while waiting")
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def load_task(self, task_id: str) -> OptimizationTask:
        return self.load_json(task_id, "task.json", OptimizationTask)

    def save_workflow(self, record: WorkflowRecord) -> ArtifactRef:
        return self.save_json(
            record.task_id,
            "state/workflow.json",
            record,
            producer="workflow",
        )

    def load_workflow(self, task_id: str) -> WorkflowRecord:
        return self.load_json(task_id, "state/workflow.json", WorkflowRecord)

    @staticmethod
    def _json_bytes(value: BaseModel | dict[str, Any] | list[Any]) -> bytes:
        if isinstance(value, BaseModel):
            serializable = value.model_dump(mode="json", by_alias=True)
        else:
            serializable = value
        return (json.dumps(serializable, indent=2, sort_keys=True) + "\n").encode()

    def save_json(
        self,
        task_id: str,
        relative_path: str | Path,
        value: BaseModel | dict[str, Any] | list[Any],
        *,
        producer: str,
    ) -> ArtifactRef:
        return self.save_bytes(
            task_id,
            relative_path,
            self._json_bytes(value),
            producer=producer,
            media_type="application/json",
        )

    def save_evidence_json(
        self,
        task_id: str,
        logical_name: str | Path,
        value: BaseModel | dict[str, Any] | list[Any],
        *,
        producer: str,
    ) -> ArtifactRef:
        """Append a JSON evidence version and never overwrite an earlier version."""

        relative_path = self._next_evidence_path(task_id, logical_name, ".json")
        return self.save_immutable_bytes(
            task_id,
            relative_path,
            self._json_bytes(value),
            producer=producer,
            media_type="application/json",
        )

    def save_immutable_json(
        self,
        task_id: str,
        relative_path: str | Path,
        value: BaseModel | dict[str, Any] | list[Any],
        *,
        producer: str,
    ) -> ArtifactRef:
        """Write one canonical JSON artifact without allowing later replacement."""

        return self.save_immutable_bytes(
            task_id,
            relative_path,
            self._json_bytes(value),
            producer=producer,
            media_type="application/json",
        )

    def save_evidence_text(
        self,
        task_id: str,
        logical_name: str | Path,
        value: str,
        *,
        producer: str,
        media_type: str = "text/plain",
    ) -> ArtifactRef:
        """Append a text evidence version and never overwrite an earlier version."""

        relative_path = self._next_evidence_path(task_id, logical_name, ".txt")
        return self.save_immutable_bytes(
            task_id,
            relative_path,
            value.encode(),
            producer=producer,
            media_type=media_type,
        )

    def load_json(
        self,
        task_id: str,
        relative_path: str | Path,
        model_type: type[ModelT] | None = None,
    ) -> ModelT | Any:
        path, _ = self._resolve(task_id, relative_path)
        if not path.is_file():
            raise StoreError(f"record not found: {relative_path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError(f"invalid JSON record: {relative_path}") from exc
        return model_type.model_validate(value) if model_type is not None else value

    def save_text(
        self,
        task_id: str,
        relative_path: str | Path,
        text: str,
        *,
        producer: str,
        media_type: str = "text/plain",
    ) -> ArtifactRef:
        return self.save_bytes(
            task_id,
            relative_path,
            text.encode(),
            producer=producer,
            media_type=media_type,
        )

    def save_bytes(
        self,
        task_id: str,
        relative_path: str | Path,
        data: bytes,
        *,
        producer: str,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        destination, portable_path = self._resolve(task_id, relative_path)
        if not self.task_dir(task_id).is_dir():
            raise StoreError(f"task not found: {task_id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if portable_path.startswith("artifacts/evidence/") and destination.exists():
            raise StoreError(f"immutable evidence already exists: {portable_path}")
        self._atomic_write(destination, data)
        artifact = ArtifactRef(
            path=portable_path,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
            producer=producer,
            media_type=media_type,
        )
        if portable_path != "artifacts/manifest.json":
            self._register_artifact(task_id, artifact)
        return artifact

    def save_immutable_bytes(
        self,
        task_id: str,
        relative_path: str | Path,
        data: bytes,
        *,
        producer: str,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        """Persist one content-addressed record, refusing replacement at its path."""

        destination, portable_path = self._resolve(task_id, relative_path)
        if not self.task_dir(task_id).is_dir():
            raise StoreError(f"task not found: {task_id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise StoreError(f"immutable artifact already exists: {portable_path}")
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            try:
                os.link(temporary_name, destination)
            except FileExistsError as exc:
                raise StoreError(
                    f"immutable artifact already exists: {portable_path}"
                ) from exc
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        artifact = ArtifactRef(
            path=portable_path,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
            producer=producer,
            media_type=media_type,
        )
        self._register_artifact(task_id, artifact)
        return artifact

    def _next_evidence_path(
        self,
        task_id: str,
        logical_name: str | Path,
        default_suffix: str,
    ) -> str:
        portable = PurePosixPath(str(logical_name).replace(os.sep, "/"))
        if portable.is_absolute() or ".." in portable.parts:
            raise StoreError("evidence name must be relative and cannot contain '..'")
        if str(portable) in {"", "."}:
            raise StoreError("evidence name cannot be empty")
        suffix = portable.suffix or default_suffix
        without_suffix = portable.with_suffix("") if portable.suffix else portable
        directory_relative = PurePosixPath("artifacts/evidence") / without_suffix
        directory, _ = self._resolve(task_id, directory_relative)
        versions: list[int] = []
        if directory.is_dir():
            for path in directory.glob(f"v*{suffix}"):
                number = path.name.removeprefix("v").removesuffix(suffix)
                if number.isdigit():
                    versions.append(int(number))
        version = max(versions, default=0) + 1
        return str(directory_relative / f"v{version:06d}{suffix}")

    def import_evidence(
        self,
        task_id: str,
        logical_name: str | Path,
        source: str | Path,
        *,
        producer: str,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        """Copy an external file into a new immutable evidence version."""

        source_path = Path(source).resolve(strict=True)
        if not source_path.is_file() or source_path.is_symlink():
            raise StoreError("artifact source must be a regular, non-symlink file")
        suffix = source_path.suffix or ".bin"
        relative_path = self._next_evidence_path(task_id, logical_name, suffix)
        return self.import_artifact(
            task_id,
            relative_path,
            source_path,
            producer=producer,
            media_type=media_type,
        )

    def import_artifact(
        self,
        task_id: str,
        relative_path: str | Path,
        source: str | Path,
        *,
        producer: str,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        source_path = Path(source).resolve(strict=True)
        if not source_path.is_file() or source_path.is_symlink():
            raise StoreError("artifact source must be a regular, non-symlink file")
        destination, portable_path = self._resolve(task_id, relative_path)
        if not self.task_dir(task_id).is_dir():
            raise StoreError(f"task not found: {task_id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if portable_path.startswith("artifacts/evidence/") and destination.exists():
            raise StoreError(f"immutable evidence already exists: {portable_path}")
        digest = hashlib.sha256()
        size = 0
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            with os.fdopen(fd, "wb") as target, source_path.open("rb") as incoming:
                while chunk := incoming.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        artifact = ArtifactRef(
            path=portable_path,
            sha256=digest.hexdigest(),
            size=size,
            producer=producer,
            media_type=media_type,
        )
        self._register_artifact(task_id, artifact)
        return artifact

    @staticmethod
    def _atomic_write(destination: Path, data: bytes) -> None:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def _register_artifact(self, task_id: str, artifact: ArtifactRef) -> None:
        manifest_path, _ = self._resolve(task_id, "artifacts/manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {"schema_version": 1, "artifacts": {}}
        manifest["artifacts"][artifact.path] = artifact.model_dump(mode="json")
        self._atomic_write(manifest_path, self._json_bytes(manifest))

    def append_event(self, task_id: str, event: str, payload: dict[str, Any]) -> None:
        path, _ = self._resolve(task_id, "events/events.jsonl")
        if not self.task_dir(task_id).is_dir():
            raise StoreError(f"task not found: {task_id}")
        line = json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "event": event,
                "payload": payload,
            },
            sort_keys=True,
        )
        with path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())

    def verify_artifact(self, task_id: str, artifact: ArtifactRef) -> bool:
        path, _ = self._resolve(task_id, artifact.path)
        if not path.is_file() or path.stat().st_size != artifact.size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest() == artifact.sha256

    def artifact_ref(self, task_id: str, relative_path: str | Path) -> ArtifactRef | None:
        """Return the latest manifest reference for a task-relative artifact path."""

        _, portable_path = self._resolve(task_id, relative_path)
        manifest_path, _ = self._resolve(task_id, "artifacts/manifest.json")
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            value = manifest.get("artifacts", {}).get(portable_path)
            return ArtifactRef.model_validate(value) if value is not None else None
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise StoreError("invalid artifact manifest") from exc

    def register_existing_artifact(
        self,
        task_id: str,
        relative_path: str | Path,
        *,
        producer: str,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        """Hash and register an already-written regular file without rewriting it."""

        path, portable_path = self._resolve(task_id, relative_path)
        if not path.is_file() or path.is_symlink():
            raise StoreError(f"artifact is not a regular file: {portable_path}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        artifact = ArtifactRef(
            path=portable_path,
            sha256=digest.hexdigest(),
            size=size,
            producer=producer,
            media_type=media_type,
        )
        self._register_artifact(task_id, artifact)
        return artifact

    def remove_empty_task(self, task_id: str) -> None:
        """Remove only an empty, failed task creation directory."""

        directory = self.task_dir(task_id)
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
