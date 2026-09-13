"""Project-local configuration for the human-facing ``gpuopt`` commands.

The configuration deliberately contains only stable locations and defaults.  A
resolved optimization task remains the source of truth for an individual run.
"""

from __future__ import annotations

import fcntl
import os
import shlex
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, field_validator

from .architecture import canonical_gfx_target

CONFIG_DIRECTORY = ".gpuopt"
CONFIG_FILENAME = "config.yaml"
CONFIG_RELATIVE_PATH = Path(CONFIG_DIRECTORY) / CONFIG_FILENAME


class ProjectConfigError(RuntimeError):
    """Raised when project discovery or configuration validation fails."""


class ProjectConfig(PydanticBaseModel):
    """Versioned, intentionally small project configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)

    schema_name: Literal["gpuopt.project-config.v1"] = Field(
        default="gpuopt.project-config.v1", alias="schema"
    )
    model_root: Path
    store_root: Path = Path(".gpuopt/store")
    llama_cpp_repo: Path | None = None
    llama_cpp_build_dir: Path | None = None
    gpu_device: int = Field(default=0, ge=0)
    gpu_gfx_target: str | None = None
    rocm_mcp_command: list[str] = Field(default_factory=lambda: ["rocm-agent-mcp"])
    default_quality_suites: list[str] = Field(
        default_factory=lambda: ["math-100.v1", "general-100.v1"]
    )

    @field_validator("rocm_mcp_command", "default_quality_suites")
    @classmethod
    def validate_nonempty_lists(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if not normalized or any(not value for value in normalized):
            raise ValueError("configuration lists must contain non-empty values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("configuration lists must not contain duplicates")
        return normalized

    @field_validator("gpu_gfx_target")
    @classmethod
    def validate_gpu_gfx_target(cls, value: str | None) -> str | None:
        return canonical_gfx_target(value) if value is not None else None


class ProjectConfigCheck(PydanticBaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: Literal["OK", "MISSING", "INVALID", "UNSET"]
    path: Path | None = None
    detail: str


_PATH_FIELDS = {
    "model_root",
    "store_root",
    "llama_cpp_repo",
    "llama_cpp_build_dir",
}
_SETTABLE_FIELDS = _PATH_FIELDS | {
    "gpu_device",
    "gpu_gfx_target",
    "rocm_mcp_command",
    "default_quality_suites",
}


def _resolved_root(project_root: str | Path) -> Path:
    candidate = Path(project_root).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProjectConfigError(f"project root does not exist: {candidate}") from exc
    if not resolved.is_dir():
        raise ProjectConfigError(f"project root is not a directory: {resolved}")
    return resolved


def project_config_path(project_root: str | Path) -> Path:
    return _resolved_root(project_root) / CONFIG_RELATIVE_PATH


def discover_project_root(start: str | Path | None = None, *, required: bool = True) -> Path | None:
    """Walk upward from *start* until a project configuration is found."""

    candidate = Path.cwd() if start is None else Path(start).expanduser()
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as exc:
        if required:
            raise ProjectConfigError(f"discovery start does not exist: {candidate}") from exc
        return None
    directory = candidate if candidate.is_dir() else candidate.parent
    for current in (directory, *directory.parents):
        config = current / CONFIG_RELATIVE_PATH
        if config.is_symlink():
            raise ProjectConfigError(f"project config must not be a symlink: {config}")
        if config.is_file():
            return current
    if required:
        raise ProjectConfigError(f"no {CONFIG_RELATIVE_PATH.as_posix()} found from {directory}")
    return None


def _normalize_paths(config: ProjectConfig, project_root: Path) -> ProjectConfig:
    updates: dict[str, Path | None] = {}
    for field in _PATH_FIELDS:
        value = getattr(config, field)
        if value is None:
            updates[field] = None
            continue
        expanded = value.expanduser()
        updates[field] = (
            expanded.resolve(strict=False)
            if expanded.is_absolute()
            else (project_root / expanded).resolve(strict=False)
        )
    return config.model_copy(update=updates)


def _config_bytes(config: ProjectConfig) -> bytes:
    payload = config.model_dump(mode="json", by_alias=True)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).encode("utf-8")


def _validate_control_directory(project_root: Path, *, create: bool) -> Path:
    control = project_root / CONFIG_DIRECTORY
    if control.is_symlink():
        raise ProjectConfigError(f"control directory must not be a symlink: {control}")
    if create:
        control.mkdir(mode=0o755, exist_ok=True)
    if not control.is_dir():
        raise ProjectConfigError(f"control directory is not a directory: {control}")
    return control


def _atomic_write(destination: Path, data: bytes) -> None:
    if destination.is_symlink():
        raise ProjectConfigError(f"refusing to replace symlink: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


@contextmanager
def _config_lock(project_root: Path) -> Iterator[None]:
    control = _validate_control_directory(project_root, create=True)
    lock_path = control / ".config.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ProjectConfigError(f"cannot open project config lock: {lock_path}") from exc
    try:
        opened = os.fstat(descriptor)
        current = lock_path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
        ):
            raise ProjectConfigError("project config lock is not a stable regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        current = lock_path.stat(follow_symlinks=False)
        if opened.st_dev != current.st_dev or opened.st_ino != current.st_ino:
            raise ProjectConfigError("project config lock was replaced while waiting")
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def save_project_config(
    project_root: str | Path, config: ProjectConfig, *, replace: bool = True
) -> Path:
    """Atomically persist a validated config, normalizing paths to absolute paths."""

    root = _resolved_root(project_root)
    normalized = _normalize_paths(config, root)
    with _config_lock(root):
        destination = root / CONFIG_RELATIVE_PATH
        if not replace and (destination.exists() or destination.is_symlink()):
            raise ProjectConfigError(f"project config already exists: {destination}")
        _atomic_write(destination, _config_bytes(normalized))
    return destination


def initialize_project_config(
    project_root: str | Path,
    *,
    model_root: str | Path,
    store_root: str | Path = Path(".gpuopt/store"),
    llama_cpp_repo: str | Path | None = None,
    llama_cpp_build_dir: str | Path | None = None,
    gpu_device: int = 0,
    gpu_gfx_target: str | None = None,
    rocm_mcp_command: list[str] | None = None,
    default_quality_suites: list[str] | None = None,
) -> ProjectConfig:
    config = ProjectConfig(
        model_root=Path(model_root),
        store_root=Path(store_root),
        llama_cpp_repo=Path(llama_cpp_repo) if llama_cpp_repo is not None else None,
        llama_cpp_build_dir=(
            Path(llama_cpp_build_dir) if llama_cpp_build_dir is not None else None
        ),
        gpu_device=gpu_device,
        gpu_gfx_target=gpu_gfx_target,
        rocm_mcp_command=rocm_mcp_command or ["rocm-agent-mcp"],
        default_quality_suites=default_quality_suites or ["math-100.v1", "general-100.v1"],
    )
    save_project_config(project_root, config, replace=False)
    return load_project_config(project_root)


def load_project_config(project_root: str | Path) -> ProjectConfig:
    root = _resolved_root(project_root)
    control = _validate_control_directory(root, create=False)
    path = control / CONFIG_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ProjectConfigError(f"project config not found or unsafe: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("configuration document must be a mapping")
        return _normalize_paths(ProjectConfig.model_validate(raw), root)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise ProjectConfigError(f"invalid project config: {path}") from exc


def _coerce_setting(key: str, value: Any) -> Any:
    if key in _PATH_FIELDS:
        if value is None or (isinstance(value, str) and value.strip().lower() in {"", "null"}):
            return None
        return Path(value)
    if key == "gpu_device":
        return int(value)
    if key == "gpu_gfx_target":
        if value is None or (isinstance(value, str) and value.strip().lower() in {"", "null"}):
            return None
        return str(value)
    if key == "rocm_mcp_command" and isinstance(value, str):
        return shlex.split(value)
    if key == "default_quality_suites" and isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def set_project_config_value(project_root: str | Path, key: str, value: Any) -> ProjectConfig:
    """Set one allow-listed key and atomically replace the configuration."""

    normalized_key = key.strip().replace("-", "_")
    if normalized_key not in _SETTABLE_FIELDS:
        allowed = ", ".join(sorted(_SETTABLE_FIELDS))
        raise ProjectConfigError(f"unsupported project setting {key!r}; allowed: {allowed}")
    root = _resolved_root(project_root)
    with _config_lock(root):
        path = root / CONFIG_RELATIVE_PATH
        if path.is_symlink() or not path.is_file():
            raise ProjectConfigError(f"project config not found or unsafe: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("configuration document must be a mapping")
            config = ProjectConfig.model_validate(raw)
            updated = config.model_copy(
                update={normalized_key: _coerce_setting(normalized_key, value)}
            )
            # model_copy does not revalidate updates in Pydantic v2.
            updated = ProjectConfig.model_validate(updated.model_dump(by_alias=True))
            updated = _normalize_paths(updated, root)
            _atomic_write(path, _config_bytes(updated))
        except (OSError, UnicodeError, yaml.YAMLError, TypeError, ValueError) as exc:
            if isinstance(exc, ProjectConfigError):
                raise
            raise ProjectConfigError(f"cannot update project config: {path}") from exc
    return load_project_config(root)


def check_project_config(config: ProjectConfig) -> list[ProjectConfigCheck]:
    """Return deterministic doctor checks without modifying the filesystem."""

    checks: list[ProjectConfigCheck] = []
    expected = (
        ("model_root", config.model_root, "directory"),
        ("store_root", config.store_root, "directory-or-creatable"),
        ("llama_cpp_repo", config.llama_cpp_repo, "directory"),
        ("llama_cpp_build_dir", config.llama_cpp_build_dir, "directory"),
    )
    for name, path, kind in expected:
        if path is None:
            checks.append(ProjectConfigCheck(name=name, status="UNSET", detail="not configured"))
            continue
        if path.is_symlink():
            status, detail = "INVALID", "symlinks are not accepted"
        elif path.is_dir():
            status, detail = "OK", "directory exists"
        elif kind == "directory-or-creatable" and path.parent.is_dir():
            status, detail = "OK", "directory can be created when first used"
        else:
            status, detail = "MISSING", "directory does not exist"
        checks.append(ProjectConfigCheck(name=name, status=status, path=path, detail=detail))
    return checks


__all__ = [
    "CONFIG_RELATIVE_PATH",
    "ProjectConfig",
    "ProjectConfigCheck",
    "ProjectConfigError",
    "check_project_config",
    "discover_project_root",
    "initialize_project_config",
    "load_project_config",
    "project_config_path",
    "save_project_config",
    "set_project_config_value",
]
