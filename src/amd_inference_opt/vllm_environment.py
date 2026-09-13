"""Deterministic native vLLM environment provenance without importing GPU packages."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import StrictModel, utc_now

DEFAULT_REQUIRED_DISTRIBUTIONS = ("vllm", "torch", "amdsmi")
DEFAULT_OPTIONAL_DISTRIBUTIONS = ("triton", "pytorch-triton-rocm", "aiter")


class VLLMEnvironmentError(RuntimeError):
    """The native runtime cannot be represented by immutable local evidence."""


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise VLLMEnvironmentError(f"cannot hash runtime file: {path}") from error
    return digest.hexdigest(), size


def _valid_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


class RuntimeFileDigest(StrictModel):
    relative_path: str
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        portable = PurePosixPath(value)
        if portable.is_absolute() or ".." in portable.parts or value in {"", "."}:
            raise ValueError("runtime file paths must be safe and relative")
        return str(portable)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _valid_sha256(value, "runtime file SHA-256")


class PythonDistributionDigest(StrictModel):
    name: str
    version: str
    status: Literal["present", "missing"]
    file_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    files_sha256: str | None = None
    files: list[RuntimeFileDigest] = Field(default_factory=list)

    @field_validator("name", "version")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\x00" in normalized or "\n" in normalized:
            raise ValueError("distribution coordinates must be non-empty single-line text")
        return normalized

    @field_validator("files_sha256")
    @classmethod
    def validate_files_sha256(cls, value: str | None) -> str | None:
        return _valid_sha256(value, "distribution files SHA-256") if value else None

    @model_validator(mode="after")
    def validate_status(self) -> PythonDistributionDigest:
        if self.status == "present":
            if self.file_count < 1 or self.files_sha256 is None:
                raise ValueError("present distributions require file evidence")
            if self.file_count != len(self.files):
                raise ValueError("distribution file_count does not match its file list")
            paths = [item.relative_path for item in self.files]
            if paths != sorted(paths) or len(paths) != len(set(paths)):
                raise ValueError("distribution files must be sorted and unique")
            if self.total_bytes != sum(item.size_bytes for item in self.files):
                raise ValueError("distribution total_bytes does not match its file list")
            if self.files_sha256 != _files_identity(self.files):
                raise ValueError("distribution files_sha256 is inconsistent with its file list")
        elif self.file_count or self.total_bytes or self.files_sha256 or self.files:
            raise ValueError("missing distributions cannot contain file evidence")
        return self


class VLLMEnvironmentManifest(StrictModel):
    schema_name: Literal["gpuopt.vllm-environment-manifest.v1"] = Field(
        default="gpuopt.vllm-environment-manifest.v1", alias="schema"
    )
    python_executable: Path
    python_executable_resolved: Path
    python_executable_sha256: str
    python_version: str
    python_prefix: Path
    python_base_prefix: Path
    pyvenv_cfg_sha256: str | None = None
    probe_sha256: str
    framework_source_sha256: str
    distributions: list[PythonDistributionDigest]
    required_distributions: list[str]
    identity_sha256: str
    captured_at: str = Field(default_factory=lambda: utc_now().isoformat())

    @field_validator(
        "python_executable_sha256",
        "pyvenv_cfg_sha256",
        "probe_sha256",
        "framework_source_sha256",
        "identity_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str | None) -> str | None:
        return _valid_sha256(value, "environment manifest SHA-256") if value else None

    @field_validator(
        "python_executable",
        "python_executable_resolved",
        "python_prefix",
        "python_base_prefix",
    )
    @classmethod
    def validate_absolute_paths(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("Python environment paths must be absolute")
        return value

    @model_validator(mode="after")
    def validate_inventory(self) -> VLLMEnvironmentManifest:
        names = [item.name.lower() for item in self.distributions]
        if len(names) != len(set(names)):
            raise ValueError("environment distributions must be unique")
        required = [item.lower() for item in self.required_distributions]
        if len(required) != len(set(required)):
            raise ValueError("required distributions must be unique")
        available = {
            item.name.lower() for item in self.distributions if item.status == "present"
        }
        missing = sorted(set(required) - available)
        if missing:
            raise ValueError("required distributions are missing: " + ", ".join(missing))
        expected = environment_identity_sha256(
            python_executable=str(self.python_executable),
            python_executable_resolved=str(self.python_executable_resolved),
            python_executable_sha256=self.python_executable_sha256,
            python_version=self.python_version,
            python_prefix=str(self.python_prefix),
            python_base_prefix=str(self.python_base_prefix),
            pyvenv_cfg_sha256=self.pyvenv_cfg_sha256,
            probe_sha256=self.probe_sha256,
            framework_source_sha256=self.framework_source_sha256,
            distributions=self.distributions,
            required_distributions=self.required_distributions,
        )
        if self.identity_sha256 != expected:
            raise ValueError("environment manifest identity_sha256 is inconsistent")
        return self


def _distribution_files(
    distribution: importlib.metadata.Distribution,
) -> list[RuntimeFileDigest]:
    selected: list[RuntimeFileDigest] = []
    for item in distribution.files or ():
        portable = PurePosixPath(str(item))
        if portable.is_absolute():
            raise VLLMEnvironmentError(
                f"distribution contains an unsafe file path: {item}"
            )
        # Bytecode and cache state are machine-local derivatives of source files.
        if "__pycache__" in portable.parts or portable.suffix in {".pyc", ".pyo"}:
            continue
        lexical = Path(distribution.locate_file(item))
        try:
            resolved = lexical.resolve(strict=True)
        except OSError as error:
            raise VLLMEnvironmentError(
                f"distribution file is missing: {portable.as_posix()}"
            ) from error
        if not resolved.is_file():
            continue
        if ".." in portable.parts:
            prefix = Path(sys.prefix).resolve()
            try:
                prefix_relative = resolved.relative_to(prefix)
            except ValueError as error:
                raise VLLMEnvironmentError(
                    "distribution file escapes the target Python prefix: "
                    f"{portable.as_posix()}"
                ) from error
            evidence_path = PurePosixPath("@prefix", prefix_relative.as_posix())
        else:
            evidence_path = portable
        digest, size = _sha256_file(resolved)
        selected.append(
            RuntimeFileDigest(
                relative_path=evidence_path.as_posix(),
                sha256=digest,
                size_bytes=size,
            )
        )
    selected.sort(key=lambda value: value.relative_path)
    if not selected:
        raise VLLMEnvironmentError(
            f"distribution has no hashable files: {distribution.metadata.get('Name')}"
        )
    return selected


def _files_identity(files: Iterable[RuntimeFileDigest]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item.size_bytes).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def capture_distribution(name: str, *, required: bool) -> PythonDistributionDigest:
    normalized = name.strip()
    if not normalized:
        raise VLLMEnvironmentError("distribution names cannot be empty")
    try:
        distribution = importlib.metadata.distribution(normalized)
    except importlib.metadata.PackageNotFoundError as error:
        if required:
            raise VLLMEnvironmentError(
                f"required Python distribution is not installed: {normalized}"
            ) from error
        return PythonDistributionDigest(
            name=normalized,
            version="missing",
            status="missing",
            file_count=0,
            total_bytes=0,
        )
    files = _distribution_files(distribution)
    return PythonDistributionDigest(
        name=str(distribution.metadata.get("Name") or normalized),
        version=distribution.version,
        status="present",
        file_count=len(files),
        total_bytes=sum(item.size_bytes for item in files),
        files_sha256=_files_identity(files),
        files=files,
    )


def environment_identity_sha256(
    *,
    python_executable: str,
    python_executable_resolved: str,
    python_executable_sha256: str,
    python_version: str,
    python_prefix: str,
    python_base_prefix: str,
    pyvenv_cfg_sha256: str | None,
    probe_sha256: str,
    framework_source_sha256: str,
    distributions: Sequence[PythonDistributionDigest],
    required_distributions: Sequence[str],
) -> str:
    document = {
        "schema": "gpuopt.vllm-environment-identity.v1",
        "python_executable": python_executable,
        "python_executable_resolved": python_executable_resolved,
        "python_executable_sha256": python_executable_sha256,
        "python_version": python_version,
        "python_prefix": python_prefix,
        "python_base_prefix": python_base_prefix,
        "pyvenv_cfg_sha256": pyvenv_cfg_sha256,
        "probe_sha256": probe_sha256,
        "framework_source_sha256": framework_source_sha256,
        "required_distributions": sorted(item.lower() for item in required_distributions),
        "distributions": [
            {
                "name": item.name.lower(),
                "version": item.version,
                "status": item.status,
                "file_count": item.file_count,
                "total_bytes": item.total_bytes,
                "files_sha256": item.files_sha256,
            }
            for item in sorted(distributions, key=lambda value: value.name.lower())
        ],
    }
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def framework_source_identity_sha256() -> str:
    """Bind all project Python modules loaded by probes and lifecycle adapters."""

    root = Path(__file__).resolve(strict=True).parent
    paths = sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    if not paths or len(paths) > 4096:
        raise VLLMEnvironmentError("framework source inventory is empty or unbounded")
    digest = hashlib.sha256()
    for path in paths:
        if path.is_symlink():
            raise VLLMEnvironmentError(f"framework source cannot be a symlink: {path}")
        before = path.stat(follow_symlinks=False)
        file_sha, size = _sha256_file(path)
        after = path.stat(follow_symlinks=False)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise VLLMEnvironmentError(f"framework source changed while hashed: {path}")
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def capture_vllm_environment(
    *,
    python_executable: str | Path | None = None,
    required_distributions: Sequence[str] = DEFAULT_REQUIRED_DISTRIBUTIONS,
    optional_distributions: Sequence[str] = DEFAULT_OPTIONAL_DISTRIBUTIONS,
) -> VLLMEnvironmentManifest:
    """Hash installed package bytes without importing vLLM, torch, or ROCm."""

    selected_input = Path(python_executable or sys.executable).expanduser()
    selected_python = Path(os.path.abspath(os.fspath(selected_input)))
    current_python = Path(os.path.abspath(sys.executable))
    try:
        same_interpreter = os.path.samefile(selected_python, current_python)
    except OSError as error:
        raise VLLMEnvironmentError(
            f"cannot inspect target Python executable: {selected_python}"
        ) from error
    # Virtual environments often symlink to the same base interpreter. Matching
    # that inode alone would label this process's packages as another venv's.
    if not same_interpreter or selected_python != current_python:
        raise VLLMEnvironmentError(
            "capture must run under the target Python executable; invoke that Python "
            "with `-m amd_inference_opt.vllm_environment`"
        )
    try:
        resolved_python = selected_python.resolve(strict=True)
    except OSError as error:
        raise VLLMEnvironmentError("target Python executable is unavailable") from error
    if not resolved_python.is_file():
        raise VLLMEnvironmentError("resolved target Python executable must be a regular file")
    python_sha, _ = _sha256_file(resolved_python)
    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    pyvenv_cfg = prefix / "pyvenv.cfg"
    pyvenv_cfg_sha = _sha256_file(pyvenv_cfg)[0] if pyvenv_cfg.is_file() else None
    probe_path = Path(__file__).resolve(strict=True)
    probe_sha, _ = _sha256_file(probe_path)
    framework_sha = framework_source_identity_sha256()
    required = tuple(dict.fromkeys(item.strip() for item in required_distributions))
    optional = tuple(
        item
        for item in dict.fromkeys(item.strip() for item in optional_distributions)
        if item and item.lower() not in {value.lower() for value in required}
    )
    if not required or any(not item for item in required):
        raise VLLMEnvironmentError("at least one non-empty required distribution is needed")
    distributions = [
        *(capture_distribution(item, required=True) for item in required),
        *(capture_distribution(item, required=False) for item in optional),
    ]
    python_version = sys.version.splitlines()[0]
    identity = environment_identity_sha256(
        python_executable=str(selected_python),
        python_executable_resolved=str(resolved_python),
        python_executable_sha256=python_sha,
        python_version=python_version,
        python_prefix=str(prefix),
        python_base_prefix=str(base_prefix),
        pyvenv_cfg_sha256=pyvenv_cfg_sha,
        probe_sha256=probe_sha,
        framework_source_sha256=framework_sha,
        distributions=distributions,
        required_distributions=required,
    )
    return VLLMEnvironmentManifest(
        python_executable=selected_python,
        python_executable_resolved=resolved_python,
        python_executable_sha256=python_sha,
        python_version=python_version,
        python_prefix=prefix,
        python_base_prefix=base_prefix,
        pyvenv_cfg_sha256=pyvenv_cfg_sha,
        probe_sha256=probe_sha,
        framework_source_sha256=framework_sha,
        distributions=distributions,
        required_distributions=list(required),
        identity_sha256=identity,
    )


def main() -> None:
    manifest = capture_vllm_environment()
    json.dump(
        manifest.model_dump(mode="json", by_alias=True),
        sys.stdout,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write(os.linesep)


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI process
    main()


__all__ = [
    "DEFAULT_OPTIONAL_DISTRIBUTIONS",
    "DEFAULT_REQUIRED_DISTRIBUTIONS",
    "PythonDistributionDigest",
    "RuntimeFileDigest",
    "VLLMEnvironmentError",
    "VLLMEnvironmentManifest",
    "capture_distribution",
    "capture_vllm_environment",
    "environment_identity_sha256",
    "framework_source_identity_sha256",
]
