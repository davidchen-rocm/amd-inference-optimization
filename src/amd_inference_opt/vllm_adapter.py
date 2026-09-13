"""Durable, shell-free vLLM server and ``vllm bench serve`` adapter.

The module is deliberately independent from the workflow models.  It can be
tested without vLLM, ROCm, a GPU, or a listening socket by injecting the small
process and HTTP protocols defined below.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

from .command import InvalidCommand, validate_argv
from .models import MetricSeries
from .process_group import (
    ProcessGroupError,
    ProcessGroupLease,
    session_has_live_members,
    terminate_uncaptured_child,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OCI_DIGEST_RE = re.compile(r"(?:[^@\s]+@)?sha256:([0-9a-f]{64})")
_RECORD_SCHEMA = "gpuopt.vllm-server-execution.v1"
_MAX_HEALTH_BODY_BYTES = 4 * 1024 * 1024
_DEFAULT_UNSET_ENV = (
    "HIP_VISIBLE_DEVICES",
    "HSA_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
)
_BENCH_RESERVED_FLAGS = {
    "--backend",
    "--base-url",
    "--endpoint",
    "--model",
    "--dataset-name",
    "--num-prompts",
    "--percentile-metrics",
    "--save-result",
    "--save-detailed",
    "--disable-tqdm",
    "--request-rate",
    "--max-concurrency",
    "--input-len",
    "--output-len",
    "--random-input-len",
    "--random-output-len",
    "--result-dir",
    "--result-filename",
}

REQUEST_THROUGHPUT_METRIC = "request_throughput_requests_per_second"
OUTPUT_THROUGHPUT_METRIC = "output_throughput_tokens_per_second"
TOTAL_THROUGHPUT_METRIC = "total_throughput_tokens_per_second"
MEAN_TTFT_METRIC = "mean_ttft_ms"
MEAN_TPOT_METRIC = "mean_tpot_ms"
MEAN_ITL_METRIC = "mean_itl_ms"
MEAN_E2EL_METRIC = "mean_e2el_ms"

# This is the single public vocabulary for required vLLM serving metrics.  The
# short names accepted/emitted by older adapter callers remain compatibility
# aliases, not gate coordinates.
CANONICAL_BENCH_METRIC_NAMES = (
    REQUEST_THROUGHPUT_METRIC,
    OUTPUT_THROUGHPUT_METRIC,
    TOTAL_THROUGHPUT_METRIC,
    MEAN_TTFT_METRIC,
    MEAN_TPOT_METRIC,
    MEAN_ITL_METRIC,
)
_SHORT_TO_CANONICAL_BENCH_METRIC = MappingProxyType(
    {
        "request_throughput": REQUEST_THROUGHPUT_METRIC,
        "output_tps": OUTPUT_THROUGHPUT_METRIC,
        "total_tps": TOTAL_THROUGHPUT_METRIC,
        "ttft": MEAN_TTFT_METRIC,
        "tpot": MEAN_TPOT_METRIC,
        "itl": MEAN_ITL_METRIC,
        "e2el": MEAN_E2EL_METRIC,
    }
)


class VLLMAdapterError(RuntimeError):
    """Base error for the vLLM adapter."""


class VLLMSpecError(VLLMAdapterError, ValueError):
    """The declared vLLM request is incomplete or unsafe."""


class VLLMRecordError(VLLMAdapterError):
    """A durable execution record is malformed or cannot be persisted."""


class VLLMServerConflictError(VLLMAdapterError):
    """A different, still-live managed server occupies this adapter record."""


class VLLMServerStartError(VLLMAdapterError):
    """The server process could not be started or exited before readiness."""


class VLLMIdentityMismatchError(VLLMServerStartError):
    """Observed runtime evidence contradicts the declared immutable identity."""

    def __init__(self, message: str, *, record: ServerExecutionRecord | None = None) -> None:
        super().__init__(message)
        self.record = record
        self.requires_explicit_stop = bool(record is not None and record.requires_explicit_stop)


class VLLMHealthTimeout(VLLMAdapterError, TimeoutError):
    """The exact server process did not become healthy before its deadline."""

    def __init__(
        self,
        message: str,
        *,
        record: ServerExecutionRecord,
        last_health: VLLMHealthResult,
    ) -> None:
        super().__init__(message)
        self.record = record
        self.last_health = last_health
        self.requires_explicit_stop = record.requires_explicit_stop


class VLLMStopError(VLLMAdapterError):
    """The exact managed process could not be stopped safely."""


class VLLMBenchmarkParseError(VLLMAdapterError, ValueError):
    """A vLLM benchmark JSON artifact lacks trustworthy required metrics."""


def canonical_sha256(value: object) -> str:
    """Return a deterministic SHA-256 for a JSON value."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise VLLMSpecError("identity values must be finite JSON values") from error
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise VLLMSpecError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized or "\x00" in normalized or any(ord(item) < 32 for item in normalized):
        raise VLLMSpecError(f"{field_name} must be non-empty printable text")
    return normalized


def _sha256(value: object, field_name: str) -> str:
    normalized = _nonempty_text(value, field_name)
    if _SHA256_RE.fullmatch(normalized) is None:
        raise VLLMSpecError(f"{field_name} must be 64 lowercase hexadecimal characters")
    return normalized


def _image_digest(value: object) -> tuple[str, str]:
    normalized = _nonempty_text(value, "image_digest")
    if _SHA256_RE.fullmatch(normalized) is not None:
        return f"sha256:{normalized}", normalized
    match = _OCI_DIGEST_RE.fullmatch(normalized)
    if match is None:
        raise VLLMSpecError(
            "image_digest must be 64 hex, sha256:<64 hex>, or image@sha256:<64 hex>"
        )
    return normalized, match.group(1)


def _frozen_environment(value: Mapping[str, str]) -> Mapping[str, str]:
    selected: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or "=" in key or "\x00" in key:
            raise VLLMSpecError(f"invalid environment key: {key!r}")
        if not isinstance(item, str) or "\x00" in item:
            raise VLLMSpecError(f"invalid environment value for {key!r}")
        selected[key] = item
    return MappingProxyType(dict(sorted(selected.items())))


def _frozen_unset_environment(value: Sequence[str], env: Mapping[str, str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise VLLMSpecError("unset_env must be a sequence of environment names")
    selected: list[str] = []
    for key in value:
        if not isinstance(key, str) or not key or "=" in key or "\x00" in key:
            raise VLLMSpecError(f"invalid environment key to unset: {key!r}")
        selected.append(key)
    if len(selected) != len(set(selected)):
        raise VLLMSpecError("unset_env must not contain duplicates")
    overlap = set(selected) & set(env)
    if overlap:
        raise VLLMSpecError(
            "environment variables cannot be both set and unset: " + ", ".join(sorted(overlap))
        )
    return tuple(sorted(selected))


@dataclass(frozen=True)
class VLLMRunIdentity:
    """Four independently comparable identities for one vLLM server run."""

    image_sha256: str
    runtime_sha256: str
    model_sha256: str
    config_sha256: str
    image_digest: str | None
    native_executable_sha256: str
    environment_manifest_sha256: str
    vllm_version: str
    model: str
    model_revision: str
    model_snapshot_sha256: str
    tensor_parallel_size: int
    dtype: str
    quantization: str | None

    def __post_init__(self) -> None:
        for name in ("image_sha256", "runtime_sha256", "model_sha256", "config_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "model_snapshot_sha256",
            _sha256(self.model_snapshot_sha256, "model_snapshot_sha256"),
        )
        if self.model_sha256 != self.model_snapshot_sha256:
            raise VLLMSpecError("model_sha256 must equal the observed model snapshot SHA-256")
        native_sha = _sha256(self.native_executable_sha256, "native_executable_sha256")
        object.__setattr__(self, "native_executable_sha256", native_sha)
        object.__setattr__(
            self,
            "environment_manifest_sha256",
            _sha256(self.environment_manifest_sha256, "environment_manifest_sha256"),
        )
        if self.image_digest is not None:
            image_reference, image_sha = _image_digest(self.image_digest)
            object.__setattr__(self, "image_digest", image_reference)
            if self.image_sha256 != image_sha:
                raise VLLMSpecError("image_sha256 does not match image_digest")
        elif self.image_sha256 != native_sha:
            raise VLLMSpecError("native image_sha256 must equal native_executable_sha256")
        for name in ("vllm_version", "model", "model_revision", "dtype"):
            object.__setattr__(self, name, _nonempty_text(getattr(self, name), name))
        expected_runtime_sha = canonical_sha256(
            {
                "execution_kind": "native-in-oci" if self.image_digest else "native",
                "native_executable_sha256": self.native_executable_sha256,
                "environment_manifest_sha256": self.environment_manifest_sha256,
                "vllm_version": self.vllm_version,
            }
        )
        if self.runtime_sha256 != expected_runtime_sha:
            raise VLLMSpecError(
                "runtime_sha256 does not bind the executable, environment manifest, "
                "vLLM version, and execution kind"
            )
        if self.quantization is not None:
            object.__setattr__(
                self, "quantization", _nonempty_text(self.quantization, "quantization")
            )
        if (
            isinstance(self.tensor_parallel_size, bool)
            or not isinstance(self.tensor_parallel_size, int)
            or self.tensor_parallel_size <= 0
        ):
            raise VLLMSpecError("tensor_parallel_size must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> VLLMRunIdentity:
        expected = {
            "image_sha256",
            "runtime_sha256",
            "model_sha256",
            "config_sha256",
            "image_digest",
            "native_executable_sha256",
            "environment_manifest_sha256",
            "vllm_version",
            "model",
            "model_revision",
            "model_snapshot_sha256",
            "tensor_parallel_size",
            "dtype",
            "quantization",
        }
        if set(value) != expected:
            raise VLLMRecordError("execution record has invalid run identity fields")
        try:
            return cls(**value)  # type: ignore[arg-type]
        except (TypeError, ValueError, VLLMAdapterError) as error:
            raise VLLMRecordError("execution record has an invalid run identity") from error


# A shorter alias is convenient for callers that already live in a vLLM namespace.
RunIdentity = VLLMRunIdentity


@dataclass(frozen=True)
class VLLMServerSpec:
    """Exact, reproducible declaration of a vLLM OpenAI server process.

    The native executable SHA-256 is mandatory because V0 manages a native PID.
    An immutable OCI digest may additionally bind the outer deployment, but V0
    rejects Docker/Podman wrapper argv and must run inside that container.
    ``argv`` is always a sequence and is passed to ``Popen`` with ``shell=False``.
    """

    argv: tuple[str, ...]
    vllm_version: str
    model: str
    model_revision: str
    model_snapshot_sha256: str
    expected_served_model: str
    environment_manifest_sha256: str
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: str | None = None
    image_digest: str | None = None
    native_executable_sha256: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    cwd: str = "."
    env: Mapping[str, str] = field(default_factory=dict)
    unset_env: tuple[str, ...] = _DEFAULT_UNSET_ENV
    host: str = "127.0.0.1"
    port: int = 8000
    health_path: str = "/v1/models"
    startup_timeout_seconds: float = 300.0
    request_timeout_seconds: float = 2.0
    shutdown_timeout_seconds: float = 15.0
    poll_interval_seconds: float = 0.5
    stdout_path: str | None = None
    stderr_path: str | None = None
    _config_json: str = field(init=False, repr=False, compare=False)
    _identity: VLLMRunIdentity = field(init=False, repr=False, compare=False)
    _request_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            frozen_argv = validate_argv(self.argv)
        except InvalidCommand as error:
            raise VLLMSpecError(str(error)) from error
        object.__setattr__(self, "argv", frozen_argv)
        executable = Path(frozen_argv[0])
        if (
            not executable.is_absolute()
            or re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable.name) is None
        ):
            raise VLLMSpecError(
                "V0 argv[0] must be an absolute Python interpreter; do not hash/use the "
                "vllm console script because /proc/PID/exe resolves to Python"
            )
        expected_server_prefix = (
            "-I",
            "-m",
            "vllm.entrypoints.openai.api_server",
        )
        if frozen_argv[1:4] != expected_server_prefix:
            raise VLLMSpecError(
                "V0 server argv must start with isolated project-owned coordinates: "
                "python -I -m vllm.entrypoints.openai.api_server"
            )
        object.__setattr__(self, "vllm_version", _nonempty_text(self.vllm_version, "vllm_version"))
        object.__setattr__(self, "model", _nonempty_text(self.model, "model"))
        object.__setattr__(
            self, "model_revision", _nonempty_text(self.model_revision, "model_revision")
        )
        object.__setattr__(
            self,
            "model_snapshot_sha256",
            _sha256(self.model_snapshot_sha256, "model_snapshot_sha256"),
        )
        object.__setattr__(self, "dtype", _nonempty_text(self.dtype, "dtype"))
        if self.quantization is not None:
            object.__setattr__(
                self, "quantization", _nonempty_text(self.quantization, "quantization")
            )
        if (
            isinstance(self.tensor_parallel_size, bool)
            or not isinstance(self.tensor_parallel_size, int)
            or self.tensor_parallel_size < 1
        ):
            raise VLLMSpecError("tensor_parallel_size must be a positive integer")
        if self.native_executable_sha256 is None:
            raise VLLMSpecError("native_executable_sha256 is required by the V0 native lifecycle")
        image_reference: str | None = None
        native_sha = _sha256(self.native_executable_sha256, "native_executable_sha256")
        object.__setattr__(self, "native_executable_sha256", native_sha)
        environment_manifest_sha = _sha256(
            self.environment_manifest_sha256,
            "environment_manifest_sha256",
        )
        object.__setattr__(
            self,
            "environment_manifest_sha256",
            environment_manifest_sha,
        )
        if self.image_digest is not None:
            image_reference, artifact_sha = _image_digest(self.image_digest)
            object.__setattr__(self, "image_digest", image_reference)
        else:
            artifact_sha = native_sha
        try:
            config_json = json.dumps(
                self.config,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            config_copy = json.loads(config_json)
        except (TypeError, ValueError) as error:
            raise VLLMSpecError("config must contain only finite JSON values") from error
        if not isinstance(config_copy, dict):
            raise VLLMSpecError("config must be a JSON object")
        object.__setattr__(self, "config", MappingProxyType(config_copy))
        object.__setattr__(self, "_config_json", config_json)
        frozen_env = _frozen_environment(self.env)
        if frozen_env.get("PYTHONNOUSERSITE") != "1":
            raise VLLMSpecError("server env must bind PYTHONNOUSERSITE=1")
        object.__setattr__(self, "env", frozen_env)
        object.__setattr__(
            self,
            "unset_env",
            _frozen_unset_environment(self.unset_env, frozen_env),
        )
        missing_required_unsets = set(_DEFAULT_UNSET_ENV) - set(self.unset_env)
        if missing_required_unsets:
            raise VLLMSpecError(
                "server unset_env must remove GPU/Python injection variables: "
                + ", ".join(sorted(missing_required_unsets))
            )

        selected_cwd = Path(self.cwd).expanduser().resolve()
        object.__setattr__(self, "cwd", str(selected_cwd))
        for path_field in ("stdout_path", "stderr_path"):
            value = getattr(self, path_field)
            if value is not None:
                object.__setattr__(self, path_field, str(Path(value).expanduser().resolve()))

        host = _nonempty_text(self.host, "host")
        if any(item in host for item in ("/", "?", "#", "@")) or "://" in host:
            raise VLLMSpecError("host must be a hostname or IP address without a URL scheme")
        object.__setattr__(self, "host", host)
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise VLLMSpecError("port must be between 1 and 65535")
        if not self.health_path.startswith("/") or "\x00" in self.health_path:
            raise VLLMSpecError("health_path must be an absolute URL path")
        object.__setattr__(
            self,
            "expected_served_model",
            _nonempty_text(self.expected_served_model, "expected_served_model"),
        )
        for field_name in (
            "startup_timeout_seconds",
            "request_timeout_seconds",
            "shutdown_timeout_seconds",
            "poll_interval_seconds",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise VLLMSpecError(f"{field_name} must be numeric")
            if not math.isfinite(float(value)) or value <= 0:
                raise VLLMSpecError(f"{field_name} must be finite and positive")

        identity = VLLMRunIdentity(
            image_sha256=artifact_sha,
            runtime_sha256=canonical_sha256(
                {
                    "execution_kind": "native-in-oci" if image_reference else "native",
                    "native_executable_sha256": native_sha,
                    "environment_manifest_sha256": environment_manifest_sha,
                    "vllm_version": self.vllm_version,
                }
            ),
            model_sha256=self.model_snapshot_sha256,
            config_sha256=canonical_sha256(
                {
                    "argv": list(self.argv),
                    "config": config_copy,
                    "cwd": self.cwd,
                    "dtype": self.dtype,
                    "env": dict(self.env),
                    "unset_env": list(self.unset_env),
                    "expected_served_model": self.expected_served_model,
                    "health_path": self.health_path,
                    "host": self.host,
                    "port": self.port,
                    "quantization": self.quantization,
                    "tensor_parallel_size": self.tensor_parallel_size,
                }
            ),
            image_digest=image_reference,
            native_executable_sha256=native_sha,
            environment_manifest_sha256=environment_manifest_sha,
            vllm_version=self.vllm_version,
            model=self.model,
            model_revision=self.model_revision,
            model_snapshot_sha256=self.model_snapshot_sha256,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            quantization=self.quantization,
        )
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(
            self,
            "_request_hash",
            canonical_sha256(
                {
                    "schema": "gpuopt.vllm-server-request.v1",
                    "identity": identity.to_dict(),
                }
            ),
        )

    @property
    def identity(self) -> VLLMRunIdentity:
        return self._identity

    @property
    def request_hash(self) -> str:
        return self._request_hash

    @property
    def health_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"http://{host}:{self.port}{self.health_path}"


class ServerExecutionStatus(StrEnum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    ORPHANED = "ORPHANED"


class RuntimeVerificationStatus(StrEnum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    MISMATCH = "MISMATCH"


@dataclass(frozen=True)
class VLLMRuntimeEvidence:
    """Observed evidence for the process/image that actually started."""

    verification: RuntimeVerificationStatus
    captured_at: str
    observed_pid: int
    observed_boot_id: str
    observed_start_ticks: int
    pid_executable_path: str | None = None
    pid_executable_sha256: str | None = None
    native_executable_matches: bool | None = None
    observed_environment_manifest_sha256: str | None = None
    process_environment_sha256: str | None = None
    declared_environment_sha256: str | None = None
    declared_environment_matches: bool | None = None
    unset_environment_absent: bool | None = None
    unexpected_inherited_environment: tuple[str, ...] = ()
    rocr_visible_devices: str | None = None
    hip_visible_devices: str | None = None
    container_id: str | None = None
    container_init_pid: int | None = None
    container_image_id: str | None = None
    container_binding_kind: str | None = None
    process_binding_id: str | None = None
    container_binding_id: str | None = None
    container_repo_digests: tuple[str, ...] = ()
    container_devices: tuple[str, ...] = ()
    container_environment_sha256: str | None = None
    image_digest_matches: bool | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.observed_pid, bool) or self.observed_pid <= 0:
            raise VLLMRecordError("runtime evidence observed_pid must be positive")
        if not isinstance(self.observed_boot_id, str) or not self.observed_boot_id.strip():
            raise VLLMRecordError("runtime evidence observed_boot_id must be non-empty")
        if isinstance(self.observed_start_ticks, bool) or self.observed_start_ticks <= 0:
            raise VLLMRecordError("runtime evidence observed_start_ticks must be positive")
        for name in (
            "pid_executable_sha256",
            "observed_environment_manifest_sha256",
            "process_environment_sha256",
            "declared_environment_sha256",
            "container_environment_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha256(value, name))
        object.__setattr__(self, "container_repo_digests", tuple(self.container_repo_digests))
        object.__setattr__(self, "container_devices", tuple(self.container_devices))
        object.__setattr__(
            self,
            "unexpected_inherited_environment",
            tuple(self.unexpected_inherited_environment),
        )
        for key in self.unexpected_inherited_environment:
            if not isinstance(key, str) or not key or "=" in key or "\x00" in key:
                raise VLLMRecordError("runtime evidence has invalid inherited environment key")
        if self.container_binding_kind is not None and self.container_binding_kind not in {
            "pid_namespace_inode",
            "cgroup_v2",
        }:
            raise VLLMRecordError("runtime evidence has unsupported container binding kind")
        for name in ("process_binding_id", "container_binding_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty_text(value, name))
        for name in ("rocr_visible_devices", "hip_visible_devices"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty_text(value, name))

    @property
    def verified(self) -> bool:
        return self.verification is RuntimeVerificationStatus.VERIFIED

    @property
    def gate_eligible(self) -> bool:
        return self.verified

    def to_dict(self) -> dict[str, object]:
        return {
            "verification": self.verification.value,
            "captured_at": self.captured_at,
            "observed_pid": self.observed_pid,
            "observed_boot_id": self.observed_boot_id,
            "observed_start_ticks": self.observed_start_ticks,
            "pid_executable_path": self.pid_executable_path,
            "pid_executable_sha256": self.pid_executable_sha256,
            "native_executable_matches": self.native_executable_matches,
            "observed_environment_manifest_sha256": (self.observed_environment_manifest_sha256),
            "process_environment_sha256": self.process_environment_sha256,
            "declared_environment_sha256": self.declared_environment_sha256,
            "declared_environment_matches": self.declared_environment_matches,
            "unset_environment_absent": self.unset_environment_absent,
            "unexpected_inherited_environment": list(self.unexpected_inherited_environment),
            "rocr_visible_devices": self.rocr_visible_devices,
            "hip_visible_devices": self.hip_visible_devices,
            "container_id": self.container_id,
            "container_init_pid": self.container_init_pid,
            "container_image_id": self.container_image_id,
            "container_binding_kind": self.container_binding_kind,
            "process_binding_id": self.process_binding_id,
            "container_binding_id": self.container_binding_id,
            "container_repo_digests": list(self.container_repo_digests),
            "container_devices": list(self.container_devices),
            "container_environment_sha256": self.container_environment_sha256,
            "image_digest_matches": self.image_digest_matches,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> VLLMRuntimeEvidence:
        expected = {
            "verification",
            "captured_at",
            "observed_pid",
            "observed_boot_id",
            "observed_start_ticks",
            "pid_executable_path",
            "pid_executable_sha256",
            "native_executable_matches",
            "observed_environment_manifest_sha256",
            "process_environment_sha256",
            "declared_environment_sha256",
            "declared_environment_matches",
            "unset_environment_absent",
            "unexpected_inherited_environment",
            "rocr_visible_devices",
            "hip_visible_devices",
            "container_id",
            "container_init_pid",
            "container_image_id",
            "container_binding_kind",
            "process_binding_id",
            "container_binding_id",
            "container_repo_digests",
            "container_devices",
            "container_environment_sha256",
            "image_digest_matches",
            "reason",
        }
        if set(value) != expected:
            raise VLLMRecordError("execution record has invalid runtime evidence fields")
        try:
            return cls(
                verification=RuntimeVerificationStatus(value["verification"]),  # type: ignore[arg-type]
                captured_at=value["captured_at"],  # type: ignore[arg-type]
                observed_pid=value["observed_pid"],  # type: ignore[arg-type]
                observed_boot_id=value["observed_boot_id"],  # type: ignore[arg-type]
                observed_start_ticks=value["observed_start_ticks"],  # type: ignore[arg-type]
                pid_executable_path=value["pid_executable_path"],  # type: ignore[arg-type]
                pid_executable_sha256=value["pid_executable_sha256"],  # type: ignore[arg-type]
                native_executable_matches=value["native_executable_matches"],  # type: ignore[arg-type]
                observed_environment_manifest_sha256=value["observed_environment_manifest_sha256"],  # type: ignore[arg-type]
                process_environment_sha256=value["process_environment_sha256"],  # type: ignore[arg-type]
                declared_environment_sha256=value["declared_environment_sha256"],  # type: ignore[arg-type]
                declared_environment_matches=value["declared_environment_matches"],  # type: ignore[arg-type]
                unset_environment_absent=value["unset_environment_absent"],  # type: ignore[arg-type]
                unexpected_inherited_environment=tuple(
                    value["unexpected_inherited_environment"]  # type: ignore[arg-type]
                ),
                rocr_visible_devices=value["rocr_visible_devices"],  # type: ignore[arg-type]
                hip_visible_devices=value["hip_visible_devices"],  # type: ignore[arg-type]
                container_id=value["container_id"],  # type: ignore[arg-type]
                container_init_pid=value["container_init_pid"],  # type: ignore[arg-type]
                container_image_id=value["container_image_id"],  # type: ignore[arg-type]
                container_binding_kind=value["container_binding_kind"],  # type: ignore[arg-type]
                process_binding_id=value["process_binding_id"],  # type: ignore[arg-type]
                container_binding_id=value["container_binding_id"],  # type: ignore[arg-type]
                container_repo_digests=tuple(value["container_repo_digests"]),  # type: ignore[arg-type]
                container_devices=tuple(value["container_devices"]),  # type: ignore[arg-type]
                container_environment_sha256=value["container_environment_sha256"],  # type: ignore[arg-type]
                image_digest_matches=value["image_digest_matches"],  # type: ignore[arg-type]
                reason=value["reason"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError, VLLMAdapterError) as error:
            if isinstance(error, VLLMRecordError):
                raise
            raise VLLMRecordError("execution record has invalid runtime evidence") from error


@dataclass(frozen=True)
class ServerExecutionRecord:
    """Durable identity for one exact OS process and server request."""

    pid: int
    boot_id: str
    start_ticks: int
    request_hash: str
    identity: VLLMRunIdentity
    runtime_evidence: VLLMRuntimeEvidence
    argv: tuple[str, ...]
    cwd: str
    health_url: str
    status: ServerExecutionStatus
    started_at: str
    updated_at: str
    shutdown_timeout_seconds: float
    requires_explicit_stop: bool
    ready_at: str | None = None
    stopped_at: str | None = None
    last_health_at: str | None = None
    error: str | None = None
    schema: str = _RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _RECORD_SCHEMA:
            raise VLLMRecordError(f"unsupported execution record schema: {self.schema!r}")
        if isinstance(self.pid, bool) or self.pid <= 0:
            raise VLLMRecordError("execution record pid must be positive")
        if isinstance(self.start_ticks, bool) or self.start_ticks <= 0:
            raise VLLMRecordError("execution record start_ticks must be positive")
        if not isinstance(self.boot_id, str) or not self.boot_id.strip():
            raise VLLMRecordError("execution record boot_id must be non-empty")
        if _SHA256_RE.fullmatch(self.request_hash) is None:
            raise VLLMRecordError("execution record request_hash must be lowercase SHA-256")
        expected_request_hash = canonical_sha256(
            {
                "schema": "gpuopt.vllm-server-request.v1",
                "identity": self.identity.to_dict(),
            }
        )
        if self.request_hash != expected_request_hash:
            raise VLLMRecordError("execution record request_hash does not match its identity")
        if (
            isinstance(self.shutdown_timeout_seconds, bool)
            or not isinstance(self.shutdown_timeout_seconds, (int, float))
            or not math.isfinite(float(self.shutdown_timeout_seconds))
            or self.shutdown_timeout_seconds <= 0
        ):
            raise VLLMRecordError("execution record shutdown timeout must be finite and positive")
        if not isinstance(self.requires_explicit_stop, bool):
            raise VLLMRecordError("execution record requires_explicit_stop must be boolean")
        try:
            object.__setattr__(self, "argv", validate_argv(self.argv))
        except InvalidCommand as error:
            raise VLLMRecordError("execution record argv is invalid") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "pid": self.pid,
            "boot_id": self.boot_id,
            "start_ticks": self.start_ticks,
            "request_hash": self.request_hash,
            "identity": self.identity.to_dict(),
            "runtime_evidence": self.runtime_evidence.to_dict(),
            "argv": list(self.argv),
            "cwd": self.cwd,
            "health_url": self.health_url,
            "status": self.status.value,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            "requires_explicit_stop": self.requires_explicit_stop,
            "ready_at": self.ready_at,
            "stopped_at": self.stopped_at,
            "last_health_at": self.last_health_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ServerExecutionRecord:
        expected = {
            "schema",
            "pid",
            "boot_id",
            "start_ticks",
            "request_hash",
            "identity",
            "runtime_evidence",
            "argv",
            "cwd",
            "health_url",
            "status",
            "started_at",
            "updated_at",
            "shutdown_timeout_seconds",
            "requires_explicit_stop",
            "ready_at",
            "stopped_at",
            "last_health_at",
            "error",
        }
        if set(value) != expected:
            raise VLLMRecordError("execution record has missing or unknown fields")
        identity_raw = value.get("identity")
        if not isinstance(identity_raw, Mapping):
            raise VLLMRecordError("execution record identity must be an object")
        runtime_evidence_raw = value.get("runtime_evidence")
        if not isinstance(runtime_evidence_raw, Mapping):
            raise VLLMRecordError("execution record runtime_evidence must be an object")
        try:
            return cls(
                schema=value["schema"],  # type: ignore[arg-type]
                pid=value["pid"],  # type: ignore[arg-type]
                boot_id=value["boot_id"],  # type: ignore[arg-type]
                start_ticks=value["start_ticks"],  # type: ignore[arg-type]
                request_hash=value["request_hash"],  # type: ignore[arg-type]
                identity=VLLMRunIdentity.from_dict(identity_raw),
                runtime_evidence=VLLMRuntimeEvidence.from_dict(runtime_evidence_raw),
                argv=tuple(value["argv"]),  # type: ignore[arg-type]
                cwd=value["cwd"],  # type: ignore[arg-type]
                health_url=value["health_url"],  # type: ignore[arg-type]
                status=ServerExecutionStatus(value["status"]),  # type: ignore[arg-type]
                started_at=value["started_at"],  # type: ignore[arg-type]
                updated_at=value["updated_at"],  # type: ignore[arg-type]
                shutdown_timeout_seconds=value["shutdown_timeout_seconds"],  # type: ignore[arg-type]
                requires_explicit_stop=value["requires_explicit_stop"],  # type: ignore[arg-type]
                ready_at=value["ready_at"],  # type: ignore[arg-type]
                stopped_at=value["stopped_at"],  # type: ignore[arg-type]
                last_health_at=value["last_health_at"],  # type: ignore[arg-type]
                error=value["error"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError, VLLMAdapterError) as error:
            if isinstance(error, VLLMRecordError):
                raise
            raise VLLMRecordError("execution record contains invalid values") from error


def load_execution_record(path: str | Path) -> ServerExecutionRecord | None:
    selected = Path(path).expanduser().resolve()
    if not selected.exists():
        return None
    if selected.is_symlink() or not selected.is_file():
        raise VLLMRecordError(f"execution record is not a regular file: {selected}")
    try:
        decoded = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VLLMRecordError(f"cannot read execution record: {selected}") from error
    if not isinstance(decoded, Mapping):
        raise VLLMRecordError("execution record must be a JSON object")
    return ServerExecutionRecord.from_dict(decoded)


def save_execution_record(path: str | Path, record: ServerExecutionRecord) -> None:
    """Atomically replace one durable execution record and fsync its directory."""

    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected.exists() and selected.is_symlink():
        raise VLLMRecordError(f"refusing to replace symlink execution record: {selected}")
    descriptor: int | None = None
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=selected.parent,
            prefix=f".{selected.name}.",
            suffix=".tmp",
            text=True,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            json.dump(
                record.to_dict(),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, selected)
        temporary_name = None
        directory_fd = os.open(selected.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise VLLMRecordError(f"cannot persist execution record: {selected}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class SpawnedProcess:
    pid: int
    boot_id: str
    start_ticks: int


class ProcessBackend(Protocol):
    """Injectable process boundary used by :class:`VLLMAdapter`."""

    def boot_id(self) -> str: ...

    def start_ticks(self, pid: int) -> int | None: ...

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        unset_env: Sequence[str],
        stdout_path: str | None,
        stderr_path: str | None,
    ) -> SpawnedProcess: ...

    def terminate_exact(
        self,
        *,
        pid: int,
        boot_id: str,
        start_ticks: int,
        timeout_seconds: float,
    ) -> bool: ...


def _proc_start_ticks(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    # Field 2 (comm) is parenthesized and may itself contain spaces or ')'.
    closing = raw.rfind(")")
    if closing < 0:
        return None
    fields_from_state = raw[closing + 1 :].strip().split()
    try:
        # starttime is field 22; this tail begins at field 3.
        value = int(fields_from_state[19])
    except (IndexError, ValueError):
        return None
    return value if value > 0 else None


class LocalProcessBackend:
    """Linux process backend with PID-reuse-safe explicit termination."""

    def __init__(
        self,
        *,
        proc_root: str | Path = "/proc",
        boot_id_path: str | Path = "/proc/sys/kernel/random/boot_id",
    ) -> None:
        self.proc_root = Path(proc_root)
        self.boot_id_path = Path(boot_id_path)
        self._children: dict[int, subprocess.Popen[bytes]] = {}

    def boot_id(self) -> str:
        try:
            value = self.boot_id_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise VLLMServerStartError("cannot read the host boot_id") from error
        if not value:
            raise VLLMServerStartError("host boot_id is empty")
        return value

    def start_ticks(self, pid: int) -> int | None:
        child = self._children.get(pid)
        if child is not None and child.poll() is not None:
            self._children.pop(pid, None)
            return None
        return _proc_start_ticks(pid, self.proc_root)

    @staticmethod
    def _open_log(path: str | None) -> Any:
        if path is None:
            return subprocess.DEVNULL
        selected = Path(path)
        selected.parent.mkdir(parents=True, exist_ok=True)
        return selected.open("ab", buffering=0)

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        unset_env: Sequence[str],
        stdout_path: str | None,
        stderr_path: str | None,
    ) -> SpawnedProcess:
        try:
            frozen = validate_argv(argv)
        except InvalidCommand as error:
            raise VLLMServerStartError(str(error)) from error
        try:
            selected_env = _frozen_environment(env)
            selected_unset = _frozen_unset_environment(unset_env, selected_env)
        except VLLMSpecError as error:
            raise VLLMServerStartError(str(error)) from error
        selected_cwd = Path(cwd)
        if not selected_cwd.is_dir():
            raise VLLMServerStartError(f"server cwd is not a directory: {selected_cwd}")
        process_env = os.environ.copy()
        for key in selected_unset:
            process_env.pop(key, None)
        process_env.update(selected_env)
        stdout = self._open_log(stdout_path)
        stderr = self._open_log(stderr_path)
        # Capture host identity before Popen.  A failure here is harmless; doing
        # it afterwards could leave an untracked GPU process alive.
        boot_id = self.boot_id()
        try:
            process = subprocess.Popen(
                list(frozen),
                cwd=selected_cwd,
                env=process_env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                start_new_session=True,
            )
        except OSError as error:
            raise VLLMServerStartError(f"cannot spawn vLLM server: {error}") from error
        finally:
            for handle in (stdout, stderr):
                if handle not in (subprocess.DEVNULL, subprocess.PIPE) and hasattr(handle, "close"):
                    handle.close()
        self._children[process.pid] = process
        rollback_lease: ProcessGroupLease | None = None
        try:
            owned_ticks = _proc_start_ticks(process.pid, self.proc_root)
            if owned_ticks is None:
                raise VLLMServerStartError("cannot capture the spawned session identity")
            rollback_lease = ProcessGroupLease(process.pid, owned_ticks, proc_root=self.proc_root)
            start_ticks = self.start_ticks(process.pid)
            if start_ticks is None:
                return_code = process.poll()
                raise VLLMServerStartError(
                    "vLLM server exited before its process identity was captured"
                    + (f" (exit code {return_code})" if return_code is not None else "")
                )
        except BaseException as error:
            try:
                if rollback_lease is not None:
                    rollback_lease.terminate(grace_seconds=5)
                    process.wait(timeout=5)
                else:
                    terminate_uncaptured_child(process, proc_root=self.proc_root)
            except (ProcessGroupError, subprocess.TimeoutExpired) as cleanup_error:
                error.add_note(f"spawned session cleanup failed: {cleanup_error}")
            self._children.pop(process.pid, None)
            raise
        return SpawnedProcess(
            pid=process.pid,
            boot_id=boot_id,
            start_ticks=start_ticks,
        )

    def _is_exact(self, pid: int, boot_id: str, start_ticks: int) -> bool:
        return self.boot_id() == boot_id and self.start_ticks(pid) == start_ticks

    def session_is_stopped(self, *, pid: int, boot_id: str) -> bool:
        if self.boot_id() != boot_id:
            return True
        try:
            return not session_has_live_members(pid, self.proc_root)
        except ProcessGroupError as error:
            raise VLLMStopError(str(error)) from error

    def terminate_exact(
        self,
        *,
        pid: int,
        boot_id: str,
        start_ticks: int,
        timeout_seconds: float,
    ) -> bool:
        if not self._is_exact(pid, boot_id, start_ticks):
            return False
        try:
            lease = ProcessGroupLease(pid, start_ticks, proc_root=self.proc_root)
            # Keep our Popen leader unreaped while verifying the session.  Its
            # exit does not prove that an EngineCore worker stopped executing.
            lease.terminate(
                grace_seconds=timeout_seconds,
                kill_seconds=min(5.0, timeout_seconds),
            )
        except ProcessGroupError as error:
            raise VLLMStopError(str(error)) from error
        child = self._children.pop(pid, None)
        if child is not None:
            child.wait(timeout=min(5.0, timeout_seconds))
        return True


class RuntimeInspector(Protocol):
    """Capture evidence from the process/container that actually started."""

    def inspect(
        self,
        spec: VLLMServerSpec,
        process: SpawnedProcess,
    ) -> VLLMRuntimeEvidence: ...


def _sha256_regular_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class LocalRuntimeInspector:
    """Inspect the native process; outer-image proof requires injected composition."""

    def __init__(self, *, proc_root: str | Path = "/proc") -> None:
        self.proc_root = Path(proc_root)

    def _pid_executable(self, pid: int) -> tuple[str | None, str | None, str | None]:
        link = self.proc_root / str(pid) / "exe"
        try:
            path = link.resolve(strict=True)
            if not path.is_file():
                return str(path), None, "pid executable is not a regular file"
            return str(path), _sha256_regular_file(path), None
        except OSError as error:
            return None, None, f"cannot hash pid executable: {error}"

    def _pid_environment(self, pid: int) -> tuple[dict[str, str] | None, str | None]:
        try:
            raw = (self.proc_root / str(pid) / "environ").read_bytes()
        except OSError as error:
            return None, f"cannot read pid environment: {error}"
        environment: dict[str, str] = {}
        for entry in raw.split(b"\0"):
            if not entry or b"=" not in entry:
                continue
            key, value = entry.split(b"=", 1)
            environment[key.decode("utf-8", errors="replace")] = value.decode(
                "utf-8", errors="replace"
            )
        return environment, None

    def inspect(
        self,
        spec: VLLMServerSpec,
        process: SpawnedProcess,
    ) -> VLLMRuntimeEvidence:
        captured_at = _utc_now()
        executable_path, executable_sha, executable_error = self._pid_executable(process.pid)
        environment, environment_error = self._pid_environment(process.pid)
        observed_declared = (
            {key: environment[key] for key in spec.env if key in environment}
            if environment is not None
            else None
        )
        unexpected_inherited = (
            tuple(sorted(key for key in spec.unset_env if key in environment))
            if environment is not None
            else ()
        )
        environment_matches = (
            environment is not None
            and not unexpected_inherited
            and all(environment.get(key) == value for key, value in spec.env.items())
        )
        common = {
            "captured_at": captured_at,
            "observed_pid": process.pid,
            "observed_boot_id": process.boot_id,
            "observed_start_ticks": process.start_ticks,
            "pid_executable_path": executable_path,
            "pid_executable_sha256": executable_sha,
            "process_environment_sha256": (
                canonical_sha256(environment) if environment is not None else None
            ),
            "declared_environment_sha256": (
                canonical_sha256(
                    {
                        "set": observed_declared,
                        "unset": sorted(key for key in spec.unset_env if key not in environment),
                    }
                )
                if observed_declared is not None and environment is not None
                else None
            ),
            "declared_environment_matches": environment_matches
            if environment is not None
            else None,
            "unset_environment_absent": (
                not unexpected_inherited if environment is not None else None
            ),
            "unexpected_inherited_environment": unexpected_inherited,
            "rocr_visible_devices": (
                environment.get("ROCR_VISIBLE_DEVICES") if environment is not None else None
            ),
            "hip_visible_devices": (
                environment.get("HIP_VISIBLE_DEVICES") if environment is not None else None
            ),
        }
        if executable_sha is None:
            return VLLMRuntimeEvidence(
                verification=RuntimeVerificationStatus.UNVERIFIED,
                **common,  # type: ignore[arg-type]
                reason=executable_error or "native executable evidence is unavailable",
            )
        native_matches = executable_sha == spec.native_executable_sha256
        if not native_matches:
            return VLLMRuntimeEvidence(
                verification=RuntimeVerificationStatus.MISMATCH,
                **common,  # type: ignore[arg-type]
                native_executable_matches=False,
                reason="pid executable SHA-256 differs from the spec",
            )
        if environment is None:
            return VLLMRuntimeEvidence(
                verification=RuntimeVerificationStatus.UNVERIFIED,
                **common,  # type: ignore[arg-type]
                native_executable_matches=True,
                reason=environment_error or "native process environment evidence is unavailable",
            )
        if not environment_matches:
            return VLLMRuntimeEvidence(
                verification=RuntimeVerificationStatus.MISMATCH,
                **common,  # type: ignore[arg-type]
                native_executable_matches=True,
                reason="native process environment differs from the declared server environment",
            )
        if spec.image_digest is not None:
            return VLLMRuntimeEvidence(
                verification=RuntimeVerificationStatus.UNVERIFIED,
                **common,  # type: ignore[arg-type]
                native_executable_matches=True,
                reason=(
                    "native executable is verified, but LocalRuntimeInspector cannot bind "
                    "the outer image digest to this process"
                ),
            )
        return VLLMRuntimeEvidence(
            verification=RuntimeVerificationStatus.VERIFIED,
            **common,  # type: ignore[arg-type]
            native_executable_matches=True,
        )


class EndpointVerifier(Protocol):
    """Detect pre-existing listeners and bind health to the managed process tree."""

    def is_listening(self, host: str, port: int, *, timeout_seconds: float) -> bool: ...

    def owned_by(self, pid: int, host: str, port: int) -> bool | None: ...


class LocalEndpointVerifier:
    """Linux endpoint ownership verifier based on ``/proc`` socket inodes."""

    def __init__(self, *, proc_root: str | Path = "/proc") -> None:
        self.proc_root = Path(proc_root)

    @staticmethod
    def _connect_host(host: str) -> str:
        if host in {"0.0.0.0", "*"}:
            return "127.0.0.1"
        if host in {"::", "[::]"}:
            return "::1"
        return host.strip("[]")

    def is_listening(self, host: str, port: int, *, timeout_seconds: float) -> bool:
        try:
            with socket.create_connection(
                (self._connect_host(host), port),
                timeout=timeout_seconds,
            ):
                return True
        except (ConnectionRefusedError, TimeoutError, socket.gaierror, OSError):
            return False

    @staticmethod
    def _stat_ppid(raw: str) -> int | None:
        closing = raw.rfind(")")
        if closing < 0:
            return None
        tail = raw[closing + 1 :].strip().split()
        try:
            return int(tail[1])
        except (IndexError, ValueError):
            return None

    def _process_tree(self, root_pid: int) -> set[int] | None:
        try:
            entries = [item for item in self.proc_root.iterdir() if item.name.isdigit()]
        except OSError:
            return None
        children: dict[int, list[int]] = {}
        for entry in entries:
            try:
                ppid = self._stat_ppid((entry / "stat").read_text(encoding="utf-8"))
            except OSError:
                continue
            if ppid is not None:
                children.setdefault(ppid, []).append(int(entry.name))
        selected = {root_pid}
        pending = [root_pid]
        while pending:
            parent = pending.pop()
            for child in children.get(parent, []):
                if child not in selected:
                    selected.add(child)
                    pending.append(child)
        return selected

    def _listening_inodes(self, port: int) -> set[str] | None:
        result: set[str] = set()
        readable = False
        for name in ("tcp", "tcp6"):
            try:
                lines = (self.proc_root / "net" / name).read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            readable = True
            for line in lines[1:]:
                fields = line.split()
                if len(fields) < 10 or fields[3] != "0A":
                    continue
                try:
                    local_port = int(fields[1].rsplit(":", 1)[1], 16)
                except (IndexError, ValueError):
                    continue
                if local_port == port:
                    result.add(fields[9])
        return result if readable else None

    def owned_by(self, pid: int, host: str, port: int) -> bool | None:
        del host  # Port/socket ownership is independent of the selected connect address.
        inodes = self._listening_inodes(port)
        pids = self._process_tree(pid)
        if inodes is None or pids is None:
            return None
        if not inodes:
            return False
        inspected_any = False
        for selected_pid in pids:
            fd_root = self.proc_root / str(selected_pid) / "fd"
            try:
                descriptors = list(fd_root.iterdir())
            except (FileNotFoundError, PermissionError, OSError):
                continue
            inspected_any = True
            for descriptor in descriptors:
                try:
                    target = os.readlink(descriptor)
                except OSError:
                    continue
                match = re.fullmatch(r"socket:\[(\d+)\]", target)
                if match is not None and match.group(1) in inodes:
                    return True
        return False if inspected_any else None


@dataclass(frozen=True)
class HTTPJSONResponse:
    status_code: int
    payload: object


class HTTPClient(Protocol):
    def get_json(self, url: str, *, timeout_seconds: float) -> HTTPJSONResponse: ...


class StdlibHTTPClient:
    """Small stdlib-only JSON client; no requests/aiohttp dependency is required."""

    def __init__(
        self,
        opener: Callable[..., Any] | None = None,
        *,
        max_body_bytes: int = _MAX_HEALTH_BODY_BYTES,
    ) -> None:
        self.opener = opener or urllib.request.urlopen
        self.max_body_bytes = max_body_bytes

    def get_json(self, url: str, *, timeout_seconds: float) -> HTTPJSONResponse:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with self.opener(request, timeout=timeout_seconds) as response:
                status = int(response.getcode())
                raw = response.read(self.max_body_bytes + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise VLLMAdapterError(f"OpenAI health request failed: {error}") from error
        if len(raw) > self.max_body_bytes:
            raise VLLMAdapterError("OpenAI health response exceeded the size limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VLLMAdapterError("OpenAI health response is not valid UTF-8 JSON") from error
        return HTTPJSONResponse(status_code=status, payload=payload)


@dataclass(frozen=True)
class VLLMHealthResult:
    healthy: bool
    process_alive: bool
    endpoint_ownership_verified: bool | None
    checked_at: str
    url: str
    status_code: int | None = None
    models: tuple[str, ...] = ()
    detail: str | None = None


@dataclass(frozen=True)
class ServerStartResult:
    record: ServerExecutionRecord
    health: VLLMHealthResult
    reused: bool

    @property
    def identity_verified(self) -> bool:
        return self.record.runtime_evidence.verified

    @property
    def gate_eligible(self) -> bool:
        return not self.gate_blockers

    @property
    def gate_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if not self.health.healthy:
            blockers.append(self.health.detail or "OpenAI endpoint is not healthy")
        if self.health.endpoint_ownership_verified is not True:
            blockers.append("health endpoint ownership is not verified")
        if not self.identity_verified:
            blockers.append(
                self.record.runtime_evidence.reason or "runtime identity is not verified"
            )
        return tuple(blockers)


VLLMServerStartResult = ServerStartResult


@dataclass(frozen=True)
class ServerStopResult:
    record: ServerExecutionRecord | None
    stopped: bool
    reason: str


VLLMServerStopResult = ServerStopResult


class VLLMAdapter:
    """Start, resume, probe, and explicitly stop one durable vLLM server."""

    def __init__(
        self,
        record_path: str | Path,
        *,
        process_backend: ProcessBackend | None = None,
        runtime_inspector: RuntimeInspector | None = None,
        http_client: HTTPClient | None = None,
        endpoint_verifier: EndpointVerifier | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.record_path = Path(record_path).expanduser().resolve()
        self.lock_path = self.record_path.with_name(f"{self.record_path.name}.lock")
        self.processes = process_backend or LocalProcessBackend()
        self.runtime_inspector = runtime_inspector or LocalRuntimeInspector()
        self.http = http_client or StdlibHTTPClient()
        self.endpoints = endpoint_verifier or LocalEndpointVerifier()
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep

    @contextmanager
    def _lock(self) -> Any:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _same_process(self, record: ServerExecutionRecord) -> bool:
        try:
            return (
                self.processes.boot_id() == record.boot_id
                and self.processes.start_ticks(record.pid) == record.start_ticks
            )
        except VLLMAdapterError:
            raise
        except Exception as error:
            raise VLLMAdapterError("cannot inspect managed server process identity") from error

    def _preflight_record_storage(self) -> None:
        """Prove a first durable record can be created before starting a server."""

        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor: int | None = None
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                dir=self.record_path.parent,
                prefix=f".{self.record_path.name}.preflight.",
                suffix=".tmp",
            )
            os.write(descriptor, b"gpuopt-vllm-record-preflight\n")
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.unlink(temporary)
            temporary = None
        except OSError as error:
            raise VLLMServerStartError(
                f"cannot persist a vLLM execution record before spawn: {error}"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def _rollback_unrecorded_process(
        self,
        process: SpawnedProcess,
        spec: VLLMServerSpec,
        error: BaseException,
    ) -> None:
        """Best-effort exact rollback when no durable record could be written."""

        try:
            stopped = self.processes.terminate_exact(
                pid=process.pid,
                boot_id=process.boot_id,
                start_ticks=process.start_ticks,
                timeout_seconds=spec.shutdown_timeout_seconds,
            )
        except Exception as cleanup_error:
            raise VLLMServerStartError(
                "vLLM start failed before a durable record was written and exact rollback "
                f"also failed for pid {process.pid}: {cleanup_error}"
            ) from error
        if not stopped:
            raise VLLMServerStartError(
                "vLLM start failed before a durable record was written; exact rollback "
                f"could not verify termination of pid {process.pid}"
            ) from error

    def _probe(self, spec: VLLMServerSpec, record: ServerExecutionRecord) -> VLLMHealthResult:
        checked_at = _utc_now()
        if record.request_hash != spec.request_hash:
            return VLLMHealthResult(
                healthy=False,
                process_alive=self._same_process(record),
                endpoint_ownership_verified=None,
                checked_at=checked_at,
                url=spec.health_url,
                detail="execution record request_hash does not match the requested server",
            )
        if not self._same_process(record):
            return VLLMHealthResult(
                healthy=False,
                process_alive=False,
                endpoint_ownership_verified=None,
                checked_at=checked_at,
                url=spec.health_url,
                detail="recorded pid identity is not alive",
            )
        try:
            response = self.http.get_json(
                spec.health_url,
                timeout_seconds=spec.request_timeout_seconds,
            )
        except Exception as error:
            return VLLMHealthResult(
                healthy=False,
                process_alive=True,
                endpoint_ownership_verified=None,
                checked_at=checked_at,
                url=spec.health_url,
                detail=f"{type(error).__name__}: {error}",
            )
        if response.status_code < 200 or response.status_code >= 300:
            return VLLMHealthResult(
                healthy=False,
                process_alive=True,
                endpoint_ownership_verified=None,
                checked_at=checked_at,
                url=spec.health_url,
                status_code=response.status_code,
                detail=f"OpenAI models endpoint returned HTTP {response.status_code}",
            )
        payload = response.payload
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
            return VLLMHealthResult(
                healthy=False,
                process_alive=True,
                endpoint_ownership_verified=None,
                checked_at=checked_at,
                url=spec.health_url,
                status_code=response.status_code,
                detail="OpenAI models endpoint did not return a data array",
            )
        models = tuple(
            item["id"]
            for item in payload["data"]
            if isinstance(item, Mapping) and isinstance(item.get("id"), str) and item["id"].strip()
        )
        if not models:
            return VLLMHealthResult(
                healthy=False,
                process_alive=True,
                endpoint_ownership_verified=None,
                checked_at=checked_at,
                url=spec.health_url,
                status_code=response.status_code,
                detail="OpenAI models endpoint returned no model ids",
            )
        if spec.expected_served_model is not None and spec.expected_served_model not in models:
            return VLLMHealthResult(
                healthy=False,
                process_alive=True,
                endpoint_ownership_verified=None,
                checked_at=checked_at,
                url=spec.health_url,
                status_code=response.status_code,
                models=models,
                detail=f"expected served model {spec.expected_served_model!r} is absent",
            )
        ownership = self.endpoints.owned_by(record.pid, spec.host, spec.port)
        if ownership is False:
            return VLLMHealthResult(
                healthy=False,
                process_alive=True,
                endpoint_ownership_verified=False,
                checked_at=checked_at,
                url=spec.health_url,
                status_code=response.status_code,
                models=models,
                detail="healthy endpoint is not owned by the managed process tree",
            )
        return VLLMHealthResult(
            healthy=True,
            process_alive=True,
            endpoint_ownership_verified=ownership,
            checked_at=checked_at,
            url=spec.health_url,
            status_code=response.status_code,
            models=models,
        )

    def _record_runtime_evidence(
        self,
        spec: VLLMServerSpec,
        process: SpawnedProcess,
        record: ServerExecutionRecord | None = None,
    ) -> tuple[VLLMRuntimeEvidence, ServerExecutionRecord | None]:
        try:
            evidence = self.runtime_inspector.inspect(spec, process)
        except Exception as error:
            evidence = VLLMRuntimeEvidence(
                verification=RuntimeVerificationStatus.UNVERIFIED,
                captured_at=_utc_now(),
                observed_pid=process.pid,
                observed_boot_id=process.boot_id,
                observed_start_ticks=process.start_ticks,
                reason=f"runtime inspection failed: {type(error).__name__}: {error}",
            )
        identity_matches = (
            evidence.observed_pid == process.pid
            and evidence.observed_boot_id == process.boot_id
            and evidence.observed_start_ticks == process.start_ticks
        )
        if not identity_matches:
            evidence = replace(
                evidence,
                verification=RuntimeVerificationStatus.MISMATCH,
                reason="runtime evidence is not bound to the spawned pid identity",
            )
        elif evidence.pid_executable_sha256 is None:
            if evidence.verification is RuntimeVerificationStatus.VERIFIED:
                evidence = replace(
                    evidence,
                    verification=RuntimeVerificationStatus.UNVERIFIED,
                    reason="VERIFIED runtime evidence omitted the pid executable SHA-256",
                )
        elif evidence.pid_executable_sha256 != spec.native_executable_sha256:
            evidence = replace(
                evidence,
                verification=RuntimeVerificationStatus.MISMATCH,
                native_executable_matches=False,
                reason="observed pid executable SHA-256 differs from the spec",
            )
        else:
            evidence = replace(evidence, native_executable_matches=True)

        expected_environment_sha = canonical_sha256(
            {"set": dict(spec.env), "unset": list(spec.unset_env)}
        )
        if evidence.verification is not RuntimeVerificationStatus.MISMATCH:
            if (
                evidence.declared_environment_matches is False
                or evidence.unset_environment_absent is False
                or bool(evidence.unexpected_inherited_environment)
                or (
                    evidence.declared_environment_sha256 is not None
                    and evidence.declared_environment_sha256 != expected_environment_sha
                )
            ):
                evidence = replace(
                    evidence,
                    verification=RuntimeVerificationStatus.MISMATCH,
                    reason="observed native process environment differs from the spec",
                )
            elif (
                evidence.declared_environment_matches is not True
                or evidence.unset_environment_absent is not True
                or evidence.declared_environment_sha256 != expected_environment_sha
                or evidence.process_environment_sha256 is None
            ):
                evidence = replace(
                    evidence,
                    verification=RuntimeVerificationStatus.UNVERIFIED,
                    reason=(
                        evidence.reason
                        or "runtime evidence did not verify the native process environment"
                    ),
                )

        if evidence.verification is not RuntimeVerificationStatus.MISMATCH:
            observed_manifest = evidence.observed_environment_manifest_sha256
            if (
                observed_manifest is not None
                and observed_manifest != spec.environment_manifest_sha256
            ):
                evidence = replace(
                    evidence,
                    verification=RuntimeVerificationStatus.MISMATCH,
                    reason="observed environment manifest SHA-256 differs from the spec",
                )
            elif observed_manifest is None:
                evidence = replace(
                    evidence,
                    verification=RuntimeVerificationStatus.UNVERIFIED,
                    reason=(
                        evidence.reason
                        or "runtime evidence omitted the independently observed "
                        "environment manifest"
                    ),
                )

        if (
            spec.image_digest is not None
            and evidence.verification is not RuntimeVerificationStatus.MISMATCH
        ):
            observed_digests = {
                match.group(1)
                for value in evidence.container_repo_digests
                if (match := _OCI_DIGEST_RE.fullmatch(value)) is not None
            }
            explicit_mismatch = (
                evidence.image_digest_matches is False
                or (bool(observed_digests) and spec.identity.image_sha256 not in observed_digests)
                or (
                    evidence.process_binding_id is not None
                    and evidence.container_binding_id is not None
                    and evidence.process_binding_id != evidence.container_binding_id
                )
            )
            # A PID namespace inode alone is not a container identity: host-PID
            # containers and hostPID pods share it with unrelated processes.
            # Gate eligibility therefore requires matching cgroup-v2 membership.
            same_container = (
                evidence.container_binding_kind == "cgroup_v2"
                and bool(evidence.process_binding_id)
                and evidence.process_binding_id == evidence.container_binding_id
            )
            outer_bound = (
                evidence.image_digest_matches is True
                and spec.identity.image_sha256 in observed_digests
                and bool(evidence.container_id)
                and bool(evidence.container_image_id)
                and same_container
            )
            if explicit_mismatch:
                evidence = replace(
                    evidence,
                    verification=RuntimeVerificationStatus.MISMATCH,
                    reason="outer container evidence contradicts the declared image/process",
                )
            elif not outer_bound:
                evidence = replace(
                    evidence,
                    verification=RuntimeVerificationStatus.UNVERIFIED,
                    reason=(
                        evidence.reason
                        or "outer image evidence lacks matching cgroup-v2 container membership"
                    ),
                )
        if record is not None:
            record = replace(record, runtime_evidence=evidence, updated_at=_utc_now())
            if evidence.verification is RuntimeVerificationStatus.MISMATCH:
                record = replace(
                    record,
                    status=ServerExecutionStatus.FAILED,
                    error=evidence.reason or "observed runtime identity mismatched the spec",
                )
            save_execution_record(self.record_path, record)
        return evidence, record

    def health(self, spec: VLLMServerSpec) -> VLLMHealthResult:
        with self._lock():
            record = load_execution_record(self.record_path)
            if record is None:
                return VLLMHealthResult(
                    healthy=False,
                    process_alive=False,
                    endpoint_ownership_verified=None,
                    checked_at=_utc_now(),
                    url=spec.health_url,
                    detail="no server execution record exists",
                )
            result = self._probe(spec, record)
            updated = replace(
                record,
                status=(ServerExecutionStatus.RUNNING if result.healthy else record.status),
                ready_at=(
                    record.ready_at or result.checked_at if result.healthy else record.ready_at
                ),
                last_health_at=result.checked_at,
                updated_at=_utc_now(),
                error=None if result.healthy else result.detail,
                requires_explicit_stop=result.process_alive,
            )
            save_execution_record(self.record_path, updated)
            return result

    def _wait_for_health(
        self,
        spec: VLLMServerSpec,
        record: ServerExecutionRecord,
        *,
        reused: bool,
    ) -> ServerStartResult:
        deadline = self._monotonic() + spec.startup_timeout_seconds
        last_health = self._probe(spec, record)
        while True:
            if last_health.healthy:
                ready = replace(
                    record,
                    status=ServerExecutionStatus.RUNNING,
                    ready_at=record.ready_at or last_health.checked_at,
                    last_health_at=last_health.checked_at,
                    updated_at=_utc_now(),
                    error=None,
                )
                save_execution_record(self.record_path, ready)
                return ServerStartResult(record=ready, health=last_health, reused=reused)
            if not last_health.process_alive:
                failed = replace(
                    record,
                    status=ServerExecutionStatus.FAILED,
                    last_health_at=last_health.checked_at,
                    updated_at=_utc_now(),
                    error=last_health.detail or "server process exited before readiness",
                    requires_explicit_stop=False,
                )
                save_execution_record(self.record_path, failed)
                raise VLLMServerStartError(failed.error or "vLLM server process exited")
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                timed_out = replace(
                    record,
                    status=ServerExecutionStatus.TIMED_OUT,
                    last_health_at=last_health.checked_at,
                    updated_at=_utc_now(),
                    error=(
                        f"server health timed out after {spec.startup_timeout_seconds} seconds"
                        + (f": {last_health.detail}" if last_health.detail else "")
                    ),
                )
                save_execution_record(self.record_path, timed_out)
                # Deliberately do not terminate here.  Only stop() may signal a process.
                raise VLLMHealthTimeout(
                    timed_out.error or "vLLM server health timed out",
                    record=timed_out,
                    last_health=last_health,
                )
            self._sleep(min(spec.poll_interval_seconds, remaining))
            last_health = self._probe(spec, record)

    def start_or_resume(self, spec: VLLMServerSpec) -> ServerStartResult:
        """Reuse the exact live request or start it once under an exclusive lock."""

        wrapper_names = {"docker", "podman", "nerdctl"}
        if any(Path(item).name.lower() in wrapper_names for item in spec.argv[:3]):
            raise VLLMSpecError(
                "V0 server lifecycle rejects Docker/Podman wrapper argv: run this adapter "
                "inside the pinned container so the recorded PID is the native vLLM server"
            )
        runtime_preflight = getattr(self.runtime_inspector, "preflight", None)
        if runtime_preflight is not None:
            try:
                runtime_preflight(spec)
            except Exception as error:
                raise VLLMServerStartError(
                    "runtime provenance preflight failed before vLLM spawn/resume: "
                    f"{type(error).__name__}: {error}"
                ) from error
        with self._lock():
            existing = load_execution_record(self.record_path)
            if existing is not None and self._same_process(existing):
                if existing.request_hash != spec.request_hash:
                    raise VLLMServerConflictError(
                        "a different managed vLLM server is still alive; explicitly stop it first"
                    )
                _, refreshed = self._record_runtime_evidence(
                    spec,
                    SpawnedProcess(
                        pid=existing.pid,
                        boot_id=existing.boot_id,
                        start_ticks=existing.start_ticks,
                    ),
                    existing,
                )
                assert refreshed is not None
                existing = refreshed
                if existing.runtime_evidence.verification is RuntimeVerificationStatus.MISMATCH:
                    raise VLLMIdentityMismatchError(
                        existing.runtime_evidence.reason
                        or "observed runtime identity mismatched the spec",
                        record=existing,
                    )
                return self._wait_for_health(spec, existing, reused=True)

            if existing is not None:
                checker = getattr(self.processes, "session_is_stopped", None)
                verified = (
                    checker(pid=existing.pid, boot_id=existing.boot_id)
                    if checker is not None
                    else existing.status is ServerExecutionStatus.STOPPED
                )
                if not verified:
                    raise VLLMServerConflictError(
                        "previous server leader exited but its session cleanup is unverified; "
                        "recover the remaining workers before starting another workload"
                    )

            if self.endpoints.is_listening(
                spec.host,
                spec.port,
                timeout_seconds=spec.request_timeout_seconds,
            ):
                raise VLLMServerConflictError(
                    "the requested health port already has a listener not bound to a live "
                    "matching execution record; no process was started"
                )

            self._preflight_record_storage()

            try:
                spawned = self.processes.spawn(
                    spec.argv,
                    cwd=spec.cwd,
                    env=spec.env,
                    unset_env=spec.unset_env,
                    stdout_path=spec.stdout_path,
                    stderr_path=spec.stderr_path,
                )
            except VLLMAdapterError:
                raise
            except Exception as error:
                raise VLLMServerStartError(f"cannot spawn vLLM server: {error}") from error
            try:
                if spawned.pid <= 0 or spawned.start_ticks <= 0 or not spawned.boot_id.strip():
                    raise VLLMServerStartError(
                        "process backend returned an invalid process identity"
                    )
                started_at = _utc_now()
                runtime_evidence, _ = self._record_runtime_evidence(spec, spawned)
                record = ServerExecutionRecord(
                    pid=spawned.pid,
                    boot_id=spawned.boot_id,
                    start_ticks=spawned.start_ticks,
                    request_hash=spec.request_hash,
                    identity=spec.identity,
                    runtime_evidence=runtime_evidence,
                    argv=spec.argv,
                    cwd=spec.cwd,
                    health_url=spec.health_url,
                    status=ServerExecutionStatus.STARTING,
                    started_at=started_at,
                    updated_at=started_at,
                    shutdown_timeout_seconds=spec.shutdown_timeout_seconds,
                    requires_explicit_stop=True,
                )
                save_execution_record(self.record_path, record)
            except BaseException as error:
                self._rollback_unrecorded_process(spawned, spec, error)
                raise
            if runtime_evidence.verification is RuntimeVerificationStatus.MISMATCH:
                failed = replace(
                    record,
                    status=ServerExecutionStatus.FAILED,
                    error=(
                        runtime_evidence.reason or "observed runtime identity mismatched the spec"
                    ),
                    updated_at=_utc_now(),
                )
                save_execution_record(self.record_path, failed)
                raise VLLMIdentityMismatchError(
                    failed.error or "runtime identity mismatch",
                    record=failed,
                )
            return self._wait_for_health(spec, record, reused=False)

    def stop(self, *, expected_request_hash: str | None = None) -> ServerStopResult:
        """Explicitly stop only the PID+boot_id+start_ticks in the durable record."""

        if (
            expected_request_hash is not None
            and _SHA256_RE.fullmatch(expected_request_hash) is None
        ):
            raise VLLMSpecError("expected_request_hash must be lowercase SHA-256")
        with self._lock():
            record = load_execution_record(self.record_path)
            if record is None:
                return ServerStopResult(record=None, stopped=False, reason="no execution record")
            if expected_request_hash is not None and record.request_hash != expected_request_hash:
                raise VLLMServerConflictError(
                    "execution record does not match expected_request_hash; no signal was sent"
                )
            if record.status is ServerExecutionStatus.STOPPED and not self._same_process(record):
                # Crash-window recovery: the process was stopped and that fact was
                # durably recorded, but the coordinator may have crashed before it
                # persisted its own stop artifact.  Preserve terminal success and
                # never reinterpret the absent process as an orphan.
                return ServerStopResult(record=record, stopped=True, reason="already stopped")
            if not self._same_process(record):
                orphaned = replace(
                    record,
                    status=ServerExecutionStatus.ORPHANED,
                    updated_at=_utc_now(),
                    error="recorded pid identity is no longer alive; no signal was sent",
                    requires_explicit_stop=False,
                )
                save_execution_record(self.record_path, orphaned)
                return ServerStopResult(
                    record=orphaned,
                    stopped=False,
                    reason="stale process identity; no signal was sent",
                )
            try:
                stopped = self.processes.terminate_exact(
                    pid=record.pid,
                    boot_id=record.boot_id,
                    start_ticks=record.start_ticks,
                    timeout_seconds=record.shutdown_timeout_seconds,
                )
            except VLLMAdapterError:
                raise
            except Exception as error:
                raise VLLMStopError(f"cannot stop vLLM server: {error}") from error
            if not stopped:
                orphaned = replace(
                    record,
                    status=ServerExecutionStatus.ORPHANED,
                    updated_at=_utc_now(),
                    error="process identity changed before termination; no signal was sent",
                    requires_explicit_stop=False,
                )
                save_execution_record(self.record_path, orphaned)
                return ServerStopResult(
                    record=orphaned,
                    stopped=False,
                    reason="process identity changed; no signal was sent",
                )
            parsed_health = urlsplit(record.health_url)
            health_host = parsed_health.hostname
            health_port = parsed_health.port
            if health_host is None or health_port is None:
                raise VLLMRecordError("execution record contains an invalid health URL")
            if self.endpoints.is_listening(
                health_host,
                health_port,
                timeout_seconds=min(record.shutdown_timeout_seconds, 2.0),
            ):
                orphaned = replace(
                    record,
                    status=ServerExecutionStatus.ORPHANED,
                    updated_at=_utc_now(),
                    error=(
                        "managed leader exited but the health port is still listening; "
                        "a descendant or unrelated process requires manual recovery"
                    ),
                    requires_explicit_stop=False,
                )
                save_execution_record(self.record_path, orphaned)
                return ServerStopResult(
                    record=orphaned,
                    stopped=False,
                    reason="health endpoint remained live after exact process termination",
                )
            stopped_at = _utc_now()
            final = replace(
                record,
                status=ServerExecutionStatus.STOPPED,
                stopped_at=stopped_at,
                updated_at=stopped_at,
                error=None,
                requires_explicit_stop=False,
            )
            save_execution_record(self.record_path, final)
            return ServerStopResult(record=final, stopped=True, reason="stopped")


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VLLMSpecError(f"{field_name} must be a positive integer")
    return value


def _positive_rate(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VLLMSpecError(f"{field_name} must be numeric")
    selected = float(value)
    if math.isnan(selected) or selected <= 0:
        raise VLLMSpecError(f"{field_name} must be positive")
    return selected


def build_bench_serve_argv(
    *,
    model: str,
    base_url: str,
    backend: str = "openai",
    endpoint: str = "/v1/completions",
    dataset_name: str = "random",
    num_prompts: int = 100,
    request_rate: float | None = None,
    max_concurrency: int | None = None,
    random_input_len: int | None = None,
    random_output_len: int | None = None,
    result_dir: str | Path | None = None,
    result_filename: str | None = None,
    save_detailed: bool = True,
    disable_tqdm: bool = True,
    percentile_metrics: Sequence[str] = ("ttft", "tpot", "itl"),
    vllm_argv: Sequence[str] = ("vllm",),
    extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    """Build a deterministic argv for current ``vllm bench serve``.

    ``vllm_argv`` permits an explicit interpreter/module prefix while retaining
    the invariant that no shell command string is accepted.
    """

    try:
        prefix = validate_argv(vllm_argv)
        extras = validate_argv(extra_args) if extra_args else ()
    except InvalidCommand as error:
        raise VLLMSpecError(str(error)) from error
    if Path(prefix[0]).name.lower() in {
        "bash",
        "cmd",
        "dash",
        "fish",
        "powershell",
        "pwsh",
        "sh",
        "zsh",
    }:
        raise VLLMSpecError("vllm_argv must not invoke a shell interpreter")
    conflicting = sorted(
        {
            item.split("=", 1)[0]
            for item in (*prefix, *extras)
            if item.split("=", 1)[0] in _BENCH_RESERVED_FLAGS
        }
    )
    if conflicting:
        raise VLLMSpecError(
            "vllm_argv/extra_args cannot override locked benchmark flags: " + ", ".join(conflicting)
        )
    model = _nonempty_text(model, "model")
    backend = _nonempty_text(backend, "backend")
    dataset_name = _nonempty_text(dataset_name, "dataset_name")
    base_url = _nonempty_text(base_url, "base_url")
    endpoint = _nonempty_text(endpoint, "endpoint")
    if not isinstance(save_detailed, bool) or not isinstance(disable_tqdm, bool):
        raise VLLMSpecError("save_detailed and disable_tqdm must be boolean")
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise VLLMSpecError("base_url must be an absolute http(s) URL")
    if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
        raise VLLMSpecError("base_url must not contain credentials, a query, or a fragment")
    if not endpoint.startswith("/") or "\x00" in endpoint:
        raise VLLMSpecError("endpoint must be an absolute URL path")
    _positive_integer(num_prompts, "num_prompts")
    for name, value in (
        ("max_concurrency", max_concurrency),
        ("random_input_len", random_input_len),
        ("random_output_len", random_output_len),
    ):
        if value is not None:
            _positive_integer(value, name)
    if request_rate is not None:
        _positive_rate(request_rate, "request_rate")
    metrics = tuple(_nonempty_text(item, "percentile metric") for item in percentile_metrics)
    if not metrics or any(item not in {"ttft", "tpot", "itl", "e2el"} for item in metrics):
        raise VLLMSpecError("percentile_metrics must use ttft, tpot, itl, or e2el")
    if len(set(metrics)) != len(metrics):
        raise VLLMSpecError("percentile_metrics must not contain duplicates")
    if result_filename is not None:
        result_filename = _nonempty_text(result_filename, "result_filename")
        if Path(result_filename).name != result_filename or result_filename in {".", ".."}:
            raise VLLMSpecError("result_filename must be a filename, not a path")

    argv = [
        *prefix,
        "bench",
        "serve",
        "--backend",
        backend,
        "--base-url",
        base_url.rstrip("/"),
        "--endpoint",
        endpoint,
        "--model",
        model,
        "--dataset-name",
        dataset_name,
        "--num-prompts",
        str(num_prompts),
        "--percentile-metrics",
        ",".join(metrics),
        "--save-result",
    ]
    if save_detailed:
        argv.append("--save-detailed")
    if disable_tqdm:
        argv.append("--disable-tqdm")
    if request_rate is not None:
        rendered_rate = "inf" if math.isinf(float(request_rate)) else str(request_rate)
        argv.extend(["--request-rate", rendered_rate])
    if max_concurrency is not None:
        argv.extend(["--max-concurrency", str(max_concurrency)])
    if random_input_len is not None:
        argv.extend(["--random-input-len", str(random_input_len)])
    if random_output_len is not None:
        argv.extend(["--random-output-len", str(random_output_len)])
    if result_dir is not None:
        argv.extend(["--result-dir", str(Path(result_dir).expanduser().resolve())])
    if result_filename is not None:
        argv.extend(["--result-filename", result_filename])
    argv.extend(extras)
    return tuple(argv)


@dataclass(frozen=True)
class BenchMetric:
    name: str
    value: float
    unit: str
    samples: tuple[float, ...]

    @property
    def sample_count(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class BenchServeResult:
    request_throughput: BenchMetric
    output_tps: BenchMetric
    total_tps: BenchMetric
    ttft: BenchMetric
    tpot: BenchMetric
    itl: BenchMetric
    e2el: BenchMetric | None
    completed_requests: int | None
    failed_requests: int | None
    request_latency_samples_ms: Mapping[str, tuple[float, ...]]
    raw: tuple[dict[str, Any], ...]

    @property
    def canonical_metrics(self) -> Mapping[str, BenchMetric]:
        """Metrics keyed only by the canonical cross-layer gate vocabulary."""

        values = {
            REQUEST_THROUGHPUT_METRIC: self.request_throughput,
            OUTPUT_THROUGHPUT_METRIC: self.output_tps,
            TOTAL_THROUGHPUT_METRIC: self.total_tps,
            MEAN_TTFT_METRIC: self.ttft,
            MEAN_TPOT_METRIC: self.tpot,
            MEAN_ITL_METRIC: self.itl,
        }
        if self.e2el is not None:
            values[MEAN_E2EL_METRIC] = self.e2el
        return MappingProxyType(values)

    @property
    def metrics(self) -> Mapping[str, BenchMetric]:
        """Deprecated short-key compatibility view; use ``canonical_metrics``."""

        values = {
            "request_throughput": self.request_throughput,
            "output_tps": self.output_tps,
            "total_tps": self.total_tps,
            "ttft": self.ttft,
            "tpot": self.tpot,
            "itl": self.itl,
        }
        if self.e2el is not None:
            values["e2el"] = self.e2el
        return MappingProxyType(values)

    def to_metric_series(self) -> dict[str, MetricSeries]:
        """Convert aggregate run samples to gate-ready ``models.MetricSeries``."""

        return {
            name: MetricSeries(unit=metric.unit, samples=list(metric.samples))
            for name, metric in self.canonical_metrics.items()
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_metrics": {
                name: asdict(metric) for name, metric in self.canonical_metrics.items()
            },
            "request_throughput": asdict(self.request_throughput),
            "output_tps": asdict(self.output_tps),
            "total_tps": asdict(self.total_tps),
            "ttft": asdict(self.ttft),
            "tpot": asdict(self.tpot),
            "itl": asdict(self.itl),
            "e2el": asdict(self.e2el) if self.e2el is not None else None,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "request_latency_samples_ms": {
                key: list(value) for key, value in self.request_latency_samples_ms.items()
            },
            "raw": list(self.raw),
        }


def _finite_positive_metric(row: Mapping[str, Any], aliases: Sequence[str], name: str) -> float:
    found = next((key for key in aliases if key in row), None)
    if found is None:
        raise VLLMBenchmarkParseError(f"required benchmark metric {name!r} is missing")
    raw = row[found]
    if isinstance(raw, Mapping):
        raw = raw.get("value", raw.get("mean"))
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise VLLMBenchmarkParseError(f"benchmark metric {name!r} must be numeric")
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise VLLMBenchmarkParseError(f"benchmark metric {name!r} must be finite and positive")
    return value


def _optional_count(
    rows: Sequence[Mapping[str, Any]], aliases: Sequence[str], name: str
) -> int | None:
    values: list[int] = []
    for row in rows:
        found = next((key for key in aliases if key in row), None)
        if found is None:
            continue
        raw = row[found]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise VLLMBenchmarkParseError(f"benchmark field {name!r} must be non-negative int")
        values.append(raw)
    return sum(values) if values else None


def _latency_samples_ms(row: Mapping[str, Any], metric: str, fallback: float) -> tuple[float, ...]:
    # vLLM detailed samples are seconds, while aggregate mean_*_ms fields are ms.
    candidates = (f"{metric}s", f"{metric}_samples_s")
    raw_samples = next((row[key] for key in candidates if key in row), None)
    multiplier = 1000.0
    if raw_samples is None:
        raw_samples = row.get(f"{metric}_samples_ms")
        multiplier = 1.0
    nested = row.get(metric)
    if raw_samples is None and isinstance(nested, Mapping):
        raw_samples = nested.get("samples")
        multiplier = 1.0 if nested.get("unit", "ms") == "ms" else 1000.0
    if raw_samples is None:
        return (fallback,)
    if not isinstance(raw_samples, list):
        raise VLLMBenchmarkParseError(f"benchmark {metric!r} samples must be an array")
    values: list[float] = []
    for item in raw_samples:
        if isinstance(item, list) and metric == "itl":
            source_items = item
        else:
            source_items = [item]
        for source in source_items:
            if isinstance(source, bool) or not isinstance(source, (int, float)):
                raise VLLMBenchmarkParseError(f"benchmark {metric!r} samples must be numeric")
            value = float(source) * multiplier
            if not math.isfinite(value) or value < 0:
                raise VLLMBenchmarkParseError(
                    f"benchmark {metric!r} samples must be finite and non-negative"
                )
            values.append(value)
    return tuple(values) if values else (fallback,)


def _decode_bench_rows(
    payload: str | bytes | Path | Mapping[str, Any] | list[Any],
) -> list[dict[str, Any]]:
    try:
        if isinstance(payload, Path):
            decoded: object = json.loads(payload.read_text(encoding="utf-8"))
        elif isinstance(payload, bytes):
            decoded = json.loads(payload.decode("utf-8"))
        elif isinstance(payload, str):
            decoded = json.loads(payload)
        else:
            decoded = payload
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VLLMBenchmarkParseError("invalid vLLM benchmark JSON") from error
    if isinstance(decoded, Mapping):
        nested = decoded.get("results")
        rows: list[Any] = nested if isinstance(nested, list) else [decoded]
    elif isinstance(decoded, list):
        rows = decoded
    else:
        raise VLLMBenchmarkParseError("vLLM benchmark JSON must be an object or array")
    if not rows:
        raise VLLMBenchmarkParseError("vLLM benchmark JSON contains no result rows")
    if any(not isinstance(row, Mapping) for row in rows):
        raise VLLMBenchmarkParseError("every vLLM benchmark result row must be an object")
    return [dict(row) for row in rows]


def parse_bench_serve_json(
    payload: str | bytes | Path | Mapping[str, Any] | list[Any],
    *,
    expected_num_prompts: int | None = None,
    require_e2el: bool = False,
) -> BenchServeResult:
    """Strictly parse official vLLM serve metrics and normalize names/units.

    Every row must contain all six gate metrics.  Missing, boolean, non-finite,
    zero, or negative aggregate values are rejected instead of being defaulted.
    Detailed latency arrays emitted in seconds are normalized to milliseconds.
    """

    rows = _decode_bench_rows(payload)
    if expected_num_prompts is not None:
        _positive_integer(expected_num_prompts, "expected_num_prompts")
    definitions = {
        "request_throughput": (
            ("request_throughput", "request_throughput_req_s"),
            "req/s",
        ),
        "output_tps": (
            ("output_throughput", "output_token_throughput", "output_tps"),
            "tok/s",
        ),
        "total_tps": (
            ("total_token_throughput", "total_throughput", "total_tps"),
            "tok/s",
        ),
        "ttft": (("mean_ttft_ms", "ttft_ms", "ttft"), "ms"),
        "tpot": (("mean_tpot_ms", "tpot_ms", "tpot"), "ms"),
        "itl": (("mean_itl_ms", "itl_ms", "itl"), "ms"),
    }
    e2el_aliases = ("mean_e2el_ms", "e2el_ms", "e2el")
    e2el_presence = [any(alias in row for alias in e2el_aliases) for row in rows]
    if require_e2el and not all(e2el_presence):
        raise VLLMBenchmarkParseError("required benchmark metric 'e2el' is missing")
    if any(e2el_presence) and not all(e2el_presence):
        raise VLLMBenchmarkParseError("benchmark runs contain inconsistent e2el metrics")
    if all(e2el_presence):
        definitions["e2el"] = (e2el_aliases, "ms")
    row_values: dict[str, list[float]] = {name: [] for name in definitions}
    latency_samples: dict[str, list[float]] = {
        name: [] for name in ("ttft", "tpot", "itl", "e2el") if name in definitions
    }
    for row in rows:
        for name, (aliases, _) in definitions.items():
            value = _finite_positive_metric(row, aliases, name)
            row_values[name].append(value)
            if name in latency_samples:
                latency_samples[name].extend(_latency_samples_ms(row, name, value))

        failed_key = next((key for key in ("failed", "failed_requests") if key in row), None)
        if failed_key is not None:
            failed_value = row[failed_key]
            if isinstance(failed_value, bool) or not isinstance(failed_value, int):
                raise VLLMBenchmarkParseError("benchmark field 'failed' must be an integer")
            if failed_value != 0:
                raise VLLMBenchmarkParseError("vLLM benchmark contains failed requests")
        row_expected = expected_num_prompts
        if row_expected is None and "num_prompts" in row:
            raw_expected = row["num_prompts"]
            if (
                isinstance(raw_expected, bool)
                or not isinstance(raw_expected, int)
                or raw_expected <= 0
            ):
                raise VLLMBenchmarkParseError("benchmark num_prompts must be a positive integer")
            row_expected = raw_expected
        if row_expected is not None:
            completed_key = next(
                (key for key in ("completed", "completed_requests") if key in row),
                None,
            )
            if completed_key is None:
                raise VLLMBenchmarkParseError(
                    "completed request count is required to verify expected_num_prompts"
                )
            completed_value = row[completed_key]
            if (
                isinstance(completed_value, bool)
                or not isinstance(completed_value, int)
                or completed_value != row_expected
            ):
                raise VLLMBenchmarkParseError(
                    "completed requests do not equal the declared num_prompts"
                )

    def metric(name: str) -> BenchMetric:
        values = tuple(row_values[name])
        return BenchMetric(
            name=_SHORT_TO_CANONICAL_BENCH_METRIC[name],
            value=sum(values) / len(values),
            unit=definitions[name][1],
            # One sample per benchmark run.  Per-request latency observations are
            # retained separately and never masquerade as stability repetitions.
            samples=values,
        )

    completed = _optional_count(rows, ("completed", "completed_requests"), "completed")
    if completed == 0:
        raise VLLMBenchmarkParseError("vLLM benchmark completed zero requests")
    failed = _optional_count(rows, ("failed", "failed_requests"), "failed")
    return BenchServeResult(
        request_throughput=metric("request_throughput"),
        output_tps=metric("output_tps"),
        total_tps=metric("total_tps"),
        ttft=metric("ttft"),
        tpot=metric("tpot"),
        itl=metric("itl"),
        e2el=metric("e2el") if "e2el" in definitions else None,
        completed_requests=completed,
        failed_requests=failed,
        request_latency_samples_ms=MappingProxyType(
            {name: tuple(values) for name, values in latency_samples.items()}
        ),
        raw=tuple(rows),
    )


__all__ = [
    "BenchMetric",
    "BenchServeResult",
    "CANONICAL_BENCH_METRIC_NAMES",
    "EndpointVerifier",
    "HTTPClient",
    "HTTPJSONResponse",
    "LocalProcessBackend",
    "LocalEndpointVerifier",
    "LocalRuntimeInspector",
    "MEAN_E2EL_METRIC",
    "MEAN_ITL_METRIC",
    "MEAN_TPOT_METRIC",
    "MEAN_TTFT_METRIC",
    "OUTPUT_THROUGHPUT_METRIC",
    "ProcessBackend",
    "REQUEST_THROUGHPUT_METRIC",
    "RunIdentity",
    "RuntimeInspector",
    "RuntimeVerificationStatus",
    "ServerExecutionRecord",
    "ServerExecutionStatus",
    "ServerStartResult",
    "ServerStopResult",
    "SpawnedProcess",
    "StdlibHTTPClient",
    "TOTAL_THROUGHPUT_METRIC",
    "VLLMAdapter",
    "VLLMAdapterError",
    "VLLMBenchmarkParseError",
    "VLLMHealthResult",
    "VLLMHealthTimeout",
    "VLLMIdentityMismatchError",
    "VLLMRecordError",
    "VLLMRunIdentity",
    "VLLMRuntimeEvidence",
    "VLLMServerConflictError",
    "VLLMServerSpec",
    "VLLMServerStartError",
    "VLLMServerStartResult",
    "VLLMServerStopResult",
    "VLLMSpecError",
    "VLLMStopError",
    "build_bench_serve_argv",
    "canonical_sha256",
    "load_execution_record",
    "parse_bench_serve_json",
    "save_execution_record",
]
