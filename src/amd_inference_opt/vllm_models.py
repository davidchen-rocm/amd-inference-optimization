"""Typed, versioned coordinates for the vLLM + MI300X V0 workflow.

The existing :mod:`amd_inference_opt.models` module intentionally keeps the
llama.cpp workload surface small.  This module adds the serving-specific
coordinates without weakening those models or pretending that a Hugging Face
snapshot directory is one model file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .command import validate_argv
from .models import (
    ArtifactRef,
    ChangeKind,
    ChangeSet,
    GateDecision,
    Hypothesis,
    OptimizationTask,
    StrictModel,
    VLLMRuntimeTarget,
    VLLMServingWorkloadConfig,
    WorkflowStatus,
    utc_now,
)
from .vllm_adapter import CANONICAL_BENCH_METRIC_NAMES


def canonical_sha256(value: Any) -> str:
    """Hash a JSON coordinate using the repository's canonical encoding."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _validate_revision(value: str, field_name: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", normalized) is None:
        raise ValueError(f"{field_name} must be an immutable 40-character commit revision")
    return normalized


def _validate_environment(values: dict[str, str]) -> dict[str, str]:
    for key, value in values.items():
        if not key or "=" in key or "\x00" in key:
            raise ValueError("environment names must be non-empty and cannot contain '=' or NUL")
        if "\x00" in value:
            raise ValueError("environment values cannot contain NUL")
    return values


def _argv_option_values(argv: list[str], names: tuple[str, ...]) -> list[str | None]:
    values: list[str | None] = []
    for index, item in enumerate(argv):
        for name in names:
            if item == name:
                values.append(argv[index + 1] if index + 1 < len(argv) else None)
            elif item.startswith(f"{name}="):
                values.append(item.removeprefix(f"{name}="))
    return values


def _require_exact_argv_option(
    argv: list[str], names: tuple[str, ...], expected: str, label: str
) -> None:
    values = _argv_option_values(argv, names)
    if values != [expected]:
        raise ValueError(
            f"argv must bind {label} exactly once to {expected!r}; observed {values!r}"
        )


class VLLMStage(StrEnum):
    """The deliberately short V0 serving workflow."""

    INSPECT = "INSPECT"
    SERVER_START = "SERVER_START"
    BASELINE = "BASELINE"
    PROFILE_APPROVAL = "PROFILE_APPROVAL"
    AGENT_DECISION = "AGENT_DECISION"
    EXPERIMENT = "EXPERIMENT"
    QUALITY = "QUALITY"
    DECIDE = "DECIDE"


class VLLMApprovalStatus(StrEnum):
    REQUESTED = "REQUESTED"
    APPROVED = "APPROVED"
    CONSUMED = "CONSUMED"


class VLLMNextActionKind(StrEnum):
    PROVIDE_INSPECTION = "PROVIDE_INSPECTION"
    START_SERVER = "START_SERVER"
    RUN_BASELINE = "RUN_BASELINE"
    STOP_BASELINE_SERVER = "STOP_BASELINE_SERVER"
    REQUEST_PROFILE_APPROVAL = "REQUEST_PROFILE_APPROVAL"
    APPROVE_PROFILE = "APPROVE_PROFILE"
    RUN_APPROVED_PROFILE = "RUN_APPROVED_PROFILE"
    RECORD_PROFILE_RESULT = "RECORD_PROFILE_RESULT"
    RESOLVE_PROFILE_ENVIRONMENT = "RESOLVE_PROFILE_ENVIRONMENT"
    SUBMIT_AGENT_DECISION = "SUBMIT_AGENT_DECISION"
    START_EXPERIMENT_SERVER = "START_EXPERIMENT_SERVER"
    RUN_EXPERIMENT = "RUN_EXPERIMENT"
    RUN_QUALITY = "RUN_QUALITY"
    STOP_EXPERIMENT_SERVER = "STOP_EXPERIMENT_SERVER"
    FINALIZE_GPU_BUDGET = "FINALIZE_GPU_BUDGET"
    EVALUATE_GATE = "EVALUATE_GATE"
    RESUME_INCONCLUSIVE = "RESUME_INCONCLUSIVE"
    COMPLETE = "COMPLETE"


class VLLMModelCoordinate(StrictModel):
    """Identity of one immutable Hugging Face snapshot directory.

    ``snapshot_digest`` is a canonical manifest/tree digest produced by the
    model preparation layer.  It is not described as the SHA-256 of one large
    model file.
    """

    model_id: str
    local_path: Path
    snapshot_manifest_path: Path
    revision: str
    snapshot_digest: str
    tokenizer_revision: str

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        normalized = value.strip()
        if (
            not normalized
            or "\x00" in normalized
            or "\n" in normalized
            or "\r" in normalized
        ):
            raise ValueError("model_id must be non-empty single-line text")
        return normalized

    @field_validator("local_path", "snapshot_manifest_path")
    @classmethod
    def validate_local_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("vLLM model and snapshot manifest paths must be absolute")
        return value

    @field_validator("revision")
    @classmethod
    def validate_model_revision(cls, value: str) -> str:
        return _validate_revision(value, "model revision")

    @field_validator("tokenizer_revision")
    @classmethod
    def validate_tokenizer_revision(cls, value: str) -> str:
        return _validate_revision(value, "tokenizer revision")

    @field_validator("snapshot_digest")
    @classmethod
    def validate_snapshot_digest(cls, value: str) -> str:
        return _validate_sha256(value, "snapshot_digest")


class MI300XDeviceBinding(StrictModel):
    """Exact logical device and partition identity for the single-GPU V0."""

    gfx_target: Literal["gfx942"] = "gfx942"
    device_ids: list[int] = Field(min_length=1, max_length=1)
    tensor_parallel_size: Literal[1] = 1
    product_name: Literal["AMD Instinct MI300X"] = "AMD Instinct MI300X"
    oam_id: int = Field(ge=0)
    xcc_count: Literal[8] = 8
    compute_partition: Literal["SPX", "DPX", "QPX", "CPX"] = "SPX"
    memory_partition: Literal["NPS1", "NPS2", "NPS4"] = "NPS1"
    device_uuid: str
    pci_bdf: str
    partition_id: int = Field(ge=0)
    amd_smi_command_sha256: str
    hip_probe_command_sha256: str

    @field_validator("device_ids")
    @classmethod
    def validate_device_ids(cls, values: list[int]) -> list[int]:
        if any(isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("device_ids must contain non-negative integer device indices")
        if len(values) != len(set(values)):
            raise ValueError("device_ids must be unique")
        return values

    @field_validator("device_uuid")
    @classmethod
    def validate_device_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]+", normalized) is None:
            raise ValueError("device_uuid must use the stable AMD SMI UUID token")
        return normalized

    @field_validator("pci_bdf")
    @classmethod
    def validate_pci_bdf(cls, value: str) -> str:
        normalized = value.strip().lower()
        if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", normalized) is None:
            raise ValueError("pci_bdf must use canonical dddd:bb:ss.f form")
        return normalized

    @field_validator("amd_smi_command_sha256", "hip_probe_command_sha256")
    @classmethod
    def validate_inspection_command_hash(cls, value: str) -> str:
        return _validate_sha256(value, "inspection command")

    @model_validator(mode="after")
    def stable_partition_identity(self) -> MI300XDeviceBinding:
        allowed = {
            "SPX": {"NPS1"},
            "DPX": {"NPS1", "NPS2"},
            "QPX": {"NPS1", "NPS4"},
            "CPX": {"NPS1", "NPS4"},
        }
        if self.memory_partition not in allowed[self.compute_partition]:
            raise ValueError(
                f"unsupported MI300X partition pair: {self.compute_partition}:"
                f"{self.memory_partition}"
            )
        return self

    @property
    def stable_device_id(self) -> str:
        return self.device_uuid


class MI300XInspectionEvidence(StrictModel):
    """Hash-bound AMD SMI observation matched before any server is started."""

    source: Literal["amd-smi+hip-probe"] = "amd-smi+hip-probe"
    scope: Literal["container_preflight", "native_preflight"]
    image_digest: str | None = None
    gfx_target: Literal["gfx942"]
    product_name: Literal["AMD Instinct MI300X"]
    oam_id: int = Field(ge=0)
    xcc_count: Literal[8]
    logical_device_id: Literal[0] = 0
    container_logical_device_id: Literal[0] = 0
    visible_device_count: Literal[1] = 1
    rocr_visible_devices: str
    hip_visible_devices: None = None
    hsa_visible_devices: None = None
    cuda_visible_devices: None = None
    gpu_device_ordinal: None = None
    device_uuid: str
    pci_bdf: str
    compute_partition: Literal["SPX", "DPX", "QPX", "CPX"]
    memory_partition: Literal["NPS1", "NPS2", "NPS4"]
    partition_id: int = Field(ge=0)
    rocm_version: str
    vllm_version: str
    pytorch_version: str
    python_version: str
    launcher_sha256: str
    environment_manifest_sha256: str
    amd_smi_artifact: ArtifactRef
    hip_probe_artifact: ArtifactRef
    amd_smi_command_sha256: str
    hip_probe_command_sha256: str
    captured_at: datetime = Field(default_factory=utc_now)

    @field_validator("device_uuid")
    @classmethod
    def validate_observed_uuid(cls, value: str | None) -> str | None:
        return MI300XDeviceBinding.validate_device_uuid(value)

    @field_validator("pci_bdf")
    @classmethod
    def validate_observed_bdf(cls, value: str) -> str:
        return MI300XDeviceBinding.validate_pci_bdf(value)

    @field_validator(
        "rocm_version",
        "vllm_version",
        "pytorch_version",
        "python_version",
    )
    @classmethod
    def validate_observed_versions(cls, value: str) -> str:
        if not value.strip() or "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("observed runtime versions must be non-empty single-line text")
        return value.strip()

    @field_validator("image_digest")
    @classmethod
    def validate_observed_image_digest(cls, value: str | None) -> str | None:
        return (
            _validate_sha256(value, "observed image_digest")
            if value is not None
            else None
        )

    @field_validator(
        "launcher_sha256",
        "environment_manifest_sha256",
        "amd_smi_command_sha256",
        "hip_probe_command_sha256",
    )
    @classmethod
    def validate_observed_launcher_digest(cls, value: str) -> str:
        return _validate_sha256(value, "observed runtime provenance digest")

    @field_validator("rocr_visible_devices")
    @classmethod
    def validate_observed_rocr_selector(cls, value: str) -> str:
        if not value.strip() or any(character.isspace() for character in value):
            raise ValueError("observed ROCR_VISIBLE_DEVICES must be one stable selector")
        return value

    @model_validator(mode="after")
    def require_stable_observed_identity(self) -> MI300XInspectionEvidence:
        if (self.scope == "container_preflight") != (self.image_digest is not None):
            raise ValueError(
                "container_preflight requires an observed RepoDigest; native_preflight "
                "must not claim one"
            )
        return self


class VLLMServingProtocol(StrictModel):
    """Exact server and ``vllm bench serve`` coordinates."""

    server_argv: list[str] = Field(min_length=1)
    benchmark_argv: list[str] = Field(min_length=1)
    cwd: Path
    server_env: dict[str, str] = Field(default_factory=dict)
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    benchmark_backend: str = "openai"
    benchmark_endpoint: str = "/v1/completions"
    benchmark_dataset: str = "random"
    tensor_parallel_size: Literal[1] = 1
    dtype: str
    quantization: str | None = None
    concurrency: int = Field(ge=1)
    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    num_prompts: int = Field(ge=1)
    seed: int = 42
    warmup_runs: int = Field(default=1, ge=0)
    sample_count: int = Field(default=3, ge=1)
    timeout_seconds: int = Field(default=900, ge=1)
    required_metrics: list[str] = Field(
        default_factory=lambda: list(CANONICAL_BENCH_METRIC_NAMES)
    )
    engine_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("server_argv", "benchmark_argv")
    @classmethod
    def validate_commands(cls, values: list[str]) -> list[str]:
        return list(validate_argv(values))

    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("vLLM serving cwd must be absolute")
        return value

    @field_validator("server_env")
    @classmethod
    def validate_server_environment(cls, values: dict[str, str]) -> dict[str, str]:
        return _validate_environment(values)

    @field_validator("host", "benchmark_backend", "benchmark_dataset")
    @classmethod
    def validate_host(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError("host must be non-empty and cannot contain whitespace")
        return normalized

    @field_validator("benchmark_endpoint")
    @classmethod
    def validate_benchmark_endpoint(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith("/") or any(character.isspace() for character in normalized):
            raise ValueError("benchmark_endpoint must be an absolute URL path")
        return normalized

    @field_validator("dtype", "quantization")
    @classmethod
    def validate_precision_coordinate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError("dtype and quantization must be non-empty tokens")
        return normalized

    @field_validator("required_metrics")
    @classmethod
    def validate_required_metrics(cls, values: list[str]) -> list[str]:
        if values != list(CANONICAL_BENCH_METRIC_NAMES):
            raise ValueError(
                "required_metrics must use the complete canonical vLLM metric vocabulary"
            )
        return values

    @model_validator(mode="after")
    def validate_request_shape(self) -> VLLMServingProtocol:
        if self.num_prompts < self.concurrency:
            raise ValueError("num_prompts must be at least concurrency")
        try:
            json.dumps(self.engine_config, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("engine_config must be finite JSON data") from error
        return self

    @property
    def coordinate_payload(self) -> dict[str, Any]:
        return {
            "schema": "amd-inference-opt.vllm-serving-protocol.v1",
            "server_argv": self.server_argv,
            "benchmark_argv": self.benchmark_argv,
            "cwd": str(self.cwd),
            "server_env": dict(sorted(self.server_env.items())),
            "host": self.host,
            "port": self.port,
            "benchmark_backend": self.benchmark_backend,
            "benchmark_endpoint": self.benchmark_endpoint,
            "benchmark_dataset": self.benchmark_dataset,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "concurrency": self.concurrency,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "num_prompts": self.num_prompts,
            "seed": self.seed,
            "warmup_runs": self.warmup_runs,
            "sample_count": self.sample_count,
            "timeout_seconds": self.timeout_seconds,
            "required_metrics": self.required_metrics,
            "engine_config": self.engine_config,
        }

    @property
    def coordinate_sha256(self) -> str:
        return canonical_sha256(self.coordinate_payload)


class VLLMProfileProtocol(StrictModel):
    """Base offline workload plus the project-owned per-attempt launcher."""

    mode: Literal["offline_one_shot", "unavailable"] = "offline_one_shot"
    profile_argv: list[str] = Field(default_factory=list)
    engine_mode: Literal["vllm_offline_generate"] = "vllm_offline_generate"
    launcher_path: Path | None = None
    launcher_sha256: str | None = None
    environment_manifest_path: Path | None = None
    cwd: Path | None = None
    environment: Literal["same_container", "native", "unavailable"]
    unavailable_reason: str | None = None
    preset: Literal["kernel-basic", "kernel-timing", "kernel-metadata"] = (
        "kernel-timing"
    )
    timeout_seconds: int = Field(default=900, ge=1)
    max_trace_bytes: int = Field(default=200_000_000, ge=1)
    max_trace_files: int = Field(default=64, ge=1)
    max_events_per_type: int = Field(default=10_000, ge=1)
    max_percentile_samples_per_kernel: int = Field(default=20, ge=1)

    @field_validator("profile_argv")
    @classmethod
    def validate_profile_argv(cls, values: list[str]) -> list[str]:
        return list(validate_argv(values)) if values else values

    @field_validator("launcher_sha256")
    @classmethod
    def validate_profile_launcher_hash(cls, value: str | None) -> str | None:
        return _validate_sha256(value, "profile launcher") if value is not None else None

    @model_validator(mode="after")
    def validate_mode(self) -> VLLMProfileProtocol:
        if self.mode == "unavailable":
            if (
                self.profile_argv
                or self.cwd is not None
                or self.launcher_path is not None
                or self.launcher_sha256 is not None
                or self.environment_manifest_path is not None
                or self.environment != "unavailable"
            ):
                raise ValueError(
                    "unavailable profiling cannot declare a command, cwd, or execution environment"
                )
            if self.unavailable_reason is None or not self.unavailable_reason.strip():
                raise ValueError("unavailable profiling requires unavailable_reason")
            return self
        if not self.profile_argv or self.cwd is None:
            raise ValueError("offline_one_shot profiling requires profile_argv and cwd")
        if (
            self.launcher_path is None
            or self.launcher_sha256 is None
            or self.environment_manifest_path is None
        ):
            raise ValueError(
                "offline_one_shot profiling requires a hash-bound launcher and "
                "environment manifest"
            )
        if (
            not self.launcher_path.is_absolute()
            or not self.environment_manifest_path.is_absolute()
        ):
            raise ValueError("profile launcher and environment manifest paths must be absolute")
        if not self.cwd.is_absolute():
            raise ValueError("profile cwd must be absolute")
        if self.environment == "unavailable":
            raise ValueError("offline_one_shot profiling requires an execution environment")
        if self.unavailable_reason is not None:
            raise ValueError("offline_one_shot profiling cannot declare unavailable_reason")
        return self

    @property
    def command_sha256(self) -> str | None:
        if self.mode == "unavailable":
            return None
        return canonical_sha256(
            {
                "offline_argv": self.profile_argv,
                "engine_mode": self.engine_mode,
                "launcher_path": str(self.launcher_path),
                "launcher_sha256": self.launcher_sha256,
                "environment_manifest_path": str(self.environment_manifest_path),
                "cwd": str(self.cwd),
                "environment": self.environment,
                "preset": self.preset,
                "timeout_seconds": self.timeout_seconds,
                "capture_limits": {
                    "max_trace_bytes": self.max_trace_bytes,
                    "max_trace_files": self.max_trace_files,
                    "max_events_per_type": self.max_events_per_type,
                    "max_percentile_samples_per_kernel": (
                        self.max_percentile_samples_per_kernel
                    ),
                },
            }
        )


class VLLMQualityProtocol(StrictModel):
    """Strict external command contract; no llama.cpp evaluator is implied."""

    argv: list[str] = Field(min_length=1)
    cwd: Path
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=14_400, ge=1)
    output_path: Path
    output_contract: Literal["amd-inference-opt.QualityResult.v1"] = (
        "amd-inference-opt.QualityResult.v1"
    )

    @field_validator("argv")
    @classmethod
    def validate_quality_argv(cls, values: list[str]) -> list[str]:
        return list(validate_argv(values))

    @field_validator("cwd", "output_path")
    @classmethod
    def validate_quality_paths(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("quality cwd and output_path must be absolute")
        return value

    @field_validator("env")
    @classmethod
    def validate_quality_environment(cls, values: dict[str, str]) -> dict[str, str]:
        return _validate_environment(values)

    @property
    def coordinate_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class VLLMCampaignConfig(StrictModel):
    """Standalone persisted configuration for one vLLM/MI300X run."""

    schema_version: Literal[1] = 1
    task: OptimizationTask
    model: VLLMModelCoordinate
    device: MI300XDeviceBinding
    serving: VLLMServingProtocol
    profile: VLLMProfileProtocol
    quality_protocol: VLLMQualityProtocol

    @property
    def id(self) -> str:
        return self.task.id

    @model_validator(mode="after")
    def validate_v0_coordinates(self) -> VLLMCampaignConfig:
        if self.task.campaign_kind.value != "vllm_mi300x":
            raise ValueError("vLLM config requires task.campaign_kind='vllm_mi300x'")
        if not isinstance(self.task.runtime, VLLMRuntimeTarget):
            raise ValueError("vLLM config requires VLLMRuntimeTarget")
        runtime = self.task.runtime
        missing_versions = [
            name
            for name, value in {
                "python_version": runtime.python_version,
                "pytorch_version": runtime.pytorch_version,
                "rocm_version": runtime.rocm_version,
            }.items()
            if value is None
        ]
        if missing_versions:
            raise ValueError(
                "vLLM runtime must bind " + ", ".join(sorted(missing_versions))
            )
        if runtime.deployment != "native":
            raise ValueError(
                "vLLM MI300X V0 runs natively inside an already pinned container"
            )
        if runtime.executable_sha256 is None:
            raise ValueError("inside-container vLLM requires executable_sha256")
        if not Path(runtime.executable).is_absolute():
            raise ValueError("native vLLM executable must use an absolute path")
        if re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(runtime.executable).name) is None:
            raise ValueError(
                "native vLLM executable must be the absolute Python interpreter observed "
                "through /proc/PID/exe"
            )
        if (runtime.image is None) != (runtime.image_digest is None):
            raise ValueError("native vLLM image and image_digest must be provided together")
        if self.task.gpu.gfx_target != "gfx942" or self.device.gfx_target != "gfx942":
            raise ValueError("vLLM MI300X requires gfx942")
        if self.device.device_ids != [self.task.gpu.device_id]:
            raise ValueError("task GPU device must exactly match the single V0 device binding")
        if self.serving.tensor_parallel_size != self.device.tensor_parallel_size:
            raise ValueError("serving and device tensor_parallel_size must match")
        workload = self.task.workload
        if not isinstance(workload, VLLMServingWorkloadConfig):
            raise ValueError("vLLM task requires the typed online serving workload")
        workload_values = {
            "input_tokens": self.serving.input_tokens,
            "output_tokens": self.serving.output_tokens,
            "num_prompts": self.serving.num_prompts,
            "concurrency": self.serving.concurrency,
            "seed": self.serving.seed,
        }
        workload_mismatches = [
            name
            for name, expected in workload_values.items()
            if getattr(workload, name) != expected
        ]
        if workload_mismatches:
            raise ValueError(
                "task online workload must exactly match serving protocol: "
                + ", ".join(workload_mismatches)
            )
        if self.task.objective.primary_metric != (
            "output_throughput_tokens_per_second"
        ):
            raise ValueError(
                "vLLM objective.primary_metric must use canonical output throughput"
            )
        if self.task.model.path != self.model.local_path:
            raise ValueError("task model path must match the HF snapshot local_path")
        if self.task.model.sha256 is not None:
            raise ValueError(
                "vLLM task.model.sha256 must be null; use model.snapshot_digest for a directory"
            )
        if self.task.model.quantization != self.serving.quantization:
            raise ValueError("task model quantization must match the serving protocol")
        if self.task.rocm.version != runtime.rocm_version:
            raise ValueError("task ROCm version must match the vLLM runtime ROCm version")
        if self.task.benchmark.protocol_hash != self.serving.coordinate_sha256:
            raise ValueError(
                "task benchmark.protocol_hash must equal the serving coordinate SHA-256"
            )
        if self.task.benchmark.sample_count != self.serving.sample_count:
            raise ValueError("task and serving sample_count must match")
        if self.task.benchmark.timeout_seconds != self.serving.timeout_seconds:
            raise ValueError("task and serving benchmark timeout must match")
        if self.task.benchmark.benchmark_command != self.serving.benchmark_argv:
            raise ValueError("task benchmark command must match serving benchmark_argv")
        if self.task.benchmark.required_metrics != self.serving.required_metrics:
            raise ValueError("task and serving required_metrics must match")
        argv = self.serving.server_argv
        launcher_kind = self.serving.engine_config.get("launcher_kind")
        launcher = self.serving.engine_config.get("launcher")
        launcher_sha256 = self.serving.engine_config.get("launcher_sha256")
        expected_server_launcher = "vllm.entrypoints.openai.api_server"
        if launcher_kind != "python_module" or launcher != expected_server_launcher:
            raise ValueError(
                "vLLM V0 server launcher must be the official OpenAI Python module"
            )
        if not isinstance(launcher_sha256, str):
            raise ValueError("engine_config.launcher_sha256 must bind launcher/package source")
        _validate_sha256(launcher_sha256, "engine_config.launcher_sha256")
        expected_server_prefix = [
            runtime.executable,
            "-I",
            "-m",
            expected_server_launcher,
        ]
        if argv[: len(expected_server_prefix)] != expected_server_prefix:
            raise ValueError(
                "server_argv must use the hash-pinned isolated Python and official "
                "vLLM OpenAI module"
            )
        if str(self.model.local_path) not in argv:
            raise ValueError("server_argv must serve the immutable local snapshot path")
        server_bindings = (
            (("--model",), str(self.model.local_path), "model snapshot"),
            (("--served-model-name",), self.model.model_id, "served model id"),
            (
                ("--tensor-parallel-size", "-tp"),
                str(self.serving.tensor_parallel_size),
                "tensor_parallel_size",
            ),
            (("--dtype",), self.serving.dtype, "dtype"),
            (("--host",), self.serving.host, "host"),
            (("--port",), str(self.serving.port), "port"),
        )
        for names, value, label in server_bindings:
            _require_exact_argv_option(argv, names, value, f"server {label}")
        quantization_values = _argv_option_values(argv, ("--quantization", "-q"))
        expected_quantization = (
            [] if self.serving.quantization is None else [self.serving.quantization]
        )
        if quantization_values != expected_quantization:
            raise ValueError(
                "server quantization must be absent or exactly match immutable config"
            )
        required_server_env = {
            "ROCR_VISIBLE_DEVICES": self.device.device_uuid,
            "PYTHONNOUSERSITE": "1",
        }
        invalid_device_env = [
            name
            for name, expected in required_server_env.items()
            if self.serving.server_env.get(name) != expected
        ]
        if invalid_device_env:
            raise ValueError(
                "native server_env must bind stable UUID isolation: "
                + ", ".join(sorted(invalid_device_env))
            )
        benchmark_argv = self.serving.benchmark_argv
        expected_benchmark_prefix = [
            runtime.executable,
            "-I",
            "-m",
            "vllm.entrypoints.cli.main",
            "bench",
            "serve",
        ]
        if benchmark_argv[: len(expected_benchmark_prefix)] != expected_benchmark_prefix:
            raise ValueError(
                "benchmark_argv must use isolated Python and the official vLLM "
                "'bench serve' module entry point"
            )
        reserved_result_flags = {"--save-result", "--result-dir", "--result-filename"}
        configured_result_flags = sorted(
            flag
            for flag in reserved_result_flags
            if any(item == flag or item.startswith(f"{flag}=") for item in benchmark_argv)
        )
        if configured_result_flags:
            raise ValueError(
                "benchmark result flags are coordinator-owned for unique durable outputs: "
                + ", ".join(configured_result_flags)
            )
        benchmark_bindings = (
            (
                ("--base-url",),
                f"http://{self.serving.host}:{self.serving.port}",
                "managed server base URL",
            ),
            (("--backend",), self.serving.benchmark_backend, "backend"),
            (("--endpoint",), self.serving.benchmark_endpoint, "endpoint"),
            (("--dataset-name",), self.serving.benchmark_dataset, "dataset"),
            (("--max-concurrency",), str(self.serving.concurrency), "concurrency"),
            (
                ("--input-len", "--random-input-len"),
                str(self.serving.input_tokens),
                "input_tokens",
            ),
            (
                ("--output-len", "--random-output-len"),
                str(self.serving.output_tokens),
                "output_tokens",
            ),
            (("--num-prompts",), str(self.serving.num_prompts), "num_prompts"),
            (("--seed",), str(self.serving.seed), "seed"),
            (("--model",), self.model.model_id, "model_id"),
        )
        for names, value, label in benchmark_bindings:
            _require_exact_argv_option(
                benchmark_argv, names, value, f"benchmark {label}"
            )
        required_locks = {
            "image_digest",
            "model_snapshot_digest",
            "model_revision",
            "tokenizer_revision",
            "gpu_device_identity",
            "gpu_partition",
            "dtype",
            "quantization",
            "tensor_parallel_size",
            "benchmark_protocol",
            "runtime_environment",
        }
        missing_locks = required_locks - set(self.task.change_policy.locked_coordinates)
        if missing_locks:
            raise ValueError(
                "vLLM V0 change policy must lock: " + ", ".join(sorted(missing_locks))
            )
        if self.task.change_policy.allowed_change_kinds != [ChangeKind.RUNTIME_CONFIG]:
            raise ValueError("vLLM V0 permits runtime_config changes only")
        forbidden_runtime_flags = {
            "--model",
            "--served-model-name",
            "--tensor-parallel-size",
            "-tp",
            "--dtype",
            "--quantization",
            "-q",
            "--host",
            "--port",
        }
        unsafe_allowlist = forbidden_runtime_flags & set(
            self.task.change_policy.allowed_runtime_args
        )
        if unsafe_allowlist:
            raise ValueError(
                "runtime allow-list contains locked vLLM coordinates: "
                + ", ".join(sorted(unsafe_allowlist))
            )
        baseline_experiment_flags = sorted(
            flag
            for flag in self.task.change_policy.allowed_runtime_args
            if _argv_option_values(argv, (flag,))
        )
        if baseline_experiment_flags:
            raise ValueError(
                "allow-listed experiment flags must be absent from baseline server argv: "
                + ", ".join(baseline_experiment_flags)
            )
        conflicting_inherited_environment = {
            "HIP_VISIBLE_DEVICES",
            "HSA_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "HSA_OVERRIDE_GFX_VERSION",
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONSTARTUP",
        }
        configured_conflicts = conflicting_inherited_environment & (
            set(self.serving.server_env) | set(self.task.mcp.env)
        )
        if configured_conflicts:
            raise ValueError(
                "vLLM V0 clears conflicting GPU/Python environment instead of setting: "
                + ", ".join(sorted(configured_conflicts))
            )
        if self.task.mcp.env.get("PYTHONNOUSERSITE") != "1":
            raise ValueError("profile MCP env must bind PYTHONNOUSERSITE=1")
        if not self.task.quality.require_correctness:
            raise ValueError("vLLM V0 requires correctness validation")
        if not self.task.quality.require_quality_evaluation:
            raise ValueError("vLLM V0 requires baseline and candidate quality evaluation")
        missing_match_fields = set(self.environment_coordinates) - set(
            self.task.environment.required_match_fields
        )
        if missing_match_fields:
            raise ValueError(
                "vLLM environment gate must compare: "
                + ", ".join(sorted(missing_match_fields))
            )
        if not self.task.environment.require_fresh_capture:
            raise ValueError("vLLM V0 requires fresh baseline and candidate captures")
        if not self.task.environment.require_stable_telemetry:
            raise ValueError("vLLM V0 requires stable observed telemetry")
        expected_profile_environment = (
            "same_container" if runtime.image_digest is not None else "native"
        )
        if (
            self.profile.mode == "offline_one_shot"
            and self.profile.environment != expected_profile_environment
        ):
            raise ValueError(
                "profile environment must match configured native/container provenance"
            )
        if self.profile.mode == "offline_one_shot":
            profile_argv = self.profile.profile_argv
            if not Path(profile_argv[0]).is_absolute():
                raise ValueError("profile argv must lock an absolute executable path")
            expected_profile_prefix = [
                runtime.executable,
                "-I",
                "-m",
                "vllm.entrypoints.cli.main",
                "bench",
                "throughput",
            ]
            if profile_argv[: len(expected_profile_prefix)] != expected_profile_prefix:
                raise ValueError(
                    "offline profile argv must use the hash-bound official vLLM "
                    "'bench throughput' Python module entry point"
                )
            profile_bindings = (
                (("--backend",), "vllm", "offline backend"),
                (("--dataset-name",), "random", "dataset"),
                (("--model",), str(self.model.local_path), "model snapshot"),
                (("--tensor-parallel-size", "-tp"), "1", "tensor_parallel_size"),
                (("--dtype",), self.serving.dtype, "dtype"),
                (("--input-len",), str(self.serving.input_tokens), "input length"),
                (("--output-len",), str(self.serving.output_tokens), "output length"),
                (("--num-prompts",), str(self.serving.num_prompts), "prompt count"),
                (("--seed",), str(self.serving.seed), "seed"),
            )
            for names, value, label in profile_bindings:
                _require_exact_argv_option(
                    profile_argv, names, value, f"profile {label}"
                )
            profile_quantization = _argv_option_values(
                profile_argv, ("--quantization", "-q")
            )
            expected_profile_quantization = (
                [] if self.serving.quantization is None else [self.serving.quantization]
            )
            if profile_quantization != expected_profile_quantization:
                raise ValueError(
                    "profile quantization must be absent or exactly match serving config"
                )
        return self

    @property
    def identity_payload(self) -> dict[str, Any]:
        runtime = self.task.runtime
        assert isinstance(runtime, VLLMRuntimeTarget)  # proved by validation
        return {
            "schema": "amd-inference-opt.vllm-mi300x-identity.v1",
            "task_id": self.task.id,
            "image": runtime.image,
            "image_digest": runtime.image_digest,
            "vllm_version": runtime.version,
            "python_executable_sha256": runtime.executable_sha256,
            "python_version": runtime.python_version,
            "pytorch_version": runtime.pytorch_version,
            "rocm_version": runtime.rocm_version,
            "environment_manifest_sha256": runtime.environment_manifest_sha256,
            "model": self.model.model_dump(mode="json"),
            "device": self.device.model_dump(mode="json"),
            "serving_protocol_sha256": self.serving.coordinate_sha256,
            "profile_command_sha256": self.profile.command_sha256,
            "quality_protocol_sha256": self.quality_protocol.coordinate_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.identity_payload)

    @property
    def environment_coordinates(self) -> dict[str, str]:
        runtime = self.task.runtime
        assert isinstance(runtime, VLLMRuntimeTarget)
        return {
            "gpu_gfx": self.device.gfx_target,
            "gpu_product_name": self.device.product_name,
            "gpu_oam_id": str(self.device.oam_id),
            "gpu_xcc_count": str(self.device.xcc_count),
            "gpu_device": str(self.device.device_ids[0]),
            "gpu_device_identity": self.device.stable_device_id,
            "gpu_device_uuid": self.device.device_uuid,
            "gpu_pci_bdf": self.device.pci_bdf,
            "gpu_compute_partition": self.device.compute_partition,
            "gpu_memory_partition": self.device.memory_partition,
            "gpu_partition_id": str(self.device.partition_id),
            "rocm_version": str(runtime.rocm_version),
            "vllm_version": runtime.version,
            "pytorch_version": str(runtime.pytorch_version),
            "python_version": str(runtime.python_version),
            "python_executable_sha256": str(runtime.executable_sha256),
            "environment_manifest_sha256": str(runtime.environment_manifest_sha256),
            "vllm_launcher_sha256": str(
                self.serving.engine_config["launcher_sha256"]
            ),
            "image_digest": runtime.image_digest or "none",
            "hf_model_id": self.model.model_id,
            "hf_revision": self.model.revision,
            "tokenizer_revision": self.model.tokenizer_revision,
            "model_snapshot_digest": self.model.snapshot_digest,
            "serving_protocol_sha256": self.serving.coordinate_sha256,
            "dtype": self.serving.dtype,
            "quantization": self.serving.quantization or "none",
            "tensor_parallel_size": str(self.serving.tensor_parallel_size),
        }


class VLLMExperimentSpec(StrictModel):
    """One V0 runtime-configuration experiment with locked workload coordinates."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str
    hypothesis_id: str
    description: str
    change: ChangeSet
    server_argv: list[str] = Field(min_length=1)
    benchmark_argv: list[str] = Field(min_length=1)
    server_env: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("id", "task_id", "hypothesis_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not value or any(character not in allowed for character in value):
            raise ValueError("identifiers may contain only letters, digits, '-' and '_'")
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("experiment description cannot be empty")
        return value

    @field_validator("server_argv", "benchmark_argv")
    @classmethod
    def validate_experiment_argv(cls, values: list[str]) -> list[str]:
        return list(validate_argv(values))

    @field_validator("server_env")
    @classmethod
    def validate_experiment_environment(cls, values: dict[str, str]) -> dict[str, str]:
        return _validate_environment(values)

    @model_validator(mode="after")
    def runtime_config_only(self) -> VLLMExperimentSpec:
        if self.change.kind != ChangeKind.RUNTIME_CONFIG:
            raise ValueError("vLLM V0 supports runtime_config experiments only")
        return self


class VLLMAgentDecision(StrictModel):
    current_stage: Literal[VLLMStage.AGENT_DECISION] = VLLMStage.AGENT_DECISION
    evidence_used: list[str] = Field(min_length=1)
    conclusion: str
    confidence: float = Field(ge=0, le=1)
    hypothesis: Hypothesis
    proposed_experiment: VLLMExperimentSpec
    proposed_next_stage: Literal[VLLMStage.EXPERIMENT] = VLLMStage.EXPERIMENT

    @field_validator("evidence_used")
    @classmethod
    def validate_evidence_used(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values) or len(values) != len(set(values)):
            raise ValueError("evidence_used must contain unique non-empty ids")
        return values

    @field_validator("conclusion")
    @classmethod
    def validate_conclusion(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("agent conclusion cannot be empty")
        return value

    @model_validator(mode="after")
    def bind_hypothesis(self) -> VLLMAgentDecision:
        if self.proposed_experiment.hypothesis_id != self.hypothesis.id:
            raise ValueError("proposed experiment must bind the decision hypothesis")
        return self


class VLLMStageCompletion(StrictModel):
    stage: VLLMStage
    evidence: dict[str, ArtifactRef]
    completed_at: datetime = Field(default_factory=utc_now)


class VLLMProfileApprovalState(StrictModel):
    attempt: int = Field(ge=1)
    request_id: str
    request_hash: str
    request: ArtifactRef
    context: ArtifactRef
    receipt: ArtifactRef | None = None
    status: VLLMApprovalStatus = VLLMApprovalStatus.REQUESTED

    @field_validator("request_hash")
    @classmethod
    def validate_request_hash(cls, value: str) -> str:
        return _validate_sha256(value, "request_hash")

    @model_validator(mode="after")
    def validate_receipt_state(self) -> VLLMProfileApprovalState:
        if self.status == VLLMApprovalStatus.REQUESTED and self.receipt is not None:
            raise ValueError("requested approval cannot already have a receipt")
        if self.status != VLLMApprovalStatus.REQUESTED and self.receipt is None:
            raise ValueError("approved or consumed approval requires a receipt artifact")
        return self


class VLLMWorkflowRecord(StrictModel):
    schema_version: Literal[1] = 1
    task_id: str
    config_sha256: str
    current_stage: VLLMStage = VLLMStage.INSPECT
    status: WorkflowStatus = WorkflowStatus.ACTIVE
    completions: list[VLLMStageCompletion] = Field(default_factory=list)
    baseline_server_request_hash: str | None = None
    baseline_server_requires_stop: bool = False
    baseline_server_start_failure: ArtifactRef | None = None
    baseline_run_failure: ArtifactRef | None = None
    baseline_server_stop: ArtifactRef | None = None
    profile_attempt_count: int = Field(default=0, ge=0)
    profile_approval: VLLMProfileApprovalState | None = None
    profile_execution_failure: ArtifactRef | None = None
    candidate_server_request_hash: str | None = None
    candidate_server_requires_stop: bool = False
    candidate_server_start_failure: ArtifactRef | None = None
    candidate_run_failure: ArtifactRef | None = None
    candidate_server_start: ArtifactRef | None = None
    pending_quality_result: ArtifactRef | None = None
    pending_candidate_result: ArtifactRef | None = None
    gpu_seconds_used: float = Field(default=0.0, ge=0)
    gpu_seconds_by_phase: dict[str, float] = Field(default_factory=dict)
    active_gpu_started_at: datetime | None = None
    active_gpu_phase: Literal["baseline_server", "candidate_server"] | None = None
    terminal_decision: GateDecision | None = None
    experiment_count: int = Field(default=0, ge=0)
    rerun_count: int = Field(default=0, ge=0)
    revision: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "config_sha256",
        "baseline_server_request_hash",
        "candidate_server_request_hash",
    )
    @classmethod
    def validate_workflow_sha256(cls, value: str | None) -> str | None:
        return _validate_sha256(value, "workflow digest") if value is not None else None

    @model_validator(mode="after")
    def validate_terminal_state(self) -> VLLMWorkflowRecord:
        if self.status == WorkflowStatus.ACTIVE and self.terminal_decision is not None:
            raise ValueError("active vLLM workflow cannot have a terminal decision")
        if self.status != WorkflowStatus.ACTIVE and self.terminal_decision is None:
            raise ValueError("terminal vLLM workflow requires a GateDecision")
        if (self.active_gpu_started_at is None) != (self.active_gpu_phase is None):
            raise ValueError("active GPU timestamp and phase must be set together")
        if any(seconds < 0 for seconds in self.gpu_seconds_by_phase.values()):
            raise ValueError("GPU phase durations cannot be negative")
        return self


class VLLMProfileExecutionPermit(StrictModel):
    request_id: str
    request_hash: str
    tool: Literal["rocm_profile_workload"] = "rocm_profile_workload"
    arguments: dict[str, Any]
    execution_context: dict[str, Any]

    @field_validator("request_hash")
    @classmethod
    def validate_permit_hash(cls, value: str) -> str:
        return _validate_sha256(value, "request_hash")


class VLLMProfileRuntimeEvidence(StrictModel):
    """Observed PID/environment/deployment proof for the offline vLLM engine."""

    verification: Literal["VERIFIED", "UNVERIFIED", "MISMATCH"]
    observed_pid: int = Field(gt=0)
    observed_boot_id: str
    observed_start_ticks: int = Field(gt=0)
    native_executable_sha256: str
    environment_manifest_sha256: str
    profile_launcher_sha256: str
    native_executable_matches: bool
    declared_environment_matches: bool
    unset_environment_absent: bool
    rocr_visible_devices: str
    image_digest_matches: bool | None = None
    container_binding_kind: Literal["cgroup_v2"] | None = None
    process_binding_id: str | None = None
    container_binding_id: str | None = None
    container_repo_digests: list[str] = Field(default_factory=list)

    @field_validator("observed_boot_id", "rocr_visible_devices")
    @classmethod
    def validate_profile_runtime_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\x00" in normalized:
            raise ValueError("profile runtime evidence text must be non-empty")
        return normalized

    @field_validator(
        "native_executable_sha256",
        "environment_manifest_sha256",
        "profile_launcher_sha256",
    )
    @classmethod
    def validate_profile_executable_hash(cls, value: str) -> str:
        return _validate_sha256(value, "profile native runtime")

    @field_validator("process_binding_id", "container_binding_id")
    @classmethod
    def validate_profile_binding_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("container binding ids cannot be blank")
        return value


class VLLMProfileResultEvidence(StrictModel):
    """ROCm MCP output plus observed same-environment execution coordinates."""

    mcp_call: dict[str, Any]
    execution_environment_artifact: ArtifactRef
    runtime_evidence: VLLMProfileRuntimeEvidence
    run_context_hash: str
    profiled_process_role: Literal["offline_vllm_engine"] = "offline_vllm_engine"
    image_digest: str | None = None
    model_snapshot_digest: str
    model_revision: str
    tokenizer_revision: str
    serving_protocol_sha256: str
    profile_command_sha256: str
    profile_launcher_sha256: str
    environment_manifest_sha256: str
    device_uuid: str
    pci_bdf: str
    partition_id: int = Field(ge=0)
    engine_kernel_count: int = Field(ge=1)

    @field_validator(
        "model_snapshot_digest",
        "serving_protocol_sha256",
        "profile_command_sha256",
        "profile_launcher_sha256",
        "environment_manifest_sha256",
    )
    @classmethod
    def validate_profile_hashes(cls, value: str) -> str:
        return _validate_sha256(value, "profile identity digest")

    @field_validator("image_digest")
    @classmethod
    def validate_profile_image_digest(cls, value: str | None) -> str | None:
        return _validate_sha256(value, "profile image digest") if value else None

    @field_validator("model_revision", "tokenizer_revision")
    @classmethod
    def validate_profile_revisions(cls, value: str) -> str:
        return _validate_revision(value, "profile revision")

    @field_validator("run_context_hash")
    @classmethod
    def validate_run_context_hash(cls, value: str) -> str:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
            raise ValueError("run_context_hash must use sha256:<64 lowercase hex>")
        return value

    @field_validator("device_uuid")
    @classmethod
    def validate_profile_uuid(cls, value: str | None) -> str | None:
        return MI300XDeviceBinding.validate_device_uuid(value)

    @field_validator("pci_bdf")
    @classmethod
    def validate_profile_bdf(cls, value: str) -> str:
        return MI300XDeviceBinding.validate_pci_bdf(value)


class VLLMNextAction(StrictModel):
    task_id: str
    stage: VLLMStage
    kind: VLLMNextActionKind
    required_artifacts: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class VLLMMetricSample(StrictModel):
    """Optional normalized single bench-serve sample for adapter implementations."""

    metrics: dict[str, float]

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, values: dict[str, float]) -> dict[str, float]:
        if not values or any(not name.strip() for name in values):
            raise ValueError("bench metrics must contain non-empty names")
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("bench metrics must be finite")
        return values


__all__ = [
    "MI300XInspectionEvidence",
    "MI300XDeviceBinding",
    "VLLMAgentDecision",
    "VLLMApprovalStatus",
    "VLLMCampaignConfig",
    "VLLMExperimentSpec",
    "VLLMMetricSample",
    "VLLMModelCoordinate",
    "VLLMNextAction",
    "VLLMNextActionKind",
    "VLLMProfileApprovalState",
    "VLLMProfileExecutionPermit",
    "VLLMProfileResultEvidence",
    "VLLMProfileRuntimeEvidence",
    "VLLMProfileProtocol",
    "VLLMQualityProtocol",
    "VLLMServingProtocol",
    "VLLMStage",
    "VLLMStageCompletion",
    "VLLMWorkflowRecord",
    "canonical_sha256",
]
