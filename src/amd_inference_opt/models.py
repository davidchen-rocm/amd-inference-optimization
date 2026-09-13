"""Versioned domain models shared by the workflow, adapters, and reports."""

from __future__ import annotations

import math
import re
import statistics
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    """Base class for persisted records; unknown fields are always an error."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class WorkflowStage(StrEnum):
    CREATE_TASK = "CREATE_TASK"
    INSPECT_TARGET = "INSPECT_TARGET"
    CAPTURE_BASELINE = "CAPTURE_BASELINE"
    DECOMPOSE_E2E = "DECOMPOSE_E2E"
    BUILD_EXECUTION_MAP = "BUILD_EXECUTION_MAP"
    DISCOVER_HOTSPOTS = "DISCOVER_HOTSPOTS"
    CLASSIFY_BOTTLENECK = "CLASSIFY_BOTTLENECK"
    ANALYZE_LIMIT = "ANALYZE_LIMIT"
    GENERATE_HYPOTHESIS = "GENERATE_HYPOTHESIS"
    CREATE_EXPERIMENT = "CREATE_EXPERIMENT"
    PATCH_AND_BUILD = "PATCH_AND_BUILD"
    MICROBENCH = "MICROBENCH"
    E2E_VALIDATION = "E2E_VALIDATION"
    QUALITY_VALIDATION = "QUALITY_VALIDATION"
    DECIDE = "DECIDE"


class WorkflowStatus(StrEnum):
    ACTIVE = "ACTIVE"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class DecisionOutcome(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


class ProfileLevel(StrEnum):
    ENVIRONMENT = "LEVEL_0"
    E2E = "LEVEL_1"
    KERNEL_TIMING = "LEVEL_2"
    TARGETED_COUNTERS = "LEVEL_3"
    ISA_THREAD_TRACE = "LEVEL_4"


class CampaignKind(StrEnum):
    """Select campaign-specific adapters without encoding policy in metadata."""

    GENERIC = "generic"
    Q4_RDNA = "q4_rdna"
    LLAMA_CPP_Q8 = "llama_cpp_q8"
    LLAMA_CPP_MIXED_QUANT = "llama_cpp_mixed_quant"
    LLAMA_CPP_SHAPE_KERNEL = "llama_cpp_shape_kernel"
    LLAMA_CPP_KV_CACHE = "llama_cpp_kv_cache"
    LLAMA_CPP_CONSUMER_AMD_FINAL = "llama_cpp_consumer_amd_final"
    VLLM_MI300X = "vllm_mi300x"


class BottleneckKind(StrEnum):
    MEMORY_BANDWIDTH = "memory_bandwidth"
    COMPUTE = "compute"
    LATENCY = "latency"
    DEPENDENCY = "dependency"
    OCCUPANCY = "occupancy"
    LAUNCH_OVERHEAD = "launch_overhead"
    SYNCHRONIZATION = "synchronization"
    CPU_SCHEDULING = "cpu_scheduling"
    UNKNOWN = "unknown"


class ChangeKind(StrEnum):
    SOURCE_PATCH = "source_patch"
    RUNTIME_CONFIG = "runtime_config"


class ChangePolicy(StrictModel):
    """Task-owned boundary for what an optimization experiment may change."""

    allowed_change_kinds: list[ChangeKind] = Field(
        default_factory=lambda: [ChangeKind.SOURCE_PATCH, ChangeKind.RUNTIME_CONFIG]
    )
    locked_coordinates: list[str] = Field(default_factory=list)
    allowed_runtime_args: list[str] = Field(default_factory=list)
    runtime_arg_arity: dict[str, Literal[0, 1]] = Field(default_factory=dict)

    @field_validator("allowed_change_kinds")
    @classmethod
    def validate_allowed_change_kinds(cls, values: list[Any]) -> list[Any]:
        if not values or any(not str(value).strip() for value in values):
            raise ValueError("change policy lists must contain non-empty unique values")
        rendered = [str(value) for value in values]
        if len(rendered) != len(set(rendered)):
            raise ValueError("change policy lists must contain non-empty unique values")
        return values

    @field_validator("locked_coordinates", "allowed_runtime_args")
    @classmethod
    def validate_optional_policy_lists(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values) or len(values) != len(set(values)):
            raise ValueError("change policy lists must contain non-empty unique values")
        return values

    @model_validator(mode="after")
    def validate_runtime_argument_policy(self) -> ChangePolicy:
        allowed = set(self.allowed_runtime_args)
        invalid = sorted(
            value
            for value in allowed
            if not value.startswith("-") or value in {"-", "--"} or "=" in value
        )
        if invalid:
            raise ValueError(
                "allowed_runtime_args must contain canonical option names: "
                + ", ".join(invalid)
            )
        unknown = sorted(set(self.runtime_arg_arity) - allowed)
        if unknown:
            raise ValueError(
                "runtime_arg_arity keys must also be allow-listed: "
                + ", ".join(unknown)
            )
        return self


class RunStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    SKIPPED = "SKIPPED"
    REUSED = "REUSED"


class QualityExecutionState(StrEnum):
    """Durable lifecycle for one exact long-running quality request."""

    STARTED = "STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    ORPHANED = "ORPHANED"


class ModelTarget(StrictModel):
    path: Path
    sha256: str | None = None
    architecture: str | None = None
    quantization: str | None = None
    sidecar_path: Path | None = None
    sidecar_sha256: str | None = None

    @field_validator("sha256", "sidecar_sha256")
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        invalid = value is not None and (
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        )
        if invalid:
            raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
        return value

    @field_validator("architecture", "quantization")
    @classmethod
    def validate_optional_model_coordinate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or any(ord(character) < 32 for character in normalized):
            raise ValueError("model coordinates must be non-empty printable text")
        return normalized


class RuntimeTarget(StrictModel):
    name: Literal["llama.cpp"] = "llama.cpp"
    repo_path: Path
    base_commit: str
    build_flags: list[str] = Field(default_factory=list)
    build_dir: Path = Path("build-amd-opt")
    backend: Literal["hip"] = "hip"
    prepared_binary_path: Path | None = None
    prepared_binary_sha256: str | None = None
    source_snapshot_sha256: str | None = None

    @field_validator("base_commit")
    @classmethod
    def nonempty_commit(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("base_commit cannot be empty")
        return value

    @field_validator("prepared_binary_sha256", "source_snapshot_sha256")
    @classmethod
    def validate_prepared_hash(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
        return value


class VLLMRuntimeTarget(StrictModel):
    """Immutable runtime coordinates for a native or containerized vLLM server."""

    name: Literal["vllm"] = "vllm"
    deployment: Literal["container", "native"] = "container"
    version: str
    executable: str = "vllm"
    executable_sha256: str | None = None
    environment_manifest_sha256: str | None = None
    image: str | None = None
    image_digest: str | None = None
    source_commit: str | None = None
    python_version: str | None = None
    pytorch_version: str | None = None
    rocm_version: str | None = None

    @field_validator(
        "version",
        "executable",
        "image",
        "source_commit",
        "python_version",
        "pytorch_version",
        "rocm_version",
    )
    @classmethod
    def validate_runtime_text(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.strip() or "\x00" in value or "\n" in value or "\r" in value
        ):
            raise ValueError("vLLM runtime coordinates must be non-empty single-line text")
        return value

    @field_validator(
        "executable_sha256",
        "environment_manifest_sha256",
        "image_digest",
    )
    @classmethod
    def validate_runtime_digest(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("vLLM runtime digests must be lowercase SHA-256 values")
        return value

    @model_validator(mode="after")
    def validate_deployment_identity(self) -> VLLMRuntimeTarget:
        if self.deployment == "container":
            if self.image is None or self.image_digest is None:
                raise ValueError("containerized vLLM requires image and image_digest")
        elif self.executable_sha256 is None or self.environment_manifest_sha256 is None:
            raise ValueError(
                "native vLLM requires executable_sha256 and "
                "environment_manifest_sha256"
            )
        return self

    @property
    def base_commit(self) -> str:
        """Compatibility revision used by read-only reports and run identities."""

        return (
            self.source_commit
            or self.image_digest
            or self.environment_manifest_sha256
            or f"vllm-{self.version}"
        )


RuntimeConfig = RuntimeTarget | VLLMRuntimeTarget


class GPUTarget(StrictModel):
    """GPU execution identity without hard-coding the current local architecture."""

    gfx_target: str = "gfx1201"
    device_id: int = Field(default=0, ge=0)
    name: str | None = None
    board_type: str | None = None

    @field_validator("gfx_target")
    @classmethod
    def validate_gfx_target(cls, value: str) -> str:
        normalized = value.strip().lower()
        if re.fullmatch(r"gfx[0-9a-f]+", normalized) is None:
            raise ValueError("gfx_target must use the canonical gfxNNNN form")
        return normalized

    @field_validator("name", "board_type")
    @classmethod
    def validate_optional_gpu_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("GPU text fields cannot be empty")
        return value


class ROCmTarget(StrictModel):
    version: str | None = None
    hip_version: str | None = None


class WorkloadConfig(StrictModel):
    kind: Literal["single_request_decode"] = "single_request_decode"
    prompt_tokens: int = Field(default=0, ge=0)
    generation_tokens: int = Field(default=128, ge=1)
    batch_size: Literal[1] = 1
    concurrency: Literal[1] = 1
    context_size: int | None = Field(default=None, ge=1)
    seed: int = 42


class VLLMServingWorkloadConfig(StrictModel):
    """Typed online-serving coordinates used by the MI300X vLLM workflow."""

    kind: Literal["online_serving"] = "online_serving"
    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    num_prompts: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    seed: int = 42

    @model_validator(mode="after")
    def validate_request_shape(self) -> VLLMServingWorkloadConfig:
        if self.num_prompts < self.concurrency:
            raise ValueError("num_prompts must be at least concurrency")
        return self


WorkloadTarget = WorkloadConfig | VLLMServingWorkloadConfig


class BenchmarkProtocol(StrictModel):
    warmup_runs: int = Field(default=2, ge=0)
    sample_count: int = Field(default=5, ge=1)
    timeout_seconds: int = Field(default=600, ge=1)
    benchmark_command: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)
    required_metrics: list[str] = Field(
        default_factory=lambda: ["tokens_per_second"]
    )
    protocol_hash: str | None = None

    @field_validator("required_metrics")
    @classmethod
    def validate_required_metrics(cls, values: list[str]) -> list[str]:
        if not values or any(not value.strip() for value in values):
            raise ValueError("required_metrics must contain non-empty metric names")
        if len(values) != len(set(values)):
            raise ValueError("required_metrics must be unique")
        return values

    @field_validator("protocol_hash")
    @classmethod
    def validate_protocol_hash(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("protocol_hash must be a lowercase SHA-256 digest")
        return value


class PerformanceMetricRequirement(StrictModel):
    metric: str
    minimum_improvement_percent: float = Field(default=0.0, ge=0)
    maximize: bool = True

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("metric cannot be empty")
        return value


class OptimizationObjective(StrictModel):
    primary_metric: str = "tokens_per_second"
    minimum_improvement_percent: float = Field(default=0.0, ge=0)
    maximize: Literal[True] = True
    metric_requirements: list[PerformanceMetricRequirement] = Field(
        default_factory=list
    )

    @field_validator("primary_metric")
    @classmethod
    def validate_primary_metric(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("primary_metric cannot be empty")
        return normalized

    @model_validator(mode="after")
    def unique_metric_requirements(self) -> OptimizationObjective:
        names = [item.metric for item in self.metric_requirements]
        if len(names) != len(set(names)):
            raise ValueError("metric_requirements must use unique metric names")
        return self

    def performance_requirements(self) -> tuple[PerformanceMetricRequirement, ...]:
        """Return explicit requirements or the backwards-compatible primary one."""

        if self.metric_requirements:
            return tuple(self.metric_requirements)
        return (
            PerformanceMetricRequirement(
                metric=self.primary_metric,
                minimum_improvement_percent=self.minimum_improvement_percent,
                maximize=self.maximize,
            ),
        )


class AccuracyRequirement(StrictModel):
    metric: str
    max_drop_percentage_points: float = Field(ge=0)

    @field_validator("metric")
    @classmethod
    def nonempty_metric(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("accuracy requirement metric cannot be empty")
        return value


class QualityConstraints(StrictModel):
    require_correctness: bool = True
    require_quality_evaluation: bool = True
    max_ppl_regression_percent: float | None = Field(default=None, ge=0)
    max_accuracy_drop_percentage_points: float | None = Field(default=None, ge=0)
    accuracy_metric: str = "math_accuracy"
    accuracy_requirements: list[AccuracyRequirement] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_accuracy_requirements(self) -> QualityConstraints:
        names = [item.metric for item in self.accuracy_requirements]
        if len(names) != len(set(names)):
            raise ValueError("accuracy_requirements must use unique metric names")
        return self

    def resolved_accuracy_requirements(self) -> tuple[AccuracyRequirement, ...]:
        if self.accuracy_requirements:
            return tuple(self.accuracy_requirements)
        if self.max_accuracy_drop_percentage_points is None:
            return ()
        return (
            AccuracyRequirement(
                metric=self.accuracy_metric,
                max_drop_percentage_points=self.max_accuracy_drop_percentage_points,
            ),
        )


class EnvironmentRequirements(StrictModel):
    required_match_fields: list[str] = Field(
        default_factory=lambda: [
            "gpu_gfx",
            "gpu_device",
            "rocm_version",
            "runtime_base_commit",
            "model_sha256",
            "build_flags_hash",
        ]
    )
    max_sample_cv_percent: float = Field(default=2.0, ge=0)
    # Telemetry is not integrated by every ROCm Issue Agent/GPU combination.
    # Tasks that have a reliable telemetry source can opt into the stricter gate.
    require_stable_telemetry: bool = False
    max_telemetry_clock_drift_percent: float = Field(default=5.0, ge=0, le=100)
    require_fresh_capture: bool = False
    require_run_identity: bool = False
    required_run_match_fields: list[str] = Field(
        default_factory=lambda: [
            "protocol_hash",
            "binary_sha256",
            "source_snapshot_sha256",
            "model_sha256",
            "runtime_libraries_hash",
            "environment_hash",
        ]
    )

    @field_validator("required_match_fields", "required_run_match_fields")
    @classmethod
    def validate_match_fields(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("comparability field names cannot be empty")
        if len(values) != len(set(values)):
            raise ValueError("comparability field names must be unique")
        return values


class TaskBudgets(StrictModel):
    max_experiments: int = Field(default=5, ge=1)
    max_gpu_minutes: float = Field(default=120.0, gt=0)
    build_timeout_seconds: int = Field(default=1800, ge=1)
    max_profile_level: ProfileLevel = ProfileLevel.KERNEL_TIMING


class MCPConfig(StrictModel):
    command: list[str] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Path | None = None
    timeout_seconds: int = Field(default=900, ge=1)


class OptimizationTask(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = Field(default_factory=utc_now)
    campaign_kind: CampaignKind = CampaignKind.GENERIC
    model: ModelTarget
    runtime: RuntimeConfig
    gpu: GPUTarget = Field(default_factory=GPUTarget)
    rocm: ROCmTarget = Field(default_factory=ROCmTarget)
    workload: WorkloadTarget = Field(default_factory=WorkloadConfig)
    benchmark: BenchmarkProtocol = Field(default_factory=BenchmarkProtocol)
    objective: OptimizationObjective = Field(default_factory=OptimizationObjective)
    quality: QualityConstraints = Field(default_factory=QualityConstraints)
    change_policy: ChangePolicy = Field(default_factory=ChangePolicy)
    environment: EnvironmentRequirements = Field(default_factory=EnvironmentRequirements)
    budgets: TaskBudgets = Field(default_factory=TaskBudgets)
    mcp: MCPConfig
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_campaign_kind(cls, value: Any) -> Any:
        """Keep persisted Q4 task documents valid while moving policy to a typed field."""

        if not isinstance(value, dict) or "campaign_kind" in value:
            return value
        metadata = value.get("metadata")
        if isinstance(metadata, dict) and metadata.get("live_q4rdna") == "true":
            migrated = dict(value)
            migrated["campaign_kind"] = CampaignKind.Q4_RDNA
            return migrated
        return value

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not value or any(c not in allowed for c in value):
            raise ValueError("id may contain only letters, digits, '-' and '_'")
        return value

    @model_validator(mode="after")
    def validate_runtime_campaign(self) -> OptimizationTask:
        if self.campaign_kind == CampaignKind.VLLM_MI300X:
            if not isinstance(self.runtime, VLLMRuntimeTarget):
                raise ValueError("vllm_mi300x campaign requires runtime.name='vllm'")
            if self.gpu.gfx_target != "gfx942":
                raise ValueError("vllm_mi300x campaign requires gpu.gfx_target='gfx942'")
            if not isinstance(self.workload, VLLMServingWorkloadConfig):
                raise ValueError(
                    "vllm_mi300x campaign requires workload.kind='online_serving'"
                )
        elif isinstance(self.runtime, VLLMRuntimeTarget):
            raise ValueError(
                "runtime.name='vllm' is supported only by campaign_kind='vllm_mi300x'"
            )
        elif isinstance(self.workload, VLLMServingWorkloadConfig):
            raise ValueError(
                "workload.kind='online_serving' is supported only by vllm_mi300x"
            )
        return self


class ArtifactRef(StrictModel):
    path: str
    sha256: str
    size: int = Field(ge=0)
    producer: str
    media_type: str = "application/octet-stream"
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("path")
    @classmethod
    def relative_safe_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("artifact path must be a safe task-relative path")
        return str(path)

    @field_validator("sha256")
    @classmethod
    def artifact_hash(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
        return value


class EvidenceRef(StrictModel):
    id: str
    kind: str
    summary: str
    artifact: ArtifactRef | None = None
    source: Literal["live", "recorded_import", "agent", "workflow"] = "live"


class MetricSeries(StrictModel):
    unit: str
    samples: list[float] = Field(min_length=1)

    @field_validator("samples")
    @classmethod
    def finite_samples(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("metric samples must be finite")
        return values

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def cv_percent(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        mean = abs(self.mean)
        if mean == 0:
            return math.inf if statistics.pstdev(self.samples) else 0.0
        return statistics.stdev(self.samples) / mean * 100


class EnvironmentFingerprint(StrictModel):
    values: dict[str, str]
    telemetry_stable: bool | None = None
    instability_reasons: list[str] = Field(default_factory=list)
    capture_id: str | None = None
    captured_at: datetime | None = None
    source: Literal["observed", "reused", "recorded_import"] = "observed"


class RunIdentity(StrictModel):
    """Typed coordinates that prove two benchmark runs are comparable."""

    protocol_hash: str
    binary_sha256: str
    source_snapshot_sha256: str
    model_sha256: str
    runtime_libraries_hash: str
    environment_hash: str
    sidecar_sha256: str | None = None
    command_hashes: dict[str, str] = Field(default_factory=dict)

    @field_validator(
        "protocol_hash",
        "binary_sha256",
        "source_snapshot_sha256",
        "model_sha256",
        "runtime_libraries_hash",
        "environment_hash",
        "sidecar_sha256",
    )
    @classmethod
    def validate_identity_hash(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("run identity values must be lowercase SHA-256 digests")
        return value

    @field_validator("command_hashes")
    @classmethod
    def validate_command_hashes(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not name.strip() for name in values):
            raise ValueError("command hash names cannot be empty")
        if any(
            len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in values.values()
        ):
            raise ValueError("command hashes must be lowercase SHA-256 digests")
        return values


class CommandResult(StrictModel):
    argv: list[str]
    cwd: Path
    status: RunStatus
    exit_code: int | None = None
    duration_seconds: float = Field(ge=0)
    stdout: ArtifactRef | None = None
    stderr: ArtifactRef | None = None


class BenchmarkResult(StrictModel):
    status: RunStatus
    metrics: dict[str, MetricSeries] = Field(default_factory=dict)
    command: CommandResult | None = None
    failure_reason: str | None = None


class QualityResult(StrictModel):
    status: RunStatus
    correctness_passed: bool | None = None
    perplexity: float | None = Field(default=None, gt=0)
    accuracies: dict[str, float] = Field(default_factory=dict)
    # Hash the evaluation protocol (dataset/version/prompts/evaluator), not the
    # candidate representation. Baseline and candidate protocol hashes must match.
    coordinate_hash: str | None = None
    representation_hash: str | None = None
    reused_from: str | None = None
    reused_from_representation_hash: str | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)

    @field_validator("accuracies")
    @classmethod
    def accuracy_range(cls, values: dict[str, float]) -> dict[str, float]:
        if any(value < 0 or value > 1 for value in values.values()):
            raise ValueError("accuracy values must be fractions in [0, 1]")
        return values


class BaselineResult(StrictModel):
    environment: EnvironmentFingerprint
    run_identity: RunIdentity | None = None
    build_status: RunStatus
    smoke_passed: bool
    benchmark: BenchmarkResult
    quality: QualityResult | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class TensorDescriptor(StrictModel):
    name: str | None = None
    shape: list[int] = Field(default_factory=list)
    dtype: str | None = None
    quantization: str | None = None


class RuntimeDescriptor(StrictModel):
    source_location: str | None = None
    operator_implementation: str | None = None
    backend: str = "hip"


class KernelDescriptor(StrictModel):
    name: str
    dispatch_count: int | None = Field(default=None, ge=0)
    duration_us: float | None = Field(default=None, ge=0)
    gpu_time_share_percent: float | None = Field(default=None, ge=0, le=100)
    grid: list[int] | None = None
    workgroup: list[int] | None = None
    metadata: dict[str, Any] | None = None


class ExecutionMapEntry(StrictModel):
    id: str
    phase: Literal["prefill", "decode"]
    layer: str | None = None
    operator: str
    tensor: TensorDescriptor = Field(default_factory=TensorDescriptor)
    runtime: RuntimeDescriptor = Field(default_factory=RuntimeDescriptor)
    kernels: list[KernelDescriptor] = Field(default_factory=list)
    hardware_behavior: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class InferenceExecutionMap(StrictModel):
    task_id: str
    entries: list[ExecutionMapEntry] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class BottleneckAssessment(StrictModel):
    kind: BottleneckKind
    evidence_ids: list[str] = Field(min_length=1)
    interpretation: str
    confidence: float = Field(ge=0, le=1)
    missing_evidence: list[str] = Field(default_factory=list)


class LimitEstimate(StrictModel):
    minimum_bytes: float | None = Field(default=None, ge=0)
    theoretical_bandwidth_gbps: float | None = Field(default=None, gt=0)
    theoretical_flops: float | None = Field(default=None, ge=0)
    theoretical_kernel_duration_us: float | None = Field(default=None, ge=0)
    hotspot_token_time_share_percent: float | None = Field(default=None, ge=0, le=100)
    maximum_e2e_improvement_percent: float = Field(ge=0)
    assumptions: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)


class Hypothesis(StrictModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    observed_evidence_ids: list[str] = Field(min_length=1)
    interpretation: str
    proposed_change: str
    expected_performance_signature: str
    expected_e2e_effect: str
    risks: list[str] = Field(default_factory=list)
    required_validation: list[str] = Field(min_length=1)
    stop_condition: str


class ChangeSet(StrictModel):
    kind: ChangeKind
    description: str
    patch_path: Path | None = None
    patch_sha256: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    unset_env: list[str] = Field(default_factory=list)
    candidate_model_path: Path | None = None
    candidate_model_sha256: str | None = None
    candidate_model_quantization: str | None = None
    runtime_args: list[str] = Field(default_factory=list)
    require_rebuild: bool | None = None

    @field_validator("candidate_model_sha256")
    @classmethod
    def validate_candidate_model_sha256(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("candidate_model_sha256 must be a lowercase SHA-256 digest")
        return value

    @field_validator("unset_env")
    @classmethod
    def validate_unset_env(cls, values: list[str]) -> list[str]:
        if any(not value or "=" in value or "\x00" in value for value in values):
            raise ValueError("unset_env entries must be non-empty environment names")
        if len(values) != len(set(values)):
            raise ValueError("unset_env entries must be unique")
        return values

    @field_validator("runtime_args")
    @classmethod
    def validate_runtime_args(cls, values: list[str]) -> list[str]:
        if any(not value or "\x00" in value for value in values):
            raise ValueError("runtime_args must contain non-empty NUL-free argv values")
        return values

    @model_validator(mode="after")
    def validate_kind(self) -> ChangeSet:
        if self.kind == ChangeKind.SOURCE_PATCH and self.patch_path is None:
            raise ValueError("source_patch requires patch_path")
        candidate_fields = (
            self.candidate_model_path,
            self.candidate_model_sha256,
            self.candidate_model_quantization,
        )
        if any(value is not None for value in candidate_fields) and not all(
            value is not None for value in candidate_fields
        ):
            raise ValueError(
                "candidate model changes require path, SHA-256, and quantization"
            )
        if self.candidate_model_path is not None and self.kind != ChangeKind.RUNTIME_CONFIG:
            raise ValueError("candidate model changes must reuse a runtime_config binary")
        if self.kind == ChangeKind.RUNTIME_CONFIG and not (
            self.env
            or self.unset_env
            or self.candidate_model_path is not None
            or self.runtime_args
        ):
            raise ValueError(
                "runtime_config requires an environment or candidate model change"
            )
        overlap = set(self.env) & set(self.unset_env)
        if overlap:
            raise ValueError(
                "environment variables cannot be both set and unset: "
                + ", ".join(sorted(overlap))
            )
        if self.require_rebuild is None:
            self.require_rebuild = self.kind == ChangeKind.SOURCE_PATCH
        return self


class StageCommandSpec(StrictModel):
    """One explicit runner step with its own process coordinates."""

    argv: list[str] = Field(min_length=1)
    cwd: Path = Path(".")
    env: dict[str, str] = Field(default_factory=dict)
    unset_env: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=600, ge=1)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, values: list[str]) -> list[str]:
        if any(not isinstance(value, str) or "\x00" in value for value in values):
            raise ValueError("argv entries must be strings without NUL bytes")
        return values

    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("stage command cwd must stay within the worktree")
        return value

    @field_validator("unset_env")
    @classmethod
    def validate_unset_env(cls, values: list[str]) -> list[str]:
        if any(not value or "=" in value or "\x00" in value for value in values):
            raise ValueError("unset_env entries must be valid environment names")
        if len(values) != len(set(values)):
            raise ValueError("unset_env entries must be unique")
        return values

    @model_validator(mode="after")
    def no_environment_overlap(self) -> StageCommandSpec:
        overlap = set(self.env) & set(self.unset_env)
        if overlap:
            raise ValueError(
                "environment variables cannot be both set and unset: "
                + ", ".join(sorted(overlap))
            )
        return self


class ProcessIdentity(StrictModel):
    """Linux process identity that remains safe when numeric PIDs are reused."""

    pid: int = Field(ge=1)
    boot_id: str = Field(min_length=1)
    start_ticks: int = Field(ge=0)


class QualityExecutionSpec(StrictModel):
    """The fully resolved request executed by a quality worker."""

    argv: list[str] = Field(min_length=1)
    cwd: Path
    env: dict[str, str] = Field(default_factory=dict)
    unset_env: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(gt=0)
    output_path: str
    stdout_path: str
    stderr_path: str

    @field_validator("argv")
    @classmethod
    def valid_quality_argv(cls, values: list[str]) -> list[str]:
        if any(not value or "\x00" in value for value in values):
            raise ValueError("quality argv entries must be non-empty and contain no NUL")
        return values

    @field_validator("output_path", "stdout_path", "stderr_path")
    @classmethod
    def task_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("quality paths must be safe task-relative paths")
        return str(path)


class QualityExecution(StrictModel):
    """Persistent attempt record used to resume without duplicate evaluation."""

    schema_version: Literal[1] = 1
    task_id: str
    experiment_id: str
    attempt_id: str
    request_hash: str
    spec_hash: str
    state: QualityExecutionState
    spec: QualityExecutionSpec
    worker: ProcessIdentity | None = None
    evaluator: ProcessIdentity | None = None
    started_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    heartbeat_at: datetime | None = None
    terminal_at: datetime | None = None
    exit_code: int | None = None
    failure_reason: str | None = None
    output_artifact: ArtifactRef | None = None
    stdout_artifact: ArtifactRef | None = None
    stderr_artifact: ArtifactRef | None = None

    @field_validator("request_hash", "spec_hash")
    @classmethod
    def quality_hash(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("quality execution hashes must be lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def valid_execution_state(self) -> QualityExecution:
        terminal = self.state in {
            QualityExecutionState.COMPLETED,
            QualityExecutionState.FAILED,
            QualityExecutionState.TIMED_OUT,
            QualityExecutionState.ORPHANED,
        }
        if terminal != (self.terminal_at is not None):
            raise ValueError("only terminal quality executions have terminal_at")
        if self.state == QualityExecutionState.RUNNING and (
            self.worker is None or self.heartbeat_at is None
        ):
            raise ValueError("running quality execution requires worker and heartbeat")
        if self.state == QualityExecutionState.COMPLETED and (
            self.exit_code != 0
            or self.output_artifact is None
            or self.stdout_artifact is None
            or self.stderr_artifact is None
        ):
            raise ValueError(
                "completed quality execution requires exit zero and bound output streams"
            )
        return self


class ExperimentSpec(StrictModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str
    hypothesis_id: str
    change: ChangeSet
    commands: dict[str, list[str]] = Field(default_factory=dict)
    stage_commands: dict[str, StageCommandSpec] = Field(default_factory=dict)
    worktree_path: Path | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ExperimentResult(StrictModel):
    experiment_id: str
    environment: EnvironmentFingerprint | None = None
    run_identity: RunIdentity | None = None
    build_status: RunStatus = RunStatus.SKIPPED
    build: CommandResult | None = None
    smoke_passed: bool | None = None
    microbenchmark: BenchmarkResult | None = None
    e2e: BenchmarkResult | None = None
    quality: QualityResult | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    failure_reason: str | None = None


class GateCheck(StrictModel):
    name: str
    passed: bool | None
    detail: str


class GateDecision(StrictModel):
    outcome: DecisionOutcome
    checks: list[GateCheck]
    reasons: list[str]
    improvement_percent: float | None = None
    metric_improvements_percent: dict[str, float] = Field(default_factory=dict)
    rerun_from_stage: WorkflowStage | None = None
    decided_at: datetime = Field(default_factory=utc_now)


class AgentDecision(StrictModel):
    current_stage: WorkflowStage
    evidence_used: list[str]
    conclusion: str
    confidence: float = Field(ge=0, le=1)
    missing_evidence: list[str] = Field(default_factory=list)
    requested_profile_level: ProfileLevel | None = None
    execution_map_updates: list[ExecutionMapEntry] = Field(default_factory=list)
    bottleneck_assessment: BottleneckAssessment | None = None
    limit_estimate: LimitEstimate | None = None
    hypothesis: Hypothesis | None = None
    proposed_experiment: ExperimentSpec | None = None
    proposed_next_stage: WorkflowStage


class StageCompletion(StrictModel):
    stage: WorkflowStage
    evidence_ids: dict[str, str]
    evidence_artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def evidence_artifacts_match_ids(self) -> StageCompletion:
        unknown = set(self.evidence_artifacts) - set(self.evidence_ids)
        if unknown:
            raise ValueError(
                "evidence_artifacts contain unknown evidence keys: "
                + ", ".join(sorted(unknown))
            )
        mismatches = [
            key
            for key, artifact in self.evidence_artifacts.items()
            if artifact.path != self.evidence_ids[key]
        ]
        if mismatches:
            raise ValueError(
                "evidence artifact paths do not match evidence ids: "
                + ", ".join(sorted(mismatches))
            )
        return self


class WorkflowRecord(StrictModel):
    schema_version: Literal[1] = 1
    task_id: str
    current_stage: WorkflowStage = WorkflowStage.CREATE_TASK
    status: WorkflowStatus = WorkflowStatus.ACTIVE
    completions: list[StageCompletion] = Field(default_factory=list)
    terminal_decision: GateDecision | None = None
    experiment_count: int = Field(default=0, ge=0)
    rerun_count: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def terminal_invariants(self) -> WorkflowRecord:
        if self.status == WorkflowStatus.ACTIVE and self.terminal_decision is not None:
            raise ValueError("active workflow cannot have a terminal decision")
        if self.status != WorkflowStatus.ACTIVE and self.terminal_decision is None:
            raise ValueError("terminal workflow requires terminal_decision")
        return self


class AgentContext(StrictModel):
    schema_version: Literal[1] = 1
    task_id: str
    current_stage: WorkflowStage
    permitted_next_stages: list[WorkflowStage]
    evidence: list[EvidenceRef]
    missing_evidence: list[str]
    experiment_count: int
    experiment_budget: int
    decision_schema: dict[str, Any]


class ApprovalRequest(StrictModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str
    tool: str
    arguments: dict[str, Any]
    request_hash: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("request_hash")
    @classmethod
    def valid_request_hash(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("request_hash must be a lowercase SHA-256 digest")
        return value


class ApprovalReceipt(StrictModel):
    request_id: str
    request_hash: str
    approved_at: datetime = Field(default_factory=utc_now)
    consumed_at: datetime | None = None

    @field_validator("request_hash")
    @classmethod
    def valid_request_hash(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("request_hash must be a lowercase SHA-256 digest")
        return value
