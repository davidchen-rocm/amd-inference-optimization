"""Evidence-bounded follow-up experiments for the accepted Q4_RDNA path."""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from .protocol import BASELINE_UNSET_ENVIRONMENT, amd_runtime_environment


class Q4RDNATuningError(ValueError):
    """A Q4_RDNA tuning coordinate or measurement is invalid."""


@dataclass(frozen=True)
class Q4RDNAFormat:
    group_weights: int = 64
    tile_rows: int = 32
    tile_bytes: int = 1088
    bits_per_weight: float = 4.25

    def __post_init__(self) -> None:
        if self.group_weights <= 0 or self.tile_rows <= 0 or self.tile_bytes <= 0:
            raise Q4RDNATuningError("Q4_RDNA format dimensions must be positive")
        if self.bits_per_weight <= 0:
            raise Q4RDNATuningError("bits_per_weight must be positive")

    def packed_matrix_bytes(self, *, rows: int, columns: int) -> int:
        if rows <= 0 or columns <= 0:
            raise Q4RDNATuningError("matrix dimensions must be positive")
        if rows % self.tile_rows or columns % self.group_weights:
            raise Q4RDNATuningError("matrix is not aligned to the Q4_RDNA tile")
        return rows // self.tile_rows * (columns // self.group_weights) * self.tile_bytes


@dataclass(frozen=True)
class GateUpShape:
    rows: int = 12288
    columns: int = 4096
    weight_matrices: int = 2
    activation_bytes_per_element: int = 4
    output_bytes_per_element: int = 4

    def __post_init__(self) -> None:
        if min(
            self.rows,
            self.columns,
            self.weight_matrices,
            self.activation_bytes_per_element,
            self.output_bytes_per_element,
        ) <= 0:
            raise Q4RDNATuningError("gate/up shape values must be positive")


@dataclass(frozen=True)
class GateUpPerformanceModel:
    shape: GateUpShape
    representation: Q4RDNAFormat
    packed_weight_bytes: int
    activation_bytes: int
    output_bytes: int
    minimum_bytes: int
    measured_kernel_us: float
    theoretical_bandwidth_gbps: float
    theoretical_kernel_us: float
    effective_bandwidth_gbps: float
    bandwidth_efficiency_percent: float
    kernel_time_headroom_percent: float
    hotspot_gpu_time_share_percent: float
    maximum_e2e_improvement_percent: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_gate_up_performance_model(
    *,
    measured_kernel_us: float,
    hotspot_gpu_time_share_percent: float,
    theoretical_bandwidth_gbps: float,
    shape: GateUpShape | None = None,
    representation: Q4RDNAFormat | None = None,
) -> GateUpPerformanceModel:
    """Estimate the remaining gate/up opportunity from compulsory byte traffic."""

    shape = shape or GateUpShape()
    representation = representation or Q4RDNAFormat()
    if not math.isfinite(measured_kernel_us) or measured_kernel_us <= 0:
        raise Q4RDNATuningError("measured_kernel_us must be finite and positive")
    if not math.isfinite(theoretical_bandwidth_gbps) or theoretical_bandwidth_gbps <= 0:
        raise Q4RDNATuningError("theoretical_bandwidth_gbps must be finite and positive")
    if not math.isfinite(hotspot_gpu_time_share_percent) or not (
        0 <= hotspot_gpu_time_share_percent <= 100
    ):
        raise Q4RDNATuningError("hotspot share must be between zero and 100")

    one_matrix = representation.packed_matrix_bytes(
        rows=shape.rows,
        columns=shape.columns,
    )
    packed_weight_bytes = one_matrix * shape.weight_matrices
    activation_bytes = shape.columns * shape.activation_bytes_per_element
    output_bytes = shape.rows * shape.output_bytes_per_element
    minimum_bytes = packed_weight_bytes + activation_bytes + output_bytes
    theoretical_kernel_us = minimum_bytes / (theoretical_bandwidth_gbps * 1000.0)
    effective_bandwidth_gbps = minimum_bytes / (measured_kernel_us * 1000.0)
    bandwidth_efficiency = 100.0 * effective_bandwidth_gbps / theoretical_bandwidth_gbps
    theoretical_fraction = min(theoretical_kernel_us / measured_kernel_us, 1.0)
    kernel_headroom = 100.0 * (1.0 - theoretical_fraction)
    total_time_reduction = (
        hotspot_gpu_time_share_percent / 100.0 * (1.0 - theoretical_fraction)
    )
    maximum_e2e = (
        100.0 * (1.0 / (1.0 - total_time_reduction) - 1.0)
        if total_time_reduction < 1.0
        else math.inf
    )
    return GateUpPerformanceModel(
        shape=shape,
        representation=representation,
        packed_weight_bytes=packed_weight_bytes,
        activation_bytes=activation_bytes,
        output_bytes=output_bytes,
        minimum_bytes=minimum_bytes,
        measured_kernel_us=measured_kernel_us,
        theoretical_bandwidth_gbps=theoretical_bandwidth_gbps,
        theoretical_kernel_us=theoretical_kernel_us,
        effective_bandwidth_gbps=effective_bandwidth_gbps,
        bandwidth_efficiency_percent=bandwidth_efficiency,
        kernel_time_headroom_percent=kernel_headroom,
        hotspot_gpu_time_share_percent=hotspot_gpu_time_share_percent,
        maximum_e2e_improvement_percent=maximum_e2e,
    )


@dataclass(frozen=True)
class GateUpMappingArm:
    split_waves: int
    environment: dict[str, str]
    unset_environment: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split_waves not in {2, 4, 8}:
            raise Q4RDNATuningError("gate/up split_waves must be 2, 4, or 8")
        if self.environment.get("LLAMA_Q4_RDNA_COOP") != str(self.split_waves):
            raise Q4RDNATuningError("mapping environment does not match split_waves")
        if self.environment.get("LLAMA_Q4_RDNA_SCOPE") != "hotspot":
            raise Q4RDNATuningError("gate/up mapping must use hotspot-only routing")
        if "LLAMA_Q4_RDNA_MAPPING" not in self.unset_environment:
            raise Q4RDNATuningError("old mapping must be explicitly unset")

    @property
    def id(self) -> str:
        return f"gate-up-split-{self.split_waves}"


def gate_up_mapping_arm(
    split_waves: int,
    *,
    sidecar_path: str | Path,
    device_id: int = 0,
    rocm_library_paths: tuple[str, ...] = (
        "/opt/rocm/core-7.14/lib",
        "/opt/rocm/lib",
    ),
) -> GateUpMappingArm:
    sidecar = Path(sidecar_path).resolve()
    if not sidecar.is_file():
        raise Q4RDNATuningError(f"Q4_RDNA sidecar does not exist: {sidecar}")
    environment = amd_runtime_environment(
        device_id=device_id,
        rocm_library_paths=rocm_library_paths,
        extra={
            "LLAMA_Q4_RDNA_SIDECAR": str(sidecar),
            "LLAMA_Q4_RDNA_SCOPE": "hotspot",
            "LLAMA_Q4_RDNA_COOP": str(split_waves),
            "LLAMA_Q4_RDNA_TRACE": "1",
        },
    )
    preserved = set(environment)
    unset = tuple(name for name in BASELINE_UNSET_ENVIRONMENT if name not in preserved)
    return GateUpMappingArm(
        split_waves=split_waves,
        environment=environment,
        unset_environment=unset,
    )


@dataclass(frozen=True)
class GateUpBenchmarkMeasurement:
    arm_id: str
    split_waves: int
    samples_tokens_per_second: tuple[float, ...]
    activation_verified: bool

    def __post_init__(self) -> None:
        if self.split_waves not in {2, 4, 8}:
            raise Q4RDNATuningError("measurement split_waves must be 2, 4, or 8")
        if not self.samples_tokens_per_second:
            raise Q4RDNATuningError("measurement requires token/s samples")
        if any(not math.isfinite(value) or value <= 0 for value in self.samples_tokens_per_second):
            raise Q4RDNATuningError("token/s samples must be finite and positive")

    @property
    def mean_tokens_per_second(self) -> float:
        return statistics.fmean(self.samples_tokens_per_second)

    @property
    def cv_percent(self) -> float:
        if len(self.samples_tokens_per_second) == 1:
            return 0.0
        return (
            statistics.stdev(self.samples_tokens_per_second)
            / self.mean_tokens_per_second
            * 100.0
        )

    def to_dict(self) -> dict[str, object]:
        document = asdict(self)
        document["mean_tokens_per_second"] = self.mean_tokens_per_second
        document["cv_percent"] = self.cv_percent
        return document


@dataclass(frozen=True)
class GateUpPairRuntimeMeasurement:
    arm_id: str
    paired_layout: bool
    samples_tokens_per_second: tuple[float, ...]
    activation_verified: bool
    paired_tensor_count: int

    def __post_init__(self) -> None:
        if not self.samples_tokens_per_second:
            raise Q4RDNATuningError("measurement requires token/s samples")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.samples_tokens_per_second
        ):
            raise Q4RDNATuningError("token/s samples must be finite and positive")
        if self.paired_tensor_count < 0:
            raise Q4RDNATuningError("paired_tensor_count cannot be negative")
        expected_pairs = 36 if self.paired_layout else 0
        if self.activation_verified and self.paired_tensor_count != expected_pairs:
            raise Q4RDNATuningError("verified arm has the wrong paired tensor count")

    @property
    def mean_tokens_per_second(self) -> float:
        return statistics.fmean(self.samples_tokens_per_second)

    @property
    def cv_percent(self) -> float:
        if len(self.samples_tokens_per_second) == 1:
            return 0.0
        return (
            statistics.stdev(self.samples_tokens_per_second)
            / self.mean_tokens_per_second
            * 100.0
        )

    def to_dict(self) -> dict[str, object]:
        document = asdict(self)
        document["mean_tokens_per_second"] = self.mean_tokens_per_second
        document["cv_percent"] = self.cv_percent
        return document


@dataclass(frozen=True)
class GateQ3RuntimeMeasurement:
    arm_id: str
    q3_gate: bool
    samples_tokens_per_second: tuple[float, ...]
    activation_verified: bool
    q3_tensor_count: int

    def __post_init__(self) -> None:
        if not self.samples_tokens_per_second:
            raise Q4RDNATuningError("measurement requires token/s samples")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.samples_tokens_per_second
        ):
            raise Q4RDNATuningError("token/s samples must be finite and positive")
        expected = 36 if self.q3_gate else 0
        if self.q3_tensor_count < 0:
            raise Q4RDNATuningError("q3_tensor_count cannot be negative")
        if self.activation_verified and self.q3_tensor_count != expected:
            raise Q4RDNATuningError("verified arm has the wrong Q3 tensor count")

    @property
    def mean_tokens_per_second(self) -> float:
        return statistics.fmean(self.samples_tokens_per_second)

    @property
    def cv_percent(self) -> float:
        if len(self.samples_tokens_per_second) == 1:
            return 0.0
        return (
            statistics.stdev(self.samples_tokens_per_second)
            / self.mean_tokens_per_second
            * 100.0
        )

    def to_dict(self) -> dict[str, object]:
        document = asdict(self)
        document["mean_tokens_per_second"] = self.mean_tokens_per_second
        document["cv_percent"] = self.cv_percent
        return document


def drop_initial_warmup_samples(
    samples: tuple[float, ...],
    *,
    warmup_samples: int,
) -> tuple[float, ...]:
    """Drop an explicit same-coordinate warmup prefix before scoring."""

    if warmup_samples < 0:
        raise Q4RDNATuningError("warmup_samples must be non-negative")
    if warmup_samples >= len(samples):
        raise Q4RDNATuningError("warmup_samples would consume every sample")
    scored = samples[warmup_samples:]
    if any(not math.isfinite(value) or value <= 0 for value in scored):
        raise Q4RDNATuningError("scored samples must be finite and positive")
    return scored


def normalize_llama_cli_generation(stdout: str) -> str:
    """Remove llama-cli's volatile timing footer without changing generated text."""

    normalized = stdout.replace("\r\n", "\n")
    normalized = re.sub(
        r"\n\[ Prompt: [^\n]+ \| Generation: [^\n]+ \]\n+(?:Exiting\.\.\.\n?)?\Z",
        "",
        normalized,
    )
    return normalized.rstrip()


class MappingScreenOutcome(StrEnum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class MappingScreenResult:
    baseline_arm_id: str
    candidate_arm_id: str
    outcome: MappingScreenOutcome
    improvement_percent: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GateUpPairFinalPolicy:
    minimum_tg128_improvement_percent: float
    minimum_tg512_improvement_percent: float
    maximum_extra_vram_percent: float

    def __post_init__(self) -> None:
        values = (
            self.minimum_tg128_improvement_percent,
            self.minimum_tg512_improvement_percent,
            self.maximum_extra_vram_percent,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise Q4RDNATuningError("final pair policy values must be finite and non-negative")
        if self.maximum_extra_vram_percent > 100:
            raise Q4RDNATuningError("maximum_extra_vram_percent cannot exceed 100")


@dataclass(frozen=True)
class GateUpPairFinalEvidence:
    tg128_improvement_percent: float
    tg512_improvement_percent: float
    baseline_kernel_average_ns: float
    candidate_kernel_average_ns: float
    extra_vram_bytes: int
    total_vram_bytes: int
    correctness_passed: bool
    profiler_complete: bool

    @property
    def kernel_average_improvement_percent(self) -> float:
        return (1.0 - self.candidate_kernel_average_ns / self.baseline_kernel_average_ns) * 100

    @property
    def extra_vram_percent(self) -> float:
        return self.extra_vram_bytes / self.total_vram_bytes * 100


@dataclass(frozen=True)
class GateUpPairFinalResult:
    outcome: MappingScreenOutcome
    checks: dict[str, bool]
    metrics: dict[str, float]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_gate_up_pair_final(
    evidence: GateUpPairFinalEvidence,
    policy: GateUpPairFinalPolicy,
) -> GateUpPairFinalResult:
    """Apply an explicit performance/correctness/VRAM Gate to the final candidate."""

    numeric = (
        evidence.tg128_improvement_percent,
        evidence.tg512_improvement_percent,
        evidence.baseline_kernel_average_ns,
        evidence.candidate_kernel_average_ns,
    )
    if (
        any(not math.isfinite(value) for value in numeric)
        or min(evidence.baseline_kernel_average_ns, evidence.candidate_kernel_average_ns) <= 0
        or evidence.extra_vram_bytes < 0
        or evidence.total_vram_bytes <= 0
    ):
        raise Q4RDNATuningError("final pair evidence contains invalid numeric values")
    checks = {
        "profiler_complete": evidence.profiler_complete,
        "correctness_passed": evidence.correctness_passed,
        "tg128_improvement": evidence.tg128_improvement_percent
        >= policy.minimum_tg128_improvement_percent,
        "tg512_improvement": evidence.tg512_improvement_percent
        >= policy.minimum_tg512_improvement_percent,
        "kernel_average_improvement": evidence.candidate_kernel_average_ns
        < evidence.baseline_kernel_average_ns,
        "extra_vram_within_budget": evidence.extra_vram_percent
        <= policy.maximum_extra_vram_percent,
    }
    metrics = {
        "tg128_improvement_percent": evidence.tg128_improvement_percent,
        "tg512_improvement_percent": evidence.tg512_improvement_percent,
        "kernel_average_improvement_percent": evidence.kernel_average_improvement_percent,
        "extra_vram_bytes": float(evidence.extra_vram_bytes),
        "extra_vram_percent": evidence.extra_vram_percent,
    }
    if not checks["profiler_complete"]:
        return GateUpPairFinalResult(
            outcome=MappingScreenOutcome.INCONCLUSIVE,
            checks=checks,
            metrics=metrics,
            reasons=("required kernel profile is incomplete",),
        )
    failed = tuple(name for name, passed in checks.items() if not passed)
    return GateUpPairFinalResult(
        outcome=(
            MappingScreenOutcome.PROMOTE
            if not failed
            else MappingScreenOutcome.REJECT
        ),
        checks=checks,
        metrics=metrics,
        reasons=(
            ("all configured final Gates passed",)
            if not failed
            else ("failed configured Gates: " + ", ".join(failed),)
        ),
    )


def screen_gate_up_mapping(
    baseline: GateUpBenchmarkMeasurement,
    candidate: GateUpBenchmarkMeasurement,
    *,
    maximum_cv_percent: float = 2.0,
    minimum_improvement_percent: float = 0.5,
) -> MappingScreenResult:
    """Screen a mapping before profiler and full-model quality work."""

    if baseline.split_waves != 8:
        raise Q4RDNATuningError("the accepted gate/up mapping baseline must use split_waves=8")
    if maximum_cv_percent <= 0 or minimum_improvement_percent < 0:
        raise Q4RDNATuningError("screen thresholds are invalid")
    if not baseline.activation_verified or not candidate.activation_verified:
        return MappingScreenResult(
            baseline_arm_id=baseline.arm_id,
            candidate_arm_id=candidate.arm_id,
            outcome=MappingScreenOutcome.INCONCLUSIVE,
            improvement_percent=None,
            reasons=("Q4_RDNA hotspot activation was not verified in both arms",),
        )
    unstable = [
        item.arm_id
        for item in (baseline, candidate)
        if item.cv_percent > maximum_cv_percent
    ]
    if unstable:
        return MappingScreenResult(
            baseline_arm_id=baseline.arm_id,
            candidate_arm_id=candidate.arm_id,
            outcome=MappingScreenOutcome.INCONCLUSIVE,
            improvement_percent=None,
            reasons=("unstable benchmark arms: " + ", ".join(unstable),),
        )
    improvement = (
        candidate.mean_tokens_per_second / baseline.mean_tokens_per_second - 1.0
    ) * 100.0
    if improvement >= minimum_improvement_percent:
        return MappingScreenResult(
            baseline_arm_id=baseline.arm_id,
            candidate_arm_id=candidate.arm_id,
            outcome=MappingScreenOutcome.PROMOTE,
            improvement_percent=improvement,
            reasons=(
                "candidate passed the short performance screen; profile and correctness "
                "remain required",
            ),
        )
    return MappingScreenResult(
        baseline_arm_id=baseline.arm_id,
        candidate_arm_id=candidate.arm_id,
        outcome=MappingScreenOutcome.REJECT,
        improvement_percent=improvement,
        reasons=("candidate did not meet the short performance threshold",),
    )


def screen_gate_up_pair_load(
    result: dict[str, object],
    *,
    maximum_cv_percent: float = 1.0,
    minimum_improvement_percent: float = 1.0,
) -> MappingScreenResult:
    """Decide whether a standalone paired-load result merits runtime integration."""

    required_numbers = (
        "max_absolute_error",
        "separate_cv_percent",
        "paired_cv_percent",
        "paired_improvement_percent",
    )
    values: dict[str, float] = {}
    for name in required_numbers:
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Q4RDNATuningError(f"pair-load result is missing numeric {name}")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise Q4RDNATuningError(f"pair-load result {name} must be finite")
        values[name] = parsed
    if values["max_absolute_error"] != 0:
        return MappingScreenResult(
            baseline_arm_id="separate-gate-up-loads",
            candidate_arm_id="paired-gate-up-load",
            outcome=MappingScreenOutcome.REJECT,
            improvement_percent=values["paired_improvement_percent"],
            reasons=("paired layout changed the mathematical result",),
        )
    if max(values["separate_cv_percent"], values["paired_cv_percent"]) > maximum_cv_percent:
        return MappingScreenResult(
            baseline_arm_id="separate-gate-up-loads",
            candidate_arm_id="paired-gate-up-load",
            outcome=MappingScreenOutcome.INCONCLUSIVE,
            improvement_percent=None,
            reasons=("pair-load microbenchmark exceeded the CV threshold",),
        )
    improvement = values["paired_improvement_percent"]
    if improvement >= minimum_improvement_percent:
        return MappingScreenResult(
            baseline_arm_id="separate-gate-up-loads",
            candidate_arm_id="paired-gate-up-load",
            outcome=MappingScreenOutcome.PROMOTE,
            improvement_percent=improvement,
            reasons=("paired load merits a source integration experiment",),
        )
    return MappingScreenResult(
        baseline_arm_id="separate-gate-up-loads",
        candidate_arm_id="paired-gate-up-load",
        outcome=MappingScreenOutcome.REJECT,
        improvement_percent=improvement,
        reasons=("paired load did not meet the microbenchmark threshold",),
    )


def screen_gate_q3_microbenchmark(
    result: dict[str, object],
    *,
    maximum_cv_percent: float = 1.0,
    minimum_improvement_percent: float = 2.0,
) -> MappingScreenResult:
    """Decide whether Q4-up/Q3-gate merits a real sidecar and runtime build."""

    required = (
        "max_absolute_error",
        "q4_cv_percent",
        "q3_cv_percent",
        "q3_improvement_percent",
        "fused_weight_byte_reduction_percent",
    )
    values: dict[str, float] = {}
    for name in required:
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Q4RDNATuningError(f"Q3 result is missing numeric {name}")
        values[name] = float(value)
        if not math.isfinite(values[name]):
            raise Q4RDNATuningError(f"Q3 result {name} must be finite")
    if values["max_absolute_error"] != 0:
        outcome = MappingScreenOutcome.REJECT
        reasons = ("Q3 packing/unpacking changed the controlled mathematical result",)
    elif max(values["q4_cv_percent"], values["q3_cv_percent"]) > maximum_cv_percent:
        outcome = MappingScreenOutcome.INCONCLUSIVE
        reasons = ("Q3 microbenchmark exceeded the CV threshold",)
    elif values["fused_weight_byte_reduction_percent"] <= 0:
        outcome = MappingScreenOutcome.REJECT
        reasons = ("Q3 candidate did not reduce fused gate/up weight bytes",)
    elif values["q3_improvement_percent"] >= minimum_improvement_percent:
        outcome = MappingScreenOutcome.PROMOTE
        reasons = ("Q3 gate passed the exact-shape performance screen",)
    else:
        outcome = MappingScreenOutcome.REJECT
        reasons = ("Q3 unpack cost consumed the expected byte-traffic benefit",)
    return MappingScreenResult(
        baseline_arm_id="q4-up-q4-gate",
        candidate_arm_id="q4-up-q3-gate",
        outcome=outcome,
        improvement_percent=(
            values["q3_improvement_percent"]
            if outcome != MappingScreenOutcome.INCONCLUSIVE
            else None
        ),
        reasons=reasons,
    )


def screen_gate_up_pair_runtime(
    baseline: GateUpPairRuntimeMeasurement,
    candidate: GateUpPairRuntimeMeasurement,
    *,
    maximum_cv_percent: float = 2.0,
    minimum_improvement_percent: float = 0.5,
) -> MappingScreenResult:
    """Screen the paired gate/up layout in the real model before profiling."""

    if baseline.paired_layout or not candidate.paired_layout:
        raise Q4RDNATuningError("pair runtime screen requires separate then paired arms")
    if maximum_cv_percent <= 0 or minimum_improvement_percent < 0:
        raise Q4RDNATuningError("screen thresholds are invalid")
    if not baseline.activation_verified or not candidate.activation_verified:
        return MappingScreenResult(
            baseline_arm_id=baseline.arm_id,
            candidate_arm_id=candidate.arm_id,
            outcome=MappingScreenOutcome.INCONCLUSIVE,
            improvement_percent=None,
            reasons=("Q4_RDNA gate/up activation was not verified in both arms",),
        )
    unstable = [
        item.arm_id
        for item in (baseline, candidate)
        if item.cv_percent > maximum_cv_percent
    ]
    if unstable:
        return MappingScreenResult(
            baseline_arm_id=baseline.arm_id,
            candidate_arm_id=candidate.arm_id,
            outcome=MappingScreenOutcome.INCONCLUSIVE,
            improvement_percent=None,
            reasons=("unstable benchmark arms: " + ", ".join(unstable),),
        )
    improvement = (
        candidate.mean_tokens_per_second / baseline.mean_tokens_per_second - 1.0
    ) * 100.0
    outcome = (
        MappingScreenOutcome.PROMOTE
        if improvement >= minimum_improvement_percent
        else MappingScreenOutcome.REJECT
    )
    reason = (
        "paired layout passed the real-model screen; profiling remains required"
        if outcome == MappingScreenOutcome.PROMOTE
        else "paired layout did not meet the real-model performance threshold"
    )
    return MappingScreenResult(
        baseline_arm_id=baseline.arm_id,
        candidate_arm_id=candidate.arm_id,
        outcome=outcome,
        improvement_percent=improvement,
        reasons=(reason,),
    )


def screen_gate_q3_runtime(
    baseline: GateQ3RuntimeMeasurement,
    candidate: GateQ3RuntimeMeasurement,
    *,
    maximum_cv_percent: float = 2.0,
    minimum_improvement_percent: float = 2.0,
) -> MappingScreenResult:
    """Screen a gate-only Q3 representation before expensive profiling/quality."""

    if baseline.q3_gate or not candidate.q3_gate:
        raise Q4RDNATuningError("Q3 runtime screen requires Q4 then Q3 gate arms")
    if maximum_cv_percent <= 0 or minimum_improvement_percent < 0:
        raise Q4RDNATuningError("screen thresholds are invalid")
    if not baseline.activation_verified or not candidate.activation_verified:
        return MappingScreenResult(
            baseline_arm_id=baseline.arm_id,
            candidate_arm_id=candidate.arm_id,
            outcome=MappingScreenOutcome.INCONCLUSIVE,
            improvement_percent=None,
            reasons=("Q4/Q3 sidecar activation was not verified in both arms",),
        )
    unstable = [
        item.arm_id
        for item in (baseline, candidate)
        if item.cv_percent > maximum_cv_percent
    ]
    if unstable:
        return MappingScreenResult(
            baseline_arm_id=baseline.arm_id,
            candidate_arm_id=candidate.arm_id,
            outcome=MappingScreenOutcome.INCONCLUSIVE,
            improvement_percent=None,
            reasons=("unstable benchmark arms: " + ", ".join(unstable),),
        )
    improvement = (
        candidate.mean_tokens_per_second / baseline.mean_tokens_per_second - 1.0
    ) * 100.0
    outcome = (
        MappingScreenOutcome.PROMOTE
        if improvement >= minimum_improvement_percent
        else MappingScreenOutcome.REJECT
    )
    return MappingScreenResult(
        baseline_arm_id=baseline.arm_id,
        candidate_arm_id=candidate.arm_id,
        outcome=outcome,
        improvement_percent=improvement,
        reasons=(
            "gate Q3 passed the real-model performance screen"
            if outcome == MappingScreenOutcome.PROMOTE
            else "gate Q3 did not meet the real-model performance threshold",
        ),
    )


__all__ = [
    "GateQ3RuntimeMeasurement",
    "GateUpBenchmarkMeasurement",
    "GateUpMappingArm",
    "GateUpPairRuntimeMeasurement",
    "GateUpPairFinalEvidence",
    "GateUpPairFinalPolicy",
    "GateUpPairFinalResult",
    "GateUpPerformanceModel",
    "GateUpShape",
    "MappingScreenOutcome",
    "MappingScreenResult",
    "Q4RDNAFormat",
    "Q4RDNATuningError",
    "build_gate_up_performance_model",
    "drop_initial_warmup_samples",
    "evaluate_gate_up_pair_final",
    "gate_up_mapping_arm",
    "normalize_llama_cli_generation",
    "screen_gate_up_mapping",
    "screen_gate_up_pair_load",
    "screen_gate_q3_microbenchmark",
    "screen_gate_q3_runtime",
    "screen_gate_up_pair_runtime",
]
