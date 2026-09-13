"""Strict same-binary HIP Graph ON/OFF A/B protocol and deterministic gate."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .command import CommandRunner
from .llama_cpp import parse_llama_bench_json, sha256_file
from .models import ArtifactRef, StrictModel
from .store import ExperimentStore

GRAPH_DISABLE_ENV = "GGML_CUDA_DISABLE_GRAPHS"
REQUIRED_METRICS = ("pp512", "tg128", "tg512")


class HipGraphABOutcome(StrEnum):
    MATERIAL_BENEFIT = "MATERIAL_BENEFIT"
    NO_MATERIAL_EFFECT = "NO_MATERIAL_EFFECT"
    HARMFUL = "HARMFUL"
    INCONCLUSIVE = "INCONCLUSIVE"


class HipGraphABSpec(StrictModel):
    schema_name: Literal["gpuopt.hip-graph-ab-spec.v1"] = Field(
        default="gpuopt.hip-graph-ab-spec.v1", alias="schema"
    )
    binary_sha256: str
    model_sha256: str
    benchmark_protocol_hash: str
    build_flags: list[str]
    common_environment: dict[str, str] = Field(default_factory=dict)
    on_environment: dict[str, str] = Field(default_factory=dict)
    off_environment: dict[str, str] = Field(
        default_factory=lambda: {GRAPH_DISABLE_ENV: "1"}
    )
    initial_paired_samples: int = Field(default=7, ge=7)
    maximum_paired_samples: int = Field(default=12, ge=7)
    max_cv_percent: float = Field(default=2.0, gt=0)
    material_delta_percent: float = Field(default=2.0, gt=0)
    maximum_regression_percent: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def exact_toggle(self) -> HipGraphABSpec:
        if not any(flag == "-DGGML_HIP_GRAPHS=ON" for flag in self.build_flags):
            raise ValueError("HIP Graph A/B binary must be compiled with GGML_HIP_GRAPHS=ON")
        if GRAPH_DISABLE_ENV in self.common_environment:
            raise ValueError("common environment cannot contain the graph toggle")
        if GRAPH_DISABLE_ENV in self.on_environment:
            raise ValueError("ON arm must leave GGML_CUDA_DISABLE_GRAPHS unset")
        if self.off_environment != {GRAPH_DISABLE_ENV: "1"}:
            raise ValueError("OFF arm may differ only by GGML_CUDA_DISABLE_GRAPHS=1")
        if self.maximum_paired_samples < self.initial_paired_samples:
            raise ValueError("maximum paired samples cannot be below initial samples")
        return self

    @property
    def coordinate_hash(self) -> str:
        value = self.model_dump(mode="json", exclude={"on_environment", "off_environment"})
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class PairedMetricSamples(StrictModel):
    unit: str
    graph_on: list[float] = Field(min_length=7)
    graph_off: list[float] = Field(min_length=7)

    @model_validator(mode="after")
    def paired_and_finite(self) -> PairedMetricSamples:
        if len(self.graph_on) != len(self.graph_off):
            raise ValueError("HIP Graph samples must be paired")
        if any(
            not math.isfinite(value) or value <= 0
            for value in [*self.graph_on, *self.graph_off]
        ):
            raise ValueError("HIP Graph samples must be positive and finite")
        return self

    @property
    def count(self) -> int:
        return len(self.graph_on)

    @staticmethod
    def _cv(values: list[float]) -> float:
        return statistics.stdev(values) / abs(statistics.fmean(values)) * 100

    @property
    def max_cv_percent(self) -> float:
        return max(self._cv(self.graph_on), self._cv(self.graph_off))

    @property
    def deltas_percent(self) -> list[float]:
        return [
            (on - off) / off * 100
            for on, off in zip(self.graph_on, self.graph_off, strict=True)
        ]

    @property
    def mean_delta_percent(self) -> float:
        return statistics.fmean(self.deltas_percent)

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        values = self.deltas_percent
        mean = statistics.fmean(values)
        t_critical = {
            7: 2.447,
            8: 2.365,
            9: 2.306,
            10: 2.262,
            11: 2.228,
            12: 2.201,
        }.get(len(values), 1.96)
        margin = t_critical * statistics.stdev(values) / math.sqrt(len(values))
        return mean - margin, mean + margin


class HipGraphTraceEvidence(StrictModel):
    graph_on_kernel_dispatches: int | None = Field(default=None, ge=0)
    graph_off_kernel_dispatches: int | None = Field(default=None, ge=0)
    graph_on_hip_api_calls: int | None = Field(default=None, ge=0)
    graph_off_hip_api_calls: int | None = Field(default=None, ge=0)
    graph_on_graph_launches: int | None = Field(default=None, ge=0)
    graph_off_graph_launches: int | None = Field(default=None, ge=0)
    graph_on_cpu_launch_gap_ns: float | None = Field(default=None, ge=0)
    graph_off_cpu_launch_gap_ns: float | None = Field(default=None, ge=0)
    graph_on_gpu_idle_gap_ns: float | None = Field(default=None, ge=0)
    graph_off_gpu_idle_gap_ns: float | None = Field(default=None, ge=0)
    missing_evidence: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class HipGraphMetricDecision(StrictModel):
    metric: str
    sample_count: int
    graph_on_mean: float
    graph_off_mean: float
    delta_percent: float
    ci95_low: float
    ci95_high: float
    max_cv_percent: float
    graph_on_latency_ms: float
    graph_off_latency_ms: float


class HipGraphABDecision(StrictModel):
    schema_name: Literal["gpuopt.hip-graph-ab-decision.v1"] = Field(
        default="gpuopt.hip-graph-ab-decision.v1", alias="schema"
    )
    outcome: HipGraphABOutcome
    metrics: list[HipGraphMetricDecision]
    trace: HipGraphTraceEvidence | None = None
    reasons: list[str]
    evidence: list[ArtifactRef] = Field(min_length=1)


def evaluate_hip_graph_ab(
    spec: HipGraphABSpec,
    metrics: dict[str, PairedMetricSamples],
    *,
    evidence: list[ArtifactRef],
    trace: HipGraphTraceEvidence | None = None,
) -> HipGraphABDecision:
    if set(metrics) != set(REQUIRED_METRICS):
        raise ValueError("HIP Graph A/B requires exactly pp512, tg128, and tg512")
    decisions: list[HipGraphMetricDecision] = []
    for name in REQUIRED_METRICS:
        samples = metrics[name]
        if samples.count > spec.maximum_paired_samples:
            raise ValueError("HIP Graph sample count exceeds configured maximum")
        low, high = samples.confidence_interval_95
        decisions.append(
            HipGraphMetricDecision(
                metric=name,
                sample_count=samples.count,
                graph_on_mean=statistics.fmean(samples.graph_on),
                graph_off_mean=statistics.fmean(samples.graph_off),
                delta_percent=samples.mean_delta_percent,
                ci95_low=low,
                ci95_high=high,
                max_cv_percent=samples.max_cv_percent,
                graph_on_latency_ms=(
                    (512 if name in {"pp512", "tg512"} else 128)
                    / statistics.fmean(samples.graph_on)
                    * 1000
                ),
                graph_off_latency_ms=(
                    (512 if name in {"pp512", "tg512"} else 128)
                    / statistics.fmean(samples.graph_off)
                    * 1000
                ),
            )
        )
    by_name = {decision.metric: decision for decision in decisions}
    unstable = [
        decision.metric
        for decision in decisions
        if decision.max_cv_percent > spec.max_cv_percent
    ]
    if unstable:
        outcome = HipGraphABOutcome.INCONCLUSIVE
        reasons = ["unstable metrics: " + ", ".join(unstable)]
    else:
        decode_benefit = all(
            by_name[name].delta_percent >= spec.material_delta_percent
            and by_name[name].ci95_low > 0
            for name in ("tg128", "tg512")
        ) and by_name["pp512"].delta_percent >= -spec.maximum_regression_percent
        prefill_benefit = (
            by_name["pp512"].delta_percent >= spec.material_delta_percent
            and by_name["pp512"].ci95_low > 0
            and all(
                by_name[name].delta_percent >= -spec.maximum_regression_percent
                for name in ("tg128", "tg512")
            )
        )
        harmful = any(
            decision.ci95_high < -spec.maximum_regression_percent
            for decision in decisions
        )
        bounded_no_effect = all(
            decision.ci95_low >= -spec.material_delta_percent
            and decision.ci95_high <= spec.material_delta_percent
            for decision in decisions
        )
        if decode_benefit or prefill_benefit:
            outcome = HipGraphABOutcome.MATERIAL_BENEFIT
            reasons = ["Graph ON produced a stable material workload improvement"]
        elif harmful:
            outcome = HipGraphABOutcome.HARMFUL
            reasons = ["Graph ON produced a statistically supported regression"]
        elif bounded_no_effect:
            outcome = HipGraphABOutcome.NO_MATERIAL_EFFECT
            reasons = ["all stable effects are bounded within the materiality band"]
        else:
            outcome = HipGraphABOutcome.INCONCLUSIVE
            reasons = ["paired confidence intervals do not support a material conclusion"]
    return HipGraphABDecision(
        outcome=outcome,
        metrics=decisions,
        trace=trace,
        reasons=reasons,
        evidence=evidence,
    )


class HipGraphRunCoordinates(StrictModel):
    """Exact benchmark coordinate used for every paired ON/OFF command."""

    binary_path: Path
    model_path: Path
    cwd: Path
    prompt_tokens: Literal[512] = 512
    generation_tokens: list[Literal[128, 512]] = Field(
        default_factory=lambda: [128, 512]
    )
    batch_size: int = Field(default=2048, gt=0)
    ubatch_size: int = Field(default=512, gt=0)
    threads: int = Field(default=12, gt=0)
    gpu_layers: int = Field(default=999, ge=0)
    device_selector: str = "ROCm0"
    same_coordinate_warmup_samples: Literal[5] = 5
    common_unset_environment: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=600, gt=0)

    @model_validator(mode="after")
    def exact_coordinate(self) -> HipGraphRunCoordinates:
        if any(
            not path.is_absolute()
            for path in (self.binary_path, self.model_path, self.cwd)
        ):
            raise ValueError("HIP Graph executable, model, and cwd paths must be absolute")
        if self.generation_tokens != [128, 512]:
            raise ValueError("HIP Graph A/B generation tokens must be [128, 512]")
        if self.ubatch_size > self.batch_size:
            raise ValueError("ubatch_size cannot exceed batch_size")
        return self

    @property
    def protocol_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={
                "binary_path",
                "model_path",
                "cwd",
                "common_unset_environment",
                "timeout_seconds",
            },
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def argv(self) -> list[str]:
        return [
            str(self.binary_path.resolve()),
            "-m",
            str(self.model_path.resolve()),
            "-p",
            "512",
            "-n",
            "128,512",
            "-b",
            str(self.batch_size),
            "-ub",
            str(self.ubatch_size),
            "-t",
            str(self.threads),
            "-r",
            str(self.same_coordinate_warmup_samples + 1),
            "-ngl",
            str(self.gpu_layers),
            "-mg",
            "0",
            "-dev",
            self.device_selector,
            "-o",
            "json",
            "-oe",
            "none",
        ]


class HipGraphABExecutionError(RuntimeError):
    """A paired command failed after its evidence was preserved."""


def run_hip_graph_ab(
    spec: HipGraphABSpec,
    coordinates: HipGraphRunCoordinates,
    *,
    task_id: str,
    store: ExperimentStore,
    runner: CommandRunner | None = None,
    trace: HipGraphTraceEvidence | None = None,
) -> tuple[HipGraphABDecision, ArtifactRef]:
    """Run alternating, same-binary ON/OFF pairs and persist every command envelope."""

    binary = coordinates.binary_path
    model = coordinates.model_path
    if binary.is_symlink() or not binary.is_file() or not binary.stat().st_mode & 0o111:
        raise HipGraphABExecutionError("llama-bench must be an executable regular file")
    if model.is_symlink() or not model.is_file():
        raise HipGraphABExecutionError("model must be a regular file")
    binary = binary.resolve(strict=True)
    model = model.resolve(strict=True)
    if sha256_file(binary) != spec.binary_sha256:
        raise HipGraphABExecutionError("llama-bench SHA-256 differs from the A/B spec")
    if sha256_file(model) != spec.model_sha256:
        raise HipGraphABExecutionError("model SHA-256 differs from the A/B spec")
    if coordinates.protocol_hash != spec.benchmark_protocol_hash:
        raise HipGraphABExecutionError("benchmark protocol hash differs from the A/B spec")

    command_runner = runner or CommandRunner()
    collected: dict[str, dict[str, list[float]]] = {
        metric: {"on": [], "off": []} for metric in REQUIRED_METRICS
    }
    evidence: list[ArtifactRef] = []
    for pair_index in range(spec.initial_paired_samples):
        arms = ("off", "on") if pair_index % 2 == 0 else ("on", "off")
        for arm in arms:
            environment = dict(spec.common_environment)
            environment.update(spec.on_environment if arm == "on" else spec.off_environment)
            unset = set(coordinates.common_unset_environment)
            if arm == "on":
                unset.add(GRAPH_DISABLE_ENV)
            else:
                unset.discard(GRAPH_DISABLE_ENV)
            result = command_runner.run(
                coordinates.argv,
                cwd=coordinates.cwd,
                env=environment,
                unset_env=sorted(unset),
                timeout_seconds=coordinates.timeout_seconds,
            )
            reference = store.save_evidence_json(
                task_id,
                f"gfx1201/hip-graph-ab/pair-{pair_index + 1:02d}-{arm}",
                {
                    "schema": "gpuopt.hip-graph-command.v1",
                    "pair_index": pair_index,
                    "arm": arm,
                    "coordinate_hash": spec.coordinate_hash,
                    "protocol_hash": coordinates.protocol_hash,
                    "command": result.to_dict(),
                },
                producer="hip-graph-ab-runner",
            )
            evidence.append(reference)
            if not result.succeeded:
                raise HipGraphABExecutionError(
                    f"HIP Graph {arm} pair {pair_index + 1} failed; evidence={reference.path}"
                )
            parsed = parse_llama_bench_json(result.stdout)
            records = parsed.by_test_id()
            if set(records) != set(REQUIRED_METRICS) or len(parsed.records) != 3:
                raise HipGraphABExecutionError(
                    f"HIP Graph {arm} output did not contain exactly pp512/tg128/tg512"
                )
            for metric in REQUIRED_METRICS:
                samples = records[metric].samples_tokens_per_second
                expected = coordinates.same_coordinate_warmup_samples + 1
                if len(samples) != expected:
                    raise HipGraphABExecutionError(
                        f"HIP Graph {arm} {metric} emitted {len(samples)} samples; "
                        f"expected {expected}"
                    )
                collected[metric][arm].append(samples[-1])

    metrics = {
        name: PairedMetricSamples(
            unit="tokens/s",
            graph_on=values["on"],
            graph_off=values["off"],
        )
        for name, values in collected.items()
    }
    decision = evaluate_hip_graph_ab(spec, metrics, evidence=evidence, trace=trace)
    decision_ref = store.save_evidence_json(
        task_id,
        "gfx1201/hip-graph-ab/decision",
        decision,
        producer="hip-graph-ab-gate",
    )
    return decision, decision_ref


__all__ = [
    "GRAPH_DISABLE_ENV",
    "HipGraphABDecision",
    "HipGraphABOutcome",
    "HipGraphABExecutionError",
    "HipGraphABSpec",
    "HipGraphRunCoordinates",
    "HipGraphMetricDecision",
    "HipGraphTraceEvidence",
    "PairedMetricSamples",
    "evaluate_hip_graph_ab",
    "run_hip_graph_ab",
]
