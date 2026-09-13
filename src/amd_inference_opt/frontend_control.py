"""Stable write boundary for the local experiment-builder UI.

The control plane never accepts shell commands, patches, or filesystem paths.  It
stores validated optimization intent in a small SQLite database.  A later workflow
step may materialize a draft into an OptimizationTask, but creating a draft never
executes a workload or mutates an ExperimentStore.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .eval_suites import (
    BALANCED_200_POLICY,
    PROVISIONAL_MATH_100_POLICY,
    QualityPolicyId,
    QualitySuiteId,
    quality_policy_suites,
)
from .frontend_api import ControlPlaneReader, FrontendReadError, MetricViewV1
from .models import OptimizationTask, StrictModel, utc_now


class MixedBitMode(StrEnum):
    DISABLED = "DISABLED"
    STOCK_SWEEP = "STOCK_SWEEP"
    SENSITIVITY_GUIDED = "SENSITIVITY_GUIDED"


class KernelExperimentMode(StrEnum):
    NO_CHANGE = "NO_CHANGE"
    EVIDENCE_THEN_TUNE = "EVIDENCE_THEN_TUNE"
    ALWAYS_CREATE_SHAPE_EXPERIMENT = "ALWAYS_CREATE_SHAPE_EXPERIMENT"


class TensorGroup(StrEnum):
    EMBEDDING = "embedding"
    OUTPUT = "output"
    ATTENTION_Q = "attention_q"
    ATTENTION_K = "attention_k"
    ATTENTION_V = "attention_v"
    ATTENTION_O = "attention_o"
    FFN_GATE = "ffn_gate"
    FFN_UP = "ffn_up"
    FFN_DOWN = "ffn_down"


class KernelShape(StrEnum):
    FFN_GATE_UP = "ffn_gate_up"
    ATTENTION_OUTPUT = "attention_output"
    SMALL_PROJECTION = "small_projection"
    VOCAB_OUTPUT = "vocab_output"


class KernelKnob(StrEnum):
    SPLIT_K = "split_k"
    WAVES_PER_BLOCK = "waves_per_block"
    ROWS_PER_WAVE = "rows_per_wave"
    VECTOR_LOAD = "vector_load"
    VGPR_LIMIT = "vgpr_limit"
    GATE_UP_FUSION = "gate_up_fusion"


class LlamaCppTechnique(StrEnum):
    MATRIX_SHAPE_WAVE_MAPPING = "matrix_shape_wave_mapping"
    WEIGHT_LAYOUT_FUSED_DEQUANT = "weight_layout_fused_dequant"
    GATE_UP_FUSION = "gate_up_fusion"
    HIP_GRAPH_AB = "hip_graph_ab"
    KV_CACHE_QUANTIZATION = "kv_cache_quantization"
    BUFFER_REUSE_AUDIT = "buffer_reuse_audit"


class ModelDraftReference(StrictModel):
    source_id: str
    run_id: str
    model_sha256: str | None = None

    @field_validator("source_id", "run_id")
    @classmethod
    def safe_identifier(cls, value: str) -> str:
        if not value or any(not (character.isalnum() or character in "-_") for character in value):
            raise ValueError(
                "model reference identifiers may contain only letters, digits, '-' and '_'"
            )
        return value


class TensorPrecisionAssignment(StrictModel):
    group: TensorGroup
    precision: Literal["Q4_K", "Q5_K", "Q6_K"]


def _default_assignments() -> list[TensorPrecisionAssignment]:
    return [
        TensorPrecisionAssignment(group=TensorGroup.EMBEDDING, precision="Q6_K"),
        TensorPrecisionAssignment(group=TensorGroup.OUTPUT, precision="Q6_K"),
        TensorPrecisionAssignment(group=TensorGroup.ATTENTION_Q, precision="Q6_K"),
        TensorPrecisionAssignment(group=TensorGroup.ATTENTION_K, precision="Q6_K"),
        TensorPrecisionAssignment(group=TensorGroup.ATTENTION_V, precision="Q6_K"),
        TensorPrecisionAssignment(group=TensorGroup.ATTENTION_O, precision="Q6_K"),
        TensorPrecisionAssignment(group=TensorGroup.FFN_GATE, precision="Q5_K"),
        TensorPrecisionAssignment(group=TensorGroup.FFN_UP, precision="Q5_K"),
        TensorPrecisionAssignment(group=TensorGroup.FFN_DOWN, precision="Q5_K"),
    ]


class MixedBitDraftV1(StrictModel):
    mode: MixedBitMode = MixedBitMode.SENSITIVITY_GUIDED
    source_precision: Literal["BF16"] = "BF16"
    sensitivity_evidence: Literal["REUSE_OR_COLLECT_IMATRIX", "REQUIRE_EXISTING"] = (
        "REUSE_OR_COLLECT_IMATRIX"
    )
    reference_candidates: list[Literal["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]] = Field(
        default_factory=lambda: ["Q6_K", "Q5_K_M", "Q4_K_M"]
    )
    assignments: list[TensorPrecisionAssignment] = Field(default_factory=_default_assignments)
    require_effective_bpw: Literal[True] = True
    require_pareto_ranking: Literal[True] = True

    @field_validator("reference_candidates")
    @classmethod
    def unique_candidates(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("reference candidates must be unique")
        return values

    @field_validator("assignments")
    @classmethod
    def unique_groups(
        cls, values: list[TensorPrecisionAssignment]
    ) -> list[TensorPrecisionAssignment]:
        groups = [value.group for value in values]
        if len(groups) != len(set(groups)):
            raise ValueError("each tensor group may have only one precision")
        return values

    @model_validator(mode="after")
    def mode_matches_assignments(self) -> MixedBitDraftV1:
        if self.mode == MixedBitMode.DISABLED and (
            self.reference_candidates or self.assignments
        ):
            raise ValueError("disabled mixed-bit workflow cannot define candidates or assignments")
        if self.mode == MixedBitMode.STOCK_SWEEP and self.assignments:
            raise ValueError("stock sweep cannot define tensor assignments")
        if self.mode == MixedBitMode.SENSITIVITY_GUIDED and not self.assignments:
            raise ValueError("sensitivity-guided workflow requires tensor assignments")
        return self


class KernelMappingDraftV1(StrictModel):
    mode: KernelExperimentMode = KernelExperimentMode.EVIDENCE_THEN_TUNE
    trigger: Literal["ONLY_IF_PROFILED_GAP", "ALWAYS"] = "ONLY_IF_PROFILED_GAP"
    shapes: list[KernelShape] = Field(
        default_factory=lambda: [
            KernelShape.FFN_GATE_UP,
            KernelShape.ATTENTION_OUTPUT,
            KernelShape.SMALL_PROJECTION,
            KernelShape.VOCAB_OUTPUT,
        ]
    )
    knobs: list[KernelKnob] = Field(
        default_factory=lambda: [
            KernelKnob.SPLIT_K,
            KernelKnob.WAVES_PER_BLOCK,
            KernelKnob.ROWS_PER_WAVE,
            KernelKnob.VECTOR_LOAD,
            KernelKnob.VGPR_LIMIT,
        ]
    )
    max_candidates: int = Field(default=4, ge=1, le=16)
    independent_clean_base: Literal[True] = True

    @field_validator("shapes", "knobs")
    @classmethod
    def unique_nonempty(cls, values: list[StrEnum]) -> list[StrEnum]:
        if not values or len(values) != len(set(values)):
            raise ValueError("kernel selections must be non-empty and unique")
        return values

    @model_validator(mode="after")
    def disabled_mode_is_empty(self) -> KernelMappingDraftV1:
        if self.mode == KernelExperimentMode.NO_CHANGE and (self.shapes or self.knobs):
            raise ValueError("NO_CHANGE kernel mode cannot select shapes or knobs")
        return self


class BenchmarkDraftV1(StrictModel):
    generation_lengths: list[Literal[128, 512]] = Field(default_factory=lambda: [128, 512])
    repetitions: int = Field(default=5, ge=3, le=12)
    max_cv_percent: float = Field(default=2.0, gt=0, le=10)

    @field_validator("generation_lengths")
    @classmethod
    def fixed_lengths(cls, values: list[int]) -> list[int]:
        if values != [128, 512]:
            raise ValueError("V1 benchmark coordinates are fixed to tg128 and tg512")
        return values


class QualityDraftV1(StrictModel):
    policy: QualityPolicyId = PROVISIONAL_MATH_100_POLICY
    math_problem_count: Literal[100] = 100
    general_problem_count: Literal[100] | None = None
    perplexity_enabled: Literal[True] = True
    greedy_canary_enabled: Literal[True] = True
    max_math_accuracy_drop_points: float = Field(default=2.0, ge=0, le=10)
    max_general_accuracy_drop_points: float = Field(default=2.0, ge=0, le=10)
    max_perplexity_regression_percent: float = Field(default=0.5, ge=0, le=10)

    @model_validator(mode="after")
    def composition_matches_policy(self) -> QualityDraftV1:
        if self.policy == BALANCED_200_POLICY and self.general_problem_count is None:
            object.__setattr__(self, "general_problem_count", 100)
        elif self.policy == PROVISIONAL_MATH_100_POLICY and (
            self.general_problem_count is not None
        ):
            raise ValueError("provisional math policy cannot include general questions")
        return self

    @property
    def suite_ids(self) -> tuple[QualitySuiteId, ...]:
        return quality_policy_suites(self.policy)


class LlamaCppOptimizationDraftV1(StrictModel):
    policy: Literal["EVIDENCE_FIRST"] = "EVIDENCE_FIRST"
    techniques: list[LlamaCppTechnique] = Field(
        default_factory=lambda: [
            LlamaCppTechnique.MATRIX_SHAPE_WAVE_MAPPING,
            LlamaCppTechnique.HIP_GRAPH_AB,
            LlamaCppTechnique.KV_CACHE_QUANTIZATION,
            LlamaCppTechnique.BUFFER_REUSE_AUDIT,
        ]
    )
    max_source_patches: int = Field(default=2, ge=0, le=4)

    @field_validator("techniques")
    @classmethod
    def unique_techniques(cls, values: list[LlamaCppTechnique]) -> list[LlamaCppTechnique]:
        if len(values) != len(set(values)):
            raise ValueError("llama.cpp techniques must be unique")
        return values


class OptimizationDraftRequestV1(StrictModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=100)
    model: ModelDraftReference
    mixed_bit: MixedBitDraftV1 = Field(default_factory=MixedBitDraftV1)
    kernel_mapping: KernelMappingDraftV1 = Field(default_factory=KernelMappingDraftV1)
    llama_cpp: LlamaCppOptimizationDraftV1 = Field(
        default_factory=LlamaCppOptimizationDraftV1
    )
    benchmark: BenchmarkDraftV1 = Field(default_factory=BenchmarkDraftV1)
    quality: QualityDraftV1 = Field(default_factory=QualityDraftV1)
    notes: str = Field(default="", max_length=1000)

    @field_validator("name", "notes")
    @classmethod
    def no_control_characters(cls, value: str) -> str:
        if any(ord(character) < 32 and character not in "\n\t" for character in value):
            raise ValueError("text contains control characters")
        return value.strip()


class OptimizationDraftRecordV1(StrictModel):
    schema_name: Literal["gpuopt.optimization-draft.v1"] = Field(
        default="gpuopt.optimization-draft.v1", alias="schema"
    )
    id: str
    status: Literal["DRAFT"] = "DRAFT"
    created_at: datetime
    content_sha256: str
    request: OptimizationDraftRequestV1


class OptimizationDraftListV1(StrictModel):
    schema_name: Literal["gpuopt.optimization-draft-list.v1"] = Field(
        default="gpuopt.optimization-draft-list.v1", alias="schema"
    )
    items: list[OptimizationDraftRecordV1]


class ModelRunMethodV1(StrictModel):
    source_id: str
    run_id: str
    model_sha256: str | None = None
    quantization: str | None = None
    gpu: str | None = None
    gfx: str | None = None
    status: str
    methods: list[str] = Field(default_factory=list)
    metrics: list[MetricViewV1] = Field(default_factory=list)


class OriginModelV1(StrictModel):
    role: Literal["ORIGIN"] = "ORIGIN"
    name: str
    source: str | None = None
    revision: str | None = None
    precision: str | None = None
    sha256: str | None = None


class VariantPerformanceV1(StrictModel):
    tg128: float | None = None
    tg512: float | None = None
    tg128_delta_percent: float | None = None
    tg512_delta_percent: float | None = None


class VariantQualityV1(StrictModel):
    policy: QualityPolicyId | None = None
    suite_ids: list[QualitySuiteId] = Field(default_factory=list)
    # Legacy aliases retained for existing frontend clients; these mirror math.
    accuracy_percent: float | None = None
    accuracy_delta_points: float | None = None
    correct: int | None = None
    total: int | None = None
    math_accuracy_percent: float | None = None
    math_accuracy_delta_points: float | None = None
    math_correct: int | None = None
    math_total: int | None = None
    general_accuracy_percent: float | None = None
    general_accuracy_delta_points: float | None = None
    general_correct: int | None = None
    general_total: int | None = None
    perplexity: float | None = None
    perplexity_delta_percent: float | None = None


class ModelVariantV1(StrictModel):
    id: str
    role: Literal["BASELINE", "CANDIDATE", "ACCEPTED"]
    label: str
    strategy: str
    quantization: str | None = None
    effective_bpw: float | None = None
    disposition: str
    selected_as_winner: bool = False
    source_id: str
    run_id: str
    performance: VariantPerformanceV1 = Field(default_factory=VariantPerformanceV1)
    quality: VariantQualityV1 = Field(default_factory=VariantQualityV1)
    reasons: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)


class ModelCatalogItemV1(StrictModel):
    id: str
    name: str
    architecture: str | None = None
    model_sha256: str | None = None
    origin: OriginModelV1
    quantizations: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    runs: list[ModelRunMethodV1] = Field(default_factory=list)
    variants: list[ModelVariantV1] = Field(default_factory=list)
    summary: str


class ModelCatalogV1(StrictModel):
    schema_name: Literal["gpuopt.model-catalog.v1"] = Field(
        default="gpuopt.model-catalog.v1", alias="schema"
    )
    generated_at: datetime = Field(default_factory=utc_now)
    items: list[ModelCatalogItemV1]


class DraftOptionsV1(StrictModel):
    schema_name: Literal["gpuopt.optimization-draft-options.v1"] = Field(
        default="gpuopt.optimization-draft-options.v1", alias="schema"
    )
    mixed_bit_modes: list[MixedBitMode] = Field(default_factory=lambda: list(MixedBitMode))
    reference_precisions: list[str] = Field(
        default_factory=lambda: ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
    )
    tensor_groups: list[TensorGroup] = Field(default_factory=lambda: list(TensorGroup))
    tensor_precisions: list[str] = Field(default_factory=lambda: ["Q4_K", "Q5_K", "Q6_K"])
    kernel_modes: list[KernelExperimentMode] = Field(
        default_factory=lambda: list(KernelExperimentMode)
    )
    kernel_shapes: list[KernelShape] = Field(default_factory=lambda: list(KernelShape))
    kernel_knobs: list[KernelKnob] = Field(default_factory=lambda: list(KernelKnob))
    llama_cpp_techniques: list[LlamaCppTechnique] = Field(
        default_factory=lambda: list(LlamaCppTechnique)
    )
    quality_policy: QualityPolicyId = PROVISIONAL_MATH_100_POLICY
    quality_policies: list[QualityPolicyId] = Field(
        default_factory=lambda: [PROVISIONAL_MATH_100_POLICY, BALANCED_200_POLICY]
    )
    execution_behavior: Literal["DRAFT_ONLY"] = "DRAFT_ONLY"


class ExperimentBuilderMetaV1(StrictModel):
    schema_name: Literal["gpuopt.experiment-builder-meta.v1"] = Field(
        default="gpuopt.experiment-builder-meta.v1", alias="schema"
    )
    draft_writes_enabled: bool
    experiment_execution_enabled: Literal[False] = False
    csrf_token: str | None = None


def _origin_identity(task: OptimizationTask) -> tuple[str, str | None, str | None, str]:
    metadata = getattr(task, "metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    hf_model = metadata.get("hf_model")
    source = metadata.get("q8_source_model_path") or metadata.get("source_model_path")
    if isinstance(hf_model, str) and hf_model:
        return hf_model, str(source) if source else hf_model, metadata.get("hf_revision"), hf_model
    if isinstance(source, str) and source:
        name = Path(source).name.removesuffix("-hf") or Path(source).name
        return name, source, None, source
    model = task.model
    name = Path(model.path).stem
    return name, str(model.path), None, model.sha256 or name


def _artifact_json(
    reader: ControlPlaneReader, source_id: str, run_id: str, path: str
) -> dict[str, object] | None:
    try:
        content = reader.source(source_id).preview(run_id, path).content
        value = json.loads(content)
    except (FrontendReadError, OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _candidate_performance(
    reader: ControlPlaneReader,
    source_id: str,
    run_id: str,
    artifacts: list[dict[str, object]],
) -> VariantPerformanceV1:
    result = VariantPerformanceV1()
    for artifact in artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or "performance" not in path:
            continue
        payload = _artifact_json(reader, source_id, run_id, path)
        if payload is None:
            continue
        improvements = payload.get("metric_improvements_percent", {})
        if isinstance(improvements, dict):
            for key, field in (("tg128", "tg128_delta_percent"), ("tg512", "tg512_delta_percent")):
                value = improvements.get(key)
                if isinstance(value, (int, float)):
                    result = result.model_copy(update={field: float(value)})
        checks = payload.get("checks", [])
        for check in checks if isinstance(checks, list) else []:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name", ""))
            if name not in {"performance.tg128", "performance.tg512"}:
                continue
            match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s+vs\s+", str(check.get("detail", "")))
            if match:
                result = result.model_copy(
                    update={"tg128" if name.endswith("tg128") else "tg512": float(match.group(1))}
                )
    return result


def _candidate_quality(
    reader: ControlPlaneReader,
    source_id: str,
    run_id: str,
    artifacts: list[dict[str, object]],
) -> VariantQualityV1:
    def counts_and_accuracy(
        value: dict[str, object], metric: Literal["math", "general"]
    ) -> tuple[float | None, int | None, int | None]:
        correct = value.get(f"{metric}_correct")
        total = value.get(f"{metric}_total")
        valid_counts = (
            isinstance(correct, int)
            and not isinstance(correct, bool)
            and isinstance(total, int)
            and not isinstance(total, bool)
            and total > 0
            and 0 <= correct <= total
        )
        accuracies = value.get("accuracies")
        raw_accuracy = (
            accuracies.get(f"{metric}_accuracy")
            if isinstance(accuracies, dict)
            else value.get(f"{metric}_accuracy")
        )
        accuracy = (
            float(raw_accuracy) * 100
            if isinstance(raw_accuracy, (int, float))
            and not isinstance(raw_accuracy, bool)
            and 0 <= float(raw_accuracy) <= 1
            else None
        )
        if accuracy is None and valid_counts:
            assert isinstance(correct, int) and isinstance(total, int)
            accuracy = correct / total * 100
        return (
            accuracy,
            correct if valid_counts and isinstance(correct, int) else None,
            total if valid_counts and isinstance(total, int) else None,
        )

    for artifact in artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or not any(
            marker in path
            for marker in (
                "quality-input",
                "quality-result",
                "provisional-quality",
                "balanced-quality",
            )
        ):
            continue
        payload = _artifact_json(reader, source_id, run_id, path)
        if payload is None:
            continue
        baseline = payload.get("baseline")
        candidate = payload.get("candidate")
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            continue
        math_accuracy, math_correct, math_total = counts_and_accuracy(candidate, "math")
        baseline_math, _, _ = counts_and_accuracy(baseline, "math")
        general_accuracy, general_correct, general_total = counts_and_accuracy(
            candidate, "general"
        )
        baseline_general, _, _ = counts_and_accuracy(baseline, "general")
        perplexity = candidate.get("perplexity")
        baseline_perplexity = baseline.get("perplexity")
        ppl_delta = (
            (float(perplexity) / float(baseline_perplexity) - 1) * 100
            if isinstance(perplexity, (int, float))
            and not isinstance(perplexity, bool)
            and isinstance(baseline_perplexity, (int, float))
            and not isinstance(baseline_perplexity, bool)
            and baseline_perplexity
            else None
        )
        raw_policy = payload.get("quality_policy", payload.get("policy"))
        policy: QualityPolicyId = (
            BALANCED_200_POLICY
            if raw_policy == BALANCED_200_POLICY or general_accuracy is not None
            else PROVISIONAL_MATH_100_POLICY
        )
        return VariantQualityV1(
            policy=policy,
            suite_ids=list(quality_policy_suites(policy)),
            accuracy_percent=math_accuracy,
            accuracy_delta_points=(
                math_accuracy - baseline_math
                if math_accuracy is not None and baseline_math is not None
                else None
            ),
            correct=math_correct,
            total=math_total,
            math_accuracy_percent=math_accuracy,
            math_accuracy_delta_points=(
                math_accuracy - baseline_math
                if math_accuracy is not None and baseline_math is not None
                else None
            ),
            math_correct=math_correct,
            math_total=math_total,
            general_accuracy_percent=general_accuracy,
            general_accuracy_delta_points=(
                general_accuracy - baseline_general
                if general_accuracy is not None and baseline_general is not None
                else None
            ),
            general_correct=general_correct,
            general_total=general_total,
            perplexity=(
                float(perplexity)
                if isinstance(perplexity, (int, float))
                and not isinstance(perplexity, bool)
                else None
            ),
            perplexity_delta_percent=ppl_delta,
        )
    return VariantQualityV1()


def _mixed_bit_variant(
    reader: ControlPlaneReader,
    source_id: str,
    run_id: str,
    candidate: dict[str, object],
    baseline_quantization: str | None,
) -> tuple[ModelVariantV1, dict[str, object] | None] | None:
    if candidate.get("strategy") != "mixed_bit":
        return None
    spec_ref = candidate.get("spec_artifact")
    spec_path = spec_ref.get("path") if isinstance(spec_ref, dict) else None
    spec = (
        _artifact_json(reader, source_id, run_id, spec_path)
        if isinstance(spec_path, str)
        else None
    )
    spec = spec or {}
    artifacts = [
        item for item in candidate.get("result_artifacts", []) if isinstance(item, dict)
    ]
    quantization = spec.get("base_quantization")
    quantization = str(quantization) if quantization else None
    selected = bool(candidate.get("selected_as_winner"))
    is_baseline = (
        quantization == baseline_quantization
        and not selected
        and candidate.get("id") in {"mixed-q8_0", "q8_0", "q8"}
    )
    role: Literal["BASELINE", "CANDIDATE", "ACCEPTED"] = (
        "ACCEPTED" if selected else "BASELINE" if is_baseline else "CANDIDATE"
    )
    evidence = [str(item["path"]) for item in artifacts if item.get("path")]
    if isinstance(spec_path, str):
        evidence.insert(0, spec_path)
    return (
        ModelVariantV1(
            id=str(candidate.get("id", "unknown")),
            role=role,
            label=str(candidate.get("label") or quantization or candidate.get("id")),
            strategy="mixed_bit",
            quantization=quantization,
            effective_bpw=(
                float(spec["effective_bpw"])
                if isinstance(spec.get("effective_bpw"), (int, float))
                else None
            ),
            disposition=str(candidate.get("disposition", "UNKNOWN")),
            selected_as_winner=selected,
            source_id=source_id,
            run_id=run_id,
            performance=_candidate_performance(
                reader, source_id, run_id, artifacts
            ),
            quality=_candidate_quality(reader, source_id, run_id, artifacts),
            reasons=[str(value) for value in candidate.get("reasons", [])],
            evidence_paths=evidence,
        ),
        spec,
    )


def build_model_catalog(reader: ControlPlaneReader) -> ModelCatalogV1:
    grouped: dict[str, dict[str, object]] = {}
    for summary in reader.runs().items:
        try:
            source = reader.source(summary.source_id)
            task = source.task(summary.run_id)
            detail = source.detail(summary.run_id)
        except (FrontendReadError, OSError, ValueError):
            continue
        origin_name, origin_source, origin_revision, family_key = _origin_identity(task)
        model_id = "model-" + hashlib.sha256(family_key.encode()).hexdigest()[:16]
        row = grouped.setdefault(
            model_id,
            {
                "name": origin_name,
                "architecture": summary.architecture,
                "model_sha256": summary.model_sha256,
                "origin": OriginModelV1(
                    name=origin_name,
                    source=origin_source,
                    revision=str(origin_revision) if origin_revision else None,
                    precision=(
                        "BF16"
                        if origin_source != str(task.model.path)
                        else task.model.quantization
                    ),
                ),
                "quantizations": set(),
                "methods": set(),
                "runs": [],
                "variants": {},
            },
        )
        quantizations = row["quantizations"]
        methods = row["methods"]
        runs = row["runs"]
        variants = row["variants"]
        assert isinstance(quantizations, set) and isinstance(methods, set)
        assert isinstance(runs, list) and isinstance(variants, dict)
        if summary.quantization:
            quantizations.add(summary.quantization)
        run_methods = {
            capability.id
            for capability in detail.capabilities
            if capability.status != "NOT_STARTED"
        }
        run_methods.update(
            str(item.get("strategy"))
            for item in detail.candidates
            if isinstance(item, dict) and item.get("strategy")
        )
        run_methods.update(
            item.strategy for item in detail.experiments if item.strategy is not None
        )
        methods.update(run_methods)
        runs.append(
            ModelRunMethodV1(
                source_id=summary.source_id,
                run_id=summary.run_id,
                model_sha256=summary.model_sha256,
                quantization=summary.quantization,
                gpu=summary.gpu,
                gfx=summary.gfx,
                status=summary.status,
                methods=sorted(run_methods),
                metrics=detail.metrics,
            )
        )
        baseline_key = f"baseline:{summary.quantization or summary.model_sha256}"
        variants.setdefault(
            baseline_key,
            ModelVariantV1(
                id=baseline_key.replace(":", "-"),
                role="BASELINE",
                label=f"{summary.quantization or 'runtime'} baseline",
                strategy="baseline",
                quantization=summary.quantization,
                disposition=summary.status,
                source_id=summary.source_id,
                run_id=summary.run_id,
            ),
        )
        for raw_candidate in detail.candidates:
            if not isinstance(raw_candidate, dict):
                continue
            projected = _mixed_bit_variant(
                reader,
                summary.source_id,
                summary.run_id,
                raw_candidate,
                summary.quantization,
            )
            if projected is None:
                continue
            variant, spec = projected
            key = (
                f"baseline:{variant.quantization}"
                if variant.role == "BASELINE"
                else f"mixed:{variant.id}"
            )
            current = variants.get(key)
            if current is None or variant.selected_as_winner or current.performance.tg128 is None:
                variants[key] = variant
            if variant.quantization:
                quantizations.add(variant.quantization)
            provenance = spec.get("provenance") if isinstance(spec, dict) else None
            source_model = provenance.get("source_model") if isinstance(provenance, dict) else None
            if isinstance(source_model, dict) and source_model.get("sha256"):
                origin = row["origin"]
                assert isinstance(origin, OriginModelV1)
                row["origin"] = origin.model_copy(
                    update={
                        "sha256": str(source_model["sha256"]),
                        "precision": str(provenance.get("source_quantization", "BF16")),
                    }
                )
    items = []
    for identifier, value in grouped.items():
        quantizations = sorted(value["quantizations"])  # type: ignore[arg-type]
        methods = sorted(value["methods"])  # type: ignore[arg-type]
        runs = value["runs"]
        variants_by_key = value["variants"]
        assert isinstance(runs, list) and isinstance(variants_by_key, dict)
        variants = sorted(
            variants_by_key.values(),
            key=lambda item: (
                {"ORIGIN": 0, "BASELINE": 1, "ACCEPTED": 2}.get(item.role, 3),
                item.label,
            ),
        )
        items.append(
            ModelCatalogItemV1(
                id=identifier,
                name=str(value["name"]),
                architecture=(str(value["architecture"]) if value["architecture"] else None),
                model_sha256=(str(value["model_sha256"]) if value["model_sha256"] else None),
                origin=value["origin"],  # type: ignore[arg-type]
                quantizations=quantizations,
                methods=methods,
                runs=runs,  # type: ignore[arg-type]
                variants=variants,
                summary=(
                    f"ORIGIN · {len(variants)} variants · {len(runs)} runs · "
                    f"{len(methods)} methods"
                ),
            )
        )
    items.sort(key=lambda item: (item.name.lower(), item.id))
    return ModelCatalogV1(items=items)


class DraftStore:
    """SQLite-backed immutable draft catalogue, separate from experiment evidence."""

    def __init__(self, database_path: str | Path) -> None:
        candidate = Path(database_path).expanduser()
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_file()):
            raise ValueError("draft database must be a regular non-symlink file")
        self.path = candidate.resolve(strict=False)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_drafts (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> OptimizationDraftRecordV1:
        request = OptimizationDraftRequestV1.model_validate_json(row["payload_json"])
        return OptimizationDraftRecordV1(
            id=row["id"],
            created_at=row["created_at"],
            content_sha256=row["content_sha256"],
            request=request,
        )

    def create(self, request: OptimizationDraftRequestV1) -> OptimizationDraftRecordV1:
        payload = request.model_dump_json()
        digest = hashlib.sha256(payload.encode()).hexdigest()
        record = OptimizationDraftRecordV1(
            id=f"draft-{uuid.uuid4().hex[:12]}",
            created_at=utc_now(),
            content_sha256=digest,
            request=request,
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO optimization_drafts VALUES (?, ?, ?, ?)",
                (record.id, record.created_at.isoformat(), digest, payload),
            )
        return record

    def list(self) -> OptimizationDraftListV1:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM optimization_drafts ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return OptimizationDraftListV1(items=[self._record(row) for row in rows])

    def get(self, draft_id: str) -> OptimizationDraftRecordV1:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM optimization_drafts WHERE id = ?", (draft_id,)
            ).fetchone()
        if row is None:
            raise KeyError("draft not found")
        return self._record(row)


def validate_draft_model(
    reader: ControlPlaneReader, request: OptimizationDraftRequestV1
) -> None:
    summary = reader.source(request.model.source_id).summary(request.model.run_id)
    if (
        request.model.model_sha256 is not None
        and summary.model_sha256 != request.model.model_sha256
    ):
        raise ValueError("model hash no longer matches the selected catalog run")


def frontend_control_schema() -> dict[str, object]:
    models = (
        ExperimentBuilderMetaV1,
        DraftOptionsV1,
        ModelCatalogV1,
        OptimizationDraftRequestV1,
        OptimizationDraftRecordV1,
        OptimizationDraftListV1,
    )
    return {
        "schema": "gpuopt.experiment-builder-schema.v1",
        "api_version": "v1",
        "execution_behavior": "DRAFT_ONLY",
        "models": {model.__name__: model.model_json_schema() for model in models},
    }


__all__ = [
    "BenchmarkDraftV1",
    "DraftOptionsV1",
    "DraftStore",
    "ExperimentBuilderMetaV1",
    "KernelExperimentMode",
    "KernelMappingDraftV1",
    "MixedBitDraftV1",
    "MixedBitMode",
    "ModelCatalogV1",
    "OptimizationDraftListV1",
    "OptimizationDraftRecordV1",
    "OptimizationDraftRequestV1",
    "QualityDraftV1",
    "TensorPrecisionAssignment",
    "build_model_catalog",
    "frontend_control_schema",
    "validate_draft_model",
]
