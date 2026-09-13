"""Configuration-driven mixed-precision planning, accounting, and Pareto ranking."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, field_validator, model_validator

from .models import ArtifactRef, StrictModel, utc_now


class Precision(StrEnum):
    Q4_K = "Q4_K"
    Q5_K = "Q5_K"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    Q4_RDNA = "Q4_RDNA"
    F16 = "F16"
    BF16 = "BF16"
    F32 = "F32"
    OTHER = "OTHER"


QUANTIZED_PRECISIONS = {
    Precision.Q4_K,
    Precision.Q5_K,
    Precision.Q6_K,
    Precision.Q8_0,
    Precision.Q4_RDNA,
}


class TensorNamingRule(StrictModel):
    group: str
    pattern: str
    protected: bool = False

    @field_validator("group", "pattern")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tensor naming fields cannot be empty")
        return value

    @field_validator("pattern")
    @classmethod
    def valid_regex(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError(f"invalid tensor regex: {error}") from error
        if not value.startswith("^") or not value.endswith("$"):
            raise ValueError("tensor naming patterns must be anchored")
        return value


class TensorNamingConfig(StrictModel):
    rules: list[TensorNamingRule] = Field(min_length=1)
    default_group: str = "other"

    @model_validator(mode="after")
    def unique_groups_and_patterns(self) -> TensorNamingConfig:
        patterns = [rule.pattern for rule in self.rules]
        if len(patterns) != len(set(patterns)):
            raise ValueError("tensor naming patterns must be unique")
        return self

    def classify(self, tensor_name: str) -> tuple[str, bool]:
        matched = [rule for rule in self.rules if re.fullmatch(rule.pattern, tensor_name)]
        if len(matched) > 1:
            raise ValueError(f"tensor matches multiple naming rules: {tensor_name}")
        if not matched:
            return self.default_group, False
        return matched[0].group, matched[0].protected


class TensorInventoryEntry(StrictModel):
    name: str
    shape: list[int] = Field(min_length=1)
    elements: int = Field(gt=0)
    storage_bytes: int = Field(ge=0)
    precision: Precision
    operator_group: str
    protected: bool = False
    quantizable: bool = True

    @model_validator(mode="after")
    def shape_matches_elements(self) -> TensorInventoryEntry:
        if math.prod(self.shape) != self.elements:
            raise ValueError("tensor shape does not match element count")
        return self


class TensorInventory(StrictModel):
    schema_name: Literal["gpuopt.tensor-inventory.v1"] = Field(
        default="gpuopt.tensor-inventory.v1", alias="schema"
    )
    model_sha256: str
    entries: list[TensorInventoryEntry] = Field(min_length=1)
    source_artifact: ArtifactRef

    @model_validator(mode="after")
    def unique_tensor_names(self) -> TensorInventory:
        names = [entry.name for entry in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("tensor inventory contains duplicate names")
        return self


class TensorSensitivityScore(StrictModel):
    tensor_name: str
    score: float = Field(ge=0)
    source_metric: str
    confidence: float = Field(default=1.0, ge=0, le=1)

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("sensitivity score must be finite")
        return value


class SensitivityEvidence(StrictModel):
    schema_name: Literal["gpuopt.sensitivity-evidence.v1"] = Field(
        default="gpuopt.sensitivity-evidence.v1", alias="schema"
    )
    model_sha256: str
    scores: list[TensorSensitivityScore] = Field(min_length=1)
    calibration_artifacts: list[ArtifactRef] = Field(min_length=1)
    quality_ablation_artifacts: list[ArtifactRef] = Field(default_factory=list)
    missing_tensors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_scores(self) -> SensitivityEvidence:
        names = [score.tensor_name for score in self.scores]
        if len(names) != len(set(names)):
            raise ValueError("sensitivity evidence contains duplicate tensor scores")
        return self


class PrecisionSearchPoint(StrictModel):
    id: str
    eligible_groups: list[str] = Field(min_length=1)
    q4_fraction: float = Field(ge=0, le=1)
    q5_fraction: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def fractions_fit(self) -> PrecisionSearchPoint:
        if self.q4_fraction + self.q5_fraction > 1:
            raise ValueError("q4_fraction + q5_fraction cannot exceed one")
        return self


class PrecisionSearchSpace(StrictModel):
    whole_model_precisions: list[Precision] = Field(
        default_factory=lambda: [
            Precision.Q8_0,
            Precision.Q6_K,
            Precision.Q5_K,
            Precision.Q4_K,
        ]
    )
    protected_group_precision: Precision = Precision.Q6_K
    search_points: list[PrecisionSearchPoint] = Field(default_factory=list)
    max_candidates: int = Field(default=8, ge=1, le=16)

    @field_validator("whole_model_precisions")
    @classmethod
    def supported_whole_precisions(cls, values: list[Precision]) -> list[Precision]:
        if len(values) != len(set(values)):
            raise ValueError("whole_model_precisions must be unique")
        if any(value not in QUANTIZED_PRECISIONS - {Precision.Q4_RDNA} for value in values):
            raise ValueError("whole-model precision must be a standard llama.cpp quant")
        return values


class PrecisionAssignmentEntry(StrictModel):
    tensor_name: str
    operator_group: str
    precision: Precision
    sensitivity_score: float | None = None
    reason: str


class PrecisionPolicy(StrictModel):
    schema_name: Literal["gpuopt.precision-policy.v1"] = Field(
        default="gpuopt.precision-policy.v1", alias="schema"
    )
    id: str
    provider: Literal["llama_cpp", "q4_rdna"]
    base_precision: Precision
    quantizer_format: str | None = None
    assignments: list[PrecisionAssignmentEntry] = Field(min_length=1)
    assignment_sha256: str
    source_evidence: list[ArtifactRef] = Field(default_factory=list)

    @field_validator("assignment_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("assignment_sha256 must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def assignment_hash_matches(self) -> PrecisionPolicy:
        if self.assignment_sha256 != assignment_hash(self.assignments):
            raise ValueError("precision assignment hash mismatch")
        names = [entry.tensor_name for entry in self.assignments]
        if len(names) != len(set(names)):
            raise ValueError("precision assignment contains duplicate tensors")
        if self.provider == "llama_cpp" and not self.quantizer_format:
            raise ValueError("llama_cpp precision policy requires quantizer_format")
        if self.provider == "q4_rdna" and self.quantizer_format is not None:
            raise ValueError("q4_rdna policy is not materialized by llama-quantize")
        return self


class PrecisionRouteRule(StrictModel):
    pattern: str
    precision: Precision
    reason: str

    @field_validator("pattern")
    @classmethod
    def anchored_pattern(cls, value: str) -> str:
        if not value.startswith("^") or not value.endswith("$"):
            raise ValueError("precision route regex must be anchored")
        re.compile(value)
        return value


def build_external_policy(
    policy_id: str,
    inventory: TensorInventory,
    *,
    provider: Literal["llama_cpp", "q4_rdna"],
    base_precision: Precision,
    routes: list[PrecisionRouteRule],
    evidence: list[ArtifactRef],
) -> PrecisionPolicy:
    """Build an exact assignment for a custom provider such as Q4_RDNA hybrid."""

    assignments: list[PrecisionAssignmentEntry] = []
    for entry in inventory.entries:
        matches = [route for route in routes if re.fullmatch(route.pattern, entry.name)]
        if len(matches) > 1:
            raise ValueError(f"external routes overlap for tensor {entry.name}")
        if not entry.quantizable:
            precision = entry.precision
            reason = "non-quantizable tensor preserved"
        elif matches:
            precision = matches[0].precision
            reason = matches[0].reason
        else:
            precision = base_precision
            reason = "external provider base representation"
        assignments.append(
            PrecisionAssignmentEntry(
                tensor_name=entry.name,
                operator_group=entry.operator_group,
                precision=precision,
                reason=reason,
            )
        )
    return PrecisionPolicy(
        id=policy_id,
        provider=provider,
        base_precision=base_precision,
        quantizer_format=None if provider == "q4_rdna" else base_precision.value,
        assignments=assignments,
        assignment_sha256=assignment_hash(assignments),
        source_evidence=evidence,
    )


def assignment_hash(assignments: list[PrecisionAssignmentEntry]) -> str:
    value = [
        entry.model_dump(mode="json")
        for entry in sorted(assignments, key=lambda item: item.tensor_name)
    ]
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@runtime_checkable
class PrecisionPolicyProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def policies(
        self,
        inventory: TensorInventory,
        sensitivity: SensitivityEvidence,
    ) -> list[PrecisionPolicy]: ...


def _assignment(
    entry: TensorInventoryEntry,
    precision: Precision,
    scores: dict[str, float],
    reason: str,
) -> PrecisionAssignmentEntry:
    return PrecisionAssignmentEntry(
        tensor_name=entry.name,
        operator_group=entry.operator_group,
        precision=precision,
        sensitivity_score=scores.get(entry.name),
        reason=reason,
    )


def plan_precision_policies(
    inventory: TensorInventory,
    sensitivity: SensitivityEvidence,
    search: PrecisionSearchSpace,
    *,
    external_policies: list[PrecisionPolicy] | None = None,
) -> list[PrecisionPolicy]:
    """Generate a deterministic, bounded candidate set from sensitivity evidence."""

    if inventory.model_sha256 != sensitivity.model_sha256:
        raise ValueError("inventory and sensitivity model hashes differ")
    scores = {score.tensor_name: score.score for score in sensitivity.scores}
    policies: list[PrecisionPolicy] = []
    for precision in search.whole_model_precisions:
        assignments = [
            _assignment(
                entry,
                (
                    entry.precision
                    if not entry.quantizable
                    else search.protected_group_precision
                    if entry.protected
                    else precision
                ),
                scores,
                (
                    "non-quantizable tensor preserved"
                    if not entry.quantizable
                    else "protected group"
                    if entry.protected
                    else "whole-model candidate"
                ),
            )
            for entry in inventory.entries
        ]
        policies.append(
            PrecisionPolicy(
                id=precision.value.lower(),
                provider="llama_cpp",
                base_precision=precision,
                quantizer_format={
                    Precision.Q4_K: "Q4_K_M",
                    Precision.Q5_K: "Q5_K_M",
                }.get(precision, precision.value),
                assignments=assignments,
                assignment_sha256=assignment_hash(assignments),
                source_evidence=sensitivity.calibration_artifacts,
            )
        )

    for point in search.search_points:
        eligible = [
            entry
            for entry in inventory.entries
            if entry.quantizable
            and entry.operator_group in point.eligible_groups
            and not entry.protected
        ]
        missing = [entry.name for entry in eligible if entry.name not in scores]
        if missing:
            raise ValueError(
                f"search point {point.id} lacks sensitivity for: " + ", ".join(missing)
            )
        eligible.sort(key=lambda entry: (scores[entry.name], entry.name))
        q4_count = round(len(eligible) * point.q4_fraction)
        q5_count = round(len(eligible) * point.q5_fraction)
        q4_names = {entry.name for entry in eligible[:q4_count]}
        q5_names = {entry.name for entry in eligible[q4_count : q4_count + q5_count]}
        assignments: list[PrecisionAssignmentEntry] = []
        for entry in inventory.entries:
            if not entry.quantizable:
                precision = entry.precision
                reason = "non-quantizable tensor preserved"
            elif entry.protected:
                precision = search.protected_group_precision
                reason = "protected group"
            elif entry.name in q4_names:
                precision = Precision.Q4_K
                reason = f"bottom sensitivity tier in {point.id}"
            elif entry.name in q5_names:
                precision = Precision.Q5_K
                reason = f"middle sensitivity tier in {point.id}"
            else:
                precision = Precision.Q6_K
                reason = f"high sensitivity or ineligible in {point.id}"
            assignments.append(_assignment(entry, precision, scores, reason))
        policies.append(
            PrecisionPolicy(
                id=point.id,
                provider="llama_cpp",
                base_precision=Precision.Q6_K,
                quantizer_format="Q6_K",
                assignments=assignments,
                assignment_sha256=assignment_hash(assignments),
                source_evidence=sensitivity.calibration_artifacts,
            )
        )

    policies.extend(external_policies or [])
    unique: list[PrecisionPolicy] = []
    seen: set[str] = set()
    for policy in policies:
        if policy.assignment_sha256 not in seen:
            unique.append(policy)
            seen.add(policy.assignment_sha256)
    if len(unique) > search.max_candidates:
        unique = unique[: search.max_candidates]
    return unique


def verify_materialized_assignment(
    policy: PrecisionPolicy,
    packed_inventory: TensorInventory,
) -> list[str]:
    expected = {entry.tensor_name: entry.precision for entry in policy.assignments}
    actual = {entry.name: entry.precision for entry in packed_inventory.entries}
    problems = [
        f"{name}: expected {precision}, observed {actual.get(name)}"
        for name, precision in expected.items()
        if actual.get(name) != precision
    ]
    problems.extend(f"unexpected tensor: {name}" for name in sorted(set(actual) - set(expected)))
    return problems


class PackedModelAccounting(StrictModel):
    schema_name: Literal["gpuopt.packed-model-accounting.v1"] = Field(
        default="gpuopt.packed-model-accounting.v1", alias="schema"
    )
    total_elements: int = Field(gt=0)
    eligible_linear_elements: int = Field(gt=0)
    logical_packed_weight_bytes: int = Field(ge=0)
    linear_packed_weight_bytes: int = Field(ge=0)
    container_bytes: int = Field(ge=0)
    auxiliary_sidecar_bytes: int = Field(default=0, ge=0)
    measured_vram_bytes: int | None = Field(default=None, ge=0)
    effective_bpw: float = Field(gt=0)
    linear_effective_bpw: float = Field(gt=0)
    precision_elements: dict[Precision, int]


def account_packed_model(
    packed_inventory: TensorInventory,
    *,
    linear_groups: set[str],
    container_bytes: int,
    auxiliary_sidecar_bytes: int = 0,
    measured_vram_bytes: int | None = None,
) -> PackedModelAccounting:
    unknown = [entry.name for entry in packed_inventory.entries if entry.storage_bytes == 0]
    if unknown:
        raise ValueError("packed inventory has unknown storage for: " + ", ".join(unknown))
    total_elements = sum(entry.elements for entry in packed_inventory.entries)
    total_bytes = sum(entry.storage_bytes for entry in packed_inventory.entries)
    linear = [entry for entry in packed_inventory.entries if entry.operator_group in linear_groups]
    linear_elements = sum(entry.elements for entry in linear)
    linear_bytes = sum(entry.storage_bytes for entry in linear)
    if not linear_elements:
        raise ValueError("linear_groups did not match any tensor")
    precision_elements: dict[Precision, int] = {}
    for entry in packed_inventory.entries:
        precision_elements[entry.precision] = (
            precision_elements.get(entry.precision, 0) + entry.elements
        )
    return PackedModelAccounting(
        total_elements=total_elements,
        eligible_linear_elements=linear_elements,
        logical_packed_weight_bytes=total_bytes,
        linear_packed_weight_bytes=linear_bytes,
        container_bytes=container_bytes,
        auxiliary_sidecar_bytes=auxiliary_sidecar_bytes,
        measured_vram_bytes=measured_vram_bytes,
        effective_bpw=8 * total_bytes / total_elements,
        linear_effective_bpw=8 * linear_bytes / linear_elements,
        precision_elements=precision_elements,
    )


class MixedPrecisionMetrics(StrictModel):
    tg128: float = Field(gt=0)
    tg512: float = Field(gt=0)
    ppl: float = Field(gt=0)
    accuracy: float = Field(ge=0, le=1)
    tg128_cv_percent: float = Field(ge=0)
    tg512_cv_percent: float = Field(ge=0)


class MixedPrecisionCandidateResult(StrictModel):
    schema_name: Literal["gpuopt.mixed-precision-result.v1"] = Field(
        default="gpuopt.mixed-precision-result.v1", alias="schema"
    )
    candidate_id: str
    policy: PrecisionPolicy
    accounting: PackedModelAccounting
    metrics: MixedPrecisionMetrics
    performance_stable: bool
    quality_passed: bool
    provisional_quality: Literal[True] = True
    evidence: list[ArtifactRef] = Field(min_length=1)


class ParetoPoint(StrictModel):
    candidate_id: str
    tg128: float
    tg512: float
    ppl: float
    accuracy: float
    effective_bpw: float
    packed_bytes: int
    dominated_by: list[str] = Field(default_factory=list)

    @property
    def on_frontier(self) -> bool:
        return not self.dominated_by


class ParetoFrontier(StrictModel):
    schema_name: Literal["gpuopt.mixed-precision-pareto.v1"] = Field(
        default="gpuopt.mixed-precision-pareto.v1", alias="schema"
    )
    points: list[ParetoPoint]
    frontier_candidate_ids: list[str]
    excluded_candidate_ids: list[str]
    selected_candidate_id: str | None


def _dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
    no_worse = (
        left.tg128 >= right.tg128
        and left.tg512 >= right.tg512
        and left.accuracy >= right.accuracy
        and left.ppl <= right.ppl
        and left.effective_bpw <= right.effective_bpw
        and left.packed_bytes <= right.packed_bytes
    )
    strictly_better = (
        left.tg128 > right.tg128
        or left.tg512 > right.tg512
        or left.accuracy > right.accuracy
        or left.ppl < right.ppl
        or left.effective_bpw < right.effective_bpw
        or left.packed_bytes < right.packed_bytes
    )
    return no_worse and strictly_better


def rank_pareto(results: list[MixedPrecisionCandidateResult]) -> ParetoFrontier:
    eligible = [result for result in results if result.performance_stable and result.quality_passed]
    excluded = [result.candidate_id for result in results if result not in eligible]
    points = [
        ParetoPoint(
            candidate_id=result.candidate_id,
            tg128=result.metrics.tg128,
            tg512=result.metrics.tg512,
            ppl=result.metrics.ppl,
            accuracy=result.metrics.accuracy,
            effective_bpw=result.accounting.effective_bpw,
            packed_bytes=result.accounting.logical_packed_weight_bytes,
        )
        for result in eligible
    ]
    ranked = [
        point.model_copy(
            update={
                "dominated_by": sorted(
                    candidate.candidate_id
                    for candidate in points
                    if candidate.candidate_id != point.candidate_id
                    and _dominates(candidate, point)
                )
            }
        )
        for point in points
    ]
    frontier = [point for point in ranked if point.on_frontier]
    winner = max(
        frontier,
        key=lambda point: (
            point.tg512,
            point.tg128,
            -point.effective_bpw,
            point.accuracy,
            point.candidate_id,
        ),
        default=None,
    )
    return ParetoFrontier(
        points=ranked,
        frontier_candidate_ids=sorted(point.candidate_id for point in frontier),
        excluded_candidate_ids=sorted(excluded),
        selected_candidate_id=winner.candidate_id if winner else None,
    )


class MixedPrecisionStage(StrEnum):
    INSPECT_MODEL = "INSPECT_MODEL"
    CAPTURE_SENSITIVITY = "CAPTURE_SENSITIVITY"
    PLAN_ASSIGNMENTS = "PLAN_ASSIGNMENTS"
    PACK = "PACK"
    BENCHMARK = "BENCHMARK"
    QUALITY = "QUALITY"
    ACCOUNT_BYTES = "ACCOUNT_BYTES"
    PARETO_RANK = "PARETO_RANK"
    DECIDE = "DECIDE"
    COMPLETE = "COMPLETE"


MIXED_PRECISION_STAGE_ORDER = tuple(MixedPrecisionStage)


class MixedPrecisionStageCompletion(StrictModel):
    stage: MixedPrecisionStage
    evidence: list[ArtifactRef] = Field(min_length=1)
    completed_at: datetime = Field(default_factory=utc_now)


class MixedPrecisionWorkflowRecord(StrictModel):
    schema_name: Literal["gpuopt.mixed-precision-workflow.v1"] = Field(
        default="gpuopt.mixed-precision-workflow.v1", alias="schema"
    )
    task_id: str
    current_stage: MixedPrecisionStage = MixedPrecisionStage.INSPECT_MODEL
    completions: list[MixedPrecisionStageCompletion] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    selected_candidate_id: str | None = None

    @model_validator(mode="after")
    def strict_stage_prefix(self) -> MixedPrecisionWorkflowRecord:
        completed = [item.stage for item in self.completions]
        if completed != list(MIXED_PRECISION_STAGE_ORDER[: len(completed)]):
            raise ValueError("mixed-precision stages must complete in strict order")
        if self.current_stage != MIXED_PRECISION_STAGE_ORDER[len(completed)]:
            raise ValueError("mixed-precision current stage does not follow completions")
        if self.current_stage == MixedPrecisionStage.COMPLETE and not self.selected_candidate_id:
            raise ValueError("completed mixed-precision workflow requires a winner")
        return self


def advance_mixed_precision(
    record: MixedPrecisionWorkflowRecord,
    evidence: list[ArtifactRef],
    *,
    candidate_ids: list[str] | None = None,
    selected_candidate_id: str | None = None,
) -> MixedPrecisionWorkflowRecord:
    if record.current_stage == MixedPrecisionStage.COMPLETE:
        raise ValueError("mixed-precision workflow is complete")
    if not evidence:
        raise ValueError("every mixed-precision stage requires immutable evidence")
    if record.current_stage == MixedPrecisionStage.PLAN_ASSIGNMENTS and not candidate_ids:
        raise ValueError("PLAN_ASSIGNMENTS requires candidate_ids")
    if record.current_stage == MixedPrecisionStage.DECIDE:
        if selected_candidate_id not in record.candidate_ids:
            raise ValueError("DECIDE winner must be a planned candidate")
    elif selected_candidate_id is not None:
        raise ValueError("winner may be selected only at DECIDE")
    completions = [
        *record.completions,
        MixedPrecisionStageCompletion(stage=record.current_stage, evidence=evidence),
    ]
    next_stage = MIXED_PRECISION_STAGE_ORDER[len(completions)]
    return MixedPrecisionWorkflowRecord(
        task_id=record.task_id,
        current_stage=next_stage,
        completions=completions,
        candidate_ids=candidate_ids or record.candidate_ids,
        selected_candidate_id=selected_candidate_id or record.selected_candidate_id,
    )


__all__ = [
    "MIXED_PRECISION_STAGE_ORDER",
    "MixedPrecisionCandidateResult",
    "MixedPrecisionMetrics",
    "MixedPrecisionStage",
    "MixedPrecisionWorkflowRecord",
    "PackedModelAccounting",
    "ParetoFrontier",
    "ParetoPoint",
    "Precision",
    "PrecisionAssignmentEntry",
    "PrecisionPolicy",
    "PrecisionPolicyProvider",
    "PrecisionRouteRule",
    "PrecisionSearchPoint",
    "PrecisionSearchSpace",
    "SensitivityEvidence",
    "TensorInventory",
    "TensorInventoryEntry",
    "TensorNamingConfig",
    "TensorNamingRule",
    "TensorSensitivityScore",
    "account_packed_model",
    "advance_mixed_precision",
    "assignment_hash",
    "build_external_policy",
    "plan_precision_policies",
    "rank_pareto",
    "verify_materialized_assignment",
]
