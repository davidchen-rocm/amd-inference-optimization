"""Evidence-first runtime memory reuse audit and bounded patch evaluation."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import ArtifactRef, DecisionOutcome, StrictModel


class MemoryAuditOutcome(StrEnum):
    OPPORTUNITY_FOUND = "OPPORTUNITY_FOUND"
    NO_ACTION = "NO_ACTION"
    INCONCLUSIVE = "INCONCLUSIVE"


class MemoryOpportunityKind(StrEnum):
    REPEATED_ALLOCATION = "repeated_allocation"
    REPEATED_WORKSPACE = "repeated_workspace"
    AVOIDABLE_COPY = "avoidable_copy"
    AVOIDABLE_MEMSET = "avoidable_memset"
    INTERMEDIATE_HBM_ROUNDTRIP = "intermediate_hbm_roundtrip"
    BUFFER_LIFETIME_REUSE = "buffer_lifetime_reuse"


class MemoryOperationAggregate(StrictModel):
    operation: str
    count: int = Field(ge=0)
    total_duration_ns: int | None = Field(default=None, ge=0)
    total_bytes: int | None = Field(default=None, ge=0)
    repeated_size_bytes: int | None = Field(default=None, ge=0)
    steady_state: bool | None = None


class BufferLifetimeEvidence(StrictModel):
    buffer_id: str
    size_bytes: int = Field(gt=0)
    first_use_ns: int = Field(ge=0)
    last_use_ns: int = Field(ge=0)
    reusable_with_buffer_id: str | None = None
    estimated_allocation_duration_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def ordered_lifetime(self) -> BufferLifetimeEvidence:
        if self.last_use_ns < self.first_use_ns:
            raise ValueError("buffer lifetime ends before it starts")
        return self


class IntermediateTrafficEvidence(StrictModel):
    producer_kernel_id: str
    consumer_kernel_id: str
    bytes_written: int | None = Field(default=None, ge=0)
    bytes_read: int | None = Field(default=None, ge=0)
    total_duration_ns: int | None = Field(default=None, ge=0)
    immediate_dependency: bool | None = None


class MemoryTraceEvidence(StrictModel):
    schema_name: Literal["gpuopt.memory-trace-evidence.v1"] = Field(
        default="gpuopt.memory-trace-evidence.v1", alias="schema"
    )
    workload_succeeded: bool
    trace_complete: bool
    steady_state_isolated: bool
    e2e_duration_ns: int = Field(gt=0)
    allocations: list[MemoryOperationAggregate] = Field(default_factory=list)
    copies: list[MemoryOperationAggregate] = Field(default_factory=list)
    lifetimes: list[BufferLifetimeEvidence] = Field(default_factory=list)
    intermediate_traffic: list[IntermediateTrafficEvidence] = Field(default_factory=list)
    coverage_percent: float | None = Field(default=None, ge=0, le=100)
    missing_evidence: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(min_length=1)

    @model_validator(mode="after")
    def lifetime_reuse_is_consistent(self) -> MemoryTraceEvidence:
        by_id = {item.buffer_id: item for item in self.lifetimes}
        if len(by_id) != len(self.lifetimes):
            raise ValueError("buffer lifetime ids must be unique")
        for item in self.lifetimes:
            if item.reusable_with_buffer_id is None:
                continue
            other = by_id.get(item.reusable_with_buffer_id)
            if other is None or other.buffer_id == item.buffer_id:
                raise ValueError("reusable buffer lifetime target is invalid")
            overlaps = not (
                item.last_use_ns < other.first_use_ns
                or other.last_use_ns < item.first_use_ns
            )
            if overlaps:
                raise ValueError("buffers marked reusable must have non-overlapping lifetimes")
        return self


class MemoryOpportunity(StrictModel):
    id: str
    kind: MemoryOpportunityKind
    summary: str
    estimated_recoverable_ns: int = Field(gt=0)
    estimated_e2e_percent: float = Field(gt=0)
    evidence: list[ArtifactRef] = Field(min_length=1)
    affected_operations: list[str] = Field(default_factory=list)
    proposed_change: str

    @field_validator("id", "summary", "proposed_change")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("memory opportunity text fields cannot be empty")
        return value


class MemoryAuditResult(StrictModel):
    schema_name: Literal["gpuopt.memory-audit-result.v1"] = Field(
        default="gpuopt.memory-audit-result.v1", alias="schema"
    )
    outcome: MemoryAuditOutcome
    opportunities: list[MemoryOpportunity] = Field(max_length=2)
    reasons: list[str]
    upper_bound_e2e_percent: float | None = Field(default=None, ge=0)
    evidence: list[ArtifactRef] = Field(min_length=1)
    missing_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def outcome_matches_opportunities(self) -> MemoryAuditResult:
        if self.outcome == MemoryAuditOutcome.OPPORTUNITY_FOUND and not self.opportunities:
            raise ValueError("OPPORTUNITY_FOUND requires at least one opportunity")
        if self.outcome != MemoryAuditOutcome.OPPORTUNITY_FOUND and self.opportunities:
            raise ValueError("only OPPORTUNITY_FOUND may carry patch opportunities")
        return self


def _opportunity(
    identifier: str,
    kind: MemoryOpportunityKind,
    summary: str,
    duration_ns: int,
    evidence: MemoryTraceEvidence,
    operation: str,
    change: str,
) -> MemoryOpportunity:
    return MemoryOpportunity(
        id=identifier,
        kind=kind,
        summary=summary,
        estimated_recoverable_ns=duration_ns,
        estimated_e2e_percent=duration_ns / evidence.e2e_duration_ns * 100,
        evidence=evidence.artifacts,
        affected_operations=[operation],
        proposed_change=change,
    )


def audit_memory_reuse(
    evidence: MemoryTraceEvidence,
    *,
    minimum_e2e_percent: float = 1.0,
) -> MemoryAuditResult:
    """Return a bounded audit result without inventing missing byte or lifetime facts."""

    if not math.isfinite(minimum_e2e_percent) or minimum_e2e_percent <= 0:
        raise ValueError("minimum_e2e_percent must be positive and finite")
    required_missing = list(evidence.missing_evidence)
    if not evidence.workload_succeeded:
        required_missing.append("workload did not succeed")
    if not evidence.trace_complete:
        required_missing.append("runtime memory trace is incomplete")
    if not evidence.steady_state_isolated:
        required_missing.append("model-load and steady-state memory events are not separated")
    if evidence.coverage_percent is None or evidence.coverage_percent < 100:
        required_missing.append("exact trace coverage is unavailable")
    if required_missing:
        return MemoryAuditResult(
            outcome=MemoryAuditOutcome.INCONCLUSIVE,
            opportunities=[],
            reasons=["required runtime memory evidence is incomplete"],
            evidence=evidence.artifacts,
            missing_evidence=sorted(set(required_missing)),
        )

    candidates: list[MemoryOpportunity] = []
    unresolved_lifetime_reuse = False
    for index, aggregate in enumerate(evidence.allocations):
        if (
            aggregate.count > 1
            and aggregate.steady_state is True
            and aggregate.total_duration_ns is not None
            and aggregate.total_duration_ns / evidence.e2e_duration_ns * 100
            >= minimum_e2e_percent
        ):
            kind = (
                MemoryOpportunityKind.REPEATED_WORKSPACE
                if "workspace" in aggregate.operation.lower()
                else MemoryOpportunityKind.REPEATED_ALLOCATION
            )
            candidates.append(
                _opportunity(
                    f"allocation-{index}",
                    kind,
                    f"Repeated steady-state {aggregate.operation}",
                    aggregate.total_duration_ns,
                    evidence,
                    aggregate.operation,
                    "retain and reuse the exact-size runtime buffer across decode iterations",
                )
            )
    for index, aggregate in enumerate(evidence.copies):
        duration = aggregate.total_duration_ns
        if (
            aggregate.count > 0
            and aggregate.steady_state is True
            and duration is not None
            and duration / evidence.e2e_duration_ns * 100 >= minimum_e2e_percent
        ):
            is_memset = "memset" in aggregate.operation.lower()
            candidates.append(
                _opportunity(
                    f"copy-{index}",
                    MemoryOpportunityKind.AVOIDABLE_MEMSET
                    if is_memset
                    else MemoryOpportunityKind.AVOIDABLE_COPY,
                    f"Material steady-state {aggregate.operation}",
                    duration,
                    evidence,
                    aggregate.operation,
                    "remove, narrow, or fuse the evidenced memory operation",
                )
            )
    for index, traffic in enumerate(evidence.intermediate_traffic):
        duration = traffic.total_duration_ns
        if (
            traffic.immediate_dependency is True
            and duration is not None
            and traffic.bytes_written is not None
            and traffic.bytes_read is not None
            and duration / evidence.e2e_duration_ns * 100 >= minimum_e2e_percent
        ):
            candidates.append(
                _opportunity(
                    f"intermediate-{index}",
                    MemoryOpportunityKind.INTERMEDIATE_HBM_ROUNDTRIP,
                    "Producer output is immediately reread from global memory",
                    duration,
                    evidence,
                    f"{traffic.producer_kernel_id}->{traffic.consumer_kernel_id}",
                    "fuse the exact producer/consumer pair after correctness validation",
                )
            )
    seen_lifetime_pairs: set[tuple[str, str]] = set()
    for lifetime in evidence.lifetimes:
        if lifetime.reusable_with_buffer_id is None:
            continue
        pair = tuple(sorted((lifetime.buffer_id, lifetime.reusable_with_buffer_id)))
        if pair in seen_lifetime_pairs:
            continue
        seen_lifetime_pairs.add(pair)
        duration = lifetime.estimated_allocation_duration_ns
        if duration is None:
            unresolved_lifetime_reuse = True
            continue
        if duration / evidence.e2e_duration_ns * 100 >= minimum_e2e_percent:
            candidates.append(
                _opportunity(
                    f"lifetime-{pair[0]}-{pair[1]}",
                    MemoryOpportunityKind.BUFFER_LIFETIME_REUSE,
                    "Two evidenced buffers have non-overlapping reusable lifetimes",
                    duration,
                    evidence,
                    f"{pair[0]}->{pair[1]}",
                    "alias the exact compatible buffers after size and alignment validation",
                )
            )

    candidates.sort(key=lambda item: (-item.estimated_recoverable_ns, item.id))
    selected = candidates[:2]
    if selected:
        return MemoryAuditResult(
            outcome=MemoryAuditOutcome.OPPORTUNITY_FOUND,
            opportunities=selected,
            reasons=["one or more evidence-backed opportunities exceed the E2E threshold"],
            evidence=evidence.artifacts,
        )
    if unresolved_lifetime_reuse:
        return MemoryAuditResult(
            outcome=MemoryAuditOutcome.INCONCLUSIVE,
            opportunities=[],
            reasons=["a buffer lifetime reuse candidate lacks a measured E2E cost"],
            evidence=evidence.artifacts,
            missing_evidence=["allocation/workspace duration for reusable lifetime pair"],
        )
    observed_ns = sum(
        aggregate.total_duration_ns or 0
        for aggregate in [*evidence.allocations, *evidence.copies]
    ) + sum(item.total_duration_ns or 0 for item in evidence.intermediate_traffic)
    observed_ns += sum(
        item.estimated_allocation_duration_ns or 0 for item in evidence.lifetimes
    )
    upper_bound = observed_ns / evidence.e2e_duration_ns * 100
    if upper_bound < minimum_e2e_percent:
        return MemoryAuditResult(
            outcome=MemoryAuditOutcome.NO_ACTION,
            opportunities=[],
            reasons=["complete trace bounds recoverable memory overhead below threshold"],
            upper_bound_e2e_percent=upper_bound,
            evidence=evidence.artifacts,
        )
    return MemoryAuditResult(
        outcome=MemoryAuditOutcome.INCONCLUSIVE,
        opportunities=[],
        reasons=["observed memory cost is material but no safe reuse transformation is proven"],
        upper_bound_e2e_percent=upper_bound,
        evidence=evidence.artifacts,
        missing_evidence=["operation-to-buffer lifetime or dependency attribution"],
    )


def memory_trace_from_rocm(
    payload: dict[str, object],
    *,
    e2e_duration_ns: int,
    steady_state_isolated: bool,
    artifacts: list[ArtifactRef],
) -> MemoryTraceEvidence:
    """Project normalized ROCm evidence without parsing private profiler CSV."""

    runtime = payload.get("runtime_trace", payload)
    if not isinstance(runtime, dict):
        raise ValueError("ROCm trace payload has no runtime_trace object")

    def aggregates(key: str) -> list[MemoryOperationAggregate]:
        value = runtime.get(key, [])
        if not isinstance(value, list):
            return []
        result: list[MemoryOperationAggregate] = []
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("operation"), str):
                continue
            result.append(MemoryOperationAggregate.model_validate(item))
        return result

    allocations = aggregates("memory_allocation_aggregates")
    copies = aggregates("memory_copy_aggregates")
    trace_complete = runtime.get("trace_status") == "completed"
    warning_details = runtime.get("warning_details", [])
    complete_coverage = trace_complete and not warning_details
    missing: list[str] = []
    if int(runtime.get("memory_allocation_count", 0) or 0) and not allocations:
        missing.append("ROCm summary lacks complete allocation aggregates")
    if int(runtime.get("memory_copy_count", 0) or 0) and not copies:
        missing.append("ROCm summary lacks complete copy duration/byte aggregates")
    if not runtime.get("memory_copy_bytes_by_direction") and copies:
        missing.append("installed profiler does not report memory-copy bytes")
    return MemoryTraceEvidence(
        workload_succeeded=runtime.get("workload_exit_code") == 0,
        trace_complete=trace_complete,
        steady_state_isolated=steady_state_isolated,
        e2e_duration_ns=e2e_duration_ns,
        allocations=allocations,
        copies=copies,
        coverage_percent=100.0 if complete_coverage else None,
        missing_evidence=missing,
        artifacts=artifacts,
    )


class MemoryPatchEvaluation(StrictModel):
    schema_name: Literal["gpuopt.memory-patch-evaluation.v1"] = Field(
        default="gpuopt.memory-patch-evaluation.v1", alias="schema"
    )
    opportunity_id: str
    correctness_passed: bool
    baseline_tg128: float = Field(gt=0)
    candidate_tg128: float = Field(gt=0)
    baseline_tg512: float = Field(gt=0)
    candidate_tg512: float = Field(gt=0)
    baseline_kernel_or_runtime_latency_ns: int | None = Field(default=None, gt=0)
    candidate_kernel_or_runtime_latency_ns: int | None = Field(default=None, gt=0)
    baseline_memory_operation_count: int = Field(ge=0)
    candidate_memory_operation_count: int = Field(ge=0)
    baseline_memory_duration_ns: int | None = Field(default=None, ge=0)
    candidate_memory_duration_ns: int | None = Field(default=None, ge=0)
    evidence: list[ArtifactRef] = Field(min_length=1)

    @model_validator(mode="after")
    def paired_optional_metrics(self) -> MemoryPatchEvaluation:
        if (self.baseline_kernel_or_runtime_latency_ns is None) != (
            self.candidate_kernel_or_runtime_latency_ns is None
        ):
            raise ValueError("memory patch latency must be present for both arms")
        if (self.baseline_memory_duration_ns is None) != (
            self.candidate_memory_duration_ns is None
        ):
            raise ValueError("memory duration must be present for both arms")
        return self


class MemoryPatchDecision(StrictModel):
    outcome: DecisionOutcome
    reasons: list[str]
    evaluation: MemoryPatchEvaluation


def evaluate_memory_patch(
    evaluation: MemoryPatchEvaluation,
    *,
    maximum_tps_regression_percent: float = 1.0,
    minimum_memory_reduction_percent: float = 5.0,
) -> MemoryPatchDecision:
    """Gate one of the at-most-two audit patches on before/after evidence."""

    if maximum_tps_regression_percent < 0 or minimum_memory_reduction_percent <= 0:
        raise ValueError("memory patch gate thresholds are invalid")
    if not evaluation.correctness_passed:
        return MemoryPatchDecision(
            outcome=DecisionOutcome.REJECT,
            reasons=["correctness validation failed"],
            evaluation=evaluation,
        )
    if (
        evaluation.baseline_kernel_or_runtime_latency_ns is None
        or evaluation.baseline_memory_duration_ns is None
    ):
        return MemoryPatchDecision(
            outcome=DecisionOutcome.INCONCLUSIVE,
            reasons=["paired latency and memory-duration evidence is required"],
            evaluation=evaluation,
        )
    tg128_delta = (
        evaluation.candidate_tg128 / evaluation.baseline_tg128 - 1
    ) * 100
    tg512_delta = (
        evaluation.candidate_tg512 / evaluation.baseline_tg512 - 1
    ) * 100
    if min(tg128_delta, tg512_delta) < -maximum_tps_regression_percent:
        return MemoryPatchDecision(
            outcome=DecisionOutcome.REJECT,
            reasons=["tg128 or tg512 regressed beyond the configured budget"],
            evaluation=evaluation,
        )
    baseline_duration = evaluation.baseline_memory_duration_ns
    memory_reduction = (
        (baseline_duration - evaluation.candidate_memory_duration_ns)
        / baseline_duration
        * 100
        if baseline_duration > 0
        else 0.0
    )
    operation_reduced = (
        evaluation.candidate_memory_operation_count
        < evaluation.baseline_memory_operation_count
    )
    latency_improved = (
        evaluation.candidate_kernel_or_runtime_latency_ns
        < evaluation.baseline_kernel_or_runtime_latency_ns
    )
    if (
        not latency_improved
        or not operation_reduced
        or memory_reduction < minimum_memory_reduction_percent
    ):
        return MemoryPatchDecision(
            outcome=DecisionOutcome.REJECT,
            reasons=["the patch did not materially reduce the evidenced memory hotspot"],
            evaluation=evaluation,
        )
    return MemoryPatchDecision(
        outcome=DecisionOutcome.ACCEPT,
        reasons=["correctness passed and the evidenced memory hotspot was reduced"],
        evaluation=evaluation,
    )


__all__ = [
    "BufferLifetimeEvidence",
    "IntermediateTrafficEvidence",
    "MemoryAuditOutcome",
    "MemoryAuditResult",
    "MemoryOperationAggregate",
    "MemoryOpportunity",
    "MemoryOpportunityKind",
    "MemoryPatchEvaluation",
    "MemoryPatchDecision",
    "MemoryTraceEvidence",
    "audit_memory_reuse",
    "evaluate_memory_patch",
    "memory_trace_from_rocm",
]
