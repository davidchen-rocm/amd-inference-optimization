"""Deterministic workflow state machine and evidence gates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import (
    ArtifactRef,
    DecisionOutcome,
    GateDecision,
    StageCompletion,
    WorkflowRecord,
    WorkflowStage,
    WorkflowStatus,
    utc_now,
)


class WorkflowError(RuntimeError):
    """Raised when a caller attempts to skip or incompletely close a stage."""


STAGE_REQUIREMENTS: dict[WorkflowStage, frozenset[str]] = {
    WorkflowStage.CREATE_TASK: frozenset({"task"}),
    WorkflowStage.INSPECT_TARGET: frozenset({"inspection"}),
    WorkflowStage.CAPTURE_BASELINE: frozenset({"baseline"}),
    WorkflowStage.DECOMPOSE_E2E: frozenset({"decomposition"}),
    WorkflowStage.BUILD_EXECUTION_MAP: frozenset({"execution_map"}),
    WorkflowStage.DISCOVER_HOTSPOTS: frozenset({"kernel_evidence"}),
    WorkflowStage.CLASSIFY_BOTTLENECK: frozenset(
        {"agent_decision", "bottleneck_assessment"}
    ),
    WorkflowStage.ANALYZE_LIMIT: frozenset({"agent_decision", "limit_estimate"}),
    WorkflowStage.GENERATE_HYPOTHESIS: frozenset({"agent_decision", "hypothesis"}),
    WorkflowStage.CREATE_EXPERIMENT: frozenset({"agent_decision", "experiment_spec"}),
    WorkflowStage.PATCH_AND_BUILD: frozenset({"build_result"}),
    WorkflowStage.MICROBENCH: frozenset({"microbenchmark_result"}),
    WorkflowStage.E2E_VALIDATION: frozenset({"e2e_result"}),
    WorkflowStage.QUALITY_VALIDATION: frozenset({"quality_result"}),
    WorkflowStage.DECIDE: frozenset({"gate_decision"}),
}


_FORWARD: dict[WorkflowStage, WorkflowStage] = {
    WorkflowStage.CREATE_TASK: WorkflowStage.INSPECT_TARGET,
    WorkflowStage.INSPECT_TARGET: WorkflowStage.CAPTURE_BASELINE,
    WorkflowStage.CAPTURE_BASELINE: WorkflowStage.DECOMPOSE_E2E,
    WorkflowStage.DECOMPOSE_E2E: WorkflowStage.BUILD_EXECUTION_MAP,
    WorkflowStage.BUILD_EXECUTION_MAP: WorkflowStage.DISCOVER_HOTSPOTS,
    WorkflowStage.DISCOVER_HOTSPOTS: WorkflowStage.CLASSIFY_BOTTLENECK,
    WorkflowStage.CLASSIFY_BOTTLENECK: WorkflowStage.ANALYZE_LIMIT,
    WorkflowStage.ANALYZE_LIMIT: WorkflowStage.GENERATE_HYPOTHESIS,
    WorkflowStage.GENERATE_HYPOTHESIS: WorkflowStage.CREATE_EXPERIMENT,
    WorkflowStage.CREATE_EXPERIMENT: WorkflowStage.PATCH_AND_BUILD,
    WorkflowStage.PATCH_AND_BUILD: WorkflowStage.MICROBENCH,
    WorkflowStage.MICROBENCH: WorkflowStage.E2E_VALIDATION,
    WorkflowStage.E2E_VALIDATION: WorkflowStage.QUALITY_VALIDATION,
    WorkflowStage.QUALITY_VALIDATION: WorkflowStage.DECIDE,
}

_DEEPER_PROFILE_STAGES = {
    WorkflowStage.CLASSIFY_BOTTLENECK,
    WorkflowStage.ANALYZE_LIMIT,
    WorkflowStage.GENERATE_HYPOTHESIS,
}


class WorkflowEngine:
    """Advance workflows only after the current stage's evidence is persisted."""

    @staticmethod
    def new(task_id: str) -> WorkflowRecord:
        return WorkflowRecord(task_id=task_id)

    @staticmethod
    def required_evidence(stage: WorkflowStage) -> frozenset[str]:
        return STAGE_REQUIREMENTS[stage]

    @staticmethod
    def permitted_next_stages(stage: WorkflowStage) -> tuple[WorkflowStage, ...]:
        if stage == WorkflowStage.DECIDE:
            return (WorkflowStage.GENERATE_HYPOTHESIS,)
        forward = _FORWARD[stage]
        if stage in _DEEPER_PROFILE_STAGES:
            return (forward, WorkflowStage.DISCOVER_HOTSPOTS)
        return (forward,)

    def complete_stage(
        self,
        record: WorkflowRecord,
        evidence_ids: Mapping[str, str | ArtifactRef],
        *,
        evidence_artifacts: Mapping[str, ArtifactRef] | None = None,
        require_artifact_bindings: bool = False,
        requested_next_stage: WorkflowStage | None = None,
        gate_decision: GateDecision | None = None,
        can_continue_after_reject: bool = False,
    ) -> WorkflowRecord:
        if record.status != WorkflowStatus.ACTIVE:
            raise WorkflowError(f"workflow is terminal: {record.status}")

        stage = record.current_stage
        normalized_ids: dict[str, str] = {}
        artifact_bindings = dict(evidence_artifacts or {})
        for key, value in evidence_ids.items():
            if isinstance(value, ArtifactRef):
                normalized_ids[key] = value.path
                artifact_bindings[key] = value
            else:
                normalized_ids[key] = value

        provided = {key for key, value in normalized_ids.items() if value}
        missing = STAGE_REQUIREMENTS[stage] - provided
        if missing:
            raise WorkflowError(
                f"cannot complete {stage}: missing evidence {', '.join(sorted(missing))}"
            )
        if any(not value.strip() for value in normalized_ids.values()):
            raise WorkflowError("evidence ids cannot be empty")
        unknown_bindings = set(artifact_bindings) - set(normalized_ids)
        if unknown_bindings:
            raise WorkflowError(
                "artifact bindings have no evidence id: "
                + ", ".join(sorted(unknown_bindings))
            )
        mismatched_bindings = [
            key
            for key, artifact in artifact_bindings.items()
            if artifact.path != normalized_ids[key]
        ]
        if mismatched_bindings:
            raise WorkflowError(
                "artifact binding paths do not match evidence ids: "
                + ", ".join(sorted(mismatched_bindings))
            )
        if require_artifact_bindings:
            unbound = set(normalized_ids) - set(artifact_bindings)
            if unbound:
                raise WorkflowError(
                    "stage evidence is not hash-bound: " + ", ".join(sorted(unbound))
                )

        updated = record.model_copy(deep=True)
        updated.completions.append(
            StageCompletion(
                stage=stage,
                evidence_ids=normalized_ids,
                evidence_artifacts=artifact_bindings,
            )
        )
        updated.updated_at = utc_now()

        if stage == WorkflowStage.CREATE_EXPERIMENT:
            updated.experiment_count += 1

        if stage == WorkflowStage.DECIDE:
            return self._apply_decision(
                updated,
                gate_decision,
                can_continue_after_reject=can_continue_after_reject,
            )

        if gate_decision is not None:
            raise WorkflowError("gate_decision is only valid in DECIDE")
        permitted = self.permitted_next_stages(stage)
        next_stage = requested_next_stage or permitted[0]
        if next_stage not in permitted:
            allowed = ", ".join(item.value for item in permitted)
            raise WorkflowError(f"illegal transition {stage} -> {next_stage}; allowed: {allowed}")
        updated.current_stage = next_stage
        return updated

    def complete_stage_from_artifacts(
        self,
        record: WorkflowRecord,
        evidence: Mapping[str, ArtifactRef],
        **kwargs: Any,
    ) -> WorkflowRecord:
        """Strict live-path helper: every completion is bound to path and SHA-256."""

        return self.complete_stage(
            record,
            evidence,
            require_artifact_bindings=True,
            **kwargs,
        )

    def complete_performance_rejection(
        self,
        record: WorkflowRecord,
        *,
        e2e_result: ArtifactRef,
        gate_decision_artifact: ArtifactRef,
        gate_decision: GateDecision,
        can_continue_after_reject: bool = False,
    ) -> WorkflowRecord:
        """Close a proven-slow experiment without fabricating quality evidence."""

        if record.current_stage != WorkflowStage.E2E_VALIDATION:
            raise WorkflowError(
                "performance rejection is only valid after E2E_VALIDATION"
            )
        if gate_decision.outcome != DecisionOutcome.REJECT:
            raise WorkflowError("early performance decision must be REJECT")
        after_e2e = self.complete_stage_from_artifacts(
            record, {"e2e_result": e2e_result}
        )
        # This is the sole code-owned bypass of QUALITY_VALIDATION. The rejected
        # performance gate proves quality cannot turn the experiment into ACCEPT.
        at_decide = after_e2e.model_copy(
            update={"current_stage": WorkflowStage.DECIDE}
        )
        return self.complete_stage_from_artifacts(
            at_decide,
            {"gate_decision": gate_decision_artifact},
            gate_decision=gate_decision,
            can_continue_after_reject=can_continue_after_reject,
        )

    @staticmethod
    def resume_inconclusive(record: WorkflowRecord) -> WorkflowRecord:
        """Resume a retryable INCONCLUSIVE workflow at its Gate-selected stage."""

        if record.status != WorkflowStatus.INCONCLUSIVE:
            raise WorkflowError("only an INCONCLUSIVE workflow can be resumed")
        decision = record.terminal_decision
        if decision is None or decision.rerun_from_stage is None:
            raise WorkflowError("INCONCLUSIVE decision does not identify a rerun stage")
        return WorkflowRecord.model_validate(
            {
                **record.model_dump(),
                "current_stage": decision.rerun_from_stage,
                "status": WorkflowStatus.ACTIVE,
                "terminal_decision": None,
                "rerun_count": record.rerun_count + 1,
                "updated_at": utc_now(),
            }
        )

    @staticmethod
    def _apply_decision(
        updated: WorkflowRecord,
        gate_decision: GateDecision | None,
        *,
        can_continue_after_reject: bool,
    ) -> WorkflowRecord:
        if gate_decision is None:
            raise WorkflowError("DECIDE requires the parsed GateDecision")

        if gate_decision.outcome == DecisionOutcome.ACCEPT:
            return WorkflowRecord.model_validate(
                {
                    **updated.model_dump(),
                    "status": WorkflowStatus.ACCEPTED,
                    "terminal_decision": gate_decision,
                }
            )
        elif gate_decision.outcome == DecisionOutcome.INCONCLUSIVE:
            return WorkflowRecord.model_validate(
                {
                    **updated.model_dump(),
                    "status": WorkflowStatus.INCONCLUSIVE,
                    "terminal_decision": gate_decision,
                }
            )
        elif can_continue_after_reject:
            updated.current_stage = WorkflowStage.GENERATE_HYPOTHESIS
        else:
            return WorkflowRecord.model_validate(
                {
                    **updated.model_dump(),
                    "status": WorkflowStatus.REJECTED,
                    "terminal_decision": gate_decision,
                }
            )
        return updated

    def validate_requested_transition(
        self,
        record: WorkflowRecord,
        requested_next_stage: WorkflowStage,
    ) -> None:
        if record.status != WorkflowStatus.ACTIVE:
            raise WorkflowError("terminal workflows cannot transition")
        if requested_next_stage not in self.permitted_next_stages(record.current_stage):
            raise WorkflowError(
                f"illegal transition {record.current_stage} -> {requested_next_stage}"
            )
