"""Deterministic single-step control plane for persisted campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .campaign import CampaignEngine, CampaignError
from .campaign_models import (
    STRATEGY_STAGES,
    CampaignRecord,
    CampaignStage,
    CampaignStatus,
    CandidateDisposition,
    CompatibilityState,
    FinalCombinationState,
    SharedBaselineReference,
)
from .campaign_planning import CampaignPlanningError, materialize_campaign_plan
from .campaign_store import CampaignStore
from .models import ArtifactRef, DecisionOutcome, GateDecision
from .quality_policy import (
    ProvisionalMath100Policy,
    ProvisionalQualityMeasurement,
    QualityPolicyError,
)
from .store import ExperimentStore, StoreError


class CampaignControlError(RuntimeError):
    """A stage input is missing, stale, or not bound to stored evidence."""


@dataclass(frozen=True)
class CampaignStepResult:
    record: CampaignRecord
    action: str
    detail: dict[str, Any]


def _artifact(store: ExperimentStore, task_id: str, path: object) -> ArtifactRef:
    if not isinstance(path, str) or not path:
        raise CampaignControlError("evidence_paths must contain non-empty strings")
    reference = store.artifact_ref(task_id, path)
    if reference is None:
        raise CampaignControlError(f"artifact is not registered in the task store: {path}")
    return reference


def _evidence(
    store: ExperimentStore,
    task_id: str,
    document: dict[str, Any] | None,
) -> list[ArtifactRef]:
    raw = [] if document is None else document.get("evidence_paths", [])
    if not isinstance(raw, list):
        raise CampaignControlError("evidence_paths must be an array")
    return [_artifact(store, task_id, path) for path in raw]


def _require_input(document: dict[str, Any] | None, stage: CampaignStage) -> dict[str, Any]:
    if document is None:
        raise CampaignControlError(f"{stage} requires --input")
    return document


def _save(
    campaign_store: CampaignStore,
    record: CampaignRecord,
    action: str,
    **detail: Any,
) -> CampaignStepResult:
    campaign_store.save(record)
    return CampaignStepResult(record=record, action=action, detail=detail)


def _candidate_gate(
    store: ExperimentStore,
    record: CampaignRecord,
    candidate_id: str,
    document: dict[str, Any],
) -> tuple[CandidateDisposition, list[ArtifactRef], list[str]]:
    """Derive a candidate disposition from registered deterministic gate evidence."""

    performance_path = document.get("performance_gate_path")
    performance_ref = _artifact(store, record.task_id, performance_path)
    performance = store.load_json(record.task_id, performance_ref.path, GateDecision)
    evidence = [performance_ref]
    reasons = list(performance.reasons)
    quality_document: dict[str, Any] | None = None
    quality_result: dict[str, Any] | None = None

    if performance.outcome == DecisionOutcome.REJECT:
        disposition = CandidateDisposition.REJECTED
    elif performance.outcome == DecisionOutcome.INCONCLUSIVE:
        disposition = CandidateDisposition.INCONCLUSIVE
    else:
        quality_path = document.get("provisional_quality_path")
        quality_ref = _artifact(store, record.task_id, quality_path)
        evidence.append(quality_ref)
        quality_document = store.load_json(record.task_id, quality_ref.path)
        if not isinstance(quality_document, dict):
            raise CampaignControlError("provisional quality evidence must be an object")
        baseline = ProvisionalQualityMeasurement.from_value(
            quality_document.get("baseline")
        )
        candidate = ProvisionalQualityMeasurement.from_value(
            quality_document.get("candidate")
        )
        policy = ProvisionalMath100Policy()
        if baseline.protocol_hash is None or candidate.protocol_hash is None:
            raise CampaignControlError(
                "campaign quality measurements must bind the provisional protocol hash"
            )
        evaluated = policy.evaluate(baseline, candidate, protocol=policy.protocol)
        quality_result = evaluated.to_dict()
        reasons.extend(evaluated.reasons)
        disposition = (
            CandidateDisposition.EXPERIMENTAL_ACCEPTED
            if evaluated.passed
            else CandidateDisposition.REJECTED
        )

    gate_document = {
        "schema": "gpuopt.campaign-candidate-gate.v1",
        "candidate_id": candidate_id,
        "disposition": disposition.value,
        "performance_gate": performance.model_dump(mode="json"),
        "provisional_quality": quality_result,
        "quality_protocol_pending_replacement": True,
        "production_ready": False,
        "source_evidence": [item.model_dump(mode="json") for item in evidence],
    }
    gate_ref = store.save_evidence_json(
        record.task_id,
        f"campaign/gates/{candidate_id}",
        gate_document,
        producer="campaign-gate",
    )
    return disposition, [*evidence, gate_ref], reasons


def run_campaign_step(
    record: CampaignRecord,
    campaign_store: CampaignStore,
    *,
    stage_input: dict[str, Any] | None = None,
) -> CampaignStepResult:
    """Advance exactly one durable revision, or report the required next input."""

    store = campaign_store.store
    engine = CampaignEngine()
    stage = record.current_stage
    try:
        if stage == CampaignStage.COMPLETE:
            return CampaignStepResult(record, "complete", {})

        if stage == CampaignStage.CREATE_CAMPAIGN:
            return _save(campaign_store, engine.advance(record), "advanced")

        if stage == CampaignStage.INSPECT_TARGET:
            if stage_input is None:
                return CampaignStepResult(
                    record,
                    "waiting_for_inspection",
                    {"required": ["evidence_paths"]},
                )
            evidence = _evidence(store, record.task_id, stage_input)
            return _save(
                campaign_store,
                engine.advance(record, evidence=evidence),
                "inspection_recorded",
            )

        if stage == CampaignStage.CAPTURE_SHARED_BASELINES:
            if stage_input is None:
                return CampaignStepResult(
                    record,
                    "waiting_for_shared_baselines",
                    {"required": ["shared_baselines"]},
                )
            raw = stage_input.get("shared_baselines")
            if not isinstance(raw, list):
                raise CampaignControlError("shared_baselines must be an array")
            baselines = [SharedBaselineReference.model_validate(item) for item in raw]
            for baseline in baselines:
                actual = _artifact(store, record.task_id, baseline.artifact.path)
                if actual.sha256 != baseline.artifact.sha256:
                    raise CampaignControlError(
                        f"shared baseline hash mismatch: {baseline.artifact.path}"
                    )
            return _save(
                campaign_store,
                engine.advance(record, shared_baselines=baselines),
                "shared_baselines_recorded",
                count=len(baselines),
            )

        if stage == CampaignStage.PLAN_CANDIDATES:
            candidates, plan = materialize_campaign_plan(record, store)
            return _save(
                campaign_store,
                engine.advance(record, evidence=[plan], candidates=candidates),
                "candidates_planned",
                candidate_count=len(candidates),
                plan_artifact=plan.path,
            )

        if stage in STRATEGY_STAGES.values():
            running = next(
                (
                    candidate
                    for candidate in record.candidates
                    if candidate.disposition == CandidateDisposition.RUNNING
                ),
                None,
            )
            if running is None:
                return _save(
                    campaign_store,
                    engine.advance(record),
                    "strategy_completed",
                )
            if stage_input is None:
                return CampaignStepResult(
                    record,
                    "waiting_for_candidate_result",
                    {
                        "candidate_id": running.id,
                        "spec_artifact": (
                            running.spec_artifact.model_dump(mode="json")
                            if running.spec_artifact
                            else None
                        ),
                        "required": ["candidate_id", "performance_gate_path"],
                        "quality_when_performance_passes": "provisional_quality_path",
                    },
                )
            if stage_input.get("candidate_id") != running.id:
                raise CampaignControlError(
                    f"current running candidate is {running.id}, not "
                    f"{stage_input.get('candidate_id')}"
                )
            if stage_input.get("disposition") == CandidateDisposition.SKIPPED.value:
                evidence = _evidence(store, record.task_id, stage_input)
                raw_reasons = stage_input.get("reasons", [])
                if (
                    not evidence
                    or not isinstance(raw_reasons, list)
                    or not raw_reasons
                    or not all(isinstance(reason, str) and reason for reason in raw_reasons)
                ):
                    raise CampaignControlError(
                        "SKIPPED requires evidence_paths and non-empty reasons"
                    )
                updated = engine.record_candidate_result(
                    record,
                    running.id,
                    CandidateDisposition.SKIPPED,
                    artifacts=evidence,
                    reasons=raw_reasons,
                )
                return _save(
                    campaign_store,
                    updated,
                    "candidate_skipped",
                    candidate_id=running.id,
                )
            disposition, evidence, gate_reasons = _candidate_gate(
                store,
                record,
                running.id,
                stage_input,
            )
            raw_reasons = stage_input.get("reasons", [])
            if not isinstance(raw_reasons, list) or not all(
                isinstance(reason, str) and reason for reason in raw_reasons
            ):
                raise CampaignControlError("reasons must be an array of non-empty strings")
            updated = engine.record_candidate_result(
                record,
                running.id,
                disposition,
                artifacts=evidence,
                reasons=[*gate_reasons, *raw_reasons],
            )
            return _save(
                campaign_store,
                updated,
                "candidate_result_recorded",
                candidate_id=running.id,
            )

        if stage == CampaignStage.SELECT_WINNERS:
            accepted = [
                candidate
                for candidate in record.candidates
                if candidate.disposition == CandidateDisposition.EXPERIMENTAL_ACCEPTED
            ]
            if len(accepted) > 1 and stage_input is None:
                return CampaignStepResult(
                    record,
                    "waiting_for_winner_selection",
                    {
                        "eligible_candidate_ids": [item.id for item in accepted],
                        "required": ["selected_candidate_ids"],
                        "constraint": (
                            "at most one mixed-bit and one KV-cache winner; "
                            "multiple independent shape winners are allowed"
                        ),
                    },
                )
            selected_ids = (
                [item.id for item in accepted]
                if stage_input is None
                else stage_input.get("selected_candidate_ids")
            )
            if not isinstance(selected_ids, list) or not all(
                isinstance(item, str) for item in selected_ids
            ):
                raise CampaignControlError("selected_candidate_ids must be a string array")
            eligible = {item.id: item for item in accepted}
            if set(selected_ids) - set(eligible):
                raise CampaignControlError(
                    "winner selection contains a candidate that was not experimentally accepted"
                )
            for strategy in ("mixed_bit", "kv_cache"):
                chosen = [
                    item
                    for item in selected_ids
                    if eligible[item].strategy.value == strategy
                ]
                if len(chosen) > 1:
                    raise CampaignControlError(
                        f"winner selection may contain at most one {strategy} candidate"
                    )
            updated = engine.advance(record, selected_candidate_ids=selected_ids)
            return _save(
                campaign_store,
                updated,
                "winners_selected",
                selected_candidate_ids=updated.selected_candidate_ids,
            )

        if stage == CampaignStage.BUILD_COMPATIBLE_COMBINATION:
            document = _require_input(stage_input, stage)
            evidence = _evidence(store, record.task_id, document)
            compatibility = CompatibilityState.model_validate(document.get("compatibility"))
            combination = FinalCombinationState.model_validate(
                document.get("final_combination")
            )
            return _save(
                campaign_store,
                engine.advance(
                    record,
                    evidence=evidence,
                    compatibility=compatibility,
                    final_combination=combination,
                ),
                "combination_recorded",
            )

        if stage == CampaignStage.FINAL_VALIDATION:
            document = _require_input(stage_input, stage)
            evidence = _evidence(store, record.task_id, document)
            combination = FinalCombinationState.model_validate(
                document.get("final_combination")
            )
            status = CampaignStatus(document.get("final_status"))
            updated = engine.advance(
                record,
                evidence=evidence,
                final_combination=combination,
                final_status=status,
            )
            result = _save(
                campaign_store,
                updated,
                "final_validation_recorded",
            )
            _write_final_report(store, updated)
            return result

        raise CampaignControlError(f"unsupported campaign stage: {stage}")
    except (
        CampaignError,
        CampaignPlanningError,
        StoreError,
        ValidationError,
        QualityPolicyError,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, CampaignControlError):
            raise
        raise CampaignControlError(str(error)) from error


def campaign_report(record: CampaignRecord) -> dict[str, Any]:
    """Return a compact report that never upgrades provisional quality evidence."""

    return {
        "schema": "gpuopt.campaign-report.v1",
        "campaign_id": record.campaign_id,
        "task_id": record.task_id,
        "stage": record.current_stage.value,
        "status": record.status.value,
        "quality_policy": record.config.quality_policy,
        "quality_protocol_pending_replacement": True,
        "production_ready": False,
        "selected_candidate_ids": record.selected_candidate_ids,
        "candidates": [
            {
                "id": candidate.id,
                "strategy": candidate.strategy.value,
                "disposition": candidate.disposition.value,
                "selected_as_winner": candidate.selected_as_winner,
                "reasons": candidate.reasons,
                "result_artifacts": [
                    artifact.model_dump(mode="json")
                    for artifact in candidate.result_artifacts
                ],
            }
            for candidate in record.candidates
        ],
        "compatibility": (
            record.compatibility.model_dump(mode="json")
            if record.compatibility is not None
            else None
        ),
        "final_combination": (
            record.final_combination.model_dump(mode="json")
            if record.final_combination is not None
            else None
        ),
    }


def _write_final_report(store: ExperimentStore, record: CampaignRecord) -> None:
    report = campaign_report(record)
    store.save_json(
        record.task_id,
        "reports/campaign-final.json",
        report,
        producer="campaign",
    )
    lines = [
        f"# Campaign {record.campaign_id}",
        "",
        f"Status: `{record.status.value}`",
        "",
        "Quality protocol: `provisional-math-100.v1` (100 questions).",
        "",
        "This is an experimental result. The quality protocol is pending replacement, "
        "so it is not production-ready.",
        "",
        "Selected candidates: "
        + (", ".join(record.selected_candidate_ids) or "none"),
        "",
    ]
    store.save_text(
        record.task_id,
        "reports/campaign-final.md",
        "\n".join(lines),
        producer="campaign",
        media_type="text/markdown",
    )


__all__ = [
    "CampaignControlError",
    "CampaignStepResult",
    "campaign_report",
    "run_campaign_step",
]
