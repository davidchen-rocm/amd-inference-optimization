"""Outer state machine coordinating independent optimization strategies."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .campaign_models import (
    STRATEGY_STAGES,
    CampaignCandidate,
    CampaignConfig,
    CampaignRecord,
    CampaignStage,
    CampaignStageCompletion,
    CampaignStatus,
    CandidateDisposition,
    CompatibilityDisposition,
    CompatibilityState,
    FinalCombinationDisposition,
    FinalCombinationState,
    SharedBaselineReference,
)
from .control_policy import (
    AttemptDisposition,
    content_fingerprint,
    record_optimization_attempt,
    select_optimization_stack,
)
from .models import ArtifactRef, DecisionOutcome, GateDecision, utc_now


class CampaignError(RuntimeError):
    """A campaign update violates the outer coordinator contract."""


class CampaignEngine:
    """Coordinate candidate families without changing the inner workflow engine."""

    @staticmethod
    def new(
        config: CampaignConfig,
        *,
        platform_evidence: ArtifactRef | None = None,
    ) -> CampaignRecord:
        return CampaignRecord(config=config, platform_evidence=platform_evidence)

    @staticmethod
    def _completion(
        stage: CampaignStage, evidence: Iterable[ArtifactRef]
    ) -> CampaignStageCompletion:
        return CampaignStageCompletion(stage=stage, evidence=list(evidence))

    @staticmethod
    def _activate_next_candidate(
        candidates: list[CampaignCandidate], stage: CampaignStage
    ) -> list[CampaignCandidate]:
        strategy = next(
            (
                strategy
                for strategy, strategy_stage in STRATEGY_STAGES.items()
                if strategy_stage == stage
            ),
            None,
        )
        if strategy is None:
            return candidates
        if any(
            candidate.disposition == CandidateDisposition.RUNNING
            for candidate in candidates
        ):
            return candidates
        for index, candidate in enumerate(candidates):
            if (
                candidate.strategy == strategy
                and candidate.disposition == CandidateDisposition.PLANNED
            ):
                updated = list(candidates)
                updated[index] = candidate.model_copy(
                    update={"disposition": CandidateDisposition.RUNNING}
                )
                return updated
        return candidates

    @staticmethod
    def _next_strategy_stage(record: CampaignRecord) -> CampaignStage | None:
        current_strategy = next(
            (
                strategy
                for strategy, stage in STRATEGY_STAGES.items()
                if stage == record.current_stage
            ),
            None,
        )
        if current_strategy is None:
            return STRATEGY_STAGES[record.config.strategy_order[0]]
        position = record.config.strategy_order.index(current_strategy)
        if position + 1 == len(record.config.strategy_order):
            return None
        return STRATEGY_STAGES[record.config.strategy_order[position + 1]]

    @staticmethod
    def _updated_record(
        record: CampaignRecord, updates: dict[str, object]
    ) -> CampaignRecord:
        return CampaignRecord.model_validate(
            {
                **record.model_dump(),
                **updates,
                "revision": record.revision + 1,
                "updated_at": utc_now(),
            }
        )

    def advance(
        self,
        record: CampaignRecord,
        *,
        evidence: Iterable[ArtifactRef] = (),
        shared_baselines: Sequence[SharedBaselineReference] = (),
        candidates: Sequence[CampaignCandidate] = (),
        selected_candidate_ids: Sequence[str] | None = None,
        compatibility: CompatibilityState | None = None,
        final_combination: FinalCombinationState | None = None,
        final_status: CampaignStatus | None = None,
    ) -> CampaignRecord:
        """Complete the current outer stage and enter its deterministic successor."""

        if record.status != CampaignStatus.ACTIVE:
            raise CampaignError(f"campaign is terminal: {record.status}")
        stage = record.current_stage
        if stage == CampaignStage.COMPLETE:
            raise CampaignError("complete campaigns cannot advance")
        evidence_items = list(evidence)
        if stage == CampaignStage.INSPECT_TARGET and not evidence_items:
            raise CampaignError("INSPECT_TARGET requires persisted evidence")
        if stage == CampaignStage.CAPTURE_SHARED_BASELINES:
            evidence_items.extend(baseline.artifact for baseline in shared_baselines)
        if stage == CampaignStage.PLAN_CANDIDATES and not evidence_items:
            raise CampaignError("PLAN_CANDIDATES requires a serialized plan artifact")
        if stage in STRATEGY_STAGES.values():
            strategy = next(
                strategy
                for strategy, strategy_stage in STRATEGY_STAGES.items()
                if strategy_stage == stage
            )
            evidence_items.extend(
                artifact
                for candidate in record.candidates
                if candidate.strategy == strategy and candidate.disposition.terminal
                for artifact in candidate.result_artifacts
            )
            if not evidence_items:
                raise CampaignError(f"{stage} requires candidate-result evidence")
        if stage in {
            CampaignStage.BUILD_COMPATIBLE_COMBINATION,
            CampaignStage.FINAL_VALIDATION,
        } and not evidence_items:
            raise CampaignError(f"{stage} requires persisted evidence")
        completions = [
            *record.completions,
            self._completion(stage, evidence_items),
        ]
        updates: dict[str, object] = {"completions": completions}

        if stage == CampaignStage.CREATE_CAMPAIGN:
            updates["current_stage"] = CampaignStage.INSPECT_TARGET
        elif stage == CampaignStage.INSPECT_TARGET:
            updates["current_stage"] = CampaignStage.CAPTURE_SHARED_BASELINES
        elif stage == CampaignStage.CAPTURE_SHARED_BASELINES:
            combined = [*record.shared_baselines, *shared_baselines]
            if not combined:
                raise CampaignError("shared baseline capture requires at least one reference")
            updates.update(
                {
                    "shared_baselines": combined,
                    "current_stage": CampaignStage.PLAN_CANDIDATES,
                }
            )
        elif stage == CampaignStage.PLAN_CANDIDATES:
            planned = [*record.candidates, *candidates]
            if not planned:
                raise CampaignError("candidate planning requires at least one candidate")
            if any(
                candidate.disposition != CandidateDisposition.PLANNED
                for candidate in planned
            ):
                raise CampaignError("new campaign candidates must be PLANNED")
            next_stage = self._next_strategy_stage(record)
            assert next_stage is not None
            planned = self._activate_next_candidate(planned, next_stage)
            updates.update({"candidates": planned, "current_stage": next_stage})
        elif stage in STRATEGY_STAGES.values():
            strategy = next(
                strategy
                for strategy, strategy_stage in STRATEGY_STAGES.items()
                if strategy_stage == stage
            )
            unfinished = [
                candidate.id
                for candidate in record.candidates
                if candidate.strategy == strategy
                and not candidate.disposition.terminal
            ]
            if unfinished:
                raise CampaignError(
                    f"{stage} has unfinished candidates: " + ", ".join(unfinished)
                )
            next_stage = self._next_strategy_stage(record)
            if next_stage is None:
                updates["current_stage"] = CampaignStage.SELECT_WINNERS
            else:
                updates.update(
                    {
                        "candidates": self._activate_next_candidate(
                            list(record.candidates), next_stage
                        ),
                        "current_stage": next_stage,
                    }
                )
        elif stage == CampaignStage.SELECT_WINNERS:
            selected = (
                list(selected_candidate_ids)
                if selected_candidate_ids is not None
                else [
                    candidate.id
                    for candidate in record.candidates
                    if candidate.disposition
                    == CandidateDisposition.EXPERIMENTAL_ACCEPTED
                ]
            )
            candidates_with_winners = [
                candidate.model_copy(
                    update={"selected_as_winner": candidate.id in selected}
                )
                for candidate in record.candidates
            ]
            try:
                optimization_ledger = select_optimization_stack(
                    record.optimization_ledger,
                    selected,
                )
            except ValueError as error:
                raise CampaignError(str(error)) from error
            updates.update(
                {
                    "candidates": candidates_with_winners,
                    "selected_candidate_ids": selected,
                    "optimization_ledger": optimization_ledger,
                    "current_stage": CampaignStage.BUILD_COMPATIBLE_COMBINATION,
                }
            )
        elif stage == CampaignStage.BUILD_COMPATIBLE_COMBINATION:
            if not record.selected_candidate_ids:
                compatibility = compatibility or CompatibilityState(
                    disposition=CompatibilityDisposition.INCOMPATIBLE,
                    reasons=["no experimentally accepted candidate was selected"],
                )
                final_combination = final_combination or FinalCombinationState(
                    disposition=FinalCombinationDisposition.REJECTED,
                    reasons=["no experimentally accepted component is available"],
                )
            if compatibility is None or final_combination is None:
                raise CampaignError(
                    "combination build requires compatibility and final-combination state"
                )
            if compatibility.disposition == CompatibilityDisposition.PENDING:
                raise CampaignError("compatibility must be resolved before final validation")
            if (
                compatibility.disposition == CompatibilityDisposition.COMPATIBLE
                and final_combination.disposition
                != FinalCombinationDisposition.BUILT
            ):
                raise CampaignError("compatible winners require a BUILT final combination")
            updates.update(
                {
                    "compatibility": compatibility,
                    "final_combination": final_combination,
                    "current_stage": CampaignStage.FINAL_VALIDATION,
                }
            )
        elif stage == CampaignStage.FINAL_VALIDATION:
            combination = final_combination or record.final_combination
            if combination is None:
                raise CampaignError("final validation requires final-combination state")
            if final_status is None or final_status == CampaignStatus.ACTIVE:
                raise CampaignError("final validation requires a terminal campaign status")
            expected_combination = {
                CampaignStatus.EXPERIMENTAL_ACCEPTED: (
                    FinalCombinationDisposition.VALIDATED
                ),
                CampaignStatus.REJECTED: FinalCombinationDisposition.REJECTED,
                CampaignStatus.INCONCLUSIVE: FinalCombinationDisposition.INCONCLUSIVE,
            }[final_status]
            if combination.disposition != expected_combination:
                raise CampaignError(
                    f"{final_status} requires final combination {expected_combination}"
                )
            updates.update(
                {
                    "final_combination": combination,
                    "status": final_status,
                    "current_stage": CampaignStage.COMPLETE,
                }
            )
        else:  # pragma: no cover - enum exhaustiveness guard
            raise CampaignError(f"unsupported campaign stage: {stage}")

        try:
            return self._updated_record(record, updates)
        except ValueError as error:
            raise CampaignError(str(error)) from error

    def record_candidate_result(
        self,
        record: CampaignRecord,
        candidate_id: str,
        disposition: CandidateDisposition | DecisionOutcome | GateDecision,
        *,
        artifacts: Sequence[ArtifactRef] = (),
        reasons: Sequence[str] = (),
    ) -> CampaignRecord:
        """Record one terminal arm and immediately activate the next planned peer."""

        if record.status != CampaignStatus.ACTIVE:
            raise CampaignError(f"campaign is terminal: {record.status}")
        strategy = next(
            (
                strategy
                for strategy, stage in STRATEGY_STAGES.items()
                if stage == record.current_stage
            ),
            None,
        )
        if strategy is None:
            raise CampaignError("candidate results are valid only in RUN_* stages")
        gate_reasons: Sequence[str] = ()
        if isinstance(disposition, GateDecision):
            gate_reasons = disposition.reasons
            disposition = disposition.outcome
        if isinstance(disposition, DecisionOutcome):
            disposition = {
                DecisionOutcome.ACCEPT: CandidateDisposition.EXPERIMENTAL_ACCEPTED,
                DecisionOutcome.REJECT: CandidateDisposition.REJECTED,
                DecisionOutcome.INCONCLUSIVE: CandidateDisposition.INCONCLUSIVE,
            }[disposition]
        else:
            try:
                disposition = CandidateDisposition(disposition)
            except ValueError as error:
                raise CampaignError(f"invalid candidate disposition: {disposition}") from error
        if not disposition.terminal:
            raise CampaignError("record_candidate_result requires a terminal disposition")

        candidates = list(record.candidates)
        index = next(
            (
                position
                for position, candidate in enumerate(candidates)
                if candidate.id == candidate_id
            ),
            None,
        )
        if index is None:
            raise CampaignError(f"unknown campaign candidate: {candidate_id}")
        candidate = candidates[index]
        if candidate.strategy != strategy:
            raise CampaignError("candidate strategy does not match the current RUN_* stage")
        if candidate.disposition != CandidateDisposition.RUNNING:
            raise CampaignError("only the RUNNING candidate may record a result")
        candidates[index] = candidate.model_copy(
            update={
                "disposition": disposition,
                "result_artifacts": [*candidate.result_artifacts, *artifacts],
                "reasons": [*candidate.reasons, *gate_reasons, *reasons],
            }
        )
        action_name = {
            "mixed_bit": "mixed_bit",
            "shape_kernel": "shape_kernel",
            "kv_cache": "kv_cache",
        }[strategy.value]
        fingerprint = candidate.content_fingerprint
        if fingerprint is None:
            fingerprint = content_fingerprint(
                action_name,
                {
                    "strategy": strategy.value,
                    "spec_sha256": (
                        candidate.spec_artifact.sha256
                        if candidate.spec_artifact is not None
                        else None
                    ),
                    "legacy_candidate_id": candidate.id,
                },
            )
            candidates[index] = candidates[index].model_copy(
                update={"content_fingerprint": fingerprint}
            )
        ledger = record.optimization_ledger
        if disposition != CandidateDisposition.SKIPPED:
            attempt_disposition = {
                CandidateDisposition.EXPERIMENTAL_ACCEPTED: AttemptDisposition.ACCEPTED,
                CandidateDisposition.REJECTED: AttemptDisposition.REJECTED,
                CandidateDisposition.INCONCLUSIVE: AttemptDisposition.INCONCLUSIVE,
            }[disposition]
            attempt_evidence = [*artifacts]
            if candidate.spec_artifact is not None:
                attempt_evidence.insert(0, candidate.spec_artifact)
            try:
                ledger = record_optimization_attempt(
                    ledger,
                    candidate_id=candidate.id,
                    action_name=action_name,
                    fingerprint=fingerprint,
                    disposition=attempt_disposition,
                    evidence=attempt_evidence,
                )
            except ValueError as error:
                raise CampaignError(str(error)) from error
        candidates = self._activate_next_candidate(candidates, record.current_stage)
        try:
            return self._updated_record(
                record,
                {"candidates": candidates, "optimization_ledger": ledger},
            )
        except ValueError as error:
            raise CampaignError(str(error)) from error


__all__ = ["CampaignEngine", "CampaignError"]
