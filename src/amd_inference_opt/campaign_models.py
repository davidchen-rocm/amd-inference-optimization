"""Versioned domain records for multi-strategy optimization campaigns."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .control_policy import CampaignControlPolicy, OptimizationLedger
from .eval_suites import QualityPolicyId
from .models import ArtifactRef, OptimizationTask, StrictModel, utc_now


class CampaignStage(StrEnum):
    CREATE_CAMPAIGN = "CREATE_CAMPAIGN"
    INSPECT_TARGET = "INSPECT_TARGET"
    CAPTURE_SHARED_BASELINES = "CAPTURE_SHARED_BASELINES"
    PLAN_CANDIDATES = "PLAN_CANDIDATES"
    RUN_MIXED_BIT = "RUN_MIXED_BIT"
    RUN_SHAPE_KERNEL = "RUN_SHAPE_KERNEL"
    RUN_KV_CACHE = "RUN_KV_CACHE"
    SELECT_WINNERS = "SELECT_WINNERS"
    BUILD_COMPATIBLE_COMBINATION = "BUILD_COMPATIBLE_COMBINATION"
    FINAL_VALIDATION = "FINAL_VALIDATION"
    COMPLETE = "COMPLETE"


class CampaignStatus(StrEnum):
    ACTIVE = "ACTIVE"
    EXPERIMENTAL_ACCEPTED = "EXPERIMENTAL_ACCEPTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class CampaignStrategy(StrEnum):
    MIXED_BIT = "mixed_bit"
    SHAPE_KERNEL = "shape_kernel"
    KV_CACHE = "kv_cache"


class CandidateDisposition(StrEnum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    EXPERIMENTAL_ACCEPTED = "EXPERIMENTAL_ACCEPTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    SKIPPED = "SKIPPED"

    @property
    def terminal(self) -> bool:
        return self not in {self.PLANNED, self.RUNNING}


class CompatibilityDisposition(StrEnum):
    PENDING = "PENDING"
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    INCONCLUSIVE = "INCONCLUSIVE"


class FinalCombinationDisposition(StrEnum):
    PLANNED = "PLANNED"
    BUILT = "BUILT"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


STRATEGY_STAGES: dict[CampaignStrategy, CampaignStage] = {
    CampaignStrategy.MIXED_BIT: CampaignStage.RUN_MIXED_BIT,
    CampaignStrategy.SHAPE_KERNEL: CampaignStage.RUN_SHAPE_KERNEL,
    CampaignStrategy.KV_CACHE: CampaignStage.RUN_KV_CACHE,
}
STAGE_STRATEGIES = {stage: strategy for strategy, stage in STRATEGY_STAGES.items()}


def _safe_identifier(value: str, *, field: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not value or any(character not in allowed for character in value):
        raise ValueError(f"{field} may contain only letters, digits, '-' and '_'")
    return value


class CampaignConfig(StrictModel):
    """Standalone campaign input suitable for JSON or YAML validation."""

    schema_version: Literal[1] = 1
    id: str
    task: OptimizationTask
    quality_policy: QualityPolicyId = "provisional-math-100.v1"
    strategy_order: list[CampaignStrategy] = Field(
        default_factory=lambda: [
            CampaignStrategy.MIXED_BIT,
            CampaignStrategy.SHAPE_KERNEL,
            CampaignStrategy.KV_CACHE,
        ]
    )
    max_candidates: int = Field(default=16, ge=1)
    planning_inputs: dict[str, Any] = Field(default_factory=dict)
    control_policy: CampaignControlPolicy = Field(default_factory=CampaignControlPolicy)

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        return _safe_identifier(value, field="campaign id")

    @field_validator("strategy_order")
    @classmethod
    def unique_nonempty_strategy_order(
        cls, values: list[CampaignStrategy]
    ) -> list[CampaignStrategy]:
        if not values:
            raise ValueError("strategy_order cannot be empty")
        if len(values) != len(set(values)):
            raise ValueError("strategy_order cannot repeat a strategy")
        return values


class SharedBaselineReference(StrictModel):
    """One immutable baseline artifact shared by multiple candidate arms."""

    id: str
    task_id: str
    kind: str
    artifact: ArtifactRef
    coordinate_hash: str | None = None
    description: str | None = None

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        return _safe_identifier(value, field="baseline id")

    @field_validator("kind")
    @classmethod
    def nonempty_kind(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("baseline kind cannot be empty")
        return value

    @field_validator("coordinate_hash")
    @classmethod
    def valid_coordinate_hash(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("coordinate_hash must be a lowercase SHA-256 digest")
        return value


# The shorter spelling is useful in campaign configuration code.
SharedBaselineRef = SharedBaselineReference


class CampaignCandidate(StrictModel):
    id: str
    strategy: CampaignStrategy
    label: str
    shared_baseline_ids: list[str] = Field(min_length=1)
    spec_artifact: ArtifactRef | None = None
    workflow_task_id: str | None = None
    disposition: CandidateDisposition = CandidateDisposition.PLANNED
    result_artifacts: list[ArtifactRef] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    selected_as_winner: bool = False
    content_fingerprint: str | None = None

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        return _safe_identifier(value, field="candidate id")

    @field_validator("label")
    @classmethod
    def nonempty_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate label cannot be empty")
        return value

    @field_validator("shared_baseline_ids")
    @classmethod
    def unique_baseline_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("shared_baseline_ids cannot contain duplicates")
        return values

    @field_validator("content_fingerprint")
    @classmethod
    def valid_content_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("content_fingerprint must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def winner_is_experimentally_accepted(self) -> CampaignCandidate:
        if self.selected_as_winner and (
            self.disposition != CandidateDisposition.EXPERIMENTAL_ACCEPTED
        ):
            raise ValueError("only experimentally accepted candidates may be winners")
        return self


class CompatibilityState(StrictModel):
    disposition: CompatibilityDisposition = CompatibilityDisposition.PENDING
    candidate_ids: list[str] = Field(default_factory=list)
    checks: dict[str, bool | None] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    evidence: list[ArtifactRef] = Field(default_factory=list)

    @field_validator("candidate_ids")
    @classmethod
    def unique_candidates(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("compatibility candidate_ids cannot contain duplicates")
        return values

    @model_validator(mode="after")
    def checks_match_disposition(self) -> CompatibilityState:
        if self.disposition == CompatibilityDisposition.COMPATIBLE and any(
            value is not True for value in self.checks.values()
        ):
            raise ValueError("COMPATIBLE compatibility checks must all pass")
        if self.disposition == CompatibilityDisposition.INCOMPATIBLE and self.checks and all(
            value is not False for value in self.checks.values()
        ):
            raise ValueError("INCOMPATIBLE compatibility requires a failed check")
        return self


class FinalCombinationState(StrictModel):
    disposition: FinalCombinationDisposition = FinalCombinationDisposition.PLANNED
    candidate_ids: list[str] = Field(default_factory=list)
    components: dict[str, str] = Field(default_factory=dict)
    checks: dict[str, bool | None] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)

    @field_validator("candidate_ids")
    @classmethod
    def unique_candidates(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("combination candidate_ids cannot contain duplicates")
        return values

    @model_validator(mode="after")
    def validation_checks_match_disposition(self) -> FinalCombinationState:
        if self.disposition == FinalCombinationDisposition.VALIDATED and any(
            value is not True for value in self.checks.values()
        ):
            raise ValueError("VALIDATED final-combination checks must all pass")
        return self


class CampaignStageCompletion(StrictModel):
    stage: CampaignStage
    evidence: list[ArtifactRef] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=utc_now)


class CampaignRecord(StrictModel):
    schema_version: Literal[1] = 1
    config: CampaignConfig
    current_stage: CampaignStage = CampaignStage.CREATE_CAMPAIGN
    status: CampaignStatus = CampaignStatus.ACTIVE
    shared_baselines: list[SharedBaselineReference] = Field(default_factory=list)
    candidates: list[CampaignCandidate] = Field(default_factory=list)
    selected_candidate_ids: list[str] = Field(default_factory=list)
    compatibility: CompatibilityState | None = None
    final_combination: FinalCombinationState | None = None
    completions: list[CampaignStageCompletion] = Field(default_factory=list)
    platform_evidence: ArtifactRef | None = None
    optimization_ledger: OptimizationLedger = Field(default_factory=OptimizationLedger)
    revision: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def campaign_id(self) -> str:
        return self.config.id

    @property
    def task_id(self) -> str:
        return self.config.task.id

    @model_validator(mode="after")
    def campaign_invariants(self) -> CampaignRecord:
        terminal = self.status != CampaignStatus.ACTIVE
        if terminal != (self.current_stage == CampaignStage.COMPLETE):
            raise ValueError("only COMPLETE campaigns may have a terminal status")

        baseline_ids = [baseline.id for baseline in self.shared_baselines]
        if len(baseline_ids) != len(set(baseline_ids)):
            raise ValueError("shared baseline ids must be unique")
        if any(baseline.task_id != self.task_id for baseline in self.shared_baselines):
            raise ValueError("shared baselines must belong to the campaign task")

        candidate_ids = [candidate.id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("campaign candidate ids must be unique")
        if len(candidate_ids) > self.config.max_candidates:
            raise ValueError("campaign exceeds max_candidates")
        fingerprints = [
            candidate.content_fingerprint
            for candidate in self.candidates
            if candidate.content_fingerprint is not None
        ]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("campaign candidates cannot repeat semantic content")
        known_baselines = set(baseline_ids)
        for candidate in self.candidates:
            if candidate.strategy not in self.config.strategy_order:
                raise ValueError(
                    f"candidate {candidate.id} uses a strategy not configured for the campaign"
                )
            unknown = set(candidate.shared_baseline_ids) - known_baselines
            if unknown:
                raise ValueError(
                    f"candidate {candidate.id} references unknown shared baselines: "
                    + ", ".join(sorted(unknown))
                )

        running = [
            candidate
            for candidate in self.candidates
            if candidate.disposition == CandidateDisposition.RUNNING
        ]
        if len(running) > 1:
            raise ValueError("only one campaign candidate may run at a time")
        current_strategy = STAGE_STRATEGIES.get(self.current_stage)
        if running and running[0].strategy != current_strategy:
            raise ValueError("running candidate does not match the current campaign stage")

        selected = set(self.selected_candidate_ids)
        if len(selected) != len(self.selected_candidate_ids):
            raise ValueError("selected_candidate_ids cannot contain duplicates")
        unknown_selected = selected - set(candidate_ids)
        if unknown_selected:
            raise ValueError("selected_candidate_ids contain unknown candidates")
        for candidate in self.candidates:
            should_be_selected = candidate.id in selected
            if candidate.selected_as_winner != should_be_selected:
                raise ValueError("candidate winner flags must match selected_candidate_ids")
            if should_be_selected and (
                candidate.disposition != CandidateDisposition.EXPERIMENTAL_ACCEPTED
            ):
                raise ValueError("selected candidates must be experimentally accepted")

        if self.compatibility is not None:
            unknown = set(self.compatibility.candidate_ids) - selected
            if unknown:
                raise ValueError("compatibility references candidates not selected as winners")
        if self.final_combination is not None:
            unknown = set(self.final_combination.candidate_ids) - selected
            if unknown:
                raise ValueError("final combination references candidates not selected as winners")
            if self.compatibility is not None and not set(
                self.final_combination.candidate_ids
            ).issubset(self.compatibility.candidate_ids):
                raise ValueError(
                    "final combination candidates must be covered by compatibility state"
                )

        if self.status == CampaignStatus.EXPERIMENTAL_ACCEPTED:
            if not selected:
                raise ValueError("EXPERIMENTAL_ACCEPTED requires a selected candidate")
            if self.compatibility is None or (
                self.compatibility.disposition != CompatibilityDisposition.COMPATIBLE
            ):
                raise ValueError("EXPERIMENTAL_ACCEPTED requires compatible winners")
            if self.final_combination is None or (
                self.final_combination.disposition
                != FinalCombinationDisposition.VALIDATED
            ):
                raise ValueError("EXPERIMENTAL_ACCEPTED requires a validated combination")
            if not self.final_combination.candidate_ids:
                raise ValueError("EXPERIMENTAL_ACCEPTED requires a nonempty combination")
        return self


__all__ = [
    "CampaignCandidate",
    "CampaignConfig",
    "CampaignRecord",
    "CampaignStage",
    "CampaignStageCompletion",
    "CampaignStatus",
    "CampaignStrategy",
    "CandidateDisposition",
    "CompatibilityDisposition",
    "CompatibilityState",
    "FinalCombinationDisposition",
    "FinalCombinationState",
    "SharedBaselineRef",
    "SharedBaselineReference",
    "STAGE_STRATEGIES",
    "STRATEGY_STAGES",
]
