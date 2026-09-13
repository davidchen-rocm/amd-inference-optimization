"""Strict v2 coordinator for closing the gfx1201 optimization phase."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .eval_suites import QualityPolicyId
from .models import ArtifactRef, OptimizationTask, StrictModel, utc_now
from .store import ExperimentStore, StoreError

GFX1201_CAMPAIGN_STATE_PATH = "state/gfx1201-campaign.json"


class Gfx1201Stage(StrEnum):
    CREATE_CAMPAIGN = "CREATE_CAMPAIGN"
    INSPECT_TARGET = "INSPECT_TARGET"
    CAPTURE_SHARED_BASELINES = "CAPTURE_SHARED_BASELINES"
    RUN_MIXED_BIT = "RUN_MIXED_BIT"
    RUN_HIP_GRAPH_AB = "RUN_HIP_GRAPH_AB"
    RUN_MEMORY_AUDIT = "RUN_MEMORY_AUDIT"
    RUN_MFMA_MMQ_EVIDENCE = "RUN_MFMA_MMQ_EVIDENCE"
    BUILD_FINAL_REPORT = "BUILD_FINAL_REPORT"
    COMPLETE = "COMPLETE"


GFX1201_STAGE_ORDER = (
    Gfx1201Stage.CREATE_CAMPAIGN,
    Gfx1201Stage.INSPECT_TARGET,
    Gfx1201Stage.CAPTURE_SHARED_BASELINES,
    Gfx1201Stage.RUN_MIXED_BIT,
    Gfx1201Stage.RUN_HIP_GRAPH_AB,
    Gfx1201Stage.RUN_MEMORY_AUDIT,
    Gfx1201Stage.RUN_MFMA_MMQ_EVIDENCE,
    Gfx1201Stage.BUILD_FINAL_REPORT,
    Gfx1201Stage.COMPLETE,
)


class Gfx1201CampaignStatus(StrEnum):
    ACTIVE = "ACTIVE"
    COMPLETE = "COMPLETE"


class CapabilityOutcome(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"
    MATERIAL_BENEFIT = "MATERIAL_BENEFIT"
    NO_MATERIAL_EFFECT = "NO_MATERIAL_EFFECT"
    HARMFUL = "HARMFUL"
    OPPORTUNITY_FOUND = "OPPORTUNITY_FOUND"
    NO_ACTION = "NO_ACTION"
    EVIDENCE_CLOSED = "EVIDENCE_CLOSED"
    COMPLETE = "COMPLETE"


STAGE_CAPABILITY: dict[Gfx1201Stage, str] = {
    Gfx1201Stage.INSPECT_TARGET: "target_inspection",
    Gfx1201Stage.CAPTURE_SHARED_BASELINES: "shared_baselines",
    Gfx1201Stage.RUN_MIXED_BIT: "mixed_precision",
    Gfx1201Stage.RUN_HIP_GRAPH_AB: "hip_graph_ab",
    Gfx1201Stage.RUN_MEMORY_AUDIT: "memory_reuse_audit",
    Gfx1201Stage.RUN_MFMA_MMQ_EVIDENCE: "mfma_mmq_evidence",
    Gfx1201Stage.BUILD_FINAL_REPORT: "gfx1201_final_report",
}


class Gfx1201CampaignConfig(StrictModel):
    schema_version: Literal[2] = 2
    id: str
    task: OptimizationTask
    quality_policy: QualityPolicyId = "provisional-math-100.v1"
    mixed_bit_max_candidates: int = Field(default=8, ge=1, le=16)

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not value or any(character not in allowed for character in value):
            raise ValueError("campaign id contains unsafe characters")
        return value

    @model_validator(mode="after")
    def target_is_gfx1201(self) -> Gfx1201CampaignConfig:
        if self.task.gpu.gfx_target != "gfx1201":
            raise ValueError("gfx1201 campaign requires task.gpu.gfx_target=gfx1201")
        return self


class CapabilityResult(StrictModel):
    capability: str
    outcome: CapabilityOutcome
    summary: str
    experiment_ids: list[str] = Field(default_factory=list)
    evidence: list[ArtifactRef] = Field(min_length=1)
    missing_evidence: list[str] = Field(default_factory=list)

    @field_validator("capability", "summary")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("capability and summary cannot be empty")
        return value


class Gfx1201StageCompletion(StrictModel):
    stage: Gfx1201Stage
    result: CapabilityResult | None = None
    completed_at: datetime = Field(default_factory=utc_now)


class Gfx1201CampaignRecord(StrictModel):
    schema_version: Literal[2] = 2
    config: Gfx1201CampaignConfig
    current_stage: Gfx1201Stage = Gfx1201Stage.CREATE_CAMPAIGN
    status: Gfx1201CampaignStatus = Gfx1201CampaignStatus.ACTIVE
    completions: list[Gfx1201StageCompletion] = Field(default_factory=list)
    selected_mixed_precision_candidate_id: str | None = None
    revision: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def campaign_id(self) -> str:
        return self.config.id

    @property
    def task_id(self) -> str:
        return self.config.task.id

    @model_validator(mode="after")
    def strict_prefix(self) -> Gfx1201CampaignRecord:
        stages = [completion.stage for completion in self.completions]
        expected = list(GFX1201_STAGE_ORDER[: len(stages)])
        if stages != expected:
            raise ValueError("campaign completions must be an exact stage-order prefix")
        expected_current = (
            GFX1201_STAGE_ORDER[len(stages)]
            if len(stages) < len(GFX1201_STAGE_ORDER)
            else None
        )
        if expected_current is not None and self.current_stage != expected_current:
            raise ValueError("current_stage does not follow completed stage prefix")
        terminal = self.current_stage == Gfx1201Stage.COMPLETE
        if terminal != (self.status == Gfx1201CampaignStatus.COMPLETE):
            raise ValueError("only COMPLETE stage may have COMPLETE status")
        if self.current_stage in {
            Gfx1201Stage.RUN_HIP_GRAPH_AB,
            Gfx1201Stage.RUN_MEMORY_AUDIT,
            Gfx1201Stage.RUN_MFMA_MMQ_EVIDENCE,
            Gfx1201Stage.BUILD_FINAL_REPORT,
            Gfx1201Stage.COMPLETE,
        } and not self.selected_mixed_precision_candidate_id:
            raise ValueError("downstream stages require the selected mixed-precision candidate")
        return self


class Gfx1201CampaignError(RuntimeError):
    pass


class Gfx1201CampaignEngine:
    @staticmethod
    def new(config: Gfx1201CampaignConfig) -> Gfx1201CampaignRecord:
        return Gfx1201CampaignRecord(config=config)

    @staticmethod
    def advance(
        record: Gfx1201CampaignRecord,
        *,
        result: CapabilityResult | None = None,
        selected_mixed_precision_candidate_id: str | None = None,
    ) -> Gfx1201CampaignRecord:
        if record.status != Gfx1201CampaignStatus.ACTIVE:
            raise Gfx1201CampaignError("campaign is already complete")
        stage = record.current_stage
        if stage == Gfx1201Stage.COMPLETE:
            raise Gfx1201CampaignError("COMPLETE cannot advance")
        expected_capability = STAGE_CAPABILITY.get(stage)
        if expected_capability is None:
            if result is not None:
                raise Gfx1201CampaignError("CREATE_CAMPAIGN does not accept capability evidence")
        elif result is None or result.capability != expected_capability:
            raise Gfx1201CampaignError(
                f"{stage} requires CapabilityResult({expected_capability})"
            )

        selected = record.selected_mixed_precision_candidate_id
        if stage == Gfx1201Stage.RUN_MIXED_BIT:
            if result is None or result.outcome != CapabilityOutcome.ACCEPT:
                raise Gfx1201CampaignError(
                    "mixed precision must ACCEPT a candidate before HIP Graph A/B"
                )
            selected = selected_mixed_precision_candidate_id
            if not selected:
                raise Gfx1201CampaignError("accepted mixed precision requires candidate id")
        elif selected_mixed_precision_candidate_id is not None:
            raise Gfx1201CampaignError("candidate selection is valid only at RUN_MIXED_BIT")

        completion = Gfx1201StageCompletion(stage=stage, result=result)
        next_stage = GFX1201_STAGE_ORDER[GFX1201_STAGE_ORDER.index(stage) + 1]
        status = (
            Gfx1201CampaignStatus.COMPLETE
            if next_stage == Gfx1201Stage.COMPLETE
            else Gfx1201CampaignStatus.ACTIVE
        )
        return Gfx1201CampaignRecord(
            config=record.config,
            current_stage=next_stage,
            status=status,
            completions=[*record.completions, completion],
            selected_mixed_precision_candidate_id=selected,
            revision=record.revision + 1,
            updated_at=utc_now(),
        )


class Gfx1201CampaignStore:
    def __init__(self, store: ExperimentStore | str | Path) -> None:
        self.store = store if isinstance(store, ExperimentStore) else ExperimentStore(store)

    def create(self, config: Gfx1201CampaignConfig) -> Gfx1201CampaignRecord:
        if not self.store.task_dir(config.task.id).exists():
            self.store.create_task(config.task)
        elif self.store.load_task(config.task.id) != config.task:
            raise Gfx1201CampaignError("persisted task differs from campaign task")
        record = Gfx1201CampaignEngine.new(config)
        with self.store.task_lock(config.task.id):
            path = self.store.task_dir(config.task.id) / GFX1201_CAMPAIGN_STATE_PATH
            if path.exists() or path.is_symlink():
                raise Gfx1201CampaignError("gfx1201 campaign already exists")
            self.store.save_json(
                config.task.id,
                GFX1201_CAMPAIGN_STATE_PATH,
                record,
                producer="gfx1201-campaign",
            )
        return record

    def load(self, task_id: str) -> Gfx1201CampaignRecord:
        try:
            return self.store.load_json(
                task_id, GFX1201_CAMPAIGN_STATE_PATH, Gfx1201CampaignRecord
            )
        except StoreError as error:
            raise Gfx1201CampaignError(str(error)) from error

    def save(self, record: Gfx1201CampaignRecord) -> ArtifactRef:
        with self.store.task_lock(record.task_id):
            current = self.load(record.task_id)
            if current.config != record.config:
                raise Gfx1201CampaignError("campaign config is immutable")
            if record.revision != current.revision + 1:
                raise Gfx1201CampaignError("campaign revision conflict")
            reference = self.store.save_json(
                record.task_id,
                GFX1201_CAMPAIGN_STATE_PATH,
                record,
                producer="gfx1201-campaign",
            )
            self.store.append_event(
                record.task_id,
                "gfx1201_campaign_saved",
                {
                    "revision": record.revision,
                    "stage": record.current_stage.value,
                    "status": record.status.value,
                },
            )
            return reference


__all__ = [
    "CapabilityOutcome",
    "CapabilityResult",
    "GFX1201_CAMPAIGN_STATE_PATH",
    "GFX1201_STAGE_ORDER",
    "Gfx1201CampaignConfig",
    "Gfx1201CampaignEngine",
    "Gfx1201CampaignError",
    "Gfx1201CampaignRecord",
    "Gfx1201CampaignStatus",
    "Gfx1201CampaignStore",
    "Gfx1201Stage",
]
