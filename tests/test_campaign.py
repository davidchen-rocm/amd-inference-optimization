from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from amd_inference_opt.campaign import CampaignEngine, CampaignError
from amd_inference_opt.campaign_control import run_campaign_step
from amd_inference_opt.campaign_models import (
    CampaignCandidate,
    CampaignConfig,
    CampaignRecord,
    CampaignStage,
    CampaignStatus,
    CampaignStrategy,
    CandidateDisposition,
    CompatibilityDisposition,
    CompatibilityState,
    FinalCombinationDisposition,
    FinalCombinationState,
    SharedBaselineReference,
)
from amd_inference_opt.campaign_store import (
    CAMPAIGN_STATE_PATH,
    CampaignStore,
    CampaignStoreError,
)
from amd_inference_opt.models import (
    ArtifactRef,
    DecisionOutcome,
    MCPConfig,
    ModelTarget,
    OptimizationTask,
    RuntimeTarget,
)


def artifact(name: str, digest: str = "a") -> ArtifactRef:
    return ArtifactRef(
        path=f"artifacts/evidence/{name}/v000001.json",
        sha256=digest * 64,
        size=2,
        producer="test",
        media_type="application/json",
    )


def task(tmp_path: Path) -> OptimizationTask:
    return OptimizationTask(
        id="campaign-task",
        model=ModelTarget(path=tmp_path / "model.gguf", sha256="1" * 64),
        runtime=RuntimeTarget(repo_path=tmp_path / "llama.cpp", base_commit="abc"),
        mcp=MCPConfig(command=["rocm-agent-mcp"]),
    )


def config(
    tmp_path: Path, *, strategies: list[CampaignStrategy] | None = None
) -> CampaignConfig:
    values: dict[str, object] = {"id": "consumer-amd", "task": task(tmp_path)}
    if strategies is not None:
        values["strategy_order"] = strategies
    return CampaignConfig(**values)


def baseline() -> SharedBaselineReference:
    return SharedBaselineReference(
        id="q6-shared",
        task_id="campaign-task",
        kind="decode-and-quality",
        artifact=artifact("baseline"),
        coordinate_hash="2" * 64,
    )


def candidate(identifier: str, strategy: CampaignStrategy) -> CampaignCandidate:
    return CampaignCandidate(
        id=identifier,
        strategy=strategy,
        label=identifier.replace("-", " "),
        shared_baseline_ids=["q6-shared"],
    )


def planned_record(tmp_path: Path) -> CampaignRecord:
    engine = CampaignEngine()
    record = engine.new(config(tmp_path))
    record = engine.advance(record)
    record = engine.advance(record, evidence=[artifact("inspection", "b")])
    record = engine.advance(record, shared_baselines=[baseline()])
    return engine.advance(
        record,
        evidence=[artifact("candidate-plan", "c")],
        candidates=[
            candidate("mixed-q5", CampaignStrategy.MIXED_BIT),
            candidate("mixed-q4", CampaignStrategy.MIXED_BIT),
            candidate("shape-a", CampaignStrategy.SHAPE_KERNEL),
            candidate("kv-q8", CampaignStrategy.KV_CACHE),
        ],
    )


def test_campaign_config_is_strict_standalone_and_has_full_capacity(
    tmp_path: Path,
) -> None:
    configuration = config(tmp_path)
    restored = CampaignConfig.model_validate_json(configuration.model_dump_json())

    assert restored == configuration
    assert configuration.quality_policy == "provisional-math-100.v1"
    assert configuration.strategy_order == [
        CampaignStrategy.MIXED_BIT,
        CampaignStrategy.SHAPE_KERNEL,
        CampaignStrategy.KV_CACHE,
    ]
    assert configuration.max_candidates == 16
    balanced = CampaignConfig.model_validate(
        {**configuration.model_dump(), "quality_policy": "balanced-200.v1"}
    )
    assert balanced.quality_policy == "balanced-200.v1"
    with pytest.raises(ValidationError, match="provisional-math-100"):
        CampaignConfig.model_validate(
            {
                **configuration.model_dump(),
                "quality_policy": "standard-accept.v1",
            }
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CampaignConfig.model_validate({**configuration.model_dump(), "extra": True})


def test_rejected_candidate_continues_to_next_arm_and_strategy(tmp_path: Path) -> None:
    engine = CampaignEngine()
    record = planned_record(tmp_path)

    assert record.current_stage == CampaignStage.RUN_MIXED_BIT
    assert record.candidates[0].disposition == CandidateDisposition.RUNNING
    record = engine.record_candidate_result(
        record,
        "mixed-q5",
        DecisionOutcome.ACCEPT,
        artifacts=[artifact("q5-result", "d")],
    )
    assert record.status == CampaignStatus.ACTIVE
    assert record.candidates[0].disposition == CandidateDisposition.EXPERIMENTAL_ACCEPTED
    assert record.candidates[1].disposition == CandidateDisposition.RUNNING

    record = engine.record_candidate_result(
        record,
        "mixed-q4",
        CandidateDisposition.REJECTED,
        artifacts=[artifact("q4-result", "e")],
        reasons=["quality regression"],
    )
    assert record.current_stage == CampaignStage.RUN_MIXED_BIT
    record = engine.advance(record)
    assert record.current_stage == CampaignStage.RUN_SHAPE_KERNEL
    assert next(item for item in record.candidates if item.id == "shape-a").disposition == (
        CandidateDisposition.RUNNING
    )

    record = engine.record_candidate_result(
        record,
        "shape-a",
        DecisionOutcome.REJECT,
        artifacts=[artifact("shape-result", "f")],
    )
    record = engine.advance(record)
    assert record.current_stage == CampaignStage.RUN_KV_CACHE
    record = engine.record_candidate_result(
        record,
        "kv-q8",
        DecisionOutcome.REJECT,
        artifacts=[artifact("kv-result", "0")],
    )
    record = engine.advance(record)
    assert record.current_stage == CampaignStage.SELECT_WINNERS

    record = engine.advance(record)
    assert record.selected_candidate_ids == ["mixed-q5"]
    assert record.current_stage == CampaignStage.BUILD_COMPATIBLE_COMBINATION
    compatibility = CompatibilityState(
        disposition=CompatibilityDisposition.COMPATIBLE,
        candidate_ids=["mixed-q5"],
        checks={"model_runtime": True, "kv_cache": True},
        evidence=[artifact("compatibility", "3")],
    )
    combination = FinalCombinationState(
        disposition=FinalCombinationDisposition.BUILT,
        candidate_ids=["mixed-q5"],
        components={"model": "mixed-q5", "runtime": "baseline"},
        artifacts=[artifact("combination", "4")],
    )
    record = engine.advance(
        record,
        evidence=[artifact("compatibility-build", "5")],
        compatibility=compatibility,
        final_combination=combination,
    )
    validated = combination.model_copy(
        update={
            "disposition": FinalCombinationDisposition.VALIDATED,
            "checks": {"performance": True, "quality": True},
        }
    )
    record = engine.advance(
        record,
        evidence=[artifact("final-validation", "6")],
        final_combination=validated,
        final_status=CampaignStatus.EXPERIMENTAL_ACCEPTED,
    )

    assert record.current_stage == CampaignStage.COMPLETE
    assert record.status == CampaignStatus.EXPERIMENTAL_ACCEPTED
    assert record.final_combination == validated
    assert "EXPERIMENTAL_ACCEPTED" not in {outcome.value for outcome in DecisionOutcome}


def test_control_plane_records_evidence_bound_candidate_skip(tmp_path: Path) -> None:
    campaign_store = CampaignStore(tmp_path / "skip-store")
    configuration = config(tmp_path, strategies=[CampaignStrategy.MIXED_BIT])
    record = campaign_store.create(configuration)
    engine = CampaignEngine()
    record = engine.advance(record)
    campaign_store.save(record)
    inspection = campaign_store.store.save_evidence_json(
        record.task_id, "inspection", {"gpu": "gfx1201"}, producer="test"
    )
    record = engine.advance(record, evidence=[inspection])
    campaign_store.save(record)
    baseline_ref = campaign_store.store.save_evidence_json(
        record.task_id, "baseline", {"tg128": 60}, producer="test"
    )
    shared = SharedBaselineReference(
        id="q6-shared",
        task_id=record.task_id,
        kind="decode",
        artifact=baseline_ref,
    )
    record = engine.advance(record, shared_baselines=[shared])
    campaign_store.save(record)
    skipped_candidate = CampaignCandidate(
        id="mixed-inapplicable",
        strategy=CampaignStrategy.MIXED_BIT,
        label="inapplicable",
        shared_baseline_ids=[shared.id],
    )
    plan_ref = campaign_store.store.save_evidence_json(
        record.task_id, "plan", {"candidate": skipped_candidate.id}, producer="test"
    )
    record = engine.advance(record, evidence=[plan_ref], candidates=[skipped_candidate])
    campaign_store.save(record)
    applicability = campaign_store.store.save_evidence_json(
        record.task_id,
        "applicability",
        {"applicable": False, "reason": "tensor precision mismatch"},
        producer="test",
    )

    result = run_campaign_step(
        record,
        campaign_store,
        stage_input={
            "candidate_id": "mixed-inapplicable",
            "disposition": "SKIPPED",
            "evidence_paths": [applicability.path],
            "reasons": ["candidate does not target the selected tensor precision"],
        },
    )

    candidate_result = result.record.candidates[0]
    assert result.action == "candidate_skipped"
    assert candidate_result.disposition == CandidateDisposition.SKIPPED
    assert candidate_result.result_artifacts == [applicability]


def test_no_accepted_candidate_reaches_deterministic_reject_without_build(
    tmp_path: Path,
) -> None:
    engine = CampaignEngine()
    configuration = config(tmp_path, strategies=[CampaignStrategy.MIXED_BIT])
    record = engine.new(configuration)
    record = engine.advance(record)
    record = engine.advance(record, evidence=[artifact("inspection", "b")])
    record = engine.advance(record, shared_baselines=[baseline()])
    record = engine.advance(
        record,
        evidence=[artifact("candidate-plan", "c")],
        candidates=[candidate("mixed-q4", CampaignStrategy.MIXED_BIT)],
    )
    record = engine.record_candidate_result(
        record,
        "mixed-q4",
        CandidateDisposition.REJECTED,
        artifacts=[artifact("q4-reject", "d")],
    )
    record = engine.advance(record)
    record = engine.advance(record)

    assert record.selected_candidate_ids == []
    record = engine.advance(record, evidence=[artifact("no-winner", "e")])
    assert record.compatibility is not None
    assert record.compatibility.disposition == CompatibilityDisposition.INCOMPATIBLE
    assert record.final_combination is not None
    assert record.final_combination.disposition == FinalCombinationDisposition.REJECTED
    record = engine.advance(
        record,
        evidence=[artifact("final-reject", "f")],
        final_status=CampaignStatus.REJECTED,
    )
    assert record.status == CampaignStatus.REJECTED
    assert record.current_stage == CampaignStage.COMPLETE


def test_campaign_requires_stage_evidence_and_known_shared_baselines(
    tmp_path: Path,
) -> None:
    engine = CampaignEngine()
    record = engine.advance(engine.new(config(tmp_path)))

    with pytest.raises(CampaignError, match="INSPECT_TARGET requires"):
        engine.advance(record)
    invalid_candidate = candidate("mixed-q5", CampaignStrategy.MIXED_BIT)
    with pytest.raises(ValidationError, match="unknown shared baselines"):
        CampaignRecord(
            config=config(tmp_path),
            candidates=[invalid_candidate],
        )


def test_campaign_store_create_load_save_and_reject_stale_revision(
    tmp_path: Path,
) -> None:
    campaign_store = CampaignStore(tmp_path / "store")
    configuration = config(tmp_path)

    created = campaign_store.create(configuration)
    advanced = CampaignEngine().advance(created)
    state_ref = campaign_store.save(advanced)

    assert campaign_store.load(configuration.task.id) == advanced
    assert campaign_store.load(configuration.id) == advanced
    assert state_ref.path == CAMPAIGN_STATE_PATH
    state_path = (
        campaign_store.store.task_dir(configuration.task.id) / CAMPAIGN_STATE_PATH
    )
    assert json.loads(state_path.read_text(encoding="utf-8"))["revision"] == 1
    with pytest.raises(CampaignStoreError, match="revision conflict"):
        campaign_store.save(advanced)
    with pytest.raises(CampaignStoreError, match="already exists"):
        campaign_store.create(configuration)
