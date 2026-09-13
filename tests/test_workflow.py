import pytest

from amd_inference_opt.agent_handoff import (
    AgentDecisionError,
    build_agent_context,
    load_current_execution_map,
    persist_execution_map_updates,
    validate_agent_decision,
)
from amd_inference_opt.models import (
    AgentDecision,
    ArtifactRef,
    DecisionOutcome,
    EvidenceRef,
    ExecutionMapEntry,
    GateDecision,
    Hypothesis,
    MCPConfig,
    ModelTarget,
    OptimizationTask,
    RuntimeTarget,
    WorkflowStage,
    WorkflowStatus,
)
from amd_inference_opt.store import ExperimentStore
from amd_inference_opt.workflow import WorkflowEngine, WorkflowError


def test_workflow_requires_evidence_and_cannot_skip() -> None:
    engine = WorkflowEngine()
    record = engine.new("task")

    with pytest.raises(WorkflowError, match="missing evidence"):
        engine.complete_stage(record, {})
    with pytest.raises(WorkflowError, match="illegal transition"):
        engine.complete_stage(
            record,
            {"task": "task.json"},
            requested_next_stage=WorkflowStage.CAPTURE_BASELINE,
        )

    advanced = engine.complete_stage(record, {"task": "task.json"})
    assert advanced.current_stage == WorkflowStage.INSPECT_TARGET
    assert record.current_stage == WorkflowStage.CREATE_TASK


def test_agent_analysis_can_request_deeper_profile_but_not_arbitrary_stage() -> None:
    engine = WorkflowEngine()
    record = engine.new("task").model_copy(
        update={"current_stage": WorkflowStage.CLASSIFY_BOTTLENECK}
    )
    evidence = {"agent_decision": "decision.json", "bottleneck_assessment": "b.json"}

    deeper = engine.complete_stage(
        record, evidence, requested_next_stage=WorkflowStage.DISCOVER_HOTSPOTS
    )
    assert deeper.current_stage == WorkflowStage.DISCOVER_HOTSPOTS
    with pytest.raises(WorkflowError, match="illegal transition"):
        engine.complete_stage(
            record, evidence, requested_next_stage=WorkflowStage.CREATE_EXPERIMENT
        )


def test_decide_accepts_or_loops_rejected_experiment() -> None:
    engine = WorkflowEngine()
    at_decide = engine.new("task").model_copy(update={"current_stage": WorkflowStage.DECIDE})
    rejected = GateDecision(
        outcome=DecisionOutcome.REJECT, checks=[], reasons=["slower"]
    )
    loop = engine.complete_stage(
        at_decide,
        {"gate_decision": "decision.json"},
        gate_decision=rejected,
        can_continue_after_reject=True,
    )
    assert loop.status == WorkflowStatus.ACTIVE
    assert loop.current_stage == WorkflowStage.GENERATE_HYPOTHESIS

    accepted = GateDecision(
        outcome=DecisionOutcome.ACCEPT, checks=[], reasons=["all gates pass"]
    )
    terminal = engine.complete_stage(
        at_decide,
        {"gate_decision": "decision.json"},
        gate_decision=accepted,
    )
    assert terminal.status == WorkflowStatus.ACCEPTED
    assert terminal.terminal_decision == accepted
    with pytest.raises(WorkflowError, match="terminal"):
        engine.complete_stage(terminal, {"gate_decision": "other.json"})


def test_agent_handoff_rejects_uncited_evidence_and_illegal_context(tmp_path) -> None:
    task = OptimizationTask(
        id="agent-task",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(repo_path=tmp_path / "runtime", base_commit="abc"),
        mcp=MCPConfig(command=["agent"]),
    )
    record = WorkflowEngine.new(task.id).model_copy(
        update={"current_stage": WorkflowStage.GENERATE_HYPOTHESIS}
    )
    evidence = [EvidenceRef(id="kernel-1", kind="kernel", summary="GEMV hotspot")]
    hypothesis = Hypothesis(
        observed_evidence_ids=["kernel-1"],
        interpretation="weight traffic dominates",
        proposed_change="reduce weight bytes",
        expected_performance_signature="kernel time falls",
        expected_e2e_effect="decode improves",
        required_validation=["e2e", "quality"],
        stop_condition="no improvement",
    )
    decision = AgentDecision(
        current_stage=WorkflowStage.GENERATE_HYPOTHESIS,
        evidence_used=["kernel-1"],
        conclusion="test a smaller representation",
        confidence=0.8,
        hypothesis=hypothesis,
        proposed_next_stage=WorkflowStage.CREATE_EXPERIMENT,
    )

    context = build_agent_context(task, record, evidence)
    assert "properties" in context.decision_schema
    assert validate_agent_decision(task, record, decision, evidence) == decision

    ungrounded = decision.model_copy(update={"evidence_used": ["invented"]})
    with pytest.raises(AgentDecisionError, match="unknown evidence"):
        validate_agent_decision(task, record, ungrounded, evidence)


def test_stage_completion_can_require_path_and_hash_binding() -> None:
    engine = WorkflowEngine()
    record = engine.new("task")
    artifact = ArtifactRef(
        path="artifacts/evidence/task/v000001.json",
        sha256="a" * 64,
        size=2,
        producer="test",
    )

    advanced = engine.complete_stage_from_artifacts(record, {"task": artifact})

    completion = advanced.completions[-1]
    assert completion.evidence_ids["task"] == artifact.path
    assert completion.evidence_artifacts["task"].sha256 == "a" * 64
    with pytest.raises(WorkflowError, match="not hash-bound"):
        engine.complete_stage(
            record,
            {"task": artifact.path},
            require_artifact_bindings=True,
        )


def test_inconclusive_workflow_resumes_at_gate_selected_stage() -> None:
    engine = WorkflowEngine()
    at_decide = engine.new("task").model_copy(
        update={"current_stage": WorkflowStage.DECIDE}
    )
    gate = GateDecision(
        outcome=DecisionOutcome.INCONCLUSIVE,
        checks=[],
        reasons=["benchmark noise"],
        rerun_from_stage=WorkflowStage.E2E_VALIDATION,
    )
    terminal = engine.complete_stage(
        at_decide,
        {"gate_decision": "gate.json"},
        gate_decision=gate,
    )

    resumed = engine.resume_inconclusive(terminal)

    assert resumed.status == WorkflowStatus.ACTIVE
    assert resumed.current_stage == WorkflowStage.E2E_VALIDATION
    assert resumed.terminal_decision is None
    assert resumed.rerun_count == 1


def test_proven_performance_rejection_skips_quality_without_fake_evidence() -> None:
    engine = WorkflowEngine()
    record = engine.new("task").model_copy(
        update={"current_stage": WorkflowStage.E2E_VALIDATION}
    )
    e2e = ArtifactRef(
        path="artifacts/evidence/e2e/v000001.json",
        sha256="a" * 64,
        size=1,
        producer="runner",
    )
    gate_artifact = ArtifactRef(
        path="artifacts/evidence/gate/v000001.json",
        sha256="b" * 64,
        size=1,
        producer="gate",
    )
    gate = GateDecision(
        outcome=DecisionOutcome.REJECT,
        checks=[],
        reasons=["old mapping is slower"],
    )

    rejected = engine.complete_performance_rejection(
        record,
        e2e_result=e2e,
        gate_decision_artifact=gate_artifact,
        gate_decision=gate,
    )

    assert rejected.status == WorkflowStatus.REJECTED
    assert [item.stage for item in rejected.completions] == [
        WorkflowStage.E2E_VALIDATION,
        WorkflowStage.DECIDE,
    ]
    assert WorkflowStage.QUALITY_VALIDATION not in {
        item.stage for item in rejected.completions
    }


def test_agent_execution_map_updates_are_merged_and_versioned(tmp_path) -> None:
    task = OptimizationTask(
        id="map-task",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(repo_path=tmp_path / "runtime", base_commit="abc"),
        mcp=MCPConfig(command=["agent"]),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    first_entry = ExecutionMapEntry(
        id="decode-gemv",
        phase="decode",
        operator="gate_proj",
        evidence_ids=["kernel-1"],
    )
    first, first_artifact = persist_execution_map_updates(
        store, task.id, [first_entry]
    )
    replacement = first_entry.model_copy(update={"hardware_behavior": "memory bound"})
    second, second_artifact = persist_execution_map_updates(
        store, task.id, [replacement], current=first
    )

    assert first_artifact.path != second_artifact.path
    assert len(second.entries) == 1
    assert second.entries[0].hardware_behavior == "memory bound"
    assert load_current_execution_map(store, task.id) == second
