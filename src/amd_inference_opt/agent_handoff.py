"""Structured, provider-neutral handoff between the workflow and an optimization agent."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import ValidationError

from .models import (
    AgentContext,
    AgentDecision,
    ArtifactRef,
    EvidenceRef,
    ExecutionMapEntry,
    InferenceExecutionMap,
    OptimizationTask,
    ProfileLevel,
    WorkflowRecord,
    WorkflowStage,
)
from .store import ExperimentStore, StoreError
from .workflow import STAGE_REQUIREMENTS, WorkflowEngine


class AgentDecisionError(ValueError):
    """Raised when an agent output is valid JSON but unsafe for this workflow context."""


_PROFILE_ORDER = list(ProfileLevel)


def artifact_evidence_id(artifact: ArtifactRef) -> str:
    """Return the immutable identity Agents must cite, not a mutable path alone."""

    return f"{artifact.path}@sha256:{artifact.sha256}"


def artifact_is_agent_evidence(artifact: ArtifactRef) -> bool:
    """Exclude mutable workflow/control state from the Agent evidence catalog."""

    return artifact.path.startswith(("artifacts/", "experiments/", "reports/"))


def build_agent_context(
    task: OptimizationTask,
    record: WorkflowRecord,
    evidence: Iterable[EvidenceRef],
    *,
    missing_evidence: Iterable[str] = (),
) -> AgentContext:
    evidence_list = list(evidence)
    return AgentContext(
        task_id=task.id,
        current_stage=record.current_stage,
        permitted_next_stages=list(
            WorkflowEngine.permitted_next_stages(record.current_stage)
        ),
        evidence=evidence_list,
        missing_evidence=list(missing_evidence),
        experiment_count=record.experiment_count,
        experiment_budget=task.budgets.max_experiments,
        decision_schema=AgentDecision.model_json_schema(mode="validation"),
    )


def validate_agent_decision(
    task: OptimizationTask,
    record: WorkflowRecord,
    decision: AgentDecision,
    available_evidence: Iterable[EvidenceRef],
) -> AgentDecision:
    if decision.current_stage != record.current_stage:
        raise AgentDecisionError(
            f"decision is for {decision.current_stage}, workflow is at {record.current_stage}"
        )
    permitted = WorkflowEngine.permitted_next_stages(record.current_stage)
    if decision.proposed_next_stage not in permitted:
        raise AgentDecisionError(
            f"agent proposed illegal transition to {decision.proposed_next_stage}"
        )

    evidence_ids = {item.id for item in available_evidence}
    referenced = set(decision.evidence_used)
    if not referenced:
        raise AgentDecisionError("agent must cite at least one available evidence id")
    unknown = referenced - evidence_ids
    if unknown:
        raise AgentDecisionError(f"agent cited unknown evidence: {', '.join(sorted(unknown))}")

    stage = record.current_stage
    required_field = {
        WorkflowStage.CLASSIFY_BOTTLENECK: "bottleneck_assessment",
        WorkflowStage.ANALYZE_LIMIT: "limit_estimate",
        WorkflowStage.GENERATE_HYPOTHESIS: "hypothesis",
        WorkflowStage.CREATE_EXPERIMENT: "proposed_experiment",
    }.get(stage)
    if required_field is not None and getattr(decision, required_field) is None:
        raise AgentDecisionError(f"{stage} requires {required_field}")

    nested_evidence: set[str] = set()
    if decision.bottleneck_assessment:
        nested_evidence.update(decision.bottleneck_assessment.evidence_ids)
    if decision.limit_estimate:
        nested_evidence.update(decision.limit_estimate.evidence_ids)
    if decision.hypothesis:
        nested_evidence.update(decision.hypothesis.observed_evidence_ids)
    update_ids = [update.id for update in decision.execution_map_updates]
    if len(update_ids) != len(set(update_ids)):
        raise AgentDecisionError("execution_map_updates must use unique entry ids")
    for update in decision.execution_map_updates:
        nested_evidence.update(update.evidence_ids)
    nested_unknown = nested_evidence - evidence_ids
    if nested_unknown:
        raise AgentDecisionError(
            f"structured analysis cited unknown evidence: {', '.join(sorted(nested_unknown))}"
        )

    if decision.proposed_experiment:
        if decision.proposed_experiment.task_id != task.id:
            raise AgentDecisionError("proposed experiment belongs to another task")
        if decision.hypothesis and (
            decision.proposed_experiment.hypothesis_id != decision.hypothesis.id
        ):
            raise AgentDecisionError("experiment hypothesis_id does not match proposed hypothesis")

    wants_deeper_profile = decision.proposed_next_stage == WorkflowStage.DISCOVER_HOTSPOTS
    if wants_deeper_profile and decision.requested_profile_level is None:
        raise AgentDecisionError("deeper profiling transition requires requested_profile_level")
    if decision.requested_profile_level is not None:
        requested_index = _PROFILE_ORDER.index(decision.requested_profile_level)
        maximum_index = _PROFILE_ORDER.index(task.budgets.max_profile_level)
        if requested_index > maximum_index:
            raise AgentDecisionError("requested profile level exceeds task budget")

    return decision


def merge_execution_map_updates(
    task_id: str,
    current: InferenceExecutionMap | None,
    updates: Iterable[ExecutionMapEntry],
) -> InferenceExecutionMap:
    """Deterministically upsert complete Agent-proposed map entries by id."""

    if current is not None and current.task_id != task_id:
        raise AgentDecisionError("execution map belongs to another task")
    ordered = list(current.entries) if current is not None else []
    positions = {entry.id: index for index, entry in enumerate(ordered)}
    seen_updates: set[str] = set()
    for update in updates:
        if update.id in seen_updates:
            raise AgentDecisionError("execution map update ids must be unique")
        seen_updates.add(update.id)
        if update.id in positions:
            ordered[positions[update.id]] = update
        else:
            positions[update.id] = len(ordered)
            ordered.append(update)
    return InferenceExecutionMap(task_id=task_id, entries=ordered)


def persist_execution_map_updates(
    store: ExperimentStore,
    task_id: str,
    updates: Iterable[ExecutionMapEntry],
    *,
    current: InferenceExecutionMap | None = None,
) -> tuple[InferenceExecutionMap, ArtifactRef]:
    """Merge updates and save a new immutable execution-map evidence version."""

    merged = merge_execution_map_updates(task_id, current, updates)
    artifact = store.save_evidence_json(
        task_id,
        "execution-map.json",
        merged,
        producer="agent_handoff",
    )
    store.save_json(
        task_id,
        "state/execution-map-pointer.json",
        {"path": artifact.path, "sha256": artifact.sha256},
        producer="agent_handoff",
    )
    store.append_event(
        task_id,
        "execution_map_updated",
        {
            "path": artifact.path,
            "sha256": artifact.sha256,
            "entry_count": len(merged.entries),
        },
    )
    return merged, artifact


def load_current_execution_map(
    store: ExperimentStore, task_id: str
) -> InferenceExecutionMap | None:
    """Load and integrity-check the execution map selected by the mutable pointer."""

    try:
        pointer = store.load_json(task_id, "state/execution-map-pointer.json")
    except StoreError as error:
        if "not found" in str(error):
            return None
        raise
    if not isinstance(pointer, dict):
        raise AgentDecisionError("execution map pointer is invalid")
    path = pointer.get("path")
    sha256 = pointer.get("sha256")
    if not isinstance(path, str) or not isinstance(sha256, str):
        raise AgentDecisionError("execution map pointer lacks path or sha256")
    artifact = store.artifact_ref(task_id, path)
    if artifact is None or artifact.sha256 != sha256:
        raise AgentDecisionError("execution map pointer does not match the manifest")
    if not store.verify_artifact(task_id, artifact):
        raise AgentDecisionError("execution map artifact failed integrity verification")
    return store.load_json(task_id, path, InferenceExecutionMap)


def write_agent_context(store: ExperimentStore, context: AgentContext) -> None:
    store.save_json(
        context.task_id,
        "state/agent-context.json",
        context,
        producer="agent_handoff",
    )
    store.append_event(
        context.task_id,
        "agent_context_created",
        {"stage": context.current_stage, "evidence_count": len(context.evidence)},
    )


def load_and_validate_agent_decision(
    store: ExperimentStore,
    task: OptimizationTask,
    record: WorkflowRecord,
    available_evidence: Iterable[EvidenceRef],
    relative_path: str = "state/agent-decision.json",
) -> AgentDecision:
    try:
        decision = store.load_json(task.id, relative_path, AgentDecision)
    except ValidationError as exc:
        raise AgentDecisionError("agent decision does not match its JSON Schema") from exc
    return validate_agent_decision(task, record, decision, available_evidence)


def evidence_keys_for_agent_stage(stage: WorkflowStage) -> frozenset[str]:
    """Expose code-owned completion keys without duplicating them in a prompt."""

    return STAGE_REQUIREMENTS[stage]
