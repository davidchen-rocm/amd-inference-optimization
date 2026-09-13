"""Command-line interface for the AMD inference optimization workflow."""

# Typer intentionally declares CLI metadata through calls in parameter defaults.
# ruff: noqa: B008

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from .agent_handoff import artifact_evidence_id, artifact_is_agent_evidence
from .cli_common import _default_store, _document, _fail, _json
from .cli_vllm import _vllm_action_payload, vllm_app
from .models import (
    AgentDecision,
    ApprovalReceipt,
    ApprovalRequest,
    ArtifactRef,
    EvidenceRef,
    GateDecision,
    InferenceExecutionMap,
    OptimizationTask,
    WorkflowRecord,
    WorkflowStage,
    WorkflowStatus,
    utc_now,
)
from .replay import ReplayError, run_recorded_replay
from .store import ExperimentStore, StoreError

app = typer.Typer(
    name="gpuopt",
    no_args_is_help=True,
    help="Evidence-gated AMD GPU inference optimization workflow.",
)
task_app = typer.Typer(no_args_is_help=True, help="Create and inspect optimization tasks.")
decision_app = typer.Typer(no_args_is_help=True, help="Submit structured agent decisions.")
campaign_app = typer.Typer(
    no_args_is_help=True,
    help="Coordinate mixed-bit, shape-kernel, and KV-cache campaigns.",
)
gfx1201_app = typer.Typer(
    no_args_is_help=True,
    help="Run the strictly ordered gfx1201 capability-closure campaign.",
)
evidence_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect providers and import hash-bound external evidence.",
)
bundle_app = typer.Typer(
    no_args_is_help=True,
    help="Build and inspect canonical experiment bundles.",
)
config_app = typer.Typer(
    no_args_is_help=True,
    help="Set stable project defaults once.",
)
model_app = typer.Typer(
    no_args_is_help=True,
    help="Scan and inspect origin/derived model-library entries.",
)
eval_app = typer.Typer(
    no_args_is_help=True,
    help="Prepare and inspect frozen quality evaluation suites.",
)
app.add_typer(task_app, name="task")
app.add_typer(decision_app, name="decision")
app.add_typer(campaign_app, name="campaign")
app.add_typer(gfx1201_app, name="gfx1201")
app.add_typer(evidence_app, name="evidence")
app.add_typer(bundle_app, name="bundle")
app.add_typer(config_app, name="config")
app.add_typer(model_app, name="model")
app.add_typer(eval_app, name="eval")
app.add_typer(vllm_app, name="vllm")


def _with_next_action(
    result: dict[str, Any], task_id: str, store_root: Path
) -> dict[str, Any]:
    """Attach one human-facing continuation instead of exposing CLI internals."""

    store = str(Path(store_root).resolve())
    paused = str(result.get("paused_for", ""))
    request_id = result.get("request_id")
    if isinstance(request_id, str) and request_id:
        action = {
            "kind": "APPROVAL_REQUIRED",
            "reason": paused or "an exact ROCm execution request needs approval",
            "command": f"gpuopt approve {task_id} {request_id} --store {store}",
            "then": f"gpuopt resume {task_id} --store {store}",
        }
    elif paused == "agent_decision":
        action = {
            "kind": "AGENT_DECISION_REQUIRED",
            "reason": "the workflow requires an evidence-grounded AgentDecision",
            "context": result.get("agent_context", "state/agent-context.json"),
            "command": (
                f"gpuopt decision submit {task_id} <decision.yaml> --store {store}"
            ),
            "then": f"gpuopt resume {task_id} --store {store}",
        }
    elif paused in {"quality_execution", "quality_running"}:
        action = {
            "kind": "WAIT_AND_RESUME",
            "reason": "the durable quality worker is still running",
            "command": f"gpuopt resume {task_id} --store {store}",
        }
    elif paused == "manual_recovery":
        action = {
            "kind": "MANUAL_RECOVERY_REQUIRED",
            "reason": str(result.get("reason", "an execution lease is orphaned")),
            "command": f"gpuopt task status {task_id} --store {store}",
        }
    elif result.get("terminal_decision") is not None or str(
        result.get("status", "")
    ) in {"ACCEPTED", "REJECTED", "INCONCLUSIVE"}:
        action = {
            "kind": "COMPLETE",
            "command": f"gpuopt report {task_id} --store {store}",
        }
    else:
        action = {
            "kind": "RESUME",
            "command": f"gpuopt resume {task_id} --store {store}",
        }
    return {**result, "next_action": action}


_AGENT_STAGES = {
    WorkflowStage.CLASSIFY_BOTTLENECK,
    WorkflowStage.ANALYZE_LIMIT,
    WorkflowStage.GENERATE_HYPOTHESIS,
    WorkflowStage.CREATE_EXPERIMENT,
}

_STAGE_ARTIFACTS: dict[WorkflowStage, dict[str, str]] = {
    WorkflowStage.INSPECT_TARGET: {"inspection": "artifacts/inspection.json"},
    WorkflowStage.CAPTURE_BASELINE: {"baseline": "artifacts/baseline.json"},
    WorkflowStage.DECOMPOSE_E2E: {"decomposition": "artifacts/decomposition.json"},
    WorkflowStage.BUILD_EXECUTION_MAP: {"execution_map": "artifacts/execution-map.json"},
    WorkflowStage.DISCOVER_HOTSPOTS: {"kernel_evidence": "artifacts/kernel-evidence.json"},
    WorkflowStage.PATCH_AND_BUILD: {"build_result": "artifacts/build-result.json"},
    WorkflowStage.MICROBENCH: {"microbenchmark_result": "artifacts/microbenchmark.json"},
    WorkflowStage.E2E_VALIDATION: {"e2e_result": "artifacts/e2e-result.json"},
    WorkflowStage.QUALITY_VALIDATION: {"quality_result": "artifacts/quality-result.json"},
    WorkflowStage.DECIDE: {"gate_decision": "artifacts/gate-decision.json"},
}


def _available_evidence(store: ExperimentStore, task_id: str) -> list[EvidenceRef]:
    try:
        manifest = store.load_json(task_id, "artifacts/manifest.json")
    except StoreError:
        return []
    evidence = []
    for path, raw in sorted(manifest.get("artifacts", {}).items()):
        artifact = ArtifactRef.model_validate(raw)
        if not artifact_is_agent_evidence(artifact):
            continue
        evidence.append(
            EvidenceRef(
                id=artifact_evidence_id(artifact),
                kind=artifact.media_type,
                summary=f"Persisted artifact {path}",
                artifact=artifact,
                source="workflow",
            )
        )
    return evidence


def _merge_execution_map_updates(
    store: ExperimentStore,
    task_id: str,
    decision: AgentDecision,
    decision_key: str,
) -> str | None:
    if not decision.execution_map_updates:
        return None
    try:
        pointer = store.load_json(task_id, "state/execution-map-current.json")
        current_path = pointer["path"]
    except (StoreError, KeyError, TypeError):
        current_path = "artifacts/execution-map.json"
    execution_map = store.load_json(task_id, current_path, InferenceExecutionMap)
    entries = {entry.id: entry for entry in execution_map.entries}
    for update in decision.execution_map_updates:
        entries[update.id] = update
    execution_map.entries = list(entries.values())
    execution_map.updated_at = utc_now()
    relative = f"artifacts/execution-maps/{decision_key}.json"
    artifact = store.save_json(
        task_id,
        relative,
        execution_map,
        producer="optimization-agent",
    )
    store.save_json(
        task_id,
        "state/execution-map-current.json",
        {"path": artifact.path, "sha256": artifact.sha256},
        producer="workflow",
    )
    return artifact.path


def _create_task(
    document: dict[str, Any], config: Path, store_root: Path
) -> tuple[OptimizationTask, Path]:
    task = OptimizationTask.model_validate(document)
    store = ExperimentStore(store_root)
    directory = store.create_task(task)
    workflow = WorkflowRecord(task_id=task.id)
    store.save_workflow(workflow)
    store.append_event(task.id, "task_created", {"config": str(config.resolve())})
    return task, directory


@evidence_app.command("capabilities")
def evidence_capabilities(
    task_id: str | None = typer.Option(None, "--task-id"),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Probe provider installation without executing a profiler or workload."""

    try:
        from .evidence_catalog import probe_evidence_capabilities

        snapshot = probe_evidence_capabilities()
        reference = None
        if task_id is not None:
            store = ExperimentStore(store_root)
            store.load_task(task_id)
            reference = store.save_evidence_json(
                task_id,
                "providers/evidence-capabilities",
                snapshot,
                producer="evidence-capability-probe",
            )
    except (ValidationError, StoreError, OSError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json({"snapshot": snapshot, "artifact": reference}))


@evidence_app.command("runtime-health")
def evidence_runtime_health(
    task_id: str,
    path: list[Path] = typer.Option([], "--path", help="Additional dependency path to stat."),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Capture host/runtime health without collecting GPU telemetry."""

    try:
        from .runtime_health import capture_runtime_health

        store = ExperimentStore(store_root)
        task = store.load_task(task_id)
        snapshot = capture_runtime_health(
            workspace=store.root,
            extra_paths=[task.runtime.repo_path, task.model.path, *path],
        )
        reference = store.save_evidence_json(
            task_id,
            "providers/runtime-health",
            snapshot,
            producer="runtime-health-probe",
        )
    except (ValidationError, StoreError, OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(_json({"snapshot": snapshot, "artifact": reference}))


@evidence_app.command("import-magpie")
def evidence_import_magpie(
    task_id: str,
    report: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Import a Magpie benchmark report and preserve its original bytes."""

    try:
        from .external_evidence import import_magpie_benchmark_report

        store = ExperimentStore(store_root)
        store.load_task(task_id)
        raw = store.import_evidence(
            task_id,
            "providers/magpie-source",
            report,
            producer="magpie",
            media_type="application/json",
        )
        evidence = import_magpie_benchmark_report(report)
        normalized = store.save_evidence_json(
            task_id,
            "providers/magpie-evidence",
            evidence,
            producer="magpie-adapter",
        )
    except (ValidationError, StoreError, OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(_json({"source": raw, "normalized": normalized, "evidence": evidence}))


@evidence_app.command("import-tracelens")
def evidence_import_tracelens(
    task_id: str,
    report: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    architecture: str = typer.Option(..., "--architecture"),
    provenance_kind: str = typer.Option(..., "--provenance-kind"),
    provenance_source: str = typer.Option(..., "--provenance-source"),
    provenance_sha256: str | None = typer.Option(None, "--provenance-sha256"),
    phase: str | None = typer.Option(None, "--phase"),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Import compact TraceLens roofline analysis with architecture provenance."""

    try:
        from .external_evidence import (
            RooflineArchitectureProvenance,
            RooflineProvenanceKind,
            import_tracelens_roofline_csv,
        )

        store = ExperimentStore(store_root)
        store.load_task(task_id)
        raw = store.import_evidence(
            task_id,
            "providers/tracelens-source",
            report,
            producer="tracelens",
            media_type="text/csv",
        )
        provenance = RooflineArchitectureProvenance(
            kind=RooflineProvenanceKind(provenance_kind),
            architecture=architecture,
            source=provenance_source,
            source_sha256=provenance_sha256,
        )
        evidence = import_tracelens_roofline_csv(
            report,
            architecture=provenance,
            phase=phase,
        )
        normalized = store.save_evidence_json(
            task_id,
            "providers/tracelens-evidence",
            evidence,
            producer="tracelens-adapter",
        )
    except (ValidationError, StoreError, OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(_json({"source": raw, "normalized": normalized, "evidence": evidence}))


@evidence_app.command("import-intellikit")
def evidence_import_intellikit(
    task_id: str,
    report: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    tool: str = typer.Option(..., "--tool"),
    capability: str = typer.Option(..., "--capability"),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Import IntelliKit JSON as analysis-only evidence pending a typed adapter."""

    try:
        from .external_evidence import IntelliKitTool, import_intellikit_json

        store = ExperimentStore(store_root)
        store.load_task(task_id)
        raw = store.import_evidence(
            task_id,
            "providers/intellikit-source",
            report,
            producer=f"intellikit-{tool}",
            media_type="application/json",
        )
        evidence = import_intellikit_json(
            report,
            tool=IntelliKitTool(tool),
            capability=capability,
        )
        normalized = store.save_evidence_json(
            task_id,
            "providers/intellikit-evidence",
            evidence,
            producer="intellikit-adapter",
        )
    except (ValidationError, StoreError, OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(_json({"source": raw, "normalized": normalized, "evidence": evidence}))


@evidence_app.command("shape-manifest")
def evidence_shape_manifest(
    task_id: str,
    kernel_evidence: Path = typer.Option(
        ..., "--kernel-evidence", exists=True, dir_okay=False, readable=True
    ),
    rules: Path = typer.Option(..., "--rules", exists=True, dir_okay=False, readable=True),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Build a config-driven shape manifest without guessing M/N/K."""

    try:
        import hashlib

        from .kernel_manifest import ShapeAttributionRule, build_kernel_shape_manifest

        store = ExperimentStore(store_root)
        store.load_task(task_id)
        payload = _document(kernel_evidence)
        rule_document = _document(rules)
        raw_rules = rule_document.get("rules")
        if not isinstance(raw_rules, list):
            raise ValueError("shape attribution document requires a rules array")
        parsed_rules = [ShapeAttributionRule.model_validate(item) for item in raw_rules]
        source_hash = hashlib.sha256(kernel_evidence.read_bytes()).hexdigest()
        manifest = build_kernel_shape_manifest(
            payload,
            parsed_rules,
            source_sha256=source_hash,
        )
        source_ref = store.import_evidence(
            task_id,
            "providers/kernel-shape-source",
            kernel_evidence,
            producer="kernel-evidence-provider",
            media_type="application/json",
        )
        rules_ref = store.import_evidence(
            task_id,
            "providers/kernel-shape-rules",
            rules,
            producer="operator-map",
            media_type="application/x-yaml",
        )
        manifest_ref = store.save_evidence_json(
            task_id,
            "providers/kernel-shape-manifest",
            manifest,
            producer="kernel-shape-manifest",
        )
    except (ValidationError, StoreError, OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(
        _json(
            {
                "source": source_ref,
                "rules": rules_ref,
                "manifest": manifest_ref,
                "summary": {
                    "resolved": manifest.resolved_variants,
                    "unresolved": manifest.unresolved_variants,
                    "ambiguous": manifest.ambiguous_variants,
                },
            }
        )
    )


@campaign_app.command("create")
def campaign_create(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, readable=True),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Create a validated, non-executing multi-strategy campaign."""

    try:
        from .campaign_models import CampaignConfig
        from .campaign_store import CampaignStore

        configured = CampaignConfig.model_validate(_document(config))
        record = CampaignStore(store_root).create(configured)
    except (ValidationError, StoreError, OSError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(
        _json(
            {
                "campaign_id": record.campaign_id,
                "task_id": record.task_id,
                "stage": record.current_stage,
                "status": record.status,
                "quality_policy": record.config.quality_policy,
            }
        )
    )


@campaign_app.command("status")
def campaign_status(
    campaign_id: str,
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Print the complete persisted campaign record."""

    try:
        from .campaign_store import CampaignStore

        record = CampaignStore(store_root).load(campaign_id)
    except (ValidationError, StoreError, OSError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(record))


@campaign_app.command("actions")
def campaign_actions() -> None:
    """Print the stable local optimization action catalogue."""

    from .control_policy import action_catalogue

    typer.echo(
        _json(
            {
                "schema": "gpuopt.action-catalogue.v1",
                "actions": list(action_catalogue()),
            }
        )
    )


@campaign_app.command("breakdown")
def campaign_breakdown(
    campaign_id: str,
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Print the current stable session-breakdown projection."""

    try:
        from .campaign_store import CampaignStore
        from .session_breakdown import SESSION_BREAKDOWN_PATH, SessionBreakdownV1

        campaign_store = CampaignStore(store_root)
        record = campaign_store.load(campaign_id)
        report = campaign_store.store.load_json(
            record.task_id,
            SESSION_BREAKDOWN_PATH,
            SessionBreakdownV1,
        )
    except (ValidationError, StoreError, OSError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(report))


def _campaign_step_payload(result: Any) -> dict[str, Any]:
    return {
        "campaign_id": result.record.campaign_id,
        "task_id": result.record.task_id,
        "stage": result.record.current_stage,
        "status": result.record.status,
        "revision": result.record.revision,
        "action": result.action,
        **result.detail,
    }


@gfx1201_app.command("create")
def gfx1201_create(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, readable=True),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Create the versioned, strictly ordered gfx1201 campaign."""

    try:
        from .gfx1201_campaign import Gfx1201CampaignConfig, Gfx1201CampaignStore

        configured = Gfx1201CampaignConfig.model_validate(_document(config))
        record = Gfx1201CampaignStore(store_root).create(configured)
    except (ValidationError, OSError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(record))


@gfx1201_app.command("status")
def gfx1201_status(
    task_id: str,
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Print the complete gfx1201 campaign record."""

    try:
        from .gfx1201_campaign import Gfx1201CampaignStore

        record = Gfx1201CampaignStore(store_root).load(task_id)
    except (ValidationError, OSError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(record))


@gfx1201_app.command("advance")
def gfx1201_advance(
    task_id: str,
    input_path: Path | None = typer.Option(
        None, "--input", exists=True, dir_okay=False, readable=True
    ),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Complete exactly the current capability stage from hash-bound evidence."""

    try:
        from .gfx1201_campaign import (
            CapabilityResult,
            Gfx1201CampaignEngine,
            Gfx1201CampaignStore,
        )

        campaign_store = Gfx1201CampaignStore(store_root)
        record = campaign_store.load(task_id)
        document = _document(input_path) if input_path is not None else {}
        raw_result = document.get("result")
        result = CapabilityResult.model_validate(raw_result) if raw_result is not None else None
        if result is not None:
            for artifact in result.evidence:
                registered = campaign_store.store.artifact_ref(task_id, artifact.path)
                if registered is None or registered.sha256 != artifact.sha256:
                    raise RuntimeError(
                        f"capability evidence is not registered in the task store: {artifact.path}"
                    )
                if not campaign_store.store.verify_artifact(task_id, registered):
                    raise RuntimeError(f"capability evidence integrity failed: {artifact.path}")
        updated = Gfx1201CampaignEngine.advance(
            record,
            result=result,
            selected_mixed_precision_candidate_id=document.get(
                "selected_mixed_precision_candidate_id"
            ),
        )
        campaign_store.save(updated)
    except (ValidationError, OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(_json(updated))


@gfx1201_app.command("mixed-plan")
def gfx1201_mixed_plan(
    task_id: str,
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False, readable=True),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Generate and persist bounded mixed-precision assignments without executing tools."""

    try:
        from .mixed_precision import (
            PrecisionPolicy,
            PrecisionSearchSpace,
            SensitivityEvidence,
            TensorInventory,
            plan_precision_policies,
        )

        document = _document(input_path)
        inventory = TensorInventory.model_validate(document.get("inventory"))
        sensitivity = SensitivityEvidence.model_validate(document.get("sensitivity"))
        search = PrecisionSearchSpace.model_validate(document.get("search"))
        external = [
            PrecisionPolicy.model_validate(value)
            for value in document.get("external_policies", [])
        ]
        policies = plan_precision_policies(
            inventory,
            sensitivity,
            search,
            external_policies=external,
        )
        store = ExperimentStore(store_root)
        reference = store.save_evidence_json(
            task_id,
            "gfx1201/mixed-precision/plan",
            {
                "schema": "gpuopt.mixed-precision-plan.v1",
                "candidate_count": len(policies),
                "policies": [policy.model_dump(mode="json", by_alias=True) for policy in policies],
            },
            producer="mixed-precision-planner",
        )
    except (ValidationError, OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(_json({"artifact": reference, "candidate_ids": [item.id for item in policies]}))


@gfx1201_app.command("mixed-rank")
def gfx1201_mixed_rank(
    task_id: str,
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False, readable=True),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Persist a deterministic Pareto frontier from completed candidate results."""

    try:
        from .mixed_precision import MixedPrecisionCandidateResult, rank_pareto

        document = _document(input_path)
        values = document.get("results")
        if not isinstance(values, list):
            raise ValueError("mixed-rank input requires a results array")
        results = [MixedPrecisionCandidateResult.model_validate(value) for value in values]
        frontier = rank_pareto(results)
        store = ExperimentStore(store_root)
        reference = store.save_evidence_json(
            task_id,
            "gfx1201/mixed-precision/pareto",
            frontier,
            producer="mixed-precision-pareto",
        )
    except (ValidationError, OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(_json({"artifact": reference, "frontier": frontier}))


@gfx1201_app.command("hip-graph-ab")
def gfx1201_hip_graph_ab(
    task_id: str,
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False, readable=True),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Run the exact local ON/OFF benchmark pairs.",
    ),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Plan or execute the strict same-binary HIP Graph A/B stage."""

    try:
        from .gfx1201_campaign import (
            CapabilityOutcome,
            CapabilityResult,
            Gfx1201CampaignEngine,
            Gfx1201CampaignStore,
            Gfx1201Stage,
        )
        from .hip_graph_ab import (
            HipGraphABSpec,
            HipGraphRunCoordinates,
            HipGraphTraceEvidence,
            run_hip_graph_ab,
        )

        campaign_store = Gfx1201CampaignStore(store_root)
        record = campaign_store.load(task_id)
        if record.current_stage != Gfx1201Stage.RUN_HIP_GRAPH_AB:
            raise RuntimeError("hip-graph-ab is valid only at RUN_HIP_GRAPH_AB")
        document = _document(input_path)
        spec = HipGraphABSpec.model_validate(document.get("spec"))
        coordinates = HipGraphRunCoordinates.model_validate(document.get("coordinates"))
        trace = (
            HipGraphTraceEvidence.model_validate(document["trace"])
            if document.get("trace") is not None
            else None
        )
        if not execute:
            typer.echo(
                _json(
                    {
                        "executed": False,
                        "coordinate_hash": spec.coordinate_hash,
                        "protocol_hash": coordinates.protocol_hash,
                        "argv": coordinates.argv,
                        "paired_samples": spec.initial_paired_samples,
                        "next_command": "repeat with --execute",
                    }
                )
            )
            return
        for artifact in trace.artifacts if trace else []:
            registered = campaign_store.store.artifact_ref(task_id, artifact.path)
            if registered != artifact or not campaign_store.store.verify_artifact(
                task_id, artifact
            ):
                raise RuntimeError(f"trace artifact is not registered: {artifact.path}")
        decision, decision_ref = run_hip_graph_ab(
            spec,
            coordinates,
            task_id=task_id,
            store=campaign_store.store,
            trace=trace,
        )
        result = CapabilityResult(
            capability="hip_graph_ab",
            outcome=CapabilityOutcome(decision.outcome.value),
            summary="; ".join(decision.reasons),
            experiment_ids=[str(document.get("experiment_id", "hip-graph-ab"))],
            evidence=[*decision.evidence, decision_ref],
            missing_evidence=trace.missing_evidence if trace else ["trace evidence not supplied"],
        )
        updated = Gfx1201CampaignEngine.advance(record, result=result)
        campaign_store.save(updated)
    except (ValidationError, OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(_json({"campaign": updated, "decision": decision, "artifact": decision_ref}))


@gfx1201_app.command("final-report")
def gfx1201_final_report(
    task_id: str,
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False, readable=True),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Validate, persist, and close the immutable gfx1201 capability report."""

    try:
        from .final_report import Gfx1201FinalReport, persist_gfx1201_final_report
        from .gfx1201_campaign import (
            CapabilityOutcome,
            CapabilityResult,
            Gfx1201CampaignEngine,
            Gfx1201CampaignStore,
            Gfx1201Stage,
        )

        campaign_store = Gfx1201CampaignStore(store_root)
        record = campaign_store.load(task_id)
        if record.current_stage != Gfx1201Stage.BUILD_FINAL_REPORT:
            raise RuntimeError("final-report is valid only at BUILD_FINAL_REPORT")
        report = Gfx1201FinalReport.model_validate(_document(input_path))
        if report.task_id != task_id or report.campaign_id != record.campaign_id:
            raise RuntimeError("final report task/campaign identity mismatch")
        json_ref, markdown_ref = persist_gfx1201_final_report(
            report, campaign_store.store
        )
        experiment_ids = sorted(
            {
                experiment_id
                for entry in [*report.model, *report.kernel, *report.runtime]
                for experiment_id in entry.experiment_ids
            }
        )
        result = CapabilityResult(
            capability="gfx1201_final_report",
            outcome=CapabilityOutcome.COMPLETE,
            summary="gfx1201 capability report is complete",
            experiment_ids=experiment_ids,
            evidence=[json_ref, markdown_ref],
        )
        updated = Gfx1201CampaignEngine.advance(record, result=result)
        campaign_store.save(updated)
    except (ValidationError, OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(_json({"campaign": updated, "json": json_ref, "markdown": markdown_ref}))


@campaign_app.command("plan")
def campaign_plan(
    campaign_id: str,
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Materialize the fixed candidate plans at PLAN_CANDIDATES."""

    try:
        from .campaign_control import run_campaign_step
        from .campaign_models import CampaignStage
        from .campaign_store import CampaignStore

        campaign_store = CampaignStore(store_root)
        record = campaign_store.load(campaign_id)
        if record.current_stage != CampaignStage.PLAN_CANDIDATES:
            raise RuntimeError(
                "campaign plan is valid only at PLAN_CANDIDATES; "
                f"current stage is {record.current_stage}"
            )
        result = run_campaign_step(record, campaign_store)
    except (ValidationError, StoreError, OSError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(_campaign_step_payload(result)))


@campaign_app.command("run")
def campaign_run(
    campaign_id: str,
    input_file: Path | None = typer.Option(
        None,
        "--input",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Stage-specific JSON/YAML input bound to registered evidence.",
    ),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Advance one durable campaign revision or report the required evidence."""

    try:
        from .campaign_control import run_campaign_step
        from .campaign_store import CampaignStore

        campaign_store = CampaignStore(store_root)
        record = campaign_store.load(campaign_id)
        stage_input = _document(input_file) if input_file is not None else None
        result = run_campaign_step(record, campaign_store, stage_input=stage_input)
    except (ValidationError, StoreError, OSError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(_campaign_step_payload(result)))


@campaign_app.command("report")
def campaign_report_command(
    campaign_id: str,
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Print the terminal campaign report or a current structured summary."""

    try:
        from .campaign_control import campaign_report
        from .campaign_store import CampaignStore

        campaign_store = CampaignStore(store_root)
        record = campaign_store.load(campaign_id)
        markdown = campaign_store.store.task_dir(record.task_id) / "reports/campaign-final.md"
        if markdown.is_file():
            typer.echo(markdown.read_text(encoding="utf-8"), nl=False)
            return
        typer.echo(_json(campaign_report(record)))
    except (ValidationError, StoreError, OSError, RuntimeError) as exc:
        _fail(str(exc))


@config_app.command("init")
def config_init(
    model_root: Path = typer.Option(..., "--model-root", file_okay=False),
    project_root: Path = typer.Option(Path.cwd(), "--project", file_okay=False),
    store_root: Path = typer.Option(Path(".gpuopt/store"), "--store"),
    llama_cpp_repo: Path | None = typer.Option(None, "--runtime-repo", file_okay=False),
    llama_cpp_build: Path | None = typer.Option(None, "--runtime-build", file_okay=False),
    gpu_device: int = typer.Option(0, "--gpu-device", min=0),
    mcp_command: str = typer.Option("rocm-agent-mcp", "--mcp-command"),
) -> None:
    """Create the project-local .gpuopt/config.yaml without replacing one."""

    try:
        from .project_config import initialize_project_config

        config = initialize_project_config(
            project_root,
            model_root=model_root,
            store_root=store_root,
            llama_cpp_repo=llama_cpp_repo,
            llama_cpp_build_dir=llama_cpp_build,
            gpu_device=gpu_device,
            rocm_mcp_command=[mcp_command],
        )
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(config))


@config_app.command("set")
def config_set(
    key: str,
    value: str,
    project_root: Path = typer.Option(Path.cwd(), "--project", file_okay=False),
) -> None:
    """Atomically set one allow-listed project setting."""

    try:
        from .project_config import set_project_config_value

        config = set_project_config_value(project_root, key, value)
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(config))


@config_app.command("show")
def config_show(
    project_root: Path = typer.Option(Path.cwd(), "--project", file_okay=False),
) -> None:
    """Print the resolved project configuration."""

    try:
        from .project_config import load_project_config

        config = load_project_config(project_root)
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(config))


@config_app.command("doctor")
def config_doctor(
    project_root: Path = typer.Option(Path.cwd(), "--project", file_okay=False),
) -> None:
    """Check configured paths without building or executing a workload."""

    try:
        from .project_config import check_project_config, load_project_config

        config = load_project_config(project_root)
        checks = check_project_config(config)
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
    status = "OK" if all(item.status == "OK" for item in checks) else "CHECK"
    typer.echo(_json({"status": status, "checks": checks}))


@model_app.command("scan")
def model_scan(
    project_root: Path = typer.Option(Path.cwd(), "--project", file_okay=False),
) -> None:
    """Hash and index configured origin/derived model artifacts."""

    try:
        from .model_library import scan_model_library

        catalog = scan_model_library(project_root)
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(catalog))


@model_app.command("list")
def model_list(
    role: str | None = typer.Option(None, "--role", help="ORIGIN or DERIVED"),
    quantization: str | None = typer.Option(None, "--quantization"),
    project_root: Path = typer.Option(Path.cwd(), "--project", file_okay=False),
) -> None:
    """List catalog entries without rescanning weight files."""

    try:
        from .model_library import ModelRole, list_models, load_model_catalog

        selected_role = ModelRole(role.upper()) if role is not None else None
        entries = list_models(
            load_model_catalog(project_root),
            role=selected_role,
            quantization=quantization,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json({"count": len(entries), "items": entries}))


@model_app.command("show")
def model_show(
    reference: str,
    project_root: Path = typer.Option(Path.cwd(), "--project", file_okay=False),
) -> None:
    """Resolve one stable model id or catalog alias."""

    try:
        from .model_library import load_model_catalog, show_model

        entry = show_model(load_model_catalog(project_root), reference)
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(entry))


@model_app.command("link")
def model_link(
    derived_reference: str,
    origin_reference: str = typer.Option(..., "--origin"),
    project_root: Path = typer.Option(Path.cwd(), "--project", file_okay=False),
) -> None:
    """Declare a derived-to-origin relation and rescan provenance."""

    try:
        from .model_library import link_model

        link = link_model(project_root, derived_reference, origin_reference)
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(link))


def _eval_suite_root(project_root: Path, suite_id: str) -> Path:
    return project_root.resolve() / ".gpuopt" / "eval-suites" / suite_id


@eval_app.command("prepare")
def eval_prepare(
    suite_id: str = typer.Argument("general-100.v1"),
    project_root: Path = typer.Option(Path.cwd(), "--project", file_okay=False),
) -> None:
    """Download once, deterministically select, and freeze a supported suite."""

    try:
        from .eval_suites import GENERAL_100_SUITE_ID, prepare_general_100
        from .full_eval_library import ENGLISH_FULL_SUITE_ID, prepare_english_full

        output = _eval_suite_root(project_root, suite_id)
        if suite_id == GENERAL_100_SUITE_ID:
            manifest = prepare_general_100(output)
        elif suite_id == ENGLISH_FULL_SUITE_ID:
            manifest = prepare_english_full(output)
        else:
            raise ValueError(f"unsupported preparable suite: {suite_id}")
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json({"suite": manifest, "path": str(output)}))


@eval_app.command("list")
def eval_list(
    project_root: Path = typer.Option(Path.cwd(), "--project", file_okay=False),
) -> None:
    """List default suite availability without contacting the network."""

    from .eval_suites import load_frozen_suite

    general = _eval_suite_root(project_root, "general-100.v1")
    try:
        manifest, _ = load_frozen_suite(general)
        general_item: dict[str, Any] = {
            "id": manifest.suite_id,
            "status": "AVAILABLE",
            "path": str(general),
            "total": manifest.total_cases,
        }
    except (OSError, ValueError, RuntimeError) as error:
        general_item = {
            "id": "general-100.v1",
            "status": "MISSING",
            "path": str(general),
            "detail": str(error),
        }
    math_manifest = (
        Path(__file__).resolve().parents[2]
        / "fixtures/q8-runtime-quality/manifest.json"
    )
    math_item = {
        "id": "math-100.v1",
        "status": "AVAILABLE" if math_manifest.is_file() else "MISSING",
        "path": str(math_manifest),
    }
    from .full_eval_library import load_english_full

    english_full = _eval_suite_root(project_root, "english-full.v1")
    try:
        full_manifest = load_english_full(english_full)
        full_item: dict[str, Any] = {
            "id": full_manifest.suite_id,
            "status": "AVAILABLE",
            "path": str(english_full),
            "total": full_manifest.total_records,
        }
    except (OSError, ValueError, RuntimeError) as error:
        full_item = {
            "id": "english-full.v1",
            "status": "MISSING",
            "path": str(english_full),
            "detail": str(error),
        }
    typer.echo(_json({"items": [math_item, general_item, full_item]}))


@eval_app.command("show")
def eval_show(
    suite_id: str,
    project_root: Path = typer.Option(Path.cwd(), "--project", file_okay=False),
) -> None:
    """Validate and print one frozen suite manifest."""

    try:
        if suite_id == "general-100.v1":
            from .eval_suites import load_frozen_suite

            manifest, cases = load_frozen_suite(_eval_suite_root(project_root, suite_id))
            result = {"manifest": manifest, "case_count": len(cases)}
        elif suite_id == "math-100.v1":
            path = Path(__file__).resolve().parents[2] / "fixtures/q8-runtime-quality/manifest.json"
            result = {"manifest": _document(path), "path": str(path)}
        elif suite_id == "english-full.v1":
            from .full_eval_library import load_english_full

            manifest = load_english_full(_eval_suite_root(project_root, suite_id))
            result = {"manifest": manifest, "record_count": manifest.total_records}
        else:
            raise ValueError(f"unknown suite: {suite_id}")
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
    typer.echo(_json(result))


@task_app.command("create")
def task_create(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, readable=True),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Create a validated task without executing a workload."""

    try:
        task, directory = _create_task(_document(config), config, store_root)
    except (ValidationError, StoreError, OSError) as exc:
        _fail(str(exc))
    typer.echo(_json({"task_id": task.id, "task_dir": str(directory), "stage": "CREATE_TASK"}))


@task_app.command("status")
def task_status(
    task_id: str,
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Print the persisted workflow status for a task."""

    try:
        store = ExperimentStore(store_root)
        task = store.load_task(task_id)
        workflow = store.load_workflow(task_id)
    except (StoreError, OSError, ValidationError) as exc:
        _fail(str(exc))
    typer.echo(_json({"task": task, "workflow": workflow}))


@app.command("advance")
def advance(
    task_id: str,
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Advance deterministic stages until agent input or approval is required."""

    try:
        from .agent_handoff import build_agent_context, write_agent_context
        from .workflow import WorkflowEngine

        store = ExperimentStore(store_root)
        task = store.load_task(task_id)
        record = store.load_workflow(task_id)
        engine = WorkflowEngine()
        completed: list[str] = []
        while record.status == WorkflowStatus.ACTIVE:
            stage = record.current_stage
            if stage in _AGENT_STAGES:
                evidence = _available_evidence(store, task_id)
                context = build_agent_context(task, record, evidence)
                write_agent_context(store, context)
                result = {
                    "task_id": task_id,
                    "stage": stage,
                    "status": record.status,
                    "completed_stages": completed,
                    "paused_for": "agent_decision",
                    "agent_context": "state/agent-context.json",
                }
                break
            if stage == WorkflowStage.CREATE_TASK:
                evidence_ids = {"task": "task.json"}
            else:
                evidence_ids = _STAGE_ARTIFACTS[stage]
                missing = [
                    path
                    for path in evidence_ids.values()
                    if not (store.task_dir(task_id) / path).is_file()
                ]
                if missing:
                    result = {
                        "task_id": task_id,
                        "stage": stage,
                        "status": record.status,
                        "completed_stages": completed,
                        "paused_for": "evidence",
                        "required_evidence": evidence_ids,
                        "missing_artifacts": missing,
                    }
                    break
            gate = None
            if stage == WorkflowStage.DECIDE:
                gate = store.load_json(task_id, "artifacts/gate-decision.json", GateDecision)
            record = engine.complete_stage(record, evidence_ids, gate_decision=gate)
            store.save_workflow(record)
            store.append_event(
                task_id,
                "stage_completed",
                {"stage": stage, "evidence_ids": evidence_ids},
            )
            completed.append(stage)
        else:
            result = {
                "task_id": task_id,
                "stage": record.current_stage,
                "status": record.status,
                "completed_stages": completed,
                "terminal_decision": record.terminal_decision,
            }
    except (StoreError, ValidationError, ValueError, RuntimeError, KeyError) as exc:
        _fail(str(exc))
    typer.echo(_json(result))


@decision_app.command("submit")
def decision_submit(
    task_id: str,
    decision_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Validate and submit an AgentDecision; the framework chooses the next state."""

    try:
        from .agent_handoff import validate_agent_decision
        from .workflow import WorkflowEngine

        decision = AgentDecision.model_validate(_document(decision_file))
        store = ExperimentStore(store_root)
        task = store.load_task(task_id)
        record = store.load_workflow(task_id)
        decision = validate_agent_decision(
            task,
            record,
            decision,
            _available_evidence(store, task_id),
        )
        relative = (
            "artifacts/agent-decisions/"
            f"{record.experiment_count:04d}-{record.current_stage.value}-{decision_file.stem}.json"
        )
        artifact = store.save_json(task_id, relative, decision, producer="optimization-agent")
        evidence_ids = {"agent_decision": artifact.path}
        execution_map_path = _merge_execution_map_updates(
            store,
            task_id,
            decision,
            f"{record.experiment_count:04d}-{record.current_stage.value}-{decision_file.stem}",
        )
        if execution_map_path is not None:
            evidence_ids["execution_map_update"] = execution_map_path
        if record.current_stage == WorkflowStage.CLASSIFY_BOTTLENECK:
            assessment = store.save_json(
                task_id,
                f"artifacts/analysis/{record.experiment_count:04d}-bottleneck.json",
                decision.bottleneck_assessment,
                producer="optimization-agent",
            )
            evidence_ids["bottleneck_assessment"] = assessment.path
        elif record.current_stage == WorkflowStage.ANALYZE_LIMIT:
            estimate = store.save_json(
                task_id,
                f"artifacts/analysis/{record.experiment_count:04d}-limit.json",
                decision.limit_estimate,
                producer="optimization-agent",
            )
            evidence_ids["limit_estimate"] = estimate.path
        elif record.current_stage == WorkflowStage.GENERATE_HYPOTHESIS:
            hypothesis = store.save_json(
                task_id,
                f"artifacts/hypotheses/{record.experiment_count:04d}.json",
                decision.hypothesis,
                producer="optimization-agent",
            )
            evidence_ids["hypothesis"] = hypothesis.path
        elif record.current_stage == WorkflowStage.CREATE_EXPERIMENT:
            if decision.proposed_experiment is None:  # validated, defensive invariant
                raise ValueError("CREATE_EXPERIMENT decision has no ExperimentSpec")
            from .experiment import freeze_experiment_inputs, materialize_q8_source_experiment

            resolved_experiment = freeze_experiment_inputs(
                task,
                decision.proposed_experiment,
                store,
            )
            resolved_experiment = materialize_q8_source_experiment(
                task,
                resolved_experiment,
                store,
            )
            experiment = store.save_json(
                task_id,
                f"experiments/{resolved_experiment.id}/spec.json",
                resolved_experiment,
                producer="optimization-agent",
            )
            store.save_json(
                task_id,
                "state/active-experiment.json",
                {
                    "experiment_id": resolved_experiment.id,
                    "spec_path": experiment.path,
                },
                producer="workflow",
            )
            evidence_ids["experiment_spec"] = experiment.path
        updated = WorkflowEngine().complete_stage(
            record,
            evidence_ids,
            requested_next_stage=decision.proposed_next_stage,
        )
        if decision.proposed_next_stage == WorkflowStage.DISCOVER_HOTSPOTS:
            try:
                active = store.load_json(task_id, "state/active-experiment.json")
                profile_target = active.get("experiment_id", "baseline")
            except StoreError:
                profile_target = "baseline"
            store.save_json(
                task_id,
                "state/requested-profile.json",
                {
                    "requested_profile_level": decision.requested_profile_level,
                    "requested_by": artifact.path,
                    "fulfilled": False,
                    "target": profile_target,
                },
                producer="workflow",
            )
        store.save_workflow(updated)
        store.append_event(
            task_id,
            "agent_decision_submitted",
            {"stage": record.current_stage, "artifact": artifact.path},
        )
        result = {
            "task_id": task_id,
            "previous_stage": record.current_stage,
            "current_stage": updated.current_stage,
            "decision_artifact": artifact.path,
        }
    except (StoreError, ValidationError, ValueError, RuntimeError, KeyError) as exc:
        _fail(str(exc))
    typer.echo(_json(result))


@app.command("approve")
def approve(
    task_id: str,
    request_id: str,
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Create a one-time receipt for the exact persisted MCP request hash."""

    try:
        from .rocm_mcp import approval_request

        store = ExperimentStore(store_root)
        request = store.load_json(
            task_id,
            f"state/approvals/{request_id}.request.json",
            ApprovalRequest,
        )
        try:
            approval_context = store.load_json(
                task_id,
                f"state/approvals/{request_id}.context.json",
            )
        except StoreError:
            approval_context = {}
        computed_hash = approval_request(
            request.tool,
            request.arguments,
            execution_context=approval_context,
        ).request_sha256
        if request.request_hash != computed_hash:
            raise StoreError("persisted approval request hash does not match its tool arguments")
        receipt_path = f"state/approvals/{request_id}.receipt.json"
        try:
            store.load_json(task_id, receipt_path, ApprovalReceipt)
        except StoreError:
            pass
        else:
            raise StoreError(f"approval already exists for request: {request_id}")
        receipt = ApprovalReceipt(request_id=request.id, request_hash=request.request_hash)
        store.save_json(task_id, receipt_path, receipt, producer="user-approval")
        store.append_event(
            task_id,
            "mcp_request_approved",
            {"request_id": request.id, "request_hash": request.request_hash},
        )
    except (StoreError, ValidationError, OSError) as exc:
        _fail(str(exc))
    typer.echo(_json(receipt))


@app.command("report")
def report(
    task_id: str,
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Print the final Markdown report, or its JSON form when Markdown is absent."""

    store = ExperimentStore(store_root)
    task_directory = store.task_dir(task_id)
    markdown = task_directory / "reports" / "final.md"
    structured = task_directory / "reports" / "final.json"
    if markdown.is_file():
        typer.echo(markdown.read_text(encoding="utf-8"), nl=False)
    elif structured.is_file():
        typer.echo(structured.read_text(encoding="utf-8"), nl=False)
    else:
        _fail(f"final report not found for task: {task_id}")


@app.command("replay")
def replay(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o", help="New replay output directory."),
) -> None:
    """Replay normalized historical evidence without executing MCP or a workload."""

    try:
        parsed = _document(config)
        task_id = parsed.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ReplayError("replay config requires task_id")
        destination = output or (Path.cwd() / "runs" / task_id)
        result = run_recorded_replay(config, destination)
    except (ReplayError, OSError) as exc:
        _fail(str(exc))
    typer.echo(
        _json(
            {
                "task_id": result.task_id,
                "output_dir": str(result.output_dir),
                "experiment_decisions": result.experiment_decisions,
                "final_decision": result.final_decision,
                "evidence_kind": "recorded_import",
            }
        )
    )


@app.command("run-live")
def run_live(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Run build/workload stages; executing ROCm MCP still pauses for exact approval.",
    ),
) -> None:
    """Initialize or resume the opt-in live vertical slice."""

    try:
        configured = OptimizationTask.model_validate(_document(config))
        store = ExperimentStore(store_root)
        directory = store.task_dir(configured.id)
        if directory.exists():
            task = store.load_task(configured.id)
            expected = configured.model_dump(exclude={"created_at"})
            actual = task.model_dump(exclude={"created_at"})
            if actual != expected:
                raise StoreError(
                    f"live config differs from persisted task: {configured.id}"
                )
        else:
            task, directory = _create_task(_document(config), config, store_root)
        if execute:
            from .live import run_live_until_pause

            result = run_live_until_pause(task, store)
            result["execution_started"] = True
            from .experiment_bundle import BundleError, refresh_task_bundles

            try:
                bundles = refresh_task_bundles(store, task.id, terminal_only=True)
                result["experiment_bundles"] = [
                    manifest.experiment_id for manifest in bundles
                ]
            except (BundleError, StoreError, ValueError, OSError) as error:
                # A derived projection must never erase or reinterpret an already
                # persisted workflow result.  Surface the maintenance error and let
                # ``gpuopt bundle rebuild`` retry it independently.
                result["bundle_warning"] = str(error)
        else:
            workflow = store.load_workflow(task.id)
            result = {
                "task_id": task.id,
                "task_dir": str(directory),
                "stage": workflow.current_stage,
                "next_command": (
                    f"gpuopt run-live {config.resolve()} --store "
                    f"{Path(store_root).resolve()} --execute"
                ),
                "execution_started": False,
                "approval_required_before_executing_mcp": True,
            }
    except (StoreError, ValidationError, ValueError, RuntimeError, OSError) as exc:
        _fail(str(exc))
    typer.echo(_json(_with_next_action(result, task.id, store_root)))


def _guided_model_reference(catalog: Any, requested: str | None) -> str:
    if requested is not None:
        return requested
    if not sys.stdin.isatty():
        raise ValueError("--model is required when stdin is not interactive")
    typer.echo("Available models:")
    for entry in catalog.entries:
        quantization = entry.quantization or entry.format.value
        typer.echo(
            f"  {entry.aliases[0]:<44} {entry.role.value:<7} "
            f"{quantization:<12} {entry.size / (1024**3):.2f} GiB"
        )
    return typer.prompt("Model id or alias")


@app.command("optimize")
def optimize(
    model_reference: str | None = typer.Option(
        None,
        "--model",
        help="Catalog id/alias. Interactive terminals can choose from a list.",
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help="Derived baseline quantization when --model selects an origin model.",
    ),
    task_id: str | None = typer.Option(None, "--task-id"),
    quality: list[str] = typer.Option(
        [],
        "--quality",
        help="Repeat for math-100.v1/general-100.v1; defaults come from config.",
    ),
    minimum_improvement: float = typer.Option(
        1.0,
        "--minimum-improvement",
        min=0,
        help="Required improvement for both tg128 and tg512.",
    ),
    max_experiments: int = typer.Option(6, "--max-experiments", min=1),
    project_root: Path = typer.Option(
        Path.cwd(), "--project", file_okay=False, help="Project or a child directory."
    ),
    store_root: Path | None = typer.Option(
        None, "--store", help="Override the configured experiment store."
    ),
    execute: bool | None = typer.Option(
        None,
        "--execute/--no-execute",
        help="Run local stages now. MCP execution still pauses for exact approval.",
    ),
    rescan_models: bool = typer.Option(
        False, "--rescan-models", help="Refresh model hashes before resolving --model."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve and validate without creating a task."
    ),
) -> None:
    """Create a complete task from project defaults, then run until a safe pause."""

    try:
        from .experiment_bundle import ModelInputProvenanceV1
        from .guided_optimize import (
            GuidedOptimizationError,
            build_optimization_task,
            resolve_runnable_model,
        )
        from .model_library import (
            ModelLibraryError,
            load_model_catalog,
            scan_model_library,
        )
        from .project_config import discover_project_root, load_project_config

        discovered = discover_project_root(project_root)
        if discovered is None:  # pragma: no cover - required discovery cannot return None
            raise ValueError("project configuration was not found")
        project = load_project_config(discovered)
        try:
            catalog = (
                scan_model_library(discovered)
                if rescan_models
                else load_model_catalog(discovered)
            )
        except ModelLibraryError:
            typer.echo("Model catalog missing; scanning configured model root...", err=True)
            catalog = scan_model_library(discovered)
        reference = _guided_model_reference(catalog, model_reference)
        selected = catalog.resolve(reference)
        selected_baseline = baseline
        if selected.role.value == "ORIGIN" and selected_baseline is None:
            if not sys.stdin.isatty():
                raise GuidedOptimizationError(
                    "an origin model requires --baseline (for example Q6_K)"
                )
            selected_baseline = typer.prompt("Baseline quantization", default="Q6_K")
        runnable = resolve_runnable_model(catalog, reference, selected_baseline)
        suites = tuple(quality or project.default_quality_suites)
        task = build_optimization_task(
            project_root=discovered,
            project=project,
            catalog=catalog,
            model=runnable,
            task_id=task_id,
            minimum_improvement_percent=minimum_improvement,
            max_experiments=max_experiments,
            quality_suites=suites,
        )
        resolved_store = (store_root or project.store_root).expanduser().resolve()
        request = {
            "schema": "gpuopt.guided-optimization-request.v1",
            "project_root": str(discovered),
            "model_reference": reference,
            "resolved_model_id": runnable.id,
            "baseline_quantization": runnable.quantization,
            "quality_suites": list(suites),
            "minimum_improvement_percent": minimum_improvement,
            "max_experiments": max_experiments,
            "task": task.model_dump(mode="json"),
        }
        if dry_run:
            create_argv = [
                "gpuopt",
                "optimize",
                "--model",
                runnable.id,
                "--task-id",
                task.id,
                "--project",
                str(discovered),
                "--store",
                str(resolved_store),
                "--minimum-improvement",
                str(minimum_improvement),
                "--max-experiments",
                str(max_experiments),
                "--no-execute",
            ]
            for suite in suites:
                create_argv.extend(["--quality", suite])
            typer.echo(
                _json(
                    {
                        **request,
                        "dry_run": True,
                        "store": str(resolved_store),
                        "next_action": {
                            "kind": "CREATE_TASK",
                            "command": shlex.join(create_argv),
                        },
                    }
                )
            )
            return

        store = ExperimentStore(resolved_store)
        directory = store.create_task(task)
        store.save_workflow(WorkflowRecord(task_id=task.id))
        request_ref = store.save_immutable_json(
            task.id,
            "artifacts/guided-request.json",
            request,
            producer="guided-cli",
        )
        provenance_ref = store.save_immutable_json(
            task.id,
            "artifacts/model-input-provenance.json",
            ModelInputProvenanceV1(
                model_path=str(runnable.model_path.resolve()),
                model_sha256=runnable.sha256,
                packed_bytes=runnable.size,
                quantization=runnable.quantization,
                architecture=runnable.architecture,
            ),
            producer="guided-cli",
        )
        store.append_event(
            task.id,
            "task_created",
            {
                "source": "guided-cli",
                "request": request_ref.model_dump(mode="json"),
                "model_provenance": provenance_ref.model_dump(mode="json"),
            },
        )
        should_execute = execute
        if should_execute is None:
            should_execute = (
                typer.confirm("Run baseline/profiling preparation now?", default=True)
                if sys.stdin.isatty()
                else False
            )
        if should_execute:
            from .experiment_bundle import BundleError, refresh_task_bundles
            from .live import run_live_until_pause

            result = run_live_until_pause(task, store)
            result["execution_started"] = True
            try:
                bundles = refresh_task_bundles(store, task.id, terminal_only=True)
                result["experiment_bundles"] = [item.experiment_id for item in bundles]
            except (BundleError, StoreError, ValueError, OSError) as error:
                result["bundle_warning"] = str(error)
        else:
            result = {
                "task_id": task.id,
                "task_dir": str(directory),
                "stage": WorkflowStage.CREATE_TASK,
                "execution_started": False,
                "approval_required_before_executing_mcp": True,
            }
        result.update(
            {
                "resolved_model": runnable.model_dump(mode="json"),
                "quality_suites": list(suites),
            }
        )
        typer.echo(_json(_with_next_action(result, task.id, resolved_store)))
    except (
        GuidedOptimizationError,
        ModelLibraryError,
        StoreError,
        ValidationError,
        ValueError,
        RuntimeError,
        OSError,
    ) as exc:
        _fail(str(exc))


@app.command("resume")
def resume(
    task_id: str,
    store_root: Path = typer.Option(_default_store(), "--store", help="Experiment store root."),
) -> None:
    """Resume a task, dispatching vLLM to its non-executing next-action view."""

    try:
        store = ExperimentStore(store_root)
        task = store.load_task(task_id)
        if task.campaign_kind.value == "vllm_mi300x":
            from .vllm_workflow import VLLMWorkflowCoordinator

            coordinator = VLLMWorkflowCoordinator(store)
            result = _vllm_action_payload(coordinator, coordinator.load(task_id))
            result.pop("record")
            typer.echo(_json(result))
            return

        from .experiment_bundle import BundleError, refresh_task_bundles
        from .live import run_live_until_pause

        result = run_live_until_pause(task, store)
        result["execution_started"] = True
        try:
            bundles = refresh_task_bundles(store, task.id, terminal_only=True)
            result["experiment_bundles"] = [item.experiment_id for item in bundles]
        except (BundleError, StoreError, ValueError, OSError) as error:
            result["bundle_warning"] = str(error)
        result = _with_next_action(result, task_id, store_root)
    except (StoreError, ValidationError, ValueError, RuntimeError, OSError) as exc:
        _fail(str(exc))
    typer.echo(_json(result))


@bundle_app.command("rebuild")
def bundle_rebuild(
    task_id: str | None = typer.Argument(
        None,
        help="Task to rebuild; omit to rebuild every task in the store.",
    ),
    experiment_id: str | None = typer.Option(
        None,
        "--experiment",
        help="Rebuild only one experiment (requires TASK_ID).",
    ),
    store_root: Path = typer.Option(_default_store(), "--store"),
) -> None:
    """Backfill canonical summaries/manifests without changing source evidence."""

    from .experiment_bundle import (
        BundleError,
        ExperimentCatalog,
        refresh_experiment_bundle,
        refresh_task_bundles,
    )

    try:
        store = ExperimentStore(store_root)
        if experiment_id is not None and task_id is None:
            raise BundleError("--experiment requires TASK_ID")
        if task_id is not None:
            manifests = (
                [refresh_experiment_bundle(store, task_id, experiment_id)]
                if experiment_id is not None
                else refresh_task_bundles(store, task_id)
            )
        else:
            manifests = []
            for directory in sorted(store.root.iterdir(), key=lambda path: path.name):
                if (
                    directory.is_dir()
                    and not directory.is_symlink()
                    and (directory / "task.json").is_file()
                ):
                    manifests.extend(refresh_task_bundles(store, directory.name))
        if experiment_id is None:
            valid = {(manifest.task_id, manifest.experiment_id) for manifest in manifests}
            ExperimentCatalog(store.root).prune(
                valid,
                task_id=task_id,
            )
    except (BundleError, StoreError, ValueError, OSError) as error:
        _fail(str(error))
    typer.echo(
        _json(
            {
                "store": str(store.root),
                "bundle_count": len(manifests),
                "experiments": [
                    {
                        "task_id": manifest.task_id,
                        "experiment_id": manifest.experiment_id,
                        "summary": manifest.summary.path,
                        "source_digest": manifest.source_digest,
                    }
                    for manifest in manifests
                ],
            }
        )
    )


@bundle_app.command("list")
def bundle_list(
    task_id: str | None = typer.Option(None, "--task"),
    store_root: Path = typer.Option(_default_store(), "--store"),
) -> None:
    """List the rebuildable SQLite catalog; JSON bundles remain authoritative."""

    from .experiment_bundle import BundleError, ExperimentCatalog

    try:
        rows = ExperimentCatalog(store_root).rows(task_id=task_id)
    except (BundleError, OSError, ValueError) as error:
        _fail(str(error))
    typer.echo(_json({"store": str(Path(store_root).resolve()), "items": rows}))


@bundle_app.command("show")
def bundle_show(
    task_id: str = typer.Argument(...),
    experiment_id: str = typer.Argument(...),
    store_root: Path = typer.Option(_default_store(), "--store"),
) -> None:
    """Show the single human-facing summary for an experiment."""

    from .experiment_bundle import BundleError, load_experiment_bundle

    try:
        summary, manifest = load_experiment_bundle(
            ExperimentStore(store_root), task_id, experiment_id
        )
    except (BundleError, StoreError, ValueError, OSError) as error:
        _fail(str(error))
    typer.echo(
        _json(
            {
                "summary": summary,
                "manifest": {
                    "path": f"experiments/{experiment_id}/manifest.json",
                    "published_at": manifest.published_at,
                    "artifact_count": len(manifest.artifacts),
                },
            }
        )
    )


@app.command("ui")
def ui(
    store_specs: list[str] | None = typer.Option(
        None,
        "--store",
        metavar="ID=PATH",
        help="Existing read-only store; repeat for multiple stores.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Loopback bind address."),
    port: int = typer.Option(4561, "--port", min=1, max=65535, help="Local UI port."),
    draft_db: Path = typer.Option(
        Path(".gpuopt-ui/control-plane.sqlite3"),
        "--draft-db",
        help="Separate SQLite database for validated UI experiment drafts.",
    ),
) -> None:
    """Serve read-only evidence plus the draft-only experiment builder."""

    try:
        from .frontend_api import ControlPlaneReader, FrontendReadError, ReadOnlyStoreSource
        from .frontend_control import DraftStore
        from .ui_server import serve_control_plane

        configured = store_specs or [f"default={_default_store()}"]
        sources = []
        for specification in configured:
            source_id, separator, raw_path = specification.partition("=")
            if not separator or not source_id or not raw_path:
                raise FrontendReadError("--store must use ID=PATH")
            sources.append(ReadOnlyStoreSource(source_id, Path(raw_path)))
        reader = ControlPlaneReader(sources)
        serve_control_plane(
            reader,
            host=host,
            port=port,
            draft_store=DraftStore(draft_db),
            announce=typer.echo,
        )
    except (FrontendReadError, OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))


if __name__ == "__main__":  # pragma: no cover
    app()
