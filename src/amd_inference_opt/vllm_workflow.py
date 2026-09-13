"""Recoverable V0 coordinator for vLLM serving on one MI300X logical GPU.

This module deliberately coordinates adapter results instead of owning GPU
execution.  A caller may connect the ports below to the local vLLM adapter,
``CommandRunner``, ROCm Issue Agent, and an external quality command.  Calling
``run_until_pause`` itself never starts a process and always returns exactly one
next action.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from .change_policy import validate_experiment_change
from .gates import GateEngine
from .models import (
    ApprovalReceipt,
    ApprovalRequest,
    ArtifactRef,
    BaselineResult,
    BenchmarkResult,
    DecisionOutcome,
    EnvironmentFingerprint,
    ExperimentResult,
    GateCheck,
    GateDecision,
    MetricSeries,
    QualityResult,
    RunStatus,
    WorkflowStage,
    WorkflowStatus,
    utc_now,
)
from .profile_contract import ProfileContract, ProfileContractError
from .rocm_mcp import MCPServerConfig, MCPToolCall, RocmIssueAgentClient, approval_request
from .store import ExperimentStore, StoreError
from .vllm_model_snapshot import (
    VLLMModelSnapshotError,
    VLLMModelSnapshotManifest,
    verify_vllm_model_snapshot,
)
from .vllm_models import (
    MI300XInspectionEvidence,
    VLLMAgentDecision,
    VLLMApprovalStatus,
    VLLMCampaignConfig,
    VLLMExperimentSpec,
    VLLMNextAction,
    VLLMNextActionKind,
    VLLMProfileApprovalState,
    VLLMProfileExecutionPermit,
    VLLMProfileResultEvidence,
    VLLMStage,
    VLLMStageCompletion,
    VLLMWorkflowRecord,
    canonical_sha256,
)

if TYPE_CHECKING:
    from .vllm_adapter import VLLMServerSpec


VLLM_CONFIG_PATH = "state/vllm-config.json"
VLLM_WORKFLOW_PATH = "state/vllm-workflow.json"
VLLM_MODEL_SNAPSHOT_MANIFEST_PATH = "artifacts/vllm-model-snapshot-manifest.json"

VLLM_STAGE_REQUIREMENTS: dict[VLLMStage, frozenset[str]] = {
    VLLMStage.INSPECT: frozenset({"inspection"}),
    VLLMStage.SERVER_START: frozenset({"server_start"}),
    VLLMStage.BASELINE: frozenset({"baseline"}),
    VLLMStage.PROFILE_APPROVAL: frozenset(
        {"server_stop", "approval_request", "approval_receipt", "profile"}
    ),
    VLLMStage.AGENT_DECISION: frozenset({"agent_decision", "experiment_spec"}),
    VLLMStage.EXPERIMENT: frozenset(
        {"candidate_server_start", "experiment_result"}
    ),
    VLLMStage.QUALITY: frozenset(
        {"quality_result", "candidate_result", "candidate_server_stop"}
    ),
    VLLMStage.DECIDE: frozenset({"gate_decision"}),
}

_NEXT_STAGE: dict[VLLMStage, VLLMStage] = {
    VLLMStage.INSPECT: VLLMStage.SERVER_START,
    VLLMStage.SERVER_START: VLLMStage.BASELINE,
    VLLMStage.BASELINE: VLLMStage.PROFILE_APPROVAL,
    VLLMStage.PROFILE_APPROVAL: VLLMStage.AGENT_DECISION,
    VLLMStage.AGENT_DECISION: VLLMStage.EXPERIMENT,
    VLLMStage.EXPERIMENT: VLLMStage.QUALITY,
    VLLMStage.QUALITY: VLLMStage.DECIDE,
}


class VLLMWorkflowError(RuntimeError):
    """The vLLM coordinator was given stale, incomplete, or unsafe input."""


@runtime_checkable
class VLLMInspectionPort(Protocol):
    """Read-only target inspection boundary."""

    def inspect(self, config: VLLMCampaignConfig) -> MI300XInspectionEvidence: ...


@runtime_checkable
class VLLMServerLifecyclePort(Protocol):
    """Structural subset implemented by :class:`vllm_adapter.VLLMAdapter`."""

    def start_or_resume(self, spec: VLLMServerSpec) -> Any: ...

    def stop(self, *, expected_request_hash: str | None = None) -> Any: ...


@runtime_checkable
class VLLMBenchmarkPort(Protocol):
    """Normalize repeated ``vllm bench serve`` samples into Gate models."""

    def capture_baseline(self, config: VLLMCampaignConfig) -> BaselineResult: ...

    def run_experiment(
        self, config: VLLMCampaignConfig, spec: VLLMExperimentSpec
    ) -> ExperimentResult: ...


@runtime_checkable
class VLLMProfilePort(Protocol):
    """Execute one permit through ROCm Issue Agent after it was consumed."""

    def execute(self, permit: VLLMProfileExecutionPermit) -> VLLMProfileResultEvidence: ...


@runtime_checkable
class VLLMQualityPort(Protocol):
    """External argv adapter whose parsed output is exactly ``QualityResult``."""

    def evaluate(
        self, config: VLLMCampaignConfig, spec: VLLMExperimentSpec
    ) -> QualityResult: ...


@runtime_checkable
class VLLMEnvironmentPort(Protocol):
    """Fresh observed environment capture used by the benchmark Gate."""

    def capture(
        self,
        config: VLLMCampaignConfig,
        *,
        phase: str,
        server_request_hash: str,
    ) -> EnvironmentFingerprint: ...


@runtime_checkable
class VLLMCommandPort(Protocol):
    """Structural subset of :class:`command.CommandRunner`."""

    def run(
        self,
        argv: list[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str],
        unset_env: tuple[str, ...],
        timeout_seconds: float,
    ) -> Any: ...


class LocalVLLMQualityPort:
    """Run the immutable external quality argv and parse only ``QualityResult``."""

    def __init__(self, store: ExperimentStore, runner: VLLMCommandPort) -> None:
        self.store = store
        self.runner = runner

    def evaluate_baseline(self, config: VLLMCampaignConfig) -> QualityResult:
        return self._evaluate(config, label="baseline")

    def evaluate(
        self, config: VLLMCampaignConfig, spec: VLLMExperimentSpec
    ) -> QualityResult:
        return self._evaluate(config, label=spec.id)

    def _evaluate(self, config: VLLMCampaignConfig, *, label: str) -> QualityResult:
        protocol = config.quality_protocol
        output = protocol.output_path
        before = self._file_identity(output)
        command = self.runner.run(
            protocol.argv,
            cwd=protocol.cwd,
            env=protocol.env,
            unset_env=(
                "HIP_VISIBLE_DEVICES",
                "HSA_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
                "HSA_OVERRIDE_GFX_VERSION",
            ),
            timeout_seconds=protocol.timeout_seconds,
        )
        command_ref = self.store.save_evidence_json(
            config.task.id,
            f"vllm/quality/{label}-command",
            _jsonable(command),
            producer="vllm-quality-command",
        )
        if not bool(getattr(command, "succeeded", False)):
            raise VLLMWorkflowError("external vLLM quality command did not succeed")
        after = self._file_identity(output)
        if after is None or after == before:
            raise VLLMWorkflowError(
                "quality command did not freshly create or replace its exact output_path"
            )
        try:
            if output.is_symlink() or not output.is_file():
                raise VLLMWorkflowError("quality output must be a regular non-symlink file")
            if output.stat().st_size > 16 * 1024 * 1024:
                raise VLLMWorkflowError("quality output exceeds the 16 MiB contract limit")
            payload = json.loads(output.read_text(encoding="utf-8"))
            quality = QualityResult.model_validate(payload)
            output_ref = self.store.import_artifact(
                config.task.id,
                f"artifacts/evidence/vllm/quality/{label}-result.json",
                output,
                producer="external-vllm-quality-adapter",
                media_type="application/json",
            )
        except (OSError, UnicodeError, json.JSONDecodeError, StoreError) as error:
            raise VLLMWorkflowError("invalid external vLLM quality output") from error
        return quality.model_copy(
            update={"artifacts": [*quality.artifacts, command_ref, output_ref]}
        )

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, int, int] | None:
        try:
            stat = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        return stat.st_ino, stat.st_size, stat.st_mtime_ns


class LocalVLLMBenchmarkPort:
    """Run repeated strict ``vllm bench serve`` samples with durable raw evidence."""

    def __init__(
        self,
        store: ExperimentStore,
        runner: VLLMCommandPort,
        environment: VLLMEnvironmentPort,
        quality: LocalVLLMQualityPort,
    ) -> None:
        self.store = store
        self.runner = runner
        self.environment = environment
        self.quality = quality

    def capture_baseline(self, config: VLLMCampaignConfig) -> BaselineResult:
        spec = build_server_spec(config, self.store)
        benchmark, artifacts = self._benchmark(config, label="baseline")
        environment = self.environment.capture(
            config,
            phase="baseline",
            server_request_hash=spec.request_hash,
        )
        return BaselineResult(
            environment=environment,
            build_status=RunStatus.REUSED,
            smoke_passed=True,
            benchmark=benchmark,
            quality=self.quality.evaluate_baseline(config),
            artifacts=artifacts,
        )

    def run_experiment(
        self, config: VLLMCampaignConfig, spec: VLLMExperimentSpec
    ) -> ExperimentResult:
        server = build_server_spec(config, self.store, experiment=spec)
        benchmark, artifacts = self._benchmark(config, label=spec.id)
        environment = self.environment.capture(
            config,
            phase=f"experiment:{spec.id}",
            server_request_hash=server.request_hash,
        )
        return ExperimentResult(
            experiment_id=spec.id,
            environment=environment,
            build_status=RunStatus.REUSED,
            smoke_passed=True,
            e2e=benchmark,
            artifacts=artifacts,
        )

    def _benchmark(
        self, config: VLLMCampaignConfig, *, label: str
    ) -> tuple[BenchmarkResult, list[ArtifactRef]]:
        from .vllm_adapter import parse_bench_serve_json

        protocol = config.serving
        attempt_id = uuid.uuid4().hex
        artifacts: list[ArtifactRef] = [
            self.store.save_evidence_json(
                config.task.id,
                f"vllm/benchmark/{label}-attempt",
                {
                    "schema": "amd-inference-opt.vllm-benchmark-attempt.v1",
                    "attempt_id": attempt_id,
                    "label": label,
                    "protocol_sha256": protocol.coordinate_sha256,
                    "base_argv": protocol.benchmark_argv,
                    "warmup_runs": protocol.warmup_runs,
                    "sample_count": protocol.sample_count,
                    "created_at": utc_now().isoformat(),
                },
                producer="vllm-benchmark-adapter",
            )
        ]
        samples: dict[str, list[float]] = {}
        units: dict[str, str] = {}
        total_runs = protocol.warmup_runs + protocol.sample_count
        for index in range(total_runs):
            result_directory = (
                self.store.task_dir(config.task.id)
                / "artifacts"
                / "vllm-bench-results"
                / attempt_id
            )
            result_directory.mkdir(parents=True, exist_ok=True)
            if result_directory.is_symlink() or not result_directory.is_dir():
                raise VLLMWorkflowError("benchmark result directory is not a safe directory")
            result_filename = f"{label}-run-{index + 1:04d}.json"
            result_path = result_directory / result_filename
            if result_path.exists() or result_path.is_symlink():
                raise VLLMWorkflowError(
                    "unique benchmark result path already exists; refusing stale reuse"
                )
            run_argv = [
                *protocol.benchmark_argv,
                "--save-result",
                "--result-dir",
                str(result_directory),
                "--result-filename",
                result_filename,
            ]
            command = self.runner.run(
                run_argv,
                cwd=protocol.cwd,
                env=protocol.server_env,
                unset_env=(
                    "HIP_VISIBLE_DEVICES",
                    "HSA_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                    "GPU_DEVICE_ORDINAL",
                    "HSA_OVERRIDE_GFX_VERSION",
                    "PYTHONHOME",
                    "PYTHONPATH",
                    "PYTHONSTARTUP",
                ),
                timeout_seconds=protocol.timeout_seconds,
            )
            artifacts.append(
                self.store.save_evidence_json(
                    config.task.id,
                    f"vllm/benchmark/{label}-run-{index + 1:04d}",
                    _jsonable(command),
                    producer="vllm-benchmark-command",
                )
            )
            if not bool(getattr(command, "succeeded", False)):
                raise VLLMWorkflowError(
                    f"vLLM benchmark command {index + 1} did not succeed"
                )
            if (
                not result_path.is_file()
                or result_path.is_symlink()
                or result_path.stat().st_size > 64 * 1024 * 1024
            ):
                raise VLLMWorkflowError(
                    "vLLM bench did not freshly write its exact regular JSON result"
                )
            parsed = parse_bench_serve_json(
                result_path,
                expected_num_prompts=protocol.num_prompts,
            )
            artifacts.append(
                self.store.import_artifact(
                    config.task.id,
                    (
                        "artifacts/evidence/vllm/benchmark/"
                        f"{attempt_id}-run-{index + 1:04d}.json"
                    ),
                    result_path,
                    producer="vllm-benchmark-adapter",
                    media_type="application/json",
                )
            )
            if index < protocol.warmup_runs:
                continue
            for name, metric in parsed.canonical_metrics.items():
                samples.setdefault(name, []).append(metric.value)
                units[name] = metric.unit
        metrics = {
            name: MetricSeries(unit=units[name], samples=values)
            for name, values in samples.items()
        }
        missing = set(protocol.required_metrics) - set(metrics)
        if missing:
            raise VLLMWorkflowError(
                "parsed bench output lacks canonical metrics: "
                + ", ".join(sorted(missing))
            )
        return BenchmarkResult(status=RunStatus.SUCCEEDED, metrics=metrics), artifacts


VLLMProfileEvidenceBuilder = Callable[
    [Mapping[str, Any], VLLMProfileExecutionPermit], VLLMProfileResultEvidence
]
VLLMMCPClientFactory = Callable[
    [MCPServerConfig, Mapping[str, Any]], AbstractAsyncContextManager[RocmIssueAgentClient]
]
VLLMProfileMaterializer = Callable[..., Any]
VLLMProfileContractReader = Callable[[MCPServerConfig], ProfileContract]


def _read_profile_contract(server: MCPServerConfig) -> ProfileContract:
    async def inspect() -> ProfileContract:
        async with RocmIssueAgentClient(server) as client:
            return await client.profile_contract()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(inspect())
    raise VLLMWorkflowError("profile contract discovery must run outside an active event loop")


class RocmMCPVLLMProfilePort:
    """Transport an already-consumed exact permit to ROCm Issue Agent once."""

    def __init__(
        self,
        config: VLLMCampaignConfig,
        *,
        resolved_environment: Mapping[str, str],
        evidence_builder: VLLMProfileEvidenceBuilder,
        client_factory: VLLMMCPClientFactory | None = None,
    ) -> None:
        self.config = config
        self.environment = dict(resolved_environment)
        self.evidence_builder = evidence_builder
        self.client_factory = client_factory or self._default_client

    @staticmethod
    def _default_client(
        server: MCPServerConfig, context: Mapping[str, Any]
    ) -> RocmIssueAgentClient:
        return RocmIssueAgentClient(server, approval_context=context)

    def _server_for_permit(self, permit: VLLMProfileExecutionPermit) -> MCPServerConfig:
        environment_sha = canonical_sha256(dict(sorted(self.environment.items())))
        if permit.execution_context.get("mcp_env_sha256") != environment_sha:
            raise VLLMWorkflowError("profile port environment differs from approval context")
        if not permit.execution_context.get("profile_contract_sha256"):
            raise VLLMWorkflowError("profile permit lacks capability negotiation; request approval")
        domain = self.config.task.mcp
        command = tuple(domain.command)
        return MCPServerConfig(
            command=command[0],
            args=command[1:],
            env=self.environment,
            cwd=str(domain.cwd) if domain.cwd is not None else None,
        )

    async def execute_async(
        self, permit: VLLMProfileExecutionPermit
    ) -> VLLMProfileResultEvidence:
        server = self._server_for_permit(permit)
        async with self.client_factory(server, permit.execution_context) as client:
            call: MCPToolCall = await client.call_tool(
                permit.tool,
                permit.arguments,
                approval_sha256=permit.request_hash,
            )
        return self.evidence_builder(call.to_dict(), permit)

    def preflight(self, permit: VLLMProfileExecutionPermit) -> None:
        """Check capability drift before the coordinator consumes the receipt."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.preflight_async(permit))
        raise VLLMWorkflowError("await preflight_async() when running inside an event loop")

    async def preflight_async(self, permit: VLLMProfileExecutionPermit) -> None:
        server = self._server_for_permit(permit)
        async with self.client_factory(server, permit.execution_context) as client:
            await client.check_profile_contract(
                permit.arguments,
                expected_sha256=permit.execution_context.get("profile_contract_sha256"),
            )

    def execute(
        self, permit: VLLMProfileExecutionPermit
    ) -> VLLMProfileResultEvidence:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute_async(permit))
        raise VLLMWorkflowError(
            "execute() cannot run inside an event loop; await execute_async()"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise VLLMWorkflowError(
        "adapter evidence must be a Pydantic model, dataclass, or mapping"
    )


def _lookup(payload: Any, *names: str) -> Any:
    current = payload
    for name in names:
        if isinstance(current, BaseModel):
            current = getattr(current, name, None)
        elif isinstance(current, Mapping):
            current = current.get(name)
        else:
            current = getattr(current, name, None)
    return current


class VLLMWorkflowEngine:
    """Pure state transitions with mandatory hash-bound evidence."""

    @staticmethod
    def new(config: VLLMCampaignConfig, config_artifact: ArtifactRef) -> VLLMWorkflowRecord:
        if config_artifact.path != VLLM_CONFIG_PATH:
            raise VLLMWorkflowError("vLLM workflow config artifact has an unexpected path")
        return VLLMWorkflowRecord(
            task_id=config.task.id,
            config_sha256=config_artifact.sha256,
        )

    @staticmethod
    def required_evidence(stage: VLLMStage) -> frozenset[str]:
        return VLLM_STAGE_REQUIREMENTS[stage]

    def complete_stage(
        self,
        record: VLLMWorkflowRecord,
        evidence: Mapping[str, ArtifactRef],
        *,
        gate_decision: GateDecision | None = None,
    ) -> VLLMWorkflowRecord:
        if record.status != WorkflowStatus.ACTIVE:
            raise VLLMWorkflowError(f"vLLM workflow is terminal: {record.status}")
        required = VLLM_STAGE_REQUIREMENTS[record.current_stage]
        provided = set(evidence)
        missing = required - provided
        unexpected = provided - required
        if missing:
            raise VLLMWorkflowError(
                f"cannot complete {record.current_stage}: missing evidence "
                + ", ".join(sorted(missing))
            )
        if unexpected:
            raise VLLMWorkflowError(
                f"cannot complete {record.current_stage}: unexpected evidence "
                + ", ".join(sorted(unexpected))
            )

        stage = record.current_stage
        completions = [
            *record.completions,
            VLLMStageCompletion(stage=stage, evidence=dict(evidence)),
        ]
        updates: dict[str, Any] = {
            "completions": completions,
            "revision": record.revision + 1,
            "updated_at": utc_now(),
        }
        if stage == VLLMStage.AGENT_DECISION:
            updates["experiment_count"] = record.experiment_count + 1
        if stage == VLLMStage.DECIDE:
            if gate_decision is None:
                raise VLLMWorkflowError("DECIDE requires the parsed GateDecision")
            updates["terminal_decision"] = gate_decision
            updates["status"] = {
                DecisionOutcome.ACCEPT: WorkflowStatus.ACCEPTED,
                DecisionOutcome.REJECT: WorkflowStatus.REJECTED,
                DecisionOutcome.INCONCLUSIVE: WorkflowStatus.INCONCLUSIVE,
            }[gate_decision.outcome]
        else:
            if gate_decision is not None:
                raise VLLMWorkflowError("gate_decision is valid only at DECIDE")
            updates["current_stage"] = _NEXT_STAGE[stage]
        return VLLMWorkflowRecord.model_validate(
            {**record.model_dump(), **updates}
        )

    @staticmethod
    def terminate_inconclusive(
        record: VLLMWorkflowRecord,
        gate_decision: GateDecision,
        gate_artifact: ArtifactRef,
        *,
        extra_evidence: Mapping[str, ArtifactRef] | None = None,
    ) -> VLLMWorkflowRecord:
        """Fail closed without manufacturing a successful profile artifact."""

        if record.status != WorkflowStatus.ACTIVE:
            raise VLLMWorkflowError("only active workflows may become inconclusive")
        if gate_decision.outcome != DecisionOutcome.INCONCLUSIVE:
            raise VLLMWorkflowError("early termination requires INCONCLUSIVE")
        completion = VLLMStageCompletion(
            stage=record.current_stage,
            evidence={"gate_decision": gate_artifact, **(extra_evidence or {})},
        )
        return VLLMWorkflowRecord.model_validate(
            {
                **record.model_dump(),
                "completions": [*record.completions, completion],
                "status": WorkflowStatus.INCONCLUSIVE,
                "terminal_decision": gate_decision,
                "revision": record.revision + 1,
                "updated_at": utc_now(),
            }
        )

    @staticmethod
    def resume_inconclusive(
        record: VLLMWorkflowRecord, *, profile_available: bool
    ) -> VLLMWorkflowRecord:
        if record.status != WorkflowStatus.INCONCLUSIVE:
            raise VLLMWorkflowError("only an INCONCLUSIVE vLLM workflow can resume")
        decision = record.terminal_decision
        if decision is None or decision.rerun_from_stage is None:
            raise VLLMWorkflowError("INCONCLUSIVE GateDecision has no rerun stage")
        mapping = {
            WorkflowStage.CAPTURE_BASELINE: VLLMStage.SERVER_START,
            WorkflowStage.DISCOVER_HOTSPOTS: VLLMStage.PROFILE_APPROVAL,
            WorkflowStage.PATCH_AND_BUILD: VLLMStage.EXPERIMENT,
            WorkflowStage.E2E_VALIDATION: VLLMStage.EXPERIMENT,
            WorkflowStage.QUALITY_VALIDATION: VLLMStage.EXPERIMENT,
        }
        stage = mapping.get(decision.rerun_from_stage)
        if stage is None:
            raise VLLMWorkflowError(
                f"vLLM V0 cannot resume Gate stage {decision.rerun_from_stage}"
            )
        if stage == VLLMStage.PROFILE_APPROVAL and not profile_available:
            raise VLLMWorkflowError(
                "profiling remains unavailable; create a corrected immutable task"
            )
        updates: dict[str, Any] = {
            "current_stage": stage,
            "status": WorkflowStatus.ACTIVE,
            "terminal_decision": None,
            "rerun_count": record.rerun_count + 1,
            "revision": record.revision + 1,
            "updated_at": utc_now(),
        }
        if stage == VLLMStage.SERVER_START:
            updates.update(
                {
                    "baseline_server_request_hash": None,
                    "baseline_server_requires_stop": False,
                    "baseline_server_start_failure": None,
                    "baseline_run_failure": None,
                    "baseline_server_stop": None,
                    "profile_approval": None,
                }
            )
        elif stage == VLLMStage.PROFILE_APPROVAL:
            updates.update(
                {"profile_approval": None, "profile_execution_failure": None}
            )
        elif stage == VLLMStage.EXPERIMENT:
            updates.update(
                {
                    "candidate_server_request_hash": None,
                    "candidate_server_requires_stop": False,
                    "candidate_server_start_failure": None,
                    "candidate_run_failure": None,
                    "candidate_server_start": None,
                    "pending_quality_result": None,
                    "pending_candidate_result": None,
                }
            )
        return VLLMWorkflowRecord.model_validate({**record.model_dump(), **updates})


class VLLMWorkflowCoordinator:
    """ExperimentStore-backed facade suitable for a future CLI command."""

    def __init__(
        self,
        store: ExperimentStore | str | Path,
        *,
        gate: GateEngine | None = None,
        profile_materializer: VLLMProfileMaterializer | None = None,
        profile_contract_reader: VLLMProfileContractReader | None = None,
    ) -> None:
        self.store = store if isinstance(store, ExperimentStore) else ExperimentStore(store)
        self.engine = VLLMWorkflowEngine()
        self.gate = gate or GateEngine()
        self.profile_materializer = profile_materializer
        self.profile_contract_reader = profile_contract_reader or _read_profile_contract

    @staticmethod
    def _load_snapshot_manifest(path: Path) -> VLLMModelSnapshotManifest:
        """Load a regular external manifest without following a manifest symlink."""

        if path.is_symlink():
            raise VLLMWorkflowError("model snapshot manifest must not be a symlink")
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_file():
                raise VLLMWorkflowError("model snapshot manifest is not a regular file")
            return VLLMModelSnapshotManifest.model_validate_json(
                resolved.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise VLLMWorkflowError(
                f"invalid model snapshot manifest: {path}"
            ) from error

    @staticmethod
    def _validate_snapshot_coordinate(
        config: VLLMCampaignConfig,
        manifest: VLLMModelSnapshotManifest,
    ) -> None:
        model = config.model
        try:
            model_root = model.local_path.resolve(strict=True)
            manifest_root = manifest.root.resolve(strict=True)
            manifest_path = model.snapshot_manifest_path.resolve(strict=True)
        except OSError as error:
            raise VLLMWorkflowError(
                "model snapshot paths must exist before campaign creation or resume"
            ) from error
        if manifest_path == model_root or model_root in manifest_path.parents:
            raise VLLMWorkflowError(
                "model snapshot manifest must live outside the hashed snapshot directory"
            )
        mismatches = [
            name
            for name, observed, expected in (
                ("root", manifest_root, model_root),
                ("model_id", manifest.model_id, model.model_id),
                ("revision", manifest.revision, model.revision),
                (
                    "tokenizer_revision",
                    manifest.tokenizer_revision,
                    model.tokenizer_revision,
                ),
                ("snapshot_digest", manifest.snapshot_digest, model.snapshot_digest),
            )
            if observed != expected
        ]
        if mismatches:
            raise VLLMWorkflowError(
                "model snapshot manifest differs from immutable config: "
                + ", ".join(mismatches)
            )
        try:
            verify_vllm_model_snapshot(manifest)
        except VLLMModelSnapshotError as error:
            raise VLLMWorkflowError(str(error)) from error

    def _verify_model_snapshot(
        self,
        config: VLLMCampaignConfig,
        *,
        require_persisted: bool,
    ) -> VLLMModelSnapshotManifest:
        external = self._load_snapshot_manifest(config.model.snapshot_manifest_path)
        self._validate_snapshot_coordinate(config, external)
        if require_persisted:
            try:
                reference = self.store.artifact_ref(
                    config.task.id, VLLM_MODEL_SNAPSHOT_MANIFEST_PATH
                )
                if reference is None or not self.store.verify_artifact(
                    config.task.id, reference
                ):
                    raise VLLMWorkflowError(
                        "persisted model snapshot manifest failed artifact verification"
                    )
                persisted = self.store.load_json(
                    config.task.id,
                    VLLM_MODEL_SNAPSHOT_MANIFEST_PATH,
                    VLLMModelSnapshotManifest,
                )
            except StoreError as error:
                raise VLLMWorkflowError(str(error)) from error
            if persisted.model_dump(mode="json", by_alias=True) != external.model_dump(
                mode="json", by_alias=True
            ):
                raise VLLMWorkflowError(
                    "external model snapshot manifest differs from its immutable import"
                )
        return external

    def create(self, config: VLLMCampaignConfig) -> VLLMWorkflowRecord:
        """Persist immutable config and revision-zero state; perform no inspection."""

        manifest = self._verify_model_snapshot(config, require_persisted=False)
        try:
            self.store.create_task(config.task)
            self.store.save_immutable_json(
                config.task.id,
                VLLM_MODEL_SNAPSHOT_MANIFEST_PATH,
                manifest,
                producer="vllm-model-snapshot",
            )
            config_ref = self.store.save_immutable_json(
                config.task.id,
                VLLM_CONFIG_PATH,
                config,
                producer="vllm-workflow",
            )
            record = self.engine.new(config, config_ref)
            self.store.save_json(
                config.task.id,
                VLLM_WORKFLOW_PATH,
                record,
                producer="vllm-workflow",
            )
            self.store.append_event(
                config.task.id,
                "vllm_workflow_created",
                {
                    "config_sha256": config_ref.sha256,
                    "identity_sha256": config.identity_sha256,
                    "stage": record.current_stage.value,
                },
            )
            return record
        except StoreError as error:
            raise VLLMWorkflowError(str(error)) from error

    def _load_config_artifact(self, task_id: str) -> VLLMCampaignConfig:
        try:
            reference = self.store.artifact_ref(task_id, VLLM_CONFIG_PATH)
            if reference is None or not self.store.verify_artifact(task_id, reference):
                raise VLLMWorkflowError("vLLM config failed artifact verification")
            config = self.store.load_json(task_id, VLLM_CONFIG_PATH, VLLMCampaignConfig)
            if config.task.id != task_id:
                raise VLLMWorkflowError("vLLM config task id does not match its directory")
            return config
        except StoreError as error:
            raise VLLMWorkflowError(str(error)) from error

    def load_config(self, task_id: str) -> VLLMCampaignConfig:
        config = self._load_config_artifact(task_id)
        self._verify_model_snapshot(config, require_persisted=True)
        return config

    def load(self, task_id: str) -> VLLMWorkflowRecord:
        try:
            self.load_config(task_id)
            record = self.store.load_json(
                task_id, VLLM_WORKFLOW_PATH, VLLMWorkflowRecord
            )
            config_ref = self.store.artifact_ref(task_id, VLLM_CONFIG_PATH)
            if record.config_sha256 != config_ref.sha256:
                raise VLLMWorkflowError("workflow is bound to a different config digest")
            if not self.store.verify_artifact(task_id, config_ref):
                raise VLLMWorkflowError("workflow config artifact is corrupt")
            return record
        except StoreError as error:
            raise VLLMWorkflowError(str(error)) from error

    def _save_update(
        self, previous: VLLMWorkflowRecord, updated: VLLMWorkflowRecord
    ) -> VLLMWorkflowRecord:
        if updated.task_id != previous.task_id:
            raise VLLMWorkflowError("workflow update changed task id")
        try:
            with self.store.task_lock(previous.task_id):
                persisted = self.store.load_json(
                    previous.task_id, VLLM_WORKFLOW_PATH, VLLMWorkflowRecord
                )
                if persisted.revision != previous.revision:
                    raise VLLMWorkflowError(
                        "vLLM workflow revision conflict: "
                        f"expected {previous.revision}, found {persisted.revision}"
                    )
                if updated.revision != previous.revision + 1:
                    raise VLLMWorkflowError("workflow updates must increment revision once")
                if updated.config_sha256 != previous.config_sha256:
                    raise VLLMWorkflowError("workflow update changed immutable config")
                self.store.save_json(
                    previous.task_id,
                    VLLM_WORKFLOW_PATH,
                    updated,
                    producer="vllm-workflow",
                )
                self.store.append_event(
                    previous.task_id,
                    "vllm_workflow_saved",
                    {
                        "revision": updated.revision,
                        "stage": updated.current_stage.value,
                        "status": updated.status.value,
                    },
                )
                return updated
        except StoreError as error:
            raise VLLMWorkflowError(str(error)) from error

    def advance(
        self,
        record: VLLMWorkflowRecord,
        evidence: Mapping[str, ArtifactRef],
        *,
        gate_decision: GateDecision | None = None,
    ) -> VLLMWorkflowRecord:
        for name, artifact in evidence.items():
            if not self.store.verify_artifact(record.task_id, artifact):
                raise VLLMWorkflowError(f"{name} failed artifact integrity verification")
        updated = self.engine.complete_stage(
            record, evidence, gate_decision=gate_decision
        )
        return self._save_update(record, updated)

    def _save_evidence(
        self,
        task_id: str,
        logical_name: str,
        value: Any,
        *,
        producer: str,
    ) -> ArtifactRef:
        try:
            return self.store.save_evidence_json(
                task_id,
                logical_name,
                _jsonable(value),
                producer=producer,
            )
        except StoreError as error:
            raise VLLMWorkflowError(str(error)) from error

    @staticmethod
    def _adapter_timestamp(value: Any, *, label: str) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise VLLMWorkflowError(
                    f"adapter {label} timestamp is invalid"
                ) from error
        else:
            raise VLLMWorkflowError(f"adapter {label} timestamp is missing")
        if parsed.tzinfo is None:
            raise VLLMWorkflowError(f"adapter {label} timestamp must include a timezone")
        return parsed

    @staticmethod
    def _gpu_budget_seconds(config: VLLMCampaignConfig) -> float:
        return config.task.budgets.max_gpu_minutes * 60.0

    def _projected_gpu_seconds(
        self, record: VLLMWorkflowRecord, *, now: datetime | None = None
    ) -> float:
        projected = record.gpu_seconds_used
        if record.active_gpu_started_at is not None:
            current = now or utc_now()
            projected += max(
                0.0, (current - record.active_gpu_started_at).total_seconds()
            )
        return projected

    def _budget_exhausted(
        self, record: VLLMWorkflowRecord, config: VLLMCampaignConfig
    ) -> bool:
        return self._projected_gpu_seconds(record) >= self._gpu_budget_seconds(config)

    def _with_active_server(
        self,
        record: VLLMWorkflowRecord,
        adapter_record: Any,
        *,
        phase: str,
    ) -> VLLMWorkflowRecord:
        if record.active_gpu_started_at is not None:
            raise VLLMWorkflowError("another GPU process is already accounted as active")
        started_at = self._adapter_timestamp(
            _lookup(adapter_record, "started_at"), label="server start"
        )
        return record.model_copy(
            update={
                "active_gpu_started_at": started_at,
                "active_gpu_phase": phase,
            }
        )

    def _server_stop_accounting(
        self, record: VLLMWorkflowRecord, result: Any
    ) -> tuple[float, dict[str, float]]:
        if record.active_gpu_started_at is None or record.active_gpu_phase is None:
            raise VLLMWorkflowError("active GPU start time is missing before server stop")
        stopped_at = self._adapter_timestamp(
            _lookup(result, "record", "stopped_at"), label="server stop"
        )
        elapsed = (stopped_at - record.active_gpu_started_at).total_seconds()
        if elapsed < 0:
            raise VLLMWorkflowError("adapter server stop predates its recorded start")
        by_phase = dict(record.gpu_seconds_by_phase)
        by_phase[record.active_gpu_phase] = (
            by_phase.get(record.active_gpu_phase, 0.0) + elapsed
        )
        return record.gpu_seconds_used + elapsed, by_phase

    def _record_profile_gpu_seconds(
        self, record: VLLMWorkflowRecord, elapsed: float
    ) -> VLLMWorkflowRecord:
        if elapsed < 0:
            raise VLLMWorkflowError("profile GPU duration cannot be negative")
        by_phase = dict(record.gpu_seconds_by_phase)
        by_phase["offline_profile"] = by_phase.get("offline_profile", 0.0) + elapsed
        updated = record.model_copy(
            update={
                "gpu_seconds_used": record.gpu_seconds_used + elapsed,
                "gpu_seconds_by_phase": by_phase,
                "revision": record.revision + 1,
                "updated_at": utc_now(),
            }
        )
        return self._save_update(record, updated)

    def finalize_gpu_budget(
        self,
        record: VLLMWorkflowRecord,
        *,
        extra_evidence: Mapping[str, ArtifactRef] | None = None,
    ) -> VLLMWorkflowRecord:
        """Close a task after its immutable paid-GPU budget is exhausted."""

        config = self.load_config(record.task_id)
        return self._finalize_gpu_budget(
            record, config, extra_evidence=extra_evidence
        )

    def _finalize_gpu_budget(
        self,
        record: VLLMWorkflowRecord,
        config: VLLMCampaignConfig,
        *,
        extra_evidence: Mapping[str, ArtifactRef] | None = None,
    ) -> VLLMWorkflowRecord:
        if record.active_gpu_started_at is not None:
            raise VLLMWorkflowError("the exact active GPU process must stop first")
        if record.gpu_seconds_used < self._gpu_budget_seconds(config):
            raise VLLMWorkflowError("GPU budget is not exhausted")
        decision = GateDecision(
            outcome=DecisionOutcome.INCONCLUSIVE,
            checks=[
                GateCheck(
                    name="gpu_time_budget",
                    passed=None,
                    detail=(
                        f"used {record.gpu_seconds_used:.3f}s of "
                        f"{self._gpu_budget_seconds(config):.3f}s"
                    ),
                )
            ],
            reasons=["paid GPU time budget was exhausted"],
        )
        gate_ref = self._save_evidence(
            record.task_id, "vllm/gpu-budget-gate", decision, producer="gate"
        )
        updated = self.engine.terminate_inconclusive(
            record,
            decision,
            gate_ref,
            extra_evidence=extra_evidence,
        )
        return self._save_update(record, updated)

    def record_inspection(
        self,
        record: VLLMWorkflowRecord,
        inspection: MI300XInspectionEvidence | Mapping[str, Any],
    ) -> VLLMWorkflowRecord:
        if record.current_stage != VLLMStage.INSPECT:
            raise VLLMWorkflowError("inspection evidence is valid only at INSPECT")
        config = self.load_config(record.task_id)
        observed = MI300XInspectionEvidence.model_validate(inspection)
        raw_artifacts = {
            "amd_smi_artifact": observed.amd_smi_artifact,
            "hip_probe_artifact": observed.hip_probe_artifact,
        }
        corrupt_raw = [
            name
            for name, artifact in raw_artifacts.items()
            if not self.store.verify_artifact(record.task_id, artifact)
        ]
        if corrupt_raw:
            raise VLLMWorkflowError(
                "MI300X raw inspection evidence failed integrity verification: "
                + ", ".join(sorted(corrupt_raw))
            )
        expected = config.device
        mismatches: list[str] = []
        comparisons = {
            "image_digest": (observed.image_digest, config.task.runtime.image_digest),
            "scope": (
                observed.scope,
                (
                    "container_preflight"
                    if config.task.runtime.image_digest is not None
                    else "native_preflight"
                ),
            ),
            "gfx_target": (observed.gfx_target, expected.gfx_target),
            "product_name": (observed.product_name, expected.product_name),
            "oam_id": (observed.oam_id, expected.oam_id),
            "xcc_count": (observed.xcc_count, expected.xcc_count),
            "device_uuid": (observed.device_uuid, expected.device_uuid),
            "pci_bdf": (observed.pci_bdf, expected.pci_bdf),
            "compute_partition": (
                observed.compute_partition,
                expected.compute_partition,
            ),
            "memory_partition": (
                observed.memory_partition,
                expected.memory_partition,
            ),
            "partition_id": (observed.partition_id, expected.partition_id),
            "rocr_visible_devices": (
                observed.rocr_visible_devices,
                expected.stable_device_id,
            ),
            "rocm_version": (observed.rocm_version, config.task.runtime.rocm_version),
            "vllm_version": (observed.vllm_version, config.task.runtime.version),
            "pytorch_version": (
                observed.pytorch_version,
                config.task.runtime.pytorch_version,
            ),
            "python_version": (
                observed.python_version,
                config.task.runtime.python_version,
            ),
            "launcher_sha256": (
                observed.launcher_sha256,
                config.serving.engine_config["launcher_sha256"],
            ),
            "environment_manifest_sha256": (
                observed.environment_manifest_sha256,
                config.task.runtime.environment_manifest_sha256,
            ),
            "amd_smi_command_sha256": (
                observed.amd_smi_command_sha256,
                expected.amd_smi_command_sha256,
            ),
            "hip_probe_command_sha256": (
                observed.hip_probe_command_sha256,
                expected.hip_probe_command_sha256,
            ),
        }
        for name, (actual, configured) in comparisons.items():
            if actual != configured:
                mismatches.append(f"{name}: observed {actual!r}, configured {configured!r}")
        if mismatches:
            raise VLLMWorkflowError(
                "MI300X inspection does not match the immutable task: "
                + "; ".join(mismatches)
            )
        artifact = self._save_evidence(
            record.task_id, "vllm/inspection", observed, producer="inspection-adapter"
        )
        return self.advance(record, {"inspection": artifact})

    def inspect(
        self, record: VLLMWorkflowRecord, adapter: VLLMInspectionPort
    ) -> VLLMWorkflowRecord:
        return self.record_inspection(record, adapter.inspect(self.load_config(record.task_id)))

    @staticmethod
    def _server_started(result: Any, spec: VLLMServerSpec) -> bool:
        status = _lookup(result, "record", "status")
        if hasattr(status, "value"):
            status = status.value
        healthy = _lookup(result, "health", "healthy")
        if healthy is None:
            healthy = _lookup(result, "health", "ok")
        identity_verified = _lookup(result, "identity_verified")
        if callable(identity_verified):
            identity_verified = identity_verified()
        gate_eligible = _lookup(result, "gate_eligible")
        if callable(gate_eligible):
            gate_eligible = gate_eligible()
        request_hash = _lookup(result, "record", "request_hash")
        identity = _lookup(result, "record", "identity")
        runtime_verified = _lookup(result, "record", "runtime_evidence", "verified")
        if callable(runtime_verified):
            runtime_verified = runtime_verified()
        argv = _lookup(result, "record", "argv")
        cwd = _lookup(result, "record", "cwd")
        runtime_evidence = _lookup(result, "record", "runtime_evidence")
        observed_pid = _lookup(runtime_evidence, "observed_pid")
        observed_boot_id = _lookup(runtime_evidence, "observed_boot_id")
        observed_start_ticks = _lookup(runtime_evidence, "observed_start_ticks")
        image_bound = spec.image_digest is None or all(
            (
                _lookup(runtime_evidence, "image_digest_matches") is True,
                _lookup(runtime_evidence, "container_binding_kind") == "cgroup_v2",
                _lookup(runtime_evidence, "process_binding_id") is not None,
                _lookup(runtime_evidence, "process_binding_id")
                == _lookup(runtime_evidence, "container_binding_id"),
            )
        )
        return all(
            (
                str(status).upper() in {"RUNNING", "HEALTHY", "READY"},
                healthy is True,
                identity_verified is True,
                gate_eligible is True,
                runtime_verified is True,
                request_hash == spec.request_hash,
                identity == spec.identity,
                tuple(argv or ()) == spec.argv,
                cwd == spec.cwd,
                _lookup(runtime_evidence, "native_executable_matches") is True,
                _lookup(runtime_evidence, "pid_executable_sha256")
                == spec.native_executable_sha256,
                _lookup(runtime_evidence, "observed_environment_manifest_sha256")
                == spec.environment_manifest_sha256,
                _lookup(runtime_evidence, "declared_environment_matches") is True,
                _lookup(runtime_evidence, "unset_environment_absent") is True,
                _lookup(runtime_evidence, "rocr_visible_devices")
                == spec.env.get("ROCR_VISIBLE_DEVICES"),
                _lookup(runtime_evidence, "hip_visible_devices")
                == spec.env.get("HIP_VISIBLE_DEVICES"),
                observed_pid == _lookup(result, "record", "pid"),
                observed_boot_id == _lookup(result, "record", "boot_id"),
                observed_start_ticks == _lookup(result, "record", "start_ticks"),
                image_bound,
            )
        )

    def record_server_start(
        self, record: VLLMWorkflowRecord, result: Any
    ) -> VLLMWorkflowRecord:
        if record.current_stage != VLLMStage.SERVER_START:
            raise VLLMWorkflowError("server evidence is valid only at SERVER_START")
        config = self.load_config(record.task_id)
        spec = build_server_spec(config, self.store)
        if not self._server_started(result, spec):
            if _lookup(result, "record", "requires_explicit_stop") is True:
                return self._record_start_failure(
                    record,
                    spec,
                    _lookup(result, "record"),
                    candidate=False,
                    message="server result failed its identity/health gate",
                )
            raise VLLMWorkflowError(
                "vLLM server is not healthy with verified immutable runtime identity"
            )
        artifact = self._save_evidence(
            record.task_id,
            "vllm/server-start",
            result,
            producer="vllm-adapter",
        )
        active = self._with_active_server(
            record, _lookup(result, "record"), phase="baseline_server"
        )
        seeded = active.model_copy(
            update={"baseline_server_request_hash": spec.request_hash}
        )
        return self.advance(seeded, {"server_start": artifact})

    def start_server(
        self, record: VLLMWorkflowRecord, adapter: VLLMServerLifecyclePort
    ) -> VLLMWorkflowRecord:
        config = self.load_config(record.task_id)
        if self._budget_exhausted(record, config):
            return self.finalize_gpu_budget(record)
        spec = build_server_spec(config, self.store)
        try:
            result = adapter.start_or_resume(spec)
        except Exception as error:
            if getattr(error, "requires_explicit_stop", False):
                return self._record_start_failure(
                    record,
                    spec,
                    getattr(error, "record", None),
                    candidate=False,
                    message=f"{type(error).__name__}: {error}",
                )
            raise
        return self.record_server_start(record, result)

    def _record_start_failure(
        self,
        record: VLLMWorkflowRecord,
        spec: VLLMServerSpec,
        adapter_record: Any,
        *,
        candidate: bool,
        message: str,
    ) -> VLLMWorkflowRecord:
        if adapter_record is None:
            raise VLLMWorkflowError(
                "adapter requires cleanup but supplied no exact process record"
            )
        if (
            _lookup(adapter_record, "request_hash") != spec.request_hash
            or _lookup(adapter_record, "identity") != spec.identity
            or _lookup(adapter_record, "requires_explicit_stop") is not True
        ):
            raise VLLMWorkflowError(
                "unsafe server failure record does not bind the requested process"
            )
        label = "candidate" if candidate else "baseline"
        failure_ref = self._save_evidence(
            record.task_id,
            f"vllm/{label}-server-start-failure",
            {
                "schema": "amd-inference-opt.vllm-server-start-failure.v1",
                "message": message,
                "request_hash": spec.request_hash,
                "record": _jsonable(adapter_record),
            },
            producer="vllm-adapter",
        )
        updates: dict[str, Any] = {
            "revision": record.revision + 1,
            "updated_at": utc_now(),
        }
        active = self._with_active_server(
            record,
            adapter_record,
            phase="candidate_server" if candidate else "baseline_server",
        )
        if candidate:
            updates.update(
                {
                    "candidate_server_request_hash": spec.request_hash,
                    "candidate_server_requires_stop": True,
                    "candidate_server_start_failure": failure_ref,
                }
            )
        else:
            updates.update(
                {
                    "baseline_server_request_hash": spec.request_hash,
                    "baseline_server_requires_stop": True,
                    "baseline_server_start_failure": failure_ref,
                }
            )
        updated = VLLMWorkflowRecord.model_validate({**active.model_dump(), **updates})
        return self._save_update(record, updated)

    @staticmethod
    def _require_benchmark_metrics(
        config: VLLMCampaignConfig, result: BaselineResult | ExperimentResult
    ) -> None:
        benchmark = result.benchmark if isinstance(result, BaselineResult) else result.e2e
        if benchmark is None:
            raise VLLMWorkflowError("vLLM benchmark result is missing")
        missing = set(config.serving.required_metrics) - set(benchmark.metrics)
        if missing:
            raise VLLMWorkflowError(
                "vLLM benchmark is missing metrics: " + ", ".join(sorted(missing))
            )

    @staticmethod
    def _require_environment_coordinates(
        config: VLLMCampaignConfig,
        result: BaselineResult | ExperimentResult,
        *,
        server_request_hash: str,
    ) -> None:
        environment = result.environment
        if environment is None:
            raise VLLMWorkflowError("vLLM run is missing an environment fingerprint")
        mismatches = [
            f"{name}: {environment.values.get(name)!r} != {expected!r}"
            for name, expected in config.environment_coordinates.items()
            if environment.values.get(name) != expected
        ]
        if environment.source != "observed":
            mismatches.append("environment source is not observed")
        if environment.capture_id is None or environment.captured_at is None:
            mismatches.append("fresh capture id/timestamp is missing")
        if environment.values.get("server_request_hash") != server_request_hash:
            mismatches.append("benchmark is not bound to the active managed server request")
        if mismatches:
            raise VLLMWorkflowError(
                "vLLM run identity does not match immutable config: "
                + "; ".join(mismatches)
            )

    @staticmethod
    def _require_baseline_quality(
        config: VLLMCampaignConfig, baseline: BaselineResult
    ) -> None:
        quality = baseline.quality
        if quality is None or quality.status != RunStatus.SUCCEEDED:
            raise VLLMWorkflowError("baseline must include successful external quality evidence")
        if quality.coordinate_hash != config.quality_protocol.coordinate_sha256:
            raise VLLMWorkflowError("baseline quality protocol hash does not match config")
        if quality.representation_hash != config.model.snapshot_digest:
            raise VLLMWorkflowError("baseline quality is not bound to the model snapshot")
        if config.task.quality.require_correctness and quality.correctness_passed is None:
            raise VLLMWorkflowError("baseline quality lacks a correctness verdict")
        missing_accuracies = {
            requirement.metric
            for requirement in config.task.quality.resolved_accuracy_requirements()
            if requirement.metric not in quality.accuracies
        }
        if missing_accuracies:
            raise VLLMWorkflowError(
                "baseline quality lacks required accuracies: "
                + ", ".join(sorted(missing_accuracies))
            )
        if (
            config.task.quality.max_ppl_regression_percent is not None
            and quality.perplexity is None
        ):
            raise VLLMWorkflowError("baseline quality lacks required perplexity")

    def record_baseline(
        self, record: VLLMWorkflowRecord, baseline: BaselineResult
    ) -> VLLMWorkflowRecord:
        if record.current_stage != VLLMStage.BASELINE:
            raise VLLMWorkflowError("baseline evidence is valid only at BASELINE")
        config = self.load_config(record.task_id)
        self._require_benchmark_metrics(config, baseline)
        if record.baseline_server_request_hash is None:
            raise VLLMWorkflowError("baseline server request identity is missing")
        self._require_environment_coordinates(
            config,
            baseline,
            server_request_hash=record.baseline_server_request_hash,
        )
        self._require_baseline_quality(config, baseline)
        artifact = self._save_evidence(
            record.task_id,
            "vllm/baseline",
            baseline,
            producer="vllm-benchmark-adapter",
        )
        return self.advance(record, {"baseline": artifact})

    def capture_baseline(
        self, record: VLLMWorkflowRecord, adapter: VLLMBenchmarkPort
    ) -> VLLMWorkflowRecord:
        try:
            config = self.load_config(record.task_id)
            if self._budget_exhausted(record, config):
                return self.request_active_server_cleanup(
                    record, reason="GPU time budget exhausted before baseline benchmark"
                )
            return self.record_baseline(record, adapter.capture_baseline(config))
        except Exception as error:
            return self.request_active_server_cleanup(
                record,
                reason=f"baseline benchmark failed: {type(error).__name__}: {error}",
            )

    def request_active_server_cleanup(
        self, record: VLLMWorkflowRecord, *, reason: str
    ) -> VLLMWorkflowRecord:
        """Persist a user abort/downstream failure before stopping an active server."""

        normalized = reason.strip()
        if not normalized:
            raise VLLMWorkflowError("server cleanup reason cannot be empty")
        baseline = record.current_stage == VLLMStage.BASELINE
        candidate = record.current_stage in {VLLMStage.EXPERIMENT, VLLMStage.QUALITY}
        if not baseline and not candidate:
            raise VLLMWorkflowError("no benchmark/quality server is active at this stage")
        request_hash = (
            record.baseline_server_request_hash
            if baseline
            else record.candidate_server_request_hash
        )
        if request_hash is None:
            raise VLLMWorkflowError("active server request identity is missing")
        failure_ref = self._save_evidence(
            record.task_id,
            "vllm/baseline-run-failure" if baseline else "vllm/candidate-run-failure",
            {
                "schema": "amd-inference-opt.vllm-active-run-failure.v1",
                "stage": record.current_stage.value,
                "request_hash": request_hash,
                "reason": normalized,
                "recorded_at": utc_now().isoformat(),
            },
            producer="vllm-workflow",
        )
        updates: dict[str, Any] = {
            "revision": record.revision + 1,
            "updated_at": utc_now(),
        }
        if baseline:
            updates.update(
                {
                    "baseline_server_requires_stop": True,
                    "baseline_run_failure": failure_ref,
                }
            )
        else:
            updates.update(
                {
                    "candidate_server_requires_stop": True,
                    "candidate_run_failure": failure_ref,
                }
            )
        updated = VLLMWorkflowRecord.model_validate(
            {**record.model_dump(), **updates}
        )
        return self._save_update(record, updated)

    @staticmethod
    def _server_stopped(result: Any, spec: VLLMServerSpec) -> bool:
        stopped = _lookup(result, "stopped")
        status = _lookup(result, "record", "status")
        if hasattr(status, "value"):
            status = status.value
        return all(
            (
                stopped is True,
                str(status).upper() == "STOPPED",
                _lookup(result, "record", "request_hash") == spec.request_hash,
                _lookup(result, "record", "identity") == spec.identity,
            )
        )

    def stop_baseline_server(
        self, record: VLLMWorkflowRecord, adapter: VLLMServerLifecyclePort
    ) -> VLLMWorkflowRecord:
        """Stop the exact baseline PID before starting an offline profile engine."""

        config = self._load_config_artifact(record.task_id)
        try:
            self._verify_model_snapshot(config, require_persisted=True)
            live_coordinate_invalid = False
        except VLLMWorkflowError:
            live_coordinate_invalid = True
        failed_start_cleanup = (
            record.current_stage == VLLMStage.SERVER_START
            and record.baseline_server_requires_stop
        )
        failed_run_cleanup = (
            record.current_stage == VLLMStage.BASELINE
            and record.baseline_server_requires_stop
        )
        budget_cleanup = (
            record.current_stage == VLLMStage.BASELINE
            and record.active_gpu_phase == "baseline_server"
            and self._budget_exhausted(record, config)
        )
        coordinate_cleanup = (
            record.current_stage == VLLMStage.BASELINE
            and record.active_gpu_phase == "baseline_server"
            and live_coordinate_invalid
        )
        if (
            not failed_start_cleanup
            and not failed_run_cleanup
            and not budget_cleanup
            and not coordinate_cleanup
            and record.current_stage != VLLMStage.PROFILE_APPROVAL
        ):
            raise VLLMWorkflowError(
                "baseline server stops after a failed run/start or before profile approval"
            )
        if record.baseline_server_stop is not None:
            return record
        if record.baseline_server_request_hash is None:
            raise VLLMWorkflowError("baseline server request identity is missing")
        spec = build_server_spec(config, self.store)
        if spec.request_hash != record.baseline_server_request_hash:
            raise VLLMWorkflowError("baseline server request hash drifted before stop")
        result = adapter.stop(expected_request_hash=spec.request_hash)
        if not self._server_stopped(result, spec):
            raise VLLMWorkflowError("baseline server stop did not verify the exact PID request")
        gpu_seconds, gpu_by_phase = self._server_stop_accounting(record, result)
        artifact = self._save_evidence(
            record.task_id,
            "vllm/baseline-server-stop",
            result,
            producer="vllm-adapter",
        )
        cleanup = (
            failed_start_cleanup
            or failed_run_cleanup
            or budget_cleanup
            or coordinate_cleanup
        )
        if cleanup:
            updated = VLLMWorkflowRecord.model_validate(
                {
                    **record.model_dump(),
                    "current_stage": VLLMStage.SERVER_START,
                    "baseline_server_request_hash": None,
                    "baseline_server_requires_stop": False,
                    "baseline_server_start_failure": None,
                    "baseline_run_failure": None,
                    "active_gpu_started_at": None,
                    "active_gpu_phase": None,
                    "gpu_seconds_used": gpu_seconds,
                    "gpu_seconds_by_phase": gpu_by_phase,
                    "revision": record.revision + 1,
                    "updated_at": utc_now(),
                }
            )
            self.store.append_event(
                record.task_id,
                "vllm_failed_server_cleanup",
                {"stop_artifact": artifact.path},
            )
            saved = self._save_update(record, updated)
            if saved.gpu_seconds_used >= self._gpu_budget_seconds(config):
                return self._finalize_gpu_budget(
                    saved, config, extra_evidence={"server_stop": artifact}
                )
            return saved
        updated = VLLMWorkflowRecord.model_validate(
            {
                **record.model_dump(),
                "baseline_server_stop": artifact,
                "active_gpu_started_at": None,
                "active_gpu_phase": None,
                "gpu_seconds_used": gpu_seconds,
                "gpu_seconds_by_phase": gpu_by_phase,
                "revision": record.revision + 1,
                "updated_at": utc_now(),
            }
        )
        saved = self._save_update(record, updated)
        if saved.gpu_seconds_used >= self._gpu_budget_seconds(config):
            return self._finalize_gpu_budget(
                saved, config, extra_evidence={"server_stop": artifact}
            )
        return saved

    def _materialize_profile_command(
        self,
        config: VLLMCampaignConfig,
        *,
        attempt_id: str,
    ) -> tuple[Any, ArtifactRef]:
        profile = config.profile
        if profile.mode == "unavailable":
            raise VLLMWorkflowError(
                "profiling is unavailable: " + str(profile.unavailable_reason)
            )
        materializer = self.profile_materializer
        if materializer is None:
            from .vllm_ports import materialize_vllm_profile_command

            materializer = materialize_vllm_profile_command
        output_root = (
            self.store.task_dir(config.task.id) / "workspaces" / "vllm-profile"
        )
        try:
            materialized = materializer(
                python_executable=config.task.runtime.executable,
                launcher_path=profile.launcher_path,
                environment_manifest_path=profile.environment_manifest_path,
                model_manifest_path=config.model.snapshot_manifest_path,
                model=config.model,
                device=config.device,
                serving=config.serving,
                offline_argv=profile.profile_argv,
                output_root=output_root,
                attempt_id=attempt_id,
            )
        except Exception as error:
            raise VLLMWorkflowError(
                f"profile command materialization failed: {error}"
            ) from error
        if getattr(materialized, "attempt_id", None) != attempt_id:
            raise VLLMWorkflowError("profile materializer changed the attempt identity")
        argv = tuple(getattr(materialized, "argv", ()))
        if len(argv) < 3 or any(not isinstance(item, str) or not item for item in argv):
            raise VLLMWorkflowError("profile materializer returned invalid argv")
        if argv[0] != config.task.runtime.executable:
            raise VLLMWorkflowError("materialized profile changed the Python executable")
        if argv[1] != "-I" or Path(argv[2]).resolve(
            strict=False
        ) != profile.launcher_path.resolve(
            strict=False
        ):
            raise VLLMWorkflowError(
                "materialized profile must use isolated Python and the approved launcher"
            )
        coordinate_path = Path(getattr(materialized, "coordinate_path", ""))
        sidecar_path = Path(getattr(materialized, "sidecar_path", ""))
        throughput_result_path = Path(
            getattr(materialized, "throughput_result_path", "")
        )
        environment_manifest_path = Path(
            getattr(materialized, "environment_manifest_path", "")
        )
        model_manifest_path = Path(getattr(materialized, "model_manifest_path", ""))
        expected_attempt_root = (output_root / attempt_id).resolve(strict=False)
        if (
            not coordinate_path.is_absolute()
            or not sidecar_path.is_absolute()
            or not throughput_result_path.is_absolute()
            or coordinate_path.parent.resolve(strict=False) != expected_attempt_root
            or sidecar_path.parent.resolve(strict=False) != expected_attempt_root
            or throughput_result_path.parent.resolve(strict=False)
            != expected_attempt_root
            or sidecar_path.exists()
            or sidecar_path.is_symlink()
            or throughput_result_path.exists()
            or throughput_result_path.is_symlink()
        ):
            raise VLLMWorkflowError(
                "profile materialization paths are not fresh and attempt-scoped"
            )
        try:
            expected_environment_manifest = profile.environment_manifest_path.resolve(
                strict=True
            )
            expected_model_manifest = config.model.snapshot_manifest_path.resolve(
                strict=True
            )
        except OSError as error:
            raise VLLMWorkflowError("profile manifest path disappeared") from error
        if (
            environment_manifest_path.resolve(strict=False)
            != expected_environment_manifest
            or model_manifest_path.resolve(strict=False) != expected_model_manifest
            or getattr(materialized, "environment_manifest_sha256", None)
            != config.task.runtime.environment_manifest_sha256
        ):
            raise VLLMWorkflowError(
                "profile materializer changed an immutable manifest coordinate"
            )
        sidecar_values = [
            argv[index + 1] if index + 1 < len(argv) else None
            for index, item in enumerate(argv)
            if item == "--gpuopt-sidecar"
        ]
        if sidecar_values != [str(sidecar_path)]:
            raise VLLMWorkflowError(
                "materialized profile argv must bind its fresh sidecar exactly once"
            )
        throughput_values = [
            argv[index + 1] if index + 1 < len(argv) else None
            for index, item in enumerate(argv)
            if item == "--output-json"
        ]
        if throughput_values != [str(throughput_result_path)]:
            raise VLLMWorkflowError(
                "materialized profile argv must bind its fresh throughput result exactly once"
            )
        try:
            coordinate = json.loads(coordinate_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise VLLMWorkflowError("materialized profile coordinate is invalid") from error
        if (
            not isinstance(coordinate, Mapping)
            or coordinate.get("profile_launcher_sha256") != profile.launcher_sha256
            or coordinate.get("environment_manifest_sha256")
            != config.task.runtime.environment_manifest_sha256
            or not isinstance(coordinate.get("workload"), Mapping)
            or coordinate["workload"].get("completion_output")
            != str(throughput_result_path)
        ):
            raise VLLMWorkflowError(
                "materialized profile coordinate differs from immutable runtime"
            )
        coordinate_sha256 = str(getattr(materialized, "coordinate_sha256", ""))
        try:
            coordinate_ref = self.store.import_evidence(
                config.task.id,
                f"vllm/profile/{attempt_id}-coordinate",
                coordinate_path,
                producer="vllm-profile-materializer",
                media_type="application/json",
            )
        except StoreError as error:
            raise VLLMWorkflowError(str(error)) from error
        if coordinate_ref.sha256 != coordinate_sha256:
            raise VLLMWorkflowError(
                "materialized profile coordinate hash differs after import"
            )
        return materialized, coordinate_ref

    @staticmethod
    def _profile_arguments(
        config: VLLMCampaignConfig, command: tuple[str, ...]
    ) -> dict[str, Any]:
        profile = config.profile
        if profile.mode == "unavailable":
            raise VLLMWorkflowError(
                "profiling is unavailable: " + str(profile.unavailable_reason)
            )
        return RocmIssueAgentClient.profile_arguments(
            command,
            preset=profile.preset,
            cwd=profile.cwd,
            timeout_seconds=profile.timeout_seconds,
            max_trace_bytes=profile.max_trace_bytes,
            max_trace_files=profile.max_trace_files,
            max_events_per_type=profile.max_events_per_type,
            max_percentile_samples_per_kernel=(
                profile.max_percentile_samples_per_kernel
            ),
        )

    @staticmethod
    def _profile_context(
        config: VLLMCampaignConfig,
        *,
        attempt_id: str,
        materialized: Any,
        coordinate_artifact: ArtifactRef,
        resolved_mcp_environment: Mapping[str, str],
    ) -> dict[str, Any]:
        return {
            "schema": "amd-inference-opt.vllm-profile-approval-context.v1",
            "attempt_id": attempt_id,
            "config_identity_sha256": config.identity_sha256,
            "runtime_identity": config.identity_payload,
            "profile_mode": config.profile.mode,
            "profile_environment": config.profile.environment,
            "deployment_scope": (
                "container_preflight"
                if config.task.runtime.image_digest is not None
                else "native_preflight"
            ),
            "profile_command_sha256": config.profile.command_sha256,
            "materialized_profile_command_sha256": canonical_sha256(
                list(materialized.argv)
            ),
            "materialized_profile": {
                "argv": list(materialized.argv),
                "coordinate_path": str(materialized.coordinate_path),
                "coordinate_sha256": materialized.coordinate_sha256,
                "coordinate_artifact": coordinate_artifact.model_dump(mode="json"),
                "sidecar_path": str(materialized.sidecar_path),
                "throughput_result_path": str(materialized.throughput_result_path),
                "environment_manifest_path": str(
                    materialized.environment_manifest_path
                ),
                "environment_manifest_sha256": (
                    materialized.environment_manifest_sha256
                ),
                "model_manifest_path": str(materialized.model_manifest_path),
                "model_manifest_sha256": materialized.model_manifest_sha256,
            },
            "profile_launcher_sha256": config.profile.launcher_sha256,
            "environment_manifest_sha256": (
                config.task.runtime.environment_manifest_sha256
            ),
            "serving_protocol_sha256": config.serving.coordinate_sha256,
            "model_snapshot_digest": config.model.snapshot_digest,
            "model_revision": config.model.revision,
            "tokenizer_revision": config.model.tokenizer_revision,
            "gfx_target": config.device.gfx_target,
            "device_ids": config.device.device_ids,
            "device_uuid": config.device.device_uuid,
            "pci_bdf": config.device.pci_bdf,
            "compute_partition": config.device.compute_partition,
            "memory_partition": config.device.memory_partition,
            "partition_id": config.device.partition_id,
            "mcp_command": config.task.mcp.command,
            "mcp_cwd": str(config.task.mcp.cwd) if config.task.mcp.cwd else None,
            "mcp_env_keys": sorted(resolved_mcp_environment),
            "mcp_env_sha256": canonical_sha256(
                dict(sorted(resolved_mcp_environment.items()))
            ),
        }

    def request_profile_approval(
        self,
        record: VLLMWorkflowRecord,
        *,
        resolved_mcp_environment: Mapping[str, str],
    ) -> tuple[VLLMWorkflowRecord, ApprovalRequest]:
        if record.current_stage != VLLMStage.PROFILE_APPROVAL:
            raise VLLMWorkflowError("profile approval is valid only at PROFILE_APPROVAL")
        if record.baseline_server_stop is None or not self.store.verify_artifact(
            record.task_id, record.baseline_server_stop
        ):
            raise VLLMWorkflowError(
                "the exact baseline server must be stopped before profile approval"
            )
        if (
            record.profile_approval is not None
            and record.profile_approval.status == VLLMApprovalStatus.CONSUMED
        ):
            raise VLLMWorkflowError(
                "consumed profile approval requires a durable result or UNKNOWN outcome "
                "before any new request"
            )
        config = self.load_config(record.task_id)
        environment = dict(resolved_mcp_environment)
        configured_mismatches = [
            name
            for name, value in config.task.mcp.env.items()
            if environment.get(name) != value
        ]
        if configured_mismatches:
            raise VLLMWorkflowError(
                "resolved MCP environment changed configured values: "
                + ", ".join(sorted(configured_mismatches))
            )
        required_device_env = {
            "ROCR_VISIBLE_DEVICES": config.device.stable_device_id,
        }
        missing_device_env = [
            name
            for name, value in required_device_env.items()
            if environment.get(name) != value
        ]
        if missing_device_env:
            raise VLLMWorkflowError(
                "resolved MCP environment lacks stable container device isolation: "
                + ", ".join(sorted(missing_device_env))
            )
        conflicting_selectors = {
            "HIP_VISIBLE_DEVICES",
            "HSA_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "HSA_OVERRIDE_GFX_VERSION",
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONSTARTUP",
        } & set(environment)
        if conflicting_selectors:
            raise VLLMWorkflowError(
                "resolved MCP environment contains conflicting GPU/Python selectors: "
                + ", ".join(sorted(conflicting_selectors))
            )
        server = MCPServerConfig.from_domain(config.task.mcp)
        server = MCPServerConfig(
            command=server.command, args=server.args, cwd=server.cwd, env=environment
        )
        try:
            contract = self.profile_contract_reader(server)
            contract_ref = self.store.save_evidence_json(
                record.task_id,
                "vllm/profile-contract",
                contract.to_dict(),
                producer="rocm-issue-agent-mcp",
            )
            # Validate limits before creating attempt-owned files or an approval.
            contract.validate(self._profile_arguments(config, tuple(config.profile.profile_argv)))
        except Exception as error:
            raise VLLMWorkflowError(f"profile capability preflight failed: {error}") from error
        if record.profile_approval is not None:
            state = record.profile_approval
            if not all(
                self.store.verify_artifact(record.task_id, ref)
                for ref in (state.request, state.context)
            ):
                raise VLLMWorkflowError("profile approval artifacts failed verification")
            previous_context = self.store.load_json(record.task_id, state.context.path)
            if (
                previous_context.get("profile_contract_sha256") == contract.sha256
                and previous_context.get("mcp_env_sha256")
                == canonical_sha256(dict(sorted(environment.items())))
            ):
                request = self.store.load_json(record.task_id, state.request.path, ApprovalRequest)
                return record, request
            # Keep the old request and unconsumed receipt immutable. A capability
            # change needs a new attempt and explicit approval of its new contract.
            self.store.append_event(record.task_id, "vllm_profile_approval_superseded", {
                "request_id": state.request_id,
                "reason": "installed profile contract or execution environment changed",
            })
        attempt = record.profile_attempt_count + 1
        attempt_id = f"baseline-profile-attempt-{attempt:04d}-{uuid.uuid4().hex}"
        materialized, coordinate_ref = self._materialize_profile_command(
            config, attempt_id=attempt_id
        )
        arguments = self._profile_arguments(config, tuple(materialized.argv))
        try:
            contract.validate(arguments)
        except ProfileContractError as error:
            raise VLLMWorkflowError(f"materialized profile contract rejected: {error}") from error
        context = self._profile_context(
            config,
            attempt_id=attempt_id,
            materialized=materialized,
            coordinate_artifact=coordinate_ref,
            resolved_mcp_environment=environment,
        )
        context["profile_contract_sha256"] = contract.sha256
        context["profile_contract_artifact"] = contract_ref.model_dump(mode="json")
        exact = approval_request(
            "rocm_profile_workload", arguments, execution_context=context
        )
        request_id = f"vllm-profile-{attempt:04d}-{exact.request_sha256[:12]}"
        request = ApprovalRequest(
            id=request_id,
            task_id=record.task_id,
            tool="rocm_profile_workload",
            arguments=arguments,
            request_hash=exact.request_sha256,
        )
        try:
            context_ref = self.store.save_immutable_json(
                record.task_id,
                f"state/approvals/{request_id}.context.json",
                context,
                producer="vllm-workflow",
            )
            request_ref = self.store.save_immutable_json(
                record.task_id,
                f"state/approvals/{request_id}.request.json",
                request,
                producer="vllm-workflow",
            )
        except StoreError as error:
            raise VLLMWorkflowError(str(error)) from error
        state = VLLMProfileApprovalState(
            attempt=attempt,
            request_id=request_id,
            request_hash=request.request_hash,
            request=request_ref,
            context=context_ref,
        )
        updated = VLLMWorkflowRecord.model_validate(
            {
                **record.model_dump(),
                "profile_approval": state,
                "profile_attempt_count": attempt,
                "profile_execution_failure": None,
                "revision": record.revision + 1,
                "updated_at": utc_now(),
            }
        )
        self.store.append_event(
            record.task_id,
            "vllm_profile_approval_requested",
            {
                "request_id": request_id,
                "request_hash": request.request_hash,
                "attempt": attempt,
            },
        )
        return self._save_update(record, updated), request

    def record_profile_approval(
        self, record: VLLMWorkflowRecord, receipt: ApprovalReceipt
    ) -> VLLMWorkflowRecord:
        """Record a receipt supplied by a human/UI; never create one implicitly."""

        state = record.profile_approval
        if record.current_stage != VLLMStage.PROFILE_APPROVAL or state is None:
            raise VLLMWorkflowError("there is no pending vLLM profile approval")
        if state.status != VLLMApprovalStatus.REQUESTED:
            raise VLLMWorkflowError("profile approval was already recorded or consumed")
        if receipt.request_id != state.request_id or receipt.request_hash != state.request_hash:
            raise VLLMWorkflowError("approval receipt does not match the exact profile request")
        if receipt.consumed_at is not None:
            raise VLLMWorkflowError("new approval receipt must not already be consumed")
        try:
            receipt_ref = self.store.save_immutable_json(
                record.task_id,
                f"state/approvals/{state.request_id}.receipt.json",
                receipt,
                producer="user-approval",
            )
        except StoreError as error:
            raise VLLMWorkflowError(str(error)) from error
        updated_state = state.model_copy(
            update={"receipt": receipt_ref, "status": VLLMApprovalStatus.APPROVED}
        )
        updated = VLLMWorkflowRecord.model_validate(
            {
                **record.model_dump(),
                "profile_approval": updated_state,
                "revision": record.revision + 1,
                "updated_at": utc_now(),
            }
        )
        return self._save_update(record, updated)

    def consume_profile_approval(
        self, record: VLLMWorkflowRecord
    ) -> tuple[VLLMWorkflowRecord, VLLMProfileExecutionPermit]:
        """Consume before transport and return the exact MCP call permit once."""

        state = record.profile_approval
        if record.current_stage != VLLMStage.PROFILE_APPROVAL or state is None:
            raise VLLMWorkflowError("there is no vLLM profile approval to consume")
        if state.status != VLLMApprovalStatus.APPROVED or state.receipt is None:
            raise VLLMWorkflowError("profile execution requires an unconsumed approval")
        if not all(
            self.store.verify_artifact(record.task_id, artifact)
            for artifact in (state.request, state.context, state.receipt)
        ):
            raise VLLMWorkflowError("profile approval artifacts failed verification")
        request = self.store.load_json(
            record.task_id, state.request.path, ApprovalRequest
        )
        context = self.store.load_json(record.task_id, state.context.path)
        try:
            contract_ref = ArtifactRef.model_validate(context.get("profile_contract_artifact"))
        except ValidationError as error:
            raise VLLMWorkflowError(
                "profile approval lacks capability negotiation evidence"
            ) from error
        if not self.store.verify_artifact(record.task_id, contract_ref):
            raise VLLMWorkflowError("profile capability negotiation artifact failed verification")
        contract_evidence = self.store.load_json(record.task_id, contract_ref.path)
        if contract_evidence.get("contract_sha256") != context.get("profile_contract_sha256"):
            raise VLLMWorkflowError("profile capability negotiation hash differs from approval")
        receipt = self.store.load_json(
            record.task_id, state.receipt.path, ApprovalReceipt
        )
        exact = approval_request(
            request.tool, request.arguments, execution_context=context
        )
        if (
            request.id != state.request_id
            or request.request_hash != state.request_hash
            or exact.request_sha256 != state.request_hash
            or receipt.request_id != state.request_id
            or receipt.request_hash != state.request_hash
            or receipt.consumed_at is not None
        ):
            raise VLLMWorkflowError("persisted profile approval is not exact and unconsumed")

        consumed = receipt.model_copy(update={"consumed_at": utc_now()})
        try:
            consumed_ref = self.store.save_immutable_json(
                record.task_id,
                f"state/approvals/{state.request_id}.consumed.json",
                consumed,
                producer="vllm-workflow",
            )
        except StoreError as error:
            raise VLLMWorkflowError(str(error)) from error
        updated_state = state.model_copy(
            update={
                "receipt": consumed_ref,
                "status": VLLMApprovalStatus.CONSUMED,
            }
        )
        updated = VLLMWorkflowRecord.model_validate(
            {
                **record.model_dump(),
                "profile_approval": updated_state,
                "revision": record.revision + 1,
                "updated_at": utc_now(),
            }
        )
        persisted = self._save_update(record, updated)
        permit = VLLMProfileExecutionPermit(
            request_id=state.request_id,
            request_hash=state.request_hash,
            arguments=request.arguments,
            execution_context=context,
        )
        return persisted, permit

    def record_profile_result(
        self,
        record: VLLMWorkflowRecord,
        profile_result: VLLMProfileResultEvidence | Mapping[str, Any],
    ) -> VLLMWorkflowRecord:
        state = record.profile_approval
        if (
            record.current_stage != VLLMStage.PROFILE_APPROVAL
            or state is None
            or state.status != VLLMApprovalStatus.CONSUMED
            or state.receipt is None
        ):
            raise VLLMWorkflowError("profile result requires a consumed exact approval")
        if not all(
            self.store.verify_artifact(record.task_id, artifact)
            for artifact in (state.request, state.context, state.receipt)
        ):
            raise VLLMWorkflowError("consumed profile approval artifacts are corrupt")
        request = self.store.load_json(
            record.task_id, state.request.path, ApprovalRequest
        )
        approval_context = self.store.load_json(record.task_id, state.context.path)
        if not isinstance(approval_context, Mapping):
            raise VLLMWorkflowError("profile approval context is not an object")
        materialized_context = approval_context.get("materialized_profile")
        approved_argv = request.arguments.get("command")
        approved_cwd = request.arguments.get("cwd")
        if (
            not isinstance(materialized_context, Mapping)
            or not isinstance(approved_argv, list)
            or materialized_context.get("argv") != approved_argv
            or approval_context.get("materialized_profile_command_sha256")
            != canonical_sha256(approved_argv)
        ):
            raise VLLMWorkflowError(
                "profile approval lost its attempt-materialized command binding"
            )
        try:
            coordinate_ref = ArtifactRef.model_validate(
                materialized_context.get("coordinate_artifact")
            )
        except ValidationError as error:
            raise VLLMWorkflowError(
                "profile approval lacks its materialized coordinate artifact"
            ) from error
        if (
            coordinate_ref.sha256 != materialized_context.get("coordinate_sha256")
            or not self.store.verify_artifact(record.task_id, coordinate_ref)
        ):
            raise VLLMWorkflowError(
                "materialized profile coordinate artifact failed verification"
            )
        throughput_result_path = materialized_context.get("throughput_result_path")
        throughput_values = [
            approved_argv[index + 1] if index + 1 < len(approved_argv) else None
            for index, item in enumerate(approved_argv)
            if item == "--output-json"
        ]
        if (
            not isinstance(throughput_result_path, str)
            or not Path(throughput_result_path).is_absolute()
            or throughput_values != [throughput_result_path]
        ):
            raise VLLMWorkflowError(
                "profile approval lost its attempt-owned throughput result binding"
            )
        config = self.load_config(record.task_id)
        evidence = VLLMProfileResultEvidence.model_validate(profile_result)
        if not self.store.verify_artifact(
            record.task_id, evidence.execution_environment_artifact
        ):
            raise VLLMWorkflowError(
                "profile execution-environment evidence failed integrity verification"
            )
        try:
            run_context = self.store.load_json(
                record.task_id, evidence.execution_environment_artifact.path
            )
        except StoreError as error:
            raise VLLMWorkflowError(str(error)) from error
        if not isinstance(run_context, Mapping):
            raise VLLMWorkflowError("profile run-context artifact is not an object")
        run_environment = run_context.get("environment")
        run_gpus = run_context.get("gpus")
        run_execution = run_context.get("vllm_execution")
        gpu_matches = isinstance(run_gpus, list) and any(
            isinstance(gpu, Mapping)
            and gpu.get("gfx_architecture") == config.device.gfx_target
            and gpu.get("pci_bdf") == config.device.pci_bdf
            and gpu.get("partition_id") == config.device.partition_id
            and gpu.get("accelerator_partition") == config.device.compute_partition
            and gpu.get("memory_partition") == config.device.memory_partition
            for gpu in run_gpus
        )
        context_problems: list[str] = []
        if run_context.get("schema") != "rocm.run-context.v1":
            context_problems.append("unexpected run-context schema")
        if run_context.get("run_context_hash") != evidence.run_context_hash:
            context_problems.append("run-context artifact hash field differs")
        hash_payload = {
            key: value
            for key, value in run_context.items()
            if key not in {"collected_at", "run_context_hash", "vllm_execution"}
        }
        recomputed_hash = f"sha256:{canonical_sha256(hash_payload)}"
        if recomputed_hash != evidence.run_context_hash:
            context_problems.append("run-context canonical content hash differs")
        try:
            canonical_approved_argv = [
                str(Path(approved_argv[0]).resolve(strict=True)),
                *approved_argv[1:],
            ]
        except OSError as error:
            raise VLLMWorkflowError(
                "approved profile executable disappeared before result validation"
            ) from error
        if run_context.get("argv") != canonical_approved_argv:
            context_problems.append("run-context argv differs from approved profile argv")
        if run_context.get("cwd") != approved_cwd:
            context_problems.append("run-context cwd differs from approved profile cwd")
        if not isinstance(run_environment, Mapping) or (
            run_environment.get("ROCR_VISIBLE_DEVICES") != config.device.device_uuid
            or run_environment.get("PYTHONNOUSERSITE") != "1"
            or any(
                name in run_environment
                for name in (
                    "HIP_VISIBLE_DEVICES",
                    "HSA_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                    "GPU_DEVICE_ORDINAL",
                    "HSA_OVERRIDE_GFX_VERSION",
                    "PYTHONHOME",
                    "PYTHONPATH",
                    "PYTHONSTARTUP",
                )
            )
        ):
            context_problems.append(
                "run-context lacks stable UUID/Python environment isolation"
            )
        expected_execution = {
            "image_digest": config.task.runtime.image_digest,
            "model_snapshot_digest": config.model.snapshot_digest,
            "model_revision": config.model.revision,
            "tokenizer_revision": config.model.tokenizer_revision,
            "serving_protocol_sha256": config.serving.coordinate_sha256,
            "profile_command_sha256": config.profile.command_sha256,
            "profile_launcher_sha256": config.profile.launcher_sha256,
            "environment_manifest_sha256": (
                config.task.runtime.environment_manifest_sha256
            ),
            "device_uuid": config.device.device_uuid,
            "pci_bdf": config.device.pci_bdf,
            "partition_id": config.device.partition_id,
            "runtime_evidence": evidence.runtime_evidence.model_dump(mode="json"),
        }
        if not isinstance(run_execution, Mapping):
            context_problems.append("enriched vLLM execution coordinates are missing")
        else:
            execution_mismatches = [
                name
                for name, expected in expected_execution.items()
                if run_execution.get(name) != expected
            ]
            if execution_mismatches:
                context_problems.append(
                    "enriched vLLM execution coordinates differ: "
                    + ", ".join(execution_mismatches)
                )
            completion = run_execution.get("offline_vllm_completion")
            if not isinstance(completion, Mapping):
                context_problems.append("offline vLLM completion evidence is missing")
            else:
                try:
                    completion_ref = ArtifactRef.model_validate(
                        completion.get("artifact")
                    )
                except ValidationError:
                    context_problems.append(
                        "offline vLLM completion artifact is invalid"
                    )
                else:
                    if (
                        completion_ref.producer != "vllm-bench-throughput"
                        or completion_ref.media_type != "application/json"
                        or not self.store.verify_artifact(
                            record.task_id, completion_ref
                        )
                    ):
                        context_problems.append(
                            "offline vLLM completion artifact failed verification"
                        )
                    else:
                        try:
                            raw_completion = self.store.load_json(
                                record.task_id, completion_ref.path
                            )
                        except StoreError:
                            context_problems.append(
                                "offline vLLM completion artifact cannot be loaded"
                            )
                        else:
                            metrics = completion.get("metrics")
                            metric_names = {
                                "num_requests",
                                "total_num_tokens",
                                "elapsed_time",
                                "requests_per_second",
                                "tokens_per_second",
                            }
                            if (
                                not isinstance(metrics, Mapping)
                                or set(metrics) != metric_names
                                or not isinstance(raw_completion, Mapping)
                                or any(
                                    raw_completion.get(name) != metrics.get(name)
                                    for name in metric_names
                                )
                            ):
                                context_problems.append(
                                    "offline vLLM completion metrics differ from artifact"
                                )
                            else:
                                num_requests = metrics["num_requests"]
                                total_tokens = metrics["total_num_tokens"]
                                elapsed = metrics["elapsed_time"]
                                request_rate = metrics["requests_per_second"]
                                token_rate = metrics["tokens_per_second"]
                                valid_integers = (
                                    not isinstance(num_requests, bool)
                                    and isinstance(num_requests, int)
                                    and num_requests == config.serving.num_prompts
                                    and not isinstance(total_tokens, bool)
                                    and isinstance(total_tokens, int)
                                    and total_tokens > 0
                                )
                                numbers = (elapsed, request_rate, token_rate)
                                valid_numbers = all(
                                    not isinstance(value, bool)
                                    and isinstance(value, (int, float))
                                    and math.isfinite(float(value))
                                    and float(value) > 0
                                    for value in numbers
                                )
                                consistent_rates = valid_integers and valid_numbers and (
                                    math.isclose(
                                        float(request_rate),
                                        num_requests / float(elapsed),
                                        rel_tol=5e-4,
                                        abs_tol=1e-6,
                                    )
                                    and math.isclose(
                                        float(token_rate),
                                        total_tokens / float(elapsed),
                                        rel_tol=5e-4,
                                        abs_tol=1e-6,
                                    )
                                )
                                if not consistent_rates:
                                    context_problems.append(
                                        "offline vLLM completion is incomplete or inconsistent"
                                    )
                    framework = evidence.mcp_call.get("framework_evidence")
                    if not isinstance(framework, Mapping):
                        context_problems.append(
                            "MCP envelope lacks framework completion evidence"
                        )
                    else:
                        try:
                            framework_ref = ArtifactRef.model_validate(
                                framework.get("offline_vllm_completion")
                            )
                        except ValidationError:
                            context_problems.append(
                                "MCP framework completion artifact is invalid"
                            )
                        else:
                            if (
                                framework_ref != completion_ref
                                or framework.get("offline_vllm_metrics")
                                != completion.get("metrics")
                            ):
                                context_problems.append(
                                    "MCP framework completion evidence differs"
                                )
        if not gpu_matches:
            context_problems.append("run-context GPU/partition identity differs")
        if context_problems:
            raise VLLMWorkflowError(
                "profile run context is not comparable: "
                + "; ".join(context_problems)
            )
        expected_identity = {
            "image_digest": config.task.runtime.image_digest,
            "model_snapshot_digest": config.model.snapshot_digest,
            "model_revision": config.model.revision,
            "tokenizer_revision": config.model.tokenizer_revision,
            "serving_protocol_sha256": config.serving.coordinate_sha256,
            "profile_command_sha256": config.profile.command_sha256,
            "profile_launcher_sha256": config.profile.launcher_sha256,
            "environment_manifest_sha256": (
                config.task.runtime.environment_manifest_sha256
            ),
            "device_uuid": config.device.device_uuid,
            "pci_bdf": config.device.pci_bdf,
            "partition_id": config.device.partition_id,
        }
        mismatches = [
            f"{name}: observed {getattr(evidence, name)!r}, expected {expected!r}"
            for name, expected in expected_identity.items()
            if getattr(evidence, name) != expected
        ]
        if mismatches:
            raise VLLMWorkflowError(
                "profile did not run in the approved engine environment: "
                + "; ".join(mismatches)
            )
        runtime = evidence.runtime_evidence
        runtime_problems: list[str] = []
        if runtime.verification != "VERIFIED":
            runtime_problems.append("profile process runtime is not VERIFIED")
        if (
            not runtime.native_executable_matches
            or runtime.native_executable_sha256
            != config.task.runtime.executable_sha256
        ):
            runtime_problems.append("profile Python executable identity differs")
        if (
            runtime.environment_manifest_sha256
            != config.task.runtime.environment_manifest_sha256
        ):
            runtime_problems.append("profile package environment manifest differs")
        if runtime.profile_launcher_sha256 != config.profile.launcher_sha256:
            runtime_problems.append("profile workload launcher digest differs")
        if not runtime.declared_environment_matches or not runtime.unset_environment_absent:
            runtime_problems.append("profile process environment was not exactly observed")
        if runtime.rocr_visible_devices != config.device.device_uuid:
            runtime_problems.append("profile process ROCR UUID differs")
        server_start = self._latest_completion(
            record, VLLMStage.SERVER_START
        ).evidence["server_start"]
        start_payload = self.store.load_json(record.task_id, server_start.path)
        baseline_runtime = _lookup(start_payload, "record", "runtime_evidence")
        if config.task.runtime.image_digest is not None:
            if runtime.image_digest_matches is not True:
                runtime_problems.append("profile container RepoDigest is unverified")
            if runtime.container_binding_kind != "cgroup_v2":
                runtime_problems.append("profile same-container binding kind is missing")
            if (
                runtime.process_binding_id is None
                or runtime.process_binding_id != runtime.container_binding_id
                or runtime.container_binding_id
                != _lookup(baseline_runtime, "container_binding_id")
            ):
                runtime_problems.append(
                    "profile engine does not share baseline container namespace/cgroup"
                )
            digest_suffix = f"sha256:{config.task.runtime.image_digest}"
            if not any(
                value == digest_suffix or value.endswith(f"@{digest_suffix}")
                for value in runtime.container_repo_digests
            ):
                runtime_problems.append("profile runtime lacks the configured RepoDigest")
        elif any(
            value is not None
            for value in (
                runtime.container_binding_kind,
                runtime.process_binding_id,
                runtime.container_binding_id,
            )
        ):
            runtime_problems.append("native profile unexpectedly claims container binding")
        if runtime_problems:
            raise VLLMWorkflowError(
                "profile runtime provenance is inconclusive: "
                + "; ".join(runtime_problems)
            )
        payload = evidence.mcp_call
        contract_error = None
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments")
        is_error = bool(payload.get("is_error"))
        structured = payload.get("structured_content")
        if not isinstance(structured, Mapping) or structured.get("schema") != (
            "rocm.mcp-kernel-evidence.v1"
        ):
            contract_error = "profile result has no ROCm kernel-evidence schema"
        if isinstance(structured, Mapping):
            kernel_evidence = structured.get("kernel_evidence")
            workload = (
                kernel_evidence.get("workload")
                if isinstance(kernel_evidence, Mapping)
                else None
            )
            profiler = (
                kernel_evidence.get("profiler")
                if isinstance(kernel_evidence, Mapping)
                else None
            )
            kernels = (
                kernel_evidence.get("kernels")
                if isinstance(kernel_evidence, Mapping)
                else None
            )
            profile_problems: list[str] = []
            if structured.get("status") != "completed":
                profile_problems.append("top-level profile status is not completed")
            if structured.get("run_context_hash") != evidence.run_context_hash:
                profile_problems.append("top-level run_context_hash differs")
            if not isinstance(kernel_evidence, Mapping):
                profile_problems.append("kernel_evidence is missing")
            else:
                if kernel_evidence.get("status") != "completed":
                    profile_problems.append("kernel evidence status is not completed")
                if kernel_evidence.get("preset") != config.profile.preset:
                    profile_problems.append("kernel evidence preset differs")
                if kernel_evidence.get("run_context_hash") != evidence.run_context_hash:
                    profile_problems.append("kernel run_context_hash differs")
                if kernel_evidence.get("workload_succeeded") is not True:
                    profile_problems.append("profiled workload did not succeed")
                if kernel_evidence.get("aggregate_timing_complete") is not True:
                    profile_problems.append("aggregate kernel timing is incomplete")
            if not isinstance(workload, Mapping) or (
                workload.get("status") != "completed" or workload.get("exit_code") != 0
            ):
                profile_problems.append("profiled workload status/exit code is invalid")
            if not isinstance(profiler, Mapping) or profiler.get("status") != "completed":
                profile_problems.append("profiler status is not completed")
            if (
                not isinstance(kernels, list)
                or not kernels
                or len(kernels) != evidence.engine_kernel_count
            ):
                profile_problems.append("actual kernel rows do not match engine_kernel_count")
            elif any(
                not isinstance(kernel, Mapping)
                or not isinstance(kernel.get("name"), str)
                or not kernel["name"].strip()
                for kernel in kernels
            ):
                profile_problems.append("kernel rows must contain non-empty engine symbols")
            if profile_problems:
                contract_error = "; ".join(profile_problems)
        if tool_name != "rocm_profile_workload" or arguments != request.arguments:
            raise VLLMWorkflowError("profile result does not match the approved MCP call")
        if is_error or contract_error:
            raise VLLMWorkflowError(
                "profile evidence is inconclusive: "
                + (contract_error or "ROCm MCP returned an error")
            )
        profile_ref = self._save_evidence(
            record.task_id,
            "vllm/profile",
            evidence,
            producer="rocm-issue-agent-mcp",
        )
        return self.advance(
            record,
            {
                "server_stop": record.baseline_server_stop,
                "approval_request": state.request,
                "approval_receipt": state.receipt,
                "profile": profile_ref,
            },
        )

    def run_approved_profile(
        self, record: VLLMWorkflowRecord, adapter: VLLMProfilePort
    ) -> VLLMWorkflowRecord:
        """Consume first, then invoke the MCP-backed profile port exactly once."""

        config = self.load_config(record.task_id)
        if self._budget_exhausted(record, config):
            return self.finalize_gpu_budget(record)
        preflight = getattr(adapter, "preflight", None)
        if preflight is not None:
            state = record.profile_approval
            if state is None or state.status != VLLMApprovalStatus.APPROVED:
                raise VLLMWorkflowError("profile execution requires an unconsumed approval")
            if not all(
                self.store.verify_artifact(record.task_id, ref)
                for ref in (state.request, state.context)
            ):
                raise VLLMWorkflowError("profile approval artifacts failed verification")
            request = self.store.load_json(record.task_id, state.request.path, ApprovalRequest)
            context = self.store.load_json(record.task_id, state.context.path)
            preflight(VLLMProfileExecutionPermit(
                request_id=state.request_id,
                request_hash=state.request_hash,
                arguments=request.arguments,
                execution_context=context,
            ))
        consumed_record, permit = self.consume_profile_approval(record)
        started = time.monotonic()
        try:
            evidence = adapter.execute(permit)
        except Exception as error:
            accounted = self._record_profile_gpu_seconds(
                consumed_record, time.monotonic() - started
            )
            return self.mark_profile_execution_inconclusive(
                accounted,
                error,
                request_id=permit.request_id,
            )
        accounted = self._record_profile_gpu_seconds(
            consumed_record, time.monotonic() - started
        )
        return self.record_profile_result(accounted, evidence)

    def mark_profile_execution_inconclusive(
        self,
        record: VLLMWorkflowRecord,
        error: Exception,
        *,
        request_id: str | None = None,
    ) -> VLLMWorkflowRecord:
        """Durably close an unknown/failed consumed attempt without replaying it."""

        state = record.profile_approval
        if (
            record.current_stage != VLLMStage.PROFILE_APPROVAL
            or state is None
            or state.status != VLLMApprovalStatus.CONSUMED
        ):
            raise VLLMWorkflowError(
                "only a consumed profile attempt can become execution-inconclusive"
            )
        if request_id is not None and request_id != state.request_id:
            raise VLLMWorkflowError("profile failure belongs to a different request")
        failure = {
            "schema": "amd-inference-opt.vllm-profile-attempt-failure.v1",
            "request_id": state.request_id,
            "request_hash": state.request_hash,
            "outcome": "UNKNOWN",
            "error_type": type(error).__name__,
            "error": str(error),
            "recorded_at": utc_now().isoformat(),
        }
        failure_ref = self._save_evidence(
            record.task_id,
            "vllm/profile-attempt-failure",
            failure,
            producer="vllm-workflow",
        )
        decision = GateDecision(
            outcome=DecisionOutcome.INCONCLUSIVE,
            checks=[
                GateCheck(
                    name="profile_execution",
                    passed=None,
                    detail=(
                        "consumed approval reached an unknown/failed transport outcome; "
                        "the receipt cannot be replayed"
                    ),
                )
            ],
            reasons=[
                "profile execution did not produce complete verified engine evidence"
            ],
            rerun_from_stage=WorkflowStage.DISCOVER_HOTSPOTS,
        )
        gate_ref = self._save_evidence(
            record.task_id,
            "vllm/gate",
            decision,
            producer="gate",
        )
        seeded = record.model_copy(
            update={"profile_execution_failure": failure_ref}
        )
        updated = self.engine.terminate_inconclusive(
            seeded,
            decision,
            gate_ref,
            extra_evidence={"profile_execution_failure": failure_ref},
        )
        return self._save_update(record, updated)

    def mark_profile_unavailable(
        self, record: VLLMWorkflowRecord
    ) -> VLLMWorkflowRecord:
        config = self.load_config(record.task_id)
        if record.current_stage != VLLMStage.PROFILE_APPROVAL:
            raise VLLMWorkflowError("profile availability is decided at PROFILE_APPROVAL")
        if record.baseline_server_stop is None:
            raise VLLMWorkflowError("baseline server must stop before closing profile Gate")
        if config.profile.mode != "unavailable":
            raise VLLMWorkflowError("configured profile is available; request exact approval")
        decision = GateDecision(
            outcome=DecisionOutcome.INCONCLUSIVE,
            checks=[
                GateCheck(
                    name="profile_environment",
                    passed=None,
                    detail=str(config.profile.unavailable_reason),
                )
            ],
            reasons=[
                "ROCm profiling is unavailable in the vLLM execution environment"
            ],
            rerun_from_stage=WorkflowStage.DISCOVER_HOTSPOTS,
        )
        gate_ref = self._save_evidence(
            record.task_id,
            "vllm/gate",
            decision,
            producer="gate",
        )
        updated = self.engine.terminate_inconclusive(record, decision, gate_ref)
        return self._save_update(record, updated)

    def record_agent_decision(
        self, record: VLLMWorkflowRecord, decision: VLLMAgentDecision
    ) -> VLLMWorkflowRecord:
        if record.current_stage != VLLMStage.AGENT_DECISION:
            raise VLLMWorkflowError("agent decisions are valid only at AGENT_DECISION")
        config = self.load_config(record.task_id)
        if self._budget_exhausted(record, config):
            return self.finalize_gpu_budget(record)
        if record.experiment_count >= config.task.budgets.max_experiments:
            raise VLLMWorkflowError("vLLM experiment budget is exhausted")
        spec = decision.proposed_experiment
        if spec.task_id != record.task_id:
            raise VLLMWorkflowError("experiment spec belongs to a different task")
        if spec.benchmark_argv != config.serving.benchmark_argv:
            raise VLLMWorkflowError("experiment changed the locked benchmark argv")
        expected_server_argv = [
            *config.serving.server_argv,
            *spec.change.runtime_args,
        ]
        if spec.server_argv != expected_server_argv:
            raise VLLMWorkflowError(
                "experiment server argv must equal baseline argv plus approved runtime_args"
            )
        expected_server_env = dict(config.serving.server_env)
        for name in spec.change.unset_env:
            expected_server_env.pop(name, None)
        expected_server_env.update(spec.change.env)
        if spec.server_env != expected_server_env:
            raise VLLMWorkflowError(
                "experiment server env must equal the declared ChangeSet delta"
            )
        if spec.change.candidate_model_path is not None:
            raise VLLMWorkflowError("vLLM V0 locks the model snapshot")
        if spec.change.env or spec.change.unset_env:
            raise VLLMWorkflowError(
                "vLLM V0 locks the process environment; use allow-listed runtime argv"
            )
        known_evidence = {
            artifact.path
            for completion in record.completions
            for artifact in completion.evidence.values()
        }
        unknown = set(decision.evidence_used) - known_evidence
        unknown.update(set(decision.hypothesis.observed_evidence_ids) - known_evidence)
        if unknown:
            raise VLLMWorkflowError(
                "agent decision cites unknown evidence: " + ", ".join(sorted(unknown))
            )
        try:
            validate_experiment_change(config.task, spec)  # structural, not llama-specific
        except ValueError as error:
            raise VLLMWorkflowError(str(error)) from error
        decision_ref = self._save_evidence(
            record.task_id,
            "vllm/agent-decision",
            decision,
            producer="optimization-agent",
        )
        spec_ref = self._save_evidence(
            record.task_id,
            "vllm/experiment-spec",
            spec,
            producer="optimization-agent",
        )
        return self.advance(
            record,
            {"agent_decision": decision_ref, "experiment_spec": spec_ref},
        )

    def _active_experiment_spec(
        self, record: VLLMWorkflowRecord
    ) -> VLLMExperimentSpec:
        completion = self._latest_completion(record, VLLMStage.AGENT_DECISION)
        return self.store.load_json(
            record.task_id,
            completion.evidence["experiment_spec"].path,
            VLLMExperimentSpec,
        )

    def record_candidate_server_start(
        self, record: VLLMWorkflowRecord, result: Any
    ) -> VLLMWorkflowRecord:
        if record.current_stage != VLLMStage.EXPERIMENT:
            raise VLLMWorkflowError("candidate server starts only at EXPERIMENT")
        if record.candidate_server_start is not None:
            raise VLLMWorkflowError("candidate server start was already recorded")
        config = self.load_config(record.task_id)
        experiment = self._active_experiment_spec(record)
        spec = build_server_spec(config, self.store, experiment=experiment)
        if not self._server_started(result, spec):
            if _lookup(result, "record", "requires_explicit_stop") is True:
                return self._record_start_failure(
                    record,
                    spec,
                    _lookup(result, "record"),
                    candidate=True,
                    message="candidate result failed its identity/health gate",
                )
            raise VLLMWorkflowError(
                "candidate server is not healthy with verified immutable identity"
            )
        artifact = self._save_evidence(
            record.task_id,
            f"vllm/experiments/{experiment.id}/server-start",
            result,
            producer="vllm-adapter",
        )
        active = self._with_active_server(
            record, _lookup(result, "record"), phase="candidate_server"
        )
        updated = VLLMWorkflowRecord.model_validate(
            {
                **active.model_dump(),
                "candidate_server_start": artifact,
                "candidate_server_request_hash": spec.request_hash,
                "revision": record.revision + 1,
                "updated_at": utc_now(),
            }
        )
        return self._save_update(record, updated)

    def start_candidate_server(
        self, record: VLLMWorkflowRecord, adapter: VLLMServerLifecyclePort
    ) -> VLLMWorkflowRecord:
        config = self.load_config(record.task_id)
        if self._budget_exhausted(record, config):
            return self.finalize_gpu_budget(record)
        experiment = self._active_experiment_spec(record)
        spec = build_server_spec(config, self.store, experiment=experiment)
        try:
            result = adapter.start_or_resume(spec)
        except Exception as error:
            if getattr(error, "requires_explicit_stop", False):
                return self._record_start_failure(
                    record,
                    spec,
                    getattr(error, "record", None),
                    candidate=True,
                    message=f"{type(error).__name__}: {error}",
                )
            raise
        return self.record_candidate_server_start(record, result)

    def stop_failed_candidate_server(
        self, record: VLLMWorkflowRecord, adapter: VLLMServerLifecyclePort
    ) -> VLLMWorkflowRecord:
        """Clean an exact leaked candidate PID before allowing another start."""

        config = self._load_config_artifact(record.task_id)
        try:
            self._verify_model_snapshot(config, require_persisted=True)
            live_coordinate_invalid = False
        except VLLMWorkflowError:
            live_coordinate_invalid = True
        budget_cleanup = (
            record.active_gpu_phase == "candidate_server"
            and self._budget_exhausted(record, config)
        )
        coordinate_cleanup = (
            record.active_gpu_phase == "candidate_server" and live_coordinate_invalid
        )
        if (
            record.current_stage not in {VLLMStage.EXPERIMENT, VLLMStage.QUALITY}
            or (
                not record.candidate_server_requires_stop
                and not budget_cleanup
                and not coordinate_cleanup
            )
            or record.candidate_server_request_hash is None
        ):
            raise VLLMWorkflowError("there is no failed candidate server to stop")
        experiment = self._active_experiment_spec(record)
        spec = build_server_spec(config, self.store, experiment=experiment)
        if spec.request_hash != record.candidate_server_request_hash:
            raise VLLMWorkflowError("failed candidate request identity drifted")
        result = adapter.stop(expected_request_hash=spec.request_hash)
        if not self._server_stopped(result, spec):
            raise VLLMWorkflowError("failed candidate server cleanup was not exact")
        gpu_seconds, gpu_by_phase = self._server_stop_accounting(record, result)
        artifact = self._save_evidence(
            record.task_id,
            f"vllm/experiments/{experiment.id}/failed-server-stop",
            result,
            producer="vllm-adapter",
        )
        updated = VLLMWorkflowRecord.model_validate(
            {
                **record.model_dump(),
                "candidate_server_request_hash": None,
                "candidate_server_requires_stop": False,
                "candidate_server_start_failure": None,
                "candidate_run_failure": None,
                "candidate_server_start": None,
                "pending_quality_result": None,
                "pending_candidate_result": None,
                "current_stage": VLLMStage.EXPERIMENT,
                "active_gpu_started_at": None,
                "active_gpu_phase": None,
                "gpu_seconds_used": gpu_seconds,
                "gpu_seconds_by_phase": gpu_by_phase,
                "revision": record.revision + 1,
                "updated_at": utc_now(),
            }
        )
        self.store.append_event(
            record.task_id,
            "vllm_failed_candidate_cleanup",
            {"stop_artifact": artifact.path},
        )
        saved = self._save_update(record, updated)
        if saved.gpu_seconds_used >= self._gpu_budget_seconds(config):
            return self._finalize_gpu_budget(
                saved, config, extra_evidence={"server_stop": artifact}
            )
        return saved

    def record_experiment(
        self, record: VLLMWorkflowRecord, result: ExperimentResult
    ) -> VLLMWorkflowRecord:
        if record.current_stage != VLLMStage.EXPERIMENT:
            raise VLLMWorkflowError("experiment evidence is valid only at EXPERIMENT")
        if record.candidate_server_start is None or not self.store.verify_artifact(
            record.task_id, record.candidate_server_start
        ):
            raise VLLMWorkflowError("candidate benchmark requires a verified server start")
        config = self.load_config(record.task_id)
        spec = self._active_experiment_spec(record)
        if result.experiment_id != spec.id:
            raise VLLMWorkflowError("experiment result id does not match the selected spec")
        self._require_benchmark_metrics(config, result)
        if record.candidate_server_request_hash is None:
            raise VLLMWorkflowError("candidate server request identity is missing")
        self._require_environment_coordinates(
            config,
            result,
            server_request_hash=record.candidate_server_request_hash,
        )
        artifact = self._save_evidence(
            record.task_id,
            f"vllm/experiments/{spec.id}/result",
            result,
            producer="vllm-benchmark-adapter",
        )
        return self.advance(
            record,
            {
                "candidate_server_start": record.candidate_server_start,
                "experiment_result": artifact,
            },
        )

    def run_experiment(
        self, record: VLLMWorkflowRecord, adapter: VLLMBenchmarkPort
    ) -> VLLMWorkflowRecord:
        try:
            config = self.load_config(record.task_id)
            if self._budget_exhausted(record, config):
                return self.request_active_server_cleanup(
                    record, reason="GPU time budget exhausted before candidate benchmark"
                )
            spec = self._active_experiment_spec(record)
            return self.record_experiment(record, adapter.run_experiment(config, spec))
        except Exception as error:
            return self.request_active_server_cleanup(
                record,
                reason=f"candidate benchmark failed: {type(error).__name__}: {error}",
            )

    def _latest_completion(
        self, record: VLLMWorkflowRecord, stage: VLLMStage
    ) -> VLLMStageCompletion:
        completion = next(
            (item for item in reversed(record.completions) if item.stage == stage),
            None,
        )
        if completion is None:
            raise VLLMWorkflowError(f"workflow has no completed {stage} evidence")
        corrupt = [
            name
            for name, artifact in completion.evidence.items()
            if not self.store.verify_artifact(record.task_id, artifact)
        ]
        if corrupt:
            raise VLLMWorkflowError(
                f"completed {stage} evidence failed integrity verification: "
                + ", ".join(sorted(corrupt))
            )
        return completion

    def record_quality(
        self, record: VLLMWorkflowRecord, quality: QualityResult
    ) -> VLLMWorkflowRecord:
        if record.current_stage != VLLMStage.QUALITY:
            raise VLLMWorkflowError("quality evidence is valid only at QUALITY")
        if record.pending_quality_result is not None:
            raise VLLMWorkflowError("candidate quality was already recorded")
        experiment_ref = self._latest_completion(
            record, VLLMStage.EXPERIMENT
        ).evidence["experiment_result"]
        candidate = self.store.load_json(
            record.task_id, experiment_ref.path, ExperimentResult
        )
        config = self.load_config(record.task_id)
        if quality.status == RunStatus.SUCCEEDED:
            if quality.coordinate_hash != config.quality_protocol.coordinate_sha256:
                raise VLLMWorkflowError(
                    "candidate quality protocol hash does not match immutable config"
                )
            if quality.representation_hash != config.model.snapshot_digest:
                raise VLLMWorkflowError(
                    "candidate quality is not bound to the model snapshot"
                )
            if (
                config.task.quality.require_correctness
                and quality.correctness_passed is None
            ):
                raise VLLMWorkflowError("candidate quality lacks a correctness verdict")
            required_accuracies = {
                item.metric
                for item in config.task.quality.resolved_accuracy_requirements()
            }
            missing_accuracies = required_accuracies - set(quality.accuracies)
            if missing_accuracies:
                raise VLLMWorkflowError(
                    "candidate quality lacks required accuracies: "
                    + ", ".join(sorted(missing_accuracies))
                )
        updated_candidate = candidate.model_copy(update={"quality": quality})
        quality_ref = self._save_evidence(
            record.task_id,
            f"vllm/experiments/{candidate.experiment_id}/quality",
            quality,
            producer="external-vllm-quality-adapter",
        )
        candidate_ref = self._save_evidence(
            record.task_id,
            f"vllm/experiments/{candidate.experiment_id}/quality-bound-result",
            updated_candidate,
            producer="vllm-workflow",
        )
        updated = VLLMWorkflowRecord.model_validate(
            {
                **record.model_dump(),
                "pending_quality_result": quality_ref,
                "pending_candidate_result": candidate_ref,
                "revision": record.revision + 1,
                "updated_at": utc_now(),
            }
        )
        return self._save_update(record, updated)

    def run_quality(
        self, record: VLLMWorkflowRecord, adapter: VLLMQualityPort
    ) -> VLLMWorkflowRecord:
        try:
            config = self.load_config(record.task_id)
            if self._budget_exhausted(record, config):
                return self.request_active_server_cleanup(
                    record, reason="GPU time budget exhausted before candidate quality"
                )
            spec = self._active_experiment_spec(record)
            return self.record_quality(record, adapter.evaluate(config, spec))
        except Exception as error:
            return self.request_active_server_cleanup(
                record,
                reason=f"candidate quality failed: {type(error).__name__}: {error}",
            )

    def stop_candidate_server(
        self, record: VLLMWorkflowRecord, adapter: VLLMServerLifecyclePort
    ) -> VLLMWorkflowRecord:
        """Stop the exact candidate only after benchmark and quality are durable."""

        if record.current_stage != VLLMStage.QUALITY:
            raise VLLMWorkflowError("candidate server stops only at QUALITY")
        if (
            record.pending_quality_result is None
            or record.pending_candidate_result is None
        ):
            raise VLLMWorkflowError("candidate quality must be durable before server stop")
        if record.candidate_server_request_hash is None:
            raise VLLMWorkflowError("candidate server request identity is missing")
        config = self._load_config_artifact(record.task_id)
        experiment = self._active_experiment_spec(record)
        spec = build_server_spec(config, self.store, experiment=experiment)
        if spec.request_hash != record.candidate_server_request_hash:
            raise VLLMWorkflowError("candidate server request hash drifted before stop")
        result = adapter.stop(expected_request_hash=spec.request_hash)
        if not self._server_stopped(result, spec):
            raise VLLMWorkflowError("candidate server stop did not verify the exact PID request")
        gpu_seconds, gpu_by_phase = self._server_stop_accounting(record, result)
        stop_ref = self._save_evidence(
            record.task_id,
            f"vllm/experiments/{experiment.id}/server-stop",
            result,
            producer="vllm-adapter",
        )
        accounted = record.model_copy(
            update={
                "active_gpu_started_at": None,
                "active_gpu_phase": None,
                "gpu_seconds_used": gpu_seconds,
                "gpu_seconds_by_phase": gpu_by_phase,
            }
        )
        saved = self.advance(
            accounted,
            {
                "quality_result": record.pending_quality_result,
                "candidate_result": record.pending_candidate_result,
                "candidate_server_stop": stop_ref,
            },
        )
        if saved.gpu_seconds_used >= self._gpu_budget_seconds(config):
            return self._finalize_gpu_budget(
                saved, config, extra_evidence={"server_stop": stop_ref}
            )
        return saved

    def decide(self, record: VLLMWorkflowRecord) -> VLLMWorkflowRecord:
        if record.current_stage != VLLMStage.DECIDE:
            raise VLLMWorkflowError("Gate evaluation is valid only at DECIDE")
        config = self.load_config(record.task_id)
        baseline_ref = self._latest_completion(
            record, VLLMStage.BASELINE
        ).evidence["baseline"]
        candidate_ref = self._latest_completion(
            record, VLLMStage.QUALITY
        ).evidence["candidate_result"]
        baseline = self.store.load_json(
            record.task_id, baseline_ref.path, BaselineResult
        )
        candidate = self.store.load_json(
            record.task_id, candidate_ref.path, ExperimentResult
        )
        decision = self.gate.evaluate(config.task, baseline, candidate)
        decision_ref = self._save_evidence(
            record.task_id,
            "vllm/gate",
            decision,
            producer="gate",
        )
        return self.advance(
            record, {"gate_decision": decision_ref}, gate_decision=decision
        )

    def resume(self, record: VLLMWorkflowRecord) -> VLLMWorkflowRecord:
        config = self.load_config(record.task_id)
        updated = self.engine.resume_inconclusive(
            record, profile_available=config.profile.mode != "unavailable"
        )
        return self._save_update(record, updated)

    def next_action(self, record: VLLMWorkflowRecord) -> VLLMNextAction:
        """Return one and only one honest action for the current durable state."""

        if record.status != WorkflowStatus.ACTIVE:
            kind = (
                VLLMNextActionKind.RESUME_INCONCLUSIVE
                if record.status == WorkflowStatus.INCONCLUSIVE
                else VLLMNextActionKind.COMPLETE
            )
            return VLLMNextAction(
                task_id=record.task_id,
                stage=record.current_stage,
                kind=kind,
                details={
                    "status": record.status.value,
                    "reasons": (
                        record.terminal_decision.reasons
                        if record.terminal_decision is not None
                        else []
                    ),
                },
            )

        stage = record.current_stage
        kind = {
            VLLMStage.INSPECT: VLLMNextActionKind.PROVIDE_INSPECTION,
            VLLMStage.SERVER_START: VLLMNextActionKind.START_SERVER,
            VLLMStage.BASELINE: VLLMNextActionKind.RUN_BASELINE,
            VLLMStage.AGENT_DECISION: VLLMNextActionKind.SUBMIT_AGENT_DECISION,
            VLLMStage.EXPERIMENT: VLLMNextActionKind.RUN_EXPERIMENT,
            VLLMStage.QUALITY: VLLMNextActionKind.RUN_QUALITY,
            VLLMStage.DECIDE: VLLMNextActionKind.EVALUATE_GATE,
        }.get(stage)
        details: dict[str, Any] = {}
        live_coordinate_error: VLLMWorkflowError | None = None
        try:
            config = self.load_config(record.task_id)
        except VLLMWorkflowError as error:
            if record.active_gpu_phase is None:
                raise
            live_coordinate_error = error
            config = self._load_config_artifact(record.task_id)
        budget_seconds = self._gpu_budget_seconds(config)
        projected_seconds = self._projected_gpu_seconds(record)
        if live_coordinate_error is not None:
            details.update(
                {
                    "reason": (
                        "live immutable coordinates failed verification while a GPU "
                        f"process may be active: {live_coordinate_error}"
                    ),
                    "request_hash": (
                        record.baseline_server_request_hash
                        if record.active_gpu_phase == "baseline_server"
                        else record.candidate_server_request_hash
                    ),
                }
            )
            kind = (
                VLLMNextActionKind.STOP_BASELINE_SERVER
                if record.active_gpu_phase == "baseline_server"
                else VLLMNextActionKind.STOP_EXPERIMENT_SERVER
            )
            details["coordinator_method"] = (
                "stop_baseline_server"
                if record.active_gpu_phase == "baseline_server"
                else "stop_failed_candidate_server"
            )
        elif projected_seconds >= budget_seconds:
            details.update(
                {
                    "gpu_seconds_used_or_active": projected_seconds,
                    "gpu_seconds_budget": budget_seconds,
                    "reason": "paid GPU time budget is exhausted",
                }
            )
            if record.active_gpu_phase == "baseline_server":
                kind = VLLMNextActionKind.STOP_BASELINE_SERVER
                details["request_hash"] = record.baseline_server_request_hash
                details["coordinator_method"] = "stop_baseline_server"
            elif record.active_gpu_phase == "candidate_server":
                kind = VLLMNextActionKind.STOP_EXPERIMENT_SERVER
                details["request_hash"] = record.candidate_server_request_hash
                details["coordinator_method"] = "stop_failed_candidate_server"
            else:
                kind = VLLMNextActionKind.FINALIZE_GPU_BUDGET
        elif stage == VLLMStage.SERVER_START and record.baseline_server_requires_stop:
            kind = VLLMNextActionKind.STOP_BASELINE_SERVER
            details["request_hash"] = record.baseline_server_request_hash
            details["reason"] = "failed start may still own a live GPU process"
        elif stage == VLLMStage.BASELINE and record.baseline_server_requires_stop:
            kind = VLLMNextActionKind.STOP_BASELINE_SERVER
            details["request_hash"] = record.baseline_server_request_hash
            details["reason"] = "baseline run failed or was aborted"
        elif stage == VLLMStage.PROFILE_APPROVAL:
            state = record.profile_approval
            if record.baseline_server_stop is None:
                kind = VLLMNextActionKind.STOP_BASELINE_SERVER
                details["request_hash"] = record.baseline_server_request_hash
            elif config.profile.mode == "unavailable":
                kind = VLLMNextActionKind.RESOLVE_PROFILE_ENVIRONMENT
                details["reason"] = config.profile.unavailable_reason
            elif state is None:
                kind = VLLMNextActionKind.REQUEST_PROFILE_APPROVAL
                details["profile_command_sha256"] = config.profile.command_sha256
            elif state.status == VLLMApprovalStatus.REQUESTED:
                kind = VLLMNextActionKind.APPROVE_PROFILE
                details.update(
                    {"request_id": state.request_id, "request_hash": state.request_hash}
                )
            elif state.status == VLLMApprovalStatus.APPROVED:
                kind = VLLMNextActionKind.RUN_APPROVED_PROFILE
                details.update(
                    {"request_id": state.request_id, "request_hash": state.request_hash}
                )
            else:
                kind = VLLMNextActionKind.RECORD_PROFILE_RESULT
                details["request_id"] = state.request_id
        elif stage == VLLMStage.EXPERIMENT:
            if record.candidate_server_requires_stop:
                kind = VLLMNextActionKind.STOP_EXPERIMENT_SERVER
                details["request_hash"] = record.candidate_server_request_hash
                details["reason"] = "failed candidate start may still own a live GPU process"
            elif record.candidate_server_start is None:
                kind = VLLMNextActionKind.START_EXPERIMENT_SERVER
            else:
                kind = VLLMNextActionKind.RUN_EXPERIMENT
                details["request_hash"] = record.candidate_server_request_hash
        elif stage == VLLMStage.QUALITY:
            if record.candidate_server_requires_stop:
                kind = VLLMNextActionKind.STOP_EXPERIMENT_SERVER
                details["request_hash"] = record.candidate_server_request_hash
                details["reason"] = "candidate benchmark/quality failed or was aborted"
            elif record.pending_quality_result is None:
                kind = VLLMNextActionKind.RUN_QUALITY
            else:
                kind = VLLMNextActionKind.STOP_EXPERIMENT_SERVER
                details["request_hash"] = record.candidate_server_request_hash
        details.setdefault("gpu_seconds_accounted", record.gpu_seconds_used)
        details.setdefault(
            "gpu_seconds_remaining",
            max(0.0, budget_seconds - projected_seconds),
        )
        details.setdefault(
            "gpu_budget_mode",
            "durable_accounting_on_coordinator_calls; external supervisor required",
        )
        assert kind is not None
        return VLLMNextAction(
            task_id=record.task_id,
            stage=stage,
            kind=kind,
            required_artifacts=sorted(VLLM_STAGE_REQUIREMENTS[stage]),
            details=details,
        )

    def run_until_pause(self, task_id: str) -> VLLMNextAction:
        """Load durable state and report work; never infer that work ran."""

        return self.next_action(self.load(task_id))


def build_server_spec(
    config: VLLMCampaignConfig,
    store: ExperimentStore,
    *,
    experiment: VLLMExperimentSpec | None = None,
) -> VLLMServerSpec:
    """Build the adapter's fully pinned server request from immutable config."""

    from .vllm_adapter import VLLMServerSpec

    runtime = config.task.runtime
    label = "baseline" if experiment is None else experiment.id
    stdout = store.task_dir(config.task.id) / f"artifacts/vllm-{label}.stdout.log"
    stderr = store.task_dir(config.task.id) / f"artifacts/vllm-{label}.stderr.log"
    argv = config.serving.server_argv if experiment is None else experiment.server_argv
    environment = (
        config.serving.server_env if experiment is None else experiment.server_env
    )
    return VLLMServerSpec(
        argv=tuple(argv),
        env=environment,
        unset_env=(
            "HIP_VISIBLE_DEVICES",
            "HSA_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "HSA_OVERRIDE_GFX_VERSION",
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONSTARTUP",
        ),
        cwd=str(config.serving.cwd),
        image_digest=(
            f"sha256:{runtime.image_digest}"
            if runtime.image_digest is not None
            else None
        ),
        native_executable_sha256=runtime.executable_sha256,
        environment_manifest_sha256=runtime.environment_manifest_sha256,
        vllm_version=runtime.version,
        model=str(config.model.local_path),
        model_revision=config.model.revision,
        model_snapshot_sha256=config.model.snapshot_digest,
        tensor_parallel_size=config.serving.tensor_parallel_size,
        dtype=config.serving.dtype,
        quantization=config.serving.quantization,
        config={
            **config.serving.engine_config,
            "campaign_identity_sha256": config.identity_sha256,
            "outer_image_digest": runtime.image_digest,
            "environment_manifest_sha256": runtime.environment_manifest_sha256,
            "model_id": config.model.model_id,
            "model_snapshot_digest": config.model.snapshot_digest,
            "tokenizer_revision": config.model.tokenizer_revision,
            "device": config.device.model_dump(mode="json"),
            "serving_protocol_sha256": config.serving.coordinate_sha256,
            "experiment_id": experiment.id if experiment is not None else None,
            "change": (
                experiment.change.model_dump(mode="json")
                if experiment is not None
                else None
            ),
        },
        host=config.serving.host,
        port=config.serving.port,
        health_path="/v1/models",
        expected_served_model=config.model.model_id,
        startup_timeout_seconds=config.serving.timeout_seconds,
        request_timeout_seconds=2,
        shutdown_timeout_seconds=15,
        poll_interval_seconds=0.5,
        stdout_path=str(stdout),
        stderr_path=str(stderr),
    )


__all__ = [
    "LocalVLLMBenchmarkPort",
    "LocalVLLMQualityPort",
    "RocmMCPVLLMProfilePort",
    "VLLMBenchmarkPort",
    "VLLMCommandPort",
    "VLLMEnvironmentPort",
    "VLLMInspectionPort",
    "VLLMProfilePort",
    "VLLMQualityPort",
    "VLLMServerLifecyclePort",
    "VLLMWorkflowCoordinator",
    "VLLMWorkflowEngine",
    "VLLMWorkflowError",
    "VLLM_CONFIG_PATH",
    "VLLM_MODEL_SNAPSHOT_MANIFEST_PATH",
    "VLLM_STAGE_REQUIREMENTS",
    "VLLM_WORKFLOW_PATH",
    "build_server_spec",
]
