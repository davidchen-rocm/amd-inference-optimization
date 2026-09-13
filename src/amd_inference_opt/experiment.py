"""Staged execution and gating for live llama.cpp experiments."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .change_policy import validate_experiment_change
from .command import CommandRunner, command_request_sha256
from .gates import GateEngine
from .llama_cpp import LlamaCppError, parse_llama_bench_json, sha256_file
from .models import (
    ArtifactRef,
    BaselineResult,
    BenchmarkResult,
    DecisionOutcome,
    ExperimentResult,
    ExperimentSpec,
    GateDecision,
    MetricSeries,
    OptimizationTask,
    QualityExecution,
    QualityExecutionState,
    QualityResult,
    RunIdentity,
    RunStatus,
    WorkflowRecord,
    WorkflowStage,
)
from .quality import (
    Q4ThreeWayQualityProtocol,
    Q8RuntimeQualityProtocol,
    QualityAdapterError,
    normalize_q4_threeway_quality,
    normalize_q8_runtime_quality,
)
from .quality_execution import QualityExecutionManager
from .runner import ExperimentRunner, runner_spec_from_domain
from .store import ExperimentStore, StoreError
from .workflow import WorkflowEngine


class ExperimentExecutionError(RuntimeError):
    """The active ExperimentSpec could not be executed or normalized."""


_LLAMA_HARDENED_CAMPAIGNS = {
    "q4_rdna",
    "llama_cpp_q8",
    "llama_cpp_mixed_quant",
    "llama_cpp_shape_kernel",
    "llama_cpp_kv_cache",
    "llama_cpp_consumer_amd_final",
}

_LLAMA_STANDARD_QUALITY_CAMPAIGNS = {
    "llama_cpp_q8",
    "llama_cpp_mixed_quant",
    "llama_cpp_shape_kernel",
    "llama_cpp_kv_cache",
    "llama_cpp_consumer_amd_final",
}


def _campaign_kind(task: OptimizationTask) -> str:
    """Return the typed campaign, retaining compatibility with older Q4 tasks."""

    configured = str(getattr(task, "campaign_kind", "generic"))
    if configured != "generic":
        return configured
    if task.metadata.get("live_q4rdna") == "true":
        return "q4_rdna"
    return configured


def _resolved_experiment_worktree(
    task: OptimizationTask,
    spec: ExperimentSpec,
    store: ExperimentStore,
) -> Path:
    if spec.worktree_path is not None:
        return spec.worktree_path.resolve()
    return (
        store.task_dir(task.id)
        / "workspaces"
        / spec.id
    ).resolve()


def _read_model_input_once(path: Path) -> tuple[str, int]:
    """Hash and size a stable regular model from one streaming read."""

    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ExperimentExecutionError("model input must be a regular file")
            digest = hashlib.sha256()
            packed_bytes = 0
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                packed_bytes += len(chunk)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise ExperimentExecutionError(f"cannot read model input: {path}") from error
    try:
        current = path.stat()
    except OSError as error:
        raise ExperimentExecutionError(f"cannot revalidate model input: {path}") from error
    if packed_bytes <= 0:
        raise ExperimentExecutionError("model input cannot be empty")
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or packed_bytes != after.st_size
        or after.st_dev != current.st_dev
        or after.st_ino != current.st_ino
        or after.st_size != current.st_size
        or after.st_mtime_ns != current.st_mtime_ns
        or after.st_ctime_ns != current.st_ctime_ns
    ):
        raise ExperimentExecutionError("model input changed while it was being hashed")
    return digest.hexdigest(), packed_bytes


def _freeze_model_input_provenance(
    task: OptimizationTask,
    spec: ExperimentSpec,
    store: ExperimentStore,
) -> ArtifactRef:
    """Bind the exact candidate model without mutating CREATE_EXPERIMENT evidence."""

    from .experiment_bundle import (
        ModelInputProvenanceV1,
        model_input_provenance_path,
    )

    candidate_declared = spec.change.candidate_model_path is not None
    raw_path = spec.change.candidate_model_path if candidate_declared else task.model.path
    expected_sha256 = (
        spec.change.candidate_model_sha256 if candidate_declared else task.model.sha256
    )
    quantization = (
        spec.change.candidate_model_quantization
        if candidate_declared
        else task.model.quantization
    )
    if raw_path is None or expected_sha256 is None:
        raise ExperimentExecutionError(
            "experiment model input requires a path and frozen SHA-256"
        )
    model_path = raw_path.expanduser().resolve(strict=False)
    actual_sha256, packed_bytes = _read_model_input_once(model_path)
    if actual_sha256 != expected_sha256:
        raise ExperimentExecutionError("model input SHA-256 does not match its declaration")
    provenance = ModelInputProvenanceV1(
        model_path=str(model_path),
        model_sha256=actual_sha256,
        packed_bytes=packed_bytes,
        quantization=quantization,
        architecture=task.model.architecture,
    )
    relative = model_input_provenance_path(spec.id)
    existing = store.artifact_ref(task.id, relative)
    if existing is None:
        return store.save_immutable_json(
            task.id,
            relative,
            provenance,
            producer="experiment-input-freezer",
        )
    if not store.verify_artifact(task.id, existing):
        raise ExperimentExecutionError("stored model input provenance is corrupt")
    try:
        bound = store.load_json(task.id, relative, ModelInputProvenanceV1)
    except (StoreError, ValidationError) as error:
        raise ExperimentExecutionError("stored model input provenance is invalid") from error
    required_coordinates = (
        "model_path",
        "model_sha256",
        "packed_bytes",
        "quantization",
        "architecture",
    )
    if any(
        getattr(bound, coordinate) != getattr(provenance, coordinate)
        for coordinate in required_coordinates
    ):
        raise ExperimentExecutionError(
            "stored model input provenance differs from the current candidate"
        )
    return existing


def freeze_experiment_inputs(
    task: OptimizationTask,
    spec: ExperimentSpec,
    store: ExperimentStore,
) -> ExperimentSpec:
    """Freeze mutable patch and model inputs before an experiment can execute."""

    frozen_spec = spec
    if str(spec.change.kind) != "source_patch":
        _freeze_model_input_provenance(task, frozen_spec, store)
        return frozen_spec
    patch = spec.change.patch_path
    declared = spec.change.patch_sha256
    if patch is None:
        raise ExperimentExecutionError(
            "source experiments require patch_path before persistence"
        )
    if patch.is_symlink() or not patch.is_file():
        raise ExperimentExecutionError(
            "source experiment patch must be a regular, non-symlink file"
        )
    patch_bytes = patch.read_bytes()
    actual = hashlib.sha256(patch_bytes).hexdigest()
    if declared is not None and actual != declared:
        raise ExperimentExecutionError("source experiment patch SHA-256 does not match")
    declared = actual
    relative = f"experiments/{spec.id}/change/patch.diff"
    existing = store.artifact_ref(task.id, relative)
    if existing is None:
        frozen = store.save_immutable_bytes(
            task.id,
            relative,
            patch_bytes,
            producer="experiment-input-freezer",
            media_type="text/x-diff",
        )
    else:
        if existing.sha256 != declared or not store.verify_artifact(task.id, existing):
            raise ExperimentExecutionError("stored source experiment patch is inconsistent")
        frozen = existing
    frozen_path = store.task_dir(task.id) / frozen.path
    frozen_spec = spec.model_copy(
        update={
            "change": spec.change.model_copy(
                update={"patch_path": frozen_path, "patch_sha256": frozen.sha256}
            )
        },
        deep=True,
    )
    _freeze_model_input_provenance(task, frozen_spec, store)
    return frozen_spec


def _bind_execution_spec(
    task: OptimizationTask,
    original_spec: ExperimentSpec,
    original_spec_path: str,
    execution_spec: ExperimentSpec,
    store: ExperimentStore,
) -> tuple[ExperimentSpec, str]:
    """Bind legacy mutable inputs without rewriting CREATE_EXPERIMENT evidence."""

    if execution_spec == original_spec:
        return execution_spec, original_spec_path
    relative = f"experiments/{execution_spec.id}/execution-spec.json"
    existing = store.artifact_ref(task.id, relative)
    if existing is None:
        artifact = store.save_immutable_json(
            task.id,
            relative,
            execution_spec,
            producer="experiment-input-freezer",
        )
    else:
        if not store.verify_artifact(task.id, existing):
            raise ExperimentExecutionError("stored execution spec is corrupt")
        bound = store.load_json(task.id, relative, ExperimentSpec)
        if bound != execution_spec:
            raise ExperimentExecutionError(
                "execution spec is already bound to different inputs"
            )
        artifact = existing
    store.save_json(
        task.id,
        "state/active-experiment.json",
        {
            "experiment_id": execution_spec.id,
            "spec_path": artifact.path,
            "create_experiment_spec_path": original_spec_path,
        },
        producer="workflow",
    )
    return execution_spec, artifact.path


def materialize_q8_source_experiment(
    task: OptimizationTask,
    spec: ExperimentSpec,
    store: ExperimentStore,
) -> ExperimentSpec:
    """Replace Agent-supplied commands with framework-owned Q8 build/run commands.

    Agents choose the patch and hypothesis.  The framework chooses the worktree,
    build directory and exact benchmark argv so a source experiment cannot
    accidentally run the baseline binary.
    """

    supported_campaigns = {
        "llama_cpp_q8",
        "llama_cpp_mixed_quant",
        "llama_cpp_shape_kernel",
        "llama_cpp_kv_cache",
        "llama_cpp_consumer_amd_final",
    }
    if _campaign_kind(task) not in supported_campaigns:
        return spec
    if str(spec.change.kind) == "runtime_config":
        candidate_model = (
            spec.change.candidate_model_path.resolve()
            if spec.change.candidate_model_path is not None
            else task.model.path.resolve()
        )
        expected_model_sha256 = (
            spec.change.candidate_model_sha256
            if spec.change.candidate_model_path is not None
            else task.model.sha256
        )
        if not candidate_model.is_file():
            raise ExperimentExecutionError(
                f"candidate model does not exist: {candidate_model}"
            )
        actual_model_sha256: str | None = None
        from .experiment_bundle import BundleError, resolve_candidate_model_provenance

        try:
            provenance, _ = resolve_candidate_model_provenance(
                store,
                task,
                spec.id,
                spec.model_dump(mode="json"),
            )
        except BundleError as error:
            raise ExperimentExecutionError(str(error)) from error
        if provenance is not None:
            actual_model_sha256 = provenance.model_sha256
        else:
            actual_model_sha256 = sha256_file(candidate_model)
        if expected_model_sha256 is None or actual_model_sha256 != expected_model_sha256:
            raise ExperimentExecutionError("candidate model SHA-256 does not match ChangeSet")
        binary = (
            task.runtime.prepared_binary_path.resolve()
            if task.runtime.prepared_binary_path is not None
            else task.runtime.build_dir.resolve() / "bin" / "llama-bench"
        )
        if not binary.is_file():
            raise ExperimentExecutionError(f"prepared llama-bench is missing: {binary}")
        from .live import _decode_protocol
        from .models import StageCommandSpec

        e2e_protocol = _decode_protocol(
            task,
            binary,
            model_path=candidate_model,
            additional_extra_args=tuple(spec.change.runtime_args),
        )
        smoke_protocol = e2e_protocol.__class__(
            llama_bench_path=str(binary),
            model_path=str(candidate_model),
            generation_tokens=(1,),
            prompt_tokens=(0,),
            batch_size=e2e_protocol.batch_size,
            ubatch_size=e2e_protocol.ubatch_size,
            threads=e2e_protocol.threads,
            repetitions=1,
            warmup_runs=e2e_protocol.warmup_runs,
            gpu_layers=e2e_protocol.gpu_layers,
            device_id=e2e_protocol.device_id,
            device_selector=e2e_protocol.device_selector,
            timeout_seconds=e2e_protocol.timeout_seconds,
            extra_args=e2e_protocol.extra_args,
            cwd=str(_resolved_experiment_worktree(task, spec, store)),
        )
        commands = {
            "smoke": StageCommandSpec(
                argv=list(smoke_protocol.argv),
                timeout_seconds=task.benchmark.timeout_seconds,
            ),
            "microbench": StageCommandSpec(
                argv=list(e2e_protocol.argv),
                timeout_seconds=task.benchmark.timeout_seconds,
            ),
            "e2e": StageCommandSpec(
                argv=list(e2e_protocol.argv),
                timeout_seconds=task.benchmark.timeout_seconds,
            ),
        }
        return spec.model_copy(
            update={
                "commands": {},
                "stage_commands": commands,
                "worktree_path": _resolved_experiment_worktree(task, spec, store),
            },
            deep=True,
        )
    if str(spec.change.kind) != "source_patch":
        return spec
    if spec.change.patch_sha256 is None:
        raise ExperimentExecutionError("Q8 source experiments require patch_sha256")
    worktree = _resolved_experiment_worktree(task, spec, store)
    build_dir = worktree / "build-gpuopt"
    binary = build_dir / "bin" / "llama-bench"
    cmake = task.metadata.get("cmake_path", "cmake")
    jobs = task.metadata.get("build_jobs", str(max(1, os.cpu_count() or 1)))
    from .live import _decode_protocol

    e2e_protocol = _decode_protocol(
        task,
        binary,
        additional_extra_args=tuple(spec.change.runtime_args),
    )
    smoke_protocol = e2e_protocol.__class__(
        llama_bench_path=str(binary),
        model_path=str(task.model.path),
        generation_tokens=(1,),
        prompt_tokens=(0,),
        batch_size=e2e_protocol.batch_size,
        ubatch_size=e2e_protocol.ubatch_size,
        threads=e2e_protocol.threads,
        repetitions=1,
        warmup_runs=e2e_protocol.warmup_runs,
        gpu_layers=e2e_protocol.gpu_layers,
        device_id=e2e_protocol.device_id,
        device_selector=e2e_protocol.device_selector,
        timeout_seconds=e2e_protocol.timeout_seconds,
        extra_args=e2e_protocol.extra_args,
        cwd=str(worktree),
    )
    from .models import StageCommandSpec

    commands = {
        "configure": StageCommandSpec(
            argv=[
                cmake,
                "-S",
                str(worktree),
                "-B",
                str(build_dir),
                *task.runtime.build_flags,
            ],
            timeout_seconds=600,
        ),
        "build": StageCommandSpec(
            argv=[
                cmake,
                "--build",
                str(build_dir),
                "--target",
                "llama-bench",
                "llama-cli",
                "llama-perplexity",
                "-j",
                jobs,
            ],
            timeout_seconds=task.budgets.build_timeout_seconds,
        ),
        "smoke": StageCommandSpec(
            argv=list(smoke_protocol.argv),
            timeout_seconds=task.benchmark.timeout_seconds,
        ),
        "microbench": StageCommandSpec(
            argv=list(e2e_protocol.argv),
            timeout_seconds=task.benchmark.timeout_seconds,
        ),
        "e2e": StageCommandSpec(
            argv=list(e2e_protocol.argv),
            timeout_seconds=task.benchmark.timeout_seconds,
        ),
    }
    return spec.model_copy(
        update={
            "commands": {},
            "stage_commands": commands,
            "worktree_path": worktree,
        },
        deep=True,
    )


def _benchmark(command: Any) -> BenchmarkResult:
    if command is None:
        return BenchmarkResult(status=RunStatus.SKIPPED, failure_reason="step not declared")
    if not command.succeeded:
        status = RunStatus.TIMED_OUT if command.timed_out else RunStatus.FAILED
        return BenchmarkResult(status=status, failure_reason="command failed")
    try:
        parsed = parse_llama_bench_json(command.stdout)
    except (json.JSONDecodeError, LlamaCppError) as error:
        return BenchmarkResult(status=RunStatus.FAILED, failure_reason=str(error))
    records = parsed.by_test_id()
    if "tg128" not in records or "tg512" not in records:
        return BenchmarkResult(
            status=RunStatus.FAILED,
            failure_reason="benchmark must contain both tg128 and tg512",
        )
    tg128 = MetricSeries(
        unit="tokens/s", samples=list(records["tg128"].samples_tokens_per_second)
    )
    tg512 = MetricSeries(
        unit="tokens/s", samples=list(records["tg512"].samples_tokens_per_second)
    )
    return BenchmarkResult(
        status=RunStatus.SUCCEEDED,
        metrics={
            "tokens_per_second": tg128,
            "tokens_per_second_tg128": tg128,
            "tokens_per_second_tg512": tg512,
        },
    )


def _benchmark_generic(command: Any, generation_tokens: int) -> BenchmarkResult:
    if command is None:
        return BenchmarkResult(status=RunStatus.SKIPPED, failure_reason="step not declared")
    if not command.succeeded:
        status = RunStatus.TIMED_OUT if command.timed_out else RunStatus.FAILED
        return BenchmarkResult(status=status, failure_reason="command failed")
    try:
        parsed = parse_llama_bench_json(command.stdout)
    except (json.JSONDecodeError, LlamaCppError) as error:
        return BenchmarkResult(status=RunStatus.FAILED, failure_reason=str(error))
    selected = parsed.by_test_id().get(f"tg{generation_tokens}")
    if selected is None and len(parsed.records) == 1:
        selected = parsed.records[0]
    if selected is None:
        return BenchmarkResult(
            status=RunStatus.FAILED,
            failure_reason=f"benchmark has no tg{generation_tokens} record",
        )
    return BenchmarkResult(
        status=RunStatus.SUCCEEDED,
        metrics={
            "tokens_per_second": MetricSeries(
                unit="tokens/s", samples=list(selected.samples_tokens_per_second)
            )
        },
    )


def _quality_generic(command: Any) -> tuple[QualityResult | None, QualityResult]:
    if command is None:
        return None, QualityResult(status=RunStatus.SKIPPED)
    if not command.succeeded:
        status = RunStatus.TIMED_OUT if command.timed_out else RunStatus.FAILED
        return None, QualityResult(status=status)
    try:
        document = json.loads(command.stdout)
        if not isinstance(document, dict):
            raise ValueError("quality stdout must be a JSON object")
        if "candidate" in document:
            candidate = QualityResult.model_validate(document["candidate"])
            baseline = (
                QualityResult.model_validate(document["baseline"])
                if "baseline" in document
                else None
            )
            return baseline, candidate
        return None, QualityResult.model_validate(document)
    except (json.JSONDecodeError, ValidationError, ValueError):
        return None, QualityResult(status=RunStatus.FAILED)
def _build_status(value: str) -> RunStatus:
    return {
        "built": RunStatus.SUCCEEDED,
        "reused": RunStatus.REUSED,
        "failed": RunStatus.FAILED,
        "not_run": RunStatus.SKIPPED,
    }[value]


def _complete(
    store: ExperimentStore,
    record: WorkflowRecord,
    evidence: dict[str, str],
) -> WorkflowRecord:
    artifacts: dict[str, ArtifactRef] = {}
    for key, path in evidence.items():
        artifact = store.artifact_ref(record.task_id, path)
        if artifact is None or not store.verify_artifact(record.task_id, artifact):
            raise ExperimentExecutionError(f"stage evidence is missing or corrupt: {path}")
        artifacts[key] = artifact
    updated = WorkflowEngine().complete_stage_from_artifacts(record, artifacts)
    store.save_workflow(updated)
    store.append_event(
        record.task_id,
        "stage_completed",
        {"stage": record.current_stage, "evidence_ids": evidence},
    )
    return updated


def _load_active(
    task: OptimizationTask, store: ExperimentStore
) -> tuple[ExperimentSpec, str]:
    pointer = store.load_json(task.id, "state/active-experiment.json")
    spec_path = pointer.get("spec_path") if isinstance(pointer, dict) else None
    if not isinstance(spec_path, str):
        raise ExperimentExecutionError("active experiment pointer is invalid")
    return store.load_json(task.id, spec_path, ExperimentSpec), spec_path


def _without_quality(spec: ExperimentSpec) -> ExperimentSpec:
    return spec.model_copy(
        update={
            "commands": {key: value for key, value in spec.commands.items() if key != "quality"},
            "stage_commands": {
                key: value for key, value in spec.stage_commands.items() if key != "quality"
            },
        },
        deep=True,
    )


def _candidate_model_coordinates(
    task: OptimizationTask,
    spec: ExperimentSpec,
    store: ExperimentStore,
) -> tuple[Path, str, str]:
    from .experiment_bundle import BundleError, resolve_candidate_model_provenance

    try:
        provenance, _ = resolve_candidate_model_provenance(
            store,
            task,
            spec.id,
            spec.model_dump(mode="json"),
        )
    except BundleError as error:
        raise ExperimentExecutionError(str(error)) from error
    if provenance is None:
        raise ExperimentExecutionError("candidate model provenance is not frozen")
    if provenance.quantization is None:
        raise ExperimentExecutionError("candidate model quantization is not frozen")
    return (
        Path(provenance.model_path),
        provenance.model_sha256,
        provenance.quantization,
    )


def _candidate_environment_and_identity(
    task: OptimizationTask,
    spec: ExperimentSpec,
    baseline: BaselineResult,
    binary: Path,
    command: tuple[str, ...],
    command_runner: CommandRunner,
    artifact_dir: Path,
    store: ExperimentStore,
    *,
    runner_result: Any | None = None,
) -> tuple[Any, RunIdentity, dict[str, Any]]:
    from .live import (
        _environment,
        _prepared_source,
        _runtime_environment,
        _runtime_environment_hash,
        _runtime_libraries_hash,
        _text_hash,
    )

    runtime_env, unset = _runtime_environment(
        task,
        candidate_env=dict(spec.change.env),
        candidate_unset=list(spec.change.unset_env),
        runtime_binary=binary if str(spec.change.kind) == "source_patch" else None,
    )
    source = (
        Path(runner_result.execution_root).resolve()
        if runner_result is not None and str(spec.change.kind) == "source_patch"
        else _prepared_source(task)
    )
    libraries = command_runner.run(
        ["ldd", str(binary)],
        cwd=source,
        env=runtime_env,
        unset_env=unset,
        timeout_seconds=60,
        stdout_path=artifact_dir / "candidate-libraries.stdout",
        stderr_path=artifact_dir / "candidate-libraries.stderr",
    )
    binary_digest = sha256_file(binary)
    from .live import _protocol_hash

    candidate_model, candidate_model_sha256, candidate_quantization = (
        _candidate_model_coordinates(task, spec, store)
    )
    protocol_hash = _protocol_hash(
        task,
        binary,
        model_path=candidate_model,
        additional_extra_args=tuple(spec.change.runtime_args),
    )
    if str(spec.change.kind) == "source_patch":
        if runner_result is None or runner_result.change is None:
            raise ExperimentExecutionError("source experiment is missing patch provenance")
        patch_sha = runner_result.change.patch_sha256
        if patch_sha is None or patch_sha != spec.change.patch_sha256:
            raise ExperimentExecutionError(
                "candidate source identity does not match the declared patch"
            )
        source_hash = _text_hash(
            json.dumps(
                {
                    "base_commit": task.runtime.base_commit,
                    "patch_sha256": patch_sha,
                    "changed_files": sorted(runner_result.change.changed_files),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        source_hash = task.metadata.get("frozen_patch_sha256") or _text_hash(
            task.runtime.base_commit
        )
    environment = _environment(
        task,
        commit=task.runtime.base_commit,
        model_sha256=candidate_model_sha256,
        binary_sha256=binary_digest,
        protocol_hash=protocol_hash,
        runtime_environment=runtime_env,
        runtime_binary=binary,
    )
    if spec.change.candidate_model_path is not None:
        values = dict(environment.values)
        values.update(
            {
                "representation_change_declared": "true",
                "source_model_sha256": baseline.environment.values["model_sha256"],
                "candidate_quantization": candidate_quantization,
            }
        )
        environment = environment.model_copy(update={"values": values})
    identity = RunIdentity(
        protocol_hash=protocol_hash,
        binary_sha256=binary_digest,
        source_snapshot_sha256=source_hash,
        model_sha256=candidate_model_sha256,
        runtime_libraries_hash=_runtime_libraries_hash(task, libraries.stdout),
        environment_hash=_runtime_environment_hash(
            runtime_env,
            runtime_binary=binary,
        ),
        sidecar_sha256=(
            task.model.sidecar_sha256
            if "LLAMA_Q4_RDNA_SIDECAR" in spec.change.env
            else None
        ),
        command_hashes={
            "e2e": command_request_sha256(
                command,
                cwd=source,
                env=runtime_env,
                unset_env=unset,
                timeout_seconds=task.benchmark.timeout_seconds,
            )
        },
    )
    detail = {
        "runtime_environment": runtime_env,
        "unset_environment": list(unset),
        "runtime_libraries": libraries.to_dict(),
        "binary": str(binary.resolve()),
        "binary_sha256": binary_digest,
        "source_snapshot_sha256": source_hash,
        "model": str(candidate_model),
        "model_sha256": candidate_model_sha256,
        "model_quantization": candidate_quantization,
        "patch_sha256": (
            runner_result.change.patch_sha256
            if runner_result is not None and runner_result.change is not None
            else None
        ),
    }
    return environment, identity, detail


def _finalize(
    task: OptimizationTask,
    spec: ExperimentSpec,
    decision: Any,
    store: ExperimentStore,
    *,
    terminal: bool,
) -> None:
    if not terminal:
        return
    report = {
        "schema_version": 1,
        "task_id": task.id,
        "experiment_id": spec.id,
        "decision": decision.model_dump(mode="json"),
        "change": spec.change.model_dump(mode="json"),
    }
    store.save_json(task.id, "reports/final.json", report, producer="gate-engine")
    if decision.outcome != DecisionOutcome.ACCEPT:
        return
    if spec.change.patch_path is not None:
        store.import_artifact(
            task.id,
            "reports/final.patch",
            spec.change.patch_path,
            producer="accepted-experiment",
            media_type="text/x-diff",
        )
    else:
        store.save_json(
            task.id,
            "reports/final-config.json",
            {
                "schema_version": 1,
                "kind": spec.change.kind,
                "env": spec.change.env,
                "unset_env": spec.change.unset_env,
                "binary_reused": spec.change.require_rebuild is False,
                "candidate_model": (
                    {
                        "path": str(spec.change.candidate_model_path.resolve()),
                        "sha256": spec.change.candidate_model_sha256,
                        "quantization": spec.change.candidate_model_quantization,
                    }
                    if spec.change.candidate_model_path is not None
                    else None
                ),
            },
            producer="accepted-experiment",
        )
        manifest_path = task.metadata.get("prepared_runtime_manifest")
        if manifest_path:
            try:
                preparation = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                frozen_patch = Path(str(preparation["frozen_patch"]))
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise ExperimentExecutionError(
                    "accepted runtime config has invalid prepared-runtime provenance"
                ) from error
            source_patch = store.import_artifact(
                task.id,
                "reports/final-source.patch",
                frozen_patch,
                producer="prepared-runtime-prerequisite",
                media_type="text/x-diff",
            )
            store.save_json(
                task.id,
                "reports/final-source-provenance.json",
                {
                    "role": "prepared runtime prerequisite",
                    "artifact": source_patch.model_dump(mode="json"),
                    "historical_old_to_split_patch": False,
                    "preparation_manifest": manifest_path,
                },
                producer="accepted-experiment",
            )


def _publish_experiment_bundle(
    task: OptimizationTask,
    spec: ExperimentSpec,
    store: ExperimentStore,
) -> str | None:
    """Publish a derived result view without changing the authoritative verdict."""

    from .experiment_bundle import BundleError, refresh_experiment_bundle

    try:
        refresh_experiment_bundle(store, task.id, spec.id)
    except (BundleError, StoreError, ValueError, OSError) as error:
        store.append_event(
            task.id,
            "experiment_bundle_failed",
            {"experiment_id": spec.id, "error": str(error)},
        )
        return str(error)
    return None


def _execute_generic_experiment(
    task: OptimizationTask,
    record: WorkflowRecord,
    store: ExperimentStore,
    runner: ExperimentRunner | None,
) -> dict[str, Any]:
    """Retain the generic MVP path for tasks outside the hardened Q4 live slice."""

    spec, spec_path = _load_active(task, store)
    frozen_spec = freeze_experiment_inputs(task, spec, store)
    spec, spec_path = _bind_execution_spec(
        task, spec, spec_path, frozen_spec, store
    )
    experiment_dir = f"experiments/{spec.id}"
    artifact_dir = store.task_dir(task.id) / experiment_dir / "runner"
    execution_spec = runner_spec_from_domain(task, spec, artifact_dir)
    execution_spec = replace(
        execution_spec,
        worktree_path=str(_resolved_experiment_worktree(task, spec, store)),
    )
    run = (runner or ExperimentRunner()).execute(execution_spec)
    raw = store.save_json(
        task.id,
        f"{experiment_dir}/runner-result.json",
        run.to_dict(),
        producer="experiment-runner",
    )
    steps = run.step_results
    smoke = steps.get("smoke")
    sidecar_required = "LLAMA_Q4_RDNA_SIDECAR" in spec.change.env
    activation_text = (
        f"{smoke.stdout}\n{smoke.stderr}" if smoke is not None else ""
    )
    sidecar_loaded = (
        "Q4_RDNA: loaded" in activation_text and "device 0" in activation_text
        if sidecar_required
        else None
    )
    activation = store.save_json(
        task.id,
        f"{experiment_dir}/q4rdna-activation.json",
        {
            "sidecar_required": sidecar_required,
            "signature": "Q4_RDNA: loaded ... device 0",
            "signature_matched": sidecar_loaded,
            "smoke_command_succeeded": smoke.succeeded if smoke is not None else None,
        },
        producer="experiment-runner",
    )
    micro = _benchmark_generic(steps.get("microbench"), task.workload.generation_tokens)
    e2e = _benchmark_generic(steps.get("e2e"), task.workload.generation_tokens)
    baseline_quality, candidate_quality = _quality_generic(steps.get("quality"))
    baseline = store.load_json(task.id, "artifacts/baseline.json", BaselineResult)
    if baseline_quality is not None:
        baseline.quality = baseline_quality
        store.save_json(
            task.id, "artifacts/baseline.json", baseline, producer="quality-evaluator"
        )
    result = ExperimentResult(
        experiment_id=spec.id,
        environment=baseline.environment.model_copy(deep=True),
        run_identity=(
            baseline.run_identity.model_copy(deep=True)
            if baseline.run_identity
            else None
        ),
        build_status=_build_status(run.build_status),
        smoke_passed=(
            smoke.succeeded and (sidecar_loaded is not False)
            if smoke is not None
            else None
        ),
        microbenchmark=micro,
        e2e=e2e,
        quality=candidate_quality,
        failure_reason=(
            f"runner stopped at {run.failure_stage}" if run.failure_stage else None
        ),
    )
    result_artifact = store.save_json(
        task.id, f"{experiment_dir}/result.json", result, producer="experiment-runner"
    )
    build_artifact = store.save_json(
        task.id,
        f"{experiment_dir}/build-result.json",
        {
            "status": result.build_status,
            "smoke_passed": result.smoke_passed,
            "runner_result": raw.path,
            "q4rdna_activation": activation.path,
        },
        producer="experiment-runner",
    )
    record = _complete(store, record, {"build_result": build_artifact.path})
    micro_artifact = store.save_json(
        task.id, f"{experiment_dir}/microbenchmark.json", micro, producer="experiment-runner"
    )
    record = _complete(store, record, {"microbenchmark_result": micro_artifact.path})
    e2e_artifact = store.save_json(
        task.id, f"{experiment_dir}/e2e-result.json", e2e, producer="experiment-runner"
    )
    record = _complete(store, record, {"e2e_result": e2e_artifact.path})
    quality_artifact = store.save_json(
        task.id,
        f"{experiment_dir}/quality-result.json",
        candidate_quality,
        producer="quality-evaluator",
    )
    record = _complete(store, record, {"quality_result": quality_artifact.path})
    decision = GateEngine().evaluate(task, baseline, result)
    gate_artifact = store.save_json(
        task.id,
        f"{experiment_dir}/gate-decision.json",
        decision,
        producer="gate-engine",
    )
    can_continue = (
        decision.outcome == DecisionOutcome.REJECT
        and record.experiment_count < task.budgets.max_experiments
    )
    record = WorkflowEngine().complete_stage_from_artifacts(
        record,
        {"gate_decision": gate_artifact},
        gate_decision=decision,
        can_continue_after_reject=can_continue,
    )
    store.save_workflow(record)
    store.append_event(
        task.id,
        "experiment_decided",
        {
            "experiment_id": spec.id,
            "spec": spec_path,
            "result": result_artifact.path,
            "decision": decision.outcome,
            "gate": gate_artifact.path,
        },
    )
    _finalize(task, spec, decision, store, terminal=not can_continue)
    bundle_warning = _publish_experiment_bundle(task, spec, store)
    return {
        "task_id": task.id,
        "experiment_id": spec.id,
        "decision": decision,
        "workflow_status": record.status,
        "current_stage": record.current_stage,
        "continue_with_next_hypothesis": can_continue,
        "bundle_warning": bundle_warning,
    }


def execute_active_experiment(
    task: OptimizationTask,
    record: WorkflowRecord,
    store: ExperimentStore,
    *,
    runner: ExperimentRunner | None = None,
) -> dict[str, Any]:
    """Run through E2E, then pause for an independently approved profile."""

    if record.current_stage != WorkflowStage.PATCH_AND_BUILD:
        raise ExperimentExecutionError(
            f"active experiment execution requires PATCH_AND_BUILD, got {record.current_stage}"
        )
    hardened_live = (
        _campaign_kind(task) in _LLAMA_HARDENED_CAMPAIGNS
        or task.runtime.prepared_binary_path is not None
        or task.environment.require_run_identity
    )
    if not hardened_live:
        return _execute_generic_experiment(task, record, store, runner)
    from .live import _prepared_source, _runtime_environment, benchmark_protocol_argv

    original_spec, spec_path = _load_active(task, store)
    validate_experiment_change(task, original_spec)
    spec = freeze_experiment_inputs(task, original_spec, store)
    spec = materialize_q8_source_experiment(task, spec, store)
    spec, spec_path = _bind_execution_spec(
        task, original_spec, spec_path, spec, store
    )
    experiment_dir = f"experiments/{spec.id}"
    artifact_dir = store.task_dir(task.id) / experiment_dir / "runner"
    if str(spec.change.kind) == "source_patch":
        worktree = _resolved_experiment_worktree(task, spec, store)
        expected_binary = worktree / "build-gpuopt" / "bin" / "llama-bench"
    else:
        expected_binary = (
            task.runtime.prepared_binary_path.resolve()
            if task.runtime.prepared_binary_path is not None
            else Path(task.runtime.build_dir).resolve() / "bin" / "llama-bench"
        )
    candidate_model, _, candidate_quantization = _candidate_model_coordinates(
        task, spec, store
    )
    exact_e2e = benchmark_protocol_argv(
        task,
        expected_binary,
        model_path=candidate_model,
        additional_extra_args=tuple(spec.change.runtime_args),
    )
    declared_e2e = spec.stage_commands.get("e2e")
    declared_argv = (
        tuple(declared_e2e.argv)
        if declared_e2e is not None
        else tuple(spec.commands.get("e2e", ()))
    )
    if declared_argv != exact_e2e:
        raise ExperimentExecutionError(
            "experiment e2e command differs from the frozen tg128/tg512 protocol"
        )
    execution_spec = runner_spec_from_domain(task, _without_quality(spec), artifact_dir)
    base_env, base_unset = _runtime_environment(
        task,
        runtime_binary=(expected_binary if str(spec.change.kind) == "source_patch" else None),
    )
    baseline = store.load_json(task.id, "artifacts/baseline.json", BaselineResult)
    execution_spec = replace(
        execution_spec,
        source_repo=str(_prepared_source(task)),
        execution_root=str(_prepared_source(task)),
        binary_path=str(expected_binary),
        expected_binary_sha256=(
            baseline.run_identity.binary_sha256
            if baseline.run_identity and str(spec.change.kind) == "runtime_config"
            else None
        ),
        base_environment=base_env,
        base_unset_environment=tuple(
            name for name in base_unset if name not in spec.change.env
        ),
    )
    selected_runner = runner or ExperimentRunner()
    run = selected_runner.execute(execution_spec)
    if str(spec.change.kind) == "source_patch" and run.failure_stage is None:
        if run.binary is None or Path(run.binary.path).resolve() != expected_binary.resolve():
            raise ExperimentExecutionError(
                "source experiment did not produce the declared worktree binary"
            )
        if run.build_status != "built":
            raise ExperimentExecutionError(
                "source experiment candidate binary is not bound to a successful build"
            )
    raw = store.save_json(
        task.id,
        f"{experiment_dir}/runner-result.json",
        run.to_dict(),
        producer="experiment-runner",
    )
    steps = run.step_results
    smoke = steps.get("smoke")
    sidecar_required = (
        _campaign_kind(task) == "q4_rdna"
        and "LLAMA_Q4_RDNA_SIDECAR" in spec.change.env
    )
    activation_text = (
        f"{smoke.stdout}\n{smoke.stderr}" if smoke is not None else ""
    )
    sidecar_loaded = (
        "Q4_RDNA: loaded" in activation_text and "device 0" in activation_text
        if sidecar_required
        else None
    )
    activation = None
    if _campaign_kind(task) == "q4_rdna":
        activation = store.save_json(
            task.id,
            f"{experiment_dir}/q4rdna-activation.json",
            {
                "sidecar_required": sidecar_required,
                "signature": "Q4_RDNA: loaded ... device 0",
                "signature_matched": sidecar_loaded,
                "smoke_command_succeeded": smoke.succeeded if smoke is not None else None,
            },
            producer="experiment-runner",
        )
    q8_offload = None
    if _campaign_kind(task) in _LLAMA_STANDARD_QUALITY_CAMPAIGNS:
        from .live import q8_offload_evidence

        try:
            parsed_smoke = (
                parse_llama_bench_json(smoke.stdout)
                if smoke is not None and smoke.succeeded
                else None
            )
            q8_offload = (
                q8_offload_evidence(
                    parsed_smoke,
                    expected_quantization=candidate_quantization,
                )
                if parsed_smoke is not None
                else {"status": "failed", "checks": {}, "observed": None}
            )
        except (json.JSONDecodeError, LlamaCppError):
            q8_offload = {"status": "failed", "checks": {}, "observed": None}
        store.save_json(
            task.id,
            f"{experiment_dir}/q8-offload.json",
            q8_offload,
            producer="experiment-runner",
        )
    micro = _benchmark(steps.get("microbench"))
    e2e = _benchmark(steps.get("e2e"))
    environment, identity, environment_detail = _candidate_environment_and_identity(
        task,
        spec,
        baseline,
        expected_binary,
        exact_e2e,
        selected_runner.commands,
        artifact_dir,
        store,
        runner_result=run,
    )
    result = ExperimentResult(
        experiment_id=spec.id,
        environment=environment,
        run_identity=identity,
        build_status=_build_status(run.build_status),
        smoke_passed=(
            smoke.succeeded
            and (sidecar_loaded is not False)
            and (q8_offload is None or q8_offload["status"] == "passed")
            if smoke is not None
            else None
        ),
        microbenchmark=micro,
        e2e=e2e,
        quality=None,
        failure_reason=(
            f"runner stopped at {run.failure_stage}" if run.failure_stage else None
        ),
    )
    result_artifact = store.save_json(
        task.id, f"{experiment_dir}/result-pre-quality.json", result, producer="experiment-runner"
    )
    store.save_json(
        task.id,
        f"{experiment_dir}/environment.json",
        environment_detail,
        producer="experiment-runner",
    )
    pre_gate = GateEngine().evaluate_performance_pre_gate(task, baseline, result)
    pre_gate_artifact = store.save_json(
        task.id,
        f"{experiment_dir}/performance-pre-gate.json",
        pre_gate,
        producer="gate-engine",
    )
    build_artifact = store.save_json(
        task.id,
        f"{experiment_dir}/build-result.json",
        {
            "status": result.build_status,
            "smoke_passed": result.smoke_passed,
            "runner_result": raw.path,
            "q4rdna_activation": activation.path if activation is not None else None,
            "q8_offload": (
                f"{experiment_dir}/q8-offload.json" if q8_offload is not None else None
            ),
        },
        producer="experiment-runner",
    )
    record = _complete(store, record, {"build_result": build_artifact.path})
    micro_artifact = store.save_json(
        task.id, f"{experiment_dir}/microbenchmark.json", micro, producer="experiment-runner"
    )
    record = _complete(store, record, {"microbenchmark_result": micro_artifact.path})
    e2e_artifact = store.save_json(
        task.id, f"{experiment_dir}/e2e-result.json", e2e, producer="experiment-runner"
    )
    if pre_gate.outcome != DecisionOutcome.REJECT:
        record = _complete(store, record, {"e2e_result": e2e_artifact.path})
    store.save_json(
        task.id,
        "state/active-execution.json",
        {
            "experiment_id": spec.id,
            "spec_path": spec_path,
            "result_path": result_artifact.path,
            "pre_gate_path": pre_gate_artifact.path,
            "e2e_path": e2e_artifact.path,
            "profile_complete": False,
            "quality_required": pre_gate.outcome == DecisionOutcome.ACCEPT,
        },
        producer="workflow",
    )
    return {
        "task_id": task.id,
        "experiment_id": spec.id,
        "current_stage": record.current_stage,
        "performance_pre_gate": pre_gate,
        "paused_for": "candidate_profile_approval",
        "profile_target": spec.id,
    }


def _run_fresh_quality(
    task: OptimizationTask,
    spec: ExperimentSpec,
    store: ExperimentStore,
    result: ExperimentResult,
) -> tuple[QualityResult, QualityResult, str] | QualityExecution:
    if _campaign_kind(task) in _LLAMA_STANDARD_QUALITY_CAMPAIGNS:
        return _run_fresh_q8_quality(task, spec, store, result)
    root = Path(task.metadata["math_rule_loop_dir"]).resolve()
    output = (
        store.task_dir(task.id)
        / "experiments"
        / spec.id
        / "quality"
        / "threeway-output.json"
    )
    protocol = Q4ThreeWayQualityProtocol(
        math_rule_loop_dir=str(root),
        hf_model_path=task.metadata["hf_model_path"],
        q4_k_model_path=str(task.model.path),
        output_path=str(output),
    )
    command = protocol.command()
    execution = QualityExecutionManager(store).start_or_resume(
        task.id,
        spec.id,
        command,
        output_path=output.relative_to(store.task_dir(task.id)).as_posix(),
        spec_coordinates={
            "adapter": "q4-three-way-v1",
            "baseline_representation_hash": task.model.sha256,
            "candidate_representation_hash": (
                result.run_identity.sidecar_sha256 if result.run_identity else None
            ),
        },
    )
    execution_path = (
        f"experiments/{spec.id}/quality/attempts/{execution.attempt_id}.json"
    )
    if execution.state in {
        QualityExecutionState.STARTED,
        QualityExecutionState.RUNNING,
        QualityExecutionState.ORPHANED,
    }:
        return execution
    if execution.state != QualityExecutionState.COMPLETED:
        status = (
            RunStatus.TIMED_OUT
            if execution.state == QualityExecutionState.TIMED_OUT
            else RunStatus.FAILED
        )
        return QualityResult(status=status), QualityResult(status=status), execution_path
    required_artifacts = {
        execution.spec.output_path: execution.output_artifact,
        execution.spec.stdout_path: execution.stdout_artifact,
        execution.spec.stderr_path: execution.stderr_artifact,
    }
    for expected_path, artifact in required_artifacts.items():
        if (
            artifact is None
            or artifact.path != expected_path
            or not store.verify_artifact(task.id, artifact)
        ):
            raise ExperimentExecutionError(
                f"completed quality artifact is missing or corrupt: {expected_path}"
            )
    try:
        pair = normalize_q4_threeway_quality(
            output,
            baseline_representation_hash=task.model.sha256,
            candidate_representation_hash=(
                result.run_identity.sidecar_sha256 if result.run_identity else None
            ),
        )
    except (OSError, json.JSONDecodeError, QualityAdapterError):
        failed = QualityResult(status=RunStatus.FAILED)
        return failed, failed, execution_path
    raw = store.import_artifact(
        task.id,
        f"experiments/{spec.id}/quality/raw-threeway.json",
        output,
        producer="quality-evaluator",
        media_type="application/json",
    )
    return pair.baseline, pair.candidate, raw.path


def _run_fresh_q8_quality(
    task: OptimizationTask,
    spec: ExperimentSpec,
    store: ExperimentStore,
    result: ExperimentResult,
    *,
    retry_failed: bool = False,
) -> tuple[QualityResult, QualityResult, str] | QualityExecution:
    raw_template = task.metadata.get("quality_command_json") or task.metadata.get(
        "q8_quality_command_json"
    )
    if raw_template is None:
        failed = QualityResult(status=RunStatus.FAILED)
        return failed, failed, "quality-command-missing"
    try:
        decoded = json.loads(raw_template)
    except json.JSONDecodeError as error:
        raise ExperimentExecutionError("quality command JSON is invalid") from error
    if not isinstance(decoded, list) or not all(
        isinstance(item, str) and item for item in decoded
    ):
        raise ExperimentExecutionError("quality command JSON must be a string list")
    candidate_model, candidate_model_sha256, _ = _candidate_model_coordinates(
        task, spec, store
    )
    if task.model.sha256 is None:
        raise ExperimentExecutionError("Q8 quality requires a frozen baseline model SHA-256")
    if candidate_model_sha256 != task.model.sha256:
        rewritten: list[str] = []
        index = 0
        while index < len(decoded):
            item = decoded[index]
            if item == "--model" and index + 1 < len(decoded):
                rewritten.extend(
                    [
                        "--baseline-model",
                        "{baseline_model}",
                        "--candidate-model",
                        "{candidate_model}",
                    ]
                )
                index += 2
                continue
            if item == "--expected-model-sha256" and index + 1 < len(decoded):
                rewritten.extend(
                    [
                        "--expected-baseline-model-sha256",
                        task.model.sha256,
                        "--expected-candidate-model-sha256",
                        candidate_model_sha256,
                    ]
                )
                index += 2
                continue
            rewritten.append(item)
            index += 1
        decoded = rewritten
    baseline_binary = task.runtime.prepared_binary_path
    if baseline_binary is None:
        baseline_binary = task.runtime.build_dir.resolve() / "bin" / "llama-bench"
    runner_result = store.load_json(
        task.id,
        f"experiments/{spec.id}/runner-result.json",
    )
    binary = runner_result.get("binary") if isinstance(runner_result, dict) else None
    candidate_binary = binary.get("path") if isinstance(binary, dict) else None
    if not isinstance(candidate_binary, str):
        raise ExperimentExecutionError("Q8 candidate runner result has no binary path")
    output = (
        store.task_dir(task.id)
        / "experiments"
        / spec.id
        / "quality"
        / "q8-runtime-output.json"
    )
    raw_env = task.metadata.get(
        "quality_env_json", task.metadata.get("q8_quality_env_json", "{}")
    )
    try:
        quality_env = json.loads(raw_env)
    except json.JSONDecodeError as error:
        raise ExperimentExecutionError("quality environment JSON is invalid") from error
    if not isinstance(quality_env, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in quality_env.items()
    ):
        raise ExperimentExecutionError("quality environment JSON must be a string mapping")
    protocol = Q8RuntimeQualityProtocol(
        command_template=tuple(decoded),
        baseline_binary=str(baseline_binary),
        candidate_binary=candidate_binary,
        model_path=str(task.model.path),
        candidate_model_path=str(candidate_model),
        output_path=str(output),
        cwd=task.metadata.get(
            "quality_cwd",
            task.metadata.get("q8_quality_cwd", str(_prepared_source_for_quality(task))),
        ),
        environment=quality_env,
        timeout_seconds=float(
            task.metadata.get(
                "quality_timeout_seconds",
                task.metadata.get("q8_quality_timeout_seconds", "14400"),
            )
        ),
    )
    command = protocol.command()
    quality_tools = {
        str(Path(item).resolve()): sha256_file(Path(item).resolve())
        for item in decoded
        if item.endswith(".py") and Path(item).resolve().is_file()
    }
    execution = QualityExecutionManager(store).start_or_resume(
        task.id,
        spec.id,
        command,
        output_path=output.relative_to(store.task_dir(task.id)).as_posix(),
        spec_coordinates={
            "adapter": "q8-runtime-pair-v1",
            "baseline_model_sha256": task.model.sha256,
            "candidate_model_sha256": candidate_model_sha256,
            "baseline_binary": str(Path(baseline_binary).resolve()),
            "candidate_binary": str(Path(candidate_binary).resolve()),
            "quality_tool_sha256": quality_tools,
        },
        retry_failed=retry_failed,
    )
    execution_path = f"experiments/{spec.id}/quality/attempts/{execution.attempt_id}.json"
    if execution.state in {
        QualityExecutionState.STARTED,
        QualityExecutionState.RUNNING,
        QualityExecutionState.ORPHANED,
    }:
        return execution
    if execution.state != QualityExecutionState.COMPLETED:
        status = (
            RunStatus.TIMED_OUT
            if execution.state == QualityExecutionState.TIMED_OUT
            else RunStatus.FAILED
        )
        return QualityResult(status=status), QualityResult(status=status), execution_path
    if execution.output_artifact is None or not store.verify_artifact(
        task.id, execution.output_artifact
    ):
        raise ExperimentExecutionError("Q8 quality output is missing or corrupt")
    try:
        pair = normalize_q8_runtime_quality(
            output,
            expected_model_sha256=task.model.sha256,
            expected_candidate_model_sha256=candidate_model_sha256,
        )
    except (OSError, json.JSONDecodeError, QualityAdapterError):
        failed = QualityResult(status=RunStatus.FAILED)
        return failed, failed, execution_path
    raw = store.import_artifact(
        task.id,
        f"experiments/{spec.id}/quality/raw-q8-runtime.json",
        output,
        producer="quality-evaluator",
        media_type="application/json",
    )
    return pair.baseline, pair.candidate, raw.path


def _prepared_source_for_quality(task: OptimizationTask) -> Path:
    configured = task.metadata.get("prepared_source_path")
    return Path(configured).resolve() if configured else task.runtime.repo_path.resolve()


def _with_semantic_environment_hash(
    result: BaselineResult | ExperimentResult,
    runtime_environment: dict[str, str],
    runtime_binary: Path,
) -> BaselineResult | ExperimentResult:
    from .live import _runtime_environment_hash

    semantic_hash = _runtime_environment_hash(
        runtime_environment,
        runtime_binary=runtime_binary,
    )
    values = dict(result.environment.values)
    values["runtime_environment_hash"] = semantic_hash
    return result.model_copy(
        update={
            "environment": result.environment.model_copy(update={"values": values}),
            "run_identity": result.run_identity.model_copy(
                update={"environment_hash": semantic_hash}
            ),
        },
        deep=True,
    )


def _baseline_for_comparison(
    task: OptimizationTask,
    store: ExperimentStore,
) -> BaselineResult:
    try:
        pointer = store.load_json(task.id, "state/performance-baseline-current.json")
        baseline_path = pointer["path"]
        if not isinstance(baseline_path, str):
            raise TypeError("performance baseline path is not a string")
        baseline = store.load_json(task.id, baseline_path, BaselineResult)
    except (StoreError, KeyError, TypeError):
        baseline = store.load_json(task.id, "artifacts/baseline.json", BaselineResult)
    try:
        detail = store.load_json(task.id, "artifacts/live-baseline/identity.json")
        runtime_environment = detail["runtime_environment"]
        runtime_binary = Path(detail["binary"])
    except (StoreError, KeyError, TypeError):
        return baseline
    if not isinstance(runtime_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in runtime_environment.items()
    ):
        return baseline
    normalized = _with_semantic_environment_hash(
        baseline,
        runtime_environment,
        runtime_binary,
    )
    if not isinstance(normalized, BaselineResult):  # defensive type narrowing
        raise ExperimentExecutionError("normalized baseline has the wrong result type")
    return normalized


def _extended_pair_verification_once(
    task: OptimizationTask,
    spec: ExperimentSpec,
    store: ExperimentStore,
    active: dict[str, Any],
    baseline: BaselineResult,
    result: ExperimentResult,
    pre_gate: GateDecision,
) -> tuple[dict[str, Any], BaselineResult, ExperimentResult, GateDecision]:
    """Resolve repeatable cold-first noise with a larger, symmetric sample set.

    Both arms use the same immutable binary, semantic protocol, environment, and
    sample count. The first sample remains in each result; no outlier is removed.
    """

    if not (
        _campaign_kind(task) in _LLAMA_STANDARD_QUALITY_CAMPAIGNS
        and pre_gate.outcome == DecisionOutcome.INCONCLUSIVE
        and pre_gate.rerun_from_stage == WorkflowStage.E2E_VALIDATION
        and any(check.name == "benchmark_stability" for check in pre_gate.checks)
        and not active.get("extended_pair_verification")
    ):
        return active, baseline, result, pre_gate
    if result.run_identity is None or baseline.run_identity is None:
        raise ExperimentExecutionError("extended pair verification requires run identities")
    repetitions = int(task.metadata.get("extended_stability_samples", "12"))
    if repetitions < task.benchmark.sample_count:
        raise ExperimentExecutionError(
            "extended_stability_samples cannot be below the baseline sample count"
        )
    runner_result = store.load_json(
        task.id, f"experiments/{spec.id}/runner-result.json"
    )
    binary_record = runner_result.get("binary") if isinstance(runner_result, dict) else None
    binary_path = binary_record.get("path") if isinstance(binary_record, dict) else None
    if not isinstance(binary_path, str):
        raise ExperimentExecutionError("extended pair verification has no candidate binary")
    binary = Path(binary_path).resolve()
    if sha256_file(binary) != result.run_identity.binary_sha256:
        raise ExperimentExecutionError("extended pair verification binary changed")
    candidate_model, candidate_model_sha256, candidate_quantization = (
        _candidate_model_coordinates(task, spec, store)
    )
    from .live import (
        _decode_protocol,
        _environment,
        _prepared_source,
        _runtime_environment,
        _runtime_environment_hash,
    )

    baseline_protocol = replace(
        _decode_protocol(task, binary, model_path=task.model.path),
        repetitions=repetitions,
    )
    candidate_protocol = replace(
        _decode_protocol(task, binary, model_path=candidate_model),
        repetitions=repetitions,
    )
    if baseline_protocol.protocol_hash != candidate_protocol.protocol_hash:
        raise ExperimentExecutionError("pair verification semantic protocols differ")
    runtime_environment, unset_environment = _runtime_environment(task)
    attempt_root = f"experiments/{spec.id}/extended-pair-verification/attempt-0001"
    absolute_root = store.task_dir(task.id) / attempt_root
    if absolute_root.exists():
        raise ExperimentExecutionError(
            f"extended pair verification evidence already exists: {attempt_root}"
        )
    request = store.save_json(
        task.id,
        f"{attempt_root}/request.json",
        {
            "schema": "gpuopt.extended-pair-verification.v1",
            "reason": "repeatable cold-first sample made both three-sample runs exceed CV",
            "outlier_policy": "retain_all_samples",
            "repetitions_per_arm": repetitions,
            "baseline_argv": list(baseline_protocol.argv),
            "candidate_argv": list(candidate_protocol.argv),
            "cwd": str(_prepared_source(task)),
            "environment": runtime_environment,
            "unset_environment": list(unset_environment),
            "binary_sha256": result.run_identity.binary_sha256,
            "baseline_model_sha256": task.model.sha256,
            "candidate_model_sha256": candidate_model_sha256,
            "protocol_hash": baseline_protocol.protocol_hash,
        },
        producer="experiment-runner",
    )
    commands = CommandRunner()
    baseline_command = commands.run(
        baseline_protocol.argv,
        cwd=_prepared_source(task),
        env=runtime_environment,
        unset_env=unset_environment,
        timeout_seconds=task.benchmark.timeout_seconds,
        stdout_path=absolute_root / "baseline.stdout",
        stderr_path=absolute_root / "baseline.stderr",
    )
    candidate_command = commands.run(
        candidate_protocol.argv,
        cwd=_prepared_source(task),
        env=runtime_environment,
        unset_env=unset_environment,
        timeout_seconds=task.benchmark.timeout_seconds,
        stdout_path=absolute_root / "candidate.stdout",
        stderr_path=absolute_root / "candidate.stderr",
    )
    baseline_benchmark = _benchmark(baseline_command)
    candidate_benchmark = _benchmark(candidate_command)
    environment_hash = _runtime_environment_hash(
        runtime_environment,
        runtime_binary=binary,
    )
    protocol_hash = baseline_protocol.protocol_hash
    baseline_environment = _environment(
        task,
        commit=task.runtime.base_commit,
        model_sha256=task.model.sha256 or baseline.run_identity.model_sha256,
        binary_sha256=result.run_identity.binary_sha256,
        protocol_hash=protocol_hash,
        runtime_environment=runtime_environment,
        runtime_binary=binary,
    )
    candidate_environment = _environment(
        task,
        commit=task.runtime.base_commit,
        model_sha256=candidate_model_sha256,
        binary_sha256=result.run_identity.binary_sha256,
        protocol_hash=protocol_hash,
        runtime_environment=runtime_environment,
        runtime_binary=binary,
    )
    candidate_values = dict(candidate_environment.values)
    candidate_values.update(
        {
            "representation_change_declared": "true",
            "source_model_sha256": baseline_environment.values["model_sha256"],
            "candidate_quantization": candidate_quantization,
        }
    )
    candidate_environment = candidate_environment.model_copy(
        update={"values": candidate_values}
    )
    cwd = _prepared_source(task)
    baseline_identity = baseline.run_identity.model_copy(
        update={
            "protocol_hash": protocol_hash,
            "environment_hash": environment_hash,
            "command_hashes": {
                "e2e": command_request_sha256(
                    baseline_protocol.argv,
                    cwd=cwd,
                    env=runtime_environment,
                    unset_env=unset_environment,
                    timeout_seconds=task.benchmark.timeout_seconds,
                )
            },
        }
    )
    candidate_identity = result.run_identity.model_copy(
        update={
            "protocol_hash": protocol_hash,
            "model_sha256": candidate_model_sha256,
            "environment_hash": environment_hash,
            "command_hashes": {
                "e2e": command_request_sha256(
                    candidate_protocol.argv,
                    cwd=cwd,
                    env=runtime_environment,
                    unset_env=unset_environment,
                    timeout_seconds=task.benchmark.timeout_seconds,
                )
            },
        }
    )
    paired_baseline = baseline.model_copy(
        update={
            "environment": baseline_environment,
            "run_identity": baseline_identity,
            "benchmark": baseline_benchmark,
        },
        deep=True,
    )
    paired_result = result.model_copy(
        update={
            "environment": candidate_environment,
            "run_identity": candidate_identity,
            "e2e": candidate_benchmark,
        },
        deep=True,
    )
    baseline_command_artifact = store.save_json(
        task.id,
        f"{attempt_root}/baseline-command.json",
        baseline_command.to_dict(),
        producer="experiment-runner",
    )
    candidate_command_artifact = store.save_json(
        task.id,
        f"{attempt_root}/candidate-command.json",
        candidate_command.to_dict(),
        producer="experiment-runner",
    )
    baseline_artifact = store.save_json(
        task.id,
        f"{attempt_root}/baseline-result.json",
        paired_baseline,
        producer="experiment-runner",
    )
    result_artifact = store.save_json(
        task.id,
        f"{attempt_root}/candidate-result.json",
        paired_result,
        producer="experiment-runner",
    )
    e2e_artifact = store.save_json(
        task.id,
        f"{attempt_root}/candidate-e2e-result.json",
        candidate_benchmark,
        producer="experiment-runner",
    )
    gate = GateEngine().evaluate_performance_pre_gate(
        task, paired_baseline, paired_result
    )
    gate_artifact = store.save_json(
        task.id,
        f"{attempt_root}/performance-pre-gate.json",
        gate,
        producer="gate-engine",
    )
    store.save_json(
        task.id,
        "state/performance-baseline-current.json",
        {"path": baseline_artifact.path, "sha256": baseline_artifact.sha256},
        producer="workflow",
    )
    active.update(
        {
            "result_path": result_artifact.path,
            "pre_gate_path": gate_artifact.path,
            "e2e_path": e2e_artifact.path,
            "quality_required": gate.outcome == DecisionOutcome.ACCEPT,
            "extended_pair_verification": {
                "state": "completed",
                "request": request.path,
                "baseline_command": baseline_command_artifact.path,
                "candidate_command": candidate_command_artifact.path,
                "baseline_result": baseline_artifact.path,
                "candidate_result": result_artifact.path,
                "candidate_e2e_result": e2e_artifact.path,
                "gate": gate_artifact.path,
            },
        }
    )
    store.save_json(task.id, "state/active-execution.json", active, producer="workflow")
    store.append_event(
        task.id,
        "extended_pair_verification_completed",
        {
            "experiment_id": spec.id,
            "repetitions_per_arm": repetitions,
            "outlier_policy": "retain_all_samples",
            "outcome": gate.outcome,
            "gate": gate_artifact.path,
        },
    )
    return active, paired_baseline, paired_result, gate


def _candidate_for_comparison(
    task: OptimizationTask,
    spec: ExperimentSpec,
    store: ExperimentStore,
    result_path: str,
) -> ExperimentResult:
    result = store.load_json(task.id, result_path, ExperimentResult)
    try:
        detail = store.load_json(task.id, f"experiments/{spec.id}/environment.json")
        runtime_environment = detail["runtime_environment"]
        runtime_binary = Path(detail["binary"])
    except (StoreError, KeyError, TypeError):
        return result
    if not isinstance(runtime_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in runtime_environment.items()
    ):
        return result
    normalized = _with_semantic_environment_hash(
        result,
        runtime_environment,
        runtime_binary,
    )
    if not isinstance(normalized, ExperimentResult):  # defensive type narrowing
        raise ExperimentExecutionError("normalized candidate has the wrong result type")
    return normalized


def _rerun_inconclusive_e2e_once(
    task: OptimizationTask,
    spec: ExperimentSpec,
    store: ExperimentStore,
    record: WorkflowRecord,
    active: dict[str, Any],
    baseline: BaselineResult,
    result: ExperimentResult,
    pre_gate: GateDecision,
) -> tuple[WorkflowRecord, dict[str, Any], ExperimentResult, GateDecision]:
    """Repeat only the exact candidate E2E command after a noisy sample set.

    The first result remains immutable.  This narrow recovery path never rebuilds,
    changes argv, or drops an outlier; it creates one complete replacement sample
    set and moves only the active pointers to the new hash-bound artifacts.
    """

    if not (
        pre_gate.outcome == DecisionOutcome.INCONCLUSIVE
        and pre_gate.rerun_from_stage == WorkflowStage.E2E_VALIDATION
    ):
        return record, active, result, pre_gate
    maximum = int(task.metadata.get("max_e2e_reruns", "1"))
    completed = int(active.get("e2e_rerun_count", 0))
    if completed >= maximum:
        return record, active, result, pre_gate
    in_flight = active.get("e2e_rerun")
    if isinstance(in_flight, dict) and in_flight.get("state") == "running":
        raise ExperimentExecutionError(
            "an E2E rerun was started but has no terminal evidence; manual recovery is required"
        )

    runner = store.load_json(task.id, f"experiments/{spec.id}/runner-result.json")
    try:
        command = runner["step_results"]["e2e"]
        argv = tuple(command["argv"])
        cwd = Path(command["cwd"]).resolve()
        environment = dict(command["environment"])
        unset_environment = tuple(command["unset_environment"])
        timeout_seconds = float(command["timeout_seconds"])
        stored_request_hash = str(command["request_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise ExperimentExecutionError("stored E2E command evidence is invalid") from error
    declared = spec.stage_commands.get("e2e")
    if declared is None or argv != tuple(declared.argv):
        raise ExperimentExecutionError("E2E rerun argv differs from the persisted ExperimentSpec")
    request_hash = command_request_sha256(
        argv,
        cwd=cwd,
        env=environment,
        unset_env=unset_environment,
        timeout_seconds=timeout_seconds,
    )
    if request_hash != stored_request_hash:
        raise ExperimentExecutionError("stored E2E command request hash does not verify")
    binary = Path(argv[0]).resolve()
    if (
        result.run_identity is None
        or sha256_file(binary) != result.run_identity.binary_sha256
    ):
        raise ExperimentExecutionError("E2E rerun binary identity changed")
    from .live import _protocol_hash, _runtime_environment_hash

    if _protocol_hash(task, binary) != result.run_identity.protocol_hash:
        raise ExperimentExecutionError("E2E rerun protocol identity changed")
    if (
        _runtime_environment_hash(environment, runtime_binary=binary)
        != result.run_identity.environment_hash
    ):
        raise ExperimentExecutionError("E2E rerun environment identity changed")

    attempt_number = completed + 1
    attempt_id = f"attempt-{attempt_number:04d}"
    attempt_root = f"experiments/{spec.id}/e2e-reruns/{attempt_id}"
    absolute_root = store.task_dir(task.id) / attempt_root
    if absolute_root.exists():
        raise ExperimentExecutionError(f"E2E rerun evidence already exists: {attempt_root}")
    request_artifact = store.save_json(
        task.id,
        f"{attempt_root}/request.json",
        {
            "schema": "gpuopt.e2e-rerun-request.v1",
            "attempt_id": attempt_id,
            "experiment_id": spec.id,
            "request_sha256": request_hash,
            "argv": list(argv),
            "cwd": str(cwd),
            "environment": environment,
            "unset_environment": list(unset_environment),
            "timeout_seconds": timeout_seconds,
            "binary_sha256": result.run_identity.binary_sha256,
            "protocol_hash": result.run_identity.protocol_hash,
            "supersedes": active["e2e_path"],
        },
        producer="experiment-runner",
    )
    active["e2e_rerun"] = {
        "attempt_id": attempt_id,
        "state": "running",
        "request": request_artifact.path,
        "request_sha256": request_hash,
    }
    store.save_json(task.id, "state/active-execution.json", active, producer="workflow")
    store.append_event(
        task.id,
        "e2e_rerun_started",
        {"experiment_id": spec.id, "attempt_id": attempt_id, "request": request_artifact.path},
    )
    command_result = CommandRunner().run(
        argv,
        cwd=cwd,
        env=environment,
        unset_env=unset_environment,
        timeout_seconds=timeout_seconds,
        stdout_path=absolute_root / "stdout.log",
        stderr_path=absolute_root / "stderr.log",
    )
    command_artifact = store.save_json(
        task.id,
        f"{attempt_root}/command-result.json",
        command_result.to_dict(),
        producer="experiment-runner",
    )
    benchmark = _benchmark(command_result)
    benchmark_artifact = store.save_json(
        task.id,
        f"{attempt_root}/e2e-result.json",
        benchmark,
        producer="experiment-runner",
    )
    updated_result = result.model_copy(update={"e2e": benchmark}, deep=True)
    result_artifact = store.save_json(
        task.id,
        f"{attempt_root}/result-pre-quality.json",
        updated_result,
        producer="experiment-runner",
    )
    updated_gate = GateEngine().evaluate_performance_pre_gate(
        task, baseline, updated_result
    )
    gate_artifact = store.save_json(
        task.id,
        f"{attempt_root}/performance-pre-gate.json",
        updated_gate,
        producer="gate-engine",
    )
    active.update(
        {
            "result_path": result_artifact.path,
            "pre_gate_path": gate_artifact.path,
            "e2e_path": benchmark_artifact.path,
            "quality_required": updated_gate.outcome == DecisionOutcome.ACCEPT,
            "e2e_rerun_count": attempt_number,
            "e2e_rerun": {
                "attempt_id": attempt_id,
                "state": "completed",
                "request": request_artifact.path,
                "command_result": command_artifact.path,
                "e2e_result": benchmark_artifact.path,
                "gate": gate_artifact.path,
            },
        }
    )
    store.save_json(task.id, "state/active-execution.json", active, producer="workflow")
    store.append_event(
        task.id,
        "e2e_rerun_completed",
        {
            "experiment_id": spec.id,
            "attempt_id": attempt_id,
            "outcome": updated_gate.outcome,
            "e2e_result": benchmark_artifact.path,
            "gate": gate_artifact.path,
        },
    )
    # The prior INCONCLUSIVE completion had already moved to QUALITY_VALIDATION.
    # Put the active workflow back at E2E so the code-owned performance-reject
    # bypass can close a stable slow rerun without fabricating quality evidence.
    updated_record = record.model_copy(
        update={
            "current_stage": WorkflowStage.E2E_VALIDATION,
            "rerun_count": record.rerun_count + 1,
        },
        deep=True,
    )
    store.save_workflow(updated_record)
    return updated_record, active, updated_result, updated_gate


def resume_active_experiment(
    task: OptimizationTask,
    record: WorkflowRecord,
    store: ExperimentStore,
) -> dict[str, Any]:
    """At QUALITY_VALIDATION, profile first, then run quality only when warranted."""

    if record.current_stage not in {
        WorkflowStage.E2E_VALIDATION,
        WorkflowStage.QUALITY_VALIDATION,
    }:
        raise ExperimentExecutionError(
            "experiment resume requires E2E_VALIDATION or QUALITY_VALIDATION, "
            f"got {record.current_stage}"
        )
    from .live import _profile_target_until_pause

    spec, spec_path = _load_active(task, store)
    with store.task_lock(task.id):
        # Re-read all mutable pointers after acquiring the cross-process lock.
        record = store.load_workflow(task.id)
        spec = freeze_experiment_inputs(task, spec, store)
        active = store.load_json(task.id, "state/active-execution.json")
        if active.get("experiment_id") != spec.id:
            raise ExperimentExecutionError("active execution does not match active experiment")
        baseline = _baseline_for_comparison(task, store)
        result = _candidate_for_comparison(task, spec, store, active["result_path"])
        refreshed_pre_gate = GateEngine().evaluate_performance_pre_gate(
            task, baseline, result
        )
        stored_pre_gate = store.load_json(
            task.id, active["pre_gate_path"], GateDecision
        )
        if refreshed_pre_gate != stored_pre_gate:
            refreshed_artifact = store.save_json(
                task.id,
                f"experiments/{spec.id}/performance-pre-gate-semantic-env-v2.json",
                refreshed_pre_gate,
                producer="gate-engine",
            )
            active["pre_gate_path"] = refreshed_artifact.path
            active["quality_required"] = (
                refreshed_pre_gate.outcome == DecisionOutcome.ACCEPT
            )
            store.save_json(
                task.id,
                "state/active-execution.json",
                active,
                producer="workflow",
            )
        record, active, result, refreshed_pre_gate = _rerun_inconclusive_e2e_once(
            task,
            spec,
            store,
            record,
            active,
            baseline,
            result,
            refreshed_pre_gate,
        )
        active, baseline, result, refreshed_pre_gate = (
            _extended_pair_verification_once(
                task,
                spec,
                store,
                active,
                baseline,
                result,
                refreshed_pre_gate,
            )
        )
    profile = _profile_target_until_pause(task, store, target=spec.id)
    if not profile.get("profile_complete"):
        return {
            "task_id": task.id,
            "experiment_id": spec.id,
            "current_stage": record.current_stage,
            **profile,
        }
    if record.current_stage == WorkflowStage.E2E_VALIDATION:
        pre_gate = store.load_json(task.id, active["pre_gate_path"], GateDecision)
        if pre_gate.outcome == DecisionOutcome.ACCEPT:
            e2e_artifact = store.artifact_ref(task.id, active["e2e_path"])
            if e2e_artifact is None:
                raise ExperimentExecutionError(
                    "accepted rerun E2E evidence is not hash-bound"
                )
            record = WorkflowEngine().complete_stage_from_artifacts(
                record,
                {"e2e_result": e2e_artifact},
            )
            store.save_workflow(record)
            store.append_event(
                task.id,
                "stage_completed",
                {
                    "stage": WorkflowStage.E2E_VALIDATION,
                    "evidence_ids": {"e2e_result": e2e_artifact.path},
                    "source": "accepted_extended_or_exact_rerun",
                },
            )
        elif pre_gate.outcome != DecisionOutcome.REJECT:
            raise ExperimentExecutionError(
                "profile completed but performance evidence remains inconclusive"
            )
        else:
            e2e_artifact = store.artifact_ref(task.id, active["e2e_path"])
            gate_artifact = store.artifact_ref(task.id, active["pre_gate_path"])
            if e2e_artifact is None or gate_artifact is None:
                raise ExperimentExecutionError("early rejection evidence is not hash-bound")
            can_continue = record.experiment_count < task.budgets.max_experiments
            record = WorkflowEngine().complete_performance_rejection(
                record,
                e2e_result=e2e_artifact,
                gate_decision_artifact=gate_artifact,
                gate_decision=pre_gate,
                can_continue_after_reject=can_continue,
            )
            store.save_workflow(record)
            store.append_event(
                task.id,
                "experiment_decided",
                {
                    "experiment_id": spec.id,
                    "spec": spec_path,
                    "profile": profile["kernel_evidence"],
                    "quality": "skipped_after_performance_reject",
                    "decision": pre_gate.outcome,
                    "gate": gate_artifact.path,
                },
            )
            _finalize(task, spec, pre_gate, store, terminal=not can_continue)
            bundle_warning = _publish_experiment_bundle(task, spec, store)
            return {
                "task_id": task.id,
                "experiment_id": spec.id,
                "decision": pre_gate,
                "workflow_status": record.status,
                "current_stage": record.current_stage,
                "continue_with_next_hypothesis": can_continue,
                "bundle_warning": bundle_warning,
            }
    if active.get("quality_required"):
        quality_run = _run_fresh_quality(task, spec, store, result)
        if isinstance(quality_run, QualityExecution):
            return {
                "task_id": task.id,
                "experiment_id": spec.id,
                "current_stage": record.current_stage,
                "paused_for": "quality_execution",
                "quality_attempt": quality_run.attempt_id,
                "quality_state": quality_run.state,
                "quality_request_hash": quality_run.request_hash,
                "quality_spec_hash": quality_run.spec_hash,
                "heartbeat_at": quality_run.heartbeat_at,
            }
        baseline_quality, candidate_quality, quality_source = quality_run
        baseline.quality = baseline_quality
        enriched_baseline = store.save_evidence_json(
            task.id,
            f"quality-baselines/{spec.id}",
            baseline,
            producer="quality-evaluator",
        )
        store.save_json(
            task.id,
            "state/quality-baseline-current.json",
            {"path": enriched_baseline.path, "sha256": enriched_baseline.sha256},
            producer="workflow",
        )
    else:
        candidate_quality = QualityResult(status=RunStatus.SKIPPED)
        quality_source = "performance-pre-gate-reject"
    result.quality = candidate_quality
    result_artifact = store.save_json(
        task.id,
        f"experiments/{spec.id}/result.json",
        result,
        producer="experiment-runner",
    )
    quality_artifact = store.save_json(
        task.id,
        f"experiments/{spec.id}/quality-result.json",
        candidate_quality,
        producer="quality-evaluator",
    )
    record = _complete(store, record, {"quality_result": quality_artifact.path})
    decision = GateEngine().evaluate(task, baseline, result)
    gate_artifact = store.save_json(
        task.id,
        f"experiments/{spec.id}/gate-decision.json",
        decision,
        producer="gate-engine",
    )
    can_continue = (
        decision.outcome == DecisionOutcome.REJECT
        and record.experiment_count < task.budgets.max_experiments
    )
    record = WorkflowEngine().complete_stage_from_artifacts(
        record,
        {"gate_decision": gate_artifact},
        gate_decision=decision,
        can_continue_after_reject=can_continue,
    )
    store.save_workflow(record)
    store.append_event(
        task.id,
        "experiment_decided",
        {
            "experiment_id": spec.id,
            "spec": spec_path,
            "result": result_artifact.path,
            "profile": profile["kernel_evidence"],
            "quality_source": quality_source,
            "decision": decision.outcome,
            "gate": gate_artifact.path,
        },
    )
    _finalize(task, spec, decision, store, terminal=not can_continue)
    bundle_warning = _publish_experiment_bundle(task, spec, store)
    return {
        "task_id": task.id,
        "experiment_id": spec.id,
        "decision": decision,
        "workflow_status": record.status,
        "current_stage": record.current_stage,
        "continue_with_next_hypothesis": can_continue,
        "bundle_warning": bundle_warning,
    }


__all__ = [
    "ExperimentExecutionError",
    "execute_active_experiment",
    "resume_active_experiment",
]
