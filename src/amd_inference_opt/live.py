"""Resumable live driver for the first AMD optimization vertical slice.

The driver executes framework-owned observation stages. It intentionally pauses
for an AgentDecision and before every executing ROCm MCP call. Candidate changes
are executed by :mod:`amd_inference_opt.runner` after the Agent has submitted an
ExperimentSpec.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from .agent_handoff import (
    artifact_evidence_id,
    artifact_is_agent_evidence,
    build_agent_context,
    write_agent_context,
)
from .architecture import (
    ArchitectureFamily,
    architecture_profile,
    canonical_gfx_target,
    cmake_gfx_target,
)
from .command import command_request_sha256
from .llama_cpp import (
    LlamaBenchmarkResult,
    LlamaBenchmarkRun,
    LlamaCppAdapter,
    LlamaCppError,
    parse_llama_bench_json,
    sha256_file,
)
from .models import (
    ApprovalReceipt,
    ApprovalRequest,
    ArtifactRef,
    BaselineResult,
    BenchmarkResult,
    CampaignKind,
    EnvironmentFingerprint,
    EvidenceRef,
    InferenceExecutionMap,
    MetricSeries,
    OptimizationTask,
    RunIdentity,
    RunStatus,
    WorkflowRecord,
    WorkflowStage,
    utc_now,
)
from .protocol import (
    BASELINE_UNSET_ENVIRONMENT,
    DecodeBenchmarkProtocol,
    KernelTimingProfileProtocol,
)
from .rocm_mcp import MCPServerConfig, MCPToolCall, RocmIssueAgentClient
from .store import ExperimentStore, StoreError
from .workflow import WorkflowEngine


class LiveWorkflowError(RuntimeError):
    """A live stage could not produce valid evidence."""


_Q4_ENV = BASELINE_UNSET_ENVIRONMENT


def _metadata_json(task: OptimizationTask, key: str, default: Any) -> Any:
    raw = task.metadata.get(key)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise LiveWorkflowError(f"task metadata {key!r} must be JSON") from error


def _runtime_environment(
    task: OptimizationTask,
    *,
    candidate_env: dict[str, str] | None = None,
    candidate_unset: list[str] | tuple[str, ...] = (),
    runtime_binary: str | Path | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Return the device/runtime environment used by direct and MCP workloads."""

    physical_device = str(task.gpu.device_id)
    isolation = {
        "HSA_VISIBLE_DEVICES": physical_device,
        "HIP_VISIBLE_DEVICES": physical_device,
        "ROCR_VISIBLE_DEVICES": physical_device,
    }
    conflicting_isolation = sorted(
        name
        for name, value in (candidate_env or {}).items()
        if name in isolation and value != isolation[name]
    )
    if conflicting_isolation:
        raise LiveWorkflowError(
            "candidate environment cannot override task GPU isolation: "
            + ", ".join(conflicting_isolation)
        )
    library_paths = task.metadata.get(
        "rocm_library_path", "/opt/rocm/core-7.14/lib:/opt/rocm/lib"
    ).split(":")
    if runtime_binary is not None:
        candidate_bin = str(Path(runtime_binary).resolve().parent)
        baseline_bins = {str(task.runtime.build_dir.resolve() / "bin")}
        if task.runtime.prepared_binary_path is not None:
            baseline_bins.add(str(task.runtime.prepared_binary_path.resolve().parent))
        library_paths = [path for path in library_paths if path not in baseline_bins]
        library_paths.insert(0, candidate_bin)
    environment = {
        **isolation,
        "LD_LIBRARY_PATH": ":".join(dict.fromkeys(library_paths)),
    }
    environment.update(candidate_env or {})
    unset = tuple(dict.fromkeys([*_Q4_ENV, *candidate_unset]))
    unset = tuple(name for name in unset if name not in environment)
    return environment, unset


def _protocol_tokens(task: OptimizationTask) -> tuple[int, ...]:
    values = _metadata_json(task, "benchmark_generation_tokens", [128, 512])
    if not isinstance(values, list) or values != [128, 512]:
        raise LiveWorkflowError(
            "Q4_RDNA live benchmark_generation_tokens must be exactly [128, 512]"
        )
    return tuple(values)


def _protocol_extra_args(task: OptimizationTask) -> tuple[str, ...]:
    values = _metadata_json(
        task,
        "benchmark_extra_args",
        [],
    )
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise LiveWorkflowError("benchmark_extra_args must be a JSON string list")
    return tuple(values)


def _profile_capture_limits(task: OptimizationTask) -> dict[str, int]:
    """Return explicit, approval-bound capture limits for ROCm profiling."""

    defaults = {
        "max_trace_bytes": 200_000_000,
        "max_trace_files": 64,
        "max_events_per_type": 10_000,
        "max_percentile_samples_per_kernel": 20,
    }
    values = _metadata_json(task, "profile_capture_limits", defaults)
    if not isinstance(values, dict):
        raise LiveWorkflowError("profile_capture_limits must be a JSON object")
    unknown = sorted(set(values) - set(defaults))
    if unknown:
        raise LiveWorkflowError(
            "profile_capture_limits contains unsupported keys: " + ", ".join(unknown)
        )
    selected = {**defaults, **values}
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in selected.values()
    ):
        raise LiveWorkflowError("profile_capture_limits values must be positive integers")
    return {name: int(selected[name]) for name in defaults}


def _decode_protocol(
    task: OptimizationTask,
    binary: str | Path,
    *,
    model_path: str | Path | None = None,
    additional_extra_args: tuple[str, ...] = (),
) -> DecodeBenchmarkProtocol:
    return DecodeBenchmarkProtocol(
        llama_bench_path=str(binary),
        model_path=str(model_path if model_path is not None else task.model.path),
        generation_tokens=_protocol_tokens(task),
        prompt_tokens=(0,),
        repetitions=task.benchmark.sample_count,
        warmup_runs=task.benchmark.warmup_runs,
        # Visibility maps the selected physical/logical host ordinal to ROCm0
        # inside the workload. Never address the pre-isolation ordinal here.
        device_id=0,
        timeout_seconds=task.benchmark.timeout_seconds,
        extra_args=(*_protocol_extra_args(task), *additional_extra_args),
        cwd=str(_prepared_source(task)),
    )


def benchmark_protocol_argv(
    task: OptimizationTask,
    binary: str | Path,
    *,
    model_path: str | Path | None = None,
    additional_extra_args: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Materialize the single authoritative, dual-decode benchmark protocol."""

    return _decode_protocol(
        task,
        binary,
        model_path=model_path,
        additional_extra_args=additional_extra_args,
    ).argv


def _profile_protocol(task: OptimizationTask, binary: str | Path) -> KernelTimingProfileProtocol:
    return KernelTimingProfileProtocol.from_e2e(_decode_protocol(task, binary))


def profile_protocol_argv(task: OptimizationTask, binary: str | Path) -> tuple[str, ...]:
    """Materialize the canonical short trace command, separate from E2E."""

    return _profile_protocol(task, binary).argv


def _prepared_source(task: OptimizationTask) -> Path:
    configured = task.metadata.get("prepared_source_path")
    return Path(configured).resolve() if configured else task.runtime.repo_path.resolve()


def _protocol_hash(
    task: OptimizationTask,
    binary: str | Path,
    *,
    model_path: str | Path | None = None,
    additional_extra_args: tuple[str, ...] = (),
) -> str:
    return _decode_protocol(
        task,
        binary,
        model_path=model_path,
        additional_extra_args=additional_extra_args,
    ).protocol_hash


def preflight(task: OptimizationTask, store: ExperimentStore) -> dict[str, Any]:
    """Validate live assets and the fixed decode protocol without executing a GPU."""

    model = task.model.path.resolve()
    source = _prepared_source(task)
    build_parent = task.runtime.build_dir.resolve().parent
    mcp_command = Path(task.mcp.command[0]).resolve()
    required_free_gib = float(task.metadata.get("minimum_free_disk_gib", "30"))
    free_gib = shutil.disk_usage(build_parent).free / 1024**3
    problems: list[str] = []
    active_architecture = architecture_profile(
        task.gpu.gfx_target,
        board_type=task.gpu.board_type,
    )
    if active_architecture.family == ArchitectureFamily.UNKNOWN:
        problems.append(f"no active local execution profile exists for {task.gpu.gfx_target}")
    build_gfx_target: str | None = None
    try:
        build_gfx_target = cmake_gfx_target(task.runtime.build_flags)
    except ValueError as error:
        problems.append(str(error))
    else:
        if build_gfx_target != task.gpu.gfx_target:
            problems.append(
                "configured GPU architecture does not match the llama.cpp build: "
                f"{task.gpu.gfx_target} != {build_gfx_target}"
            )
    if not model.is_file():
        problems.append(f"model does not exist: {model}")
    if not source.is_dir() or not (source / "CMakeLists.txt").is_file():
        problems.append(f"prepared llama.cpp source is invalid: {source}")
    preparation_path = task.metadata.get("prepared_runtime_manifest")
    preparation: dict[str, Any] | None = None
    if preparation_path:
        path = Path(preparation_path).resolve()
        try:
            preparation = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"prepared runtime manifest is invalid: {path}: {error}")
        else:
            if preparation.get("worktree") != str(source):
                problems.append("prepared runtime manifest does not own configured source")
            if task.campaign_kind == CampaignKind.Q4_RDNA:
                patch_path = Path(str(preparation.get("frozen_patch", "")))
                expected_patch_sha = task.metadata.get("frozen_patch_sha256")
                if not patch_path.is_file():
                    problems.append(f"frozen Q4 patch does not exist: {patch_path}")
                elif expected_patch_sha and sha256_file(patch_path) != expected_patch_sha:
                    problems.append("frozen Q4 patch SHA-256 does not match task metadata")
    if not mcp_command.is_file() or not os.access(mcp_command, os.X_OK):
        problems.append(f"ROCm MCP command is not executable: {mcp_command}")
    if (
        task.campaign_kind == CampaignKind.Q4_RDNA
        and (task.model.sidecar_path is None or not task.model.sidecar_path.resolve().is_file())
    ):
        problems.append("Q4_RDNA sidecar is required and must exist")
    if task.campaign_kind == CampaignKind.LLAMA_CPP_Q8 and task.model.quantization != "Q8_0":
        problems.append("llama_cpp_q8 campaign requires model.quantization='Q8_0'")
    if free_gib < required_free_gib:
        problems.append(f"only {free_gib:.1f} GiB free; task requires {required_free_gib:.1f} GiB")
    environment, unset = _runtime_environment(task)
    binary = task.runtime.build_dir.resolve() / "bin" / "llama-bench"
    if task.runtime.prepared_binary_path is not None:
        binary = task.runtime.prepared_binary_path.resolve()
        if not binary.is_file():
            problems.append(f"prepared binary does not exist: {binary}")
        elif (
            task.runtime.prepared_binary_sha256 is not None
            and sha256_file(binary) != task.runtime.prepared_binary_sha256
        ):
            problems.append("prepared binary SHA-256 does not match task")
    argv = benchmark_protocol_argv(task, binary)
    payload = {
        "schema_version": 1,
        "status": "ready" if not problems else "failed",
        "campaign_kind": task.campaign_kind,
        "architecture": active_architecture.model_dump(mode="json"),
        "architecture_coordinates": {
            "configured_gfx_target": task.gpu.gfx_target,
            "cmake_gfx_target": build_gfx_target,
            "matches": build_gfx_target == task.gpu.gfx_target,
        },
        "problems": problems,
        "free_disk_gib": free_gib,
        "required_free_disk_gib": required_free_gib,
        "source": str(source),
        "build_dir": str(task.runtime.build_dir.resolve()),
        "model": str(model),
        "sidecar": str(task.model.sidecar_path.resolve()) if task.model.sidecar_path else None,
        "mcp_command": str(mcp_command),
        "benchmark_argv": list(argv),
        "runtime_environment": environment,
        "unset_environment": list(unset),
        "protocol_hash": _protocol_hash(task, binary),
        "prepared_runtime": preparation,
    }
    store.save_json(task.id, "artifacts/preflight.json", payload, producer="live-preflight")
    if problems:
        raise LiveWorkflowError("; ".join(problems))
    return payload


def _save_stage(
    store: ExperimentStore,
    record: WorkflowRecord,
    evidence_ids: dict[str, str],
) -> WorkflowRecord:
    evidence_artifacts: dict[str, ArtifactRef] = {}
    for key, path in evidence_ids.items():
        artifact = store.artifact_ref(record.task_id, path)
        if artifact is None or not store.verify_artifact(record.task_id, artifact):
            raise LiveWorkflowError(f"stage evidence is missing or corrupt: {path}")
        evidence_artifacts[key] = artifact
    updated = WorkflowEngine().complete_stage_from_artifacts(record, evidence_artifacts)
    store.save_workflow(updated)
    store.append_event(
        record.task_id,
        "stage_completed",
        {"stage": record.current_stage, "evidence_ids": evidence_ids},
    )
    return updated


def _build_flags_hash(flags: list[str]) -> str:
    encoded = json.dumps(flags, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _runtime_environment_hash(
    runtime_environment: dict[str, str],
    *,
    runtime_binary: Path | None = None,
) -> str:
    """Hash runtime settings without treating the compared build path as drift."""

    normalized = dict(runtime_environment)
    if runtime_binary is not None and "LD_LIBRARY_PATH" in normalized:
        binary_dir = runtime_binary.resolve().parent
        library_paths = []
        for entry in normalized["LD_LIBRARY_PATH"].split(":"):
            if entry and Path(entry).resolve() == binary_dir:
                library_paths.append("{runtime_binary_dir}")
            else:
                library_paths.append(entry)
        normalized["LD_LIBRARY_PATH"] = ":".join(library_paths)
    return _text_hash(json.dumps(normalized, sort_keys=True, separators=(",", ":")))


def _runtime_libraries_hash(task: OptimizationTask, ldd_output: str) -> str:
    """Hash stable library coordinates, excluding per-process ASLR addresses."""

    normalized = []
    for raw_line in ldd_output.splitlines():
        line = re.sub(r"\s*\(0x[0-9a-fA-F]+\)\s*$", "", raw_line.strip())
        if line:
            normalized.append(" ".join(line.split()))
    prepared_library_hash = None
    manifest_path = task.metadata.get("prepared_runtime_manifest")
    if manifest_path:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            prepared_library_hash = manifest.get("build", {}).get("libggml_hip_sha256")
        except (OSError, json.JSONDecodeError, AttributeError):
            prepared_library_hash = None
    return _text_hash(
        json.dumps(
            {
                "libraries": sorted(normalized),
                "prepared_libggml_hip_sha256": prepared_library_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _environment(
    task: OptimizationTask,
    *,
    commit: str,
    model_sha256: str,
    binary_sha256: str,
    protocol_hash: str,
    runtime_environment: dict[str, str],
    runtime_binary: Path | None = None,
) -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        values={
            "gpu_gfx": task.gpu.gfx_target,
            "gpu_device": str(task.gpu.device_id),
            "rocm_version": task.rocm.version or "unknown",
            "runtime_base_commit": commit,
            "model_sha256": model_sha256,
            "build_flags_hash": _build_flags_hash(task.runtime.build_flags),
            "binary_sha256": binary_sha256,
            "benchmark_protocol_hash": protocol_hash,
            "runtime_environment_hash": _runtime_environment_hash(
                runtime_environment,
                runtime_binary=runtime_binary,
            ),
        },
        # Do not fabricate stability when the configured observer has no telemetry.
        telemetry_stable=None,
        instability_reasons=["GPU telemetry was not integrated by this observation path"],
        capture_id=uuid.uuid4().hex,
        captured_at=utc_now(),
        source="observed",
    )


async def _inspect_mcp(task: OptimizationTask) -> tuple[dict[str, Any], list[MCPToolCall]]:
    domain = MCPServerConfig.from_domain(task.mcp)
    runtime_environment, unset_environment = _runtime_environment(task)
    server_environment = dict(domain.env)
    for name in unset_environment:
        server_environment.pop(name, None)
    server_environment.update(runtime_environment)
    config = MCPServerConfig(
        command=domain.command,
        args=domain.args,
        env=server_environment,
        cwd=domain.cwd,
    )
    async with RocmIssueAgentClient(config) as client:
        connection = client.connection_evidence
        if connection is None:  # defensive invariant
            raise LiveWorkflowError("ROCm MCP initialized without connection evidence")
        calls = list(await client.inspect_environment())
        return connection.to_dict(), calls


def _call_problem(call: MCPToolCall) -> str | None:
    return call.contract_error


def _observed_architecture_evidence(
    task: OptimizationTask,
    calls: list[MCPToolCall],
) -> tuple[dict[str, Any], list[str]]:
    """Validate the selected logical device against hash-bound MCP evidence."""

    hip_call = next(
        (call for call in calls if call.tool_name == "rocm_hip_capabilities"),
        None,
    )
    evidence: dict[str, Any] = {
        "source_tool": "rocm_hip_capabilities",
        "configured_gfx_target": task.gpu.gfx_target,
        "host_device_id": task.gpu.device_id,
        "visible_logical_device_id": 0,
        "raw_targets": None,
        "observed_gfx_targets": [],
        "matches": False,
    }
    if hip_call is None:
        return evidence, ["ROCm inspection did not return rocm_hip_capabilities"]
    if hip_call.contract_error is not None:
        return evidence, []
    structured = hip_call.structured_content
    if not isinstance(structured, dict):  # contract_error guards this defensively
        return evidence, ["ROCm HIP capabilities returned no structured object"]
    raw_targets = structured.get("detected_gfx_targets")
    evidence["raw_targets"] = raw_targets
    if not isinstance(raw_targets, list) or not raw_targets:
        return evidence, ["ROCm HIP capabilities did not identify the selected device architecture"]
    observed: set[str] = set()
    for raw_target in raw_targets:
        if not isinstance(raw_target, str):
            return evidence, ["ROCm HIP capabilities contains a non-string GFX target"]
        try:
            # ROCm may append feature qualifiers (for example xnack) to the ISA
            # identity. Build/task coordinates intentionally use the bare target.
            observed.add(canonical_gfx_target(raw_target.split(":", 1)[0]))
        except ValueError:
            return evidence, [
                f"ROCm HIP capabilities contains an invalid GFX target: {raw_target!r}"
            ]
    normalized = sorted(observed)
    evidence["observed_gfx_targets"] = normalized
    if len(normalized) != 1:
        return evidence, [
            "ROCm HIP capabilities must report exactly one architecture after GPU "
            f"visibility isolation; observed {normalized}"
        ]
    evidence["matches"] = normalized[0] == task.gpu.gfx_target
    if not evidence["matches"]:
        return evidence, [
            "observed GPU architecture does not match the configured task: "
            f"{normalized[0]} != {task.gpu.gfx_target}"
        ]
    return evidence, []


def inspect_target(
    task: OptimizationTask,
    record: WorkflowRecord,
    store: ExperimentStore,
    *,
    llama: LlamaCppAdapter | None = None,
) -> WorkflowRecord:
    """Inspect llama.cpp/model identity and read-only ROCm capabilities."""

    if record.current_stage != WorkflowStage.INSPECT_TARGET:
        raise LiveWorkflowError(f"inspect_target called at {record.current_stage}")
    preflight_result = preflight(task, store)
    adapter = llama or LlamaCppAdapter()
    local = adapter.inspect_target(
        _prepared_source(task),
        task.model.path,
        expected_commit=task.runtime.base_commit,
        expected_model_sha256=task.model.sha256,
    )
    sidecar: dict[str, Any] | None = None
    if task.model.sidecar_path is not None:
        path = task.model.sidecar_path.resolve()
        if not path.is_file():
            raise LiveWorkflowError(f"sidecar does not exist: {path}")
        from .llama_cpp import sha256_file

        digest = sha256_file(path)
        sidecar = {
            "path": str(path),
            "sha256": digest,
            "expected_sha256_matches": (
                digest == task.model.sidecar_sha256
                if task.model.sidecar_sha256 is not None
                else None
            ),
        }

    connection, calls = asyncio.run(_inspect_mcp(task))
    mcp_calls = {call.tool_name: call.to_dict() for call in calls}
    problems = [problem for call in calls if (problem := _call_problem(call))]
    observed_architecture, architecture_problems = _observed_architecture_evidence(task, calls)
    problems.extend(architecture_problems)
    if local.expected_commit_matches is False:
        problems.append("llama.cpp commit does not match the task")
    if local.expected_model_sha256_matches is False:
        problems.append("GGUF SHA-256 does not match the task")
    if sidecar and sidecar["expected_sha256_matches"] is False:
        problems.append("sidecar SHA-256 does not match the task")

    payload = {
        "schema_version": 1,
        "target": local.to_dict(),
        "sidecar": sidecar,
        "mcp_connection": connection,
        "mcp_calls": mcp_calls,
        "observed_architecture": observed_architecture,
        "problems": problems,
        "preflight": preflight_result,
    }
    artifact = store.save_json(
        task.id, "artifacts/inspection.json", payload, producer="live-inspection"
    )
    # Preserve each complete MCP envelope independently for contract audits.
    for call in calls:
        store.save_json(
            task.id,
            f"artifacts/mcp/inspect/{call.tool_name}.json",
            call.to_dict(),
            producer="rocm-issue-agent-mcp",
        )
    if problems:
        store.append_event(task.id, "inspection_inconclusive", {"problems": problems})
        raise LiveWorkflowError("; ".join(problems))
    return _save_stage(store, record, {"inspection": artifact.path})


def _baseline_metric(run: LlamaBenchmarkRun, generation_tokens: int) -> MetricSeries:
    if run.benchmark is None:
        raise LiveWorkflowError("llama-bench produced no parsed benchmark")
    test_id = f"tg{generation_tokens}"
    record = run.benchmark.by_test_id().get(test_id)
    if record is None:
        available = ", ".join(sorted(run.benchmark.by_test_id()))
        raise LiveWorkflowError(f"baseline missing {test_id}; available records: {available}")
    return MetricSeries(unit="tokens/s", samples=list(record.samples_tokens_per_second))


def _run_exact_benchmark(
    adapter: LlamaCppAdapter,
    argv: tuple[str, ...],
    *,
    cwd: Path,
    environment: dict[str, str],
    unset_environment: tuple[str, ...],
    timeout_seconds: float,
    artifact_dir: Path,
) -> LlamaBenchmarkRun:
    command = adapter.commands.run(
        argv,
        cwd=cwd,
        env=environment,
        unset_env=unset_environment,
        timeout_seconds=timeout_seconds,
        stdout_path=artifact_dir / "stdout.log",
        stderr_path=artifact_dir / "stderr.log",
    )
    benchmark = None
    if command.succeeded:
        try:
            benchmark = parse_llama_bench_json(command.stdout)
        except (json.JSONDecodeError, LlamaCppError):
            benchmark = None
    return LlamaBenchmarkRun(command=command, benchmark=benchmark)


def q8_offload_evidence(
    run: LlamaBenchmarkRun | LlamaBenchmarkResult,
    *,
    expected_quantization: str = "Q8_0",
) -> dict[str, Any]:
    """Prove a quantized-model smoke run stayed on the selected ROCm device."""

    benchmark = run.benchmark if isinstance(run, LlamaBenchmarkRun) else run
    rows = list(benchmark.raw) if benchmark is not None else []
    row = rows[0] if len(rows) == 1 else None
    checks = {
        "single_result": row is not None,
        "quantized_model_type": bool(
            row and expected_quantization in str(row.get("model_type", ""))
        ),
        "rocm_backend": bool(row and "ROCm" in str(row.get("backends", ""))),
        "device_rocm0": bool(row and row.get("devices") == "ROCm0"),
        "main_gpu_zero": bool(row and row.get("main_gpu") == 0),
        "all_gpu_layers_requested": bool(row and row.get("n_gpu_layers") == 999),
        "kv_offload_enabled": bool(row and row.get("no_kv_offload") is False),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "observed": row,
    }


def capture_baseline(
    task: OptimizationTask,
    record: WorkflowRecord,
    store: ExperimentStore,
    *,
    llama: LlamaCppAdapter | None = None,
) -> WorkflowRecord:
    """Build llama.cpp and capture a direct, full-output decode baseline."""

    if record.current_stage != WorkflowStage.CAPTURE_BASELINE:
        raise LiveWorkflowError(f"capture_baseline called at {record.current_stage}")
    adapter = llama or LlamaCppAdapter()
    task_dir = store.task_dir(task.id)
    live_dir = task_dir / "artifacts" / "live-baseline"
    source = _prepared_source(task)
    runtime_env, baseline_unset = _runtime_environment(task)
    baseline_build_status = RunStatus.SUCCEEDED
    if task.runtime.prepared_binary_path is not None:
        binary = task.runtime.prepared_binary_path.resolve()
        if not binary.is_file():
            raise LiveWorkflowError(f"prepared llama-bench does not exist: {binary}")
        actual_binary_sha256 = sha256_file(binary)
        if (
            task.runtime.prepared_binary_sha256 is not None
            and actual_binary_sha256 != task.runtime.prepared_binary_sha256
        ):
            raise LiveWorkflowError("prepared llama-bench SHA-256 does not match task")
        store.save_json(
            task.id,
            "artifacts/live-baseline/build.json",
            {
                "status": "REUSED",
                "binary": str(binary),
                "binary_sha256": actual_binary_sha256,
                "preparation_manifest": task.metadata.get("prepared_runtime_manifest"),
            },
            producer="prepared-runtime",
        )
        baseline_build_status = RunStatus.REUSED
    else:
        build = adapter.build(
            source,
            task.runtime.build_dir,
            cmake_flags=task.runtime.build_flags,
            jobs=max(1, os.cpu_count() or 1),
            env=runtime_env,
            build_timeout_seconds=task.budgets.build_timeout_seconds,
            artifact_dir=live_dir / "build",
        )
        store.save_json(
            task.id,
            "artifacts/live-baseline/build.json",
            build.to_dict(),
            producer="llama-cpp-adapter",
        )
        if not build.succeeded:
            raise LiveWorkflowError("baseline llama.cpp build failed; see live-baseline/build.json")
        binary = Path(build.llama_bench_path).resolve()
    if not binary.is_file():
        raise LiveWorkflowError(f"built llama-bench does not exist: {binary}")
    smoke_argv = DecodeBenchmarkProtocol(
        llama_bench_path=str(binary),
        model_path=str(task.model.path),
        generation_tokens=(1,),
        prompt_tokens=(0,),
        repetitions=1,
        warmup_runs=task.benchmark.warmup_runs,
        device_id=0,
        timeout_seconds=task.benchmark.timeout_seconds,
        extra_args=_protocol_extra_args(task),
        cwd=str(source),
    ).argv
    smoke = _run_exact_benchmark(
        adapter,
        smoke_argv,
        cwd=source,
        environment=runtime_env,
        unset_environment=baseline_unset,
        timeout_seconds=task.benchmark.timeout_seconds,
        artifact_dir=live_dir / "smoke",
    )
    store.save_json(
        task.id,
        "artifacts/live-baseline/smoke.json",
        {
            "command": smoke.command.to_dict(),
            "benchmark": smoke.benchmark.to_dict() if smoke.benchmark else None,
        },
        producer="llama-cpp-adapter",
    )
    if not smoke.succeeded:
        raise LiveWorkflowError("baseline smoke test failed; see live-baseline/smoke.json")
    if task.campaign_kind == CampaignKind.LLAMA_CPP_Q8:
        offload = q8_offload_evidence(smoke)
        store.save_json(
            task.id,
            "artifacts/live-baseline/q8-offload.json",
            offload,
            producer="llama-cpp-adapter",
        )
        if offload["status"] != "passed":
            raise LiveWorkflowError(
                "Q8 baseline did not prove full ROCm0 offload; see q8-offload.json"
            )

    baseline_argv = benchmark_protocol_argv(task, binary)
    baseline_run = _run_exact_benchmark(
        adapter,
        baseline_argv,
        cwd=source,
        environment=runtime_env,
        unset_environment=baseline_unset,
        timeout_seconds=task.benchmark.timeout_seconds,
        artifact_dir=live_dir / "benchmark",
    )
    store.save_json(
        task.id,
        "artifacts/live-baseline/benchmark.json",
        {
            "command": baseline_run.command.to_dict(),
            "benchmark": (baseline_run.benchmark.to_dict() if baseline_run.benchmark else None),
        },
        producer="llama-cpp-adapter",
    )
    if not baseline_run.succeeded:
        raise LiveWorkflowError("baseline benchmark failed; see live-baseline/benchmark.json")

    ldd = adapter.commands.run(
        ["ldd", str(binary)],
        cwd=source,
        env=runtime_env,
        unset_env=baseline_unset,
        timeout_seconds=60,
        stdout_path=live_dir / "runtime-libraries.stdout",
        stderr_path=live_dir / "runtime-libraries.stderr",
    )
    binary_sha256 = sha256_file(binary)
    protocol_hash = _protocol_hash(task, binary)
    source_snapshot_sha256 = (
        task.runtime.source_snapshot_sha256
        or task.metadata.get("frozen_patch_sha256")
        or _text_hash(task.runtime.base_commit)
    )
    libraries_hash = _runtime_libraries_hash(task, ldd.stdout)
    environment_hash = _runtime_environment_hash(
        runtime_env,
        runtime_binary=binary,
    )
    identity = store.save_json(
        task.id,
        "artifacts/live-baseline/identity.json",
        {
            "binary": str(binary),
            "binary_sha256": binary_sha256,
            "benchmark_argv": list(baseline_argv),
            "runtime_environment": runtime_env,
            "unset_environment": list(baseline_unset),
            "benchmark_protocol_hash": protocol_hash,
            "runtime_libraries": ldd.to_dict(),
        },
        producer="live-baseline",
    )

    inspection = store.load_json(task.id, "artifacts/inspection.json")
    target = inspection["target"]
    baseline = BaselineResult(
        environment=_environment(
            task,
            commit=target["commit"],
            model_sha256=target["model_sha256"],
            binary_sha256=binary_sha256,
            protocol_hash=protocol_hash,
            runtime_environment=runtime_env,
            runtime_binary=binary,
        ),
        run_identity=RunIdentity(
            protocol_hash=protocol_hash,
            binary_sha256=binary_sha256,
            source_snapshot_sha256=source_snapshot_sha256,
            model_sha256=target["model_sha256"],
            runtime_libraries_hash=libraries_hash,
            environment_hash=environment_hash,
            sidecar_sha256=None,
            command_hashes={
                "smoke": command_request_sha256(
                    smoke_argv,
                    cwd=source,
                    env=runtime_env,
                    unset_env=baseline_unset,
                    timeout_seconds=task.benchmark.timeout_seconds,
                ),
                "e2e": command_request_sha256(
                    baseline_argv,
                    cwd=source,
                    env=runtime_env,
                    unset_env=baseline_unset,
                    timeout_seconds=task.benchmark.timeout_seconds,
                ),
            },
        ),
        build_status=baseline_build_status,
        smoke_passed=True,
        benchmark=BenchmarkResult(
            status=RunStatus.SUCCEEDED,
            metrics={
                task.objective.primary_metric: _baseline_metric(
                    baseline_run, _protocol_tokens(task)[0]
                ),
                "tokens_per_second_tg128": _baseline_metric(baseline_run, 128),
                "tokens_per_second_tg512": _baseline_metric(baseline_run, 512),
            },
            failure_reason=None,
        ),
        # Quality can be produced once and attached at QUALITY_VALIDATION. The
        # Gate will remain INCONCLUSIVE if it is still absent.
        quality=None,
        artifacts=[identity],
    )
    artifact = store.save_json(
        task.id, "artifacts/baseline.json", baseline, producer="live-baseline"
    )
    return _save_stage(store, record, {"baseline": artifact.path})


def decompose_and_seed_map(
    task: OptimizationTask,
    record: WorkflowRecord,
    store: ExperimentStore,
) -> WorkflowRecord:
    """Record the MVP's decode-only decomposition and a deliberately partial map."""

    if record.current_stage == WorkflowStage.DECOMPOSE_E2E:
        decomposition = store.save_json(
            task.id,
            "artifacts/decomposition.json",
            {
                "schema_version": 1,
                "prefill": {"status": "not_requested"},
                "decode": {
                    "status": "measured",
                    "generation_tokens": task.workload.generation_tokens,
                },
                "runtime_http_scheduler_overhead": {"status": "not_applicable"},
            },
            producer="workflow",
        )
        return _save_stage(store, record, {"decomposition": decomposition.path})
    if record.current_stage == WorkflowStage.BUILD_EXECUTION_MAP:
        execution_map = InferenceExecutionMap(task_id=task.id)
        artifact = store.save_json(
            task.id,
            "artifacts/execution-map.json",
            execution_map,
            producer="workflow",
        )
        return _save_stage(store, record, {"execution_map": artifact.path})
    raise LiveWorkflowError(f"decompose_and_seed_map called at {record.current_stage}")


def _profile_binding(
    task: OptimizationTask,
    store: ExperimentStore,
    target: str,
) -> dict[str, Any]:
    binary = (
        task.runtime.prepared_binary_path.resolve()
        if task.runtime.prepared_binary_path is not None
        else Path(task.runtime.build_dir).resolve() / "bin" / "llama-bench"
    )
    profile_cwd = _prepared_source(task)
    binary_source = "baseline"
    candidate_env: dict[str, str] = {}
    candidate_unset: list[str] = []
    experiment_id: str | None = None
    if target != "baseline":
        from .models import ExperimentSpec

        pointer = store.load_json(task.id, "state/active-experiment.json")
        if not isinstance(pointer, dict) or pointer.get("experiment_id") != target:
            raise LiveWorkflowError(f"profile target is not the active experiment: {target}")
        spec = store.load_json(task.id, str(pointer["spec_path"]), ExperimentSpec)
        experiment_id = spec.id
        candidate_env = dict(spec.change.env)
        candidate_unset = list(getattr(spec.change, "unset_env", []))
        if spec.change.kind == "source_patch":
            active = store.load_json(task.id, "state/active-execution.json")
            if not isinstance(active, dict) or active.get("experiment_id") != target:
                raise LiveWorkflowError(
                    f"source-patch profile has no matching active execution: {target}"
                )
            runner_result = store.load_json(
                task.id, f"experiments/{target}/runner-result.json"
            )
            binary_identity = (
                runner_result.get("binary") if isinstance(runner_result, dict) else None
            )
            if not isinstance(binary_identity, dict):
                raise LiveWorkflowError(
                    f"source-patch profile has no candidate binary identity: {target}"
                )
            candidate_path = binary_identity.get("path")
            candidate_sha256 = binary_identity.get("sha256")
            if not isinstance(candidate_path, str) or not isinstance(candidate_sha256, str):
                raise LiveWorkflowError(
                    f"source-patch profile has an invalid candidate binary identity: {target}"
                )
            binary = Path(candidate_path).resolve()
            if not binary.is_file() or sha256_file(binary) != candidate_sha256:
                raise LiveWorkflowError(
                    f"source-patch candidate binary is missing or changed: {binary}"
                )
            execution_root = runner_result.get("execution_root")
            if not isinstance(execution_root, str) or not Path(execution_root).resolve().is_dir():
                raise LiveWorkflowError(
                    f"source-patch profile has an invalid execution root: {target}"
                )
            profile_cwd = Path(execution_root).resolve()
            if not binary.is_relative_to(profile_cwd):
                raise LiveWorkflowError(
                    "source-patch candidate binary is outside its execution root: "
                    f"{binary}"
                )
            binary_source = "candidate_runner"
    candidate_model_path: Path | None = None
    candidate_model_sha256: str | None = None
    if target != "baseline" and spec.change.candidate_model_path is not None:
        candidate_model_path = spec.change.candidate_model_path.resolve()
        candidate_model_sha256 = spec.change.candidate_model_sha256
        if not candidate_model_path.is_file():
            raise LiveWorkflowError(
                f"candidate model is missing before profiling: {candidate_model_path}"
            )
        if sha256_file(candidate_model_path) != candidate_model_sha256:
            raise LiveWorkflowError("candidate model SHA-256 changed before profiling")
    e2e_protocol = _decode_protocol(task, binary, model_path=candidate_model_path)
    profile_protocol = KernelTimingProfileProtocol.from_e2e(e2e_protocol)
    command = profile_protocol.argv
    runtime_env, unset = _runtime_environment(
        task,
        candidate_env=candidate_env,
        candidate_unset=candidate_unset,
        runtime_binary=binary if binary_source == "candidate_runner" else None,
    )
    arguments = RocmIssueAgentClient.profile_arguments(
        command,
        preset="kernel-timing",
        cwd=profile_cwd,
        timeout_seconds=task.mcp.timeout_seconds,
        **_profile_capture_limits(task),
    )
    return {
        "target": target,
        "experiment_id": experiment_id,
        "arguments": arguments,
        "runtime_environment": runtime_env,
        "unset_environment": list(unset),
        "binary_sha256": sha256_file(binary) if binary.is_file() else None,
        "binary_path": str(binary),
        "binary_source": binary_source,
        "model_path": str(candidate_model_path or task.model.path.resolve()),
        "model_sha256": candidate_model_sha256 or task.model.sha256,
        "e2e_protocol_hash": e2e_protocol.protocol_hash,
        "profile_command_hash": profile_protocol.command_hash,
        "profile_protocol_hash": profile_protocol.protocol_hash,
        "profile_protocol": profile_protocol.details,
    }


async def _approved_profile(
    task: OptimizationTask,
    binding: dict[str, Any],
    approval_context: dict[str, Any],
    arguments: dict[str, Any],
    approval_hash: str,
) -> tuple[dict[str, Any], MCPToolCall, MCPToolCall | None]:
    domain = MCPServerConfig.from_domain(task.mcp)
    server_env = dict(domain.env)
    for name in binding["unset_environment"]:
        server_env.pop(name, None)
    server_env.update(binding["runtime_environment"])
    config = MCPServerConfig(
        command=domain.command,
        args=domain.args,
        env=server_env,
        cwd=domain.cwd,
    )
    async with RocmIssueAgentClient(
        config,
        approval_context=approval_context,
    ) as client:
        connection = client.connection_evidence
        if connection is None:
            raise LiveWorkflowError("ROCm MCP initialized without connection evidence")
        profile = await client.call_tool(
            "rocm_profile_workload", arguments, approval_sha256=approval_hash
        )
        case_id = None
        if isinstance(profile.structured_content, dict):
            value = profile.structured_content.get("case_id")
            case_id = value if isinstance(value, str) else None
        summary = await client.get_case_summary(case_id) if case_id else None
        return connection.to_dict(), profile, summary


_RETRYABLE_PROFILE_STATUSES = {"transport_failed", "evidence_inconclusive"}
_PERCENTILE_SAMPLE_LIMIT_WARNING = (
    "p50/p95 were omitted for kernels whose duration sample limit was exceeded"
)


def _profile_failure_reason(attempt: dict[str, Any]) -> str:
    transport_error = attempt.get("transport_error")
    if isinstance(transport_error, str) and transport_error:
        return transport_error
    problems = attempt.get("problems")
    if isinstance(problems, list) and problems:
        return "; ".join(str(problem) for problem in problems)
    return f"profile attempt ended with status {attempt.get('status')!r}"


def _record_profile_failure(profile_state: dict[str, Any], attempt: dict[str, Any]) -> None:
    """Preserve the first failure and an append-only per-attempt history."""

    reason = _profile_failure_reason(attempt)
    attempt.setdefault("failure_reason", reason)
    raw_history = profile_state.setdefault("failure_history", [])
    if not isinstance(raw_history, list):
        raise LiveWorkflowError("invalid profile failure history")
    attempt_id = str(attempt.get("attempt_id"))
    if not any(
        isinstance(entry, dict) and entry.get("attempt_id") == attempt_id for entry in raw_history
    ):
        entry: dict[str, Any] = {
            "attempt_id": attempt_id,
            "status": attempt.get("status"),
            "reason": reason,
        }
        for key in ("request_id", "evidence", "problems", "transport_error"):
            if key in attempt:
                entry[key] = attempt[key]
        raw_history.append(entry)
    profile_state.setdefault("first_failure_reason", reason)
    profile_state.setdefault("first_failed_attempt_id", attempt_id)


def _migrate_profile_failure_history(profile_state: dict[str, Any]) -> None:
    """Backfill failure history for profile state written by older drivers."""

    attempts = profile_state.get("attempts")
    if not isinstance(attempts, list):
        raise LiveWorkflowError("invalid profile attempts")
    for attempt in attempts:
        if isinstance(attempt, dict) and attempt.get("status") in _RETRYABLE_PROFILE_STATUSES:
            _record_profile_failure(profile_state, attempt)
    profile_state["schema_version"] = 2


def _create_profile_attempt(
    task: OptimizationTask,
    store: ExperimentStore,
    *,
    target: str,
    binding: dict[str, Any],
    attempts: list[Any],
) -> dict[str, Any]:
    ordinal = len(attempts) + 1
    attempt_id = f"{target}-attempt-{ordinal:04d}"
    approval_context = {
        "attempt_id": attempt_id,
        "target": target,
        "runtime_environment": binding["runtime_environment"],
        "unset_environment": binding["unset_environment"],
        "binary_sha256": binding["binary_sha256"],
        "e2e_protocol_hash": binding["e2e_protocol_hash"],
        "profile_command_hash": binding["profile_command_hash"],
        "profile_protocol_hash": binding["profile_protocol_hash"],
        "profile_protocol": binding["profile_protocol"],
    }
    client = RocmIssueAgentClient(
        MCPServerConfig.from_domain(task.mcp),
        approval_context=approval_context,
    )
    approval = client.make_approval_request("rocm_profile_workload", binding["arguments"])
    request_id = f"profile-{attempt_id}-{approval.request_sha256[:12]}"
    request = ApprovalRequest(
        id=request_id,
        task_id=task.id,
        tool="rocm_profile_workload",
        arguments=binding["arguments"],
        request_hash=approval.request_sha256,
    )
    context_artifact = store.save_json(
        task.id,
        f"state/approvals/{request_id}.context.json",
        approval_context,
        producer="workflow",
    )
    store.save_json(
        task.id,
        f"state/approvals/{request_id}.request.json",
        request,
        producer="workflow",
    )
    attempt = {
        "attempt_id": attempt_id,
        "request_id": request_id,
        "request_hash": request.request_hash,
        "context": context_artifact.path,
        "binding": binding,
        "status": "awaiting_approval",
    }
    attempts.append(attempt)
    store.append_event(
        task.id,
        "mcp_approval_requested",
        {
            "attempt_id": attempt_id,
            "target": target,
            "request_id": request_id,
            "request_hash": request.request_hash,
        },
    )
    return attempt


def _stored_profile_calls(
    store: ExperimentStore,
    task_id: str,
    evidence_path: str,
) -> tuple[MCPToolCall, MCPToolCall | None]:
    """Reconstruct immutable MCP envelopes for deterministic Gate re-evaluation."""

    payload = store.load_json(task_id, evidence_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("profile"), dict):
        raise LiveWorkflowError(f"stored profile evidence is malformed: {evidence_path}")
    try:
        profile = MCPToolCall(**payload["profile"])
        raw_summary = payload.get("case_summary")
        summary = MCPToolCall(**raw_summary) if isinstance(raw_summary, dict) else None
    except TypeError as error:
        raise LiveWorkflowError(
            f"stored MCP envelope is malformed: {evidence_path}"
        ) from error
    return profile, summary


def _reclassify_stored_profile_attempt(
    task: OptimizationTask,
    store: ExperimentStore,
    *,
    target: str,
    binding: dict[str, Any],
    profile_state: dict[str, Any],
) -> dict[str, Any] | None:
    """Recover evidence rejected only by an older, overly strict deterministic Gate.

    This never calls MCP. It re-evaluates the immutable raw MCP envelope, and it
    only supersedes later approval requests that have neither a receipt nor an
    execution attempt.
    """

    attempts = profile_state.get("attempts")
    if not isinstance(attempts, list):
        raise LiveWorkflowError(f"invalid profile state for {target}")
    for index in range(len(attempts) - 1, -1, -1):
        attempt = attempts[index]
        if not isinstance(attempt, dict):
            continue
        if attempt.get("status") != "evidence_inconclusive":
            continue
        if attempt.get("binding") != binding:
            continue
        evidence_path = attempt.get("evidence")
        if not isinstance(evidence_path, str) or not evidence_path:
            continue
        try:
            profile, summary = _stored_profile_calls(store, task.id, evidence_path)
        except (LiveWorkflowError, StoreError):
            continue
        problems = _profile_problems(profile)
        if summary is None:
            problems.append("stored profile response did not contain a case summary")
        elif summary.contract_error:
            problems.append(summary.contract_error)
        if problems:
            continue

        superseded: list[str] = []
        can_recover = True
        for later in attempts[index + 1 :]:
            if not isinstance(later, dict) or later.get("status") != "awaiting_approval":
                can_recover = False
                break
            request_id = later.get("request_id")
            if not isinstance(request_id, str):
                can_recover = False
                break
            try:
                store.load_json(
                    task.id,
                    f"state/approvals/{request_id}.receipt.json",
                    ApprovalReceipt,
                )
            except StoreError:
                continue
            can_recover = False
            break
        if not can_recover:
            continue

        reclassified_at = utc_now().isoformat()
        original_problems = list(attempt.get("problems") or [])
        attempt["status"] = "succeeded"
        attempt["problems"] = []
        attempt["reclassified_at"] = reclassified_at
        attempt["reclassification_reason"] = (
            "aggregate kernel timing is complete; only bounded percentile samples "
            "are unavailable"
        )
        for entry in profile_state.get("failure_history", []):
            if isinstance(entry, dict) and entry.get("attempt_id") == attempt.get(
                "attempt_id"
            ):
                entry["reclassified_as"] = "succeeded"
                entry["reclassified_at"] = reclassified_at
                entry["original_problems"] = original_problems
        for later in attempts[index + 1 :]:
            later["status"] = "superseded_before_execution"
            later["superseded_by"] = attempt.get("attempt_id")
            later["superseded_at"] = reclassified_at
            superseded.append(str(later.get("attempt_id")))
        store.append_event(
            task.id,
            "profiling_evidence_reclassified",
            {
                "attempt_id": attempt.get("attempt_id"),
                "target": target,
                "evidence": evidence_path,
                "original_problems": original_problems,
                "superseded_attempts": superseded,
            },
        )
        return attempt
    return None


def _approval_pause(
    task: OptimizationTask,
    store: ExperimentStore,
    latest: dict[str, Any],
    *,
    paused_for: str = "approval",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_id = str(latest["request_id"])
    request = store.load_json(
        task.id,
        f"state/approvals/{request_id}.request.json",
        ApprovalRequest,
    )
    result = {
        "paused_for": paused_for,
        "target": latest["binding"]["target"],
        "attempt_id": latest["attempt_id"],
        "request_id": request.id,
        "request_hash": request.request_hash,
        "tool": request.tool,
        "arguments": request.arguments,
        "execution_context": store.load_json(task.id, str(latest["context"])),
    }
    result.update(extra or {})
    return result


def _profile_target_until_pause(
    task: OptimizationTask,
    store: ExperimentStore,
    *,
    target: str,
) -> dict[str, Any]:
    """Request/consume one exact, persistent Level-2 profile attempt."""

    binding = _profile_binding(task, store, target)
    imported = _stored_raw_profile(task, store, target=target, binding=binding)
    if imported is not None:
        return imported
    if task.metadata.get("raw_rocprof_authorized") == "true":
        return _raw_profile_target(task, store, target=target, binding=binding)
    profile_state_path = f"state/profiles/{target}.json"
    try:
        profile_state = store.load_json(task.id, profile_state_path)
    except StoreError:
        profile_state = {"schema_version": 2, "target": target, "attempts": []}
    if not isinstance(profile_state, dict):
        raise LiveWorkflowError(f"invalid profile state for {target}")
    _migrate_profile_failure_history(profile_state)
    attempts = profile_state.get("attempts")
    if not isinstance(attempts, list):
        raise LiveWorkflowError(f"invalid profile state for {target}")
    reclassified = _reclassify_stored_profile_attempt(
        task,
        store,
        target=target,
        binding=binding,
        profile_state=profile_state,
    )
    if reclassified is not None:
        store.save_json(task.id, profile_state_path, profile_state, producer="workflow")
    succeeded = next(
        (
            attempt
            for attempt in reversed(attempts)
            if isinstance(attempt, dict) and attempt.get("status") == "succeeded"
        ),
        None,
    )
    if succeeded is not None:
        return {
            "profile_complete": True,
            "target": target,
            "attempt_id": succeeded["attempt_id"],
            "kernel_evidence": succeeded["evidence"],
            "case_ownership": succeeded.get("case_ownership"),
        }
    latest = attempts[-1] if attempts else None
    create_attempt = latest is None or (
        isinstance(latest, dict) and latest.get("status") in _RETRYABLE_PROFILE_STATUSES
    )
    if create_attempt:
        latest = _create_profile_attempt(
            task,
            store,
            target=target,
            binding=binding,
            attempts=attempts,
        )
        store.save_json(task.id, profile_state_path, profile_state, producer="workflow")
    if not isinstance(latest, dict):
        raise LiveWorkflowError(f"invalid latest profile attempt for {target}")
    request_id = str(latest["request_id"])
    request = store.load_json(
        task.id,
        f"state/approvals/{request_id}.request.json",
        ApprovalRequest,
    )
    receipt_path = f"state/approvals/{request_id}.receipt.json"
    try:
        receipt = store.load_json(task.id, receipt_path, ApprovalReceipt)
    except StoreError:
        return _approval_pause(task, store, latest)
    if receipt.request_hash != request.request_hash:
        raise LiveWorkflowError("approval receipt hash does not match the exact MCP request")
    if receipt.consumed_at is not None:
        raise LiveWorkflowError("approval receipt was already consumed")

    from .models import utc_now

    receipt.consumed_at = utc_now()
    store.save_json(task.id, receipt_path, receipt, producer="workflow")
    latest["status"] = "transport_started"
    store.save_json(task.id, profile_state_path, profile_state, producer="workflow")
    store.append_event(
        task.id,
        "mcp_approval_consumed",
        {
            "attempt_id": latest["attempt_id"],
            "request_id": request.id,
            "request_hash": request.request_hash,
        },
    )
    approval_context = store.load_json(task.id, str(latest["context"]))
    attempt_binding = latest.get("binding")
    if not isinstance(attempt_binding, dict):
        raise LiveWorkflowError("profile attempt is missing its immutable binding")
    try:
        connection, profile, summary = asyncio.run(
            _approved_profile(
                task,
                attempt_binding,
                approval_context,
                request.arguments,
                request.request_hash,
            )
        )
    except Exception as error:
        latest["status"] = "transport_failed"
        latest["transport_error"] = f"{type(error).__name__}: {error}"
        _record_profile_failure(profile_state, latest)
        store.append_event(
            task.id,
            "mcp_profile_transport_failed",
            {
                "attempt_id": latest["attempt_id"],
                "request_id": request.id,
                "error": latest["transport_error"],
            },
        )
        failed_attempt = latest
        latest = _create_profile_attempt(
            task,
            store,
            target=target,
            binding=binding,
            attempts=attempts,
        )
        store.save_json(task.id, profile_state_path, profile_state, producer="workflow")
        return _approval_pause(
            task,
            store,
            latest,
            paused_for="new_approval_after_transport_failure",
            extra={
                "failed_attempt_id": failed_attempt["attempt_id"],
                "failure_reason": failed_attempt["failure_reason"],
            },
        )
    case_id = None
    if isinstance(profile.structured_content, dict):
        raw_case_id = profile.structured_content.get("case_id")
        case_id = raw_case_id if isinstance(raw_case_id, str) else None
    ownership = {
        "case_id": case_id,
        "owner": "rocm-issue-agent",
        "case_store_root": task.mcp.env.get("ROCM_AGENT_HOME"),
        "raw_artifacts_managed_by": "rocm-issue-agent-case-store",
    }
    attempt_root = f"artifacts/mcp/profiles/{target}/{latest['attempt_id']}"
    payload = {
        "schema_version": 1,
        "attempt_id": latest["attempt_id"],
        "target": target,
        "binding": attempt_binding,
        "approval_context": approval_context,
        "connection": connection,
        "profile": profile.to_dict(),
        "case_summary": summary.to_dict() if summary else None,
        "case_ownership": ownership,
    }
    artifact = store.save_json(
        task.id,
        f"{attempt_root}/evidence.json",
        payload,
        producer="rocm-issue-agent-mcp",
    )
    problems = _profile_problems(profile)
    if summary and summary.contract_error:
        problems.append(summary.contract_error)
    if summary is None:
        problems.append("profile response did not contain a case_id")
    latest["status"] = "succeeded" if not problems else "evidence_inconclusive"
    latest["evidence"] = artifact.path
    latest["case_ownership"] = ownership
    latest["problems"] = problems
    store.save_json(task.id, profile_state_path, profile_state, producer="workflow")
    if problems:
        _record_profile_failure(profile_state, latest)
        failed_attempt = latest
        store.append_event(
            task.id,
            "profiling_inconclusive",
            {
                "attempt_id": failed_attempt["attempt_id"],
                "target": target,
                "problems": problems,
            },
        )
        latest = _create_profile_attempt(
            task,
            store,
            target=target,
            binding=binding,
            attempts=attempts,
        )
        store.save_json(task.id, profile_state_path, profile_state, producer="workflow")
        return _approval_pause(
            task,
            store,
            latest,
            paused_for="new_approval_after_inconclusive_evidence",
            extra={
                "failed_attempt_id": failed_attempt["attempt_id"],
                "failure_reason": failed_attempt["failure_reason"],
                "problems": problems,
                "kernel_evidence": artifact.path,
            },
        )
    return {
        "profile_complete": True,
        "target": target,
        "attempt_id": latest["attempt_id"],
        "kernel_evidence": artifact.path,
        "case_ownership": ownership,
    }


def _stored_raw_profile(
    task: OptimizationTask,
    store: ExperimentStore,
    *,
    target: str,
    binding: dict[str, Any],
) -> dict[str, Any] | None:
    """Reuse hash-bound raw evidence, including explicitly finalized imports."""

    state_path = f"state/raw-profiles/{target}.json"
    try:
        state = store.load_json(task.id, state_path)
    except StoreError:
        return None
    attempts = state.get("attempts") if isinstance(state, dict) else None
    if not isinstance(attempts, list):
        raise LiveWorkflowError(f"invalid raw profile state for {target}")
    binding_hash = _text_hash(
        json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    previous = next(
        (
            attempt
            for attempt in reversed(attempts)
            if isinstance(attempt, dict)
            and attempt.get("binding_hash") == binding_hash
            and attempt.get("status") in {"completed", "partial", "failed", "timeout"}
        ),
        None,
    )
    if previous is None:
        return None
    evidence_path = previous.get("evidence")
    artifact = (
        store.artifact_ref(task.id, evidence_path)
        if isinstance(evidence_path, str)
        else None
    )
    if artifact is None or not store.verify_artifact(task.id, artifact):
        return None
    return {
        "profile_complete": True,
        "target": target,
        "attempt_id": previous["attempt_id"],
        "kernel_evidence": artifact.path,
        "evidence_status": previous["status"],
        "profile_backend": "raw_rocprofv3",
        "authorization": previous.get("authorization", "explicit raw profile evidence"),
        "reused": True,
    }


def _raw_profile_target(
    task: OptimizationTask,
    store: ExperimentStore,
    *,
    target: str,
    binding: dict[str, Any],
) -> dict[str, Any]:
    """Execute or reuse one explicitly authorized raw rocprofv3 fallback."""

    from .raw_rocprof import RawRocprofAdapter

    state_path = f"state/raw-profiles/{target}.json"
    binding_hash = _text_hash(
        json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    reusable = _stored_raw_profile(task, store, target=target, binding=binding)
    if reusable is not None:
        return reusable
    try:
        state = store.load_json(task.id, state_path)
    except StoreError:
        state = {"schema_version": 1, "target": target, "attempts": []}
    attempts = state.get("attempts") if isinstance(state, dict) else None
    if not isinstance(attempts, list):
        raise LiveWorkflowError(f"invalid raw profile state for {target}")
    attempt_id = f"raw-{target}-attempt-{len(attempts) + 1:04d}"
    attempt_root = f"artifacts/raw-rocprof/profiles/{target}/{attempt_id}"
    output_dir = store.task_dir(task.id) / attempt_root / "capture"
    attempt: dict[str, Any] = {
        "attempt_id": attempt_id,
        "binding_hash": binding_hash,
        "binding": binding,
        "status": "started",
        "authorization": "task.metadata.raw_rocprof_authorized=true",
    }
    attempts.append(attempt)
    store.save_json(task.id, state_path, state, producer="workflow")
    store.append_event(
        task.id,
        "raw_profile_started",
        {
            "attempt_id": attempt_id,
            "target": target,
            "binding_hash": binding_hash,
            "authorization": attempt["authorization"],
        },
    )
    arguments = binding["arguments"]
    adapter = RawRocprofAdapter(
        rocprofv3_path=task.metadata.get("rocprofv3_path", "/usr/bin/rocprofv3")
    )
    try:
        result = adapter.profile(
            arguments["command"],
            cwd=arguments["cwd"],
            environment=binding["runtime_environment"],
            unset_environment=binding["unset_environment"],
            output_dir=output_dir,
            timeout_seconds=arguments["timeout_seconds"],
            max_trace_bytes=arguments["max_trace_bytes"],
            max_trace_files=arguments["max_trace_files"],
            max_events_per_type=arguments["max_events_per_type"],
            max_percentile_samples_per_kernel=arguments[
                "max_percentile_samples_per_kernel"
            ],
        )
        payload = result.to_dict()
        status = str(result.kernel_evidence["status"])
    except Exception as error:
        # The fallback is terminal evidence for this exact binding. Persisting the
        # failure lets a performance-negative experiment finish without inventing
        # an MCP receipt, while the Gate/Agent still sees that no timing claim exists.
        status = "failed"
        payload = {
            "schema": "gpuopt.raw-rocprof-result.v1",
            "source": "raw_rocprofv3",
            "authorization": {
                "mode": "task_configuration",
                "field": "metadata.raw_rocprof_authorized",
                "value": "true",
            },
            "binding": binding,
            "error": f"{type(error).__name__}: {error}",
            "kernel_evidence": {
                "schema": "gpuopt.raw-kernel-evidence.v1",
                "status": "failed",
                "kernels": [],
                "aggregate_timing_complete": False,
                "hotspot_ranking_reliable": False,
                "coverage_percent": 0.0,
                "warnings": ["raw rocprofv3 fallback failed"],
            },
        }
    artifact = store.save_json(
        task.id,
        f"{attempt_root}/evidence.json",
        payload,
        producer="raw-rocprofv3",
    )
    attempt["status"] = status
    attempt["evidence"] = artifact.path
    store.save_json(task.id, state_path, state, producer="workflow")
    store.append_event(
        task.id,
        "raw_profile_completed",
        {
            "attempt_id": attempt_id,
            "target": target,
            "status": status,
            "evidence": artifact.path,
            "mcp_approval_created": False,
        },
    )
    return {
        "profile_complete": True,
        "target": target,
        "attempt_id": attempt_id,
        "kernel_evidence": artifact.path,
        "evidence_status": status,
        "profile_backend": "raw_rocprofv3",
        "authorization": attempt["authorization"],
        "reused": False,
    }


def finalize_existing_raw_profile(
    task: OptimizationTask,
    store: ExperimentStore,
    *,
    target: str,
    trace_root: str | Path,
    profiler_argv: list[str] | tuple[str, ...],
    normalization_max_events: int,
    authorization_reference: str,
    workload_argv: list[str] | tuple[str, ...] | None = None,
    environment: dict[str, str] | None = None,
    workload_exit_code: int = 0,
) -> dict[str, Any]:
    """Normalize and bind an existing trace so live resume performs no new execution."""

    from .raw_rocprof import normalize_existing_trace

    binding = _profile_binding(task, store, target)
    binding_hash = _text_hash(
        json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    reusable = _stored_raw_profile(task, store, target=target, binding=binding)
    if reusable is not None:
        return reusable
    arguments = binding["arguments"]
    captured_workload = tuple(workload_argv or arguments["command"])
    if captured_workload != tuple(arguments["command"]):
        raise LiveWorkflowError(
            "existing raw trace workload differs from the workflow profile binding"
        )
    captured_environment = dict(environment or binding["runtime_environment"])
    capture_budget = {
        name: arguments[name]
        for name in (
            "max_trace_bytes",
            "max_trace_files",
            "max_events_per_type",
            "max_percentile_samples_per_kernel",
        )
    }
    payload = normalize_existing_trace(
        trace_root,
        profiler_argv=profiler_argv,
        workload_argv=captured_workload,
        cwd=arguments["cwd"],
        environment=captured_environment,
        unset_environment=binding["unset_environment"],
        capture_budget=capture_budget,
        normalization_max_events=normalization_max_events,
        workload_exit_code=workload_exit_code,
        authorization_reference=authorization_reference,
    )
    payload["workflow_binding"] = binding
    evidence = payload["kernel_evidence"]
    status = str(evidence["status"])
    state_path = f"state/raw-profiles/{target}.json"
    try:
        state = store.load_json(task.id, state_path)
    except StoreError:
        state = {"schema_version": 1, "target": target, "attempts": []}
    attempts = state.get("attempts") if isinstance(state, dict) else None
    if not isinstance(attempts, list):
        raise LiveWorkflowError(f"invalid raw profile state for {target}")
    attempt_id = f"raw-import-{target}-attempt-{len(attempts) + 1:04d}"
    artifact = store.save_json(
        task.id,
        f"artifacts/raw-rocprof/profiles/{target}/{attempt_id}/evidence.json",
        payload,
        producer="raw-rocprofv3-import",
    )
    attempts.append(
        {
            "attempt_id": attempt_id,
            "binding_hash": binding_hash,
            "binding": binding,
            "status": status,
            "evidence": artifact.path,
            "authorization": authorization_reference,
            "capture_origin": "existing_trace",
        }
    )
    store.save_json(task.id, state_path, state, producer="workflow")
    store.append_event(
        task.id,
        "raw_profile_imported",
        {
            "attempt_id": attempt_id,
            "target": target,
            "status": status,
            "evidence": artifact.path,
            "trace_root": str(Path(trace_root).resolve()),
            "normalization_max_events": normalization_max_events,
            "mcp_approval_created": False,
            "authorization_reference": authorization_reference,
        },
    )
    return {
        "profile_complete": True,
        "target": target,
        "attempt_id": attempt_id,
        "kernel_evidence": artifact.path,
        "evidence_status": status,
        "profile_backend": "raw_rocprofv3",
        "authorization": authorization_reference,
        "reused": False,
    }


def _profile_problems(profile: MCPToolCall) -> list[str]:
    problem = profile.contract_error
    if problem:
        return [problem]
    payload = profile.structured_content
    if not isinstance(payload, dict):
        return ["profile returned no structured evidence"]
    evidence = payload.get("kernel_evidence")
    if not isinstance(evidence, dict):
        return ["profile response is missing kernel_evidence"]
    problems: list[str] = []
    status = evidence.get("status")
    if status not in {"completed", "partial"}:
        problems.append(f"kernel evidence status is {status!r}")
    preset = evidence.get("preset")
    if preset != "kernel-timing":
        problems.append(f"kernel evidence preset is {preset!r}")
    profiler = evidence.get("profiler")
    if not isinstance(profiler, dict):
        problems.append("kernel evidence is missing profiler status")
    elif profiler.get("status") != "completed":
        problems.append(f"profiler status is {profiler.get('status')!r}")
    workload = evidence.get("workload")
    if not isinstance(workload, dict):
        problems.append("kernel evidence is missing workload status")
    else:
        if workload.get("status") != "completed":
            problems.append(f"profiled workload status is {workload.get('status')!r}")
        if workload.get("exit_code") != 0:
            problems.append(f"profiled workload exit_code is {workload.get('exit_code')!r}")
    kernels = evidence.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        problems.append("kernel evidence contains no parsed kernels")
    warnings = evidence.get("warnings", [])
    if status == "partial":
        normalized_warnings = {
            str(item).strip() for item in warnings if str(item).strip()
        }
        percentile_only = normalized_warnings == {_PERCENTILE_SAMPLE_LIMIT_WARNING}
        aggregates_complete = isinstance(kernels, list) and bool(kernels) and all(
            _kernel_aggregate_complete(kernel) for kernel in kernels
        )
        if not percentile_only or not aggregates_complete:
            problems.append("partial kernel evidence has warnings that can bias hotspot ranking")
    return problems


def _kernel_aggregate_complete(kernel: Any) -> bool:
    if not isinstance(kernel, dict):
        return False
    count = kernel.get("dispatch_count")
    total = kernel.get("total_duration_ns")
    average = kernel.get("average_duration_ns")
    share = kernel.get("gpu_kernel_time_share_percent")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        return False
    numeric = (int, float)
    if any(
        not isinstance(value, numeric) or isinstance(value, bool)
        for value in (total, average, share)
    ):
        return False
    if total <= 0 or average <= 0 or not 0 <= share <= 100:
        return False
    expected_average = total / count
    if abs(float(average) - expected_average) > max(1e-6, expected_average * 1e-9):
        return False
    if kernel.get("percentiles_reliable") is False and (
        kernel.get("p50_duration_ns") is not None
        or kernel.get("p95_duration_ns") is not None
    ):
        return False
    return True


def _available_evidence(store: ExperimentStore, task_id: str) -> list[EvidenceRef]:
    try:
        manifest = store.load_json(task_id, "artifacts/manifest.json")
    except StoreError:
        return []
    evidence: list[EvidenceRef] = []
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


def discover_hotspots(
    task: OptimizationTask,
    record: WorkflowRecord,
    store: ExperimentStore,
) -> tuple[WorkflowRecord, dict[str, Any]]:
    """Create an exact approval request or consume it once to collect timing evidence."""

    if record.current_stage != WorkflowStage.DISCOVER_HOTSPOTS:
        raise LiveWorkflowError(f"discover_hotspots called at {record.current_stage}")
    try:
        requested_profile = store.load_json(task.id, "state/requested-profile.json")
    except StoreError:
        requested_profile = None
    if (
        isinstance(requested_profile, dict)
        and requested_profile.get("fulfilled") is False
        and requested_profile.get("requested_profile_level") not in {None, "LEVEL_2"}
    ):
        level = requested_profile.get("requested_profile_level")
        artifact = store.save_json(
            task.id,
            "artifacts/profiling-unavailable.json",
            {
                "schema_version": 1,
                "requested_profile_level": level,
                "status": "unavailable",
                "reason": (
                    "The current ROCm Issue Agent integration exposes kernel timing but "
                    "does not integrate targeted counters, kernel metadata, ISA, thread "
                    "trace, or PC sampling for this workflow."
                ),
            },
            producer="workflow",
        )
        # The request is answered with explicit unavailable evidence, never by
        # silently substituting a lower profiling level.
        requested_profile["fulfilled"] = True
        requested_profile["evidence"] = artifact.path
        store.save_json(
            task.id,
            "state/requested-profile.json",
            requested_profile,
            producer="workflow",
        )
        updated = _save_stage(store, record, {"kernel_evidence": artifact.path})
        return updated, {
            "evidence_status": "unavailable",
            "problems": [f"requested profiling level {level} is not integrated"],
            "kernel_evidence": artifact.path,
        }
    target = "baseline"
    if isinstance(requested_profile, dict):
        configured_target = requested_profile.get("target")
        if isinstance(configured_target, str) and configured_target:
            target = configured_target
    detail = _profile_target_until_pause(task, store, target=target)
    if not detail.get("profile_complete"):
        return record, detail
    updated = _save_stage(
        store,
        record,
        {"kernel_evidence": str(detail["kernel_evidence"])},
    )
    if isinstance(requested_profile, dict):
        requested_profile["fulfilled"] = True
        requested_profile["evidence"] = detail["kernel_evidence"]
        store.save_json(
            task.id,
            "state/requested-profile.json",
            requested_profile,
            producer="workflow",
        )
    return updated, detail


def run_live_until_pause(task: OptimizationTask, store: ExperimentStore) -> dict[str, Any]:
    """Run deterministic live stages until Agent input, approval, or failure is needed."""

    if task.campaign_kind == CampaignKind.VLLM_MI300X:
        raise LiveWorkflowError(
            "vllm_mi300x tasks cannot enter the llama.cpp live runner; "
            "use the vLLM workflow coordinator"
        )
    record = store.load_workflow(task.id)
    completed: list[str] = []
    while True:
        stage = record.current_stage
        if stage == WorkflowStage.CREATE_TASK:
            record = _save_stage(store, record, {"task": "task.json"})
        elif stage == WorkflowStage.INSPECT_TARGET:
            record = inspect_target(task, record, store)
        elif stage == WorkflowStage.CAPTURE_BASELINE:
            record = capture_baseline(task, record, store)
        elif stage in {WorkflowStage.DECOMPOSE_E2E, WorkflowStage.BUILD_EXECUTION_MAP}:
            record = decompose_and_seed_map(task, record, store)
        elif stage == WorkflowStage.DISCOVER_HOTSPOTS:
            record, detail = discover_hotspots(task, record, store)
            if record.current_stage == WorkflowStage.DISCOVER_HOTSPOTS:
                return {
                    "task_id": task.id,
                    "stage": record.current_stage,
                    "completed_stages": completed,
                    **detail,
                }
        elif stage in {
            WorkflowStage.CLASSIFY_BOTTLENECK,
            WorkflowStage.ANALYZE_LIMIT,
            WorkflowStage.GENERATE_HYPOTHESIS,
            WorkflowStage.CREATE_EXPERIMENT,
        }:
            context = build_agent_context(task, record, _available_evidence(store, task.id))
            write_agent_context(store, context)
            return {
                "task_id": task.id,
                "stage": stage,
                "completed_stages": completed,
                "paused_for": "agent_decision",
                "agent_context": "state/agent-context.json",
            }
        elif stage == WorkflowStage.PATCH_AND_BUILD:
            from .experiment import execute_active_experiment

            result = execute_active_experiment(task, record, store)
            result["completed_stages"] = completed
            return result
        elif stage in {
            WorkflowStage.E2E_VALIDATION,
            WorkflowStage.QUALITY_VALIDATION,
        }:
            from .experiment import resume_active_experiment

            result = resume_active_experiment(task, record, store)
            result["completed_stages"] = completed
            return result
        else:
            return {
                "task_id": task.id,
                "stage": stage,
                "completed_stages": completed,
                "paused_for": "manual_recovery",
                "detail": "workflow is inside an incomplete experiment; inspect stored evidence",
            }
        completed.append(stage.value)


__all__ = [
    "LiveWorkflowError",
    "capture_baseline",
    "decompose_and_seed_map",
    "discover_hotspots",
    "finalize_existing_raw_profile",
    "inspect_target",
    "run_live_until_pause",
]
