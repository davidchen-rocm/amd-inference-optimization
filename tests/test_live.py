from __future__ import annotations

from pathlib import Path

import pytest

import amd_inference_opt.live as live_module
import amd_inference_opt.raw_rocprof as raw_module
from amd_inference_opt.command import argv_sha256
from amd_inference_opt.live import (
    LiveWorkflowError,
    _profile_binding,
    _profile_target_until_pause,
    _runtime_environment,
    _runtime_environment_hash,
    _runtime_libraries_hash,
    benchmark_protocol_argv,
    discover_hotspots,
    finalize_existing_raw_profile,
    preflight,
    profile_protocol_argv,
    q8_offload_evidence,
)
from amd_inference_opt.llama_cpp import parse_llama_bench_json
from amd_inference_opt.models import (
    ApprovalReceipt,
    CampaignKind,
    ChangeSet,
    ExperimentSpec,
    MCPConfig,
    ModelTarget,
    OptimizationTask,
    RuntimeTarget,
    WorkflowRecord,
    WorkflowStage,
)
from amd_inference_opt.rocm_mcp import MCPToolCall, approval_request
from amd_inference_opt.store import ExperimentStore


def test_explicit_raw_profile_mode_creates_no_mcp_approval(
    tmp_path: Path, monkeypatch
) -> None:
    task = OptimizationTask(
        id="raw-profile",
        campaign_kind=CampaignKind.LLAMA_CPP_Q8,
        model=ModelTarget(path=tmp_path / "model.gguf", quantization="Q8_0"),
        runtime=RuntimeTarget(
            repo_path=tmp_path / "repo",
            base_commit="abc",
            build_dir=tmp_path / "build",
        ),
        mcp=MCPConfig(command=["/unavailable/rocm-agent-mcp"]),
        metadata={"raw_rocprof_authorized": "true"},
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)

    class _Result:
        kernel_evidence = {
            "status": "completed",
            "aggregate_timing_complete": True,
            "hotspot_ranking_reliable": True,
            "coverage_percent": 100.0,
            "kernels": [{"name": "q8_kernel", "dispatch_count": 128}],
        }

        def to_dict(self):
            return {
                "schema": "gpuopt.raw-rocprof-result.v1",
                "source": "raw_rocprofv3",
                "profiler_argv": ["/usr/bin/rocprofv3"],
                "environment": {"HIP_VISIBLE_DEVICES": "0"},
                "raw_artifacts": [
                    {"path": "kernel_trace.csv", "sha256": "a" * 64, "size_bytes": 10}
                ],
                "kernel_evidence": self.kernel_evidence,
            }

    def fake_profile(self, *args, **kwargs):
        del self, args, kwargs
        return _Result()

    monkeypatch.setattr(raw_module.RawRocprofAdapter, "profile", fake_profile)

    detail = _profile_target_until_pause(task, store, target="baseline")

    assert detail["profile_complete"] is True
    assert detail["profile_backend"] == "raw_rocprofv3"
    assert detail["evidence_status"] == "completed"
    assert not (store.task_dir(task.id) / "state" / "approvals").exists()
    evidence = store.load_json(task.id, detail["kernel_evidence"])
    assert evidence["source"] == "raw_rocprofv3"
    events = (store.task_dir(task.id) / "events" / "events.jsonl").read_text()
    assert '"mcp_approval_created": false' in events


def test_existing_raw_trace_can_be_finalized_and_reused_without_task_metadata(
    tmp_path: Path,
) -> None:
    task = OptimizationTask(
        id="raw-import",
        campaign_kind=CampaignKind.LLAMA_CPP_Q8,
        model=ModelTarget(path=tmp_path / "model.gguf", quantization="Q8_0"),
        runtime=RuntimeTarget(
            repo_path=tmp_path / "repo",
            base_commit="abc",
            build_dir=tmp_path / "build",
        ),
        mcp=MCPConfig(command=["/unavailable/rocm-agent-mcp"]),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    trace_root = tmp_path / "existing-trace"
    trace_root.mkdir()
    (trace_root / "kernel_trace.csv").write_text(
        '"Kernel_Name","Start_Timestamp","End_Timestamp"\n'
        '"q8_kernel",100,200\n',
        encoding="utf-8",
    )
    workload = profile_protocol_argv(
        task, tmp_path / "build" / "bin" / "llama-bench"
    )
    profiler = [
        "/usr/bin/rocprofv3",
        "--kernel-trace",
        "--output-directory",
        str(trace_root),
        "--",
    ]

    imported = finalize_existing_raw_profile(
        task,
        store,
        target="baseline",
        trace_root=trace_root,
        profiler_argv=profiler,
        normalization_max_events=2,
        authorization_reference="user-authorized-unattended-raw-fallback",
    )
    reused = _profile_target_until_pause(task, store, target="baseline")

    assert imported["evidence_status"] == "completed"
    assert reused["attempt_id"] == imported["attempt_id"]
    assert reused["reused"] is True
    assert not (store.task_dir(task.id) / "state" / "approvals").exists()
    evidence = store.load_json(task.id, imported["kernel_evidence"])
    assert evidence["workload_argv"] == list(workload)
    assert evidence["kernel_evidence"]["capture_budget"]["max_events_per_type"] == 10_000
    assert evidence["kernel_evidence"]["normalization_budget"]["max_events"] == 2


def _preflight_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-q8")
    source = tmp_path / "llama.cpp"
    source.mkdir()
    (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n")
    mcp = tmp_path / "rocm-agent-mcp"
    mcp.write_text("#!/bin/sh\nexit 0\n")
    mcp.chmod(0o755)
    return model, source, mcp


def test_q8_preflight_does_not_require_q4_sidecar(tmp_path: Path) -> None:
    model, source, mcp = _preflight_assets(tmp_path)
    task = OptimizationTask(
        id="q8-preflight",
        campaign_kind=CampaignKind.LLAMA_CPP_Q8,
        model=ModelTarget(path=model, quantization="Q8_0"),
        runtime=RuntimeTarget(
            repo_path=source,
            base_commit="abc",
            build_flags=["-DGGML_HIP=ON", "-DAMDGPU_TARGETS=gfx1201"],
            build_dir=tmp_path / "build",
        ),
        mcp=MCPConfig(command=[str(mcp)]),
        metadata={"minimum_free_disk_gib": "0"},
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)

    result = preflight(task, store)

    assert result["status"] == "ready"
    assert result["campaign_kind"] == "llama_cpp_q8"
    assert result["sidecar"] is None


def test_gfx942_preflight_accepts_matching_registered_build_target(
    tmp_path: Path,
) -> None:
    model, source, mcp = _preflight_assets(tmp_path)
    task = OptimizationTask(
        id="mi300x-preflight",
        campaign_kind=CampaignKind.LLAMA_CPP_Q8,
        model=ModelTarget(path=model, quantization="Q8_0"),
        runtime=RuntimeTarget(
            repo_path=source,
            base_commit="abc",
            build_flags=["-DGGML_HIP=ON", "-DAMDGPU_TARGETS=gfx942"],
            build_dir=tmp_path / "build",
        ),
        gpu={"gfx_target": "gfx942", "device_id": 2},
        mcp=MCPConfig(command=[str(mcp)]),
        metadata={"minimum_free_disk_gib": "0"},
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)

    result = preflight(task, store)

    assert result["architecture"]["family"] == "cdna3"
    assert result["architecture"]["wavefront_size"] == 64
    assert result["architecture_coordinates"] == {
        "configured_gfx_target": "gfx942",
        "cmake_gfx_target": "gfx942",
        "matches": True,
    }
    assert result["runtime_environment"]["HIP_VISIBLE_DEVICES"] == "2"
    assert result["benchmark_argv"][result["benchmark_argv"].index("-dev") + 1] == "ROCm0"
    assert result["benchmark_argv"][result["benchmark_argv"].index("-mg") + 1] == "0"


@pytest.mark.parametrize(
    ("build_flags", "message"),
    [
        (["-DGGML_HIP=ON"], "missing AMDGPU_TARGETS"),
        (["-DAMDGPU_TARGETS=gfx942;gfx1201"], "exactly one GFX target"),
        (["-DAMDGPU_TARGETS=mi300x"], "invalid GFX target"),
        (
            ["-DAMDGPU_TARGETS=gfx942", "-DAMDGPU_TARGETS=gfx1201"],
            "exactly one AMDGPU_TARGETS",
        ),
        (["-DAMDGPU_TARGETS=gfx1201"], "does not match"),
    ],
)
def test_preflight_fails_closed_for_unbound_architecture_coordinates(
    tmp_path: Path,
    build_flags: list[str],
    message: str,
) -> None:
    model, source, mcp = _preflight_assets(tmp_path)
    task = OptimizationTask(
        id="architecture-fail-closed",
        campaign_kind=CampaignKind.LLAMA_CPP_Q8,
        model=ModelTarget(path=model, quantization="Q8_0"),
        runtime=RuntimeTarget(
            repo_path=source,
            base_commit="abc",
            build_flags=build_flags,
            build_dir=tmp_path / "build",
        ),
        gpu={"gfx_target": "gfx942"},
        mcp=MCPConfig(command=[str(mcp)]),
        metadata={"minimum_free_disk_gib": "0"},
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)

    with pytest.raises(LiveWorkflowError, match=message):
        preflight(task, store)

    saved = store.load_json(task.id, "artifacts/preflight.json")
    assert saved["status"] == "failed"


def test_legacy_q4_preflight_still_requires_sidecar(tmp_path: Path) -> None:
    model, source, mcp = _preflight_assets(tmp_path)
    task = OptimizationTask(
        id="q4-preflight",
        model=ModelTarget(path=model, quantization="Q4_K_M"),
        runtime=RuntimeTarget(
            repo_path=source,
            base_commit="abc",
            build_flags=["-DGGML_HIP=ON", "-DAMDGPU_TARGETS=gfx1201"],
            build_dir=tmp_path / "build",
        ),
        mcp=MCPConfig(command=[str(mcp)]),
        metadata={"live_q4rdna": "true", "minimum_free_disk_gib": "0"},
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)

    assert task.campaign_kind == CampaignKind.Q4_RDNA
    with pytest.raises(LiveWorkflowError, match="sidecar"):
        preflight(task, store)


def test_runtime_environment_hash_ignores_compared_binary_directory(tmp_path: Path) -> None:
    baseline_binary = tmp_path / "baseline" / "bin" / "llama-bench"
    candidate_binary = tmp_path / "candidate" / "bin" / "llama-bench"
    baseline_environment = {
        "HIP_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": f"{baseline_binary.parent}:/opt/rocm/lib",
    }
    candidate_environment = {
        "HIP_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": f"{candidate_binary.parent}:/opt/rocm/lib",
    }

    assert _runtime_environment_hash(
        baseline_environment,
        runtime_binary=baseline_binary,
    ) == _runtime_environment_hash(
        candidate_environment,
        runtime_binary=candidate_binary,
    )
    changed_device = dict(candidate_environment, HIP_VISIBLE_DEVICES="1")
    assert _runtime_environment_hash(
        baseline_environment,
        runtime_binary=baseline_binary,
    ) != _runtime_environment_hash(
        changed_device,
        runtime_binary=candidate_binary,
    )


def test_source_patch_profile_binds_candidate_runner_binary(tmp_path: Path) -> None:
    baseline_binary = tmp_path / "baseline" / "llama-bench"
    baseline_binary.parent.mkdir()
    baseline_binary.write_bytes(b"baseline")
    candidate_root = tmp_path / "candidate-worktree"
    candidate_root.mkdir()
    candidate_binary = candidate_root / "build" / "bin" / "llama-bench"
    candidate_binary.parent.mkdir(parents=True)
    candidate_binary.write_bytes(b"candidate")
    task = OptimizationTask(
        id="q8-candidate-profile",
        campaign_kind=CampaignKind.LLAMA_CPP_Q8,
        model=ModelTarget(path=tmp_path / "model.gguf", quantization="Q8_0"),
        runtime=RuntimeTarget(
            repo_path=tmp_path / "repo",
            base_commit="abc",
            prepared_binary_path=baseline_binary,
        ),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    patch = tmp_path / "change.patch"
    patch.write_text("patch")
    spec = ExperimentSpec(
        id="candidate",
        task_id=task.id,
        hypothesis_id="hypothesis",
        change=ChangeSet(
            kind="source_patch",
            description="candidate kernel",
            patch_path=patch,
        ),
    )
    spec_artifact = store.save_json(
        task.id, "experiments/candidate/spec.json", spec, producer="test"
    )
    store.save_json(
        task.id,
        "state/active-experiment.json",
        {"experiment_id": spec.id, "spec_path": spec_artifact.path},
        producer="test",
    )
    store.save_json(
        task.id,
        "state/active-execution.json",
        {"experiment_id": spec.id},
        producer="test",
    )
    store.save_json(
        task.id,
        "experiments/candidate/runner-result.json",
        {
            "experiment_id": spec.id,
            "execution_root": str(candidate_root),
            "binary": {
                "path": str(candidate_binary),
                "sha256": live_module.sha256_file(candidate_binary),
            },
        },
        producer="test",
    )

    binding = _profile_binding(task, store, spec.id)

    assert binding["arguments"]["command"][0] == str(candidate_binary.resolve())
    assert binding["arguments"]["cwd"] == str(candidate_root.resolve())
    assert binding["binary_source"] == "candidate_runner"
    assert binding["binary_sha256"] == live_module.sha256_file(candidate_binary)
    assert binding["arguments"]["max_trace_bytes"] == 200_000_000


def test_discover_hotspots_persists_exact_approval_before_execution(
    tmp_path: Path,
) -> None:
    task = OptimizationTask(
        id="live-approval",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(
            repo_path=tmp_path / "llama.cpp",
            base_commit="abc",
            build_dir=tmp_path / "build",
        ),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    record = WorkflowRecord(
        task_id=task.id,
        current_stage=WorkflowStage.DISCOVER_HOTSPOTS,
    )
    store.save_workflow(record)

    unchanged, detail = discover_hotspots(task, record, store)

    assert unchanged.current_stage == WorkflowStage.DISCOVER_HOTSPOTS
    assert detail["paused_for"] == "approval"
    assert detail["tool"] == "rocm_profile_workload"
    assert detail["arguments"]["preset"] == "kernel-timing"
    assert detail["arguments"]["command"][detail["arguments"]["command"].index("-n") + 1] == "128"
    assert detail["arguments"]["command"][detail["arguments"]["command"].index("-r") + 1] == "1"
    assert detail["arguments"]["command"][0] == str(
        (tmp_path / "build" / "bin" / "llama-bench").resolve()
    )
    assert detail["execution_context"]["runtime_environment"] == {
        "HSA_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/opt/rocm/core-7.14/lib:/opt/rocm/lib",
    }
    assert detail["execution_context"]["e2e_protocol_hash"]
    assert detail["execution_context"]["profile_command_hash"] == argv_sha256(
        detail["arguments"]["command"]
    )
    assert detail["execution_context"]["profile_protocol"] == {
        "schema_version": 1,
        "kind": "kernel-timing-short",
        "preset": "kernel-timing",
        "prompt_tokens": [0],
        "generation_tokens": [128],
        "repetitions": 1,
        "warmup": True,
        "batch_size": 2048,
        "ubatch_size": 512,
        "threads": 12,
        "gpu_layers": 999,
        "visible_device": 0,
        "device_selector": "ROCm0",
        "profile_command_hash": detail["execution_context"]["profile_command_hash"],
    }
    bound = approval_request(
        detail["tool"],
        detail["arguments"],
        execution_context=detail["execution_context"],
    )
    changed_e2e = dict(detail["execution_context"])
    changed_e2e["e2e_protocol_hash"] = "0" * 64
    changed_profile = dict(detail["execution_context"])
    changed_profile["profile_command_hash"] = "1" * 64
    assert bound.request_sha256 == detail["request_hash"]
    assert approval_request(
        detail["tool"], detail["arguments"], execution_context=changed_e2e
    ).request_sha256 != detail["request_hash"]
    assert approval_request(
        detail["tool"], detail["arguments"], execution_context=changed_profile
    ).request_sha256 != detail["request_hash"]
    request_path = (
        store.task_dir(task.id) / "state" / "approvals" / f"{detail['request_id']}.request.json"
    )
    assert request_path.is_file()

    second_record, second_detail = discover_hotspots(task, record, store)
    assert second_record == record
    assert second_detail["request_hash"] == detail["request_hash"]


def test_deeper_profile_does_not_silently_fall_back_to_kernel_timing(
    tmp_path: Path,
) -> None:
    task = OptimizationTask(
        id="deeper-profile",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(repo_path=tmp_path / "repo", base_commit="abc"),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    record = WorkflowRecord(task_id=task.id, current_stage=WorkflowStage.DISCOVER_HOTSPOTS)
    store.save_workflow(record)
    store.save_json(
        task.id,
        "state/requested-profile.json",
        {
            "requested_profile_level": "LEVEL_3",
            "requested_by": "decision.json",
            "fulfilled": False,
        },
        producer="test",
    )

    updated, detail = discover_hotspots(task, record, store)

    assert updated.current_stage == WorkflowStage.CLASSIFY_BOTTLENECK
    assert updated.completions[-1].evidence_artifacts["kernel_evidence"].sha256
    assert detail["evidence_status"] == "unavailable"
    assert "LEVEL_3" in detail["problems"][0]
    approvals = store.task_dir(task.id) / "state" / "approvals"
    assert not approvals.exists()


def test_fixed_protocol_isolates_host_device_and_uses_logical_device_zero(
    tmp_path: Path,
) -> None:
    task = OptimizationTask(
        id="protocol",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(repo_path=tmp_path / "repo", base_commit="abc"),
        gpu={"gfx_target": "gfx942", "device_id": 3},
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )

    environment, unset = _runtime_environment(task)
    argv = benchmark_protocol_argv(task, tmp_path / "llama-bench")

    assert environment["HSA_VISIBLE_DEVICES"] == "3"
    assert environment["HIP_VISIBLE_DEVICES"] == "3"
    assert environment["ROCR_VISIBLE_DEVICES"] == "3"
    assert argv[argv.index("-n") + 1] == "128,512"
    assert argv[argv.index("-dev") + 1] == "ROCm0"
    assert argv[argv.index("-mg") + 1] == "0"
    assert "LLAMA_Q4_RDNA_TRACE" in unset
    assert "SMITHY_CONFIG" in unset


def test_runtime_environment_rejects_candidate_device_override(tmp_path: Path) -> None:
    task = OptimizationTask(
        id="device-override",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(repo_path=tmp_path / "repo", base_commit="abc"),
        gpu={"gfx_target": "gfx942", "device_id": 3},
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )

    with pytest.raises(LiveWorkflowError, match="cannot override task GPU isolation"):
        _runtime_environment(
            task,
            candidate_env={"HIP_VISIBLE_DEVICES": "0"},
        )


def test_observed_architecture_uses_mcp_hip_evidence_and_feature_base(
    tmp_path: Path,
) -> None:
    task = OptimizationTask(
        id="observed-architecture",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(repo_path=tmp_path / "repo", base_commit="abc"),
        gpu={"gfx_target": "gfx942", "device_id": 2},
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    call = MCPToolCall(
        tool_name="rocm_hip_capabilities",
        arguments={},
        called_at="2026-08-23T00:00:00Z",
        is_error=False,
        structured_content={
            "schema": "rocm.hip-capabilities.v1",
            "detected_gfx_targets": ["gfx942:sramecc+:xnack-"],
        },
        raw_result={},
    )

    evidence, problems = live_module._observed_architecture_evidence(task, [call])

    assert problems == []
    assert evidence["observed_gfx_targets"] == ["gfx942"]
    assert evidence["host_device_id"] == 2
    assert evidence["visible_logical_device_id"] == 0
    assert evidence["matches"] is True

    mismatch = MCPToolCall(
        tool_name="rocm_hip_capabilities",
        arguments={},
        called_at="2026-08-23T00:00:00Z",
        is_error=False,
        structured_content={
            "schema": "rocm.hip-capabilities.v1",
            "detected_gfx_targets": ["gfx1201"],
        },
        raw_result={},
    )
    _, mismatch_problems = live_module._observed_architecture_evidence(task, [mismatch])
    assert "does not match" in mismatch_problems[0]


def test_source_candidate_runtime_does_not_load_baseline_shared_libraries(
    tmp_path: Path,
) -> None:
    baseline_binary = tmp_path / "baseline-build/bin/llama-bench"
    candidate_binary = tmp_path / "candidate-build/bin/llama-bench"
    task = OptimizationTask(
        id="candidate-libraries",
        campaign_kind="llama_cpp_q8",
        model=ModelTarget(path=tmp_path / "model.gguf", quantization="Q8_0"),
        runtime=RuntimeTarget(
            repo_path=tmp_path / "repo",
            base_commit="abc",
            build_dir=tmp_path / "baseline-build",
            prepared_binary_path=baseline_binary,
        ),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
        metadata={
            "rocm_library_path": (
                f"{baseline_binary.parent}:/opt/rocm/core-7.14/lib"
            )
        },
    )

    environment, _ = _runtime_environment(task, runtime_binary=candidate_binary)

    assert environment["LD_LIBRARY_PATH"].split(":") == [
        str(candidate_binary.parent.resolve()),
        "/opt/rocm/core-7.14/lib",
    ]
    assert str(baseline_binary.parent.resolve()) not in environment[
        "LD_LIBRARY_PATH"
    ].split(":")


def test_q8_offload_requires_q8_rocm0_and_all_gpu_layers() -> None:
    parsed = parse_llama_bench_json(
        {
            "n_prompt": 0,
            "n_gen": 1,
            "avg_ts": 34.0,
            "samples_ts": [34.0],
            "model_type": "qwen3 8B Q8_0",
            "backends": "ROCm",
            "devices": "ROCm0",
            "main_gpu": 0,
            "n_gpu_layers": 999,
            "no_kv_offload": False,
        }
    )

    assert q8_offload_evidence(parsed)["status"] == "passed"
    parsed.raw[0]["devices"] = "CPU"
    assert q8_offload_evidence(parsed)["status"] == "failed"


def test_baseline_old_and_split_profiles_share_short_argv_but_not_runtime_env(
    tmp_path: Path,
) -> None:
    task = OptimizationTask(
        id="profile-variants",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(
            repo_path=tmp_path / "repo",
            base_commit="abc",
            build_dir=tmp_path / "build",
        ),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    baseline = _profile_binding(task, store, "baseline")

    bindings = {"baseline": baseline}
    for variant, env in (
        ("old", {"LLAMA_Q4_RDNA_MAPPING": "old"}),
        ("split", {"LLAMA_Q4_RDNA_SIDECAR": "/models/model.q4rdna"}),
    ):
        spec = ExperimentSpec(
            id=variant,
            task_id=task.id,
            hypothesis_id=f"hypothesis-{variant}",
            change=ChangeSet(
                kind="runtime_config",
                description=variant,
                env=env,
            ),
            # Profiling must not inherit a wide or otherwise mutated E2E argv.
            commands={"e2e": ["/tmp/not-the-canonical-profile", "-n", "999"]},
        )
        artifact = store.save_json(
            task.id, f"experiments/{variant}/spec.json", spec, producer="test"
        )
        store.save_json(
            task.id,
            "state/active-experiment.json",
            {"experiment_id": variant, "spec_path": artifact.path},
            producer="test",
        )
        bindings[variant] = _profile_binding(task, store, variant)

    commands = {tuple(value["arguments"]["command"]) for value in bindings.values()}
    command_hashes = {value["profile_command_hash"] for value in bindings.values()}
    assert len(commands) == 1
    assert len(command_hashes) == 1
    assert (
        baseline["arguments"]["command"][baseline["arguments"]["command"].index("-n") + 1] == "128"
    )
    assert bindings["old"]["runtime_environment"]["LLAMA_Q4_RDNA_MAPPING"] == "old"
    assert (
        bindings["split"]["runtime_environment"]["LLAMA_Q4_RDNA_SIDECAR"] == "/models/model.q4rdna"
    )


def test_transport_failure_requires_a_new_approval_attempt(tmp_path: Path, monkeypatch) -> None:
    task = OptimizationTask(
        id="retry-profile",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(
            repo_path=tmp_path / "repo",
            base_commit="abc",
            build_dir=tmp_path / "build",
        ),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    record = WorkflowRecord(task_id=task.id, current_stage=WorkflowStage.DISCOVER_HOTSPOTS)
    store.save_workflow(record)
    _, requested = discover_hotspots(task, record, store)
    store.save_json(
        task.id,
        f"state/approvals/{requested['request_id']}.receipt.json",
        ApprovalReceipt(
            request_id=requested["request_id"],
            request_hash=requested["request_hash"],
        ),
        producer="test",
    )

    async def fail_transport(*args, **kwargs):
        del args, kwargs
        raise ConnectionError("stdio closed")

    monkeypatch.setattr(live_module, "_approved_profile", fail_transport)
    _, failed = discover_hotspots(task, record, store)
    assert failed["paused_for"] == "new_approval_after_transport_failure"
    assert failed["attempt_id"] == "baseline-attempt-0002"
    assert failed["request_id"] != requested["request_id"]
    _, retried = discover_hotspots(task, record, store)
    assert retried["paused_for"] == "approval"
    assert retried["attempt_id"] == "baseline-attempt-0002"
    assert retried["request_id"] != requested["request_id"]
    old_receipt = store.load_json(
        task.id,
        f"state/approvals/{requested['request_id']}.receipt.json",
        ApprovalReceipt,
    )
    assert old_receipt.consumed_at is not None
    profile_state = store.load_json(task.id, "state/profiles/baseline.json")
    assert profile_state["first_failed_attempt_id"] == "baseline-attempt-0001"
    assert profile_state["first_failure_reason"] == "ConnectionError: stdio closed"
    assert profile_state["failure_history"][0]["attempt_id"] == "baseline-attempt-0001"


def test_inconclusive_evidence_preserves_failure_and_requests_a_new_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    task = OptimizationTask(
        id="retry-inconclusive",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(
            repo_path=tmp_path / "repo",
            base_commit="abc",
            build_dir=tmp_path / "build",
        ),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    record = WorkflowRecord(task_id=task.id, current_stage=WorkflowStage.DISCOVER_HOTSPOTS)
    store.save_workflow(record)
    _, requested = discover_hotspots(task, record, store)
    store.save_json(
        task.id,
        f"state/approvals/{requested['request_id']}.receipt.json",
        ApprovalReceipt(
            request_id=requested["request_id"],
            request_hash=requested["request_hash"],
        ),
        producer="test",
    )

    async def inconclusive(*args, **kwargs):
        del args, kwargs
        profile = MCPToolCall(
            tool_name="rocm_profile_workload",
            arguments={},
            called_at="2026-01-01T00:00:00+00:00",
            is_error=False,
            structured_content={
                "schema": "rocm.mcp-kernel-evidence.v1",
                "case_id": "CASE-TEST",
                "kernel_evidence": {
                    "status": "partial",
                    "workload": {"status": "failed", "exit_code": -9},
                    "kernels": [{"name": "partial"}],
                    "warnings": ["trace exceeded byte limit"],
                },
            },
            raw_result={},
        )
        return {"server": "test"}, profile, None

    monkeypatch.setattr(live_module, "_approved_profile", inconclusive)
    _, retried = discover_hotspots(task, record, store)

    assert retried["paused_for"] == "new_approval_after_inconclusive_evidence"
    assert retried["failed_attempt_id"] == "baseline-attempt-0001"
    assert retried["attempt_id"] == "baseline-attempt-0002"
    assert retried["request_id"] != requested["request_id"]
    assert retried["arguments"]["command"][retried["arguments"]["command"].index("-n") + 1] == "128"
    state = store.load_json(task.id, "state/profiles/baseline.json")
    assert [attempt["status"] for attempt in state["attempts"]] == [
        "evidence_inconclusive",
        "awaiting_approval",
    ]
    assert state["first_failed_attempt_id"] == "baseline-attempt-0001"
    assert state["failure_history"][0]["problems"]


def test_existing_inconclusive_state_migrates_and_creates_fresh_approval(
    tmp_path: Path,
) -> None:
    task = OptimizationTask(
        id="migrate-inconclusive",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(
            repo_path=tmp_path / "repo",
            base_commit="abc",
            build_dir=tmp_path / "build",
        ),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    record = WorkflowRecord(
        task_id=task.id, current_stage=WorkflowStage.DISCOVER_HOTSPOTS
    )
    store.save_workflow(record)
    _, first = discover_hotspots(task, record, store)
    state = store.load_json(task.id, "state/profiles/baseline.json")
    state["schema_version"] = 1
    state["attempts"][0]["status"] = "evidence_inconclusive"
    state["attempts"][0]["problems"] = ["profiled workload exit_code is -9"]
    store.save_json(
        task.id, "state/profiles/baseline.json", state, producer="old-driver"
    )

    _, fresh = discover_hotspots(task, record, store)

    assert fresh["paused_for"] == "approval"
    assert fresh["attempt_id"] == "baseline-attempt-0002"
    assert fresh["request_id"] != first["request_id"]
    assert fresh["execution_context"]["profile_protocol"]["generation_tokens"] == [
        128
    ]
    migrated = store.load_json(task.id, "state/profiles/baseline.json")
    assert migrated["schema_version"] == 2
    assert migrated["first_failure_reason"] == "profiled workload exit_code is -9"
    assert len(migrated["failure_history"]) == 1


def _percentile_only_profile() -> MCPToolCall:
    return MCPToolCall(
        tool_name="rocm_profile_workload",
        arguments={},
        called_at="2026-01-01T00:00:00+00:00",
        is_error=False,
        structured_content={
            "schema": "rocm.mcp-kernel-evidence.v1",
            "case_id": "CASE-PERCENTILE",
            "kernel_evidence": {
                "preset": "kernel-timing",
                "status": "partial",
                "profiler": {"status": "completed", "tool": "rocprofv3"},
                "workload": {"status": "completed", "exit_code": 0},
                "kernels": [
                    {
                        "name": "hot_kernel",
                        "dispatch_count": 20_000,
                        "total_duration_ns": 1_000_000,
                        "average_duration_ns": 50.0,
                        "gpu_kernel_time_share_percent": 80.0,
                        "p50_duration_ns": None,
                        "p95_duration_ns": None,
                        "percentiles_reliable": False,
                    },
                    {
                        "name": "small_kernel",
                        "dispatch_count": 2,
                        "total_duration_ns": 250_000,
                        "average_duration_ns": 125_000.0,
                        "gpu_kernel_time_share_percent": 20.0,
                        "p50_duration_ns": 100_000,
                        "p95_duration_ns": 150_000,
                        "percentiles_reliable": True,
                    },
                ],
                "warnings": [
                    "p50/p95 were omitted for kernels whose duration sample limit was exceeded"
                ],
            },
        },
        raw_result={},
    )


def test_profile_gate_accepts_only_percentile_sample_cap_with_complete_aggregates() -> None:
    profile = _percentile_only_profile()

    assert live_module._profile_problems(profile) == []

    evidence = profile.structured_content["kernel_evidence"]
    evidence["warnings"].append("Kernel evidence CSV was truncated: trace.csv")
    assert live_module._profile_problems(profile) == [
        "partial kernel evidence has warnings that can bias hotspot ranking"
    ]
    evidence["warnings"] = [
        "p50/p95 were omitted for kernels whose duration sample limit was exceeded"
    ]
    evidence["profiler"]["status"] = "partial"
    assert "profiler status is 'partial'" in live_module._profile_problems(profile)


def test_reclassifies_stored_percentile_only_evidence_and_supersedes_unapproved_retry(
    tmp_path: Path,
) -> None:
    task = OptimizationTask(
        id="reclassify-profile",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(
            repo_path=tmp_path / "repo",
            base_commit="abc",
            build_dir=tmp_path / "build",
        ),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    record = WorkflowRecord(
        task_id=task.id, current_stage=WorkflowStage.DISCOVER_HOTSPOTS
    )
    store.save_workflow(record)
    _, requested = discover_hotspots(task, record, store)
    state = store.load_json(task.id, "state/profiles/baseline.json")
    first = state["attempts"][0]
    profile = _percentile_only_profile()
    summary = MCPToolCall(
        tool_name="rocm_get_case_summary",
        arguments={"case_id": "CASE-PERCENTILE"},
        called_at="2026-01-01T00:00:01+00:00",
        is_error=False,
        structured_content={"schema": "rocm.mcp-case-summary.v1"},
        raw_result={},
    )
    evidence = store.save_json(
        task.id,
        "artifacts/mcp/profiles/baseline/baseline-attempt-0001/evidence.json",
        {"profile": profile.to_dict(), "case_summary": summary.to_dict()},
        producer="test",
    )
    first["status"] = "evidence_inconclusive"
    first["evidence"] = evidence.path
    first["problems"] = [
        "partial kernel evidence has warnings that can bias hotspot ranking"
    ]
    live_module._record_profile_failure(state, first)
    live_module._create_profile_attempt(
        task,
        store,
        target="baseline",
        binding=first["binding"],
        attempts=state["attempts"],
    )
    store.save_json(
        task.id, "state/profiles/baseline.json", state, producer="old-gate"
    )

    updated, detail = discover_hotspots(task, record, store)

    assert requested["attempt_id"] == "baseline-attempt-0001"
    assert updated.current_stage == WorkflowStage.CLASSIFY_BOTTLENECK
    assert detail["profile_complete"] is True
    assert detail["attempt_id"] == "baseline-attempt-0001"
    recovered = store.load_json(task.id, "state/profiles/baseline.json")
    assert [attempt["status"] for attempt in recovered["attempts"]] == [
        "succeeded",
        "superseded_before_execution",
    ]
    assert recovered["failure_history"][0]["reclassified_as"] == "succeeded"
    assert recovered["attempts"][1]["superseded_by"] == "baseline-attempt-0001"


def test_runtime_library_hash_ignores_aslr_addresses(tmp_path: Path) -> None:
    task = OptimizationTask(
        id="libraries",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(repo_path=tmp_path / "repo", base_commit="abc"),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    first = "libamdhip64.so => /opt/rocm/lib/libamdhip64.so (0x00007f01)\n"
    second = "libamdhip64.so => /opt/rocm/lib/libamdhip64.so (0x0000abcd)\n"

    assert _runtime_libraries_hash(task, first) == _runtime_libraries_hash(task, second)


def test_available_evidence_identity_binds_path_and_sha256(tmp_path: Path) -> None:
    task = OptimizationTask(
        id="evidence-identity",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(repo_path=tmp_path / "repo", base_commit="abc"),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    artifact = store.save_evidence_json(
        task.id, "timing.json", {"status": "captured"}, producer="test"
    )
    store.save_json(
        task.id, "state/agent-context.json", {"mutable": True}, producer="test"
    )

    evidence = live_module._available_evidence(store, task.id)

    identity = f"{artifact.path}@sha256:{artifact.sha256}"
    assert any(item.id == identity and item.artifact == artifact for item in evidence)
    assert all(not item.id.startswith("state/") for item in evidence)
