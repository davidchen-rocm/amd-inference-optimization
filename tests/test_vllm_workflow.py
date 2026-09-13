from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from amd_inference_opt.change_policy import validate_experiment_change
from amd_inference_opt.cli import app
from amd_inference_opt.mi300x_observation import observation_request_sha256
from amd_inference_opt.models import (
    AccuracyRequirement,
    ApprovalReceipt,
    BaselineResult,
    BenchmarkProtocol,
    BenchmarkResult,
    CampaignKind,
    ChangeKind,
    ChangePolicy,
    ChangeSet,
    DecisionOutcome,
    EnvironmentFingerprint,
    EnvironmentRequirements,
    ExperimentResult,
    GPUTarget,
    Hypothesis,
    MCPConfig,
    MetricSeries,
    ModelTarget,
    OptimizationObjective,
    OptimizationTask,
    PerformanceMetricRequirement,
    QualityConstraints,
    QualityResult,
    ROCmTarget,
    RunStatus,
    TaskBudgets,
    VLLMRuntimeTarget,
    VLLMServingWorkloadConfig,
    WorkflowStatus,
    utc_now,
)
from amd_inference_opt.profile_contract import ProfileContract
from amd_inference_opt.store import ExperimentStore
from amd_inference_opt.vllm_adapter import (
    RuntimeVerificationStatus,
    ServerExecutionRecord,
    ServerExecutionStatus,
    ServerStartResult,
    ServerStopResult,
    VLLMHealthResult,
    VLLMRuntimeEvidence,
)
from amd_inference_opt.vllm_model_snapshot import capture_vllm_model_snapshot
from amd_inference_opt.vllm_models import (
    MI300XDeviceBinding,
    MI300XInspectionEvidence,
    VLLMAgentDecision,
    VLLMCampaignConfig,
    VLLMExperimentSpec,
    VLLMModelCoordinate,
    VLLMNextActionKind,
    VLLMProfileProtocol,
    VLLMProfileResultEvidence,
    VLLMProfileRuntimeEvidence,
    VLLMQualityProtocol,
    VLLMServingProtocol,
    VLLMStage,
    canonical_sha256,
)
from amd_inference_opt.vllm_workflow import (
    LocalVLLMBenchmarkPort,
    LocalVLLMQualityPort,
    VLLMWorkflowCoordinator,
    VLLMWorkflowError,
    build_server_spec,
)

UUID = "GPU-deadbeef"
IMAGE_SHA = "4" * 64
EXECUTABLE_SHA = "2" * 64
ENVIRONMENT_SHA = "3" * 64
SERVER_LAUNCHER_SHA = "1" * 64
PROFILE_LAUNCHER_SHA = "5" * 64
AMD_SMI_COMMAND_SHA = "6" * 64
HIP_PROBE_COMMAND_SHA = "7" * 64
MODEL_REVISION = "a" * 40
TOKENIZER_REVISION = "c" * 40
CANONICAL_METRICS = [
    "request_throughput_requests_per_second",
    "output_throughput_tokens_per_second",
    "total_throughput_tokens_per_second",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
]
CLI_RUNNER = CliRunner()


def _config(tmp_path: Path, **overrides: Any) -> VLLMCampaignConfig:
    model_path = tmp_path / "models" / "org" / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "config.json").write_text(
        '{"architectures":["TestForCausalLM"]}\n', encoding="utf-8"
    )
    manifest = capture_vllm_model_snapshot(
        model_path,
        model_id="org/model",
        revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
    )
    manifest_path = tmp_path / "model-snapshot-manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(by_alias=True), encoding="utf-8"
    )
    python = tmp_path / "venv" / "bin" / "python3"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"test-python-interpreter")
    profile_launcher = tmp_path / "gpuopt" / "profile_vllm_offline.py"
    environment_manifest = tmp_path / "vllm-environment-manifest.json"
    environment_manifest.write_text("{}\n", encoding="utf-8")
    server_argv = [
        str(python),
        "-I",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        "org/model",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
    ]
    benchmark_argv = [
        str(python),
        "-I",
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        "http://127.0.0.1:8000",
        "--endpoint",
        "/v1/completions",
        "--model",
        "org/model",
        "--dataset-name",
        "random",
        "--num-prompts",
        "16",
        "--seed",
        "42",
        "--max-concurrency",
        "4",
        "--random-input-len",
        "128",
        "--random-output-len",
        "32",
    ]
    serving = VLLMServingProtocol(
        server_argv=server_argv,
        benchmark_argv=benchmark_argv,
        cwd=tmp_path,
        server_env={
            "ROCR_VISIBLE_DEVICES": UUID,
            "PYTHONNOUSERSITE": "1",
        },
        dtype="bfloat16",
        concurrency=4,
        input_tokens=128,
        output_tokens=32,
        num_prompts=16,
        warmup_runs=1,
        sample_count=3,
        timeout_seconds=900,
        required_metrics=CANONICAL_METRICS,
        engine_config={
            "launcher_kind": "python_module",
            "launcher": "vllm.entrypoints.openai.api_server",
            "launcher_sha256": SERVER_LAUNCHER_SHA,
        },
    )
    quality = VLLMQualityProtocol(
        argv=[str(tmp_path / "tools" / "quality"), "--model", "org/model"],
        cwd=tmp_path,
        output_path=tmp_path / "quality-result.json",
    )
    environment_fields = [
        "gpu_gfx",
        "gpu_product_name",
        "gpu_oam_id",
        "gpu_xcc_count",
        "gpu_device",
        "gpu_device_identity",
        "gpu_device_uuid",
        "gpu_pci_bdf",
        "gpu_compute_partition",
        "gpu_memory_partition",
        "gpu_partition_id",
        "rocm_version",
        "vllm_version",
        "pytorch_version",
        "python_version",
        "python_executable_sha256",
        "environment_manifest_sha256",
        "vllm_launcher_sha256",
        "image_digest",
        "hf_model_id",
        "hf_revision",
        "tokenizer_revision",
        "model_snapshot_digest",
        "serving_protocol_sha256",
        "dtype",
        "quantization",
        "tensor_parallel_size",
    ]
    task = OptimizationTask(
        id="vllm-mi300x-test",
        campaign_kind=CampaignKind.VLLM_MI300X,
        model=ModelTarget(path=model_path),
        runtime=VLLMRuntimeTarget(
            deployment="native",
            version="0.10.1",
            executable=str(python),
            executable_sha256=EXECUTABLE_SHA,
            environment_manifest_sha256=ENVIRONMENT_SHA,
            image="rocm/vllm",
            image_digest=IMAGE_SHA,
            python_version="3.12.4",
            pytorch_version="2.7.1",
            rocm_version="6.4.1",
        ),
        gpu=GPUTarget(gfx_target="gfx942", device_id=1, name="AMD Instinct MI300X"),
        rocm=ROCmTarget(version="6.4.1"),
        workload=VLLMServingWorkloadConfig(
            input_tokens=128,
            output_tokens=32,
            num_prompts=16,
            concurrency=4,
            seed=42,
        ),
        benchmark=BenchmarkProtocol(
            warmup_runs=1,
            sample_count=3,
            timeout_seconds=900,
            benchmark_command=benchmark_argv,
            required_metrics=CANONICAL_METRICS,
            protocol_hash=serving.coordinate_sha256,
        ),
        objective=OptimizationObjective(
            primary_metric="output_throughput_tokens_per_second",
            metric_requirements=[
                PerformanceMetricRequirement(
                    metric="output_throughput_tokens_per_second",
                    minimum_improvement_percent=5,
                )
            ]
        ),
        quality=QualityConstraints(
            accuracy_requirements=[
                AccuracyRequirement(
                    metric="math_accuracy", max_drop_percentage_points=0.01
                )
            ]
        ),
        change_policy=ChangePolicy(
            allowed_change_kinds=[ChangeKind.RUNTIME_CONFIG],
            locked_coordinates=[
                "image_digest",
                "model_snapshot_digest",
                "model_revision",
                "tokenizer_revision",
                "gpu_device_identity",
                "gpu_partition",
                "dtype",
                "quantization",
                "tensor_parallel_size",
                "benchmark_protocol",
                "runtime_environment",
            ],
            allowed_runtime_args=["--max-num-seqs"],
            runtime_arg_arity={"--max-num-seqs": 1},
        ),
        environment=EnvironmentRequirements(
            required_match_fields=environment_fields,
            max_sample_cv_percent=2,
            require_stable_telemetry=True,
            require_fresh_capture=True,
        ),
        budgets=TaskBudgets(max_experiments=2, max_gpu_minutes=60),
        mcp=MCPConfig(
            command=[str(tmp_path / "agent" / "mcp")],
            env={
                "ROCR_VISIBLE_DEVICES": UUID,
                "PYTHONNOUSERSITE": "1",
            },
            cwd=tmp_path,
        ),
    )
    profile = VLLMProfileProtocol(
        profile_argv=[
            str(python),
            "-I",
            "-m",
            "vllm.entrypoints.cli.main",
            "bench",
            "throughput",
            "--backend",
            "vllm",
            "--dataset-name",
            "random",
            "--model",
            str(model_path),
            "--tensor-parallel-size",
            "1",
            "--dtype",
            "bfloat16",
            "--input-len",
            "128",
            "--output-len",
            "32",
            "--num-prompts",
            "16",
            "--seed",
            "42",
        ],
        launcher_path=profile_launcher,
        launcher_sha256=PROFILE_LAUNCHER_SHA,
        environment_manifest_path=environment_manifest,
        cwd=tmp_path,
        environment="same_container",
    )
    values: dict[str, Any] = {
        "task": task,
        "model": VLLMModelCoordinate(
            model_id="org/model",
            local_path=model_path,
            snapshot_manifest_path=manifest_path,
            revision=MODEL_REVISION,
            snapshot_digest=manifest.snapshot_digest,
            tokenizer_revision=TOKENIZER_REVISION,
        ),
        "device": MI300XDeviceBinding(
            device_ids=[1],
            oam_id=1,
            device_uuid=UUID,
            pci_bdf="0000:41:00.0",
            partition_id=0,
            amd_smi_command_sha256=AMD_SMI_COMMAND_SHA,
            hip_probe_command_sha256=HIP_PROBE_COMMAND_SHA,
        ),
        "serving": serving,
        "profile": profile,
        "quality_protocol": quality,
    }
    values.update(overrides)
    return VLLMCampaignConfig(**values)


def _inspection(
    config: VLLMCampaignConfig, store: ExperimentStore
) -> MI300XInspectionEvidence:
    amd_smi = store.save_evidence_json(
        config.task.id,
        "vllm/raw-inspection/amd-smi",
        {"command_sha256": AMD_SMI_COMMAND_SHA, "product": "MI300X"},
        producer="amd-smi-inspection",
    )
    hip_probe = store.save_evidence_json(
        config.task.id,
        "vllm/raw-inspection/hip-probe",
        {"command_sha256": HIP_PROBE_COMMAND_SHA, "visible_device_count": 1},
        producer="hip-probe-inspection",
    )
    return MI300XInspectionEvidence(
        scope="container_preflight",
        image_digest=IMAGE_SHA,
        gfx_target="gfx942",
        product_name="AMD Instinct MI300X",
        oam_id=1,
        xcc_count=8,
        rocr_visible_devices=UUID,
        device_uuid=UUID,
        pci_bdf="0000:41:00.0",
        compute_partition="SPX",
        memory_partition="NPS1",
        partition_id=0,
        rocm_version="6.4.1",
        vllm_version="0.10.1",
        pytorch_version="2.7.1",
        python_version="3.12.4",
        launcher_sha256=SERVER_LAUNCHER_SHA,
        environment_manifest_sha256=ENVIRONMENT_SHA,
        amd_smi_artifact=amd_smi,
        hip_probe_artifact=hip_probe,
        amd_smi_command_sha256=AMD_SMI_COMMAND_SHA,
        hip_probe_command_sha256=HIP_PROBE_COMMAND_SHA,
    )


def _fake_profile_materializer(**values: Any) -> Any:
    attempt_id = values["attempt_id"]
    attempt_root = Path(values["output_root"]) / attempt_id
    attempt_root.mkdir(parents=True)
    coordinate_path = attempt_root / "profile-coordinate.json"
    sidecar_path = attempt_root / "profile-launch-sidecar.json"
    throughput_result_path = attempt_root / "vllm-throughput-result.json"
    coordinate_path.write_text(
        json.dumps(
            {
                "schema": "gpuopt.vllm-profile-coordinate.v1",
                "profile_launcher_sha256": PROFILE_LAUNCHER_SHA,
                "environment_manifest_sha256": ENVIRONMENT_SHA,
                "workload": {"completion_output": str(throughput_result_path)},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    coordinate_sha = hashlib.sha256(coordinate_path.read_bytes()).hexdigest()
    model_manifest = Path(values["model_manifest_path"])
    model_manifest_sha = hashlib.sha256(model_manifest.read_bytes()).hexdigest()
    argv = (
        str(values["python_executable"]),
        "-I",
        str(values["launcher_path"]),
        "--gpuopt-coordinate",
        str(coordinate_path),
        "--gpuopt-coordinate-sha256",
        coordinate_sha,
        "--gpuopt-model-manifest",
        str(model_manifest),
        "--gpuopt-model-manifest-sha256",
        model_manifest_sha,
        "--gpuopt-sidecar",
        str(sidecar_path),
        "--",
        *values["offline_argv"],
        "--output-json",
        str(throughput_result_path),
    )
    return SimpleNamespace(
        attempt_id=attempt_id,
        argv=argv,
        coordinate_path=coordinate_path,
        coordinate_sha256=coordinate_sha,
        sidecar_path=sidecar_path,
        throughput_result_path=throughput_result_path,
        environment_manifest_path=Path(values["environment_manifest_path"]),
        environment_manifest_sha256=ENVIRONMENT_SHA,
        model_manifest_path=model_manifest,
        model_manifest_sha256=model_manifest_sha,
    )


def _coordinator(tmp_path: Path) -> VLLMWorkflowCoordinator:
    contract = ProfileContract(**json.loads(
        (Path(__file__).parent / "fixtures" / "rocm-profile-contract.json").read_text()
    ))
    return VLLMWorkflowCoordinator(
        tmp_path / "store", profile_materializer=_fake_profile_materializer,
        profile_contract_reader=lambda server: contract,
    )


def _environment(
    config: VLLMCampaignConfig, capture_id: str, server_request_hash: str
) -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        values={
            **config.environment_coordinates,
            "server_request_hash": server_request_hash,
            "telemetry_gfxclk_mean_mhz": "1700.0",
            "telemetry_uclk_mean_mhz": "1300.0",
        },
        capture_id=capture_id,
        captured_at=utc_now(),
        source="observed",
        telemetry_stable=True,
    )


def _quality(config: VLLMCampaignConfig, accuracy: float = 0.8) -> QualityResult:
    return QualityResult(
        status=RunStatus.SUCCEEDED,
        correctness_passed=True,
        accuracies={"math_accuracy": accuracy},
        coordinate_hash=config.quality_protocol.coordinate_sha256,
        representation_hash=config.model.snapshot_digest,
    )


def _metrics(scale: float = 1.0) -> dict[str, MetricSeries]:
    return {
        "request_throughput_requests_per_second": MetricSeries(
            unit="req/s", samples=[10 * scale, 10.01 * scale, 9.99 * scale]
        ),
        "output_throughput_tokens_per_second": MetricSeries(
            unit="tok/s", samples=[100 * scale, 100.1 * scale, 99.9 * scale]
        ),
        "total_throughput_tokens_per_second": MetricSeries(
            unit="tok/s", samples=[120 * scale, 120.1 * scale, 119.9 * scale]
        ),
        "mean_ttft_ms": MetricSeries(unit="ms", samples=[20, 20.01, 19.99]),
        "mean_tpot_ms": MetricSeries(unit="ms", samples=[5, 5.01, 4.99]),
        "mean_itl_ms": MetricSeries(unit="ms", samples=[4, 4.01, 3.99]),
    }


class FakeLifecycle:
    def __init__(self) -> None:
        self.current: ServerExecutionRecord | None = None
        self.pid = 4100

    def start_or_resume(self, spec: Any) -> ServerStartResult:
        self.pid += 1
        now = utc_now().isoformat()
        runtime = VLLMRuntimeEvidence(
            verification=RuntimeVerificationStatus.VERIFIED,
            captured_at="2026-08-23T00:00:00+00:00",
            observed_pid=self.pid,
            observed_boot_id="boot-test",
            observed_start_ticks=9000 + self.pid,
            pid_executable_path=spec.argv[0],
            pid_executable_sha256=spec.native_executable_sha256,
            native_executable_matches=True,
            observed_environment_manifest_sha256=spec.environment_manifest_sha256,
            declared_environment_matches=True,
            unset_environment_absent=True,
            rocr_visible_devices=UUID,
            container_id="container-test",
            container_init_pid=1,
            container_image_id="sha256:image-id",
            container_binding_kind="cgroup_v2",
            process_binding_id="cgroup:test",
            container_binding_id="cgroup:test",
            container_repo_digests=(f"rocm/vllm@sha256:{IMAGE_SHA}",),
            image_digest_matches=True,
        )
        self.current = ServerExecutionRecord(
            pid=self.pid,
            boot_id="boot-test",
            start_ticks=9000 + self.pid,
            request_hash=spec.request_hash,
            identity=spec.identity,
            runtime_evidence=runtime,
            argv=spec.argv,
            cwd=spec.cwd,
            health_url=spec.health_url,
            status=ServerExecutionStatus.RUNNING,
            started_at=now,
            updated_at=now,
            shutdown_timeout_seconds=15,
            requires_explicit_stop=True,
            ready_at=now,
        )
        health = VLLMHealthResult(
            healthy=True,
            process_alive=True,
            endpoint_ownership_verified=True,
            checked_at=now,
            url=spec.health_url,
            status_code=200,
            models=("org/model",),
        )
        return ServerStartResult(record=self.current, health=health, reused=False)

    def stop(self, *, expected_request_hash: str | None = None) -> ServerStopResult:
        assert self.current is not None
        assert expected_request_hash == self.current.request_hash
        now = utc_now().isoformat()
        self.current = replace(
            self.current,
            status=ServerExecutionStatus.STOPPED,
            requires_explicit_stop=False,
            stopped_at=now,
            updated_at=now,
        )
        return ServerStopResult(record=self.current, stopped=True, reason="stopped exact PID")


def _profile_evidence(
    coordinator: VLLMWorkflowCoordinator,
    record: Any,
    permit: Any,
    config: VLLMCampaignConfig,
) -> VLLMProfileResultEvidence:
    runtime = VLLMProfileRuntimeEvidence(
        verification="VERIFIED",
        observed_pid=5001,
        observed_boot_id="boot-test",
        observed_start_ticks=15001,
        native_executable_sha256=EXECUTABLE_SHA,
        environment_manifest_sha256=ENVIRONMENT_SHA,
        profile_launcher_sha256=PROFILE_LAUNCHER_SHA,
        native_executable_matches=True,
        declared_environment_matches=True,
        unset_environment_absent=True,
        rocr_visible_devices=UUID,
        image_digest_matches=True,
        container_binding_kind="cgroup_v2",
        process_binding_id="cgroup:test",
        container_binding_id="cgroup:test",
        container_repo_digests=[f"rocm/vllm@sha256:{IMAGE_SHA}"],
    )
    base_context = {
        "schema": "rocm.run-context.v1",
        "argv": [
            str(Path(permit.arguments["command"][0]).resolve()),
            *permit.arguments["command"][1:],
        ],
        "cwd": permit.arguments["cwd"],
        "environment": {
            "ROCR_VISIBLE_DEVICES": UUID,
            "PYTHONNOUSERSITE": "1",
        },
        "gpus": [
            {
                "gfx_architecture": "gfx942",
                "pci_bdf": "0000:41:00.0",
                "partition_id": 0,
                "accelerator_partition": "SPX",
                "memory_partition": "NPS1",
            }
        ],
    }
    run_context_hash = f"sha256:{canonical_sha256(base_context)}"
    completion_metrics = {
        "num_requests": config.serving.num_prompts,
        "total_num_tokens": config.serving.num_prompts
        * (config.serving.input_tokens + config.serving.output_tokens),
        "elapsed_time": 2.0,
        "requests_per_second": config.serving.num_prompts / 2.0,
        "tokens_per_second": (
            config.serving.num_prompts
            * (config.serving.input_tokens + config.serving.output_tokens)
            / 2.0
        ),
    }
    completion_ref = coordinator.store.save_evidence_json(
        record.task_id,
        "vllm/profile-throughput-completion",
        completion_metrics,
        producer="vllm-bench-throughput",
    )
    execution = {
        "image_digest": IMAGE_SHA,
        "model_snapshot_digest": config.model.snapshot_digest,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "serving_protocol_sha256": config.serving.coordinate_sha256,
        "profile_command_sha256": config.profile.command_sha256,
        "profile_launcher_sha256": PROFILE_LAUNCHER_SHA,
        "environment_manifest_sha256": ENVIRONMENT_SHA,
        "device_uuid": UUID,
        "pci_bdf": "0000:41:00.0",
        "partition_id": 0,
        "runtime_evidence": runtime.model_dump(mode="json"),
        "offline_vllm_completion": {
            "artifact": completion_ref.model_dump(mode="json"),
            "metrics": completion_metrics,
        },
    }
    context_ref = coordinator.store.save_evidence_json(
        record.task_id,
        "vllm/profile-run-context",
        {
            **base_context,
            "collected_at": "2026-08-23T00:02:00+00:00",
            "run_context_hash": run_context_hash,
            "vllm_execution": execution,
        },
        producer="observed-profile-sidecar",
    )
    kernel = {"name": "vllm::paged_attention", "dispatch_count": 16}
    structured = {
        "schema": "rocm.mcp-kernel-evidence.v1",
        "status": "completed",
        "run_context_hash": run_context_hash,
        "kernel_evidence": {
            "status": "completed",
            "preset": config.profile.preset,
            "run_context_hash": run_context_hash,
            "workload_succeeded": True,
            "aggregate_timing_complete": True,
            "workload": {"status": "completed", "exit_code": 0},
            "profiler": {"status": "completed", "tool": "rocprofv3"},
            "kernels": [kernel],
        },
    }
    return VLLMProfileResultEvidence(
        mcp_call={
            "tool_name": "rocm_profile_workload",
            "arguments": permit.arguments,
            "is_error": False,
            "structured_content": structured,
            "framework_evidence": {
                "offline_vllm_completion": completion_ref.model_dump(mode="json"),
                "offline_vllm_metrics": completion_metrics,
            },
        },
        execution_environment_artifact=context_ref,
        runtime_evidence=runtime,
        run_context_hash=run_context_hash,
        image_digest=IMAGE_SHA,
        model_snapshot_digest=config.model.snapshot_digest,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        serving_protocol_sha256=config.serving.coordinate_sha256,
        profile_command_sha256=config.profile.command_sha256,
        profile_launcher_sha256=PROFILE_LAUNCHER_SHA,
        environment_manifest_sha256=ENVIRONMENT_SHA,
        device_uuid=UUID,
        pci_bdf="0000:41:00.0",
        partition_id=0,
        engine_kernel_count=1,
    )


def test_config_fails_closed_on_duplicate_locked_argv_and_bad_partition(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    duplicate = config.serving.model_copy(
        update={"server_argv": [*config.serving.server_argv, "--model", "/tmp/other"]}
    )
    duplicate_task = config.task.model_copy(
        update={
            "benchmark": config.task.benchmark.model_copy(
                update={"protocol_hash": duplicate.coordinate_sha256}
            )
        }
    )
    with pytest.raises(ValidationError, match="model snapshot exactly once"):
        _config(tmp_path, serving=duplicate, task=duplicate_task)
    with pytest.raises(ValidationError, match="unsupported MI300X partition pair"):
        MI300XDeviceBinding(
            device_ids=[1],
            oam_id=1,
            device_uuid=UUID,
            pci_bdf="0000:41:00.0",
            partition_id=0,
            compute_partition="SPX",
            memory_partition="NPS4",
            amd_smi_command_sha256=AMD_SMI_COMMAND_SHA,
            hip_probe_command_sha256=HIP_PROBE_COMMAND_SHA,
        )


def test_config_to_adapter_spec_binds_native_container_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = ExperimentStore(tmp_path / "store")
    store.create_task(config.task)
    spec = build_server_spec(config, store)
    assert spec.argv == tuple(config.serving.server_argv)
    assert spec.expected_served_model == "org/model"
    assert spec.native_executable_sha256 == EXECUTABLE_SHA
    assert spec.environment_manifest_sha256 == ENVIRONMENT_SHA
    assert spec.image_digest == f"sha256:{IMAGE_SHA}"
    assert spec.env == {
        "ROCR_VISIBLE_DEVICES": UUID,
        "PYTHONNOUSERSITE": "1",
    }
    assert "HIP_VISIBLE_DEVICES" in spec.unset_env
    assert "HSA_OVERRIDE_GFX_VERSION" in spec.unset_env
    assert spec.tensor_parallel_size == 1


def test_boolean_runtime_flag_uses_declared_zero_arity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    policy = config.task.change_policy.model_copy(
        update={
            "allowed_runtime_args": ["--max-num-seqs", "--enforce-eager"],
            "runtime_arg_arity": {"--max-num-seqs": 1, "--enforce-eager": 0},
        }
    )
    task = config.task.model_copy(update={"change_policy": policy})
    change = ChangeSet(
        kind=ChangeKind.RUNTIME_CONFIG,
        description="disable graph capture",
        runtime_args=["--enforce-eager"],
    )
    experiment = VLLMExperimentSpec(
        task_id=task.id,
        hypothesis_id="hypothesis-boolean",
        description="boolean runtime flag",
        change=change,
        server_argv=[*config.serving.server_argv, "--enforce-eager"],
        benchmark_argv=config.serving.benchmark_argv,
        server_env=config.serving.server_env,
    )
    validate_experiment_change(task, experiment)


def test_failed_start_that_may_be_live_forces_exact_cleanup(tmp_path: Path) -> None:
    config = _config(tmp_path)
    coordinator = _coordinator(tmp_path)
    lifecycle = FakeLifecycle()
    record = coordinator.create(config)
    record = coordinator.record_inspection(
        record, _inspection(config, coordinator.store)
    )
    spec = build_server_spec(config, coordinator.store)
    started = lifecycle.start_or_resume(spec)
    unverified_runtime = replace(
        started.record.runtime_evidence,
        verification=RuntimeVerificationStatus.UNVERIFIED,
        reason="manifest observer unavailable",
    )
    lifecycle.current = replace(started.record, runtime_evidence=unverified_runtime)
    failed = ServerStartResult(
        record=lifecycle.current,
        health=started.health,
        reused=False,
    )
    record = coordinator.record_server_start(record, failed)
    assert record.current_stage == VLLMStage.SERVER_START
    assert record.baseline_server_requires_stop is True
    assert coordinator.next_action(record).kind == VLLMNextActionKind.STOP_BASELINE_SERVER
    record = coordinator.stop_baseline_server(record, lifecycle)
    assert record.baseline_server_requires_stop is False
    assert coordinator.next_action(record).kind == VLLMNextActionKind.START_SERVER


def test_model_snapshot_manifest_and_live_files_are_reverified_on_resume(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    coordinator = _coordinator(tmp_path)
    lifecycle = FakeLifecycle()
    record = coordinator.create(config)
    imported = coordinator.store.artifact_ref(
        config.task.id, "artifacts/vllm-model-snapshot-manifest.json"
    )
    assert imported is not None
    assert coordinator.store.verify_artifact(config.task.id, imported)
    record = coordinator.record_inspection(
        record, _inspection(config, coordinator.store)
    )
    record = coordinator.start_server(record, lifecycle)

    (config.model.local_path / "config.json").write_text(
        '{"architectures":["ChangedForCausalLM"]}\n', encoding="utf-8"
    )
    with pytest.raises(VLLMWorkflowError, match="no longer matches"):
        coordinator.load(config.task.id)
    assert coordinator.next_action(record).kind == VLLMNextActionKind.STOP_BASELINE_SERVER
    record = coordinator.stop_baseline_server(record, lifecycle)
    assert record.active_gpu_started_at is None


def test_baseline_failure_forces_stop_before_retry(tmp_path: Path) -> None:
    config = _config(tmp_path)
    coordinator = _coordinator(tmp_path)
    lifecycle = FakeLifecycle()
    record = coordinator.create(config)
    record = coordinator.record_inspection(
        record, _inspection(config, coordinator.store)
    )
    record = coordinator.start_server(record, lifecycle)

    class FailingBenchmark:
        def capture_baseline(self, selected: VLLMCampaignConfig) -> BaselineResult:
            raise TimeoutError(f"benchmark timed out for {selected.task.id}")

    record = coordinator.capture_baseline(record, FailingBenchmark())
    assert record.baseline_run_failure is not None
    assert record.baseline_server_requires_stop is True
    assert coordinator.next_action(record).kind == VLLMNextActionKind.STOP_BASELINE_SERVER
    record = coordinator.stop_baseline_server(record, lifecycle)
    assert record.current_stage == VLLMStage.SERVER_START
    assert record.active_gpu_started_at is None
    assert record.gpu_seconds_by_phase["baseline_server"] >= 0


def test_paid_gpu_budget_forces_exact_stop_then_inconclusive(tmp_path: Path) -> None:
    base = _config(tmp_path)
    task = base.task.model_copy(
        update={"budgets": base.task.budgets.model_copy(update={"max_gpu_minutes": 1e-12})}
    )
    config = _config(tmp_path, task=task)
    coordinator = _coordinator(tmp_path)
    lifecycle = FakeLifecycle()
    record = coordinator.create(config)
    record = coordinator.record_inspection(
        record, _inspection(config, coordinator.store)
    )
    record = coordinator.start_server(record, lifecycle)
    action = coordinator.next_action(record)
    assert action.kind == VLLMNextActionKind.STOP_BASELINE_SERVER
    assert action.details["reason"] == "paid GPU time budget is exhausted"
    record = coordinator.stop_baseline_server(record, lifecycle)
    assert record.status == WorkflowStatus.INCONCLUSIVE
    assert record.active_gpu_started_at is None
    assert record.gpu_seconds_used > 0


def test_full_recoverable_workflow_accepts_only_after_exact_stops(tmp_path: Path) -> None:
    config = _config(tmp_path)
    coordinator = _coordinator(tmp_path)
    lifecycle = FakeLifecycle()
    record = coordinator.create(config)
    assert coordinator.run_until_pause(record.task_id).kind == (
        VLLMNextActionKind.PROVIDE_INSPECTION
    )

    record = coordinator.record_inspection(
        record, _inspection(config, coordinator.store)
    )
    record = coordinator.start_server(record, lifecycle)
    baseline_hash = record.baseline_server_request_hash
    assert baseline_hash is not None
    record = coordinator.record_baseline(
        record,
        BaselineResult(
            environment=_environment(config, "baseline-capture", baseline_hash),
            build_status=RunStatus.REUSED,
            smoke_passed=True,
            benchmark=BenchmarkResult(status=RunStatus.SUCCEEDED, metrics=_metrics()),
            quality=_quality(config),
        ),
    )
    assert coordinator.next_action(record).kind == VLLMNextActionKind.STOP_BASELINE_SERVER
    record = coordinator.stop_baseline_server(record, lifecycle)
    assert lifecycle.current is not None
    assert lifecycle.current.status == ServerExecutionStatus.STOPPED

    record, request = coordinator.request_profile_approval(
        record,
        resolved_mcp_environment={
            "ROCR_VISIBLE_DEVICES": UUID,
            "PYTHONNOUSERSITE": "1",
        },
    )
    assert request.arguments["command"] != config.profile.profile_argv
    assert record.profile_approval is not None
    approval_context = coordinator.store.load_json(
        record.task_id, record.profile_approval.context.path
    )
    materialized = approval_context["materialized_profile"]
    assert materialized["argv"] == request.arguments["command"]
    coordinate_ref = coordinator.store.artifact_ref(
        record.task_id, materialized["coordinate_artifact"]["path"]
    )
    assert coordinate_ref is not None
    assert coordinator.store.verify_artifact(record.task_id, coordinate_ref)
    record = coordinator.record_profile_approval(
        record,
        ApprovalReceipt(request_id=request.id, request_hash=request.request_hash),
    )
    record, permit = coordinator.consume_profile_approval(record)
    with pytest.raises(VLLMWorkflowError, match="unconsumed approval"):
        coordinator.consume_profile_approval(record)
    with pytest.raises(VLLMWorkflowError, match="durable result or UNKNOWN"):
        coordinator.request_profile_approval(
            record,
            resolved_mcp_environment={
                "ROCR_VISIBLE_DEVICES": UUID,
                "PYTHONNOUSERSITE": "1",
            },
        )
    profile_evidence = _profile_evidence(coordinator, record, permit, config)
    tampered_completion = profile_evidence.model_copy(deep=True)
    tampered_completion.mcp_call["framework_evidence"][
        "offline_vllm_metrics"
    ]["num_requests"] = config.serving.num_prompts - 1
    with pytest.raises(VLLMWorkflowError, match="framework completion evidence"):
        coordinator.record_profile_result(record, tampered_completion)
    record = coordinator.record_profile_result(record, profile_evidence)
    assert record.current_stage == VLLMStage.AGENT_DECISION

    profile_ref = record.completions[-1].evidence["profile"]
    hypothesis = Hypothesis(
        id="hypothesis-1",
        observed_evidence_ids=[profile_ref.path],
        interpretation="scheduler capacity is conservative",
        proposed_change="increase max-num-seqs",
        expected_performance_signature="higher output throughput",
        expected_e2e_effect="at least five percent",
        required_validation=["bench", "quality"],
        stop_condition="quality regression",
    )
    change = ChangeSet(
        kind=ChangeKind.RUNTIME_CONFIG,
        description="increase scheduler capacity",
        runtime_args=["--max-num-seqs", "64"],
    )
    experiment = VLLMExperimentSpec(
        id="candidate-1",
        task_id=config.task.id,
        hypothesis_id=hypothesis.id,
        description="max-num-seqs 64",
        change=change,
        server_argv=[*config.serving.server_argv, *change.runtime_args],
        benchmark_argv=config.serving.benchmark_argv,
        server_env=config.serving.server_env,
    )
    record = coordinator.record_agent_decision(
        record,
        VLLMAgentDecision(
            evidence_used=[profile_ref.path],
            conclusion="run one allow-listed scheduler experiment",
            confidence=0.8,
            hypothesis=hypothesis,
            proposed_experiment=experiment,
        ),
    )
    record = coordinator.start_candidate_server(record, lifecycle)
    candidate_hash = record.candidate_server_request_hash
    assert candidate_hash is not None and candidate_hash != baseline_hash
    record = coordinator.record_experiment(
        record,
        ExperimentResult(
            experiment_id=experiment.id,
            environment=_environment(config, "candidate-capture", candidate_hash),
            build_status=RunStatus.REUSED,
            smoke_passed=True,
            e2e=BenchmarkResult(status=RunStatus.SUCCEEDED, metrics=_metrics(1.1)),
        ),
    )
    record = coordinator.record_quality(record, _quality(config))
    assert coordinator.next_action(record).kind == (
        VLLMNextActionKind.STOP_EXPERIMENT_SERVER
    )
    record = coordinator.stop_candidate_server(record, lifecycle)
    assert record.current_stage == VLLMStage.DECIDE
    record = coordinator.decide(record)
    assert record.status == WorkflowStatus.ACCEPTED
    assert record.terminal_decision is not None
    assert record.terminal_decision.outcome == DecisionOutcome.ACCEPT


def test_profile_transport_failure_is_durable_and_receipt_is_not_replayed(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    coordinator = _coordinator(tmp_path)
    lifecycle = FakeLifecycle()
    record = coordinator.create(config)
    record = coordinator.record_inspection(
        record, _inspection(config, coordinator.store)
    )
    record = coordinator.start_server(record, lifecycle)
    request_hash = record.baseline_server_request_hash
    assert request_hash is not None
    record = coordinator.record_baseline(
        record,
        BaselineResult(
            environment=_environment(config, "baseline", request_hash),
            build_status=RunStatus.REUSED,
            smoke_passed=True,
            benchmark=BenchmarkResult(status=RunStatus.SUCCEEDED, metrics=_metrics()),
            quality=_quality(config),
        ),
    )
    record = coordinator.stop_baseline_server(record, lifecycle)
    record, request = coordinator.request_profile_approval(
        record,
        resolved_mcp_environment={
            "ROCR_VISIBLE_DEVICES": UUID,
            "PYTHONNOUSERSITE": "1",
        },
    )
    first_materialized_command = request.arguments["command"]
    record = coordinator.record_profile_approval(
        record,
        ApprovalReceipt(request_id=request.id, request_hash=request.request_hash),
    )

    class FailingProfile:
        def execute(self, permit: Any) -> Any:
            raise TimeoutError(f"unknown transport outcome for {permit.request_id}")

    record = coordinator.run_approved_profile(record, FailingProfile())
    assert record.status == WorkflowStatus.INCONCLUSIVE
    assert record.profile_execution_failure is not None
    assert coordinator.next_action(record).kind == VLLMNextActionKind.RESUME_INCONCLUSIVE
    record = coordinator.resume(record)
    record, retry = coordinator.request_profile_approval(
        record,
        resolved_mcp_environment={
            "ROCR_VISIBLE_DEVICES": UUID,
            "PYTHONNOUSERSITE": "1",
        },
    )
    assert retry.id != request.id
    assert retry.arguments["command"] != first_materialized_command
    assert record.profile_attempt_count == 2


def _ready_for_profile(coordinator: VLLMWorkflowCoordinator, config: VLLMCampaignConfig):
    lifecycle = FakeLifecycle()
    record = coordinator.create(config)
    record = coordinator.record_inspection(record, _inspection(config, coordinator.store))
    record = coordinator.start_server(record, lifecycle)
    record = coordinator.record_baseline(record, BaselineResult(
        environment=_environment(config, "baseline", record.baseline_server_request_hash),
        build_status=RunStatus.REUSED,
        smoke_passed=True,
        benchmark=BenchmarkResult(status=RunStatus.SUCCEEDED, metrics=_metrics()),
        quality=_quality(config),
    ))
    return coordinator.stop_baseline_server(record, lifecycle)


def test_profile_budget_negotiation_precedes_materialization_and_approval(tmp_path: Path) -> None:
    base = _config(tmp_path)
    config = _config(tmp_path, profile=base.profile.model_copy(update={
        "max_trace_bytes": 500_000_000,
    }))
    coordinator = _coordinator(tmp_path)
    record = _ready_for_profile(coordinator, config)
    materializations: list[Any] = []
    coordinator.profile_materializer = lambda **values: materializations.append(values)
    with pytest.raises(VLLMWorkflowError, match="max_trace_bytes.*200000000"):
        coordinator.request_profile_approval(record, resolved_mcp_environment={
            "ROCR_VISIBLE_DEVICES": UUID, "PYTHONNOUSERSITE": "1",
        })
    assert materializations == []
    saved = coordinator.load(record.task_id)
    assert saved.profile_attempt_count == 0
    assert saved.profile_approval is None
    assert saved.gpu_seconds_used == record.gpu_seconds_used


def test_profile_preflight_drift_preserves_receipt_and_can_request_new_approval(tmp_path: Path):
    config = _config(tmp_path)
    coordinator = _coordinator(tmp_path)
    record = _ready_for_profile(coordinator, config)
    environment = {"ROCR_VISIBLE_DEVICES": UUID, "PYTHONNOUSERSITE": "1"}
    record, request = coordinator.request_profile_approval(
        record, resolved_mcp_environment=environment
    )
    record = coordinator.record_profile_approval(record, ApprovalReceipt(
        request_id=request.id, request_hash=request.request_hash,
    ))
    receipt_ref = record.profile_approval.receipt
    context = coordinator.store.load_json(record.task_id, record.profile_approval.context.path)
    contract_ref = context["profile_contract_artifact"]
    assert coordinator.store.artifact_ref(record.task_id, contract_ref["path"]) is not None

    class ChangedProfile:
        def preflight(self, permit):
            assert permit.execution_context["profile_contract_sha256"]
            raise VLLMWorkflowError("ROCm profile contract changed after approval")

        def execute(self, permit):
            pytest.fail("capability failure must not execute a GPU workload")

    with pytest.raises(VLLMWorkflowError, match="changed after approval"):
        coordinator.run_approved_profile(record, ChangedProfile())
    assert coordinator.load(record.task_id).profile_approval.status.value == "APPROVED"
    receipt = coordinator.store.load_json(record.task_id, receipt_ref.path, ApprovalReceipt)
    assert receipt.consumed_at is None
    contract = coordinator.profile_contract_reader(None)
    contract.input_schema["properties"]["max_trace_bytes"]["anyOf"][0]["maximum"] = 300_000_000
    updated, refreshed = coordinator.request_profile_approval(
        record, resolved_mcp_environment=environment
    )
    assert refreshed.id != request.id
    assert updated.profile_attempt_count == 2
    assert coordinator.store.verify_artifact(record.task_id, receipt_ref)


@dataclass
class FakeCommand:
    stdout: str
    succeeded: bool = True


def test_local_benchmark_and_quality_ports_parse_strict_contracts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = ExperimentStore(tmp_path / "store")
    store.create_task(config.task)
    bench_payload = json.dumps(
        {
            "request_throughput": 10.0,
            "output_throughput": 100.0,
            "total_token_throughput": 120.0,
            "mean_ttft_ms": 20.0,
            "mean_tpot_ms": 5.0,
            "mean_itl_ms": 4.0,
            "completed": 16,
            "failed": 0,
        }
    )

    class Runner:
        def run(self, argv: list[str], **kwargs: Any) -> FakeCommand:
            del kwargs
            result_dir = Path(argv[argv.index("--result-dir") + 1])
            result_file = argv[argv.index("--result-filename") + 1]
            (result_dir / result_file).write_text(bench_payload, encoding="utf-8")
            return FakeCommand(stdout=bench_payload)

    class Environment:
        def capture(
            self,
            selected: VLLMCampaignConfig,
            *,
            phase: str,
            server_request_hash: str,
        ) -> EnvironmentFingerprint:
            return _environment(selected, phase, server_request_hash)

    quality = SimpleNamespace(evaluate_baseline=lambda selected: _quality(selected))
    port = LocalVLLMBenchmarkPort(store, Runner(), Environment(), quality)
    baseline = port.capture_baseline(config)
    assert baseline.benchmark.status == RunStatus.SUCCEEDED
    assert set(baseline.benchmark.metrics) == set(CANONICAL_METRICS)
    assert all(len(series.samples) == 3 for series in baseline.benchmark.metrics.values())

    class QualityRunner:
        def run(self, argv: list[str], **kwargs: Any) -> FakeCommand:
            del argv, kwargs
            config.quality_protocol.output_path.write_text(
                _quality(config).model_dump_json(), encoding="utf-8"
            )
            return FakeCommand(stdout="")

    quality_port = LocalVLLMQualityPort(store, QualityRunner())
    result = quality_port.evaluate_baseline(config)
    assert result.correctness_passed is True
    assert len(result.artifacts) == 2


def _config_with_probe_hashes(tmp_path: Path) -> VLLMCampaignConfig:
    config = _config(tmp_path)
    runtime = config.task.runtime
    static_argv = [
        runtime.executable,
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "static",
        "--uuid",
        config.device.device_uuid,
    ]
    hip_argv = [
        runtime.executable,
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "hip",
    ]
    raw = config.model_dump(mode="json", by_alias=True)
    raw["device"]["amd_smi_command_sha256"] = observation_request_sha256(
        static_argv,
        cwd=config.serving.cwd,
        device_uuid=config.device.device_uuid,
        timeout_seconds=60.0,
    )
    raw["device"]["hip_probe_command_sha256"] = observation_request_sha256(
        hip_argv,
        cwd=config.serving.cwd,
        device_uuid=config.device.device_uuid,
        timeout_seconds=60.0,
    )
    return VLLMCampaignConfig.model_validate(raw)


def test_profile_protocol_requires_official_isolated_vllm_throughput_entry(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    assert config.profile.profile_argv[:6] == [
        config.task.runtime.executable,
        "-I",
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "throughput",
    ]
    assert "--max-concurrency" not in config.profile.profile_argv
    raw = config.model_dump(mode="json", by_alias=True)
    raw["profile"]["profile_argv"][3] = "attacker.fake_vllm"
    with pytest.raises(ValidationError, match="official vLLM"):
        VLLMCampaignConfig.model_validate(raw)


def test_server_and_benchmark_require_isolated_official_python_entries(
    tmp_path: Path,
) -> None:
    def sync_protocol_hash(raw: dict[str, Any]) -> None:
        raw["task"]["benchmark"]["protocol_hash"] = (
            VLLMServingProtocol.model_validate(raw["serving"]).coordinate_sha256
        )

    config = _config(tmp_path)
    raw_server = config.model_dump(mode="json", by_alias=True)
    raw_server["serving"]["server_argv"].remove("-I")
    sync_protocol_hash(raw_server)
    with pytest.raises(ValidationError, match="isolated Python"):
        VLLMCampaignConfig.model_validate(raw_server)

    raw_benchmark = config.model_dump(mode="json", by_alias=True)
    raw_benchmark["serving"]["benchmark_argv"] = [
        "/opt/venv/bin/vllm",
        "bench",
        "serve",
        *raw_benchmark["serving"]["benchmark_argv"][7:],
    ]
    raw_benchmark["task"]["benchmark"]["benchmark_command"] = raw_benchmark[
        "serving"
    ]["benchmark_argv"]
    sync_protocol_hash(raw_benchmark)
    with pytest.raises(ValidationError, match="official vLLM"):
        VLLMCampaignConfig.model_validate(raw_benchmark)

    raw_environment = config.model_dump(mode="json", by_alias=True)
    raw_environment["serving"]["server_env"].pop("PYTHONNOUSERSITE")
    sync_protocol_hash(raw_environment)
    with pytest.raises(ValidationError, match="PYTHONNOUSERSITE"):
        VLLMCampaignConfig.model_validate(raw_environment)


def test_vllm_example_is_schema_valid_and_protocol_hash_bound() -> None:
    project_root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (project_root / "examples" / "vllm-mi300x-task.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = VLLMCampaignConfig.model_validate(payload)
    assert config.task.benchmark.protocol_hash == config.serving.coordinate_sha256


def test_vllm_cli_is_non_executing_and_generic_resume_dispatches_next(
    tmp_path: Path,
) -> None:
    config = _config_with_probe_hashes(tmp_path)
    config_path = tmp_path / "vllm.json"
    config_path.write_text(config.model_dump_json(by_alias=True), encoding="utf-8")
    store_root = tmp_path / "store"

    plan = CLI_RUNNER.invoke(
        app,
        ["vllm", "probe-plan", "--config", str(config_path)],
    )
    assert plan.exit_code == 0, plan.output
    plan_payload = json.loads(plan.stdout)
    assert plan_payload["accepted"] is True
    assert plan_payload["execution_started"] is False
    assert {item["id"] for item in plan_payload["preconditions"]} == {
        "rocm_minimum",
        "deployment_observability",
        "rocr_uuid_selector",
    }

    created = CLI_RUNNER.invoke(
        app,
        [
            "vllm",
            "create",
            "--config",
            str(config_path),
            "--store",
            str(store_root),
        ],
    )
    assert created.exit_code == 0, created.output
    created_payload = json.loads(created.stdout)
    assert created_payload["next_action"]["kind"] == "PROVIDE_INSPECTION"
    assert created_payload["execution"]["gpu_execution_started"] is False

    status = CLI_RUNNER.invoke(
        app, ["vllm", "status", config.id, "--store", str(store_root)]
    )
    assert status.exit_code == 0, status.output
    assert json.loads(status.stdout)["record"]["revision"] == 0

    next_action = CLI_RUNNER.invoke(
        app, ["vllm", "next", config.id, "--store", str(store_root)]
    )
    assert next_action.exit_code == 0, next_action.output
    next_payload = json.loads(next_action.stdout)
    assert "record" not in next_payload
    assert next_payload["execution"]["can_execute_next_action"] is False

    resumed = CLI_RUNNER.invoke(
        app, ["resume", config.id, "--store", str(store_root)]
    )
    assert resumed.exit_code == 0, resumed.output
    resume_payload = json.loads(resumed.stdout)
    assert resume_payload["next_action"]["kind"] == "PROVIDE_INSPECTION"
    assert resume_payload["execution"]["next_action_executed"] is False
    assert VLLMWorkflowCoordinator(ExperimentStore(store_root)).load(config.id).revision == 0


def test_vllm_probe_plan_rejects_hash_drift_without_execution(tmp_path: Path) -> None:
    config = _config_with_probe_hashes(tmp_path)
    raw = config.model_dump(mode="json", by_alias=True)
    raw["device"]["hip_probe_command_sha256"] = "f" * 64
    config_path = tmp_path / "mismatch.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    result = CLI_RUNNER.invoke(
        app,
        ["vllm", "probe-plan", "--config", str(config_path)],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["accepted"] is False
    assert payload["execution_started"] is False
    hip = next(item for item in payload["commands"] if item["name"] == "hip_logical_device")
    assert hip["hash_matches"] is False
    assert "no probe was executed" in result.stderr
