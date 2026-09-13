from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import amd_inference_opt.vllm_ports as ports_module
from amd_inference_opt.models import (
    EnvironmentFingerprint,
    QualityResult,
    RunStatus,
    utc_now,
)
from amd_inference_opt.rocm_mcp import (
    MCPToolCall,
    RocmIssueAgentClient,
    approval_request,
)
from amd_inference_opt.store import ExperimentStore
from amd_inference_opt.vllm_environment import capture_vllm_environment
from amd_inference_opt.vllm_model_snapshot import capture_vllm_model_snapshot
from amd_inference_opt.vllm_models import (
    MI300XDeviceBinding,
    VLLMModelCoordinate,
    VLLMProfileExecutionPermit,
    VLLMServingProtocol,
    canonical_sha256,
)
from amd_inference_opt.vllm_ports import (
    CommandVLLMBenchmarkPort,
    CommandVLLMQualityPort,
    VLLMEnvironmentCapture,
    VLLMPortError,
    VLLMRocmProfilePort,
    materialize_bench_serve_argv,
    materialize_vllm_profile_command,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
REVISION = "d" * 40
TOKENIZER_REVISION = "e" * 40
UUID = "GPU-deadbeef"
CANONICAL_METRICS = [
    "request_throughput_requests_per_second",
    "output_throughput_tokens_per_second",
    "total_throughput_tokens_per_second",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
]


def _empty_store(tmp_path: Path, task_id: str = "vllm-port-test") -> ExperimentStore:
    store = ExperimentStore(tmp_path / "store")
    task = store.task_dir(task_id)
    task.mkdir()
    for name in ("state", "events", "artifacts", "experiments", "workspaces", "reports"):
        (task / name).mkdir()
    return store


def _write_executable(path: Path, content: bytes = b"#!/bin/false\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o755)
    return path


@dataclass
class _Result:
    argv: tuple[str, ...]
    succeeded: bool = True
    stdout: str = "ignored human output"
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "succeeded": self.succeeded,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class _BenchRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: Any, **kwargs: Any) -> _Result:
        del kwargs
        frozen = tuple(argv)
        self.calls.append(frozen)
        result_dir = Path(frozen[frozen.index("--result-dir") + 1])
        filename = frozen[frozen.index("--result-filename") + 1]
        index = len(self.calls)
        payload = {
            "request_throughput": 10.0 + index,
            "output_throughput": 100.0 + index,
            "total_token_throughput": 120.0 + index,
            "mean_ttft_ms": 12.0 + index,
            "mean_tpot_ms": 4.0 + index,
            "mean_itl_ms": 3.0 + index,
            "completed": 8,
            "failed": 0,
        }
        (result_dir / filename).write_text(json.dumps(payload), encoding="utf-8")
        return _Result(frozen)


class _Telemetry:
    def __init__(self, store: ExperimentStore, values: dict[str, str]) -> None:
        self.store = store
        self.values = values
        self.counter = 0

    def begin(self, config: Any, *, phase: str, server_request_hash: str) -> str:
        del config, server_request_hash
        return phase

    def finish(
        self,
        config: Any,
        *,
        phase: str,
        server_request_hash: str,
        before: Any,
    ) -> VLLMEnvironmentCapture:
        assert before == phase
        self.counter += 1
        pre = self.store.save_evidence_json(
            config.task.id,
            f"telemetry/{phase}-pre",
            {"phase": "before", "clock_mhz": 1600, "partition": "SPX:NPS1"},
            producer="test-amd-smi",
        )
        post = self.store.save_evidence_json(
            config.task.id,
            f"telemetry/{phase}-post",
            {
                "phase": "after",
                "clock_mhz": 1600,
                "temperature_c": 55,
                "power_w": 300,
                "throttle": "UNTHROTTLED",
                "partition": "SPX:NPS1",
            },
            producer="test-amd-smi",
        )
        return VLLMEnvironmentCapture(
            fingerprint=EnvironmentFingerprint(
                values={
                    **self.values,
                    "server_request_hash": server_request_hash,
                    "telemetry_gfxclk_mean_mhz": "1600.0",
                    "telemetry_uclk_mean_mhz": "1200.0",
                },
                telemetry_stable=True,
                capture_id=f"capture-{self.counter}",
                captured_at=utc_now(),
                source="observed",
            ),
            before_artifacts=(pre,),
            after_artifacts=(post,),
        )


class _Quality:
    def evaluate_baseline(self, config: Any) -> QualityResult:
        return QualityResult(
            status=RunStatus.SUCCEEDED,
            correctness_passed=True,
            coordinate_hash=config.quality_protocol.coordinate_sha256,
            representation_hash=config.model.snapshot_digest,
        )


def _benchmark_config(tmp_path: Path, model_manifest: Any) -> Any:
    runtime_executable_sha, _ = _digest(Path(sys.executable).resolve())
    benchmark_argv = [
        str(Path(sys.executable).absolute()),
        "-I",
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--num-prompts",
        "8",
    ]
    environment_coordinates = {
        "gpu_gfx": "gfx942",
        "gpu_device_uuid": UUID,
        "model_snapshot_digest": model_manifest.snapshot_digest,
    }
    serving = SimpleNamespace(
        benchmark_argv=benchmark_argv,
        cwd=tmp_path,
        server_env={"ROCR_VISIBLE_DEVICES": UUID, "PYTHONNOUSERSITE": "1"},
        timeout_seconds=30,
        warmup_runs=1,
        sample_count=2,
        num_prompts=8,
        required_metrics=CANONICAL_METRICS,
        coordinate_sha256=SHA_A,
        engine_config={"launcher_sha256": SHA_B},
    )
    runtime = SimpleNamespace(
        executable=str(Path(sys.executable).absolute()),
        executable_sha256=runtime_executable_sha,
        environment_manifest_sha256="1" * 64,
        image_digest="2" * 64,
    )
    model = SimpleNamespace(
        local_path=model_manifest.root,
        model_id=model_manifest.model_id,
        revision=model_manifest.revision,
        tokenizer_revision=model_manifest.tokenizer_revision,
        snapshot_digest=model_manifest.snapshot_digest,
    )
    return SimpleNamespace(
        task=SimpleNamespace(
            id="vllm-port-test",
            runtime=runtime,
            environment=SimpleNamespace(require_stable_telemetry=True),
        ),
        device=SimpleNamespace(device_uuid=UUID),
        model=model,
        serving=serving,
        quality_protocol=SimpleNamespace(coordinate_sha256="3" * 64),
        environment_coordinates=environment_coordinates,
    )


def test_materializer_owns_unique_result_file_and_preserves_native_warmups(
    tmp_path: Path,
) -> None:
    executable = _write_executable(tmp_path / "vllm")
    base = [str(executable), "bench", "serve"]

    argv = materialize_bench_serve_argv(
        base,
        result_dir=tmp_path,
        result_filename="run-0001.json",
    )

    assert argv[-4:] == (
        "--result-dir",
        str(tmp_path.resolve()),
        "--result-filename",
        "run-0001.json",
    )
    assert argv.count("--save-result") == 1
    with pytest.raises(VLLMPortError, match="coordinator-owned"):
        materialize_bench_serve_argv(
            [*base, "--result-filename", "stale.json"],
            result_dir=tmp_path,
            result_filename="fresh.json",
        )
    warmed = materialize_bench_serve_argv(
        [*base, "--num-warmups", "32"],
        result_dir=tmp_path,
        result_filename="fresh.json",
    )
    assert warmed[:len(base) + 2] == (*base, "--num-warmups", "32")


@pytest.mark.parametrize("warmup_args", [
    ["--num-warmups"], ["--num-warmups", "-1"], ["--num-warmups=nan"],
    ["--num-warmups", "1", "--num-warmups=2"],
])
def test_materializer_rejects_ambiguous_native_warmups(
    tmp_path: Path, warmup_args: list[str],
) -> None:
    with pytest.raises(VLLMPortError, match="num-warmups"):
        materialize_bench_serve_argv(
            ["vllm", "bench", "serve", *warmup_args],
            result_dir=tmp_path, result_filename="fresh.json",
        )


def test_captured_manifests_materialize_and_launch_exact_offline_process(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "model.safetensors").write_bytes(b"weights")
    model_manifest = capture_vllm_model_snapshot(
        model_root,
        model_id="org/model",
        revision=REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
    )
    model_manifest_path = tmp_path / "model-manifest.json"
    model_manifest_path.write_text(model_manifest.model_dump_json(by_alias=True), encoding="utf-8")
    environment_manifest = capture_vllm_environment(
        python_executable=sys.executable,
        required_distributions=("pydantic",),
        optional_distributions=(),
    )
    environment_manifest_path = tmp_path / "environment-manifest.json"
    environment_manifest_path.write_text(
        environment_manifest.model_dump_json(by_alias=True), encoding="utf-8"
    )
    model = VLLMModelCoordinate(
        model_id="org/model",
        local_path=model_root,
        snapshot_manifest_path=model_manifest_path.resolve(),
        revision=REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        snapshot_digest=model_manifest.snapshot_digest,
    )
    device = MI300XDeviceBinding(
        device_ids=[1],
        oam_id=1,
        device_uuid=UUID,
        pci_bdf="0000:41:00.0",
        partition_id=0,
        amd_smi_command_sha256=SHA_A,
        hip_probe_command_sha256=SHA_B,
    )
    serving = VLLMServingProtocol(
        server_argv=[sys.executable, "-m", "server"],
        benchmark_argv=[
            sys.executable,
            "-m",
            "bench",
            "bench",
            "serve",
        ],
        cwd=tmp_path,
        server_env={"ROCR_VISIBLE_DEVICES": UUID, "PYTHONNOUSERSITE": "1"},
        dtype="bfloat16",
        concurrency=2,
        input_tokens=16,
        output_tokens=4,
        num_prompts=4,
        seed=42,
        warmup_runs=0,
        sample_count=1,
        required_metrics=CANONICAL_METRICS,
        engine_config={"launcher_sha256": SHA_C},
    )
    offline = [
        str(Path(sys.executable).absolute()),
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
        str(model_root),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--input-len",
        "16",
        "--output-len",
        "4",
        "--num-prompts",
        "4",
        "--seed",
        "42",
    ]
    launcher = Path(__file__).parents[1] / "tools" / "vllm_profile_launcher.py"

    materialized = materialize_vllm_profile_command(
        python_executable=Path(sys.executable).absolute(),
        launcher_path=launcher.resolve(),
        environment_manifest_path=environment_manifest_path.resolve(),
        model_manifest_path=model_manifest_path.resolve(),
        model=model,
        device=device,
        serving=serving,
        offline_argv=offline,
        output_root=(tmp_path / "profiles").resolve(),
        attempt_id="attempt-0001",
    )
    spec = importlib.util.spec_from_file_location("test_vllm_profile_launcher", launcher)
    assert spec is not None and spec.loader is not None
    launcher_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher_module)
    executed: dict[str, Any] = {}

    class ExecCalled(Exception):
        pass

    def fake_execve(path: str, argv: list[str], env: dict[str, str]) -> None:
        executed.update(path=path, argv=argv, env=env)
        raise ExecCalled

    original_argv = sys.argv
    original_rocr = os.environ.get("ROCR_VISIBLE_DEVICES")
    original_no_user_site = os.environ.get("PYTHONNOUSERSITE")
    conflicting = {
        name: os.environ.pop(name, None)
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
    }
    os.environ["ROCR_VISIBLE_DEVICES"] = UUID
    os.environ["PYTHONNOUSERSITE"] = "1"
    assert materialized.argv[1] == "-I"
    sys.argv = list(materialized.argv[2:])
    original_execve = launcher_module.os.execve
    launcher_module.os.execve = fake_execve
    try:
        with pytest.raises(ExecCalled):
            launcher_module.main()
    finally:
        launcher_module.os.execve = original_execve
        sys.argv = original_argv
        if original_rocr is None:
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        else:
            os.environ["ROCR_VISIBLE_DEVICES"] = original_rocr
        if original_no_user_site is None:
            os.environ.pop("PYTHONNOUSERSITE", None)
        else:
            os.environ["PYTHONNOUSERSITE"] = original_no_user_site
        for name, value in conflicting.items():
            if value is not None:
                os.environ[name] = value

    delimiter = materialized.argv.index("--")
    assert executed["path"] == offline[0]
    assert executed["argv"] == list(materialized.argv[delimiter + 1 :])
    sidecar = json.loads(materialized.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["schema"] == "gpuopt.vllm-profile-launch.v1"
    assert sidecar["model_snapshot_digest"] == model.snapshot_digest
    assert sidecar["selected_environment"]["ROCR_VISIBLE_DEVICES"] == UUID
    assert sidecar["native_executable_sha256"] == (environment_manifest.python_executable_sha256)
    assert sidecar["throughput_result_path"] == str(materialized.throughput_result_path)
    assert not materialized.throughput_result_path.exists()
    with pytest.raises(VLLMPortError, match="bench throughput"):
        materialize_vllm_profile_command(
            python_executable=Path(sys.executable).absolute(),
            launcher_path=launcher.resolve(),
            environment_manifest_path=environment_manifest_path.resolve(),
            model_manifest_path=model_manifest_path.resolve(),
            model=model,
            device=device,
            serving=serving,
            offline_argv=[sys.executable, "-c", "raise SystemExit(0)"],
            output_root=(tmp_path / "profiles").resolve(),
            attempt_id="attempt-marker-rejected",
        )
    with pytest.raises(VLLMPortError, match="already exists"):
        materialize_vllm_profile_command(
            python_executable=Path(sys.executable).absolute(),
            launcher_path=launcher.resolve(),
            environment_manifest_path=environment_manifest_path.resolve(),
            model_manifest_path=model_manifest_path.resolve(),
            model=model,
            device=device,
            serving=serving,
            offline_argv=offline,
            output_root=(tmp_path / "profiles").resolve(),
            attempt_id="attempt-0001",
        )


def test_benchmark_port_reads_each_result_file_and_maps_six_canonical_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _empty_store(tmp_path)
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "model.safetensors").write_bytes(b"weights")
    manifest = capture_vllm_model_snapshot(
        model_root,
        model_id="org/model",
        revision=REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
    )
    config = _benchmark_config(tmp_path, manifest)
    telemetry = _Telemetry(store, config.environment_coordinates)
    runner = _BenchRunner()
    monkeypatch.setattr(
        ports_module,
        "build_server_spec",
        lambda *args, **kwargs: SimpleNamespace(request_hash="4" * 64),
    )
    port = CommandVLLMBenchmarkPort(
        store,
        telemetry,
        _Quality(),  # type: ignore[arg-type]
        model_snapshot_manifest=manifest,
        runner=runner,
    )

    baseline = port.capture_baseline(config)

    assert len(runner.calls) == 3  # one warmup plus two measured repetitions
    assert set(baseline.benchmark.metrics) == set(CANONICAL_METRICS)
    assert baseline.benchmark.metrics["output_throughput_tokens_per_second"].samples == [
        102.0,
        103.0,
    ]
    assert baseline.run_identity is not None
    assert baseline.run_identity.model_sha256 == manifest.snapshot_digest
    assert baseline.environment.telemetry_stable is True
    raw_results = [item for item in baseline.artifacts if "result" in item.path]
    assert len(raw_results) == 3
    assert all(store.verify_artifact(config.task.id, item) for item in baseline.artifacts)

    (model_root / "model.safetensors").write_bytes(b"mutated")
    with pytest.raises(Exception, match="snapshot"):
        port.capture_baseline(config)


def test_benchmark_rejects_nonisolated_console_script_even_with_pinned_shebang(
    tmp_path: Path,
) -> None:
    python = str(Path(sys.executable).absolute())
    python_sha, _ = _digest(Path(sys.executable).resolve())
    console = _write_executable(
        tmp_path / "bin" / "vllm",
        f"#!{python}\nraise SystemExit(0)\n".encode(),
    )

    with pytest.raises(VLLMPortError, match="pinned lexical venv Python"):
        ports_module._benchmark_launcher_identity(
            [str(console), "bench", "serve"],
            runtime_executable=python,
            runtime_executable_sha256=python_sha,
        )


class _QualityRunner:
    def __init__(self, output: Path, protocol_sha: str, snapshot_sha: str) -> None:
        self.output = output
        self.protocol_sha = protocol_sha
        self.snapshot_sha = snapshot_sha

    def run(self, argv: Any, **kwargs: Any) -> _Result:
        del kwargs
        temporary = self.output.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "status": "SUCCEEDED",
                    "correctness_passed": True,
                    "perplexity": 8.5,
                    "accuracies": {"math_accuracy": 0.8},
                    "coordinate_hash": self.protocol_sha,
                    "representation_hash": self.snapshot_sha,
                }
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.output)
        return _Result(tuple(argv))


def test_quality_port_binds_executable_protocol_and_model_snapshot(tmp_path: Path) -> None:
    store = _empty_store(tmp_path)
    executable = _write_executable(tmp_path / "bin" / "quality")
    output = tmp_path / "quality.json"
    protocol_sha = "5" * 64
    snapshot_sha = "6" * 64
    protocol = SimpleNamespace(
        argv=[str(executable), "--suite", "general-100"],
        cwd=tmp_path,
        env={},
        timeout_seconds=30,
        output_path=output,
        coordinate_sha256=protocol_sha,
    )
    config = SimpleNamespace(
        task=SimpleNamespace(
            id="vllm-port-test",
            runtime=SimpleNamespace(environment_manifest_sha256="7" * 64),
        ),
        quality_protocol=protocol,
        model=SimpleNamespace(snapshot_digest=snapshot_sha),
    )
    port = CommandVLLMQualityPort(
        store,
        _QualityRunner(output, protocol_sha, snapshot_sha),
    )

    baseline = port.evaluate_baseline(config)

    assert baseline.status == RunStatus.SUCCEEDED
    assert baseline.coordinate_hash == protocol_sha
    assert baseline.representation_hash == snapshot_sha
    executable.write_bytes(b"changed executable")
    with pytest.raises(VLLMPortError, match="differs from baseline"):
        port.evaluate(config, SimpleNamespace(id="candidate"))


class _FakeMCPClient:
    def __init__(self, call: MCPToolCall) -> None:
        self.call = call

    async def call_tool(
        self, tool: str, arguments: dict[str, Any], approval_sha256: str
    ) -> MCPToolCall:
        assert tool == self.call.tool_name
        assert arguments == self.call.arguments
        assert approval_sha256
        return self.call


class _FakeClientContext:
    def __init__(self, call: MCPToolCall) -> None:
        self.client = _FakeMCPClient(call)

    async def __aenter__(self) -> _FakeMCPClient:
        return self.client

    async def __aexit__(self, *args: Any) -> None:
        return None


def _digest(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def test_profile_port_copies_and_verifies_case_artifacts_before_returning(
    tmp_path: Path,
) -> None:
    store = _empty_store(tmp_path)
    agent = _write_executable(tmp_path / "agent" / "mcp")
    launcher = _write_executable(tmp_path / "launcher.py")
    launcher_sha, _ = _digest(launcher)
    sidecar = tmp_path / "profile-sidecar.json"
    coordinate_sha = "7" * 64
    model_manifest_sha = "8" * 64
    coordinate_path = tmp_path / "profile-coordinate.json"
    environment_manifest_path = tmp_path / "environment-manifest.json"
    model_manifest_path = tmp_path / "model-manifest.json"
    throughput_result_path = tmp_path / "vllm-throughput-result.json"
    offline_profile_argv = [
        str(Path(sys.executable).absolute()),
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
        str(tmp_path / "model"),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--input-len",
        "16",
        "--output-len",
        "4",
        "--num-prompts",
        "4",
        "--seed",
        "42",
    ]
    materialized_profile_argv = [
        str(Path(sys.executable).absolute()),
        "-I",
        str(launcher),
        "--gpuopt-environment-manifest",
        str(environment_manifest_path),
        "--gpuopt-environment-manifest-sha256",
        SHA_B,
        "--gpuopt-model-manifest",
        str(model_manifest_path),
        "--gpuopt-model-manifest-sha256",
        model_manifest_sha,
        "--gpuopt-coordinate",
        str(coordinate_path),
        "--gpuopt-coordinate-sha256",
        coordinate_sha,
        "--gpuopt-sidecar",
        str(sidecar),
        "--",
        *offline_profile_argv,
        "--output-json",
        str(throughput_result_path),
    ]
    runtime = SimpleNamespace(
        executable_sha256=SHA_A,
        environment_manifest_sha256=SHA_B,
        image_digest=None,
    )
    profile = SimpleNamespace(
        # The immutable config stores the base offline command.  The approved
        # permit stores a distinct, per-attempt outer launcher command.
        profile_argv=offline_profile_argv,
        cwd=tmp_path,
        preset="kernel-timing",
        max_trace_files=8,
        max_trace_bytes=1_000_000,
        command_sha256=SHA_C,
        launcher_sha256=launcher_sha,
    )
    device = SimpleNamespace(
        gfx_target="gfx942",
        device_uuid=UUID,
        pci_bdf="0000:41:00.0",
        partition_id=0,
        compute_partition="SPX",
        memory_partition="NPS1",
    )
    config = SimpleNamespace(
        task=SimpleNamespace(
            id="vllm-port-test",
            runtime=runtime,
            mcp=SimpleNamespace(command=[str(agent)], env={}, cwd=tmp_path),
        ),
        model=SimpleNamespace(
            snapshot_digest="9" * 64,
            revision=REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
        ),
        serving=SimpleNamespace(coordinate_sha256="1" * 64, num_prompts=4),
        profile=profile,
        device=device,
        identity_sha256="2" * 64,
    )
    sidecar.write_text(
        json.dumps(
            {
                "schema": "gpuopt.vllm-profile-launch.v1",
                "pid": 123,
                "boot_id": "boot-test",
                "start_ticks": 456,
                "native_executable_sha256": SHA_A,
                "environment_manifest_sha256": SHA_B,
                "profile_launcher_sha256": launcher_sha,
                "profile_coordinate_sha256": coordinate_sha,
                "model_manifest_sha256": model_manifest_sha,
                "model_snapshot_digest": "9" * 64,
                "model_revision": REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
                "serving_protocol_sha256": "1" * 64,
                "device_uuid": UUID,
                "pci_bdf": "0000:41:00.0",
                "partition_id": 0,
                "throughput_result_path": str(throughput_result_path),
                "offline_argv": [
                    *offline_profile_argv,
                    "--output-json",
                    str(throughput_result_path),
                ],
                "offline_argv_sha256": canonical_sha256(
                    [
                        *offline_profile_argv,
                        "--output-json",
                        str(throughput_result_path),
                    ]
                ),
                "selected_environment": {
                    "ROCR_VISIBLE_DEVICES": UUID,
                    "PYTHONNOUSERSITE": "1",
                    "HIP_VISIBLE_DEVICES": None,
                    "HSA_VISIBLE_DEVICES": None,
                    "CUDA_VISIBLE_DEVICES": None,
                    "GPU_DEVICE_ORDINAL": None,
                },
                "declared_environment_matches": True,
                "unset_environment_absent": True,
                "cgroup": {"kind": "cgroup_v2", "binding_id": "cgroup:/test"},
            }
        ),
        encoding="utf-8",
    )
    base_context = {
        "schema": "rocm.run-context.v1",
        # ROCm Issue Agent resolves only argv[0].  On a normal virtualenv this
        # differs from the lexical Python path bound by the approval.
        "argv": [
            str(Path(materialized_profile_argv[0]).resolve()),
            *materialized_profile_argv[1:],
        ],
        "cwd": str(tmp_path),
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
    run_hash = f"sha256:{canonical_sha256(base_context)}"
    run_context = {**base_context, "run_context_hash": run_hash}
    case_root = tmp_path / "cases"
    case = case_root / "CASE-TEST-001"
    (case / "raw" / "observation").mkdir(parents=True)
    context_path = case / "raw" / "observation" / "run-context.json"
    context_path.write_text(json.dumps(run_context), encoding="utf-8")
    trace = case / "raw" / "trace.csv"
    trace.write_text("Kernel_Name,Start_Timestamp,End_Timestamp\n", encoding="utf-8")
    entries = []
    for path in (context_path, trace):
        digest, size = _digest(path)
        entries.append(
            {
                "path": path.relative_to(case).as_posix(),
                "sha256": digest,
                "size_bytes": size,
                "content_type": "application/json" if path.suffix == ".json" else "text/csv",
                "ownership_scope": "case",
                "retention_policy": "local_case_lifetime",
            }
        )
    arguments = RocmIssueAgentClient.profile_arguments(
        materialized_profile_argv,
        preset="kernel-timing",
        cwd=tmp_path,
        timeout_seconds=900,
        max_trace_bytes=1_000_000,
        max_trace_files=8,
        max_events_per_type=10_000,
        max_percentile_samples_per_kernel=20,
    )
    environment = {"ROCR_VISIBLE_DEVICES": UUID, "PYTHONNOUSERSITE": "1"}
    execution_context = {
        "config_identity_sha256": config.identity_sha256,
        "profile_contract_sha256": "b" * 64,
        "mcp_env_sha256": canonical_sha256(environment),
        "profile_command_sha256": SHA_C,
        "materialized_profile_command_sha256": canonical_sha256(
            materialized_profile_argv
        ),
        "profile_launcher_sha256": launcher_sha,
        "environment_manifest_sha256": SHA_B,
        "serving_protocol_sha256": "1" * 64,
        "model_snapshot_digest": "9" * 64,
        "materialized_profile": {
            "argv": materialized_profile_argv,
            "coordinate_path": str(coordinate_path),
            "coordinate_sha256": coordinate_sha,
            "environment_manifest_path": str(environment_manifest_path),
            "environment_manifest_sha256": SHA_B,
            "model_manifest_path": str(model_manifest_path),
            "model_manifest_sha256": model_manifest_sha,
            "sidecar_path": str(sidecar),
            "throughput_result_path": str(throughput_result_path),
        },
    }
    exact = approval_request(
        "rocm_profile_workload", arguments, execution_context=execution_context
    )
    permit = VLLMProfileExecutionPermit(
        request_id="profile-test",
        request_hash=exact.request_sha256,
        arguments=arguments,
        execution_context=execution_context,
    )
    structured = {
        "schema": "rocm.mcp-kernel-evidence.v1",
        "status": "completed",
        "case_id": case.name,
        "run_context_hash": run_hash,
        "artifact_manifest": entries,
        "kernel_evidence": {
            "status": "completed",
            "preset": "kernel-timing",
            "run_context_hash": run_hash,
            "workload_succeeded": True,
            "aggregate_timing_complete": True,
            "workload": {"status": "completed", "exit_code": 0},
            "profiler": {"status": "completed", "tool": "rocprofv3"},
            "kernels": [{"name": "vllm::paged_attention", "dispatch_count": 8}],
        },
    }
    call = MCPToolCall(
        tool_name="rocm_profile_workload",
        arguments=arguments,
        called_at="2026-08-23T00:00:00+00:00",
        is_error=False,
        structured_content=structured,
        raw_result={"structuredContent": structured},
    )
    throughput_result_path.write_text(
        json.dumps(
            {
                "elapsed_time": 2.0,
                "num_requests": 4,
                "total_num_tokens": 80,
                "requests_per_second": 2.0,
                "tokens_per_second": 40.0,
            }
        ),
        encoding="utf-8",
    )
    port = VLLMRocmProfilePort(
        config,
        store,
        resolved_environment=environment,
        artifact_roots=[case_root],
        client_factory=lambda server, context: _FakeClientContext(call),
    )

    result = port.execute(permit)

    imported = result.mcp_call["framework_evidence"]["imported_raw_artifacts"]
    assert len(imported) == 2
    assert result.mcp_call["framework_evidence"]["offline_vllm_metrics"] == {
        "num_requests": 4,
        "total_num_tokens": 80,
        "elapsed_time": 2.0,
        "requests_per_second": 2.0,
        "tokens_per_second": 40.0,
    }
    assert result.engine_kernel_count == 1
    assert store.verify_artifact(config.task.id, result.execution_environment_artifact)
    enriched = store.load_json(config.task.id, result.execution_environment_artifact.path)
    assert enriched["vllm_execution"]["model_snapshot_digest"] == "9" * 64


def test_profile_port_fails_inconclusive_when_case_artifacts_are_not_accessible(
    tmp_path: Path,
) -> None:
    # This narrow regression verifies the P0 policy without constructing a
    # second complete profile fixture: the import helper must retain the exact
    # manifest reference and return a warning, never manufacture bytes.
    store = _empty_store(tmp_path)
    config = SimpleNamespace(
        task=SimpleNamespace(id="vllm-port-test"),
        profile=SimpleNamespace(max_trace_files=4, max_trace_bytes=1024),
    )
    port = object.__new__(VLLMRocmProfilePort)
    port.config = config
    port.store = store
    port.artifact_roots = ()
    permit = SimpleNamespace(request_id="missing")
    structured = {
        "case_id": "CASE-MISSING-001",
        "artifact_manifest": [
            {
                "path": "raw/trace.csv",
                "sha256": "f" * 64,
                "size_bytes": 12,
                "content_type": "text/csv",
                "ownership_scope": "case",
                "retention_policy": "local_case_lifetime",
            }
        ],
    }

    result = port._import_agent_artifacts(structured, permit)

    assert not result.references
    assert result.retention_refs[0]["reason"] == "case_artifact_not_locally_accessible"
    assert result.warnings
