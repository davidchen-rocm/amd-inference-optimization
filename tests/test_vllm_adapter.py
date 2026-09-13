from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from amd_inference_opt.vllm_adapter import (
    CANONICAL_BENCH_METRIC_NAMES,
    MEAN_E2EL_METRIC,
    HTTPJSONResponse,
    RuntimeVerificationStatus,
    ServerExecutionStatus,
    SpawnedProcess,
    StdlibHTTPClient,
    VLLMAdapter,
    VLLMAdapterError,
    VLLMBenchmarkParseError,
    VLLMHealthTimeout,
    VLLMIdentityMismatchError,
    VLLMRuntimeEvidence,
    VLLMServerConflictError,
    VLLMServerSpec,
    VLLMServerStartError,
    VLLMSpecError,
    build_bench_serve_argv,
    canonical_sha256,
    load_execution_record,
    parse_bench_serve_json,
)

NATIVE_SHA = "a" * 64
SNAPSHOT_SHA = "b" * 64
IMAGE_SHA = "c" * 64
ENVIRONMENT_MANIFEST_SHA = "d" * 64


def _spec(tmp_path: Path, **overrides: Any) -> VLLMServerSpec:
    values: dict[str, Any] = {
        "argv": (
            "/opt/venv/bin/python",
            "-I",
            "-m",
            "vllm.entrypoints.openai.api_server",
        ),
        "vllm_version": "0.10.1+rocm",
        "model": "org/model",
        "model_revision": "deadbeef",
        "model_snapshot_sha256": SNAPSHOT_SHA,
        "expected_served_model": "served-model",
        "native_executable_sha256": NATIVE_SHA,
        "environment_manifest_sha256": ENVIRONMENT_MANIFEST_SHA,
        "tensor_parallel_size": 2,
        "dtype": "float16",
        "quantization": "fp8",
        "config": {"max_num_seqs": 64},
        "cwd": str(tmp_path),
        "env": {
            "PYTHONNOUSERSITE": "1",
            "ROCR_VISIBLE_DEVICES": "GPU-uuid",
        },
        "port": 8123,
        "startup_timeout_seconds": 0.25,
        "request_timeout_seconds": 0.05,
        "shutdown_timeout_seconds": 7.5,
        "poll_interval_seconds": 0.1,
    }
    values.update(overrides)
    return VLLMServerSpec(**values)


class FakeProcessBackend:
    def __init__(self) -> None:
        self.current_boot_id = "boot-fixture"
        self.alive: dict[int, int] = {}
        self.spawn_calls: list[dict[str, Any]] = []
        self.terminate_calls: list[dict[str, Any]] = []
        self.next_pid = 4100
        self.next_ticks = 9000

    def boot_id(self) -> str:
        return self.current_boot_id

    def start_ticks(self, pid: int) -> int | None:
        return self.alive.get(pid)

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        unset_env: Sequence[str],
        stdout_path: str | None,
        stderr_path: str | None,
    ) -> SpawnedProcess:
        call = {
            "argv": tuple(argv),
            "cwd": cwd,
            "env": dict(env),
            "unset_env": tuple(unset_env),
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
        }
        self.spawn_calls.append(call)
        pid = self.next_pid
        ticks = self.next_ticks
        self.next_pid += 1
        self.next_ticks += 10
        self.alive[pid] = ticks
        return SpawnedProcess(pid=pid, boot_id=self.current_boot_id, start_ticks=ticks)

    def terminate_exact(
        self,
        *,
        pid: int,
        boot_id: str,
        start_ticks: int,
        timeout_seconds: float,
    ) -> bool:
        call = {
            "pid": pid,
            "boot_id": boot_id,
            "start_ticks": start_ticks,
            "timeout_seconds": timeout_seconds,
        }
        self.terminate_calls.append(call)
        if boot_id != self.current_boot_id or self.alive.get(pid) != start_ticks:
            return False
        self.alive.pop(pid)
        return True


class FakeRuntimeInspector:
    def __init__(
        self,
        *,
        verification: RuntimeVerificationStatus = RuntimeVerificationStatus.VERIFIED,
        executable_sha: str = NATIVE_SHA,
        bind_outer_image: bool = True,
        binding_kind: str = "cgroup_v2",
        observed_environment_manifest_sha: str | None = ENVIRONMENT_MANIFEST_SHA,
        leaked_unset_environment: str | None = None,
    ) -> None:
        self.verification = verification
        self.executable_sha = executable_sha
        self.bind_outer_image = bind_outer_image
        self.binding_kind = binding_kind
        self.observed_environment_manifest_sha = observed_environment_manifest_sha
        self.leaked_unset_environment = leaked_unset_environment
        self.calls = 0

    def inspect(
        self,
        spec: VLLMServerSpec,
        process: SpawnedProcess,
    ) -> VLLMRuntimeEvidence:
        self.calls += 1
        outer: dict[str, Any] = {}
        if spec.image_digest is not None and self.bind_outer_image:
            outer = {
                "container_id": "container-123",
                "container_init_pid": 1,
                "container_image_id": f"sha256:{IMAGE_SHA}",
                "container_binding_kind": self.binding_kind,
                "process_binding_id": "cgroup:/docker/container-123",
                "container_binding_id": "cgroup:/docker/container-123",
                "container_repo_digests": (
                    f"registry.invalid/vllm@sha256:{spec.identity.image_sha256}",
                ),
                "image_digest_matches": True,
            }
        observed_unset = [key for key in spec.unset_env if key != self.leaked_unset_environment]
        return VLLMRuntimeEvidence(
            verification=self.verification,
            captured_at="2026-01-01T00:00:00+00:00",
            observed_pid=process.pid,
            observed_boot_id=process.boot_id,
            observed_start_ticks=process.start_ticks,
            pid_executable_path=spec.argv[0],
            pid_executable_sha256=self.executable_sha,
            native_executable_matches=(self.executable_sha == spec.native_executable_sha256),
            observed_environment_manifest_sha256=self.observed_environment_manifest_sha,
            process_environment_sha256=canonical_sha256(
                {
                    **dict(spec.env),
                    "PATH": "/fixture",
                    **(
                        {self.leaked_unset_environment: "leaked"}
                        if self.leaked_unset_environment
                        else {}
                    ),
                }
            ),
            declared_environment_sha256=canonical_sha256(
                {"set": dict(spec.env), "unset": observed_unset}
            ),
            declared_environment_matches=self.leaked_unset_environment is None,
            unset_environment_absent=self.leaked_unset_environment is None,
            unexpected_inherited_environment=(
                (self.leaked_unset_environment,)
                if self.leaked_unset_environment is not None
                else ()
            ),
            rocr_visible_devices=spec.env.get("ROCR_VISIBLE_DEVICES"),
            hip_visible_devices=spec.env.get("HIP_VISIBLE_DEVICES"),
            **outer,
        )


class FailingPreflightRuntimeInspector(FakeRuntimeInspector):
    def preflight(self, spec: VLLMServerSpec) -> None:
        del spec
        raise RuntimeError("package bytes changed")


class FakeHTTPClient:
    def __init__(self, values: Sequence[HTTPJSONResponse | Exception]) -> None:
        self.values = list(values)
        self.calls = 0

    def get_json(self, url: str, *, timeout_seconds: float) -> HTTPJSONResponse:
        del url, timeout_seconds
        index = min(self.calls, len(self.values) - 1)
        self.calls += 1
        value = self.values[index]
        if isinstance(value, Exception):
            raise value
        return value


class FakeEndpointVerifier:
    def __init__(self, *, listening: bool = False, ownership: bool | None = True) -> None:
        self.listening = listening
        self.ownership = ownership
        self.preflight_calls = 0
        self.ownership_calls = 0

    def is_listening(self, host: str, port: int, *, timeout_seconds: float) -> bool:
        del host, port, timeout_seconds
        self.preflight_calls += 1
        return self.listening

    def owned_by(self, pid: int, host: str, port: int) -> bool | None:
        del pid, host, port
        self.ownership_calls += 1
        return self.ownership


class FakeClock:
    def __init__(self, on_sleep: Any = None) -> None:
        self.value = 10.0
        self.on_sleep = on_sleep

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds
        if self.on_sleep is not None:
            self.on_sleep()


def _healthy() -> HTTPJSONResponse:
    return HTTPJSONResponse(
        status_code=200,
        payload={"object": "list", "data": [{"id": "served-model", "object": "model"}]},
    )


def _adapter(
    tmp_path: Path,
    *,
    processes: FakeProcessBackend | None = None,
    inspector: FakeRuntimeInspector | None = None,
    http: FakeHTTPClient | None = None,
    endpoints: FakeEndpointVerifier | None = None,
    clock: FakeClock | None = None,
) -> tuple[VLLMAdapter, FakeProcessBackend, FakeRuntimeInspector, FakeHTTPClient]:
    selected_processes = processes or FakeProcessBackend()
    selected_inspector = inspector or FakeRuntimeInspector()
    selected_http = http or FakeHTTPClient([_healthy()])
    selected_endpoints = endpoints or FakeEndpointVerifier()
    selected_clock = clock or FakeClock()
    adapter = VLLMAdapter(
        tmp_path / "server-execution.json",
        process_backend=selected_processes,
        runtime_inspector=selected_inspector,
        http_client=selected_http,
        endpoint_verifier=selected_endpoints,
        monotonic=selected_clock.monotonic,
        sleep=selected_clock.sleep,
    )
    return adapter, selected_processes, selected_inspector, selected_http


def test_spec_binds_snapshot_runtime_config_argv_and_environment(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    changed_env = _spec(
        tmp_path,
        env={
            "PYTHONNOUSERSITE": "1",
            "ROCR_VISIBLE_DEVICES": "GPU-other-uuid",
        },
    )
    changed_manifest = _spec(tmp_path, environment_manifest_sha256="e" * 64)
    outer = _spec(tmp_path, image_digest=f"registry.invalid/vllm@sha256:{IMAGE_SHA}")
    raw_outer = _spec(tmp_path, image_digest=IMAGE_SHA)

    assert spec.identity.model_sha256 == SNAPSHOT_SHA
    assert spec.identity.image_sha256 == NATIVE_SHA
    assert outer.identity.image_sha256 == IMAGE_SHA
    assert raw_outer.image_digest == f"sha256:{IMAGE_SHA}"
    assert spec.identity.native_executable_sha256 == NATIVE_SHA
    assert spec.identity.environment_manifest_sha256 == ENVIRONMENT_MANIFEST_SHA
    assert spec.unset_env == (
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HIP_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "HSA_VISIBLE_DEVICES",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
    )
    assert spec.request_hash != changed_env.request_hash
    assert spec.identity.config_sha256 != changed_env.identity.config_sha256
    assert spec.request_hash != changed_manifest.request_hash
    assert spec.identity.runtime_sha256 != changed_manifest.identity.runtime_sha256
    assert spec.health_url == "http://127.0.0.1:8123/v1/models"


def test_spec_rejects_command_strings_mutable_identity_and_missing_model_name(
    tmp_path: Path,
) -> None:
    with pytest.raises(VLLMSpecError, match="argv sequence"):
        _spec(tmp_path, argv="vllm serve unsafe")
    with pytest.raises(VLLMSpecError, match="native_executable_sha256"):
        _spec(tmp_path, native_executable_sha256=None)
    with pytest.raises(VLLMSpecError, match="environment_manifest_sha256"):
        _spec(tmp_path, environment_manifest_sha256="")
    with pytest.raises(VLLMSpecError, match="lowercase"):
        _spec(tmp_path, environment_manifest_sha256="D" * 64)
    with pytest.raises(VLLMSpecError, match="expected_served_model"):
        _spec(tmp_path, expected_served_model="")
    with pytest.raises(VLLMSpecError, match="absolute Python interpreter"):
        _spec(tmp_path, argv=("/opt/venv/bin/vllm", "serve", "org/model"))
    with pytest.raises(VLLMSpecError, match="finite JSON"):
        _spec(tmp_path, config={"bad": math.nan})
    with pytest.raises(VLLMSpecError, match="both set and unset"):
        _spec(
            tmp_path,
            env={"CUDA_VISIBLE_DEVICES": "0", "PYTHONNOUSERSITE": "1"},
        )


def test_build_bench_serve_argv_is_literal_and_complete(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    literal = f"$(touch {marker})"
    argv = build_bench_serve_argv(
        model="org/model",
        base_url="http://127.0.0.1:8123/",
        num_prompts=32,
        request_rate=math.inf,
        max_concurrency=8,
        random_input_len=128,
        random_output_len=64,
        result_dir=tmp_path,
        result_filename="result.json",
        extra_args=("--metadata", literal),
    )

    assert argv[:3] == ("vllm", "bench", "serve")
    assert argv[argv.index("--request-rate") + 1] == "inf"
    assert argv[argv.index("--result-filename") + 1] == "result.json"
    assert argv[-1] == literal
    assert "--save-result" in argv and "--save-detailed" in argv
    assert not marker.exists()

    with pytest.raises(VLLMSpecError, match="argv sequence"):
        build_bench_serve_argv(
            model="org/model",
            base_url="http://localhost:8000",
            vllm_argv="python -m vllm",
        )
    with pytest.raises(VLLMSpecError, match="shell interpreter"):
        build_bench_serve_argv(
            model="org/model",
            base_url="http://localhost:8000",
            vllm_argv=("/bin/bash", "-c", "vllm"),
        )
    with pytest.raises(VLLMSpecError, match="locked benchmark flags"):
        build_bench_serve_argv(
            model="org/model",
            base_url="http://localhost:8000",
            extra_args=("--num-prompts=999",),
        )


def _bench_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "num_prompts": 2,
        "completed": 2,
        "failed": 0,
        "request_throughput": 2.5,
        "output_throughput": 200.0,
        "total_token_throughput": 450.0,
        "mean_ttft_ms": 12.0,
        "mean_tpot_ms": 4.0,
        "mean_itl_ms": 3.5,
        "mean_e2el_ms": 40.0,
        "ttfts": [0.010, 0.014],
        "tpots": [0.003, 0.005],
        "itls": [[0.003, 0.004], [0.0035]],
        "e2els": [0.035, 0.045],
    }
    row.update(overrides)
    return row


def test_parser_normalizes_units_and_separates_run_from_request_samples() -> None:
    result = parse_bench_serve_json(json.dumps(_bench_row()))

    assert result.request_throughput.value == pytest.approx(2.5)
    assert result.request_throughput.unit == "req/s"
    assert result.output_tps.unit == "tok/s"
    assert result.ttft.unit == "ms"
    assert result.e2el is not None and result.e2el.value == pytest.approx(40.0)
    assert result.ttft.samples == (12.0,)
    assert result.request_latency_samples_ms["ttft"] == pytest.approx((10.0, 14.0))
    assert result.request_latency_samples_ms["itl"] == pytest.approx((3.0, 4.0, 3.5))
    assert result.completed_requests == 2
    assert result.failed_requests == 0
    assert tuple(name for name in result.canonical_metrics if name != MEAN_E2EL_METRIC) == (
        CANONICAL_BENCH_METRIC_NAMES
    )
    assert result.request_throughput.name == CANONICAL_BENCH_METRIC_NAMES[0]
    series = result.to_metric_series()
    assert set(series) == {*CANONICAL_BENCH_METRIC_NAMES, MEAN_E2EL_METRIC}
    assert series[CANONICAL_BENCH_METRIC_NAMES[0]].samples == [2.5]
    assert series[CANONICAL_BENCH_METRIC_NAMES[0]].unit == "req/s"


def test_parser_preserves_one_stability_sample_per_run() -> None:
    result = parse_bench_serve_json([_bench_row(mean_ttft_ms=10.0), _bench_row(mean_ttft_ms=14.0)])

    assert result.ttft.value == pytest.approx(12.0)
    assert result.ttft.samples == (10.0, 14.0)
    assert result.ttft.sample_count == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mean_itl_ms": None}, "must be numeric"),
        ({"mean_ttft_ms": float("nan")}, "finite and positive"),
        ({"output_throughput": 0}, "finite and positive"),
        ({"failed": 1}, "failed requests"),
        ({"completed": 1}, "do not equal"),
        ({"completed": 3}, "do not equal"),
    ],
)
def test_parser_rejects_missing_nonfinite_failed_or_incomplete_metrics(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(VLLMBenchmarkParseError, match=message):
        parse_bench_serve_json(_bench_row(**overrides))

    missing = _bench_row()
    missing.pop("request_throughput")
    with pytest.raises(VLLMBenchmarkParseError, match="missing"):
        parse_bench_serve_json(missing)

    no_e2el = _bench_row()
    no_e2el.pop("mean_e2el_ms")
    no_e2el.pop("e2els")
    with pytest.raises(VLLMBenchmarkParseError, match="e2el.*missing"):
        parse_bench_serve_json(no_e2el, require_e2el=True)


def test_runtime_provenance_preflight_fails_before_spawn(tmp_path: Path) -> None:
    inspector = FailingPreflightRuntimeInspector()
    adapter, processes, _, _ = _adapter(tmp_path, inspector=inspector)

    with pytest.raises(VLLMServerStartError, match="before vLLM spawn"):
        adapter.start_or_resume(_spec(tmp_path))

    assert not processes.spawn_calls
    assert not (tmp_path / "server-execution.json").exists()


def test_start_persists_actual_identity_and_resumes_without_duplicate(tmp_path: Path) -> None:
    adapter, processes, inspector, _ = _adapter(tmp_path)
    spec = _spec(tmp_path)

    first = adapter.start_or_resume(spec)
    second = adapter.start_or_resume(spec)
    record = load_execution_record(tmp_path / "server-execution.json")

    assert not first.reused
    assert second.reused
    assert first.gate_eligible
    assert len(processes.spawn_calls) == 1
    assert inspector.calls == 2
    assert record is not None
    assert record.pid == first.record.pid
    assert record.boot_id == "boot-fixture"
    assert record.start_ticks == 9000
    assert record.request_hash == spec.request_hash
    assert record.runtime_evidence.pid_executable_sha256 == NATIVE_SHA
    assert record.runtime_evidence.verified
    assert record.status is ServerExecutionStatus.RUNNING
    assert record.requires_explicit_stop
    assert processes.spawn_calls[0]["argv"] == spec.argv
    assert processes.spawn_calls[0]["env"] == dict(spec.env)
    assert processes.spawn_calls[0]["unset_env"] == spec.unset_env


def test_preexisting_port_and_docker_wrapper_fail_before_spawn(tmp_path: Path) -> None:
    endpoints = FakeEndpointVerifier(listening=True)
    adapter, processes, _, _ = _adapter(tmp_path, endpoints=endpoints)
    with pytest.raises(VLLMServerConflictError, match="already has a listener"):
        adapter.start_or_resume(_spec(tmp_path))
    assert not processes.spawn_calls

    with pytest.raises(VLLMSpecError, match="absolute Python interpreter"):
        _spec(
            tmp_path,
            argv=("docker", "run", "registry.invalid/vllm@sha256:" + IMAGE_SHA),
            image_digest="sha256:" + IMAGE_SHA,
        )
    assert not processes.spawn_calls


def test_first_record_failure_rolls_back_exact_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, processes, _, _ = _adapter(tmp_path)
    spec = _spec(tmp_path)

    def fail_record(path: str | Path, record: Any) -> None:
        del path, record
        raise OSError("simulated full disk")

    monkeypatch.setattr(
        "amd_inference_opt.vllm_adapter.save_execution_record",
        fail_record,
    )

    with pytest.raises(OSError, match="simulated full disk"):
        adapter.start_or_resume(spec)

    assert len(processes.spawn_calls) == 1
    assert len(processes.terminate_calls) == 1
    assert processes.alive == {}

def test_live_different_request_is_a_conflict_not_a_second_server(tmp_path: Path) -> None:
    adapter, processes, _, _ = _adapter(tmp_path)
    adapter.start_or_resume(_spec(tmp_path))

    with pytest.raises(VLLMServerConflictError, match="different managed"):
        adapter.start_or_resume(_spec(tmp_path, dtype="bfloat16"))
    assert len(processes.spawn_calls) == 1


def test_health_timeout_is_durable_and_never_implicitly_stops(tmp_path: Path) -> None:
    clock = FakeClock()
    http = FakeHTTPClient([HTTPJSONResponse(status_code=503, payload={"error": "loading"})])
    adapter, processes, _, _ = _adapter(tmp_path, http=http, clock=clock)

    with pytest.raises(VLLMHealthTimeout) as caught:
        adapter.start_or_resume(_spec(tmp_path, startup_timeout_seconds=0.2))

    record = load_execution_record(tmp_path / "server-execution.json")
    assert caught.value.last_health.status_code == 503
    assert record is not None and record.status is ServerExecutionStatus.TIMED_OUT
    assert record.requires_explicit_stop
    assert caught.value.requires_explicit_stop
    assert not processes.terminate_calls
    assert record.pid in processes.alive


def test_process_exit_before_health_is_a_durable_failure(tmp_path: Path) -> None:
    processes = FakeProcessBackend()
    clock = FakeClock(on_sleep=processes.alive.clear)
    http = FakeHTTPClient([HTTPJSONResponse(status_code=503, payload={})])
    adapter, _, _, _ = _adapter(tmp_path, processes=processes, http=http, clock=clock)

    with pytest.raises(VLLMServerStartError, match="pid identity"):
        adapter.start_or_resume(_spec(tmp_path))

    record = load_execution_record(tmp_path / "server-execution.json")
    assert record is not None and record.status is ServerExecutionStatus.FAILED
    assert not processes.terminate_calls


def test_runtime_mismatch_is_recorded_and_not_killed_implicitly(tmp_path: Path) -> None:
    inspector = FakeRuntimeInspector(executable_sha="f" * 64)
    adapter, processes, _, http = _adapter(tmp_path, inspector=inspector)

    with pytest.raises(VLLMIdentityMismatchError, match="executable") as caught:
        adapter.start_or_resume(_spec(tmp_path))

    record = load_execution_record(tmp_path / "server-execution.json")
    assert record is not None and record.status is ServerExecutionStatus.FAILED
    assert record.runtime_evidence.verification is RuntimeVerificationStatus.MISMATCH
    assert caught.value.requires_explicit_stop
    assert http.calls == 0
    assert not processes.terminate_calls


def test_inherited_secondary_gpu_filter_is_a_runtime_mismatch(tmp_path: Path) -> None:
    inspector = FakeRuntimeInspector(leaked_unset_environment="CUDA_VISIBLE_DEVICES")
    adapter, processes, _, _ = _adapter(tmp_path, inspector=inspector)

    with pytest.raises(VLLMIdentityMismatchError, match="environment"):
        adapter.start_or_resume(_spec(tmp_path))

    record = load_execution_record(tmp_path / "server-execution.json")
    assert record is not None
    assert record.runtime_evidence.unexpected_inherited_environment == ("CUDA_VISIBLE_DEVICES",)
    assert not processes.terminate_calls


def test_environment_manifest_must_be_independently_observed(tmp_path: Path) -> None:
    missing = FakeRuntimeInspector(observed_environment_manifest_sha=None)
    adapter, _, _, _ = _adapter(tmp_path / "missing", inspector=missing)
    result = adapter.start_or_resume(_spec(tmp_path))

    assert result.health.healthy
    assert result.record.runtime_evidence.verification is RuntimeVerificationStatus.UNVERIFIED
    assert "environment manifest" in (result.record.runtime_evidence.reason or "")
    assert not result.gate_eligible

    mismatched = FakeRuntimeInspector(observed_environment_manifest_sha="e" * 64)
    adapter, processes, _, _ = _adapter(tmp_path / "mismatched", inspector=mismatched)
    with pytest.raises(VLLMIdentityMismatchError, match="environment manifest"):
        adapter.start_or_resume(_spec(tmp_path))
    assert not processes.terminate_calls


def test_unbound_outer_image_is_healthy_but_not_gate_eligible(tmp_path: Path) -> None:
    inspector = FakeRuntimeInspector(bind_outer_image=False)
    adapter, _, _, _ = _adapter(tmp_path, inspector=inspector)
    result = adapter.start_or_resume(
        _spec(tmp_path, image_digest=f"registry.invalid/vllm@sha256:{IMAGE_SHA}")
    )

    assert result.health.healthy
    assert result.record.runtime_evidence.verification is RuntimeVerificationStatus.UNVERIFIED
    assert not result.gate_eligible


def test_bound_outer_image_uses_cgroup_not_pid_equality(tmp_path: Path) -> None:
    adapter, _, _, _ = _adapter(tmp_path)
    result = adapter.start_or_resume(
        _spec(tmp_path, image_digest=f"registry.invalid/vllm@sha256:{IMAGE_SHA}")
    )

    evidence = result.record.runtime_evidence
    assert evidence.container_init_pid == 1
    assert result.record.pid != evidence.container_init_pid
    assert evidence.process_binding_id == evidence.container_binding_id
    assert result.gate_eligible


def test_pid_namespace_alone_never_binds_outer_container(tmp_path: Path) -> None:
    inspector = FakeRuntimeInspector(binding_kind="pid_namespace_inode")
    adapter, _, _, _ = _adapter(tmp_path, inspector=inspector)
    result = adapter.start_or_resume(
        _spec(tmp_path, image_digest=f"registry.invalid/vllm@sha256:{IMAGE_SHA}")
    )

    evidence = result.record.runtime_evidence
    assert evidence.container_binding_kind == "pid_namespace_inode"
    assert evidence.verification is RuntimeVerificationStatus.UNVERIFIED
    assert "cgroup-v2" in (evidence.reason or "")
    assert not result.gate_eligible


def test_health_owner_mismatch_never_accepts_an_old_server(tmp_path: Path) -> None:
    endpoints = FakeEndpointVerifier(ownership=False)
    clock = FakeClock()
    adapter, processes, _, _ = _adapter(tmp_path, endpoints=endpoints, clock=clock)

    with pytest.raises(VLLMHealthTimeout):
        adapter.start_or_resume(_spec(tmp_path, startup_timeout_seconds=0.1))

    record = load_execution_record(tmp_path / "server-execution.json")
    assert record is not None and record.status is ServerExecutionStatus.TIMED_OUT
    assert not processes.terminate_calls


def test_explicit_stop_uses_recorded_timeout_and_stale_pid_is_never_signaled(
    tmp_path: Path,
) -> None:
    adapter, processes, _, _ = _adapter(tmp_path)
    spec = _spec(tmp_path)
    started = adapter.start_or_resume(spec)
    stopped = adapter.stop(expected_request_hash=spec.request_hash)

    assert stopped.stopped
    assert stopped.record is not None
    assert stopped.record.status is ServerExecutionStatus.STOPPED
    assert not stopped.record.requires_explicit_stop
    assert processes.terminate_calls[0]["timeout_seconds"] == 7.5
    assert processes.terminate_calls[0]["start_ticks"] == started.record.start_ticks

    # A coordinator crash after the durable STOPPED write must be recoverable:
    # retrying stop succeeds idempotently without signalling or rewriting ORPHANED.
    stopped_again = adapter.stop(expected_request_hash=spec.request_hash)
    assert stopped_again.stopped
    assert stopped_again.reason == "already stopped"
    assert stopped_again.record is not None
    assert stopped_again.record.status is ServerExecutionStatus.STOPPED
    assert len(processes.terminate_calls) == 1

    second_adapter, second_processes, _, _ = _adapter(tmp_path / "second")
    second_spec = _spec(tmp_path)
    second = second_adapter.start_or_resume(second_spec)
    second_processes.alive[second.record.pid] += 1  # Simulate PID reuse.
    stale = second_adapter.stop()

    assert not stale.stopped
    assert stale.record is not None and stale.record.status is ServerExecutionStatus.ORPHANED
    assert not second_processes.terminate_calls


def test_stop_is_not_successful_while_health_port_remains_live(tmp_path: Path) -> None:
    endpoints = FakeEndpointVerifier(listening=False)
    adapter, processes, _, _ = _adapter(tmp_path, endpoints=endpoints)
    spec = _spec(tmp_path)
    adapter.start_or_resume(spec)
    endpoints.listening = True

    stopped = adapter.stop(expected_request_hash=spec.request_hash)

    assert processes.terminate_calls
    assert not stopped.stopped
    assert stopped.record is not None
    assert stopped.record.status is ServerExecutionStatus.ORPHANED
    assert "endpoint remained live" in stopped.reason


def test_stdlib_health_client_uses_get_and_rejects_invalid_json() -> None:
    requests: list[Any] = []

    class Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def getcode(self) -> int:
            return 200

        def read(self, size: int) -> bytes:
            return self.body[:size]

    def opener(request: Any, *, timeout: float) -> Response:
        assert timeout == 1.5
        requests.append(request)
        return Response(b'{"data": [{"id": "served-model"}]}')

    client = StdlibHTTPClient(opener=opener)
    response = client.get_json("http://127.0.0.1:8000/v1/models", timeout_seconds=1.5)

    assert response.status_code == 200
    assert requests[0].get_method() == "GET"
    assert response.payload == {"data": [{"id": "served-model"}]}

    invalid = StdlibHTTPClient(opener=lambda *_args, **_kwargs: Response(b"not-json"))
    with pytest.raises(VLLMAdapterError, match="valid UTF-8 JSON"):
        invalid.get_json("http://127.0.0.1:8000/v1/models", timeout_seconds=1.5)
