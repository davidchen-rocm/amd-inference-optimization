from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from amd_inference_opt.command import CommandResult, command_request_sha256
from amd_inference_opt.mi300x_observation import (
    CommandMI300XEnvironmentWindow,
    CommandMI300XInspectionPort,
    MI300XHIPProbeV1,
    MI300XObservationError,
    MI300XStaticProbeV1,
    _hip_uuid_token,
    _normalize_mi300x_product_name,
    observation_request_sha256,
)
from amd_inference_opt.models import ArtifactRef

UUID = "GPU-11111111-2222-3333-4444-555555555555"
BDF = "0000:41:00.0"


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, Any]] = []

    def save_evidence_json(
        self, task_id: str, logical_name: str, value: Any, *, producer: str
    ) -> ArtifactRef:
        payload = json.dumps(value, default=str, sort_keys=True).encode()
        reference = ArtifactRef(
            path=f"artifacts/evidence/{logical_name}/v{len(self.saved) + 1:06d}.json",
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
            producer=producer,
            media_type="application/json",
        )
        self.saved.append((logical_name, value))
        return reference

    def verify_artifact(self, task_id: str, reference: ArtifactRef) -> bool:
        return bool(task_id and reference.sha256)


class FakeRunner:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: str | Path,
        env: dict[str, str],
        unset_env: tuple[str, ...],
        timeout_seconds: float,
    ) -> CommandResult:
        stdout = self.outputs.pop(0)
        return CommandResult(
            argv=tuple(argv),
            cwd=str(cwd),
            started_at="2026-08-23T00:00:00+00:00",
            duration_seconds=0.1,
            exit_code=0,
            stdout=stdout,
            stderr="",
            environment=env,
            unset_environment=tuple(unset_env),
            timeout_seconds=timeout_seconds,
            request_sha256=command_request_sha256(
                argv,
                cwd=cwd,
                env=env,
                unset_env=unset_env,
                timeout_seconds=timeout_seconds,
            ),
        )


def _static() -> dict[str, Any]:
    return {
        "schema": "gpuopt.mi300x-static-probe.v1",
        "captured_at": "2026-08-23T00:00:00Z",
        "device_uuid": UUID,
        "pci_bdf": BDF,
        "product_name": "AMD Instinct MI300X",
        "oam_id": 0,
        "xcc_count": 8,
        "compute_partition": "SPX",
        "memory_partition": "NPS1",
        "partition_id": 0,
        "raw": {"source": "amdsmi"},
    }


def _hip() -> dict[str, Any]:
    return {
        "schema": "gpuopt.mi300x-hip-probe.v1",
        "captured_at": "2026-08-23T00:00:01Z",
        "visible_device_count": 1,
        "logical_device_id": 0,
        "device_name": "AMD Instinct MI300X",
        "gfx_target": "gfx942",
        "device_uuid": UUID,
        "pci_bdf": BDF,
        "mapping_basis": "hip-runtime-uuid+bdf",
        "rocr_visible_devices": UUID,
        "hip_visible_devices": None,
        "hsa_visible_devices": None,
        "cuda_visible_devices": None,
        "gpu_device_ordinal": None,
        "rocm_version": "7.2.0",
        "pytorch_version": "2.8.0+rocm7.2",
        "vllm_version": "0.10.1",
        "python_version": "3.12.4",
    }


def _telemetry(
    *,
    throttle: bool = False,
    partition: str = "SPX",
    gfxclk_mhz: float = 1700,
    uclk_mhz: float = 1300,
) -> dict[str, Any]:
    return {
        "schema": "gpuopt.mi300x-telemetry.v1",
        "captured_at": "2026-08-23T00:00:02Z",
        "device_uuid": UUID,
        "pci_bdf": BDF,
        "compute_partition": partition,
        "memory_partition": "NPS1",
        "partition_id": 0,
        "current_gfxclk_mhz": gfxclk_mhz,
        "average_gfxclk_mhz": 1650,
        "current_uclk_mhz": uclk_mhz,
        "socket_power_w": 500,
        "hotspot_temperature_c": 65,
        "hbm_temperatures_c": [58, 59],
        "throttle_active": throttle,
        "power_violation_percent": 0,
        "thermal_violation_percent": 0,
        "raw": {"source": "amdsmi"},
    }


def _config(tmp_path: Path, static_argv: tuple[str, ...], hip_argv: tuple[str, ...]) -> Any:
    timeout = 30.0
    device = SimpleNamespace(
        device_uuid=UUID,
        pci_bdf=BDF,
        product_name="AMD Instinct MI300X",
        oam_id=0,
        xcc_count=8,
        compute_partition="SPX",
        memory_partition="NPS1",
        partition_id=0,
        gfx_target="gfx942",
        amd_smi_command_sha256=observation_request_sha256(
            static_argv,
            cwd=tmp_path,
            device_uuid=UUID,
            timeout_seconds=timeout,
        ),
        hip_probe_command_sha256=observation_request_sha256(
            hip_argv,
            cwd=tmp_path,
            device_uuid=UUID,
            timeout_seconds=timeout,
        ),
    )
    runtime = SimpleNamespace(
        image_digest=None,
        environment_manifest_sha256="1" * 64,
        executable_sha256=hashlib.sha256(
            Path(static_argv[0]).resolve().read_bytes()
        ).hexdigest(),
    )
    return SimpleNamespace(
        task=SimpleNamespace(
            id="mi300-test",
            runtime=runtime,
            environment=SimpleNamespace(max_telemetry_clock_drift_percent=5.0),
        ),
        device=device,
        serving=SimpleNamespace(engine_config={"launcher_sha256": "2" * 64}),
        environment_coordinates={
            "gpu_gfx": "gfx942",
            "gpu_device_uuid": UUID,
        },
    )


def test_static_probe_rejects_mi300a_even_though_gfx_can_match() -> None:
    value = _static()
    value["product_name"] = "AMD Instinct MI300A"
    with pytest.raises(ValueError, match="did not identify"):
        MI300XStaticProbeV1.model_validate(value)


def test_cloud_vf_market_name_normalizes_without_accepting_other_gfx942_products() -> None:
    assert _normalize_mi300x_product_name("AMD Instinct MI300X") == (
        "AMD Instinct MI300X"
    )
    assert _normalize_mi300x_product_name("AMD Instinct MI300X VF") == (
        "AMD Instinct MI300X"
    )

    with pytest.raises(MI300XObservationError, match="product family"):
        _normalize_mi300x_product_name("AMD Instinct MI325X")


def test_hip_ascii_uuid_is_not_hex_encoded_twice() -> None:
    assert _hip_uuid_token(b"cb8383781156855c") == "GPU-cb8383781156855c"

    with pytest.raises(MI300XObservationError, match="16-digit hexadecimal"):
        _hip_uuid_token(b"zzzzzzzzzzzzzzzz")


def test_hip_probe_allows_explicitly_unavailable_cloud_vf_display_name() -> None:
    value = _hip()
    value["device_name"] = "unavailable"

    assert MI300XHIPProbeV1.model_validate(value).device_name == "unavailable"


def test_inspection_executes_hash_bound_raw_commands(tmp_path: Path) -> None:
    static_argv = (
        "/usr/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "static",
        "--uuid",
        UUID,
    )
    hip_argv = (
        "/usr/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "hip",
    )
    config = _config(tmp_path, static_argv, hip_argv)
    store = FakeStore()
    port = CommandMI300XInspectionPort(
        store,  # type: ignore[arg-type]
        amd_smi_argv=static_argv,
        hip_probe_argv=hip_argv,
        cwd=tmp_path,
        runner=FakeRunner([json.dumps(_static()), json.dumps(_hip())]),
        timeout_seconds=30,
    )

    evidence = port.inspect(config)

    assert evidence.device_uuid == UUID
    assert evidence.visible_device_count == 1
    assert evidence.scope == "native_preflight"
    assert [item[0] for item in store.saved] == [
        "vllm/raw-inspection/amd-smi",
        "vllm/raw-inspection/hip-probe",
    ]


def test_container_inspection_cannot_copy_digest_from_config(tmp_path: Path) -> None:
    static_argv = (
        "/usr/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "static",
        "--uuid",
        UUID,
    )
    hip_argv = (
        "/usr/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "hip",
    )
    config = _config(tmp_path, static_argv, hip_argv)
    config.task.runtime.image_digest = "3" * 64
    port = CommandMI300XInspectionPort(
        FakeStore(),  # type: ignore[arg-type]
        amd_smi_argv=static_argv,
        hip_probe_argv=hip_argv,
        cwd=tmp_path,
        runner=FakeRunner([]),
        timeout_seconds=30,
    )
    with pytest.raises(MI300XObservationError, match="trusted RepoDigest"):
        port.inspect(config)


def test_telemetry_window_marks_throttle_inconclusive(tmp_path: Path) -> None:
    static_argv = (
        "/usr/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "static",
        "--uuid",
        UUID,
    )
    hip_argv = (
        "/usr/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "hip",
    )
    config = _config(tmp_path, static_argv, hip_argv)
    store = FakeStore()
    telemetry_argv = (
        "/usr/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "telemetry",
        "--uuid",
        UUID,
    )
    port = CommandMI300XEnvironmentWindow(
        store,  # type: ignore[arg-type]
        telemetry_argv=telemetry_argv,
        cwd=tmp_path,
        runner=FakeRunner(
            [json.dumps(_telemetry()), json.dumps(_telemetry(throttle=True))]
        ),
        timeout_seconds=30,
    )

    start = port.begin(config, phase="baseline", server_request_hash="4" * 64)
    capture = port.finish(
        config,
        phase="baseline",
        server_request_hash="4" * 64,
        before=start,
    )

    assert capture.fingerprint.telemetry_stable is False
    assert any("throttle" in reason for reason in capture.fingerprint.instability_reasons)
    assert len(capture.before_artifacts) == len(capture.after_artifacts) == 1


def test_telemetry_window_is_stable_only_with_complete_identity(tmp_path: Path) -> None:
    static_argv = (
        "/usr/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "static",
        "--uuid",
        UUID,
    )
    hip_argv = (
        "/usr/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "hip",
    )
    config = _config(tmp_path, static_argv, hip_argv)
    store = FakeStore()
    port = CommandMI300XEnvironmentWindow(
        store,  # type: ignore[arg-type]
        telemetry_argv=(
            "/usr/bin/python3",
            "-I",
            "-m",
            "amd_inference_opt.mi300x_observation",
            "--mode",
            "telemetry",
            "--uuid",
            UUID,
        ),
        cwd=tmp_path,
        runner=FakeRunner([json.dumps(_telemetry()), json.dumps(_telemetry())]),
        timeout_seconds=30,
    )
    start = port.begin(config, phase="candidate", server_request_hash="5" * 64)
    capture = port.finish(
        config,
        phase="candidate",
        server_request_hash="5" * 64,
        before=start,
    )

    assert capture.fingerprint.telemetry_stable is True
    assert capture.fingerprint.instability_reasons == []
    assert capture.fingerprint.values["gpu_device_uuid"] == UUID
    assert capture.fingerprint.values["telemetry_gfxclk_mean_mhz"] == "1700.000000"


def test_telemetry_clock_drift_is_inconclusive(tmp_path: Path) -> None:
    static_argv = (
        "/usr/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "static",
        "--uuid",
        UUID,
    )
    hip_argv = (
        "/usr/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "hip",
    )
    config = _config(tmp_path, static_argv, hip_argv)
    telemetry_argv = (
        "/usr/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.mi300x_observation",
        "--mode",
        "telemetry",
        "--uuid",
        UUID,
    )
    port = CommandMI300XEnvironmentWindow(
        FakeStore(),  # type: ignore[arg-type]
        telemetry_argv=telemetry_argv,
        cwd=tmp_path,
        runner=FakeRunner(
            [
                json.dumps(_telemetry(gfxclk_mhz=1000, uclk_mhz=800)),
                json.dumps(_telemetry(gfxclk_mhz=2000, uclk_mhz=1300)),
            ]
        ),
        timeout_seconds=30,
    )

    start = port.begin(config, phase="candidate", server_request_hash="6" * 64)
    capture = port.finish(
        config,
        phase="candidate",
        server_request_hash="6" * 64,
        before=start,
    )

    assert capture.fingerprint.telemetry_stable is False
    assert any("gfxclk drift" in item for item in capture.fingerprint.instability_reasons)
