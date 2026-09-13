"""Concrete, read-only MI300X inspection and telemetry adapters.

The vLLM workflow intentionally consumes normalized evidence, but normalization
must never mean that hardware coordinates were typed into a JSON file by hand.
This module executes two small project-owned probes, stores their complete
command envelopes, and only then projects them into the workflow DTOs.

No method in this module changes clocks, power limits, or partition settings.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError, field_validator, model_validator

from .command import CommandRunner, command_request_sha256, validate_argv
from .models import ArtifactRef, EnvironmentFingerprint, StrictModel, utc_now
from .store import ExperimentStore
from .vllm_environment import framework_source_identity_sha256
from .vllm_models import MI300XInspectionEvidence, VLLMCampaignConfig
from .vllm_ports import VLLMEnvironmentCapture, VLLMPortError

_CONFLICTING_GPU_ENV = (
    "HIP_VISIBLE_DEVICES",
    "HSA_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
)
_OBSERVATION_IMPLEMENTATION_ENV = "GPUOPT_MI300X_OBSERVATION_SHA256"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MI300XObservationError(VLLMPortError):
    """Raw device evidence is unavailable, malformed, or inconsistent."""


def observation_implementation_sha256() -> str:
    return framework_source_identity_sha256()


def _executable_sha256(value: str) -> str:
    lexical = Path(value)
    if not lexical.is_absolute():
        raise MI300XObservationError("observation Python executable must be absolute")
    try:
        before_link = lexical.resolve(strict=True)
        descriptor = os.open(
            before_link,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise MI300XObservationError(
            "observation Python executable is unavailable"
        ) from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise MI300XObservationError(
                "observation Python executable is not a regular file"
            )
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise MI300XObservationError(
            "observation Python executable changed while it was hashed"
        )
    try:
        after_link = lexical.resolve(strict=True)
    except OSError as error:
        raise MI300XObservationError(
            "observation Python executable changed while it was resolved"
        ) from error
    if after_link != before_link:
        raise MI300XObservationError(
            "observation Python executable symlink changed during verification"
        )
    return digest.hexdigest()


def observation_environment(device_uuid: str) -> dict[str, str]:
    return {
        "PYTHONNOUSERSITE": "1",
        "ROCR_VISIBLE_DEVICES": device_uuid,
        _OBSERVATION_IMPLEMENTATION_ENV: observation_implementation_sha256(),
    }


def observation_unset_environment() -> tuple[str, ...]:
    return _CONFLICTING_GPU_ENV


def observation_request_sha256(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    device_uuid: str,
    timeout_seconds: float,
) -> str:
    return command_request_sha256(
        argv,
        cwd=cwd,
        env=observation_environment(device_uuid),
        unset_env=_CONFLICTING_GPU_ENV,
        timeout_seconds=timeout_seconds,
    )


class ObservationCommandPort(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str],
        unset_env: Sequence[str],
        timeout_seconds: float,
    ) -> Any: ...


def _single_line(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized or any(character in normalized for character in "\x00\r\n"):
        raise ValueError(f"{label} must be non-empty single-line text")
    return normalized


def _uuid(value: str) -> str:
    normalized = _single_line(value, "device_uuid")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]+", normalized) is None:
        raise ValueError("device_uuid is not a valid AMD SMI/HIP UUID token")
    return normalized


def _bdf(value: str) -> str:
    normalized = _single_line(value, "pci_bdf").lower()
    if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", normalized) is None:
        raise ValueError("pci_bdf must use canonical dddd:bb:ss.f form")
    return normalized


class MI300XStaticProbeV1(StrictModel):
    schema_id: Literal["gpuopt.mi300x-static-probe.v1"] = Field(alias="schema")
    captured_at: datetime
    device_uuid: str
    pci_bdf: str
    product_name: str
    oam_id: int = Field(ge=0)
    xcc_count: int = Field(ge=1)
    compute_partition: str
    memory_partition: str
    partition_id: int = Field(ge=0)
    raw: dict[str, Any]

    @field_validator("device_uuid")
    @classmethod
    def valid_uuid(cls, value: str) -> str:
        return _uuid(value)

    @field_validator("pci_bdf")
    @classmethod
    def valid_bdf(cls, value: str) -> str:
        return _bdf(value)

    @field_validator("product_name", "compute_partition", "memory_partition")
    @classmethod
    def valid_text(cls, value: str) -> str:
        return _single_line(value, "static probe field")

    @model_validator(mode="after")
    def valid_mi300x(self) -> MI300XStaticProbeV1:
        if self.product_name != "AMD Instinct MI300X":
            raise ValueError("AMD SMI did not identify an AMD Instinct MI300X")
        if self.xcc_count != 8:
            raise ValueError("MI300X V0 requires observed total XCC count 8")
        allowed = {
            "SPX": {"NPS1"},
            "DPX": {"NPS1", "NPS2"},
            "QPX": {"NPS1", "NPS4"},
            "CPX": {"NPS1", "NPS4"},
        }
        if self.compute_partition not in allowed:
            raise ValueError("unsupported MI300X compute partition")
        if self.memory_partition not in allowed[self.compute_partition]:
            raise ValueError("unsupported MI300X compute/memory partition pair")
        return self


class MI300XHIPProbeV1(StrictModel):
    schema_id: Literal["gpuopt.mi300x-hip-probe.v1"] = Field(alias="schema")
    captured_at: datetime
    visible_device_count: Literal[1]
    logical_device_id: Literal[0]
    device_name: str
    gfx_target: Literal["gfx942"]
    device_uuid: str
    pci_bdf: str
    mapping_basis: Literal["hip-runtime-uuid+bdf"]
    rocr_visible_devices: str
    hip_visible_devices: None = None
    hsa_visible_devices: None = None
    cuda_visible_devices: None = None
    gpu_device_ordinal: None = None
    rocm_version: str
    pytorch_version: str
    vllm_version: str
    python_version: str

    @field_validator(
        "device_name",
        "rocr_visible_devices",
        "rocm_version",
        "pytorch_version",
        "vllm_version",
        "python_version",
    )
    @classmethod
    def valid_text(cls, value: str) -> str:
        return _single_line(value, "HIP probe field")

    @field_validator("device_uuid")
    @classmethod
    def valid_uuid(cls, value: str) -> str:
        return _uuid(value)

    @field_validator("pci_bdf")
    @classmethod
    def valid_bdf(cls, value: str) -> str:
        return _bdf(value)


class MI300XTelemetryProbeV1(StrictModel):
    schema_id: Literal["gpuopt.mi300x-telemetry.v1"] = Field(alias="schema")
    captured_at: datetime
    device_uuid: str
    pci_bdf: str
    compute_partition: str
    memory_partition: str
    partition_id: int = Field(ge=0)
    current_gfxclk_mhz: float | None = None
    average_gfxclk_mhz: float | None = None
    current_uclk_mhz: float | None = None
    socket_power_w: float | None = None
    hotspot_temperature_c: float | None = None
    hbm_temperatures_c: list[float] = Field(default_factory=list)
    throttle_active: bool | None = None
    power_violation_percent: float | None = None
    thermal_violation_percent: float | None = None
    raw: dict[str, Any]

    @field_validator("device_uuid")
    @classmethod
    def valid_uuid(cls, value: str) -> str:
        return _uuid(value)

    @field_validator("pci_bdf")
    @classmethod
    def valid_bdf(cls, value: str) -> str:
        return _bdf(value)

    @field_validator(
        "current_gfxclk_mhz",
        "average_gfxclk_mhz",
        "current_uclk_mhz",
        "socket_power_w",
        "hotspot_temperature_c",
        "power_violation_percent",
        "thermal_violation_percent",
    )
    @classmethod
    def finite_optional(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("telemetry metrics must be finite and non-negative")
        return value

    @field_validator("hbm_temperatures_c")
    @classmethod
    def finite_temperatures(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("HBM temperatures must be finite and non-negative")
        return values


class ObservedContainerIdentity(StrictModel):
    """Hash-bound observation produced by a trusted outer-container launcher."""

    image_digest: str
    binding_kind: Literal["cgroup_v2"]
    binding_id: str
    evidence: ArtifactRef

    @field_validator("image_digest")
    @classmethod
    def valid_image_digest(cls, value: str) -> str:
        normalized = value.removeprefix("sha256:")
        if _SHA256.fullmatch(normalized) is None:
            raise ValueError("container RepoDigest must contain a SHA-256 digest")
        return normalized

    @field_validator("binding_id")
    @classmethod
    def valid_binding(cls, value: str) -> str:
        return _single_line(value, "container cgroup binding")


def _json_stdout(result: Any, model: type[StrictModel], label: str) -> Any:
    if not bool(getattr(result, "succeeded", False)):
        raise MI300XObservationError(f"{label} command failed")
    stdout = getattr(result, "stdout", None)
    if not isinstance(stdout, str) or len(stdout.encode("utf-8")) > 8 * 1024 * 1024:
        raise MI300XObservationError(f"{label} output is absent or exceeds 8 MiB")
    try:
        return model.model_validate_json(stdout)
    except ValidationError as error:
        raise MI300XObservationError(f"{label} output has an invalid schema") from error


def _validate_probe_timestamp(
    result: Any,
    captured_at: datetime,
    label: str,
    *,
    tolerance_seconds: float = 5.0,
) -> None:
    try:
        started = datetime.fromisoformat(str(result.started_at).replace("Z", "+00:00"))
        duration = float(result.duration_seconds)
    except (AttributeError, TypeError, ValueError) as error:
        raise MI300XObservationError(
            f"{label} command has no auditable execution window"
        ) from error
    if started.tzinfo is None or captured_at.tzinfo is None or duration < 0:
        raise MI300XObservationError(
            f"{label} command/probe timestamps must be timezone-aware and valid"
        )
    start_utc = started.astimezone(UTC)
    captured_utc = captured_at.astimezone(UTC)
    lower = start_utc - timedelta(seconds=tolerance_seconds)
    upper = start_utc + timedelta(seconds=duration + tolerance_seconds)
    if not lower <= captured_utc <= upper:
        raise MI300XObservationError(
            f"{label} captured_at falls outside its command execution window"
        )


def _command_envelope(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        value = result.to_dict()
    elif isinstance(result, Mapping):
        value = dict(result)
    else:
        value = {
            name: getattr(result, name, None)
            for name in (
                "argv",
                "cwd",
                "started_at",
                "duration_seconds",
                "exit_code",
                "stdout",
                "stderr",
                "timed_out",
                "spawn_error",
                "environment",
                "unset_environment",
                "timeout_seconds",
                "argv_sha256",
                "request_sha256",
            )
        }
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def _require_executed_request(result: Any, expected_sha256: str, label: str) -> None:
    if getattr(result, "request_sha256", None) != expected_sha256:
        raise MI300XObservationError(f"{label} execution differs from the bound request")


class CommandMI300XInspectionPort:
    """Execute and persist exact AMD SMI + HIP preflight commands."""

    def __init__(
        self,
        store: ExperimentStore,
        *,
        amd_smi_argv: Sequence[str],
        hip_probe_argv: Sequence[str],
        cwd: str | Path,
        runner: ObservationCommandPort | None = None,
        timeout_seconds: float = 60,
        container_identity: ObservedContainerIdentity | None = None,
    ) -> None:
        self.store = store
        self.amd_smi_argv = validate_argv(amd_smi_argv)
        self.hip_probe_argv = validate_argv(hip_probe_argv)
        static_prefix = (
            "-I",
            "-m",
            "amd_inference_opt.mi300x_observation",
            "--mode",
            "static",
        )
        hip_prefix = (
            "-I",
            "-m",
            "amd_inference_opt.mi300x_observation",
            "--mode",
            "hip",
        )
        if self.amd_smi_argv[1:6] != static_prefix:
            raise MI300XObservationError(
                "AMD SMI probe must use isolated project-owned module coordinates"
            )
        if self.hip_probe_argv[1:6] != hip_prefix:
            raise MI300XObservationError(
                "HIP probe must use isolated project-owned module coordinates"
            )
        self.cwd = Path(cwd).resolve()
        self.runner = runner or CommandRunner()
        self.timeout_seconds = float(timeout_seconds)
        self.container_identity = container_identity

    def _request_hash(self, argv: Sequence[str], uuid: str) -> str:
        return observation_request_sha256(
            argv,
            cwd=self.cwd,
            device_uuid=uuid,
            timeout_seconds=self.timeout_seconds,
        )

    def inspect(self, config: VLLMCampaignConfig) -> MI300XInspectionEvidence:
        uuid = config.device.device_uuid
        static_hash = self._request_hash(self.amd_smi_argv, uuid)
        hip_hash = self._request_hash(self.hip_probe_argv, uuid)
        if static_hash != config.device.amd_smi_command_sha256:
            raise MI300XObservationError("AMD SMI request hash differs from task binding")
        if hip_hash != config.device.hip_probe_command_sha256:
            raise MI300XObservationError("HIP probe request hash differs from task binding")
        runtime = config.task.runtime
        for label, argv in (
            ("AMD SMI", self.amd_smi_argv),
            ("HIP", self.hip_probe_argv),
        ):
            if _executable_sha256(argv[0]) != runtime.executable_sha256:
                raise MI300XObservationError(
                    f"{label} probe Python bytes differ from the pinned runtime"
                )
        if runtime.image_digest is not None:
            observed = self.container_identity
            if observed is None:
                raise MI300XObservationError(
                    "container task requires trusted RepoDigest+cgroup preflight evidence"
                )
            if observed.image_digest != runtime.image_digest:
                raise MI300XObservationError("observed container RepoDigest differs")
            if not self.store.verify_artifact(config.task.id, observed.evidence):
                raise MI300XObservationError("container identity artifact failed integrity")
        elif self.container_identity is not None:
            raise MI300XObservationError("native task cannot claim container identity")

        environment = observation_environment(uuid)
        static_result = self.runner.run(
            self.amd_smi_argv,
            cwd=self.cwd,
            env=environment,
            unset_env=_CONFLICTING_GPU_ENV,
            timeout_seconds=self.timeout_seconds,
        )
        hip_result = self.runner.run(
            self.hip_probe_argv,
            cwd=self.cwd,
            env=environment,
            unset_env=_CONFLICTING_GPU_ENV,
            timeout_seconds=self.timeout_seconds,
        )
        _require_executed_request(static_result, static_hash, "AMD SMI probe")
        _require_executed_request(hip_result, hip_hash, "HIP probe")
        static = _json_stdout(static_result, MI300XStaticProbeV1, "AMD SMI probe")
        hip = _json_stdout(hip_result, MI300XHIPProbeV1, "HIP probe")
        _validate_probe_timestamp(static_result, static.captured_at, "AMD SMI probe")
        _validate_probe_timestamp(hip_result, hip.captured_at, "HIP probe")
        static_ref = self.store.save_evidence_json(
            config.task.id,
            "vllm/raw-inspection/amd-smi",
            {
                "command": _command_envelope(static_result),
                "parsed": static.model_dump(mode="json", by_alias=True),
                "container_identity": (
                    self.container_identity.model_dump(mode="json")
                    if self.container_identity is not None
                    else None
                ),
            },
            producer="mi300x-inspection-port",
        )
        hip_ref = self.store.save_evidence_json(
            config.task.id,
            "vllm/raw-inspection/hip-probe",
            {
                "command": _command_envelope(hip_result),
                "parsed": hip.model_dump(mode="json", by_alias=True),
            },
            producer="mi300x-inspection-port",
        )
        mismatches = {
            "device_uuid": (static.device_uuid, uuid),
            "pci_bdf": (static.pci_bdf, config.device.pci_bdf),
            "HIP pci_bdf": (hip.pci_bdf, config.device.pci_bdf),
            "product_name": (static.product_name, config.device.product_name),
            "oam_id": (static.oam_id, config.device.oam_id),
            "xcc_count": (static.xcc_count, config.device.xcc_count),
            "compute_partition": (
                static.compute_partition,
                config.device.compute_partition,
            ),
            "memory_partition": (
                static.memory_partition,
                config.device.memory_partition,
            ),
            "partition_id": (static.partition_id, config.device.partition_id),
            "gfx_target": (hip.gfx_target, config.device.gfx_target),
            "HIP ROCR selector": (hip.rocr_visible_devices, uuid),
        }
        bad = [name for name, (actual, expected) in mismatches.items() if actual != expected]
        if _uuid_payload(hip.device_uuid) != _uuid_payload(uuid):
            bad.append("HIP device_uuid")
        if bad:
            raise MI300XObservationError(
                "MI300X probe differs from immutable config: " + ", ".join(sorted(bad))
            )
        return MI300XInspectionEvidence(
            scope=(
                "container_preflight"
                if runtime.image_digest is not None
                else "native_preflight"
            ),
            image_digest=runtime.image_digest,
            gfx_target=hip.gfx_target,
            product_name=static.product_name,
            oam_id=static.oam_id,
            xcc_count=static.xcc_count,
            rocr_visible_devices=hip.rocr_visible_devices,
            device_uuid=static.device_uuid,
            pci_bdf=static.pci_bdf,
            compute_partition=static.compute_partition,
            memory_partition=static.memory_partition,
            partition_id=static.partition_id,
            rocm_version=hip.rocm_version,
            vllm_version=hip.vllm_version,
            pytorch_version=hip.pytorch_version,
            python_version=hip.python_version,
            launcher_sha256=str(config.serving.engine_config["launcher_sha256"]),
            environment_manifest_sha256=str(runtime.environment_manifest_sha256),
            amd_smi_artifact=static_ref,
            hip_probe_artifact=hip_ref,
            amd_smi_command_sha256=static_hash,
            hip_probe_command_sha256=hip_hash,
        )


@dataclass(frozen=True)
class _TelemetryStart:
    phase: str
    server_request_hash: str
    request_hash: str
    parsed: MI300XTelemetryProbeV1
    artifact: ArtifactRef


class CommandMI300XEnvironmentWindow:
    """Capture before/after telemetry and fail closed on missing stability evidence."""

    def __init__(
        self,
        store: ExperimentStore,
        *,
        telemetry_argv: Sequence[str],
        cwd: str | Path,
        runner: ObservationCommandPort | None = None,
        timeout_seconds: float = 30,
        max_temperature_c: float = 90.0,
    ) -> None:
        self.store = store
        self.telemetry_argv = validate_argv(telemetry_argv)
        telemetry_prefix = (
            "-I",
            "-m",
            "amd_inference_opt.mi300x_observation",
            "--mode",
            "telemetry",
        )
        if self.telemetry_argv[1:6] != telemetry_prefix:
            raise MI300XObservationError(
                "telemetry probe must use isolated project-owned module coordinates"
            )
        self.cwd = Path(cwd).resolve()
        self.runner = runner or CommandRunner()
        self.timeout_seconds = float(timeout_seconds)
        self.max_temperature_c = max_temperature_c

    def _capture(
        self,
        config: VLLMCampaignConfig,
        *,
        phase: str,
        position: Literal["before", "after"],
    ) -> tuple[MI300XTelemetryProbeV1, ArtifactRef, str]:
        uuid = config.device.device_uuid
        if (
            _executable_sha256(self.telemetry_argv[0])
            != config.task.runtime.executable_sha256
        ):
            raise MI300XObservationError(
                "telemetry Python bytes differ from the pinned runtime"
            )
        environment = observation_environment(uuid)
        request_hash = observation_request_sha256(
            self.telemetry_argv,
            cwd=self.cwd,
            device_uuid=uuid,
            timeout_seconds=self.timeout_seconds,
        )
        result = self.runner.run(
            self.telemetry_argv,
            cwd=self.cwd,
            env=environment,
            unset_env=_CONFLICTING_GPU_ENV,
            timeout_seconds=self.timeout_seconds,
        )
        _require_executed_request(result, request_hash, "AMD SMI telemetry")
        parsed = _json_stdout(result, MI300XTelemetryProbeV1, "AMD SMI telemetry")
        _validate_probe_timestamp(result, parsed.captured_at, "AMD SMI telemetry")
        reference = self.store.save_evidence_json(
            config.task.id,
            f"vllm/telemetry/{phase}-{position}",
            {
                "command": _command_envelope(result),
                "parsed": parsed.model_dump(mode="json", by_alias=True),
            },
            producer="mi300x-telemetry-port",
        )
        return parsed, reference, request_hash

    def begin(
        self,
        config: VLLMCampaignConfig,
        *,
        phase: str,
        server_request_hash: str,
    ) -> _TelemetryStart:
        parsed, artifact, request_hash = self._capture(
            config, phase=phase, position="before"
        )
        return _TelemetryStart(
            phase=phase,
            server_request_hash=server_request_hash,
            request_hash=request_hash,
            parsed=parsed,
            artifact=artifact,
        )

    def finish(
        self,
        config: VLLMCampaignConfig,
        *,
        phase: str,
        server_request_hash: str,
        before: Any,
    ) -> VLLMEnvironmentCapture:
        if not isinstance(before, _TelemetryStart):
            raise MI300XObservationError("telemetry window start token is invalid")
        if before.phase != phase or before.server_request_hash != server_request_hash:
            raise MI300XObservationError("telemetry window binding changed")
        after, after_ref, request_hash = self._capture(
            config, phase=phase, position="after"
        )
        if request_hash != before.request_hash:
            raise MI300XObservationError("telemetry command identity changed")
        expected = config.device
        reasons: list[str] = []
        for label, sample in (("before", before.parsed), ("after", after)):
            if sample.device_uuid != expected.device_uuid:
                reasons.append(f"{label} device UUID differs")
            if sample.pci_bdf != expected.pci_bdf:
                reasons.append(f"{label} PCI BDF differs")
            if sample.compute_partition != expected.compute_partition:
                reasons.append(f"{label} compute partition differs")
            if sample.memory_partition != expected.memory_partition:
                reasons.append(f"{label} memory partition differs")
            if sample.partition_id != expected.partition_id:
                reasons.append(f"{label} partition id differs")
            required = (
                sample.current_gfxclk_mhz,
                sample.current_uclk_mhz,
                sample.socket_power_w,
                sample.hotspot_temperature_c,
                sample.throttle_active,
            )
            if any(value is None for value in required):
                reasons.append(f"{label} mandatory telemetry is missing")
            if sample.throttle_active is not False:
                reasons.append(f"{label} throttle state is active or unknown")
            if (
                sample.hotspot_temperature_c is not None
                and sample.hotspot_temperature_c > self.max_temperature_c
            ):
                reasons.append(f"{label} hotspot temperature exceeds limit")
            for kind, value in (
                ("power", sample.power_violation_percent),
                ("thermal", sample.thermal_violation_percent),
            ):
                if value is not None and value > 0:
                    reasons.append(f"{label} {kind} violation is nonzero")
        clock_tolerance = config.task.environment.max_telemetry_clock_drift_percent
        for name, left, right in (
            (
                "gfxclk",
                before.parsed.current_gfxclk_mhz,
                after.current_gfxclk_mhz,
            ),
            ("uclk", before.parsed.current_uclk_mhz, after.current_uclk_mhz),
        ):
            if left is None or right is None:
                continue
            denominator = max(abs(left), abs(right))
            drift = math.inf if denominator == 0 else abs(right - left) / denominator * 100
            if drift > clock_tolerance:
                reasons.append(
                    f"before/after {name} drift {drift:.3f}% exceeds "
                    f"{clock_tolerance:.3f}%"
                )
        gfx_values = [
            value
            for value in (
                before.parsed.current_gfxclk_mhz,
                after.current_gfxclk_mhz,
            )
            if value is not None
        ]
        uclk_values = [
            value
            for value in (
                before.parsed.current_uclk_mhz,
                after.current_uclk_mhz,
            )
            if value is not None
        ]
        fingerprint = EnvironmentFingerprint(
            values={
                **config.environment_coordinates,
                "server_request_hash": server_request_hash,
                "telemetry_command_sha256": request_hash,
                "telemetry_gfxclk_mean_mhz": (
                    f"{sum(gfx_values) / len(gfx_values):.6f}"
                    if gfx_values
                    else "missing"
                ),
                "telemetry_uclk_mean_mhz": (
                    f"{sum(uclk_values) / len(uclk_values):.6f}"
                    if uclk_values
                    else "missing"
                ),
                "telemetry_before_captured_at": before.parsed.captured_at.isoformat(),
                "telemetry_after_captured_at": after.captured_at.isoformat(),
            },
            telemetry_stable=not reasons,
            instability_reasons=sorted(set(reasons)),
            capture_id=hashlib.sha256(
                (
                    f"{before.artifact.sha256}\0{after_ref.sha256}\0"
                    f"{server_request_hash}"
                ).encode()
            ).hexdigest(),
            captured_at=utc_now(),
            source="observed",
        )
        return VLLMEnvironmentCapture(
            fingerprint=fingerprint,
            before_artifacts=(before.artifact,),
            after_artifacts=(after_ref,),
        )


def _enum_text(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value if isinstance(value.value, str) else value.name
    text = str(value).strip()
    for separator in (".", "::"):
        if separator in text:
            text = text.rsplit(separator, 1)[-1]
    return text.upper()


def _safe_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return _safe_json(value.value if isinstance(value.value, (str, int, float)) else value.name)
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_mi300x_product_name(market_name: str) -> str:
    """Map AMD SMI's bare-metal and SR-IOV names to one product family.

    Cloud MI300X devices are exposed as ``AMD Instinct MI300X VF``.  Keep the
    observed market name in the raw evidence, but use the stable product-family
    coordinate for configuration comparison.  Deliberately do not accept
    other gfx942 products such as MI300A or MI325X.
    """

    normalized = market_name.strip()
    if normalized not in {"AMD Instinct MI300X", "AMD Instinct MI300X VF"}:
        raise MI300XObservationError(
            "AMD SMI did not identify an AMD Instinct MI300X product family"
        )
    return "AMD Instinct MI300X"


def _select_amdsmi_device(amdsmi: Any, selector: str) -> Any:
    handles = amdsmi.amdsmi_get_processor_handles()
    selected: list[Any] = []
    physical_matches: list[Any] = []
    for handle in handles:
        enumeration = amdsmi.amdsmi_get_gpu_enumeration_info(handle)
        hip_uuid = enumeration.get("hip_uuid") if isinstance(enumeration, Mapping) else None
        if isinstance(hip_uuid, str) and hip_uuid == selector:
            selected.append(handle)
        if str(amdsmi.amdsmi_get_gpu_device_uuid(handle)) == selector:
            physical_matches.append(handle)
    if not selected:
        selected = physical_matches
    if len(selected) != 1:
        raise MI300XObservationError(
            "HIP UUID selector did not resolve exactly one AMD SMI partition; physical "
            "UUIDs shared by multiple pre-ROCm-7.13 partitions are not accepted"
        )
    return selected[0]


def _xcc_count(profile: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    active = profile.get("partition_profile")
    if not isinstance(active, Mapping):
        raise MI300XObservationError("AMD SMI lacks active accelerator profile")
    profile_index = active.get("profile_index")
    profiles = config.get("profiles")
    if not isinstance(profiles, list):
        raise MI300XObservationError("AMD SMI lacks accelerator profile config")
    matches = [
        item
        for item in profiles
        if isinstance(item, Mapping) and item.get("profile_index") == profile_index
    ]
    if len(matches) != 1:
        raise MI300XObservationError("active accelerator profile is ambiguous")
    resources = matches[0].get("resources")
    partitions = matches[0].get("num_partitions")
    if (
        not isinstance(resources, list)
        or isinstance(partitions, bool)
        or not isinstance(partitions, int)
    ):
        raise MI300XObservationError("accelerator profile resource evidence is incomplete")
    xcc = [
        item
        for item in resources
        if isinstance(item, Mapping) and _enum_text(item.get("resource_type")) == "XCC"
    ]
    if len(xcc) != 1:
        raise MI300XObservationError("accelerator profile lacks one XCC resource")
    per_partition = xcc[0].get("partition_resource")
    if isinstance(per_partition, bool) or not isinstance(per_partition, int):
        raise MI300XObservationError("accelerator profile XCC count is invalid")
    return per_partition * partitions


def collect_mi300x_static(selector: str) -> MI300XStaticProbeV1:
    """Collect the allowlisted AMD SMI identity fields for one UUID."""

    try:
        import amdsmi  # type: ignore[import-not-found]
    except ImportError as error:
        raise MI300XObservationError("the official amdsmi Python package is unavailable") from error
    initialized = False
    try:
        amdsmi.amdsmi_init()
        initialized = True
        handle = _select_amdsmi_device(amdsmi, selector)
        asic = amdsmi.amdsmi_get_gpu_asic_info(handle)
        enumeration = amdsmi.amdsmi_get_gpu_enumeration_info(handle)
        profile = amdsmi.amdsmi_get_gpu_accelerator_partition_profile(handle)
        profile_config = amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config(handle)
        memory_config = amdsmi.amdsmi_get_gpu_memory_partition_config(handle)
        compute = _enum_text(amdsmi.amdsmi_get_gpu_compute_partition(handle))
        memory = _enum_text(amdsmi.amdsmi_get_gpu_memory_partition(handle))
        profile_ids = profile.get("partition_id")
        if not isinstance(profile_ids, list) or not profile_ids:
            raise MI300XObservationError("AMD SMI lacks active partition id")
        partition_id = profile_ids[0]
        if isinstance(partition_id, bool) or not isinstance(partition_id, int):
            raise MI300XObservationError("AMD SMI partition id is invalid")
        bdf = str(amdsmi.amdsmi_get_gpu_device_bdf(handle))
        physical_uuid = str(amdsmi.amdsmi_get_gpu_device_uuid(handle))
        hip_uuid = enumeration.get("hip_uuid")
        if not isinstance(hip_uuid, str) or not hip_uuid.strip():
            if physical_uuid != selector:
                raise MI300XObservationError(
                    "AMD SMI enumeration lacks the HIP UUID needed for partition mapping"
                )
            hip_uuid = physical_uuid
        market = str(asic.get("market_name", ""))
        product_name = _normalize_mi300x_product_name(market)
        oam_id = enumeration.get("oam_id", asic.get("oam_id"))
        if isinstance(oam_id, bool) or not isinstance(oam_id, int):
            raise MI300XObservationError("AMD SMI OAM id is unavailable")
        return MI300XStaticProbeV1(
            schema="gpuopt.mi300x-static-probe.v1",
            captured_at=utc_now(),
            device_uuid=hip_uuid,
            pci_bdf=bdf,
            product_name=product_name,
            oam_id=oam_id,
            xcc_count=_xcc_count(profile, profile_config),
            compute_partition=compute,
            memory_partition=memory,
            partition_id=partition_id,
            raw={
                "asic": {
                    "market_name": market,
                    "oam_id": asic.get("oam_id"),
                    "device_id": asic.get("device_id"),
                    "subsystem_id": asic.get("subsystem_id"),
                    "num_compute_units": asic.get("num_compute_units"),
                    "target_graphics_version": asic.get("target_graphics_version"),
                },
                "enumeration": _safe_json(enumeration),
                "amdsmi_device_uuid": physical_uuid,
                "accelerator_partition": _safe_json(profile),
                "accelerator_partition_config": _safe_json(profile_config),
                "memory_partition_config": _safe_json(memory_config),
            },
        )
    except MI300XObservationError:
        raise
    except Exception as error:
        raise MI300XObservationError(f"AMD SMI inspection failed: {error}") from error
    finally:
        if initialized:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception:
                pass


class _HIPUUID(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


def _hip_uuid_token(raw_uuid: bytes) -> str:
    if len(raw_uuid) != 16:
        raise MI300XObservationError("HIP UUID payload must contain exactly 16 bytes")
    try:
        payload = raw_uuid.decode("ascii")
    except UnicodeDecodeError as error:
        raise MI300XObservationError("HIP UUID payload is not ASCII") from error
    if re.fullmatch(r"[0-9A-Fa-f]{16}", payload) is None:
        raise MI300XObservationError("HIP UUID payload is not a 16-digit hexadecimal token")
    return f"GPU-{payload.lower()}"


def _hip_runtime_identity() -> tuple[str, str]:
    library: Any | None = None
    errors: list[str] = []
    for name in ("libamdhip64.so.7", "libamdhip64.so"):
        try:
            library = ctypes.CDLL(name)
            break
        except OSError as error:
            errors.append(str(error))
    if library is None:
        raise MI300XObservationError(
            "HIP runtime library is unavailable: " + "; ".join(errors)
        )
    try:
        library.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.hipGetDeviceCount.restype = ctypes.c_int
        library.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.hipGetDevice.restype = ctypes.c_int
        library.hipDeviceGetUuid.argtypes = [ctypes.POINTER(_HIPUUID), ctypes.c_int]
        library.hipDeviceGetUuid.restype = ctypes.c_int
        library.hipDeviceGetPCIBusId.argtypes = [
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.hipDeviceGetPCIBusId.restype = ctypes.c_int
    except AttributeError as error:
        raise MI300XObservationError(
            "HIP runtime lacks UUID/BDF identity functions"
        ) from error
    count = ctypes.c_int()
    current = ctypes.c_int()
    if library.hipGetDeviceCount(ctypes.byref(count)) != 0 or count.value != 1:
        raise MI300XObservationError("HIP runtime did not expose exactly one device")
    if library.hipGetDevice(ctypes.byref(current)) != 0 or current.value != 0:
        raise MI300XObservationError("HIP current logical device is not zero")
    uuid = _HIPUUID()
    if library.hipDeviceGetUuid(ctypes.byref(uuid), 0) != 0:
        raise MI300XObservationError("HIP could not observe logical device zero UUID")
    raw_uuid = bytes(uuid.bytes)
    if not any(raw_uuid):
        raise MI300XObservationError("HIP returned an empty logical device UUID")
    bdf_buffer = ctypes.create_string_buffer(64)
    if library.hipDeviceGetPCIBusId(bdf_buffer, len(bdf_buffer), 0) != 0:
        raise MI300XObservationError("HIP could not observe logical device zero PCI BDF")
    try:
        bdf = bdf_buffer.value.decode("ascii")
    except UnicodeDecodeError as error:
        raise MI300XObservationError("HIP returned a non-ASCII PCI BDF") from error
    return _hip_uuid_token(raw_uuid), _bdf(bdf)


def _uuid_payload(value: str) -> str:
    payload = value.lower().removeprefix("gpu-")
    return "".join(character for character in payload if character in "0123456789abcdef")


def collect_mi300x_hip() -> MI300XHIPProbeV1:
    """Initialize HIP through PyTorch after ROCR UUID isolation is already set."""

    try:
        import importlib.metadata

        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise MI300XObservationError("PyTorch ROCm is unavailable") from error
    count = torch.cuda.device_count()
    logical = torch.cuda.current_device() if count else -1
    properties = torch.cuda.get_device_properties(0) if count == 1 else None
    gfx = None
    if properties is not None:
        gfx = getattr(properties, "gcnArchName", None) or getattr(
            properties, "gcn_arch_name", None
        )
    if isinstance(gfx, str):
        gfx = gfx.split(":", 1)[0]
    runtime_uuid, runtime_bdf = _hip_runtime_identity()
    selector = os.environ.get("ROCR_VISIBLE_DEVICES", "")
    if _uuid_payload(runtime_uuid) != _uuid_payload(selector):
        raise MI300XObservationError(
            "HIP logical device zero UUID differs from ROCR_VISIBLE_DEVICES"
        )
    device_name = ""
    if count == 1:
        device_name = str(torch.cuda.get_device_name(0)).strip()
    return MI300XHIPProbeV1(
        schema="gpuopt.mi300x-hip-probe.v1",
        captured_at=utc_now(),
        visible_device_count=count,
        logical_device_id=logical,
        device_name=device_name or "unavailable",
        gfx_target=gfx,
        device_uuid=runtime_uuid,
        pci_bdf=runtime_bdf,
        mapping_basis="hip-runtime-uuid+bdf",
        rocr_visible_devices=selector,
        hip_visible_devices=os.environ.get("HIP_VISIBLE_DEVICES"),
        hsa_visible_devices=os.environ.get("HSA_VISIBLE_DEVICES"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_device_ordinal=os.environ.get("GPU_DEVICE_ORDINAL"),
        rocm_version=str(torch.version.hip or ""),
        pytorch_version=str(torch.__version__),
        vllm_version=importlib.metadata.version("vllm"),
        python_version=".".join(str(value) for value in sys.version_info[:3]),
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _active(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return None


def collect_mi300x_telemetry(selector: str) -> MI300XTelemetryProbeV1:
    """Collect one allowlisted AMD SMI endpoint sample for stability gating."""

    try:
        import amdsmi  # type: ignore[import-not-found]
    except ImportError as error:
        raise MI300XObservationError("the official amdsmi Python package is unavailable") from error
    initialized = False
    try:
        amdsmi.amdsmi_init()
        initialized = True
        handle = _select_amdsmi_device(amdsmi, selector)
        metrics = amdsmi.amdsmi_get_gpu_metrics_info(handle)
        profile = amdsmi.amdsmi_get_gpu_accelerator_partition_profile(handle)
        profile_data = profile.get("partition_profile")
        ids = profile.get("partition_id")
        if not isinstance(profile_data, Mapping) or not isinstance(ids, list) or not ids:
            raise MI300XObservationError("AMD SMI telemetry lacks partition identity")
        memory = _enum_text(amdsmi.amdsmi_get_gpu_memory_partition(handle))
        compute = _enum_text(amdsmi.amdsmi_get_gpu_compute_partition(handle))
        hbm = metrics.get("temperature_hbm", [])
        if not isinstance(hbm, list):
            hbm = []
        return MI300XTelemetryProbeV1(
            schema="gpuopt.mi300x-telemetry.v1",
            captured_at=utc_now(),
            device_uuid=selector,
            pci_bdf=str(amdsmi.amdsmi_get_gpu_device_bdf(handle)),
            compute_partition=compute,
            memory_partition=memory,
            partition_id=ids[0],
            current_gfxclk_mhz=_number(metrics.get("current_gfxclk")),
            average_gfxclk_mhz=_number(metrics.get("average_gfxclk_frequency")),
            current_uclk_mhz=_number(metrics.get("current_uclk")),
            socket_power_w=_number(
                metrics.get("current_socket_power", metrics.get("average_socket_power"))
            ),
            hotspot_temperature_c=_number(metrics.get("temperature_hotspot")),
            hbm_temperatures_c=[value for item in hbm if (value := _number(item)) is not None],
            throttle_active=_active(
                metrics.get("throttle_status", metrics.get("indep_throttle_status"))
            ),
            power_violation_percent=_number(
                metrics.get("active_gfx_clk_below_host_limit_pwr")
            ),
            thermal_violation_percent=_number(
                metrics.get("active_gfx_clk_below_host_limit_thm")
            ),
            raw={
                "metrics": _safe_json(metrics),
                "accelerator_partition": _safe_json(profile),
                "amdsmi_device_uuid": str(amdsmi.amdsmi_get_gpu_device_uuid(handle)),
                "enumeration": _safe_json(
                    amdsmi.amdsmi_get_gpu_enumeration_info(handle)
                ),
            },
        )
    except MI300XObservationError:
        raise
    except Exception as error:
        raise MI300XObservationError(f"AMD SMI telemetry failed: {error}") from error
    finally:
        if initialized:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception:
                pass


def probe_main(argv: Sequence[str] | None = None) -> int:
    """Small ``python -m`` entry point used by immutable command coordinates."""

    import argparse

    parser = argparse.ArgumentParser(description="Read-only MI300X evidence probe")
    parser.add_argument("--mode", required=True, choices=("static", "hip", "telemetry"))
    parser.add_argument("--uuid")
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    expected_implementation = os.environ.get(_OBSERVATION_IMPLEMENTATION_ENV)
    if expected_implementation != observation_implementation_sha256():
        parser.error(
            f"{_OBSERVATION_IMPLEMENTATION_ENV} does not bind this probe implementation"
        )
    if arguments.mode != "hip" and not arguments.uuid:
        parser.error("--uuid is required for AMD SMI modes")
    if arguments.mode == "static":
        result = collect_mi300x_static(arguments.uuid)
    elif arguments.mode == "telemetry":
        result = collect_mi300x_telemetry(arguments.uuid)
    else:
        result = collect_mi300x_hip()
    print(result.model_dump_json(by_alias=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(probe_main())
