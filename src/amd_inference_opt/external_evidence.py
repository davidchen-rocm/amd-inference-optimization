"""Narrow, hash-bound importers for external AMD evidence artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from .models import StrictModel, utc_now


class ExternalEvidenceError(RuntimeError):
    """An external artifact is unsafe, malformed or lacks required evidence."""


class ImportStatus(StrEnum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class SourceArtifact(StrictModel):
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    ownership: Literal["external_provider", "gpuopt_store_copy"] = "external_provider"
    retention: str = "provider_owned"


class MagpieKernelGap(StrictModel):
    name: str
    calls: int = Field(ge=0)
    total_duration_us: float = Field(ge=0)
    average_duration_us: float | None = Field(default=None, ge=0)
    percent_total: float | None = Field(default=None, ge=0, le=100)


class EndpointTelemetryAggregate(StrictModel):
    samples: int = Field(ge=0)
    average_power_w: float | None = None
    maximum_power_w: float | None = None
    average_temperature_c: float | None = None
    maximum_temperature_c: float | None = None
    average_clock_mhz: float | None = None
    missing_fields: list[str] = Field(default_factory=list)


class MagpieBenchmarkEvidence(StrictModel):
    schema_name: Literal["gpuopt.magpie-benchmark-evidence.v1"] = Field(
        default="gpuopt.magpie-benchmark-evidence.v1", alias="schema"
    )
    imported_at: datetime = Field(default_factory=utc_now)
    status: ImportStatus
    provider: Literal["magpie"] = "magpie"
    framework: str | None = None
    model: str | None = None
    request_throughput: float | None = Field(default=None, ge=0)
    output_token_throughput: float | None = Field(default=None, ge=0)
    total_token_throughput: float | None = Field(default=None, ge=0)
    completed_requests: int | None = Field(default=None, ge=0)
    latency_ms: dict[str, float]
    gap_kernels: list[MagpieKernelGap]
    telemetry: EndpointTelemetryAggregate | None = None
    trace_outputs: list[str]
    source_artifact: SourceArtifact
    gate_eligible: bool
    warnings: list[str] = Field(default_factory=list)


class RooflineProvenanceKind(StrEnum):
    LIVE_MEASURED = "LIVE_MEASURED"
    STATIC_FALLBACK = "STATIC_FALLBACK"
    UNKNOWN = "UNKNOWN"


class RooflineArchitectureProvenance(StrictModel):
    kind: RooflineProvenanceKind
    architecture: str
    source: str
    source_sha256: str | None = None

    @field_validator("source_sha256")
    @classmethod
    def valid_hash(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("source_sha256 must be lowercase hexadecimal")
        return value


class TraceLensKernelRoofline(StrictModel):
    name: str
    calls: int | None = Field(default=None, ge=0)
    total_duration_us: float | None = Field(default=None, ge=0)
    average_duration_us: float | None = Field(default=None, ge=0)
    time_share_percent: float | None = Field(default=None, ge=0, le=100)
    arithmetic_intensity: float | None = Field(default=None, ge=0)
    achieved_bandwidth_gbps: float | None = Field(default=None, ge=0)
    achieved_tflops: float | None = Field(default=None, ge=0)
    efficiency_percent: float | None = Field(default=None, ge=0)
    bound: str | None = None
    m: int | None = Field(default=None, gt=0)
    n: int | None = Field(default=None, gt=0)
    k: int | None = Field(default=None, gt=0)
    params_json: dict[str, Any] | None = None


class TraceLensRooflineEvidence(StrictModel):
    schema_name: Literal["gpuopt.tracelens-roofline-evidence.v1"] = Field(
        default="gpuopt.tracelens-roofline-evidence.v1", alias="schema"
    )
    imported_at: datetime = Field(default_factory=utc_now)
    provider: Literal["tracelens"] = "tracelens"
    status: ImportStatus
    architecture: RooflineArchitectureProvenance
    phase: str | None = None
    kernels: list[TraceLensKernelRoofline]
    source_artifact: SourceArtifact
    gate_eligible: Literal[False] = False
    warnings: list[str] = Field(default_factory=list)


class IntelliKitTool(StrEnum):
    KERNCAP = "kerncap"
    METRIX = "metrix"
    LINEX = "linex"
    NEXUS = "nexus"
    ACCORDO = "accordo"
    UPROF_MCP = "uprof_mcp"


class IntelliKitEvidenceEnvelope(StrictModel):
    schema_name: Literal["gpuopt.intellikit-evidence-envelope.v1"] = Field(
        default="gpuopt.intellikit-evidence-envelope.v1", alias="schema"
    )
    imported_at: datetime = Field(default_factory=utc_now)
    provider: Literal["intellikit"] = "intellikit"
    tool: IntelliKitTool
    capability: str
    source_schema: str | None = None
    provider_status: str | None = None
    source_artifact: SourceArtifact
    gate_eligible: Literal[False] = False
    facts: dict[str, str | int | float | bool | None]
    warnings: list[str] = Field(default_factory=list)


def _safe_file(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    if selected.is_symlink() or not selected.is_file():
        raise ExternalEvidenceError("external evidence must be a regular non-symlink file")
    return selected.resolve(strict=True)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> SourceArtifact:
    return SourceArtifact(
        path=str(path),
        sha256=_hash_file(path),
        size_bytes=path.stat().st_size,
    )


def _finite(value: Any, *, minimum: float = 0) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= minimum else None


def _integer(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= minimum else None


def _flatten_latency(value: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for category, raw in value.items():
        if isinstance(raw, Mapping):
            for statistic, candidate in raw.items():
                parsed = _finite(candidate)
                if parsed is not None:
                    result[f"{category}.{statistic}"] = parsed
        else:
            parsed = _finite(raw)
            if parsed is not None:
                result[str(category)] = parsed
    return dict(sorted(result.items()))


def _telemetry(samples: Any) -> EndpointTelemetryAggregate | None:
    if isinstance(samples, Mapping):
        values = [samples]
    elif isinstance(samples, list):
        values = [item for item in samples if isinstance(item, Mapping)]
    else:
        values = []
    if not values:
        return None

    def numbers(*names: str) -> list[float]:
        result: list[float] = []
        for sample in values:
            for name in names:
                parsed = _finite(sample.get(name))
                if parsed is not None:
                    result.append(parsed)
                    break
        return result

    power = numbers("power_w", "power")
    temperature = numbers("temperature_c", "temperature")
    clock = numbers("clock_mhz", "sclk_mhz")
    missing = []
    if not power:
        missing.append("power")
    if not temperature:
        missing.append("temperature")
    if not clock:
        missing.append("clock")
    return EndpointTelemetryAggregate(
        samples=len(values),
        average_power_w=sum(power) / len(power) if power else None,
        maximum_power_w=max(power) if power else None,
        average_temperature_c=(sum(temperature) / len(temperature) if temperature else None),
        maximum_temperature_c=max(temperature) if temperature else None,
        average_clock_mhz=sum(clock) / len(clock) if clock else None,
        missing_fields=missing,
    )


def import_magpie_benchmark_report(path: str | Path) -> MagpieBenchmarkEvidence:
    """Import Magpie's report without treating wrapper success as correctness."""

    selected = _safe_file(path)
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalEvidenceError("invalid Magpie benchmark report") from error
    if not isinstance(payload, Mapping):
        raise ExternalEvidenceError("Magpie benchmark report must be an object")
    throughput = payload.get("throughput")
    throughput = throughput if isinstance(throughput, Mapping) else {}
    completed = _integer(throughput.get("completed_requests"))
    total = _finite(throughput.get("total_token_throughput"))
    output = _finite(throughput.get("output_throughput"))
    request = _finite(throughput.get("request_throughput"))
    warnings: list[str] = []
    if payload.get("success") is not True:
        warnings.append("Magpie wrapper did not report success")
    if completed is None or completed <= 0:
        warnings.append("completed request evidence is missing or zero")
    if total is None and output is None:
        warnings.append("positive token throughput is missing")

    gap = payload.get("gap_analysis")
    gap = gap if isinstance(gap, Mapping) else {}
    gap_rows: list[MagpieKernelGap] = []
    for raw in gap.get("top_kernels", []) if isinstance(gap.get("top_kernels"), list) else []:
        if not isinstance(raw, Mapping) or not str(raw.get("name", "")).strip():
            continue
        duration = _finite(raw.get("self_cuda_total_us", raw.get("total_duration_us")))
        calls = _integer(raw.get("calls"))
        if duration is None or calls is None:
            continue
        gap_rows.append(
            MagpieKernelGap(
                name=str(raw["name"]),
                calls=calls,
                total_duration_us=duration,
                average_duration_us=_finite(raw.get("avg_time_us"))
                or (duration / calls if calls else None),
                percent_total=_finite(raw.get("pct_total", raw.get("time_pct"))),
            )
        )
    analysis = payload.get("tracelens_analysis")
    analysis = analysis if isinstance(analysis, Mapping) else {}
    outputs = analysis.get("output_files")
    trace_outputs = [str(value) for value in outputs] if isinstance(outputs, list) else []
    gate_eligible = payload.get("success") is True and bool(
        completed and completed > 0 and (total is not None or output is not None)
    )
    status = (
        ImportStatus.COMPLETED
        if gate_eligible
        else ImportStatus.PARTIAL
        if any(value is not None for value in (total, output, request))
        else ImportStatus.FAILED
    )
    return MagpieBenchmarkEvidence(
        status=status,
        framework=str(payload.get("framework")) if payload.get("framework") else None,
        model=str(payload.get("model")) if payload.get("model") else None,
        request_throughput=request,
        output_token_throughput=output,
        total_token_throughput=total,
        completed_requests=completed,
        latency_ms=_flatten_latency(
            payload.get("latency") if isinstance(payload.get("latency"), Mapping) else {}
        ),
        gap_kernels=gap_rows,
        telemetry=_telemetry(payload.get("gpu_monitor")),
        trace_outputs=trace_outputs,
        source_artifact=_artifact(selected),
        gate_eligible=gate_eligible,
        warnings=warnings,
    )


def _normalized(row: Mapping[str, Any]) -> dict[str, Any]:
    return {re.sub(r"[^a-z0-9]", "", str(key).lower()): value for key, value in row.items()}


def _first(row: Mapping[str, Any], *names: str) -> Any:
    normalized = _normalized(row)
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if normalized.get(key) not in (None, ""):
            return normalized[key]
    return None


def import_tracelens_roofline_csv(
    path: str | Path,
    *,
    architecture: RooflineArchitectureProvenance,
    phase: str | None = None,
) -> TraceLensRooflineEvidence:
    """Import a compact TraceLens roofline CSV as analysis-only evidence."""

    selected = _safe_file(path)
    try:
        with selected.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise ExternalEvidenceError("invalid TraceLens roofline CSV") from error
    kernels: list[TraceLensKernelRoofline] = []
    warnings: list[str] = []
    for row in rows:
        name = _first(row, "Name", "Kernel Name", "kernel_name")
        if name is None:
            continue
        params: dict[str, Any] | None = None
        params_raw = _first(row, "params_json")
        if params_raw:
            try:
                candidate = json.loads(str(params_raw))
                params = candidate if isinstance(candidate, dict) else None
            except json.JSONDecodeError:
                warnings.append(f"invalid params_json for kernel {name}")
        kernels.append(
            TraceLensKernelRoofline(
                name=str(name),
                calls=_integer(_first(row, "Calls", "count")),
                total_duration_us=_finite(
                    _first(row, "Self CUDA total (us)", "total_duration_us", "total_time_us")
                ),
                average_duration_us=_finite(
                    _first(row, "Avg time (us)", "average_duration_us", "mean_us")
                ),
                time_share_percent=_finite(
                    _first(row, "% Total", "time_pct", "gpu_time_share_percent")
                ),
                arithmetic_intensity=_finite(
                    _first(row, "arithmetic_intensity", "arithmetic intensity")
                ),
                achieved_bandwidth_gbps=_finite(
                    _first(row, "achieved_bandwidth_gbps", "bandwidth_gbps")
                ),
                achieved_tflops=_finite(_first(row, "achieved_tflops", "tflops")),
                efficiency_percent=_finite(
                    _first(row, "efficiency_pct", "efficiency_percent")
                ),
                bound=(
                    str(bound_value)
                    if (bound_value := _first(row, "bound_type", "bound")) is not None
                    else None
                ),
                m=_integer(_first(row, "M", "gemm_m"), minimum=1),
                n=_integer(_first(row, "N", "gemm_n"), minimum=1),
                k=_integer(_first(row, "K", "gemm_k"), minimum=1),
                params_json=params,
            )
        )
    if not kernels:
        warnings.append("no recognizable kernel rows were imported")
    if architecture.kind == RooflineProvenanceKind.UNKNOWN:
        warnings.append("roofline architecture provenance is unknown")
    return TraceLensRooflineEvidence(
        status=ImportStatus.COMPLETED if kernels else ImportStatus.FAILED,
        architecture=architecture,
        phase=phase,
        kernels=kernels,
        source_artifact=_artifact(selected),
        warnings=warnings,
    )


def import_intellikit_json(
    path: str | Path,
    *,
    tool: IntelliKitTool,
    capability: str,
) -> IntelliKitEvidenceEnvelope:
    """Preserve IntelliKit JSON without silently promoting an unknown schema."""

    selected = _safe_file(path)
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalEvidenceError("invalid IntelliKit JSON evidence") from error
    if not isinstance(payload, Mapping):
        raise ExternalEvidenceError("IntelliKit evidence must be a JSON object")
    facts: dict[str, str | int | float | bool | None] = {}
    for name in (
        "kernel_name",
        "status",
        "result",
        "is_valid",
        "num_arrays_validated",
        "duration_ns",
        "total_duration_ns",
    ):
        value = payload.get(name)
        if isinstance(value, (str, int, float, bool)) or value is None:
            facts[name] = value
    source_schema = payload.get("schema") or payload.get("schema_version")
    return IntelliKitEvidenceEnvelope(
        tool=tool,
        capability=capability,
        source_schema=str(source_schema) if source_schema else None,
        provider_status=(str(payload.get("status")) if payload.get("status") else None),
        source_artifact=_artifact(selected),
        facts=facts,
        warnings=[
            "generic IntelliKit envelope is analysis-only until a tool-specific "
            "schema adapter validates completeness"
        ],
    )


__all__ = [
    "EndpointTelemetryAggregate",
    "ExternalEvidenceError",
    "ImportStatus",
    "IntelliKitEvidenceEnvelope",
    "IntelliKitTool",
    "MagpieBenchmarkEvidence",
    "MagpieKernelGap",
    "RooflineArchitectureProvenance",
    "RooflineProvenanceKind",
    "SourceArtifact",
    "TraceLensKernelRoofline",
    "TraceLensRooflineEvidence",
    "import_intellikit_json",
    "import_magpie_benchmark_report",
    "import_tracelens_roofline_csv",
]
