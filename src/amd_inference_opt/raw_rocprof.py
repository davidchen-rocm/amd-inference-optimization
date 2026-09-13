"""Explicitly authorized raw rocprofv3 fallback for kernel-timing evidence.

This adapter is intentionally small: it launches only the fixed kernel-trace
preset, preserves the raw files, and projects dispatch CSV rows into the same
aggregate fields consumed by the workflow.  It does not impersonate an MCP call
or manufacture an MCP approval receipt.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .command import CommandResult, CommandRunner, validate_argv


class RawRocprofError(RuntimeError):
    """The bounded raw profiler request or its output is invalid."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _field(row: Mapping[str, str], *names: str) -> str | None:
    normalized = {
        re.sub(r"[^a-z0-9]", "", key.lower()): value for key, value in row.items()
    }
    for name in names:
        value = normalized.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if value is not None and value.strip():
            return value.strip()
    return None


def _integer(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value, 0)
    except ValueError:
        try:
            floating = float(value)
        except ValueError:
            return None
        if not math.isfinite(floating) or not floating.is_integer():
            return None
        parsed = int(floating)
    return parsed


def _percentile(samples: list[int], fraction: float) -> int | None:
    if not samples:
        return None
    ordered = sorted(samples)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


@dataclass(frozen=True)
class RawArtifact:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class RawProfileResult:
    schema: str
    source: str
    authorization: dict[str, str]
    profiler_argv: tuple[str, ...]
    workload_argv: tuple[str, ...]
    cwd: str
    environment: dict[str, str]
    unset_environment: tuple[str, ...]
    environment_hash: str
    command: CommandResult
    kernel_evidence: dict[str, Any]
    raw_artifacts: tuple[RawArtifact, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_kernel_csvs(
    trace_root: str | Path,
    *,
    max_trace_bytes: int,
    max_trace_files: int,
    max_events_per_type: int,
    max_percentile_samples_per_kernel: int,
) -> tuple[dict[str, Any], tuple[RawArtifact, ...]]:
    """Normalize bounded kernel dispatch CSV files and hash every raw artifact."""

    limits = {
        "max_trace_bytes": max_trace_bytes,
        "max_trace_files": max_trace_files,
        "max_events_per_type": max_events_per_type,
        "max_percentile_samples_per_kernel": max_percentile_samples_per_kernel,
    }
    if any(isinstance(value, bool) or value <= 0 for value in limits.values()):
        raise RawRocprofError("raw rocprof limits must be positive integers")
    root = Path(trace_root).resolve()
    if not root.is_dir():
        raise RawRocprofError(f"raw rocprof output directory does not exist: {root}")

    all_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    artifacts = tuple(
        RawArtifact(
            path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )
        for path in all_files
    )
    csv_files = [
        path
        for path in all_files
        if path.suffix.lower() == ".csv" and "kernel" in path.name.lower()
    ]
    selected: list[Path] = []
    selected_bytes = 0
    for path in csv_files:
        size = path.stat().st_size
        if len(selected) >= max_trace_files or selected_bytes + size > max_trace_bytes:
            continue
        selected.append(path)
        selected_bytes += size

    aggregates: dict[str, dict[str, Any]] = {}
    valid_rows = 0
    observed_valid_rows = 0
    malformed_rows = 0
    event_limit_reached = False
    for path in selected:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = _field(row, "Kernel_Name", "KernelName", "Kernel Name")
                start = _integer(
                    _field(row, "Start_Timestamp", "StartTimestamp", "StartNs")
                )
                end = _integer(_field(row, "End_Timestamp", "EndTimestamp", "EndNs"))
                if name is None or start is None or end is None or end < start:
                    malformed_rows += 1
                    continue
                observed_valid_rows += 1
                if valid_rows >= max_events_per_type:
                    event_limit_reached = True
                    continue
                valid_rows += 1
                duration = end - start
                agent = _field(row, "Agent_Id", "AgentId", "Agent")
                grid = tuple(
                    _integer(_field(row, f"Grid_Size_{axis}", f"GridSize{axis}"))
                    for axis in "XYZ"
                )
                workgroup = tuple(
                    _integer(
                        _field(row, f"Workgroup_Size_{axis}", f"WorkgroupSize{axis}")
                    )
                    for axis in "XYZ"
                )
                resource_usage = {
                    "vgpr_count": _integer(_field(row, "VGPR_Count", "VGPRCount")),
                    "accum_vgpr_count": _integer(
                        _field(row, "Accum_VGPR_Count", "AccumVGPRCount")
                    ),
                    "sgpr_count": _integer(_field(row, "SGPR_Count", "SGPRCount")),
                    "lds_bytes": _integer(
                        _field(row, "LDS_Block_Size", "LDSBlockSize")
                    ),
                    "scratch_bytes": _integer(_field(row, "Scratch_Size", "ScratchSize")),
                }
                identity = {
                    "name": name,
                    "agent": agent,
                    "grid": grid,
                    "workgroup": workgroup,
                    "resource_usage": resource_usage,
                }
                aggregate_key = _canonical_sha256(identity)
                aggregate = aggregates.setdefault(
                    aggregate_key,
                    {
                        "name": name,
                        "kernel_id": "sha256:" + aggregate_key,
                        "agent": agent,
                        "grid": list(grid),
                        "workgroup": list(workgroup),
                        "resource_usage": resource_usage,
                        "dispatch_count": 0,
                        "total_duration_ns": 0,
                        "min_duration_ns": duration,
                        "max_duration_ns": duration,
                        "samples": [],
                    },
                )
                aggregate["dispatch_count"] += 1
                aggregate["total_duration_ns"] += duration
                aggregate["min_duration_ns"] = min(aggregate["min_duration_ns"], duration)
                aggregate["max_duration_ns"] = max(aggregate["max_duration_ns"], duration)
                if len(aggregate["samples"]) < max_percentile_samples_per_kernel:
                    aggregate["samples"].append(duration)

    total_duration = sum(item["total_duration_ns"] for item in aggregates.values())
    kernels: list[dict[str, Any]] = []
    for aggregate in aggregates.values():
        samples = aggregate.pop("samples")
        count = aggregate["dispatch_count"]
        percentiles_reliable = len(samples) == count
        aggregate.update(
            {
                "average_duration_ns": aggregate["total_duration_ns"] / count,
                "p50_duration_ns": _percentile(samples, 0.50),
                "p95_duration_ns": _percentile(samples, 0.95),
                "percentiles_reliable": percentiles_reliable,
                "gpu_kernel_time_share_percent": (
                    aggregate["total_duration_ns"] / total_duration * 100
                    if total_duration
                    else 0.0
                ),
            }
        )
        kernels.append(aggregate)
    kernels.sort(key=lambda item: item["total_duration_ns"], reverse=True)

    total_csv_bytes = sum(path.stat().st_size for path in csv_files)
    byte_coverage = selected_bytes / total_csv_bytes * 100 if total_csv_bytes else 0.0
    event_coverage = (
        valid_rows / observed_valid_rows * 100 if observed_valid_rows else 0.0
    )
    coverage = min(byte_coverage, event_coverage) if csv_files else 0.0
    files_complete = len(selected) == len(csv_files)
    aggregate_complete = bool(kernels) and files_complete and not event_limit_reached
    warnings: list[str] = []
    if not csv_files:
        warnings.append("no kernel trace CSV was produced")
    if not files_complete:
        warnings.append("raw kernel CSV coverage was limited by trace file/byte budget")
    if event_limit_reached:
        warnings.append("raw kernel event budget was exceeded")
    if malformed_rows:
        warnings.append(f"ignored {malformed_rows} malformed kernel rows")
    return (
        {
            "schema": "gpuopt.raw-kernel-evidence.v1",
            "preset": "kernel-timing",
            "status": "completed" if aggregate_complete else "partial",
            "profiler": {"tool": "rocprofv3", "status": "completed"},
            "workload": {"status": "completed", "exit_code": 0},
            "kernels": kernels,
            "aggregate_timing_complete": aggregate_complete,
            "hotspot_ranking_reliable": aggregate_complete,
            "ranking_complete": aggregate_complete,
            "coverage_percent": coverage,
            "coverage_basis": "minimum_of_kernel_csv_bytes_and_valid_dispatch_rows",
            "byte_coverage_percent": byte_coverage,
            "event_coverage_percent": event_coverage,
            "share_denominator_ns": total_duration,
            "capture_budget": limits,
            "capture_usage": {
                "trace_bytes": sum(item.size_bytes for item in artifacts),
                "trace_files": len(artifacts),
                "kernel_events_observed": observed_valid_rows,
                "kernel_events_normalized": valid_rows,
            },
            "malformed_rows": malformed_rows,
            "warnings": warnings,
        },
        artifacts,
    )


def normalize_existing_trace(
    trace_root: str | Path,
    *,
    profiler_argv: Sequence[str],
    workload_argv: Sequence[str],
    cwd: str | Path,
    environment: Mapping[str, str],
    unset_environment: Sequence[str],
    capture_budget: Mapping[str, int],
    normalization_max_events: int,
    workload_exit_code: int = 0,
    authorization_reference: str,
) -> dict[str, Any]:
    """Normalize an already captured trace without launching a GPU workload.

    ``capture_budget`` records the original capture coordinate.  A larger
    ``normalization_max_events`` may be selected after capture so complete raw
    CSV data is fully aggregated instead of inheriting a percentile/sample cap.
    Both values remain visible in the persisted evidence.
    """

    required_limits = {
        "max_trace_bytes",
        "max_trace_files",
        "max_events_per_type",
        "max_percentile_samples_per_kernel",
    }
    if set(capture_budget) != required_limits:
        raise RawRocprofError(
            "capture_budget must define exactly: " + ", ".join(sorted(required_limits))
        )
    if not authorization_reference.strip():
        raise RawRocprofError("authorization_reference must not be empty")
    profiler = validate_argv(profiler_argv)
    workload = validate_argv(workload_argv)
    evidence, artifacts = normalize_kernel_csvs(
        trace_root,
        max_trace_bytes=int(capture_budget["max_trace_bytes"]),
        max_trace_files=int(capture_budget["max_trace_files"]),
        max_events_per_type=normalization_max_events,
        max_percentile_samples_per_kernel=int(
            capture_budget["max_percentile_samples_per_kernel"]
        ),
    )
    evidence["capture_budget"] = dict(capture_budget)
    evidence["normalization_budget"] = {
        "max_events": normalization_max_events,
        "purpose": "offline complete aggregate scan",
    }
    workload_status = "completed" if workload_exit_code == 0 else "failed"
    evidence["workload"] = {
        "status": workload_status,
        "exit_code": workload_exit_code,
    }
    if workload_exit_code != 0:
        evidence["status"] = "failed"
        evidence["aggregate_timing_complete"] = False
        evidence["hotspot_ranking_reliable"] = False
        evidence["ranking_complete"] = False
        evidence["warnings"].append("captured workload exit code was non-zero")
    return {
        "schema": "gpuopt.raw-rocprof-result.v1",
        "source": "raw_rocprofv3",
        "capture_origin": "existing_trace",
        "authorization": {
            "mode": "explicit_reference",
            "reference": authorization_reference,
            "mcp_approval_created": False,
        },
        "trace_root": str(Path(trace_root).resolve()),
        "profiler_argv": list(profiler),
        "workload_argv": list(workload),
        "cwd": str(Path(cwd).resolve()),
        "environment": dict(environment),
        "unset_environment": list(unset_environment),
        "environment_hash": _canonical_sha256(
            {
                "environment": dict(sorted(environment.items())),
                "unset_environment": sorted(unset_environment),
            }
        ),
        "kernel_evidence": evidence,
        "raw_artifacts": [asdict(artifact) for artifact in artifacts],
    }


class RawRocprofAdapter:
    """Run one fixed rocprofv3 kernel trace after framework-level authorization."""

    def __init__(
        self,
        command_runner: CommandRunner | None = None,
        *,
        rocprofv3_path: str | Path = "/usr/bin/rocprofv3",
    ) -> None:
        self.commands = command_runner or CommandRunner()
        self.rocprofv3_path = Path(rocprofv3_path).resolve()

    def profile(
        self,
        workload_argv: Sequence[str],
        *,
        cwd: str | Path,
        environment: Mapping[str, str],
        unset_environment: Sequence[str],
        output_dir: str | Path,
        timeout_seconds: float,
        max_trace_bytes: int,
        max_trace_files: int,
        max_events_per_type: int,
        max_percentile_samples_per_kernel: int,
    ) -> RawProfileResult:
        workload = validate_argv(workload_argv)
        if not self.rocprofv3_path.is_file():
            raise RawRocprofError(f"rocprofv3 does not exist: {self.rocprofv3_path}")
        root = Path(output_dir).resolve()
        if root.exists():
            raise RawRocprofError(f"raw rocprof output already exists: {root}")
        trace_dir = root / "trace"
        trace_dir.mkdir(parents=True)
        profiler_argv = (
            str(self.rocprofv3_path),
            "--kernel-trace",
            "--stats",
            "--output-format",
            "csv",
            "--output-directory",
            str(trace_dir),
            "--",
            *workload,
        )
        command = self.commands.run(
            profiler_argv,
            cwd=cwd,
            env=environment,
            unset_env=unset_environment,
            timeout_seconds=timeout_seconds,
            stdout_path=root / "stdout.log",
            stderr_path=root / "stderr.log",
        )
        evidence, artifacts = normalize_kernel_csvs(
            root,
            max_trace_bytes=max_trace_bytes,
            max_trace_files=max_trace_files,
            max_events_per_type=max_events_per_type,
            max_percentile_samples_per_kernel=max_percentile_samples_per_kernel,
        )
        workload_status = (
            "timeout" if command.timed_out else "completed" if command.succeeded else "failed"
        )
        evidence["workload"] = {
            "status": workload_status,
            "exit_code": command.exit_code,
        }
        evidence["profiler"]["status"] = (
            "timeout" if command.timed_out else "completed" if command.succeeded else "failed"
        )
        if not command.succeeded:
            evidence["status"] = workload_status
            evidence["aggregate_timing_complete"] = False
            evidence["hotspot_ranking_reliable"] = False
            evidence["ranking_complete"] = False
            evidence["warnings"].append("raw rocprofv3 command did not complete successfully")
        return RawProfileResult(
            schema="gpuopt.raw-rocprof-result.v1",
            source="raw_rocprofv3",
            authorization={
                "mode": "task_configuration",
                "field": "metadata.raw_rocprof_authorized",
                "value": "true",
            },
            profiler_argv=profiler_argv,
            workload_argv=workload,
            cwd=str(Path(cwd).resolve()),
            environment=dict(environment),
            unset_environment=tuple(unset_environment),
            environment_hash=_canonical_sha256(
                {
                    "environment": dict(sorted(environment.items())),
                    "unset_environment": sorted(unset_environment),
                }
            ),
            command=command,
            kernel_evidence=evidence,
            raw_artifacts=artifacts,
        )


__all__ = [
    "RawArtifact",
    "RawProfileResult",
    "RawRocprofAdapter",
    "RawRocprofError",
    "normalize_kernel_csvs",
    "normalize_existing_trace",
]
