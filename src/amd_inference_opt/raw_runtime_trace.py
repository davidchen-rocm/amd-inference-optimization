"""Normalize bounded raw rocprof HIP/kernel/copy CSV evidence.

This is the unattended fallback for hosts where executing MCP calls cannot be
pre-approved one request at a time.  It never labels raw evidence as MCP output.
"""

from __future__ import annotations

import csv
import hashlib
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class RawRuntimeTraceError(RuntimeError):
    """The raw runtime trace is missing or exceeds its declared budget."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _rows(paths: list[Path], *, max_events: int) -> tuple[list[dict[str, str]], bool]:
    rows: list[dict[str, str]] = []
    exceeded = False
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if len(rows) >= max_events:
                    exceeded = True
                    continue
                rows.append(dict(row))
    return rows, exceeded


def _mean_gap(intervals: list[tuple[int, int]]) -> tuple[float | None, int]:
    if len(intervals) < 2:
        return None, 0
    ordered = sorted(intervals)
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps = [
        max(0, right[0] - left[1])
        for left, right in zip(merged, merged[1:], strict=False)
    ]
    return (statistics.fmean(gaps) if gaps else 0.0), sum(gaps)


def normalize_raw_runtime_trace(
    trace_root: str | Path,
    *,
    workload_exit_code: int,
    max_events_per_type: int = 250_000,
) -> dict[str, Any]:
    """Project public rocprof CSV columns into runtime and audit aggregates."""

    root = Path(trace_root).resolve()
    if not root.is_dir() or max_events_per_type <= 0:
        raise RawRuntimeTraceError("trace root and event budget must be valid")
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    by_kind = {
        "hip": [path for path in files if "hip_api_trace" in path.name],
        "kernel": [path for path in files if "kernel_trace" in path.name],
        "copy": [path for path in files if "memory_copy_trace" in path.name],
    }
    missing = [kind for kind, paths in by_kind.items() if not paths]
    hip_rows, hip_limited = _rows(by_kind["hip"], max_events=max_events_per_type)
    kernel_rows, kernel_limited = _rows(by_kind["kernel"], max_events=max_events_per_type)
    copy_rows, copy_limited = _rows(by_kind["copy"], max_events=max_events_per_type)
    malformed = 0
    api_counts: Counter[str] = Counter()
    api_durations: dict[str, int] = defaultdict(int)
    launch_intervals: list[tuple[int, int]] = []
    allocation_names = {
        "hipMalloc",
        "hipFree",
        "hipHostMalloc",
        "hipHostFree",
        "hipMallocAsync",
        "hipFreeAsync",
    }
    for row in hip_rows:
        name = row.get("Function")
        start = _integer(row.get("Start_Timestamp"))
        end = _integer(row.get("End_Timestamp"))
        if not name or start is None or end is None or end < start:
            malformed += 1
            continue
        api_counts[name] += 1
        api_durations[name] += end - start
        if name in {"hipLaunchKernel", "hipGraphLaunch"}:
            launch_intervals.append((start, end))
    kernel_intervals: list[tuple[int, int]] = []
    for row in kernel_rows:
        start = _integer(row.get("Start_Timestamp"))
        end = _integer(row.get("End_Timestamp"))
        if start is None or end is None or end < start:
            malformed += 1
            continue
        kernel_intervals.append((start, end))
    copy_counts: Counter[str] = Counter()
    copy_durations: dict[str, int] = defaultdict(int)
    for row in copy_rows:
        direction = row.get("Direction")
        start = _integer(row.get("Start_Timestamp"))
        end = _integer(row.get("End_Timestamp"))
        if not direction or start is None or end is None or end < start:
            malformed += 1
            continue
        copy_counts[direction] += 1
        copy_durations[direction] += end - start
    cpu_gap, cpu_gap_total = _mean_gap(launch_intervals)
    gpu_gap, gpu_gap_total = _mean_gap(kernel_intervals)
    all_intervals = [*launch_intervals, *kernel_intervals]
    trace_span_ns = (
        max(end for _, end in all_intervals) - min(start for start, _ in all_intervals)
        if all_intervals
        else None
    )
    limited = hip_limited or kernel_limited or copy_limited
    warnings = [f"missing {kind} CSV" for kind in missing]
    if limited:
        warnings.append("event normalization budget exceeded")
    if malformed:
        warnings.append(f"ignored {malformed} malformed rows")
    trace_complete = not warnings and workload_exit_code == 0
    return {
        "schema": "gpuopt.raw-runtime-trace.v1",
        "source": "raw_rocprofv3",
        "trace_status": "completed" if trace_complete else "partial",
        "workload_exit_code": workload_exit_code,
        "hip_api_counts": dict(sorted(api_counts.items())),
        "hip_api_duration_ns": dict(sorted(api_durations.items())),
        "hip_kernel_launch_count": api_counts["hipLaunchKernel"],
        "graph_launch_count": api_counts["hipGraphLaunch"],
        "kernel_dispatch_count": len(kernel_intervals),
        "memory_allocation_count": sum(api_counts[name] for name in allocation_names),
        "memory_allocation_aggregates": [
            {
                "operation": name,
                "count": api_counts[name],
                "total_duration_ns": api_durations[name],
                "total_bytes": None,
                "repeated_size_bytes": None,
                "steady_state": None,
            }
            for name in sorted(allocation_names)
            if api_counts[name]
        ],
        "memory_copy_count": sum(copy_counts.values()),
        "memory_copy_aggregates": [
            {
                "operation": direction,
                "count": count,
                "total_duration_ns": copy_durations[direction],
                "total_bytes": None,
                "repeated_size_bytes": None,
                "steady_state": None,
            }
            for direction, count in sorted(copy_counts.items())
        ],
        "memory_copy_bytes_by_direction": {},
        "cpu_launch_gap_mean_ns": cpu_gap,
        "cpu_launch_gap_total_ns": cpu_gap_total,
        "gpu_idle_gap_mean_ns": gpu_gap,
        "gpu_idle_gap_total_ns": gpu_gap_total,
        "trace_span_ns": trace_span_ns,
        "warning_details": warnings,
        "capture_usage": {
            "hip_api_events": len(hip_rows),
            "kernel_events": len(kernel_rows),
            "memory_copy_events": len(copy_rows),
            "max_events_per_type": max_events_per_type,
        },
        "raw_artifacts": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }


__all__ = ["RawRuntimeTraceError", "normalize_raw_runtime_trace"]
