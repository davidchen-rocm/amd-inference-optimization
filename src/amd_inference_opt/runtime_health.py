"""Bounded read-only health evidence for the local workflow runtime.

This intentionally does not collect GPU telemetry.  ROCm Issue Agent owns GPU
state; this probe covers the host/runtime failure modes around an experiment.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from .evidence_catalog import EvidenceCapabilitySnapshot, probe_evidence_capabilities
from .models import StrictModel, utc_now

_PROCESS_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "inference_server": (
        "llama-server",
        "sglang.srt",
        "sglang.launch_server",
        "vllm.entrypoints",
        "vllm serve",
    ),
    "benchmark": ("llama-bench", "Magpie", "benchmark_serving", "inferencex"),
    "profiler": ("rocprofv3", "rocprof-compute", "rocprof-sys"),
    "build": ("hipcc", "amdclang", "cmake --build", "ninja"),
    "quality": ("llama-perplexity", "q8_runtime_quality_eval"),
}


class MountHealth(StrictModel):
    path: str
    total_bytes: int | None = Field(default=None, ge=0)
    used_bytes: int | None = Field(default=None, ge=0)
    free_bytes: int | None = Field(default=None, ge=0)
    used_percent: float | None = Field(default=None, ge=0, le=100)
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    warning: str | None = None


class ProcessHealth(StrictModel):
    pid: int = Field(gt=0)
    category: str
    matched_pattern: str
    command_name: str


class PathHealth(StrictModel):
    path: str
    status: Literal["AVAILABLE", "MISSING", "UNREADABLE"]
    stat_latency_ms: float | None = Field(default=None, ge=0)


class RuntimeHealthSnapshot(StrictModel):
    schema_name: Literal["gpuopt.runtime-health.v1"] = Field(
        default="gpuopt.runtime-health.v1", alias="schema"
    )
    collected_at: datetime = Field(default_factory=utc_now)
    host_pid: int = Field(gt=0)
    mounts: list[MountHealth]
    processes_known: bool
    relevant_processes: list[ProcessHealth]
    dependency_capabilities: EvidenceCapabilitySnapshot
    paths: list[PathHealth]
    warnings: list[str] = Field(default_factory=list)


def _mount(path: Path) -> MountHealth:
    try:
        usage = shutil.disk_usage(path)
    except OSError as error:
        return MountHealth(
            path=str(path),
            status="UNAVAILABLE",
            warning=str(error),
        )
    used = usage.total - usage.free
    return MountHealth(
        path=str(path),
        total_bytes=usage.total,
        used_bytes=used,
        free_bytes=usage.free,
        used_percent=(used / usage.total * 100 if usage.total else None),
        status="AVAILABLE",
    )


def _processes(proc_root: Path, *, limit: int) -> tuple[bool, list[ProcessHealth], list[str]]:
    if limit <= 0:
        raise ValueError("process limit must be positive")
    if not proc_root.is_dir():
        return False, [], [f"process filesystem is unavailable: {proc_root}"]
    values: list[ProcessHealth] = []
    warnings: list[str] = []
    try:
        entries = sorted(
            (item for item in proc_root.iterdir() if item.name.isdigit()),
            key=lambda item: int(item.name),
        )
    except OSError as error:
        return False, [], [f"process filesystem could not be listed: {error}"]
    for entry in entries:
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if not command:
            continue
        matched: tuple[str, str] | None = None
        lowered = command.lower()
        for category, patterns in _PROCESS_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in lowered:
                    matched = (category, pattern)
                    break
            if matched:
                break
        if matched is None:
            continue
        command_name = Path(command.split()[0]).name[:120]
        values.append(
            ProcessHealth(
                pid=int(entry.name),
                category=matched[0],
                matched_pattern=matched[1],
                command_name=command_name,
            )
        )
        if len(values) >= limit:
            warnings.append(f"relevant process list was truncated at {limit} entries")
            break
    return True, values, warnings


def _path_health(path: Path) -> PathHealth:
    start = time.monotonic_ns()
    try:
        path.stat()
    except FileNotFoundError:
        return PathHealth(path=str(path), status="MISSING")
    except OSError:
        return PathHealth(path=str(path), status="UNREADABLE")
    latency = (time.monotonic_ns() - start) / 1_000_000
    return PathHealth(path=str(path), status="AVAILABLE", stat_latency_ms=latency)


def capture_runtime_health(
    *,
    workspace: str | Path,
    proc_root: str | Path = "/proc",
    extra_paths: Sequence[str | Path] = (),
    command_overrides: Mapping[str, str] | None = None,
    process_limit: int = 256,
) -> RuntimeHealthSnapshot:
    """Capture host health without launching a workload or reading secrets."""

    selected_workspace = Path(workspace).expanduser().resolve()
    mount_paths = [selected_workspace]
    shared_memory = Path("/dev/shm")
    if shared_memory.exists() and shared_memory != selected_workspace:
        mount_paths.append(shared_memory)
    known, processes, process_warnings = _processes(
        Path(proc_root), limit=process_limit
    )
    paths = [_path_health(Path(value).expanduser()) for value in extra_paths]
    warnings = list(process_warnings)
    mounts = [_mount(path) for path in mount_paths]
    for item in mounts:
        if item.status == "AVAILABLE" and item.used_percent is not None and item.used_percent >= 95:
            warnings.append(f"mount usage is at least 95%: {item.path}")
    return RuntimeHealthSnapshot(
        host_pid=os.getpid(),
        mounts=mounts,
        processes_known=known,
        relevant_processes=processes,
        dependency_capabilities=probe_evidence_capabilities(
            command_overrides=command_overrides
        ),
        paths=paths,
        warnings=warnings,
    )


__all__ = [
    "MountHealth",
    "PathHealth",
    "ProcessHealth",
    "RuntimeHealthSnapshot",
    "capture_runtime_health",
]
