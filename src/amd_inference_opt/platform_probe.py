# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Adapted from AMD-AGI/Hyperloom v1.0.0b2 common/platform_probe.py.
# Local changes: removed Hyperloom provenance imports, added gpuopt Pydantic
# evidence, and limited the probe to read-only local host facts.
"""Read-only local host topology evidence with per-field degradation."""

from __future__ import annotations

import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from .architecture import ArchitectureProfile, architecture_profile
from .models import GPUTarget, StrictModel, utc_now

DEFAULT_ROOT = Path("/")
_CPU_ROOT = "sys/devices/system/cpu"
_NODE_ROOT = "sys/devices/system/node"
_AMDGPU_DRIVER_ROOT = "sys/bus/pci/drivers/amdgpu"


def read_kernel_file(path: Path | str, *, root: Path = DEFAULT_ROOT) -> str:
    candidate = Path(path)
    target = candidate if candidate.is_absolute() else root / candidate
    try:
        return target.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def sysfs_available(*, root: Path = DEFAULT_ROOT) -> bool:
    return bool(read_kernel_file(f"{_CPU_ROOT}/smt/active", root=root)) or (
        root / _CPU_ROOT / "cpu0"
    ).exists()


def _smt_state(*, root: Path) -> str | None:
    raw = read_kernel_file(f"{_CPU_ROOT}/smt/active", root=root)
    return {"1": "on", "0": "off"}.get(raw)


def _socket_count(*, root: Path) -> int | None:
    try:
        values = {
            read_kernel_file(path)
            for path in (root / _CPU_ROOT).glob(
                "cpu[0-9]*/topology/physical_package_id"
            )
        }
    except OSError:
        return None
    return len(values - {""}) or None


def _numa_node_count(*, root: Path) -> int | None:
    try:
        return len(list((root / _NODE_ROOT).glob("node[0-9]*"))) or None
    except OSError:
        return None


def _nodes_per_socket(*, root: Path) -> str | None:
    sockets = _socket_count(root=root)
    nodes = _numa_node_count(root=root)
    if not sockets or not nodes:
        return None
    return f"NPS{nodes // sockets}"


def _cpu_model(*, root: Path) -> str:
    for line in read_kernel_file("proc/cpuinfo", root=root).splitlines():
        if line.startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return "unknown"


def _amdgpu_device_count(*, root: Path) -> int | None:
    try:
        return len(list((root / _AMDGPU_DRIVER_ROOT).glob("*:*:*.*"))) or None
    except OSError:
        return None


class CpuPlatformEvidence(StrictModel):
    model: str
    smt: Literal["on", "off"] | None = None
    sockets: int | None = Field(default=None, ge=1)
    numa_nodes: int | None = Field(default=None, ge=1)
    nps: str | None = None
    governor: str
    boost: Literal["on", "off", "unknown"]


class LocalPlatformEvidence(StrictModel):
    schema_name: Literal["gpuopt.local-platform.v1"] = Field(
        default="gpuopt.local-platform.v1", alias="schema"
    )
    captured_at: datetime = Field(default_factory=utc_now)
    status: Literal["ok", "partial", "unavailable"]
    host: str
    kernel: str
    cpu: CpuPlatformEvidence | None = None
    amdgpu_host_device_count: int | None = Field(default=None, ge=1)
    architecture: ArchitectureProfile
    warnings: list[str] = Field(default_factory=list)


def probe_local_platform(
    gpu: GPUTarget,
    *,
    root: Path = DEFAULT_ROOT,
) -> LocalPlatformEvidence:
    """Collect provenance only; this never replaces ROCm Issue Agent evidence."""

    architecture = architecture_profile(
        gpu.gfx_target,
        board_type=gpu.board_type,
    )
    host = socket.gethostname() if root == DEFAULT_ROOT else "fixture-host"
    kernel = (
        platform.release()
        if root == DEFAULT_ROOT
        else read_kernel_file("proc/sys/kernel/osrelease", root=root) or "unknown"
    )
    if not sysfs_available(root=root):
        return LocalPlatformEvidence(
            status="unavailable",
            host=host,
            kernel=kernel,
            architecture=architecture,
            warnings=["local CPU sysfs is unavailable"],
        )
    governor = (
        read_kernel_file(
            f"{_CPU_ROOT}/cpu0/cpufreq/scaling_governor", root=root
        )
        or "unknown"
    )
    boost = {
        "1": "on",
        "0": "off",
    }.get(read_kernel_file(f"{_CPU_ROOT}/cpufreq/boost", root=root), "unknown")
    cpu = CpuPlatformEvidence(
        model=_cpu_model(root=root),
        smt=_smt_state(root=root),
        sockets=_socket_count(root=root),
        numa_nodes=_numa_node_count(root=root),
        nps=_nodes_per_socket(root=root),
        governor=governor,
        boost=boost,
    )
    warnings = []
    if governor == "unknown" or boost == "unknown":
        warnings.append("CPU performance tuning state is partially unavailable")
    return LocalPlatformEvidence(
        status="partial" if warnings else "ok",
        host=host,
        kernel=kernel,
        cpu=cpu,
        amdgpu_host_device_count=_amdgpu_device_count(root=root),
        architecture=architecture,
        warnings=warnings,
    )


__all__ = [
    "CpuPlatformEvidence",
    "DEFAULT_ROOT",
    "LocalPlatformEvidence",
    "probe_local_platform",
    "read_kernel_file",
    "sysfs_available",
]
