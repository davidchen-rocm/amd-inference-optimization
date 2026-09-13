"""Stable catalogue for external and local evidence providers.

The framework owns evidence requirements and verdicts.  Providers own capture
and analysis.  This module makes that boundary explicit so an installed tool is
not confused with a tool that the workflow actually invokes.
"""

from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import StrictModel, utc_now


class ProviderIntegration(StrEnum):
    DIRECT = "DIRECT"
    ADAPTER = "ADAPTER"
    TRANSITIVE = "TRANSITIVE"
    AVAILABLE_NOT_INTEGRATED = "AVAILABLE_NOT_INTEGRATED"


class ProviderAvailability(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class EvidenceDomain(StrEnum):
    ENVIRONMENT = "environment"
    BENCHMARK = "benchmark"
    TRACE = "trace"
    KERNEL = "kernel"
    HARDWARE_COUNTER = "hardware_counter"
    SOURCE_LINE = "source_line"
    ISA = "isa"
    CORRECTNESS = "correctness"
    HOST = "host"
    DISTRIBUTED = "distributed"
    HEALTH = "health"
    PROVENANCE = "provenance"


class ProviderCapability(StrictModel):
    id: str
    domain: EvidenceDomain
    description: str
    gate_eligible: bool
    requires_execution: bool
    output_schemas: list[str] = Field(default_factory=list)

    @field_validator("id", "description")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provider capability text cannot be empty")
        return value


class EvidenceProviderSpec(StrictModel):
    id: str
    name: str
    integration: ProviderIntegration
    authority: str
    license: str | None = None
    commands: list[str] = Field(default_factory=list)
    python_modules: list[str] = Field(default_factory=list)
    capabilities: list[ProviderCapability]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_capabilities(self) -> EvidenceProviderSpec:
        ids = [item.id for item in self.capabilities]
        if len(ids) != len(set(ids)):
            raise ValueError("provider capability ids must be unique")
        return self


class ProviderProbe(StrictModel):
    provider_id: str
    availability: ProviderAvailability
    integration: ProviderIntegration
    detected_commands: dict[str, str | None] = Field(default_factory=dict)
    detected_modules: dict[str, bool] = Field(default_factory=dict)
    detail: str


class EvidenceCapabilitySnapshot(StrictModel):
    schema_name: Literal["gpuopt.evidence-capabilities.v1"] = Field(
        default="gpuopt.evidence-capabilities.v1", alias="schema"
    )
    collected_at: datetime = Field(default_factory=utc_now)
    providers: list[ProviderProbe]
    catalogue: list[EvidenceProviderSpec]
    warnings: list[str] = Field(default_factory=list)


def _capability(
    identifier: str,
    domain: EvidenceDomain,
    description: str,
    *,
    gate: bool,
    executes: bool,
    schemas: Sequence[str] = (),
) -> ProviderCapability:
    return ProviderCapability(
        id=identifier,
        domain=domain,
        description=description,
        gate_eligible=gate,
        requires_execution=executes,
        output_schemas=list(schemas),
    )


EVIDENCE_PROVIDERS: tuple[EvidenceProviderSpec, ...] = (
    EvidenceProviderSpec(
        id="rocm-issue-agent",
        name="ROCm Issue Agent",
        integration=ProviderIntegration.DIRECT,
        authority="local MCP server",
        commands=["rocm-agent-mcp"],
        capabilities=[
            _capability(
                "rocm-environment",
                EvidenceDomain.ENVIRONMENT,
                "ROCm, HIP, GPU and profiler capability snapshot.",
                gate=True,
                executes=False,
                schemas=["rocm.mcp-snapshot.v1", "rocm.hip-capabilities.v1"],
            ),
            _capability(
                "kernel-timing",
                EvidenceDomain.KERNEL,
                "Bounded kernel dispatch timing, coverage and endpoint telemetry.",
                gate=True,
                executes=True,
                schemas=["rocm.mcp-kernel-evidence.v1"],
            ),
            _capability(
                "runtime-trace",
                EvidenceDomain.TRACE,
                "HIP API, kernel and memory-copy runtime trace.",
                gate=True,
                executes=True,
                schemas=["rocm.mcp-trace-result.v1"],
            ),
            _capability(
                "system-validation",
                EvidenceDomain.HEALTH,
                "ROCm system validation; not model quality evaluation.",
                gate=True,
                executes=True,
                schemas=["rocm.mcp-validation-result.v1"],
            ),
        ],
        notes=["Primary local GPU evidence provider; the framework does not reimplement it."],
    ),
    EvidenceProviderSpec(
        id="raw-rocprofv3",
        name="Raw rocprofv3 fallback",
        integration=ProviderIntegration.DIRECT,
        authority="local framework adapter",
        commands=["rocprofv3"],
        capabilities=[
            _capability(
                "raw-kernel-timing",
                EvidenceDomain.KERNEL,
                "Hash-bound kernel timing fallback with explicit normalization budgets.",
                gate=True,
                executes=True,
                schemas=["gpuopt.raw-kernel-evidence.v1"],
            ),
            _capability(
                "raw-runtime-trace",
                EvidenceDomain.TRACE,
                "HIP, kernel and copy CSV normalization fallback.",
                gate=True,
                executes=True,
                schemas=["gpuopt.raw-runtime-trace.v1"],
            ),
        ],
    ),
    EvidenceProviderSpec(
        id="magpie",
        name="AMD-AGI Magpie",
        integration=ProviderIntegration.ADAPTER,
        authority="external MIT package",
        license="MIT",
        python_modules=["Magpie"],
        capabilities=[
            _capability(
                "framework-benchmark",
                EvidenceDomain.BENCHMARK,
                "vLLM/SGLang/Atom throughput, latency and completed-request evidence.",
                gate=True,
                executes=True,
                schemas=["gpuopt.magpie-benchmark-evidence.v1"],
            ),
            _capability(
                "gap-analysis",
                EvidenceDomain.KERNEL,
                "Steady-state kernel call, duration and share aggregation.",
                gate=False,
                executes=False,
                schemas=["gpuopt.magpie-benchmark-evidence.v1"],
            ),
            _capability(
                "multi-rank-trace",
                EvidenceDomain.DISTRIBUTED,
                "Per-rank trace and collective report discovery.",
                gate=False,
                executes=True,
            ),
        ],
        notes=["Adapter imports reports; Magpie remains responsible for execution."],
    ),
    EvidenceProviderSpec(
        id="tracelens",
        name="AMD-AGI TraceLens",
        integration=ProviderIntegration.ADAPTER,
        authority="external MIT package",
        license="MIT",
        commands=["TraceLens_generate_perf_report_rocprof"],
        capabilities=[
            _capability(
                "trace-analysis",
                EvidenceDomain.TRACE,
                "Hierarchical PyTorch/JAX/rocprof trace analysis.",
                gate=False,
                executes=False,
                schemas=["gpuopt.tracelens-roofline-evidence.v1"],
            ),
            _capability(
                "semantic-roofline",
                EvidenceDomain.KERNEL,
                "Operator-semantic roofline evidence with explicit architecture provenance.",
                gate=False,
                executes=False,
                schemas=["gpuopt.tracelens-roofline-evidence.v1"],
            ),
            _capability(
                "collective-analysis",
                EvidenceDomain.DISTRIBUTED,
                "Multi-rank collective and communication analysis.",
                gate=False,
                executes=False,
            ),
        ],
        notes=["TraceLens analyzes traces; it is not a live GPU observer."],
    ),
    EvidenceProviderSpec(
        id="intellikit",
        name="AMDResearch IntelliKit",
        integration=ProviderIntegration.AVAILABLE_NOT_INTEGRATED,
        authority="external MIT packages",
        license="MIT",
        python_modules=["kerncap", "metrix", "linex", "nexus", "accordo"],
        capabilities=[
            _capability(
                "kernel-isolation",
                EvidenceDomain.KERNEL,
                "Kerncap dispatch capture and VA-faithful standalone replay.",
                gate=False,
                executes=True,
            ),
            _capability(
                "hardware-counters",
                EvidenceDomain.HARDWARE_COUNTER,
                "Metrix counter projection for bandwidth, cache and compute behavior.",
                gate=False,
                executes=True,
            ),
            _capability(
                "source-line-stalls",
                EvidenceDomain.SOURCE_LINE,
                "Linex source-line timing and stall evidence.",
                gate=False,
                executes=True,
            ),
            _capability(
                "hsa-isa-inspection",
                EvidenceDomain.ISA,
                "Nexus HSA packet, assembly and HIP source inspection.",
                gate=False,
                executes=True,
            ),
            _capability(
                "kernel-correctness",
                EvidenceDomain.CORRECTNESS,
                "Accordo reference/candidate snapshot comparison.",
                gate=False,
                executes=True,
            ),
        ],
        notes=[
            "HyperLoom reaches only selected IntelliKit capabilities through Magpie; "
            "installation does not imply workflow integration."
        ],
    ),
    EvidenceProviderSpec(
        id="uprof-mcp",
        name="AMD uProf MCP",
        integration=ProviderIntegration.AVAILABLE_NOT_INTEGRATED,
        authority="IntelliKit external MCP server",
        commands=["uprof-profiler-mcp"],
        capabilities=[
            _capability(
                "cpu-hotspots",
                EvidenceDomain.HOST,
                "Host CPU hotspot profiling through AMD uProf.",
                gate=False,
                executes=True,
            )
        ],
    ),
    EvidenceProviderSpec(
        id="gpuopt-local-health",
        name="gpuopt local runtime health",
        integration=ProviderIntegration.DIRECT,
        authority="local framework read-only probe",
        capabilities=[
            _capability(
                "runtime-health",
                EvidenceDomain.HEALTH,
                "Disk, shared-memory, process and dependency availability snapshot.",
                gate=False,
                executes=False,
                schemas=["gpuopt.runtime-health.v1"],
            ),
            _capability(
                "platform-provenance",
                EvidenceDomain.PROVENANCE,
                "CPU, NUMA, governor, boost and configured GPU architecture provenance.",
                gate=True,
                executes=False,
            ),
        ],
    ),
    EvidenceProviderSpec(
        id="gpuopt-orchestration-audit",
        name="gpuopt orchestration audit",
        integration=ProviderIntegration.DIRECT,
        authority="local workflow/store",
        capabilities=[
            _capability(
                "workflow-events",
                EvidenceDomain.PROVENANCE,
                "Stage, decision, approval, attempt and artifact event history.",
                gate=True,
                executes=False,
            ),
            _capability(
                "durable-execution-health",
                EvidenceDomain.HEALTH,
                "Task lock and long-running quality execution identity/lease evidence.",
                gate=True,
                executes=False,
            ),
        ],
    ),
    EvidenceProviderSpec(
        id="hyperloom-host-probe",
        name="HyperLoom Python host probe concept",
        integration=ProviderIntegration.AVAILABLE_NOT_INTEGRATED,
        authority="external HyperLoom component",
        license="MIT",
        capabilities=[
            _capability(
                "python-host-roundtrips",
                EvidenceDomain.HOST,
                "PyTorch item/tolist/cpu/numpy, synchronization, H2D and collective calls.",
                gate=False,
                executes=True,
            ),
            _capability(
                "framework-repeat-fingerprints",
                EvidenceDomain.HOST,
                "Strict and loose Python framework-call repetition fingerprints.",
                gate=False,
                executes=True,
            ),
        ],
        notes=[
            "Not copied: it is PyTorch-specific and deep profiling can distort host timing. "
            "A llama.cpp implementation needs a separate HIP/C++ evidence leg."
        ],
    ),
    EvidenceProviderSpec(
        id="hyperloom-platform-bmc-audit",
        name="HyperLoom BIOS/BMC platform audit concept",
        integration=ProviderIntegration.AVAILABLE_NOT_INTEGRATED,
        authority="external HyperLoom operator tool",
        license="MIT",
        commands=["ipmitool"],
        capabilities=[
            _capability(
                "bios-performance-profile",
                EvidenceDomain.ENVIRONMENT,
                "High Performance profile, APBDIS and DF C-state evidence.",
                gate=False,
                executes=False,
            )
        ],
        notes=[
            "Not integrated: BMC access and temporary account creation are a separate "
            "privileged risk class and require explicit operator policy."
        ],
    ),
    EvidenceProviderSpec(
        id="hyperloom-multinode-probes",
        name="HyperLoom multi-node probe concept",
        integration=ProviderIntegration.AVAILABLE_NOT_INTEGRATED,
        authority="external HyperLoom component",
        license="MIT",
        capabilities=[
            _capability(
                "remote-gpu-identity",
                EvidenceDomain.DISTRIBUTED,
                "Ray/SSH GPU identity discovery on the actual inference nodes.",
                gate=False,
                executes=True,
            ),
            _capability(
                "serving-leg-health",
                EvidenceDomain.DISTRIBUTED,
                "Health, model registration and real generated-token checks for every leg.",
                gate=False,
                executes=True,
            ),
        ],
        notes=["Deferred until the future MI300X remote/multi-node phase."],
    ),
    EvidenceProviderSpec(
        id="hyperloom-robustness-monitor",
        name="HyperLoom robustness monitor concept",
        integration=ProviderIntegration.AVAILABLE_NOT_INTEGRATED,
        authority="external HyperLoom component",
        license="MIT",
        capabilities=[
            _capability(
                "cluster-and-agent-health",
                EvidenceDomain.HEALTH,
                (
                    "GPU leak, Ray, server, disk/shm, FD, heartbeat, lease, WAL "
                    "and dependency signals."
                ),
                gate=False,
                executes=False,
            )
        ],
        notes=[
            "gpuopt implements a bounded local subset; cluster-wide and agent-reactor "
            "signals are not claimed."
        ],
    ),
)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def probe_evidence_capabilities(
    *,
    command_overrides: Mapping[str, str] | None = None,
) -> EvidenceCapabilitySnapshot:
    """Probe installation only; never execute a profiler or workload."""

    overrides = dict(command_overrides or {})
    probes: list[ProviderProbe] = []
    for provider in EVIDENCE_PROVIDERS:
        commands = {
            command: overrides.get(command) or shutil.which(command)
            for command in provider.commands
        }
        modules = {name: _module_available(name) for name in provider.python_modules}
        signals = [value is not None for value in commands.values()] + list(modules.values())
        if (
            not provider.commands
            and not provider.python_modules
            and provider.integration == ProviderIntegration.DIRECT
        ):
            availability = ProviderAvailability.AVAILABLE
            detail = "built into gpuopt"
        elif not provider.commands and not provider.python_modules:
            availability = ProviderAvailability.UNKNOWN
            detail = "catalogued design boundary; no local entry point is integrated"
        elif signals and all(signals):
            availability = ProviderAvailability.AVAILABLE
            detail = "all declared provider entry points were detected"
        elif any(signals):
            availability = ProviderAvailability.UNKNOWN
            detail = "only part of the provider entry points were detected"
        else:
            availability = ProviderAvailability.UNAVAILABLE
            detail = "no provider entry point was detected; this is not execution validation"
        probes.append(
            ProviderProbe(
                provider_id=provider.id,
                availability=availability,
                integration=provider.integration,
                detected_commands=commands,
                detected_modules=modules,
                detail=detail,
            )
        )
    return EvidenceCapabilitySnapshot(
        providers=probes,
        catalogue=list(EVIDENCE_PROVIDERS),
    )


__all__ = [
    "EVIDENCE_PROVIDERS",
    "EvidenceCapabilitySnapshot",
    "EvidenceDomain",
    "EvidenceProviderSpec",
    "ProviderAvailability",
    "ProviderCapability",
    "ProviderIntegration",
    "ProviderProbe",
    "probe_evidence_capabilities",
]
