"""Shape-scoped MMQ and matrix-instruction evidence closure."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .models import ArtifactRef, BottleneckKind, StrictModel


class MatrixInstructionFamily(StrEnum):
    MFMA = "MFMA"
    WMMA = "WMMA"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class MeaningfulOptimizationGap(StrEnum):
    YES = "YES"
    NO = "NO"
    INCONCLUSIVE = "INCONCLUSIVE"


class ShapeCoordinate(StrictModel):
    id: str
    operator: str
    phase: Literal["prefill", "decode"]
    m: int | None = Field(default=None, gt=0)
    n: int = Field(gt=0)
    k: int = Field(gt=0)


class KernelResourceEvidence(StrictModel):
    vgpr: int | None = Field(default=None, ge=0)
    sgpr: int | None = Field(default=None, ge=0)
    lds_bytes: int | None = Field(default=None, ge=0)
    scratch_bytes: int | None = Field(default=None, ge=0)
    occupancy_percent: float | None = Field(default=None, ge=0, le=100)
    wave_size: int | None = Field(default=None, gt=0)
    unavailable: list[str] = Field(default_factory=list)


class StallEvidence(StrictModel):
    compute_percent: float | None = Field(default=None, ge=0, le=100)
    memory_percent: float | None = Field(default=None, ge=0, le=100)
    dependency_percent: float | None = Field(default=None, ge=0, le=100)
    unavailable: list[str] = Field(default_factory=list)


class MatrixPathEvidence(StrictModel):
    schema_name: Literal["gpuopt.matrix-path-evidence.v1"] = Field(
        default="gpuopt.matrix-path-evidence.v1", alias="schema"
    )
    gfx_target: str
    shape: ShapeCoordinate
    selected_kernel: str
    kernel_id: str | None = None
    mmq_enabled: bool | None = None
    matrix_instruction_family: MatrixInstructionFamily
    matrix_instructions: list[str] = Field(default_factory=list)
    isa_complete: bool
    grid: list[int] | None = None
    workgroup: list[int] | None = None
    wave_mapping: str | None = None
    resources: KernelResourceEvidence
    average_duration_ns: float = Field(gt=0)
    total_duration_ns: int = Field(gt=0)
    gpu_time_share_percent: float = Field(ge=0, le=100)
    stalls: StallEvidence
    bottleneck: BottleneckKind
    meaningful_gap: MeaningfulOptimizationGap
    estimated_recoverable_e2e_percent: float | None = Field(default=None, ge=0)
    missing_evidence: list[str] = Field(default_factory=list)
    evidence: list[ArtifactRef] = Field(min_length=1)

    @model_validator(mode="after")
    def instruction_claim_has_isa(self) -> MatrixPathEvidence:
        lowered = [instruction.lower() for instruction in self.matrix_instructions]
        if self.matrix_instruction_family == MatrixInstructionFamily.MFMA and not any(
            instruction.startswith("v_mfma") for instruction in lowered
        ):
            raise ValueError("MFMA claim requires an actual v_mfma ISA instruction")
        if self.matrix_instruction_family == MatrixInstructionFamily.WMMA and not any(
            instruction.startswith("v_wmma") for instruction in lowered
        ):
            raise ValueError("WMMA claim requires an actual v_wmma ISA instruction")
        if self.matrix_instruction_family == MatrixInstructionFamily.UNKNOWN and self.isa_complete:
            raise ValueError("complete ISA must resolve matrix instruction family")
        if self.meaningful_gap == MeaningfulOptimizationGap.YES and (
            self.estimated_recoverable_e2e_percent is None
            or self.estimated_recoverable_e2e_percent < 2
        ):
            raise ValueError("meaningful YES requires at least 2% estimated E2E recovery")
        return self


def infer_matrix_instruction_family(
    instructions: list[str], *, isa_complete: bool
) -> MatrixInstructionFamily:
    lowered = [instruction.lower() for instruction in instructions]
    if any(instruction.startswith("v_mfma") for instruction in lowered):
        return MatrixInstructionFamily.MFMA
    if any(instruction.startswith("v_wmma") for instruction in lowered):
        return MatrixInstructionFamily.WMMA
    return MatrixInstructionFamily.NONE if isa_complete else MatrixInstructionFamily.UNKNOWN


_INSTRUCTION = re.compile(r"^\s*(?:[0-9a-f]+:)?\s*(?P<op>[a-z][a-z0-9_.]+)\b")


def extract_kernel_instructions(disassembly: str, kernel_marker: str) -> list[str]:
    """Extract opcodes from one exact demangled llvm-objdump symbol block."""

    lines = disassembly.splitlines()
    start = next(
        (
            index + 1
            for index, line in enumerate(lines)
            if line.rstrip().endswith(":") and kernel_marker in line
        ),
        None,
    )
    if start is None:
        return []
    instructions: list[str] = []
    for line in lines[start:]:
        if line and not line[0].isspace() and line.rstrip().endswith(":"):
            break
        match = _INSTRUCTION.match(line)
        if match:
            instructions.append(match.group("op"))
    return instructions


def parse_amdgpu_kernel_metadata(metadata: str, kernel_marker: str) -> KernelResourceEvidence:
    """Parse official llvm-readobj AMDGPU metadata for one exact kernel symbol."""

    lines = metadata.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if kernel_marker in line),
        None,
    )
    if start is None:
        return KernelResourceEvidence(
            unavailable=["kernel was not found in AMDGPU code-object metadata"]
        )
    block = "\n".join(lines[start : start + 100])

    def integer(*keys: str) -> int | None:
        for key in keys:
            match = re.search(rf"{re.escape(key)}\s*[:=]\s*([0-9]+)", block)
            if match:
                return int(match.group(1))
        return None

    values = {
        "vgpr": integer(".vgpr_count", "vgpr_count"),
        "sgpr": integer(".sgpr_count", "sgpr_count"),
        "lds_bytes": integer(".group_segment_fixed_size", "group_segment_fixed_size"),
        "scratch_bytes": integer(
            ".private_segment_fixed_size", "private_segment_fixed_size"
        ),
        "wave_size": integer(".wavefront_size", "wavefront_size"),
    }
    unavailable = [name for name, value in values.items() if value is None]
    return KernelResourceEvidence(**values, unavailable=unavailable)


def classify_meaningful_gap(
    *,
    gpu_time_share_percent: float,
    estimated_recoverable_e2e_percent: float | None,
    core_evidence_complete: bool,
) -> MeaningfulOptimizationGap:
    if not core_evidence_complete or estimated_recoverable_e2e_percent is None:
        return MeaningfulOptimizationGap.INCONCLUSIVE
    if gpu_time_share_percent >= 5 and estimated_recoverable_e2e_percent >= 2:
        return MeaningfulOptimizationGap.YES
    if estimated_recoverable_e2e_percent < 2:
        return MeaningfulOptimizationGap.NO
    return MeaningfulOptimizationGap.INCONCLUSIVE


class MFMAMMQClosure(StrictModel):
    schema_name: Literal["gpuopt.mfma-mmq-closure.v1"] = Field(
        default="gpuopt.mfma-mmq-closure.v1", alias="schema"
    )
    gfx_target: str
    shapes: list[MatrixPathEvidence] = Field(min_length=1)
    required_shape_ids: list[str] = Field(min_length=1)
    missing_shape_ids: list[str] = Field(default_factory=list)
    evidence_closed: bool
    summary: str

    @model_validator(mode="after")
    def coverage_matches(self) -> MFMAMMQClosure:
        observed = {item.shape.id for item in self.shapes}
        expected_missing = sorted(set(self.required_shape_ids) - observed)
        if sorted(self.missing_shape_ids) != expected_missing:
            raise ValueError("missing_shape_ids do not match required shape coverage")
        if self.evidence_closed != (not self.missing_shape_ids):
            raise ValueError("evidence_closed must match required shape coverage")
        if any(item.gfx_target != self.gfx_target for item in self.shapes):
            raise ValueError("all matrix path evidence must use the closure gfx target")
        return self


def close_mfma_mmq_evidence(
    gfx_target: str,
    shapes: list[MatrixPathEvidence],
    *,
    required_shape_ids: list[str],
) -> MFMAMMQClosure:
    observed = {item.shape.id for item in shapes}
    missing = sorted(set(required_shape_ids) - observed)
    return MFMAMMQClosure(
        gfx_target=gfx_target,
        shapes=shapes,
        required_shape_ids=required_shape_ids,
        missing_shape_ids=missing,
        evidence_closed=not missing,
        summary=(
            "All required shapes have explicit matrix-path and availability evidence"
            if not missing
            else "Required shape evidence remains incomplete"
        ),
    )


__all__ = [
    "KernelResourceEvidence",
    "MFMAMMQClosure",
    "MatrixInstructionFamily",
    "MatrixPathEvidence",
    "MeaningfulOptimizationGap",
    "ShapeCoordinate",
    "StallEvidence",
    "classify_meaningful_gap",
    "close_mfma_mmq_evidence",
    "extract_kernel_instructions",
    "infer_matrix_instruction_family",
    "parse_amdgpu_kernel_metadata",
]
