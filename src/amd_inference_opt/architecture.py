"""Architecture identities used by local optimization workflows.

The registry deliberately separates a gfx ISA target from a board product.  A
future board may share an ISA with another product, so callers must not infer a
product name from ``gfx_target`` alone.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import Field, field_validator

from .models import StrictModel

_GFX_TARGET = re.compile(r"^gfx[0-9a-f]+$")


class ArchitectureFamily(StrEnum):
    CDNA3 = "cdna3"
    RDNA4 = "rdna4"
    UNKNOWN = "unknown"


class ArchitectureProfile(StrictModel):
    """Stable architecture facts; observed run state belongs in evidence."""

    gfx_target: str
    family: ArchitectureFamily
    wavefront_size: int | None = Field(default=None, ge=1)
    compute_units: int | None = Field(default=None, ge=1)
    board_type: str | None = None
    theoretical_memory_bandwidth_gbps: float | None = Field(default=None, gt=0)
    source: str

    @field_validator("gfx_target")
    @classmethod
    def validate_gfx_target(cls, value: str) -> str:
        normalized = value.strip().lower()
        if _GFX_TARGET.fullmatch(normalized) is None:
            raise ValueError("gfx_target must use the canonical gfxNNNN form")
        return normalized

    @field_validator("board_type", "source")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("architecture text fields cannot be empty")
        return value


_LOCAL_PROFILES: dict[str, ArchitectureProfile] = {
    "gfx942": ArchitectureProfile(
        gfx_target="gfx942",
        family=ArchitectureFamily.CDNA3,
        wavefront_size=64,
        source="gpuopt.local-profile.v1",
    ),
    "gfx1201": ArchitectureProfile(
        gfx_target="gfx1201",
        family=ArchitectureFamily.RDNA4,
        wavefront_size=32,
        source="gpuopt.local-profile.v1",
    ),
}


def canonical_gfx_target(value: str) -> str:
    """Return one canonical bare GFX target or reject the coordinate."""

    normalized = value.strip().lower()
    if _GFX_TARGET.fullmatch(normalized) is None:
        raise ValueError("gfx target must use the canonical gfxNNNN form")
    return normalized


def cmake_gfx_target(build_flags: Sequence[str]) -> str:
    """Extract exactly one ``AMDGPU_TARGETS`` coordinate from CMake flags.

    A build containing no target, multiple target options, or a target list is
    not architecture-bound evidence.  Callers therefore fail closed instead of
    choosing the first value or assuming the local Radeon default.
    """

    prefix = "-DAMDGPU_TARGETS="
    typed_prefix = "-DAMDGPU_TARGETS:STRING="
    values = [
        flag[len(prefix) :] if flag.startswith(prefix) else flag[len(typed_prefix) :]
        for flag in build_flags
        if flag.startswith(prefix) or flag.startswith(typed_prefix)
    ]
    if not values:
        raise ValueError("CMake build flags are missing AMDGPU_TARGETS")
    if len(values) != 1:
        raise ValueError("CMake build flags must contain exactly one AMDGPU_TARGETS option")
    raw = values[0].strip()
    if ";" in raw:
        raise ValueError("AMDGPU_TARGETS must contain exactly one GFX target")
    try:
        return canonical_gfx_target(raw)
    except ValueError as error:
        raise ValueError(f"AMDGPU_TARGETS has an invalid GFX target: {raw!r}") from error


_LOCAL_BOARD_FACTS: dict[tuple[str, str], dict[str, object]] = {
    ("gfx1201", "radeon-rx-9070-xt"): {"compute_units": 64},
}


def architecture_profile(
    gfx_target: str,
    *,
    board_type: str | None = None,
) -> ArchitectureProfile:
    """Resolve known local facts while preserving unknown future targets."""

    normalized = canonical_gfx_target(gfx_target)
    known = _LOCAL_PROFILES.get(normalized)
    if known is None:
        return ArchitectureProfile(
            gfx_target=normalized,
            family=ArchitectureFamily.UNKNOWN,
            board_type=board_type,
            source="gpuopt.unrecognized-profile.v1",
        )
    if board_type is None:
        return known
    normalized_board = board_type.strip().lower()
    return known.model_copy(
        update={
            "board_type": normalized_board,
            **_LOCAL_BOARD_FACTS.get((normalized, normalized_board), {}),
        }
    )


def registered_architectures() -> tuple[ArchitectureProfile, ...]:
    return tuple(_LOCAL_PROFILES[key] for key in sorted(_LOCAL_PROFILES))


__all__ = [
    "ArchitectureFamily",
    "ArchitectureProfile",
    "architecture_profile",
    "canonical_gfx_target",
    "cmake_gfx_target",
    "registered_architectures",
]
