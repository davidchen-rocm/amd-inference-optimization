"""Config-driven, shape-aware kernel evidence manifest.

Profiler rows can identify a launch but usually cannot identify the model
operator or its mathematical M/N/K coordinates.  Attribution rules therefore
live outside profiler output, are hash-bound, and remain explicit in the
manifest.  Missing or ambiguous attribution is preserved instead of guessed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .models import StrictModel, utc_now
from .shape_kernels import KernelDispatchGroup, group_mcp_kernel_evidence


class KernelManifestError(RuntimeError):
    """Kernel evidence or an attribution rule violates the manifest contract."""


class AttributionStatus(StrEnum):
    EXACT_RULE = "EXACT_RULE"
    UNRESOLVED = "UNRESOLVED"
    AMBIGUOUS = "AMBIGUOUS"


class TensorCoordinate(StrictModel):
    operator: str | None = None
    phase: str | None = None
    bucket: str | None = None
    dtype: str | None = None
    quantization: str | None = None
    m: int | None = Field(default=None, gt=0)
    n: int | None = Field(default=None, gt=0)
    k: int | None = Field(default=None, gt=0)
    batch: int | None = Field(default=None, gt=0)
    groups: int | None = Field(default=None, gt=0)
    input_layout: str | None = None
    output_layout: str | None = None
    scale_layout: str | None = None
    epilogue: str | None = None


class ShapeAttributionRule(StrictModel):
    id: str
    kernel_name_regex: str
    source_kernel_id: str | None = None
    grid: tuple[int, int, int] | None = None
    workgroup: tuple[int, int, int] | None = None
    graph_variant: str | None = None
    node_ordinal: int | None = Field(default=None, ge=0)
    coordinate: TensorCoordinate
    evidence_basis: str

    @field_validator("id", "kernel_name_regex", "evidence_basis")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("shape attribution rule text cannot be empty")
        return value

    @field_validator("kernel_name_regex")
    @classmethod
    def valid_regex(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError("kernel_name_regex is invalid") from error
        return value

    @field_validator("grid", "workgroup")
    @classmethod
    def valid_geometry(
        cls, value: tuple[int, int, int] | None
    ) -> tuple[int, int, int] | None:
        if value is not None and any(item <= 0 for item in value):
            raise ValueError("launch geometry must contain positive values")
        return value


class KernelResourceUsage(StrictModel):
    vgpr_count: int | None = Field(default=None, ge=0)
    accum_vgpr_count: int | None = Field(default=None, ge=0)
    sgpr_count: int | None = Field(default=None, ge=0)
    lds_bytes: int | None = Field(default=None, ge=0)
    scratch_bytes: int | None = Field(default=None, ge=0)
    occupancy: float | None = Field(default=None, ge=0)


class KernelVariantEvidence(StrictModel):
    signature: str
    source_kernel_id: str | None = None
    name: str
    graph_variant: str | None = None
    node_ordinal: int | None = None
    grid: tuple[int, int, int]
    workgroup: tuple[int, int, int]
    coordinate: TensorCoordinate
    attribution_status: AttributionStatus
    attribution_rule_id: str | None = None
    attribution_basis: str | None = None
    dispatch_count: int = Field(ge=0)
    total_duration_ns: float | None = Field(default=None, ge=0)
    average_duration_ns: float | None = Field(default=None, ge=0)
    gpu_time_share_percent: float | None = Field(default=None, ge=0, le=100)
    resource_usage: KernelResourceUsage
    warnings: list[str] = Field(default_factory=list)


class KernelShapeManifest(StrictModel):
    schema_name: Literal["gpuopt.kernel-shape-manifest.v1"] = Field(
        default="gpuopt.kernel-shape-manifest.v1", alias="schema"
    )
    generated_at: datetime = Field(default_factory=utc_now)
    source_schema: str | None = None
    source_sha256: str
    attribution_rules_sha256: str
    variants: list[KernelVariantEvidence]
    resolved_variants: int = Field(ge=0)
    unresolved_variants: int = Field(ge=0)
    ambiguous_variants: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def counts_match(self) -> KernelShapeManifest:
        counts = {
            AttributionStatus.EXACT_RULE: self.resolved_variants,
            AttributionStatus.UNRESOLVED: self.unresolved_variants,
            AttributionStatus.AMBIGUOUS: self.ambiguous_variants,
        }
        for status, expected in counts.items():
            if sum(item.attribution_status == status for item in self.variants) != expected:
                raise ValueError("kernel shape manifest attribution counts do not match")
        return self


def _canonical_hash(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True)  # type: ignore[union-attr]
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _matches(rule: ShapeAttributionRule, group: KernelDispatchGroup) -> bool:
    if re.search(rule.kernel_name_regex, group.name) is None:
        return False
    if rule.source_kernel_id is not None and rule.source_kernel_id != group.kernel_id:
        return False
    if rule.grid is not None and rule.grid != group.grid:
        return False
    return rule.workgroup is None or rule.workgroup == group.workgroup


def _resource_usage(kernel: Mapping[str, Any]) -> KernelResourceUsage:
    raw = kernel.get("resource_usage")
    raw = raw if isinstance(raw, Mapping) else {}
    metadata = kernel.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}

    def value(*names: str) -> Any:
        for source in (raw, metadata, kernel):
            for name in names:
                if source.get(name) is not None:
                    return source[name]
        return None

    return KernelResourceUsage(
        vgpr_count=value("vgpr_count", "vgpr"),
        accum_vgpr_count=value("accum_vgpr_count", "accum_vgpr"),
        sgpr_count=value("sgpr_count", "sgpr"),
        lds_bytes=value("lds_bytes", "lds_size"),
        scratch_bytes=value("scratch_bytes", "scratch_size"),
        occupancy=value("occupancy"),
    )


def _share(kernel: Mapping[str, Any]) -> float | None:
    for name in (
        "gpu_kernel_time_share_percent",
        "gpu_time_share_percent",
        "gpu_time_share",
    ):
        raw = kernel.get(name)
        if raw is None:
            continue
        try:
            result = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result) and 0 <= result <= 100:
            return result
    return None


def _kernel_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    nested = payload.get("kernel_evidence")
    evidence = nested if isinstance(nested, Mapping) else payload
    rows = evidence.get("kernels")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise KernelManifestError("kernel evidence must contain a kernels array")
    if any(not isinstance(item, Mapping) for item in rows):
        raise KernelManifestError("every kernel evidence entry must be an object")
    return list(rows)  # type: ignore[return-value]


def build_kernel_shape_manifest(
    payload: Mapping[str, Any],
    rules: Sequence[ShapeAttributionRule],
    *,
    source_sha256: str | None = None,
) -> KernelShapeManifest:
    """Build a shape manifest without deriving M/N/K from launch geometry."""

    if len({rule.id for rule in rules}) != len(rules):
        raise KernelManifestError("shape attribution rule ids must be unique")
    rows = _kernel_rows(payload)
    groups = group_mcp_kernel_evidence(payload)
    row_index: dict[
        tuple[str | None, str, tuple[int, ...], tuple[int, ...]], Mapping[str, Any]
    ] = {}
    for row in rows:
        identifier = row.get("kernel_id")
        metadata = row.get("metadata")
        if identifier is None and isinstance(metadata, Mapping):
            identifier = metadata.get("kernel_id")
        grid = tuple(row.get("grid", ()))
        workgroup = tuple(row.get("workgroup", ()))
        key = (
            str(identifier) if identifier is not None else None,
            str(row.get("name")),
            grid,
            workgroup,
        )
        row_index[key] = row

    variants: list[KernelVariantEvidence] = []
    manifest_warnings: list[str] = []
    for group in groups:
        matches = [rule for rule in rules if _matches(rule, group)]
        if len(matches) == 1:
            rule = matches[0]
            status = AttributionStatus.EXACT_RULE
            coordinate = rule.coordinate
            warnings: list[str] = []
        elif len(matches) > 1:
            rule = None
            status = AttributionStatus.AMBIGUOUS
            coordinate = TensorCoordinate()
            warnings = ["multiple shape attribution rules matched this launch"]
            manifest_warnings.append(
                f"ambiguous attribution for {group.name} grid={group.grid}"
            )
        else:
            rule = None
            status = AttributionStatus.UNRESOLVED
            coordinate = TensorCoordinate()
            warnings = ["operator and M/N/K are not observable and no rule matched"]
        raw = row_index.get((group.kernel_id, group.name, group.grid, group.workgroup), {})
        identity = {
            "source_kernel_id": group.kernel_id,
            "name": group.name,
            "graph_variant": rule.graph_variant if rule else None,
            "node_ordinal": rule.node_ordinal if rule else None,
            "grid": group.grid,
            "workgroup": group.workgroup,
            "coordinate": coordinate.model_dump(mode="json"),
            "resource_usage": _resource_usage(raw).model_dump(mode="json"),
        }
        variants.append(
            KernelVariantEvidence(
                signature="sha256:" + _canonical_hash(identity),
                source_kernel_id=group.kernel_id,
                name=group.name,
                graph_variant=rule.graph_variant if rule else None,
                node_ordinal=rule.node_ordinal if rule else None,
                grid=group.grid,
                workgroup=group.workgroup,
                coordinate=coordinate,
                attribution_status=status,
                attribution_rule_id=rule.id if rule else None,
                attribution_basis=rule.evidence_basis if rule else None,
                dispatch_count=group.dispatch_count,
                total_duration_ns=group.total_duration_ns,
                average_duration_ns=group.average_duration_ns,
                gpu_time_share_percent=_share(raw),
                resource_usage=_resource_usage(raw),
                warnings=warnings,
            )
        )
    variants.sort(
        key=lambda item: (
            -(item.total_duration_ns if item.total_duration_ns is not None else -1),
            item.signature,
        )
    )
    source_hash = source_sha256 or _canonical_hash(payload)
    if len(source_hash) != 64 or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
        raise KernelManifestError("source_sha256 must be lowercase hexadecimal")
    return KernelShapeManifest(
        source_schema=str(payload.get("schema")) if payload.get("schema") else None,
        source_sha256=source_hash,
        attribution_rules_sha256=_canonical_hash(
            [rule.model_dump(mode="json") for rule in rules]
        ),
        variants=variants,
        resolved_variants=sum(
            item.attribution_status == AttributionStatus.EXACT_RULE for item in variants
        ),
        unresolved_variants=sum(
            item.attribution_status == AttributionStatus.UNRESOLVED for item in variants
        ),
        ambiguous_variants=sum(
            item.attribution_status == AttributionStatus.AMBIGUOUS for item in variants
        ),
        warnings=sorted(set(manifest_warnings)),
    )


__all__ = [
    "AttributionStatus",
    "KernelManifestError",
    "KernelResourceUsage",
    "KernelShapeManifest",
    "KernelVariantEvidence",
    "ShapeAttributionRule",
    "TensorCoordinate",
    "build_kernel_shape_manifest",
]
