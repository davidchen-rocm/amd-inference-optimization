"""Shape-aware projections for raw rocprof and ROCm MCP kernel evidence.

Kernel names (and even rocprof kernel IDs) are shared by several matrix shapes.
This module therefore treats launch geometry as part of dispatch identity and
keeps shape attribution explicitly tied to a locked model/operator map.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ShapeKernelError(RuntimeError):
    """Kernel evidence is incomplete or cannot be projected safely."""


def _normalized_field(row: Mapping[str, Any], *names: str) -> Any:
    normalized = {re.sub(r"[^a-z0-9]", "", str(key).lower()): value for key, value in row.items()}
    for name in names:
        value = normalized.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if value is not None and (not isinstance(value, str) or value.strip()):
            return value.strip() if isinstance(value, str) else value
    return None


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ShapeKernelError(f"{field} must be an integer")
    try:
        parsed = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as error:
        raise ShapeKernelError(f"{field} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise ShapeKernelError(f"{field} must be an integer")
    if parsed < (1 if positive else 0):
        condition = "positive" if positive else "non-negative"
        raise ShapeKernelError(f"{field} must be {condition}")
    return parsed


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ShapeKernelError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ShapeKernelError(f"{field} must be numeric") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ShapeKernelError(f"{field} must be finite and non-negative")
    return parsed


def _dimensions_from_components(
    row: Mapping[str, Any],
    prefix: str,
) -> tuple[int, int, int] | None:
    values = tuple(
        _normalized_field(
            row,
            f"{prefix}_Size_{axis}",
            f"{prefix}Size{axis}",
            f"{prefix}_{axis}",
        )
        for axis in "XYZ"
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ShapeKernelError(f"{prefix.lower()} must provide x, y, and z")
    return tuple(
        _integer(value, f"{prefix.lower()}_{axis.lower()}", positive=True)
        for axis, value in zip("XYZ", values, strict=True)
    )  # type: ignore[return-value]


def _dimensions(row: Mapping[str, Any], field: str) -> tuple[int, int, int]:
    direct = _normalized_field(row, field)
    if direct is not None:
        if not isinstance(direct, Sequence) or isinstance(direct, (str, bytes)):
            raise ShapeKernelError(f"{field} must contain exactly three dimensions")
        if len(direct) != 3:
            raise ShapeKernelError(f"{field} must contain exactly three dimensions")
        return tuple(
            _integer(value, f"{field}[{index}]", positive=True)
            for index, value in enumerate(direct)
        )  # type: ignore[return-value]
    components = _dimensions_from_components(row, field)
    if components is None:
        raise ShapeKernelError(f"kernel evidence is missing {field}")
    return components


def _kernel_identity(kernel_id: str | None, name: str) -> str:
    return f"kernel_id:{kernel_id}" if kernel_id else f"kernel_name:{name}"


@dataclass(frozen=True)
class KernelDispatchGroup:
    """One kernel specialization at one exact grid/workgroup geometry."""

    source: str
    identity: str
    kernel_id: str | None
    name: str
    grid: tuple[int, int, int]
    workgroup: tuple[int, int, int]
    dispatch_count: int
    total_duration_ns: float | None

    @property
    def average_duration_ns(self) -> float | None:
        if self.total_duration_ns is None or self.dispatch_count == 0:
            return None
        return self.total_duration_ns / self.dispatch_count

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["grid"] = list(self.grid)
        result["workgroup"] = list(self.workgroup)
        result["average_duration_ns"] = self.average_duration_ns
        return result


@dataclass
class _MutableGroup:
    source: str
    identity: str
    kernel_id: str | None
    name: str
    grid: tuple[int, int, int]
    workgroup: tuple[int, int, int]
    dispatch_count: int = 0
    total_duration_ns: float | None = 0.0

    def add(self, count: int, duration_ns: float | None) -> None:
        self.dispatch_count += count
        if duration_ns is None:
            self.total_duration_ns = None
        elif self.total_duration_ns is not None:
            self.total_duration_ns += duration_ns

    def freeze(self) -> KernelDispatchGroup:
        return KernelDispatchGroup(
            source=self.source,
            identity=self.identity,
            kernel_id=self.kernel_id,
            name=self.name,
            grid=self.grid,
            workgroup=self.workgroup,
            dispatch_count=self.dispatch_count,
            total_duration_ns=self.total_duration_ns,
        )


def _add_group(
    groups: dict[tuple[str, str, tuple[int, int, int], tuple[int, int, int]], _MutableGroup],
    *,
    source: str,
    kernel_id: str | None,
    name: str,
    grid: tuple[int, int, int],
    workgroup: tuple[int, int, int],
    count: int,
    duration_ns: float | None,
) -> None:
    identity = _kernel_identity(kernel_id, name)
    # Keep the name in the key as a defensive guard against an inconsistent
    # producer reusing a kernel ID for more than one symbol.
    key = (identity, name, grid, workgroup)
    aggregate = groups.setdefault(
        key,
        _MutableGroup(source, identity, kernel_id, name, grid, workgroup),
    )
    aggregate.add(count, duration_ns)


def _freeze_groups(groups: Mapping[object, _MutableGroup]) -> tuple[KernelDispatchGroup, ...]:
    frozen = [group.freeze() for group in groups.values()]
    frozen.sort(
        key=lambda item: (
            -(item.total_duration_ns if item.total_duration_ns is not None else -1),
            item.identity,
            item.name,
            item.grid,
            item.workgroup,
        )
    )
    return tuple(frozen)


def group_raw_dispatch_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[KernelDispatchGroup, ...]:
    """Group raw rocprof rows without merging different launch geometries."""

    groups: dict[tuple[str, str, tuple[int, int, int], tuple[int, int, int]], _MutableGroup] = {}
    for row_number, row in enumerate(rows, start=2):
        name_value = _normalized_field(row, "Kernel_Name", "KernelName", "Kernel Name")
        start_value = _normalized_field(row, "Start_Timestamp", "StartTimestamp", "StartNs")
        end_value = _normalized_field(row, "End_Timestamp", "EndTimestamp", "EndNs")
        if name_value is None or start_value is None or end_value is None:
            raise ShapeKernelError(f"raw kernel row {row_number} is incomplete")
        name = str(name_value)
        start = _integer(start_value, f"row {row_number} start")
        end = _integer(end_value, f"row {row_number} end")
        if end < start:
            raise ShapeKernelError(f"raw kernel row {row_number} ends before it starts")
        kernel_id_value = _normalized_field(row, "Kernel_Id", "KernelId", "Kernel ID")
        kernel_id = str(kernel_id_value) if kernel_id_value is not None else None
        _add_group(
            groups,
            source="raw_rocprofv3",
            kernel_id=kernel_id,
            name=name,
            grid=_dimensions(row, "grid"),
            workgroup=_dimensions(row, "workgroup"),
            count=1,
            duration_ns=float(end - start),
        )
    return _freeze_groups(groups)


def group_raw_kernel_csvs(
    paths: str | Path | Sequence[str | Path],
) -> tuple[KernelDispatchGroup, ...]:
    """Read and group one or more raw rocprof kernel-trace CSV files."""

    selected = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not selected:
        raise ShapeKernelError("at least one raw kernel CSV is required")

    def rows() -> Iterable[Mapping[str, Any]]:
        for untyped_path in selected:
            path = Path(untyped_path)
            if not path.is_file() or path.is_symlink():
                raise ShapeKernelError(f"raw kernel CSV is not a regular file: {path}")
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                yield from csv.DictReader(handle)

    return group_raw_dispatch_rows(rows())


def _mcp_duration_ns(kernel: Mapping[str, Any], count: int) -> float | None:
    total = _normalized_field(kernel, "total_duration_ns", "TotalDurationNs")
    if total is not None:
        return _number(total, "total_duration_ns")
    average = _normalized_field(kernel, "average_duration_ns", "AverageNs")
    if average is not None:
        return _number(average, "average_duration_ns") * count
    duration_us = _normalized_field(kernel, "duration_us")
    if duration_us is not None:
        return _number(duration_us, "duration_us") * 1000.0
    return None


def group_mcp_kernel_evidence(
    payload: Mapping[str, Any],
) -> tuple[KernelDispatchGroup, ...]:
    """Project MCP structured content while preserving shape identity.

    ``payload`` may be the structured result itself or a raw fallback result
    containing a nested ``kernel_evidence`` object.
    """

    nested = payload.get("kernel_evidence")
    evidence = nested if isinstance(nested, Mapping) else payload
    kernels = evidence.get("kernels")
    if not isinstance(kernels, Sequence) or isinstance(kernels, (str, bytes)):
        raise ShapeKernelError("MCP kernel evidence must contain a kernels array")
    groups: dict[tuple[str, str, tuple[int, int, int], tuple[int, int, int]], _MutableGroup] = {}
    for index, untyped_kernel in enumerate(kernels):
        if not isinstance(untyped_kernel, Mapping):
            raise ShapeKernelError(f"MCP kernel entry {index} must be an object")
        name_value = _normalized_field(untyped_kernel, "name", "kernel_name")
        if name_value is None:
            raise ShapeKernelError(f"MCP kernel entry {index} has no name")
        metadata = untyped_kernel.get("metadata")
        metadata_id = metadata.get("kernel_id") if isinstance(metadata, Mapping) else None
        direct_id = _normalized_field(untyped_kernel, "kernel_id")
        kernel_id_value = direct_id if direct_id is not None else metadata_id
        count_value = _normalized_field(untyped_kernel, "dispatch_count", "calls")
        count = _integer(count_value if count_value is not None else 1, "dispatch_count")
        if count == 0:
            continue
        _add_group(
            groups,
            source="rocm_mcp",
            kernel_id=str(kernel_id_value) if kernel_id_value is not None else None,
            name=str(name_value),
            grid=_dimensions(untyped_kernel, "grid"),
            workgroup=_dimensions(untyped_kernel, "workgroup"),
            count=count,
            duration_ns=_mcp_duration_ns(untyped_kernel, count),
        )
    return _freeze_groups(groups)


@dataclass(frozen=True)
class LockedQwenShape:
    """A model/operator mapping locked outside profiler-observable fields."""

    id: str
    n: int
    k: int
    operator: str
    kernel_type: str
    ncols_dst: int
    architecture: str
    baseline_grid: tuple[int, int, int]
    candidate_grid: tuple[int, int, int]
    workgroup: tuple[int, int, int]
    rows_per_block: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for field in ("baseline_grid", "candidate_grid", "workgroup"):
            result[field] = list(getattr(self, field))
        return result


LOCKED_QWEN3_8B_Q6_K_SHAPES = (
    LockedQwenShape(
        id="qwen3-8b-q6-k-ffn-gate-up-n12288-k4096",
        n=12288,
        k=4096,
        operator="FFN gate/up projection",
        kernel_type="Q6_K",
        ncols_dst=1,
        architecture="gfx1201",
        baseline_grid=(393216, 8, 1),
        candidate_grid=(49152, 8, 1),
        workgroup=(32, 8, 1),
        rows_per_block=8,
    ),
    LockedQwenShape(
        id="qwen3-8b-q6-k-attention-hidden-n4096-k4096",
        n=4096,
        k=4096,
        operator="self-attention query/output projections",
        kernel_type="Q6_K",
        ncols_dst=1,
        architecture="gfx1201",
        baseline_grid=(131072, 8, 1),
        candidate_grid=(16384, 8, 1),
        workgroup=(32, 8, 1),
        rows_per_block=8,
    ),
)


def _is_q6_k_ncols_one(name: str) -> bool:
    return bool(
        re.search(
            r"mul_mat_vec_q<\s*(?:\(ggml_type\)14|GGML_TYPE_Q6_K)\s*,\s*1\s*,",
            name,
        )
    )


def map_locked_qwen_shapes(
    groups: Iterable[KernelDispatchGroup],
    *,
    shapes: Sequence[LockedQwenShape] = LOCKED_QWEN3_8B_Q6_K_SHAPES,
) -> tuple[dict[str, Any], ...]:
    """Attach locked Qwen N/K/operator labels to exact dispatch signatures.

    K and operator are not observable in rocprof geometry.  The result marks
    those fields as locked attribution so consumers cannot mistake them for
    values measured by the profiler.
    """

    projected: list[dict[str, Any]] = []
    for group in groups:
        if not _is_q6_k_ncols_one(group.name):
            continue
        matches = [
            (shape, "baseline" if group.grid == shape.baseline_grid else "candidate")
            for shape in shapes
            if group.workgroup == shape.workgroup
            and group.grid in (shape.baseline_grid, shape.candidate_grid)
        ]
        if len(matches) > 1:
            raise ShapeKernelError(
                f"dispatch signature maps to multiple locked Qwen shapes: {group.grid}"
            )
        if not matches:
            continue
        shape, launch = matches[0]
        projected.append(
            {
                "dispatch": group.to_dict(),
                "shape": shape.to_dict(),
                "launch": launch,
                "attribution": {
                    "n_basis": "locked_model_shape_and_exact_grid",
                    "k_basis": "locked_model_operator_map_not_profiler_observable",
                    "operator_basis": "locked_model_operator_map_not_profiler_observable",
                },
            }
        )
    return tuple(projected)


__all__ = [
    "KernelDispatchGroup",
    "LOCKED_QWEN3_8B_Q6_K_SHAPES",
    "LockedQwenShape",
    "ShapeKernelError",
    "group_mcp_kernel_evidence",
    "group_raw_dispatch_rows",
    "group_raw_kernel_csvs",
    "map_locked_qwen_shapes",
]
