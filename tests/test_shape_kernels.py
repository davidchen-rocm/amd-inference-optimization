from __future__ import annotations

import csv
from pathlib import Path

import pytest

from amd_inference_opt.shape_kernels import (
    ShapeKernelError,
    group_mcp_kernel_evidence,
    group_raw_dispatch_rows,
    group_raw_kernel_csvs,
    map_locked_qwen_shapes,
)

Q6_FUSED = "void mul_mat_vec_q<(ggml_type)14, 1, true, false>(void const*, void const*)"
Q6_UNFUSED = "void mul_mat_vec_q<(ggml_type)14, 1, false, false>(void const*, void const*)"


def _raw_row(
    *,
    grid_x: int,
    start: int,
    end: int,
    kernel_id: str | None = "252",
    name: str = Q6_FUSED,
) -> dict[str, str]:
    row = {
        "Kernel_Name": name,
        "Start_Timestamp": str(start),
        "End_Timestamp": str(end),
        "Grid_Size_X": str(grid_x),
        "Grid_Size_Y": "8",
        "Grid_Size_Z": "1",
        "Workgroup_Size_X": "32",
        "Workgroup_Size_Y": "8",
        "Workgroup_Size_Z": "1",
    }
    if kernel_id is not None:
        row["Kernel_Id"] = kernel_id
    return row


def test_raw_projection_keeps_shared_kernel_id_shapes_separate() -> None:
    groups = group_raw_dispatch_rows(
        [
            _raw_row(grid_x=393216, start=10, end=30),
            _raw_row(grid_x=131072, start=40, end=50),
            _raw_row(grid_x=393216, start=60, end=90),
        ]
    )

    assert [(group.grid, group.dispatch_count) for group in groups] == [
        ((393216, 8, 1), 2),
        ((131072, 8, 1), 1),
    ]
    assert groups[0].kernel_id == "252"
    assert groups[0].total_duration_ns == 50
    assert groups[0].average_duration_ns == 25


def test_raw_projection_falls_back_to_name_and_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "kernel_trace.csv"
    rows = [
        _raw_row(grid_x=393216, start=1, end=4, kernel_id=None),
        _raw_row(grid_x=393216, start=5, end=11, kernel_id=None),
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (group,) = group_raw_kernel_csvs(csv_path)

    assert group.identity == f"kernel_name:{Q6_FUSED}"
    assert group.dispatch_count == 2
    assert group.total_duration_ns == 9


def test_mcp_projection_groups_only_exact_identity_and_geometry() -> None:
    groups = group_mcp_kernel_evidence(
        {
            "schema": "rocm.mcp-kernel-evidence.v1",
            "kernels": [
                {
                    "kernel_id": "q6-fused",
                    "name": Q6_FUSED,
                    "grid": [393216, 8, 1],
                    "workgroup": [32, 8, 1],
                    "dispatch_count": 4,
                    "average_duration_ns": 100,
                },
                {
                    "metadata": {"kernel_id": "q6-fused"},
                    "name": Q6_FUSED,
                    "grid": [393216, 8, 1],
                    "workgroup": [32, 8, 1],
                    "dispatch_count": 2,
                    "total_duration_ns": 150,
                },
                {
                    "kernel_id": "q6-fused",
                    "name": Q6_FUSED,
                    "grid": [131072, 8, 1],
                    "workgroup": [32, 8, 1],
                    "dispatch_count": 3,
                    "total_duration_ns": 120,
                },
            ],
        }
    )

    assert [(group.grid, group.dispatch_count, group.total_duration_ns) for group in groups] == [
        ((393216, 8, 1), 6, 550),
        ((131072, 8, 1), 3, 120),
    ]


def test_locked_qwen_map_marks_inferred_k_and_expected_candidate_grid() -> None:
    baseline_groups = group_raw_dispatch_rows(
        [
            _raw_row(grid_x=393216, start=0, end=20),
            _raw_row(
                grid_x=131072,
                start=20,
                end=30,
                kernel_id="181",
                name=Q6_UNFUSED,
            ),
        ]
    )
    projected = map_locked_qwen_shapes(baseline_groups)

    assert [item["shape"]["n"] for item in projected] == [12288, 4096]
    assert projected[0]["shape"]["k"] == 4096
    assert projected[0]["shape"]["operator"] == "FFN gate/up projection"
    assert projected[0]["shape"]["candidate_grid"] == [49152, 8, 1]
    assert projected[0]["attribution"]["k_basis"].endswith("not_profiler_observable")
    assert all(item["launch"] == "baseline" for item in projected)


def test_shape_projection_rejects_missing_geometry_instead_of_collapsing() -> None:
    with pytest.raises(ShapeKernelError, match="missing grid"):
        group_mcp_kernel_evidence(
            {
                "kernels": [
                    {
                        "name": Q6_FUSED,
                        "workgroup": [32, 8, 1],
                        "dispatch_count": 1,
                    }
                ]
            }
        )


def test_locked_qwen_map_ignores_wrong_kernel_type_or_workgroup() -> None:
    q8_name = Q6_FUSED.replace("(ggml_type)14", "(ggml_type)8")
    q8_group = group_raw_dispatch_rows([_raw_row(grid_x=393216, start=0, end=1, name=q8_name)])
    wrong_workgroup = _raw_row(grid_x=393216, start=0, end=1)
    wrong_workgroup["Workgroup_Size_Y"] = "4"
    geometry_group = group_raw_dispatch_rows([wrong_workgroup])

    assert map_locked_qwen_shapes(q8_group) == ()
    assert map_locked_qwen_shapes(geometry_group) == ()
