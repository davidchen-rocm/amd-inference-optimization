from pathlib import Path

from amd_inference_opt.raw_runtime_trace import normalize_raw_runtime_trace


def test_normalizes_runtime_launch_copy_and_allocation_evidence(tmp_path: Path) -> None:
    (tmp_path / "1_hip_api_trace.csv").write_text(
        '"Function","Start_Timestamp","End_Timestamp"\n'
        '"hipMalloc",0,10\n'
        '"hipLaunchKernel",20,25\n'
        '"hipGraphLaunch",30,35\n',
        encoding="utf-8",
    )
    (tmp_path / "1_kernel_trace.csv").write_text(
        '"Kernel_Name","Start_Timestamp","End_Timestamp"\n'
        '"a",25,28\n'
        '"b",40,45\n',
        encoding="utf-8",
    )
    (tmp_path / "1_memory_copy_trace.csv").write_text(
        '"Direction","Start_Timestamp","End_Timestamp"\n'
        '"MEMORY_COPY_HOST_TO_DEVICE",11,19\n',
        encoding="utf-8",
    )

    result = normalize_raw_runtime_trace(tmp_path, workload_exit_code=0)

    assert result["trace_status"] == "completed"
    assert result["hip_kernel_launch_count"] == 1
    assert result["graph_launch_count"] == 1
    assert result["kernel_dispatch_count"] == 2
    assert result["memory_allocation_count"] == 1
    assert result["memory_copy_count"] == 1
    assert result["cpu_launch_gap_mean_ns"] == 5
    assert result["gpu_idle_gap_mean_ns"] == 12
    assert len(result["raw_artifacts"]) == 3


def test_missing_csv_and_event_limit_are_partial(tmp_path: Path) -> None:
    (tmp_path / "1_hip_api_trace.csv").write_text(
        '"Function","Start_Timestamp","End_Timestamp"\n'
        '"hipLaunchKernel",0,1\n'
        '"hipLaunchKernel",2,3\n',
        encoding="utf-8",
    )

    result = normalize_raw_runtime_trace(
        tmp_path,
        workload_exit_code=0,
        max_events_per_type=1,
    )

    assert result["trace_status"] == "partial"
    assert "event normalization budget exceeded" in result["warning_details"]
    assert "missing kernel CSV" in result["warning_details"]
