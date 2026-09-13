from __future__ import annotations

import hashlib
from pathlib import Path

from amd_inference_opt.command import CommandResult
from amd_inference_opt.raw_rocprof import RawRocprofAdapter, normalize_kernel_csvs


def _kernel_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '"Kind","Agent_Id","Kernel_Name","Start_Timestamp","End_Timestamp"\n'
        '"KERNEL_DISPATCH",1,"hot_kernel",100,300\n'
        '"KERNEL_DISPATCH",1,"hot_kernel",400,500\n'
        '"KERNEL_DISPATCH",1,"small_kernel",600,650\n',
        encoding="utf-8",
    )


def test_normalizes_raw_kernel_csv_and_hashes_artifacts(tmp_path: Path) -> None:
    trace_root = tmp_path / "capture"
    csv_path = trace_root / "trace" / "kernel_trace.csv"
    _kernel_csv(csv_path)
    (trace_root / "stdout.log").write_text("bench output\n", encoding="utf-8")

    evidence, artifacts = normalize_kernel_csvs(
        trace_root,
        max_trace_bytes=1_000_000,
        max_trace_files=8,
        max_events_per_type=100,
        max_percentile_samples_per_kernel=10,
    )

    assert evidence["status"] == "completed"
    assert evidence["aggregate_timing_complete"] is True
    assert evidence["hotspot_ranking_reliable"] is True
    assert evidence["coverage_percent"] == 100.0
    assert [kernel["name"] for kernel in evidence["kernels"]] == [
        "hot_kernel",
        "small_kernel",
    ]
    hot = evidence["kernels"][0]
    assert hot["dispatch_count"] == 2
    assert hot["total_duration_ns"] == 300
    assert hot["average_duration_ns"] == 150
    assert hot["gpu_kernel_time_share_percent"] == 300 / 350 * 100
    manifest = {artifact.path: artifact for artifact in artifacts}
    assert manifest["trace/kernel_trace.csv"].sha256 == hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    assert manifest["stdout.log"].size_bytes == len("bench output\n")


def test_event_budget_marks_raw_hotspot_ranking_partial(tmp_path: Path) -> None:
    trace_root = tmp_path / "capture"
    _kernel_csv(trace_root / "kernel_trace.csv")

    evidence, _ = normalize_kernel_csvs(
        trace_root,
        max_trace_bytes=1_000_000,
        max_trace_files=8,
        max_events_per_type=1,
        max_percentile_samples_per_kernel=10,
    )

    assert evidence["status"] == "partial"
    assert evidence["aggregate_timing_complete"] is False
    assert evidence["hotspot_ranking_reliable"] is False
    assert "raw kernel event budget was exceeded" in evidence["warnings"]


def test_same_kernel_name_is_split_by_launch_geometry_and_resources(tmp_path: Path) -> None:
    trace_root = tmp_path / "geometry"
    csv_path = trace_root / "kernel_trace.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(
        '"Kernel_Name","Start_Timestamp","End_Timestamp","Agent_Id",'
        '"Grid_Size_X","Grid_Size_Y","Grid_Size_Z","Workgroup_Size_X",'
        '"Workgroup_Size_Y","Workgroup_Size_Z","VGPR_Count","LDS_Block_Size",'
        '"Scratch_Size","SGPR_Count","Accum_VGPR_Count"\n'
        '"shared_name",0,100,"GPU 0",393216,8,1,32,8,1,24,2048,0,32,0\n'
        '"shared_name",200,240,"GPU 0",131072,8,1,32,8,1,16,1024,0,32,0\n',
        encoding="utf-8",
    )

    evidence, _ = normalize_kernel_csvs(
        trace_root,
        max_trace_bytes=1_000_000,
        max_trace_files=8,
        max_events_per_type=100,
        max_percentile_samples_per_kernel=10,
    )

    assert len(evidence["kernels"]) == 2
    assert [item["grid"] for item in evidence["kernels"]] == [
        [393216, 8, 1],
        [131072, 8, 1],
    ]
    assert evidence["kernels"][0]["resource_usage"]["vgpr_count"] == 24
    assert evidence["kernels"][1]["resource_usage"]["lds_bytes"] == 1024
    assert evidence["kernels"][0]["kernel_id"] != evidence["kernels"][1]["kernel_id"]


class _FakeRunner:
    def run(self, argv, **kwargs):
        trace_dir = Path(argv[argv.index("--output-directory") + 1])
        _kernel_csv(trace_dir / "kernel_trace.csv")
        Path(kwargs["stdout_path"]).write_text("stdout\n", encoding="utf-8")
        Path(kwargs["stderr_path"]).write_text("", encoding="utf-8")
        return CommandResult(
            argv=tuple(argv),
            cwd=str(Path(kwargs["cwd"]).resolve()),
            started_at="2026-01-01T00:00:00+00:00",
            duration_seconds=1.0,
            exit_code=0,
            stdout="stdout\n",
            stderr="",
            environment=dict(kwargs["env"]),
            unset_environment=tuple(kwargs["unset_env"]),
            timeout_seconds=kwargs["timeout_seconds"],
        )


def test_raw_adapter_builds_fixed_argv_and_preserves_execution_coordinates(
    tmp_path: Path,
) -> None:
    profiler = tmp_path / "rocprofv3"
    profiler.write_text("fixture", encoding="utf-8")
    adapter = RawRocprofAdapter(_FakeRunner(), rocprofv3_path=profiler)

    result = adapter.profile(
        ["/opt/llama-bench", "-n", "128"],
        cwd=tmp_path,
        environment={"HIP_VISIBLE_DEVICES": "0"},
        unset_environment=["LLAMA_Q4_RDNA_SIDECAR"],
        output_dir=tmp_path / "capture",
        timeout_seconds=60,
        max_trace_bytes=1_000_000,
        max_trace_files=8,
        max_events_per_type=100,
        max_percentile_samples_per_kernel=10,
    )

    assert result.profiler_argv[:6] == (
        str(profiler.resolve()),
        "--kernel-trace",
        "--stats",
        "--output-format",
        "csv",
        "--output-directory",
    )
    assert result.profiler_argv[-4:] == ("--", "/opt/llama-bench", "-n", "128")
    assert result.environment == {"HIP_VISIBLE_DEVICES": "0"}
    assert result.unset_environment == ("LLAMA_Q4_RDNA_SIDECAR",)
    assert result.kernel_evidence["status"] == "completed"
    assert {artifact.path for artifact in result.raw_artifacts} >= {
        "stdout.log",
        "stderr.log",
        "trace/kernel_trace.csv",
    }
