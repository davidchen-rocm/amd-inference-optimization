"""Bounded ROCm BF16 GEMM evidence; this module imports no accelerator runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark import _ProcessGroupCleanup

SCHEMA = "macfit-rocm-hardware-probe-v1"
SIZES = (512, 2048, 4096, 8192)
WARMUPS = 2
REPETITIONS = 5
MAX_REPORT_BYTES = 1024 * 1024 - 1
MAX_WALLTIME_SECONDS = 180
RELATIVE_L2_TOLERANCE = 0.01
NORMALIZED_MAX_TOLERANCE = 0.02


class ProbeTimeout(RuntimeError):
    """The isolated probe exhausted its total budget, including lock waiting."""


class ProbeFailed(RuntimeError):
    """No complete, verified hardware evidence was produced."""


def positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite positive number.")
    measured = float(value)
    if not math.isfinite(measured) or measured <= 0:
        raise ValueError(f"{label} must be a finite positive number.")
    return measured


def throughput_tflops(size: int, milliseconds: float) -> float:
    """Count conventional dense square GEMM FLOPs, never an LLM throughput estimate."""
    if type(size) is not int or size not in SIZES:
        raise ValueError("Only the bounded square GEMM sizes are supported.")
    duration = positive_finite(milliseconds, "GEMM milliseconds")
    return positive_finite(2 * size**3 / (duration * 1e9), "GEMM TFLOPs/s")


def summarize_times(size: int, values: list[float]) -> dict[str, float]:
    if not isinstance(values, list) or len(values) != REPETITIONS:
        raise ValueError("Each measured GEMM requires exactly five timing samples.")
    durations = [positive_finite(value, "GEMM milliseconds") for value in values]
    median = statistics.median(durations)
    return {
        "median_ms": median,
        "min_ms": min(durations),
        "max_ms": max(durations),
        "median_tflops_per_second": throughput_tflops(size, median),
    }


def correctness_passed(relative_l2_error: float, normalized_max_error: float) -> bool:
    for value in (relative_l2_error, normalized_max_error):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(value) or value < 0:
            return False
    return (
        relative_l2_error <= RELATIVE_L2_TOLERANCE
        and normalized_max_error <= NORMALIZED_MAX_TOLERANCE
    )


def _encoded_report(report: dict[str, Any]) -> bytes:
    try:
        contents = (json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("The hardware report must be finite JSON.") from exc
    if len(contents) > MAX_REPORT_BYTES:
        raise ValueError("The hardware report exceeds the one MiB evidence bound.")
    return contents


def _write_atomic(path: Path, report: dict[str, Any]) -> None:
    contents = _encoded_report(report)
    descriptor, temporary = tempfile.mkstemp(prefix=".probe-pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_report(report: Any) -> dict[str, Any]:
    """Reject partial or malformed child output before exposing success evidence."""
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError("The child did not produce a hardware-probe report.")
    _encoded_report(report)
    if report.get("status") != "succeeded" or report.get("microbenchmark_not_llm") is not True:
        raise ValueError("The hardware-probe report is not a complete microbenchmark.")
    check = report.get("correctness")
    if not isinstance(check, dict) or check.get("finite") is not True:
        raise ValueError("A finite correctness check is required.")
    if not correctness_passed(check.get("relative_l2_error"), check.get("normalized_max_error")):
        raise ValueError("The BF16 result did not pass its FP32 reference check.")
    runtime = report.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("rocm") or not runtime.get("gpu_name"):
        raise ValueError("The report must identify the real ROCm GPU runtime.")
    positive_finite(runtime.get("total_vram_bytes"), "Total VRAM")
    cases = report.get("gemm")
    if not isinstance(cases, list) or len(cases) != len(SIZES):
        raise ValueError("All bounded GEMM sizes must be measured.")
    for size, case in zip(SIZES, cases, strict=True):
        if not isinstance(case, dict) or case.get("size") != size:
            raise ValueError("The measured GEMM shapes are incomplete or unordered.")
        if case.get("finite_output") is not True:
            raise ValueError("Every measured GEMM output must be finite.")
        if case.get("warmups") != WARMUPS or case.get("repetitions") != REPETITIONS:
            raise ValueError("Every GEMM must use the documented sampling protocol.")
        for key in ("device", "synchronized_wall"):
            durations = case.get(key + "_samples_ms")
            expected = summarize_times(size, durations)
            if case.get(key) != expected:
                raise ValueError("A GEMM summary does not match its actual timing samples.")
    return report


def _correctness(torch: Any) -> dict[str, Any]:
    # Reference inputs are quantized first, so input rounding is not blamed on GEMM.
    generator = torch.Generator(device="cpu").manual_seed(142)
    a_cpu = torch.randn((64, 64), generator=generator).to(torch.bfloat16)
    b_cpu = torch.randn((64, 64), generator=generator).to(torch.bfloat16)
    reference = a_cpu.float() @ b_cpu.float()
    actual = (a_cpu.to("cuda:0") @ b_cpu.to("cuda:0")).float().cpu()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    difference = actual - reference
    l2 = float(torch.linalg.vector_norm(difference).item())
    reference_l2 = float(torch.linalg.vector_norm(reference).item())
    maximum = float(difference.abs().max().item())
    reference_maximum = float(reference.abs().max().item())
    relative = l2 / max(reference_l2, 1e-12)
    normalized = maximum / max(reference_maximum, 1e-12)
    if not finite or not correctness_passed(relative, normalized):
        raise ProbeFailed("BF16 GEMM did not pass the finite FP32 reference check.")
    return {
        "shape": [64, 64],
        "reference": "CPU FP32 matmul of the identical BF16-quantized inputs",
        "finite": finite,
        "relative_l2_error": relative,
        "relative_l2_tolerance": RELATIVE_L2_TOLERANCE,
        "absolute_max_error": maximum,
        "reference_absolute_max": reference_maximum,
        "normalized_max_error": normalized,
        "normalized_max_tolerance": NORMALIZED_MAX_TOLERANCE,
    }


def _measure_gemm(torch: Any, size: int) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats(0)
    a = torch.randn((size, size), dtype=torch.bfloat16, device="cuda:0")
    b = torch.randn((size, size), dtype=torch.bfloat16, device="cuda:0")
    output = torch.empty_like(a)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUPS):
        start.record()
        torch.mm(a, b, out=output)
        end.record()
    torch.cuda.synchronize(0)
    device_samples, wall_samples = [], []
    for _ in range(REPETITIONS):
        torch.cuda.synchronize(0)
        wall_start = time.perf_counter()
        start.record()
        torch.mm(a, b, out=output)
        end.record()
        torch.cuda.synchronize(0)
        wall_samples.append((time.perf_counter() - wall_start) * 1000)
        device_samples.append(float(start.elapsed_time(end)))
    finite = bool(torch.isfinite(output).all().item())
    if not finite:
        raise ProbeFailed("A measured BF16 GEMM output was not finite.")
    return {
        "size": size,
        "shape": [size, size, size],
        "warmups": WARMUPS,
        "repetitions": REPETITIONS,
        "finite_output": finite,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "device_samples_ms": device_samples,
        "synchronized_wall_samples_ms": wall_samples,
        "device": summarize_times(size, device_samples),
        "synchronized_wall": summarize_times(size, wall_samples),
    }


def collect_report() -> dict[str, Any]:
    """Child-only GPU operations while holding the same lock as training workers."""
    from amd_inference_opt.resource_lock import exclusive_gpu_lock

    lock = Path(os.environ.get("MACFIT_GPU_LOCK", "/tmp/macfit-training-gpu.lock"))
    with exclusive_gpu_lock(lock):
        import torch

        if not torch.cuda.is_available() or not torch.version.hip:
            raise ProbeFailed("An available ROCm GPU is required.")
        torch.cuda.set_device(0)
        torch.manual_seed(142)
        torch.cuda.manual_seed_all(142)
        properties = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        with torch.inference_mode():
            correctness = _correctness(torch)
            cases = [_measure_gemm(torch, size) for size in SIZES]
        return {
            "schema": SCHEMA,
            "status": "succeeded",
            "created_at": datetime.now(UTC).isoformat(),
            "microbenchmark_not_llm": True,
            "flop_accounting": "2 * n^3 for one dense square BF16 GEMM",
            "dtype": "bfloat16",
            "seed": 142,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": {
                "gpu_name": str(properties.name),
                "architecture": str(getattr(properties, "gcnArchName", "unknown")),
                "total_vram_bytes": int(properties.total_memory),
                "driver_visible_total_bytes": int(total),
                "free_vram_before_probe_bytes": int(free),
                "torch": str(torch.__version__),
                "rocm": str(torch.version.hip),
                "python": sys.version.split()[0],
            },
            "correctness": correctness,
            "gemm": cases,
            "limitations": [
                "Isolated dense GEMM throughput does not predict LLM tokens/second.",
                "Device events and synchronized host wall time measure different intervals.",
                "The small reference check does not verify all large GEMM outputs elementwise.",
                "GPU virtualization, contention, clock state and shape affect the result.",
                "Lock waiting, runtime startup and allocation count toward the parent deadline.",
            ],
        }


def run_probe(
    output: Path,
    *,
    walltime_seconds: int = 120,
    child_command: list[str] | None = None,
) -> dict[str, Any]:
    """Supervise a real subprocess; injected commands exercise CPU process boundaries."""
    if type(walltime_seconds) is not int or not 1 <= walltime_seconds <= MAX_WALLTIME_SECONDS:
        raise ValueError("The total wall-time budget must be 1 to 180 integer seconds.")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Use a fresh output path; existing evidence will not be overwritten.")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("The evidence parent must be an existing real directory.")
    command = child_command or [sys.executable, "-m", "macfit_training.hardware_probe", "--worker"]
    with tempfile.TemporaryDirectory(prefix=".hardware-probe-", dir=output.parent) as temporary:
        pending = Path(temporary) / "report.json"
        process = subprocess.Popen(
            [*command, "--output", str(pending)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        cleanup = _ProcessGroupCleanup(process)
        try:
            try:
                code = process.wait(timeout=walltime_seconds)
            except subprocess.TimeoutExpired as exc:
                raise ProbeTimeout(
                    "The hardware probe reached its total wall-time deadline."
                ) from exc
            cleanup()
            if code != 0:
                raise ProbeFailed("The child failed; no verified hardware report was published.")
            descriptor = os.open(pending, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_REPORT_BYTES:
                    raise ProbeFailed("The child report is not a bounded regular JSON file.")
                contents = stream.read(MAX_REPORT_BYTES + 1)
            if len(contents) > MAX_REPORT_BYTES:
                raise ProbeFailed("The hardware report exceeded its evidence bound.")
            report = validate_report(json.loads(contents))
            # Link a complete file atomically without replacing evidence created in a race.
            os.link(pending, output)
            return report
        except BaseException as error:
            try:
                cleanup()
            except Exception as cleanup_error:
                error.add_note("Process-group cleanup failed: " + type(cleanup_error).__name__)
                raise error from cleanup_error
            raise
        finally:
            cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure bounded real ROCm BF16 GEMM throughput.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--walltime-seconds", type=int, default=120)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.umask(0o077)

    def interrupted(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.worker:
            _write_atomic(args.output, validate_report(collect_report()))
        else:
            run_probe(args.output, walltime_seconds=args.walltime_seconds)
        return 0
    except KeyboardInterrupt:
        print(
            "The hardware probe was interrupted; no complete evidence was published.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        # Exception text from device libraries may contain local paths or environment details.
        label = "deadline" if isinstance(exc, ProbeTimeout) else "validation or runtime failure"
        print(
            f"Hardware probe stopped: {label}. No complete evidence was published.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
