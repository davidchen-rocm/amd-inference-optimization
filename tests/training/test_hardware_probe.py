import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from macfit_training import benchmark
from macfit_training.hardware_probe import (
    MAX_REPORT_BYTES,
    REPETITIONS,
    SCHEMA,
    SIZES,
    WARMUPS,
    ProbeFailed,
    ProbeTimeout,
    _encoded_report,
    correctness_passed,
    run_probe,
    summarize_times,
    throughput_tflops,
    validate_report,
)


def report():
    cases = []
    for size in SIZES:
        device = [0.5, 1.0, 1.5, 2.0, 2.5]
        wall = [1.0, 1.5, 2.0, 2.5, 3.0]
        cases.append(
            {
                "size": size,
                "finite_output": True,
                "warmups": WARMUPS,
                "repetitions": REPETITIONS,
                "device_samples_ms": device,
                "synchronized_wall_samples_ms": wall,
                "device": summarize_times(size, device),
                "synchronized_wall": summarize_times(size, wall),
            }
        )
    return {
        "schema": SCHEMA,
        "status": "succeeded",
        "microbenchmark_not_llm": True,
        "correctness": {
            "finite": True,
            "relative_l2_error": 0.001,
            "normalized_max_error": 0.002,
        },
        "runtime": {"rocm": "fixture", "gpu_name": "fixture", "total_vram_bytes": 1024},
        "gemm": cases,
    }


def test_dense_gemm_flop_accounting_converts_milliseconds_and_uses_real_samples():
    assert throughput_tflops(512, 1.0) == pytest.approx(0.268435456)
    assert throughput_tflops(8192, 1.0) == pytest.approx(1099.511627776)
    measured = summarize_times(512, [5.0, 1.0, 4.0, 2.0, 3.0])
    assert measured["median_ms"] == 3.0
    assert measured["min_ms"] == 1.0
    assert measured["max_ms"] == 5.0
    assert measured["median_tflops_per_second"] == pytest.approx(0.268435456 / 3)


@pytest.mark.parametrize("duration", [0, -1, True, "1", float("nan"), float("inf")])
def test_timing_cannot_publish_nonfinite_zero_negative_or_non_numeric_throughput(duration):
    with pytest.raises(ValueError):
        throughput_tflops(512, duration)


@pytest.mark.parametrize("shape", [0, 513, 16384, 512.0, True])
def test_only_bounded_integer_gemm_shapes_are_allowed(shape):
    with pytest.raises(ValueError):
        throughput_tflops(shape, 1.0)


def test_correctness_tolerance_checks_both_norms_and_rejects_nan():
    assert correctness_passed(0.01, 0.02)
    assert not correctness_passed(0.01001, 0.0)
    assert not correctness_passed(0.0, 0.02001)
    for value in (float("nan"), float("inf"), -0.1, True, None, "0"):
        assert not correctness_passed(value, 0.0)
        assert not correctness_passed(0.0, value)


def test_report_requires_all_finite_shapes_and_summaries_matching_raw_samples():
    valid = report()
    assert validate_report(valid) == valid
    incomplete = copy.deepcopy(valid)
    incomplete["gemm"].pop()
    changed_summary = copy.deepcopy(valid)
    changed_summary["gemm"][0]["device"]["median_tflops_per_second"] *= 2
    nonfinite_output = copy.deepcopy(valid)
    nonfinite_output["gemm"][0]["finite_output"] = False
    failed_check = copy.deepcopy(valid)
    failed_check["correctness"]["relative_l2_error"] = 0.1
    no_runtime = copy.deepcopy(valid)
    no_runtime["runtime"]["rocm"] = None
    for invalid in (incomplete, changed_summary, nonfinite_output, failed_check, no_runtime):
        with pytest.raises(ValueError):
            validate_report(invalid)


def test_evidence_has_a_strict_sub_one_mib_bound_and_rejects_nan():
    assert len(_encoded_report(report())) < MAX_REPORT_BYTES
    with pytest.raises(ValueError):
        _encoded_report({"large": "x" * MAX_REPORT_BYTES})
    with pytest.raises(ValueError):
        _encoded_report({"value": float("nan")})


def test_existing_evidence_and_symlinks_are_preserved(tmp_path):
    output = tmp_path / "probe.json"
    output.write_text('{"verified":true}')
    with pytest.raises(ValueError, match="fresh output"):
        run_probe(output)
    assert output.read_text() == '{"verified":true}'
    alias = tmp_path / "alias.json"
    alias.symlink_to(output)
    with pytest.raises(ValueError, match="fresh output"):
        run_probe(alias)


@pytest.mark.parametrize("budget", [0, -1, 181, True, 1.5])
def test_parent_rejects_an_unbounded_or_non_integer_budget_before_spawning(tmp_path, budget):
    with pytest.raises(ValueError):
        run_probe(tmp_path / "unused.json", walltime_seconds=budget)
    assert not list(tmp_path.iterdir())


def test_real_child_deadline_interrupts_lock_wait_without_importing_gpu(tmp_path):
    import fcntl

    lock = tmp_path / "gpu.lock"
    output = tmp_path / "probe.json"
    script = tmp_path / "wait.py"
    script.write_text(
        "import os\n"
        "from amd_inference_opt.resource_lock import exclusive_gpu_lock\n"
        "with exclusive_gpu_lock(os.environ['MACFIT_GPU_LOCK']):\n"
        " raise RuntimeError('The locked child must never acquire this lease')\n"
    )
    env = os.environ.copy()
    os.environ["MACFIT_GPU_LOCK"] = str(lock)
    try:
        with lock.open("w") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            started = time.monotonic()
            with pytest.raises(ProbeTimeout):
                run_probe(output, walltime_seconds=1, child_command=[sys.executable, str(script)])
            assert time.monotonic() - started < 5
    finally:
        if "MACFIT_GPU_LOCK" in env:
            os.environ["MACFIT_GPU_LOCK"] = env["MACFIT_GPU_LOCK"]
        else:
            os.environ.pop("MACFIT_GPU_LOCK", None)
    assert not output.exists()
    assert not list(tmp_path.glob(".hardware-probe-*"))


def test_parent_atomic_publication_accepts_complete_child_evidence(tmp_path):
    script = tmp_path / "child.py"
    fixture = json.dumps(report())
    script.write_text(
        "import json,os,sys\n"
        f"report=json.loads({fixture!r})\n"
        "path=sys.argv[sys.argv.index('--output')+1]\n"
        "with open(path,'w') as stream: json.dump(report,stream)\n"
        "os.chmod(path,0o600)\n"
    )
    output = tmp_path / "probe.json"
    result = run_probe(output, child_command=[sys.executable, str(script)])
    assert result == report()
    assert json.loads(output.read_text()) == result
    assert output.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".hardware-probe-*"))


def test_nonzero_child_exit_never_publishes_success_evidence(tmp_path):
    output = tmp_path / "probe.json"
    with pytest.raises(ProbeFailed):
        run_probe(output, child_command=[sys.executable, "-c", "raise SystemExit(2)"])
    assert not output.exists()


def test_descendant_stops_even_after_successful_group_leader_exit(tmp_path):
    marker = tmp_path / "descendant-heartbeat"
    descendant = (
        "import signal,time\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        f"with open({str(marker)!r},'a') as stream:\n"
        " while True:\n"
        "  stream.write('x'); stream.flush(); time.sleep(0.03)\n"
    )
    script = tmp_path / "child-with-descendant.py"
    fixture = json.dumps(report())
    script.write_text(
        "import json,os,subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}])\n"
        f"while not os.path.exists({str(marker)!r}): time.sleep(0.01)\n"
        f"report=json.loads({fixture!r})\n"
        "path=sys.argv[sys.argv.index('--output')+1]\n"
        "with open(path,'w') as stream: json.dump(report,stream)\n"
    )
    run_probe(tmp_path / "probe.json", child_command=[sys.executable, str(script)])
    saved_size = marker.stat().st_size
    assert saved_size > 0
    time.sleep(0.2)
    assert marker.stat().st_size == saved_size


def boundary_worker(tmp_path, path):
    source = tmp_path / "boundary-child.py"
    if path == "succeeded":
        fixture = json.dumps(report())
        source.write_text(
            "import json,sys\n"
            "path=sys.argv[sys.argv.index('--output')+1]\n"
            f"with open(path,'w') as stream: json.dump(json.loads({fixture!r}),stream)\n"
        )
    elif path == "timed_out":
        source.write_text("import time\ntime.sleep(30)\n")
    else:
        source.write_text("raise SystemExit(2)\n")
    return [sys.executable, str(source)]


@pytest.mark.parametrize("path", ["succeeded", "timed_out", "worker_failed"])
def test_cleanup_signals_the_real_child_group_exactly_once_on_each_path(
    tmp_path, monkeypatch, path
):
    original = benchmark.terminate_group
    calls = []

    def count_cleanup(process):
        calls.append(process.pid)
        if len(calls) > 1:
            raise PermissionError("An already-cleaned process group must not be signalled again")
        original(process)

    monkeypatch.setattr(benchmark, "terminate_group", count_cleanup)
    options = {"walltime_seconds": 1, "child_command": boundary_worker(tmp_path, path)}
    if path == "succeeded":
        assert run_probe(tmp_path / "bounded.json", **options) == report()
    else:
        error_type = ProbeTimeout if path == "timed_out" else ProbeFailed
        with pytest.raises(error_type):
            run_probe(tmp_path / "bounded.json", **options)
    assert len(calls) == 1


def test_cleanup_permission_error_is_not_retried_or_published_as_success(tmp_path, monkeypatch):
    original = benchmark.terminate_group
    calls = []

    def forbidden_cleanup(process):
        calls.append(process.pid)
        original(process)
        raise PermissionError("Private process ownership details")

    monkeypatch.setattr(benchmark, "terminate_group", forbidden_cleanup)
    output = tmp_path / "probe.json"
    with pytest.raises(PermissionError):
        run_probe(output, child_command=boundary_worker(tmp_path, "succeeded"))
    assert len(calls) == 1
    assert not output.exists()
    assert not list(tmp_path.glob(".hardware-probe-*"))


def test_original_deadline_exception_survives_cleanup_permission_failure(tmp_path, monkeypatch):
    original = benchmark.terminate_group
    calls = []

    def forbidden_cleanup(process):
        calls.append(process.pid)
        original(process)
        raise PermissionError("Private process ownership details")

    monkeypatch.setattr(benchmark, "terminate_group", forbidden_cleanup)
    output = tmp_path / "probe.json"
    with pytest.raises(ProbeTimeout) as error:
        run_probe(output, walltime_seconds=1, child_command=boundary_worker(tmp_path, "timed_out"))
    assert isinstance(error.value.__cause__, PermissionError)
    assert error.value.__notes__ == ["Process-group cleanup failed: PermissionError"]
    assert len(calls) == 1
    assert not output.exists()
    assert not list(tmp_path.glob(".hardware-probe-*"))


def test_cpu_import_and_cli_help_do_not_import_accelerator_libraries():
    root = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys\n"
        "import macfit_training.hardware_probe as probe\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'transformers' not in sys.modules\n"
        "try: probe.main(['--help'])\n"
        "except SystemExit as exc: assert exc.code == 0\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
        stdout=subprocess.DEVNULL,
    )
