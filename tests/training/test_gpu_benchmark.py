import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import macfit_training.benchmark as benchmark
from macfit_training.benchmark import (
    BENCHMARK_MODELS,
    MAX_EVIDENCE_BYTES,
    integer_grid,
    parse_stop_at,
    run_campaign,
    summarize_samples,
    validate_plan,
    write_atomic,
)
from macfit_training.config import MODELS
from macfit_training.service.backup import export_archive
from macfit_training.service.settings import Settings
from macfit_training.service.store import JobStore


def plan(**changes):
    return {
        "models": ["qwen3-0-6b"],
        "batch_sizes": [1, 4],
        "prompt_tokens": [128, 512],
        "decode_steps": 16,
        "warmup_repetitions": 1,
        "repetitions": 2,
        "walltime_seconds": 30,
        "model_timeout_seconds": 10,
        "seed": 42,
        "stop_at": None,
        **changes,
    }


def sample(**changes):
    return {
        "prefill_seconds": 0.1,
        "decode_seconds": 0.2,
        "prefill_tokens_per_second": 1280.0,
        "decode_tokens_per_second": 80.0,
        "finite_logits": True,
        "output_ids_sha256": "a" * 64,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        **changes,
    }


def test_benchmark_larger_models_are_pinned_and_not_added_to_web_training():
    assert "qwen3-14b" not in MODELS and "qwen3-32b" not in MODELS
    pinned = validate_plan(plan(models=["qwen3-14b", "qwen3-32b"]))
    assert pinned["model_identities"] == [
        BENCHMARK_MODELS["qwen3-14b"],
        BENCHMARK_MODELS["qwen3-32b"],
    ]
    assert all(len(item["revision"]) == 40 for item in pinned["model_identities"])
    assert validate_plan(pinned) == pinned


@pytest.mark.parametrize(
    "changes",
    [
        {"models": ["arbitrary"]},
        {"models": ["qwen3-4b", "qwen3-4b"]},
        {"batch_sizes": [65]},
        {"prompt_tokens": [32769]},
        {"prompt_tokens": [128, 128]},
        {"decode_steps": 0},
        {"repetitions": 1},
        {"warmup_repetitions": 0},
        {"walltime_seconds": 21601},
        {"stop_at": "2026-09-30T23:45:00"},
    ],
)
def test_benchmark_rejects_unbounded_or_ambiguous_operator_protocol(changes):
    with pytest.raises(ValueError):
        validate_plan(plan(**changes))


def test_grids_and_deadline_preserve_exact_coordinates():
    assert integer_grid("1,4,16", 64) == [1, 4, 16]
    with pytest.raises(ValueError):
        integer_grid("1,1", 64)
    assert parse_stop_at("2026-09-30T23:45:00-04:00").isoformat() == "2026-10-01T03:45:00+00:00"


def test_summary_exposes_variation_and_does_not_claim_quality():
    result = summarize_samples(
        [sample(), sample(decode_tokens_per_second=120.0, peak_allocated_bytes=150)]
    )
    assert result["decode"]["median_tokens_per_second"] == 100.0
    assert result["decode"]["coefficient_of_variation"] > 0
    assert result["peak_allocated_bytes"] == 150
    assert result["correctness"] == {
        "finite_logits": True,
        "logit_checks": "prefill_and_final_decode",
        "greedy_repeats_identical": True,
        "semantic_quality_evaluated": False,
    }
    result = summarize_samples([sample(), sample(output_ids_sha256="b" * 64)])
    assert result["correctness"]["greedy_repeats_identical"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"finite_logits": False},
        {"prefill_seconds": float("nan")},
        {"decode_seconds": 0},
        {"decode_tokens_per_second": float("inf")},
    ],
)
def test_nonfinite_or_unmeasured_samples_are_not_successful_evidence(changes):
    with pytest.raises(ValueError):
        summarize_samples([sample(), sample(**changes)])


def test_atomic_evidence_rejects_oversize_or_nan_without_changing_verified_file(tmp_path):
    path = tmp_path / "evidence.json"
    write_atomic(path, {"verified": True})
    original = path.read_bytes()
    for value in ({"text": "x" * 1000}, {"x": float("nan")}):
        with pytest.raises(ValueError):
            write_atomic(path, value, max_bytes=100)
        assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def fake_worker(tmp_path, code):
    source = tmp_path / "worker.py"
    source.write_text(
        "import json, sys, time\n"
        "config=json.load(open(sys.argv[1]))\n"
        "model=config['identity']['id']\n"
        "def emit(stage, **kw):\n"
        " print(json.dumps({'model_id': model, 'stage': stage, **kw}), flush=True)\n" + code
    )
    return [sys.executable, str(source)]


def test_child_oom_is_recorded_and_next_model_runs_with_durable_partial_evidence(tmp_path):
    command = fake_worker(
        tmp_path,
        "if model == 'qwen3-0-6b':\n"
        " emit('out_of_memory', error_type='OutOfMemoryError')\n"
        " sys.exit(2)\n"
        "emit('runtime_ready', runtime={'test_fixture': True})\n"
        "emit('case_completed', result={'test_case': True})\n"
        "emit('succeeded')\n",
    )
    output = tmp_path / "gpu-capability-test.json"
    result = run_campaign(plan(models=["qwen3-0-6b", "qwen3-4b"]), output, worker_command=command)
    assert result["status"] == "incomplete"
    assert [item["status"] for item in result["models"]] == ["out_of_memory", "succeeded"]
    assert json.loads(output.read_text()) == result
    model_snapshot = json.loads((tmp_path / "gpu-capability-test-qwen3-4b.json").read_text())
    assert model_snapshot["cases"] == [{"test_case": True}]
    events = [
        json.loads(line) for line in output.with_suffix(".events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["stage"] == "campaign_finished"
    assert events[1]["stage"] == "out_of_memory"
    assert len(model_snapshot["identity"]["revision"]) == 40


def test_hard_child_timeout_includes_waiting_and_stops_campaign_without_false_success(tmp_path):
    command = fake_worker(tmp_path, "emit('waiting_for_gpu')\ntime.sleep(30)\n")
    output = tmp_path / "gpu-capability-timeout.json"
    started = time.monotonic()
    result = run_campaign(plan(model_timeout_seconds=1), output, worker_command=command)
    assert time.monotonic() - started < 5
    assert result["status"] == "incomplete"
    assert result["models"][0]["status"] == "timed_out"
    assert any(event["stage"] == "model_timeout" for event in result["events"])


def test_timeout_kills_descendant_after_its_group_leader_has_exited(tmp_path):
    marker = tmp_path / "descendant-heartbeat"
    descendant = (
        "import signal,time\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        f"with open({str(marker)!r},'a') as stream:\n"
        " while True:\n"
        "  stream.write('x'); stream.flush(); time.sleep(0.05)\n"
    )
    command = fake_worker(
        tmp_path,
        "import subprocess\n"
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}])\n"
        "emit('succeeded')\n",
    )
    result = run_campaign(
        plan(model_timeout_seconds=1),
        tmp_path / "gpu-capability-descendant.json",
        worker_command=command,
    )
    assert result["models"][0]["status"] == "timed_out"
    saved_size = marker.stat().st_size
    assert saved_size > 0
    time.sleep(0.2)
    assert marker.stat().st_size == saved_size


def test_successful_leader_cleanup_kills_descendant_that_redirected_stdio(tmp_path):
    marker = tmp_path / "successful-descendant-heartbeat"
    descendant = (
        "import signal,time\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        f"with open({str(marker)!r},'a') as stream:\n"
        " while True:\n"
        "  stream.write('x'); stream.flush(); time.sleep(0.05)\n"
    )
    command = fake_worker(
        tmp_path,
        "import subprocess\nfrom pathlib import Path\n"
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}], "
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        f"while not Path({str(marker)!r}).exists():\n time.sleep(0.01)\n"
        "emit('succeeded')\n",
    )
    result = run_campaign(
        plan(), tmp_path / "gpu-capability-successful-descendant.json", worker_command=command
    )
    assert result["status"] == "succeeded"
    saved_size = marker.stat().st_size
    assert saved_size > 0
    time.sleep(0.2)
    assert marker.stat().st_size == saved_size


@pytest.mark.parametrize("path", ["succeeded", "timed_out", "invalid_event"])
def test_process_group_cleanup_runs_exactly_once_across_all_supervisor_paths(
    tmp_path, monkeypatch, path
):
    bodies = {
        "succeeded": "emit('succeeded')\n",
        "timed_out": "time.sleep(30)\n",
        "invalid_event": "print('invalid-json',flush=True)\ntime.sleep(30)\n",
    }
    command = fake_worker(tmp_path, bodies[path])
    original = benchmark.terminate_group
    cleanup_calls = []

    def cleanup_once(process):
        cleanup_calls.append(process.pid)
        if len(cleanup_calls) > 1:
            raise PermissionError("The already-cleaned process group cannot be signalled again")
        original(process)

    monkeypatch.setattr(benchmark, "terminate_group", cleanup_once)
    output = tmp_path / "gpu-capability-cleanup-once.json"
    if path == "invalid_event":
        with pytest.raises(json.JSONDecodeError):
            run_campaign(plan(model_timeout_seconds=1), output, worker_command=command)
    else:
        result = run_campaign(plan(model_timeout_seconds=1), output, worker_command=command)
        assert result["models"][0]["status"] == path
    assert len(cleanup_calls) == 1


def test_cleanup_permission_error_remains_visible_after_successful_worker(tmp_path, monkeypatch):
    command = fake_worker(tmp_path, "emit('succeeded')\n")
    original = benchmark.terminate_group
    cleanup_calls = []

    def forbidden_cleanup(process):
        cleanup_calls.append(process.pid)
        original(process)
        raise PermissionError("Simulated process-group ownership failure")

    monkeypatch.setattr(benchmark, "terminate_group", forbidden_cleanup)
    output = tmp_path / "gpu-capability-cleanup-forbidden.json"
    with pytest.raises(PermissionError, match="ownership failure"):
        run_campaign(plan(), output, worker_command=command)
    assert len(cleanup_calls) == 1
    document = json.loads(output.read_text())
    assert document["status"] == "cleanup_failed"
    assert document["models"][0]["cleanup_error_type"] == "PermissionError"


def test_cleanup_error_does_not_replace_original_supervisor_exception(tmp_path, monkeypatch):
    command = fake_worker(tmp_path, "print('invalid-json',flush=True)\ntime.sleep(30)\n")
    original = benchmark.terminate_group
    cleanup_calls = []

    def forbidden_cleanup(process):
        cleanup_calls.append(process.pid)
        original(process)
        raise PermissionError("Simulated process-group ownership failure")

    monkeypatch.setattr(benchmark, "terminate_group", forbidden_cleanup)
    output = tmp_path / "gpu-capability-original-failure.json"
    with pytest.raises(json.JSONDecodeError) as error:
        run_campaign(plan(), output, worker_command=command)
    assert isinstance(error.value.__cause__, PermissionError)
    assert len(cleanup_calls) == 1
    document = json.loads(output.read_text())
    assert document["status"] == "supervisor_failed"
    assert document["models"][0]["cleanup_error_type"] == "PermissionError"


def test_expired_absolute_deadline_starts_no_model_child(tmp_path):
    command = fake_worker(tmp_path, "raise RuntimeError('This must never run')\n")
    deadline = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    output = tmp_path / "gpu-capability-expired.json"
    result = run_campaign(plan(stop_at=deadline), output, worker_command=command)
    assert result["models"][0]["status"] == "skipped_deadline"
    assert not list(tmp_path.glob("*.stderr.log"))


def test_existing_evidence_is_preserved_instead_of_replayed(tmp_path):
    output = tmp_path / "gpu-capability-existing.json"
    output.write_text('{"verified": true}')
    original = copy.deepcopy(output.read_bytes())
    with pytest.raises(ValueError, match="fresh output"):
        run_campaign(plan(), output)
    assert output.read_bytes() == original


def test_module_cli_default_child_imports_stable_module_and_reaches_worker(tmp_path):
    # Block Torch imports even on a GPU-equipped test host; this regression checks
    # real module/subprocess launch without downloading a checkpoint or using GPU.
    (tmp_path / "torch.py").write_text(
        "raise ImportError('Accelerator imports blocked by CPU test')\n"
    )
    output = tmp_path / "gpu-capability-default-launch.json"
    source = Path(__file__).resolve().parents[2] / "src"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(source))),
        "MACFIT_GPU_LOCK": str(tmp_path / "gpu.lock"),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "macfit_training.benchmark",
            "--models",
            "qwen3-0-6b",
            "--batch-sizes",
            "1",
            "--prompt-tokens",
            "32",
            "--decode-steps",
            "1",
            "--repetitions",
            "2",
            "--walltime-seconds",
            "10",
            "--model-timeout-seconds",
            "5",
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    document = json.loads(output.read_text())
    assert document["status"] == "incomplete"
    stages = [event["stage"] for event in document["events"]]
    assert "waiting_for_gpu" in stages
    assert any(
        event["stage"] == "worker_failed" and event["error_type"] == "ImportError"
        for event in document["events"]
    )
    assert document["models"][0]["exit_code"] == 2


def test_maximum_legal_matrix_preserves_every_sample_and_exports_within_backup_bounds(tmp_path):
    scored = summarize_samples(
        [
            sample(
                prefill_seconds=1.123456789012345,
                decode_seconds=12.123456789012345,
                prefill_tokens_per_second=12345.123456789012,
                decode_tokens_per_second=1234.123456789012,
                peak_allocated_bytes=123456789012,
                peak_reserved_bytes=234567890123,
            )
            for _ in range(10)
        ]
    )
    template = {
        "decode_steps_per_sequence": 512,
        "scored_repetitions": 10,
        "warmup_repetitions_excluded": 5,
        **scored,
    }
    command = fake_worker(
        tmp_path,
        f"template=json.loads({json.dumps(template)!r})\n"
        "for prompt in config['plan']['prompt_tokens']:\n"
        " for batch in config['plan']['batch_sizes']:\n"
        "  case={**template,'prompt_tokens_per_sequence':prompt,'batch_size':batch}\n"
        "  emit('case_completed',result=case)\n"
        "emit('succeeded')\n",
    )
    data = tmp_path / "data"
    JobStore(Settings(data, "g" * 32))
    output = data / "evidence/gpu-capability-maximum.json"
    result = run_campaign(
        plan(
            models=list(BENCHMARK_MODELS),
            batch_sizes=list(range(1, 9)),
            prompt_tokens=[128 * i for i in range(1, 9)],
            decode_steps=512,
            repetitions=10,
            warmup_repetitions=5,
            walltime_seconds=180,
            model_timeout_seconds=30,
        ),
        output,
        worker_command=command,
    )
    assert result["status"] == "succeeded"
    assert result["schema_version"] == 2
    assert len(result["models"]) == 6
    saved_samples = 0
    for index in result["models"]:
        path = output.parent / index["evidence_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == index["evidence_sha256"]
        full = json.loads(path.read_text())
        assert len(full["cases"]) == len(index["cases"]) == 64
        assert all("samples" not in case for case in index["cases"])
        for coordinate, detailed in zip(index["cases"], full["cases"], strict=True):
            assert len(detailed["samples"]) == 10
            assert detailed["samples"] == template["samples"]
            assert coordinate == {key: value for key, value in detailed.items() if key != "samples"}
            saved_samples += len(detailed["samples"])
    assert saved_samples == 6 * 64 * 10
    assert all("result" not in event for event in result["events"])
    assert all(path.stat().st_size <= MAX_EVIDENCE_BYTES for path in output.parent.glob("*.json"))
    archive_bytes = io.BytesIO()
    export_archive(data, archive_bytes)
    archive_bytes.seek(0)
    with tarfile.open(fileobj=archive_bytes, mode="r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        backed_up = [entry for entry in manifest["files"] if entry["path"].startswith("evidence/")]
        assert len(backed_up) == 7
        assert all(entry["size_bytes"] < 1024**2 for entry in backed_up)
        for entry in backed_up:
            original = data / entry["path"]
            assert archive.extractfile(entry["path"]).read() == original.read_bytes()
