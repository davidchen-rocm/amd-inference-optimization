from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from macfit_training import attention_probe as probe
from macfit_training import benchmark
from macfit_training.service.backup import export_archive
from macfit_training.service.settings import Settings
from macfit_training.service.store import JobStore


def protocol(**changes):
    return {
        "models": ["qwen3-8b"],
        "batch_sizes": [1, 4],
        "prompt_tokens": [2048, 8192],
        "decode_steps": 64,
        "warmup_repetitions": 2,
        "repetitions": 5,
        "seed": 42,
        "walltime_seconds": 30,
        "model_timeout_seconds": 10,
        "stop_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        **changes,
    }


def sample():
    return {
        "prefill_seconds": 0.1,
        "decode_seconds": 0.2,
        "prefill_tokens_per_second": 20480.0,
        "decode_tokens_per_second": 320.0,
        "finite_logits": True,
        "output_ids_sha256": "a" * 64,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
    }


def event(name, device="DeviceType.CPU"):
    return SimpleNamespace(name=name, device_type=device)


def test_controlled_plan_is_pinned_and_has_no_http_registry_side_effect():
    for model in ("qwen3-8b", "qwen3-14b"):
        result = probe.validate_protocol(protocol(models=[model]))
        assert result["model_identities"][0] == benchmark.BENCHMARK_MODELS[model]
    from macfit_training.config import MODELS

    assert "qwen3-14b" not in MODELS


@pytest.mark.parametrize(
    "change",
    [
        {"models": ["qwen3-32b"]},
        {"models": ["qwen3-8b", "qwen3-14b"]},
        {"batch_sizes": [16]},
        {"prompt_tokens": [32768]},
        {"decode_steps": 128},
        {"warmup_repetitions": 1},
        {"repetitions": 3},
        {"seed": 43},
        {"stop_at": None},
        {"walltime_seconds": 3601},
    ],
)
def test_uncontrolled_protocol_coordinates_are_rejected(change):
    with pytest.raises(ValueError):
        probe.validate_protocol(protocol(**change))


def test_profile_records_actual_device_names_and_does_not_claim_library_identity():
    result = probe.summarize_profile(
        [
            event("aten::_scaled_dot_product_flash_attention"),
            event("aten::matmul"),
            event("ck-kernel", "DeviceType.CUDA"),
            event("ck-kernel", "DeviceType.CUDA"),
        ]
    )
    assert result["status"] == "captured"
    assert result["gpu_kernels"]["display_names"] == [{"name": "ck-kernel", "count": 2}]
    assert result["flash_operator_observed"] is True
    assert result["math_operator_observed"] is False
    assert result["timed_samples_include_profiling"] is False
    assert "backend_identity" not in result
    math = probe.summarize_profile([event("aten::_scaled_dot_product_attention_math")])
    assert math["status"] == "dispatch_unverified"


def test_profile_fingerprint_includes_names_outside_capped_display():
    names = [event(f"kernel-{i:03d}", "DeviceType.CUDA") for i in range(129)]
    before = probe.summarize_profile(names)
    names[-1] = event("z-hidden-different-kernel", "DeviceType.CUDA")
    after = probe.summarize_profile(names)
    assert before["gpu_kernels"]["display_names"] == after["gpu_kernels"]["display_names"]
    assert (
        before["gpu_kernels"]["full_name_set_sha256"]
        != after["gpu_kernels"]["full_name_set_sha256"]
    )
    assert before["gpu_kernels"]["display_truncated"] is True
    with pytest.raises(ValueError, match="event bound"):
        probe.summarize_profile([event("x")] * (probe.MAX_PROFILE_EVENTS + 1))


def fake_torch(monkeypatch):
    accelerator = ModuleType("torch")
    accelerator.cuda = SimpleNamespace(is_available=lambda: True)
    accelerator.version = SimpleNamespace(hip="fake-runtime")
    preference_calls = []
    current = "aotriton"

    def preference(value=None):
        nonlocal current
        preference_calls.append(value)
        if value is not None:
            current = value
        return SimpleNamespace(name=current)

    accelerator.backends = SimpleNamespace(
        cuda=SimpleNamespace(preferred_rocm_fa_library=preference)
    )
    attention = ModuleType("torch.nn.attention")
    contexts = []

    @contextmanager
    def force(backends):
        contexts.append(backends)
        yield

    attention.SDPBackend = SimpleNamespace(FLASH_ATTENTION="flash-only")
    attention.sdpa_kernel = force
    profiler_calls = []

    @contextmanager
    def profile(**options):
        profiler_calls.append(options)
        yield SimpleNamespace(
            events=lambda: [
                event("aten::_scaled_dot_product_flash_attention"),
                event("fake-kernel", "DeviceType.CUDA"),
            ]
        )

    accelerator.profiler = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
        supported_activities=lambda: {"cpu", "cuda"},
        profile=profile,
    )
    monkeypatch.setitem(sys.modules, "torch", accelerator)
    monkeypatch.setitem(sys.modules, "torch.nn", ModuleType("torch.nn"))
    monkeypatch.setitem(sys.modules, "torch.nn.attention", attention)
    return accelerator, preference_calls, contexts, profiler_calls


def test_worker_forces_flash_and_profiles_before_both_warmups_without_gpu(
    tmp_path, monkeypatch, capsys
):
    accelerator, preference_calls, contexts, profiler_calls = fake_torch(monkeypatch)
    monkeypatch.setenv("MACFIT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    measured_steps = []

    def measured(_model, _ids, _batch, steps, _accelerator):
        measured_steps.append(steps)
        return sample()

    def loaded(_identity, plan, emit):
        emit("runtime_ready", runtime={"test_fixture": True})
        for _ in range(plan["warmup_repetitions"]):
            benchmark._measure_sample(None, [1] * 2048, 1, 64, accelerator)
        scored = [benchmark._measure_sample(None, [1] * 2048, 1, 64, accelerator) for _ in range(5)]
        emit("case_completed", result=benchmark.summarize_samples(scored))
        emit("succeeded")
        return 0

    monkeypatch.setattr(benchmark, "_measure_sample", measured)
    monkeypatch.setattr(benchmark, "_benchmark_loaded_model", loaded)
    plan = probe.validate_protocol(protocol(batch_sizes=[1], prompt_tokens=[2048]))
    assert probe._worker({"plan": plan, "identity": plan["model_identities"][0]}, "aotriton") == 0
    assert preference_calls == ["aotriton", None]
    assert contexts == [["flash-only"]]
    assert measured_steps == [1, *([64] * 7)]
    assert len(profiler_calls) == 1 and profiler_calls[0]["with_stack"] is False
    assert benchmark._measure_sample is measured
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    runtime = [event["runtime"] for event in events if event["stage"] == "runtime_ready"][-1]
    assert runtime["preference_requested"] == runtime["preference_returned"] == "aotriton"
    assert runtime["profile"]["decode_steps"] == 1


def test_native_kernel_failure_is_unsupported_and_measure_hook_is_restored(
    tmp_path, monkeypatch, capsys
):
    fake_torch(monkeypatch)
    monkeypatch.setenv("MACFIT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    original = benchmark._measure_sample

    def unavailable(*_args):
        raise RuntimeError("CK has not been compiled into this test runtime")

    monkeypatch.setattr(benchmark, "_benchmark_loaded_model", unavailable)
    plan = probe.validate_protocol(protocol())
    assert probe._worker({"plan": plan, "identity": plan["model_identities"][0]}, "ck") == 2
    assert benchmark._measure_sample is original
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["code"] == "attention_unsupported"


def test_profiler_limitation_is_explicit_but_actual_forward_errors_propagate(monkeypatch):
    accelerator, *_ = fake_torch(monkeypatch)
    accelerator.profiler.supported_activities = lambda: {"cpu"}
    assert probe.capture_profile(accelerator, lambda: None) == {
        "status": "unsupported",
        "reason": "gpu_profiler_unavailable",
    }
    accelerator.profiler.supported_activities = lambda: {"cpu", "cuda"}

    def failed_forward():
        raise RuntimeError("No available kernel")

    with pytest.raises(RuntimeError, match="kernel"):
        probe.capture_profile(accelerator, failed_forward)


def fake_workers(tmp_path):
    template = {
        "decode_steps_per_sequence": 64,
        "warmup_repetitions_excluded": 2,
        "scored_repetitions": 5,
        **benchmark.summarize_samples([sample()] * 5),
    }
    commands = {}
    for arm in probe.ARMS:
        script = tmp_path / f"fixture-{arm}.py"
        profile = probe.summarize_profile(
            [
                event("aten::_scaled_dot_product_flash_attention"),
                event(arm + "-kernel", "DeviceType.CUDA"),
            ]
        )
        runtime = {"preference_requested": arm, "preference_returned": arm, "profile": profile}
        script.write_text(
            "import json,sys\nconfig=json.load(open(sys.argv[1]))\n"
            "model=config['identity']['id']\n"
            "def emit(stage,**values):\n"
            " print(json.dumps({'model_id':model,'stage':stage,**values}),flush=True)\n"
            f"emit('runtime_ready',runtime=json.loads({json.dumps(runtime)!r}))\n"
            f"template=json.loads({json.dumps(template)!r})\n"
            "for prompt in config['plan']['prompt_tokens']:\n"
            " for batch in config['plan']['batch_sizes']:\n"
            "  case={**template,'prompt_tokens_per_sequence':prompt,'batch_size':batch}\n"
            "  emit('case_completed',result=case)\n"
            "emit('succeeded')\n"
        )
        commands[arm] = [sys.executable, str(script)]
    return commands


def test_two_cpu_fixture_arms_preserve_samples_and_backup_all_bounded_evidence(tmp_path):
    commands = fake_workers(tmp_path)
    data = tmp_path / "data"
    JobStore(Settings(data, "g" * 32))
    output = data / "evidence/gpu-capability-attention.json"
    result = probe.run_probe(protocol(), output, worker_commands=commands)
    assert result["status"] == "succeeded"
    assert result["comparison"]["label"] == "preference_arms"
    assert result["comparison"]["backend_difference_established"] is False
    assert result["comparison"]["kernel_name_fingerprints_differ"] is True
    assert len(result["comparison"]["cases"]) == 4
    assert all(case["greedy_token_hash_sets_equal"] for case in result["comparison"]["cases"])
    for arm in result["arms"]:
        path = output.parent / arm["evidence_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == arm["evidence_sha256"]
        assert sum(len(case["samples"]) for case in arm["model"]["cases"]) == 20
    assert len(list(output.parent.glob("*.json"))) == 5
    assert all(path.stat().st_size < 1024**2 for path in output.parent.glob("*.json"))
    archive = io.BytesIO()
    export_archive(data, archive)
    archive.seek(0)
    with tarfile.open(fileobj=archive, mode="r:gz") as reader:
        manifest = json.load(reader.extractfile("manifest.json"))
        assert (
            len([item for item in manifest["files"] if item["path"].startswith("evidence/")]) == 5
        )


def test_expired_absolute_deadline_runs_neither_arm(tmp_path):
    result = probe.run_probe(
        protocol(stop_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat()),
        tmp_path / "expired-attention.json",
        worker_commands={arm: ["must-not-run"] for arm in probe.ARMS},
    )
    assert result["status"] == "incomplete"
    assert [arm["status"] for arm in result["arms"]] == ["skipped_deadline", "skipped_deadline"]


def test_global_budget_and_model_deadline_bound_actual_cpu_child(tmp_path):
    script = tmp_path / "blocked.py"
    script.write_text("import time\ntime.sleep(30)\n")
    started = time.monotonic()
    result = probe.run_probe(
        protocol(walltime_seconds=2, model_timeout_seconds=1),
        tmp_path / "timeout-attention.json",
        worker_commands={arm: [sys.executable, str(script)] for arm in probe.ARMS},
    )
    assert time.monotonic() - started < 5
    assert result["status"] == "incomplete"
    assert all(arm["model"]["status"] == "timed_out" for arm in result["arms"])


def test_evidence_slot_budget_refuses_before_creating_any_probe_files(tmp_path):
    for index in range(probe.MAX_BACKUP_EVIDENCE_FILES - 5):
        (tmp_path / f"old-{index}.json").write_text("{}")
    output = tmp_path / "would-overflow.json"
    with pytest.raises(ValueError, match="evidence-file bound"):
        probe.run_probe(protocol(), output)
    assert not output.exists()


def test_lazy_import_and_default_child_module_launch_never_use_gpu(tmp_path):
    source = Path(__file__).resolve().parents[2] / "src"
    (tmp_path / "torch.py").write_text(
        "raise ImportError('Accelerator imports blocked by CPU regression')\n"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(source))),
        "MACFIT_GPU_LOCK": str(tmp_path / "gpu.lock"),
    }
    lazy = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import macfit_training.attention_probe; assert 'torch' not in sys.modules",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert lazy.returncode == 0, lazy.stderr
    output = tmp_path / "default-attention.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "macfit_training.attention_probe",
            "--batch-sizes",
            "1",
            "--prompt-tokens",
            "2048",
            "--walltime-seconds",
            "10",
            "--model-timeout-seconds",
            "5",
            "--stop-at",
            protocol()["stop_at"],
            "--output",
            str(output),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 2, result.stderr
    state = json.loads(output.read_text())
    assert state["status"] == "unsupported"
    assert [arm["status"] for arm in state["arms"]] == ["unsupported", "unsupported"]


def test_comparison_rejects_different_coordinates_and_model_evidence_tampering(tmp_path):
    commands = fake_workers(tmp_path)
    state = probe.run_probe(
        protocol(), tmp_path / "compare-attention.json", worker_commands=commands
    )
    arms = copy.deepcopy(state["arms"])
    arms[1]["model"]["cases"].pop()
    with pytest.raises(ValueError, match="different coordinates"):
        probe.comparison(arms)
    path = tmp_path / "model.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="byte hash"):
        probe.read_verified_model(
            tmp_path, {"evidence_file": path.name, "evidence_sha256": "a" * 64}
        )
