"""CPU artifact integrity and real subprocess lifecycle checks for LoRA regression."""

import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from amd_inference_opt.vllm_model_snapshot import (
    VLLMModelSnapshotError,
    capture_vllm_model_snapshot,
)
from macfit_training import benchmark, regression
from macfit_training.artifacts import finish_artifacts, write_json
from macfit_training.config import validate_job_input
from macfit_training.experiments import build_input


@pytest.fixture
def completed_job(tmp_path):
    job_dir, model_dir = tmp_path / "job", tmp_path / "base"
    job_dir.mkdir()
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"not real model weights")
    job = validate_job_input("training", build_input())
    base = job["base_model"]
    snapshot = capture_vllm_model_snapshot(
        model_dir,
        model_id=base["repo_id"],
        revision=base["revision"],
        tokenizer_revision=base["revision"],
    )
    write_json(job_dir / "input.json", job)
    write_json(job_dir / "model-snapshot.json", snapshot.model_dump(mode="json"))
    adapter = job_dir / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter fixture")
    write_json(
        adapter / "adapter_config.json",
        {
            "base_model_name_or_path": base["repo_id"],
            "revision": base["revision"],
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
        },
    )
    result = {
        "method": "lora_sft",
        "base_model": base,
        "evaluation": {},
        "provenance": {"model_snapshot_sha256": snapshot.snapshot_digest},
    }
    result["artifacts"] = finish_artifacts(job_dir, job, result, adapter_dir=adapter)
    write_json(job_dir / "result.json", result)
    return job_dir, model_dir


def perfect_samples():
    return [
        {
            **case,
            **{
                stage: {"text": case["expected"], "stopped_by_limit": False, "passed": True}
                for stage in ("base", "adapter")
            },
        }
        for case in regression.CASES
    ]


def test_generic_protocol_is_balanced_and_independent_from_policy_task():
    assert len(regression.CASES) == 10
    assert len({case["id"] for case in regression.CASES}) == 10
    assert Counter(case["language"] for case in regression.CASES) == {"en": 5, "zh": 5}
    assert "Juniper" not in regression.SYSTEM
    assert "Juniper" not in str(regression.CASES)
    assert regression.protocol()["decoding"] == {
        "mode": "greedy",
        "enable_thinking": False,
        "max_new_tokens": 64,
    }
    helper_file = Path(regression.parse_strict_json.__code__.co_filename)
    assert (
        regression.protocol()["strict_scoring_helpers_source_sha256"]
        == hashlib.sha256(helper_file.read_bytes()).hexdigest()
    )


def test_regression_helper_source_is_bound_separately(tmp_path, monkeypatch):
    helper_source = tmp_path / "strict_helpers.py"
    helper_source.write_text("first helper source\n")
    original = regression.parse_strict_json

    def helper(*args, **kwargs):
        return original(*args, **kwargs)

    helper.__code__ = helper.__code__.replace(co_filename=str(helper_source))
    monkeypatch.setattr(regression, "parse_strict_json", helper)
    first = regression.protocol()
    helper_source.write_text("second helper source\n")
    second = regression.protocol()
    assert first["source_sha256"] == second["source_sha256"]
    assert (
        first["strict_scoring_helpers_source_sha256"]
        != second["strict_scoring_helpers_source_sha256"]
    )


def test_exact_text_normalizer_accepts_nfkc_but_not_explanations():
    case = regression.CASES[0]
    assert regression.answer_passes(case, " ４２\n")
    assert not regression.answer_passes(case, "The answer is 42.")
    assert not regression.answer_passes(case, "```\n42\n```")
    assert not regression.answer_passes(case, 42)


def test_json_equality_requires_exact_keys_and_types_but_not_key_order():
    case = next(case for case in regression.CASES if case["id"] == "json-add-en")
    assert regression.answer_passes(case, '{ "is_even":false, "total":7 }')
    for value in (
        '{"total":7.0,"is_even":false}',
        '{"total":7,"is_even":0}',
        '{"total":7,"is_even":false,"explanation":"seven"}',
        '{"total":7,"total":8,"is_even":false}',
        '```json\n{"total":7,"is_even":false}\n```',
    ):
        assert not regression.answer_passes(case, value)


def test_summary_reports_individual_regressions_and_improvements():
    samples = perfect_samples()
    samples[0]["base"]["passed"] = False
    samples[1]["adapter"]["passed"] = False
    result = regression.summarize(samples)
    assert result["base_pass_fraction"] == result["adapter_pass_fraction"] == 0.9
    assert result["regressed_case_ids"] == ["multiply-zh"]
    assert result["improved_case_ids"] == ["add-en"]
    assert result["pass_fraction_change"] == 0.0
    with pytest.raises(ValueError):
        regression.summarize(samples[:-1])


@pytest.mark.parametrize("mutation", ["flag", "question", "metrics", "missing"])
def test_parent_rejects_unsubstantiated_child_success(mutation):
    samples = perfect_samples()
    result = {"samples": samples, "metrics": regression.summarize(samples)}
    if mutation == "flag":
        samples[0]["adapter"]["text"] = "incorrect"
    elif mutation == "question":
        samples[0]["question"] = "changed question"
    elif mutation == "metrics":
        result["metrics"]["adapter_pass_fraction"] = 0.0
    else:
        result.pop("metrics")
    with pytest.raises(ValueError):
        regression.validate_completed_child(result)


def test_original_artifacts_and_snapshot_are_verified(completed_job):
    job_dir, model_dir = completed_job
    job, result, snapshot, base = regression.verified_model(job_dir)
    assert base == model_dir
    assert snapshot.snapshot_digest == result["provenance"]["model_snapshot_sha256"]
    assert job["model_id"] == "qwen3-0-6b"


@pytest.mark.parametrize("target", ["base", "adapter", "provenance"])
def test_tampering_is_rejected_before_loading_any_model(completed_job, target):
    job_dir, model_dir = completed_job
    if target == "base":
        (model_dir / "model.safetensors").write_bytes(b"changed")
    elif target == "adapter":
        (job_dir / "artifacts/adapter.tar.gz").write_bytes(b"changed")
    else:
        result = json.loads((job_dir / "result.json").read_text())
        result["provenance"]["model_snapshot_sha256"] = "0" * 64
        write_json(job_dir / "result.json", result)
    with pytest.raises((ValueError, VLLMModelSnapshotError)):
        regression.verified_model(job_dir)


def test_loader_uses_local_safetensors_on_rocm_gpu_only(tmp_path, monkeypatch):
    calls = {}

    class Model:
        @classmethod
        def from_pretrained(cls, path, **options):
            calls["model"] = options
            return cls()

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            return self

        def parameters(self):
            return [SimpleNamespace(device=SimpleNamespace(type="cuda"))]

    class Tokenizer:
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, path, **options):
            calls["tokenizer"] = options
            return cls()

    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        manual_seed_all=lambda _seed: None,
        get_device_name=lambda _index: "test GPU",
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=cuda,
            version=SimpleNamespace(hip="test"),
            bfloat16="bf16",
            manual_seed=lambda _seed: None,
            __version__="test",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=Model,
            AutoTokenizer=Tokenizer,
        ),
    )
    monkeypatch.setattr(regression.importlib.metadata, "version", lambda _name: "test")
    _, _, runtime = regression.load_base(tmp_path)
    assert calls["device"] == "cuda:0"
    assert calls["model"]["local_files_only"] is True
    assert calls["model"]["trust_remote_code"] is False
    assert calls["model"]["use_safetensors"] is True
    assert calls["tokenizer"]["local_files_only"] is True
    assert runtime["full_gpu_weights"] is True
    cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="ROCm"):
        regression.load_base(tmp_path)


def test_child_uses_hash_bound_adapter_and_generic_instructions(
    completed_job, tmp_path, monkeypatch
):
    job_dir, _ = completed_job
    (job_dir / "adapter/adapter_model.safetensors").write_bytes(b"untrusted mutable adapter")
    output = tmp_path / "child.json"
    monkeypatch.setenv("MACFIT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setattr(
        regression,
        "load_base",
        lambda _path: (
            SimpleNamespace(stage="base"),
            object(),
            {"test": True},
        ),
    )

    class Adapter:
        @classmethod
        def from_pretrained(cls, model, path, **options):
            assert (path / "adapter_model.safetensors").read_bytes() == b"adapter fixture"
            assert options["is_trainable"] is False
            assert options["local_files_only"] is True
            assert options["torch_device"] == "cuda:0"
            assert options["device_map"] == {"": "cuda:0"}
            return SimpleNamespace(stage="adapter", eval=lambda: SimpleNamespace(stage="adapter"))

    expected = {case["question"]: case["expected"] for case in regression.CASES}

    def generate(model, tokenizer, messages, **options):
        assert messages[0] == {"role": "system", "content": regression.SYSTEM}
        assert options == {"max_input_tokens": 1024, "max_new_tokens": 64}
        question = messages[1]["content"]
        assert question in expected
        return {
            "text": expected[question],
            "stopped_by_limit": False,
            "input_tokens": 30,
            "output_tokens": 1,
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=Adapter))
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
    )
    monkeypatch.setattr("macfit_training.evaluation.generate_text", generate)
    assert regression.run_child({"job": str(job_dir), "child_output": str(output)}) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "succeeded"
    assert (
        result["metrics"]["base_pass_fraction"] == result["metrics"]["adapter_pass_fraction"] == 1.0
    )
    regression.validate_completed_child(result)
    assert output.stat().st_size < regression.MAX_CHILD_BYTES


def test_parent_default_command_imports_stable_module_and_publishes_actual_outputs(
    tmp_path, monkeypatch
):
    job = tmp_path / "completed"
    job.mkdir()
    script = tmp_path / "child.py"
    result = {
        "status": "succeeded",
        "samples": perfect_samples(),
        "metrics": regression.summarize(perfect_samples()),
    }
    script.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "config=json.loads(Path(sys.argv[1]).read_text())\n"
        f"Path(config['child_output']).write_text({json.dumps(json.dumps(result))})\n"
    )
    observed = []
    original = subprocess.Popen

    def popen(command, **options):
        observed.append(command)
        return original([sys.executable, str(script), command[-1]], **options)

    monkeypatch.setattr(regression.subprocess, "Popen", popen)
    monkeypatch.setattr(regression, "__name__", "__main__")
    output = tmp_path / "evidence.json"
    result = regression.run_regressions([job], output, walltime_seconds=20)
    assert observed[0][1:4] == ["-m", "macfit_training.regression", "--child-config"]
    assert result["status"] == "succeeded"
    assert result["jobs"][0]["metrics"]["adapter_pass_fraction"] == 1.0
    assert json.loads(output.read_text())["status"] == "succeeded"
    assert output.stat().st_size < regression.MAX_EVIDENCE_BYTES
    with pytest.raises(ValueError, match="fresh output"):
        regression.run_regressions([job], output)


def test_expired_stop_time_skips_every_child(tmp_path, monkeypatch):
    job = tmp_path / "completed"
    job.mkdir()
    monkeypatch.setattr(
        regression.subprocess, "Popen", lambda *_a, **_k: pytest.fail("No child allowed")
    )
    result = regression.run_regressions(
        [job], tmp_path / "expired.json", stop_at="2000-01-01T00:00:00Z"
    )
    assert result["status"] == "incomplete"
    assert result["jobs"][0]["status"] == "skipped_deadline"


def test_timeout_kills_process_group_once_and_preserves_partial_output(tmp_path, monkeypatch):
    job = tmp_path / "completed"
    job.mkdir()
    script = tmp_path / "hang.py"
    script.write_text(
        "import json,sys,time,signal,os,subprocess\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "config=json.loads(Path(sys.argv[1]).read_text())\n"
        "Path(config['child_output']).write_text(json.dumps({'status':'loading_base','samples':[]}))\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        f"Path({str(tmp_path / 'pid.json')!r}).write_text(json.dumps([os.getpid(),child.pid]))\n"
        "time.sleep(60)\n"
    )
    cleanup_calls = []
    original = benchmark.terminate_group

    def cleanup(process):
        cleanup_calls.append(process.pid)
        original(process)

    monkeypatch.setattr(benchmark, "terminate_group", cleanup)
    started = time.monotonic()
    result = regression.run_regressions(
        [job],
        tmp_path / "timeout.json",
        walltime_seconds=20,
        job_timeout_seconds=1,
        worker_command=[sys.executable, str(script)],
    )
    assert time.monotonic() - started < 10
    assert result["jobs"][0]["status"] == "timed_out"
    assert result["jobs"][0]["samples"] == []
    assert len(cleanup_calls) == 1
    leader, child = json.loads((tmp_path / "pid.json").read_text())
    # A terminated orphan may briefly remain a zombie; it must not execute further.
    for pid in (leader, child):
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        )
        assert not result.stdout.strip() or result.stdout.strip().startswith("Z")


@pytest.mark.parametrize("field,bad", [("walltime_seconds", 9), ("job_timeout_seconds", 3601)])
def test_plan_limits_are_cpu_validated(tmp_path, field, bad):
    job = tmp_path / "completed"
    job.mkdir()
    values = {"walltime_seconds": 1800, "job_timeout_seconds": 600, "stop_at": None}
    values[field] = bad
    with pytest.raises(ValueError):
        regression.validate_plan([job], **values)
    with pytest.raises(ValueError, match="timezone"):
        regression.validate_plan([job], 1800, 600, "2026-10-01T03:45:00")


def test_child_size_limit_rejects_oversized_evidence(tmp_path):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (regression.MAX_CHILD_BYTES + 1))
    with pytest.raises(ValueError, match="size bound"):
        regression.read_child(path)


def test_exited_leader_descendants_are_reaped_before_success(tmp_path, monkeypatch):
    job = tmp_path / "completed"
    job.mkdir()
    script, pid_file = tmp_path / "finished-leader.py", tmp_path / "child-pid.json"
    child_result = {
        "status": "succeeded",
        "samples": perfect_samples(),
        "metrics": regression.summarize(perfect_samples()),
    }
    script.write_text(
        "import json,sys,subprocess\nfrom pathlib import Path\n"
        "config=json.loads(Path(sys.argv[1]).read_text())\n"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])\n"
        f"Path({str(pid_file)!r}).write_text(json.dumps(child.pid))\n"
        f"Path(config['child_output']).write_text({json.dumps(json.dumps(child_result))})\n"
    )
    calls = []
    original = benchmark.terminate_group

    def cleanup(process):
        calls.append((process.pid, process.poll()))
        original(process)

    monkeypatch.setattr(benchmark, "terminate_group", cleanup)
    result = regression.run_regressions(
        [job],
        tmp_path / "evidence.json",
        walltime_seconds=20,
        worker_command=[sys.executable, str(script)],
    )
    assert result["status"] == "succeeded"
    assert len(calls) == 1 and calls[0][1] == 0
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(json.loads(pid_file.read_text()))],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    assert not state or state.startswith("Z")


def test_cleanup_failure_chains_under_original_error_and_never_retries(tmp_path, monkeypatch):
    job = tmp_path / "completed"
    job.mkdir()
    script = tmp_path / "oversized-worker.py"
    script.write_text(
        "import json,sys,time\nfrom pathlib import Path\n"
        "config=json.loads(Path(sys.argv[1]).read_text())\n"
        "Path(config['child_output']).write_text(' '*(64*1024+1))\n"
        "time.sleep(2)\n"
    )
    calls = []
    original = benchmark.terminate_group

    def cleanup(process):
        calls.append(process.pid)
        original(process)  # Reap the real CPU process before simulating an OS error.
        raise PermissionError("private diagnostic must not be copied into evidence")

    monkeypatch.setattr(benchmark, "terminate_group", cleanup)
    evidence_path = tmp_path / "evidence.json"
    with pytest.raises(ValueError, match="size bound") as failure:
        regression.run_regressions(
            [job],
            evidence_path,
            walltime_seconds=20,
            worker_command=[sys.executable, str(script)],
        )
    assert isinstance(failure.value.__cause__, PermissionError)
    assert len(calls) == 1
    evidence = json.loads(evidence_path.read_text())
    assert evidence["status"] == "cleanup_failed"
    assert evidence["error_type"] == "ValueError"
    assert evidence["jobs"][0]["cleanup_error_type"] == "PermissionError"
    assert evidence["jobs"][0]["error_type"] == "ValueError"
    assert "private diagnostic" not in evidence_path.read_text()


def test_cleanup_failure_on_completed_leader_cannot_claim_success(tmp_path, monkeypatch):
    job = tmp_path / "completed"
    job.mkdir()
    script = tmp_path / "completed-worker.py"
    child_result = {
        "status": "succeeded",
        "samples": perfect_samples(),
        "metrics": regression.summarize(perfect_samples()),
    }
    script.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "config=json.loads(Path(sys.argv[1]).read_text())\n"
        f"Path(config['child_output']).write_text({json.dumps(json.dumps(child_result))})\n"
    )
    calls = []
    original = benchmark.terminate_group

    def cleanup(process):
        calls.append(process.pid)
        original(process)
        raise PermissionError("private process details")

    monkeypatch.setattr(benchmark, "terminate_group", cleanup)
    evidence_path = tmp_path / "evidence.json"
    with pytest.raises(PermissionError):
        regression.run_regressions(
            [job],
            evidence_path,
            walltime_seconds=20,
            worker_command=[sys.executable, str(script)],
        )
    assert len(calls) == 1
    evidence = json.loads(evidence_path.read_text())
    assert evidence["status"] == "cleanup_failed"
    assert evidence["jobs"][0]["status"] == "cleanup_failed"
    assert "private process details" not in evidence_path.read_text()
