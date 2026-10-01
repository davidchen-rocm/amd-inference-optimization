"""CPU protocol and real process-lifecycle tests; fixture bytes are not models."""

import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from amd_inference_opt.vllm_model_snapshot import capture_vllm_model_snapshot
from macfit_training import benchmark
from macfit_training import export_verification as verification
from macfit_training.artifacts import canonical_sha256, finish_artifacts, write_json
from macfit_training.config import validate_job_input
from macfit_training.data import evaluation_messages
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


def plan_for(completed_job, tmp_path):
    job_dir, model_dir = completed_job
    return verification.validate_plan(
        job_dir,
        tmp_path / "merged",
        tmp_path / "evidence.json",
        base_model_dir=model_dir,
        question_ids=[
            "eval-juniper-1",
            "eval-juniper-2",
            "eval-juniper-3",
            "eval-juniper-4",
            "eval-cedar-1",
            "eval-cedar-3",
        ],
    )


def prediction(text, *, ids=None):
    ids = ids or [21, 22]
    result = {
        "text": text,
        "input_tokens": 50,
        "input_ids_sha256": canonical_sha256([1, 2, 3]),
        "output_tokens": len(ids),
        "output_ids": ids,
        "output_ids_sha256": canonical_sha256(ids),
        "elapsed_seconds": 0.01,
        "finite_logits": True,
        "finite_checks": "prompt_last_token_and_generated_sequence_last_token",
        "stopped_by_limit": False,
    }
    result["score"] = {**verification.expected_score(text, text), "passed": True}
    return result


def fixture_comparison(plan, merged_dir):
    merged_dir.mkdir()
    (merged_dir / "model.safetensors").write_bytes(b"merged fixture")
    (merged_dir / "tokenizer.json").write_text('{"fake":true}\n')
    from macfit_training.artifacts import describe_artifact

    manifest = {
        "schema": "macfit-merged-export.v1",
        "dtype": "bfloat16",
        "device": "cpu",
        "format": "transformers_safetensors",
        **{key: plan[key] for key in verification.SOURCE_KEYS[:4]},
        "files": [
            describe_artifact(path, "merged_model_file") for path in sorted(merged_dir.iterdir())
        ],
    }
    write_json(merged_dir / "export-manifest.json", manifest)
    samples = [
        {
            **{key: row[key] for key in ("id", "question", "expected")},
            "peft_cpu": prediction(row["expected"]),
            "merged_cpu": prediction(row["expected"]),
        }
        for row in plan["selected_questions"]
    ]
    return {
        "status": "succeeded",
        **{key: plan[key] for key in verification.SOURCE_KEYS},
        "selected_questions_sha256": plan["selected_questions_sha256"],
        "selected_question_ids": plan["question_ids"],
        "samples": samples,
        "metrics": verification.compare_samples(samples),
        "export_manifest": manifest,
        "export_manifest_file": verification.verify_export(merged_dir, manifest),
        "parameter_dtypes": {
            "peft_cpu": {"torch.bfloat16": 7, "torch.float32": 2},
            "merged_cpu": {"torch.bfloat16": 7},
        },
        "runtime": {
            "device": "cpu",
            "cpu_threads": 4,
            "cpu_interop_threads": 1,
            "local_files_only": True,
            "trust_remote_code": False,
            "gpu_visibility": {
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
            },
        },
    }


@pytest.mark.parametrize(
    "expected,actual,passed",
    [
        ('{"a":1,"b":[true,null]}', '{"b":[true,null],"a":1}', True),
        ('{"a":1}', '{"a":true}', False),
        ('{"a":1}', '{"a":1,"a":1}', False),
        ('{"a":1}', '```json\n{"a":1}\n```', False),
        ('{"a":1}', '{"a":1,"extra":0}', False),
        ("[1,2]", "[1,2]", True),
        ("null", "null", True),
        ("42", "42", True),
        ('"hello"', '"hello"', True),
        ("true", "1", False),
        ("hello world", "  hello\nworld ", True),
        ("４２ apples", "42 apples", True),
        ("Hello", "hello", False),
        ("hello", "hello explanation", False),
    ],
)
def test_generic_expected_scoring_preserves_types_and_strict_structure(expected, actual, passed):
    assert verification.expected_score(expected, actual)["exact_expected"] is passed


def test_selection_is_original_disjoint_and_not_hardcoded_to_policy_fixture():
    job = validate_job_input("training", build_input())
    first = verification.select_questions(job)
    assert first == verification.select_questions(job)
    assert len({row["id"] for row in first}) == 6
    assert all(row in job["evaluation"] for row in first)
    generic = {
        "evaluation": [
            {"id": f"user-{index}", "question": f"Question {index}", "expected": str(index)}
            for index in range(8)
        ]
    }
    ids = [f"user-{index}" for index in (7, 3, 6, 2, 5, 1)]
    assert [row["id"] for row in verification.select_questions(generic, ids)] == ids
    for invalid in (ids[:-1], ids[:-1] + [ids[0]], ids[:-1] + ["missing"]):
        with pytest.raises(ValueError):
            verification.select_questions(generic, invalid)
    with pytest.raises(ValueError, match="at least six"):
        verification.select_questions({"evaluation": generic["evaluation"][:5]})


def test_plan_binds_declared_original_artifact_and_model_metadata(completed_job, tmp_path):
    plan = plan_for(completed_job, tmp_path)
    job_dir, _ = completed_job
    job, identity = verification.source_identity(job_dir, verify_bytes=True)
    assert {key: plan[key] for key in verification.SOURCE_KEYS} == identity
    assert plan["selected_questions_sha256"] == canonical_sha256(
        verification.select_questions(job, plan["question_ids"])
    )
    result = json.loads((job_dir / "result.json").read_text())
    result["provenance"]["model_snapshot_sha256"] = "0" * 64
    write_json(job_dir / "result.json", result)
    with pytest.raises(ValueError, match="provenance"):
        plan_for(completed_job, tmp_path)


@pytest.mark.parametrize("walltime", [9, 601, True, 600.0])
def test_hard_walltime_bounds_are_validated(completed_job, tmp_path, walltime):
    with pytest.raises(ValueError, match="10 to 600"):
        verification.validate_plan(
            completed_job[0], tmp_path / "merged", tmp_path / "evidence", walltime_seconds=walltime
        )


def test_private_fresh_separate_paths_are_required(completed_job, tmp_path):
    job_dir, _ = completed_job
    for merged, evidence in (
        (job_dir / "merged", tmp_path / "evidence"),
        (tmp_path / "merged", job_dir / "evidence"),
        (tmp_path / "merged", tmp_path / "merged/evidence"),
    ):
        with pytest.raises(ValueError):
            verification.validate_plan(job_dir, merged, evidence)
    (tmp_path / "merged").mkdir()
    with pytest.raises(ValueError, match="fresh"):
        plan_for(completed_job, tmp_path)


@pytest.mark.parametrize("target", ["weight", "extra", "link", "marker", "manifest"])
def test_saved_export_inventory_rejects_mutation(completed_job, tmp_path, target):
    plan = plan_for(completed_job, tmp_path)
    merged = tmp_path / "merged"
    result = fixture_comparison(plan, merged)
    if target == "weight":
        (merged / "model.safetensors").write_bytes(b"changed")
    elif target == "extra":
        (merged / "unexpected.txt").write_text("unexpected")
    elif target == "link":
        (merged / "model.safetensors").unlink()
        (merged / "model.safetensors").symlink_to(completed_job[1] / "model.safetensors")
    elif target == "marker":
        write_json(merged / "EXPORT_INCOMPLETE.json", {"complete": False})
    else:
        write_json(merged / "export-manifest.json", {"schema": "changed"})
    with pytest.raises(ValueError):
        verification.verify_export(merged, result["export_manifest"])


@pytest.mark.parametrize("field", verification.SOURCE_KEYS)
def test_success_must_bind_every_source_identity(completed_job, tmp_path, field):
    plan = plan_for(completed_job, tmp_path)
    result = fixture_comparison(plan, tmp_path / "merged")
    verification.validate_completed(result, plan)
    result[field] = "different"
    with pytest.raises(ValueError):
        verification.validate_completed(result, plan)


@pytest.mark.parametrize(
    "target",
    [
        "manifest_source",
        "question",
        "token_hash",
        "score",
        "finite",
        "device",
        "threads",
        "dtype_inventory",
    ],
)
def test_success_cannot_claim_stale_outputs_or_cpu_placement(completed_job, tmp_path, target):
    plan = plan_for(completed_job, tmp_path)
    result = fixture_comparison(plan, tmp_path / "merged")
    if target == "manifest_source":
        result["export_manifest"]["source_adapter_sha256"] = "0" * 64
    elif target == "question":
        result["samples"][0]["question"] = "not the source question"
    elif target in {"token_hash", "score", "finite"}:
        predicted = result["samples"][0]["merged_cpu"]
        if target == "token_hash":
            predicted["output_ids_sha256"] = "0" * 64
        elif target == "score":
            predicted["score"]["passed"] = False
        else:
            predicted["finite_logits"] = False
    elif target == "device":
        result["runtime"]["gpu_visibility"]["HIP_VISIBLE_DEVICES"] = "0"
    elif target == "threads":
        result["runtime"]["cpu_threads"] = 128
    else:
        result["parameter_dtypes"] = {}
    with pytest.raises(ValueError):
        verification.validate_completed(result, plan)


def test_exact_raw_text_and_token_differences_are_preserved(completed_job, tmp_path):
    plan = plan_for(completed_job, tmp_path)
    result = fixture_comparison(plan, tmp_path / "merged")
    changed = result["samples"][1]
    changed["merged_cpu"] = prediction(changed["expected"], ids=[21, 23])
    changed["merged_cpu"]["text"] = "\n" + changed["expected"] + " "
    result["metrics"] = verification.compare_samples(result["samples"])
    verification.validate_completed(result, plan)
    assert result["metrics"]["peft_cpu_passes"] == result["metrics"]["merged_cpu_passes"] == 6
    assert result["metrics"]["identical_token_output_count"] == 5
    assert result["metrics"]["identical_text_output_count"] == 5
    assert result["metrics"]["changed_case_ids"] == [changed["id"]]


def test_cpu_loader_disables_network_remote_code_and_gpu_device_maps(tmp_path, monkeypatch):
    calls = {}

    class Model:
        @classmethod
        def from_pretrained(cls, path, **options):
            calls["model"] = options
            return cls()

        def eval(self):
            return self

        def parameters(self):
            return [
                SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype="bf16", numel=lambda: 7)
            ]

    class Tokenizer:
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, path, **options):
            calls["tokenizer"] = options
            return cls()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=Model,
            AutoTokenizer=Tokenizer,
        ),
    )
    verification.load_cpu_model(tmp_path)
    assert calls["model"]["device_map"] == "cpu"
    assert calls["model"]["use_safetensors"] is True
    assert calls["model"]["torch_dtype"] == "bf16"
    assert all(options["local_files_only"] is True for options in calls.values())
    assert all(options["trust_remote_code"] is False for options in calls.values())
    bad = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type="cuda"))])
    with pytest.raises(RuntimeError, match="CPU"):
        verification.verify_cpu_parameters(bad)


def test_actual_export_helpers_use_verified_archive_and_original_job_prompts(
    completed_job,
    tmp_path,
    monkeypatch,
):
    plan = plan_for(completed_job, tmp_path)
    job_dir, _ = completed_job
    evidence = tmp_path / "child.json"
    # This mutable directory is explicitly ignored in favor of the registered tar hash.
    (job_dir / "adapter/adapter_model.safetensors").write_bytes(b"untrusted change")
    events, messages_seen = [], []

    class Model:
        def __init__(self, stage):
            self.stage = stage

        def eval(self):
            return self

        def parameters(self):
            return [
                SimpleNamespace(
                    device=SimpleNamespace(type="cpu"), dtype="torch.bfloat16", numel=lambda: 7
                )
            ]

    class Adapter:
        @classmethod
        def from_pretrained(cls, model, path, **options):
            assert (path / "adapter_model.safetensors").read_bytes() == b"adapter fixture"
            assert options["torch_device"] == "cpu"
            assert options["device_map"] == {"": "cpu"}
            assert options["local_files_only"] is True
            assert options["is_trainable"] is False
            return Model("peft_cpu")

    torch = SimpleNamespace(
        set_num_threads=lambda count: events.append(("threads", count)),
        set_num_interop_threads=lambda count: events.append(("interop", count)),
        get_num_threads=lambda: 4,
        get_num_interop_threads=lambda: 1,
        manual_seed=lambda _seed: None,
        __version__="test",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=Adapter))
    monkeypatch.setattr(verification.importlib.metadata, "version", lambda _name: "test")

    def merge(base, adapter, destination):
        assert events == [("threads", 4), ("interop", 1)]
        assert all(
            verification.os.environ[name] == ""
            for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        )
        assert (adapter / "adapter_model.safetensors").read_bytes() == b"adapter fixture"
        (destination / "model.safetensors").write_bytes(b"fake merged weights")
        write_json(destination / "tokenizer_config.json", {"test": True})

    monkeypatch.setattr("macfit_training.export.merge_cpu", merge)
    monkeypatch.setattr(
        verification,
        "load_cpu_model",
        lambda directory: (Model("merged_cpu" if directory.name == "merged" else "base"), object()),
    )
    by_question = {row["question"]: row for row in plan["selected_questions"]}
    job = json.loads((job_dir / "input.json").read_text())

    def generate(model, tokenizer, messages):
        question = messages[-1]["content"]
        row = by_question[question]
        assert messages == evaluation_messages(row, job)
        messages_seen.append((model.stage, question))
        return prediction(row["expected"])

    monkeypatch.setattr(verification, "generate_cpu", generate)
    assert verification.run_child({**plan, "child_output": str(evidence)}) == 0
    result = json.loads(evidence.read_text())
    verification.validate_completed(result, plan)
    assert len(messages_seen) == 12
    assert result["runtime"]["cpu_threads"] == 4
    assert evidence.stat().st_size < verification.MAX_CHILD_BYTES
    # The second phase performs actual byte/inventory checks, without importing Torch.
    inventory = tmp_path / "inventory.json"
    assert (
        verification.run_child(
            {
                **plan,
                "child_output": str(inventory),
                "inventory_only": True,
                "export_manifest": result["export_manifest"],
            }
        )
        == 0
    )
    verification.validate_inventory(json.loads(inventory.read_text()), result, plan)
    (tmp_path / "merged/model.safetensors").write_bytes(b"changed after child exit")
    assert (
        verification.run_child(
            {
                **plan,
                "child_output": str(inventory),
                "inventory_only": True,
                "export_manifest": result["export_manifest"],
            }
        )
        == 2
    )
    assert json.loads(inventory.read_text())["status"] == "worker_failed"


def worker_script(tmp_path, result):
    """A real CPU process substitutes only generation; final inventory uses actual code."""
    result_file, script = tmp_path / "worker-result.json", tmp_path / "worker.py"
    write_json(result_file, result)
    script.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "from macfit_training.export_verification import run_child\n"
        "config=json.loads(Path(sys.argv[-1]).read_text())\n"
        "if config.get('inventory_only'):\n"
        " raise SystemExit(run_child(config))\n"
        f"result=json.loads(Path({str(result_file)!r}).read_text())\n"
        "Path(config['child_output']).write_text(json.dumps(result))\n"
    )
    return script


def test_parent_stable_module_and_post_exit_byte_verification(completed_job, tmp_path, monkeypatch):
    plan = plan_for(completed_job, tmp_path)
    result = fixture_comparison(plan, tmp_path / "generated-merged")
    # The actual parent needs a fresh destination at planning time; fake comparison
    # worker simulates its completed export by moving prepared private fixture files.
    script = worker_script(tmp_path, result)
    original = subprocess.Popen
    observed, cleanup_pids = [], []
    cleanup_original = benchmark.terminate_group

    def popen(command, **options):
        observed.append((command, options))
        config = json.loads(Path(command[-1]).read_text())
        if not config.get("inventory_only"):
            (tmp_path / "generated-merged").rename(tmp_path / "merged")
        return original([sys.executable, str(script), command[-1]], **options)

    def cleanup(process):
        cleanup_pids.append(process.pid)
        cleanup_original(process)

    monkeypatch.setattr(verification.subprocess, "Popen", popen)
    monkeypatch.setattr(benchmark, "terminate_group", cleanup)
    monkeypatch.setattr(verification, "__name__", "__main__")
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "parent-value")
    document = verification.run_verification(
        completed_job[0],
        tmp_path / "merged",
        tmp_path / "evidence.json",
        base_model_dir=completed_job[1],
        question_ids=plan["question_ids"],
        walltime_seconds=30,
    )
    assert document["status"] == "succeeded"
    assert len(observed) == len(set(cleanup_pids)) == len(cleanup_pids) == 2
    assert all(
        command[1:4] == ["-m", verification.MODULE, "--child-config"] for command, _ in observed
    )
    assert all(options["start_new_session"] is True for _, options in observed)
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        assert all(options["env"][name] == "" for _, options in observed)
        assert verification.os.environ[name] == "parent-value"
    assert document["result"]["metrics"]["peft_cpu_passes"] == 6
    assert document["post_exit_validation"]["status"] == "succeeded"
    output = tmp_path / "evidence.json"
    assert json.loads(output.read_text())["status"] == "succeeded"
    assert output.stat().st_size < verification.MAX_EVIDENCE_BYTES
    assert (
        document["protocol"]["strict_scoring_helpers_source_sha256"]
        == hashlib.sha256(
            Path(verification.parse_strict_json.__code__.co_filename).read_bytes()
        ).hexdigest()
    )


def test_post_exit_child_rejects_export_tampering(completed_job, tmp_path, monkeypatch):
    plan = plan_for(completed_job, tmp_path)
    result = fixture_comparison(plan, tmp_path / "prepared")
    script = worker_script(tmp_path, result)
    original = subprocess.Popen

    def popen(command, **options):
        config = json.loads(Path(command[-1]).read_text())
        if config.get("inventory_only"):
            (tmp_path / "merged/model.safetensors").write_bytes(b"post-comparison tamper")
        else:
            (tmp_path / "prepared").rename(tmp_path / "merged")
        return original([sys.executable, str(script), command[-1]], **options)

    monkeypatch.setattr(verification.subprocess, "Popen", popen)
    result = verification.run_verification(
        completed_job[0],
        tmp_path / "merged",
        tmp_path / "evidence.json",
        question_ids=plan["question_ids"],
        walltime_seconds=30,
    )
    assert result["status"] == "worker_failed"
    assert result["post_exit_validation"]["status"] == "worker_failed"


def test_expired_absolute_deadline_starts_no_process(completed_job, tmp_path, monkeypatch):
    monkeypatch.setattr(
        verification.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("No process")
    )
    result = verification.run_verification(
        completed_job[0],
        tmp_path / "merged",
        tmp_path / "evidence.json",
        stop_at="2000-01-01T00:00:00Z",
    )
    assert result["status"] == "skipped_deadline"
    assert not (tmp_path / "merged").exists()


def test_real_timeout_reaps_group_once_and_retains_partial_evidence(
    completed_job,
    tmp_path,
    monkeypatch,
):
    script, pid_file = tmp_path / "sleeping.py", tmp_path / "pid.json"
    script.write_text(
        "import json,os,signal,subprocess,sys,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "config=json.loads(Path(sys.argv[-1]).read_text())\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        f"Path({str(pid_file)!r}).write_text(json.dumps([os.getpid(),child.pid]))\n"
        "Path(config['child_output']).write_text(json.dumps({'status':'exporting_on_cpu'}))\n"
        "time.sleep(60)\n"
    )
    calls = []
    original = benchmark.terminate_group

    def cleanup(process):
        calls.append(process.pid)
        original(process)

    monkeypatch.setattr(benchmark, "terminate_group", cleanup)
    started = time.monotonic()
    result = verification.run_verification(
        completed_job[0],
        tmp_path / "merged",
        tmp_path / "evidence.json",
        walltime_seconds=10,
        worker_command=[sys.executable, str(script)],
    )
    assert result["status"] == "timed_out"
    assert result["result"]["status"] == "exporting_on_cpu"
    assert len(calls) == 1
    assert time.monotonic() - started < 10
    for pid in json.loads(pid_file.read_text()):
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], text=True, capture_output=True, check=False
        ).stdout.strip()
        assert not state or state.startswith("Z")


def test_cleanup_permission_failure_preserves_prior_exception_and_never_retries(
    completed_job,
    tmp_path,
    monkeypatch,
):
    script = tmp_path / "oversized.py"
    script.write_text(
        "import json,sys,time\nfrom pathlib import Path\n"
        "config=json.loads(Path(sys.argv[-1]).read_text())\n"
        "Path(config['child_output']).write_text(' '*(128*1024+1))\n"
        "time.sleep(2)\n"
    )
    calls = []
    original = benchmark.terminate_group

    def cleanup(process):
        calls.append(process.pid)
        original(process)  # Real process is always reaped before simulated OS error.
        raise PermissionError("sensitive text that must not be persisted")

    monkeypatch.setattr(benchmark, "terminate_group", cleanup)
    output = tmp_path / "evidence.json"
    with pytest.raises(ValueError, match="exceeded") as failure:
        verification.run_verification(
            completed_job[0],
            tmp_path / "merged",
            output,
            walltime_seconds=20,
            worker_command=[sys.executable, str(script)],
        )
    assert isinstance(failure.value.__cause__, PermissionError)
    assert len(calls) == 1
    evidence = json.loads(output.read_text())
    assert evidence["status"] == "cleanup_failed"
    assert evidence["cleanup_error_type"] == "PermissionError"
    assert evidence["error_type"] == "ValueError"
    assert "sensitive text" not in output.read_text()


def test_inventory_attestation_cannot_claim_a_different_manifest(completed_job, tmp_path):
    plan = plan_for(completed_job, tmp_path)
    comparison = fixture_comparison(plan, tmp_path / "merged")
    inventory = {
        "status": "succeeded",
        **{key: plan[key] for key in verification.SOURCE_KEYS},
        "export_manifest_sha256": canonical_sha256(comparison["export_manifest"]),
        "export_manifest_file": comparison["export_manifest_file"],
    }
    verification.validate_inventory(inventory, comparison, plan)
    for field in ("source_adapter_sha256", "export_manifest_sha256", "export_manifest_file"):
        changed = copy.deepcopy(inventory)
        changed[field] = "changed"
        with pytest.raises(ValueError, match="final export inventory"):
            verification.validate_inventory(changed, comparison, plan)


@pytest.mark.parametrize(
    "finite_prompt,finite_output,raises",
    [
        (True, True, False),
        (False, True, True),
        (True, False, True),
    ],
)
def test_generation_checks_raw_logit_boundaries_and_keeps_actual_greedy_tokens(
    monkeypatch,
    finite_prompt,
    finite_output,
    raises,
):
    from contextlib import nullcontext

    class Tensor(list):
        def __getitem__(self, index):
            value = super().__getitem__(index)
            return Tensor(value) if isinstance(index, slice) else value

        def unsqueeze(self, _axis):
            return Tensor([list(self)])

        def cpu(self):
            return self

        def tolist(self):
            return list(self)

    options = {}
    torch = SimpleNamespace(
        long="long",
        tensor=lambda value, **_options: Tensor(value),
        inference_mode=nullcontext,
        isfinite=lambda value: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: value)),
        ones_like=lambda value: Tensor([1] * len(value)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return [1, 4, 5]

        def decode(self, tokens, **kwargs):
            assert kwargs == {"skip_special_tokens": True}
            assert tokens.tolist() == [10, 2]
            return " 42\n"

    class Model:
        calls = 0

        def __call__(self, **kwargs):
            assert kwargs["use_cache"] is False
            assert kwargs["logits_to_keep"] == 1
            self.calls += 1
            return SimpleNamespace(logits=finite_prompt if self.calls == 1 else finite_output)

        def generate(self, **kwargs):
            options.update(kwargs)
            return [Tensor([1, 4, 5, 10, 2])]

    model = Model()
    if raises:
        with pytest.raises(RuntimeError, match="nonfinite"):
            verification.generate_cpu(model, Tokenizer(), [{"role": "user", "content": "6*7?"}])
        assert model.calls == (1 if not finite_prompt else 2)
    else:
        output = verification.generate_cpu(
            model, Tokenizer(), [{"role": "user", "content": "6*7?"}]
        )
        assert output["text"] == " 42\n"
        assert output["output_ids"] == [10, 2]
        assert output["output_ids_sha256"] == canonical_sha256([10, 2])
        assert output["finite_logits"] is True
        assert output["stopped_by_limit"] is False
        assert options["max_new_tokens"] == 96
        assert options["do_sample"] is False
        assert options["temperature"] is None
        assert options["top_p"] is None
        assert options["top_k"] is None


def test_saved_export_rejects_hardlinks_even_with_matching_bytes(completed_job, tmp_path):
    plan = plan_for(completed_job, tmp_path)
    result = fixture_comparison(plan, tmp_path / "merged")
    verification.os.link(tmp_path / "merged/model.safetensors", tmp_path / "shared-copy")
    with pytest.raises(ValueError, match="hard links"):
        verification.verify_export(tmp_path / "merged", result["export_manifest"])
