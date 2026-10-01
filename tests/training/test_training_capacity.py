"""CPU protocol, evidence and real process-boundary checks; never a GPU simulation."""

import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from amd_inference_opt.vllm_model_snapshot import (
    VLLMModelSnapshotManifest,
    VLLMSnapshotFile,
    capture_vllm_model_snapshot,
    snapshot_identity_sha256,
)
from macfit_training import benchmark
from macfit_training import training_capacity as capacity
from macfit_training.benchmark import BENCHMARK_MODELS
from macfit_training.config import MODELS


def result(identity=None):
    identity = identity or BENCHMARK_MODELS["qwen3-14b"]
    files = [
        VLLMSnapshotFile(
            relative_path="model.safetensors", sha256="a" * 64, size_bytes=10, storage="regular"
        )
    ]
    digest = snapshot_identity_sha256(
        model_id=identity["repo_id"],
        revision=identity["revision"],
        tokenizer_revision=identity["revision"],
        files=files,
    )
    snapshot = VLLMModelSnapshotManifest(
        model_id=identity["repo_id"],
        revision=identity["revision"],
        tokenizer_revision=identity["revision"],
        root=Path("/snapshot-fixture"),
        file_count=1,
        total_bytes=10,
        files=files,
        snapshot_digest=digest,
        captured_at="2026-09-30T20:00:00+00:00",
    )
    samples = [
        {"seconds": seconds, "loss": 1.0, "gradient_norm": 0.5, "finite_gradients": True}
        for seconds in (0.5, 1.0, 1.5)
    ]
    return {
        "identity": identity,
        "status": "succeeded",
        "adapter_published": False,
        "quality_evaluated": False,
        "training_finished_adapter": False,
        "optimizer_step_calls": 5,
        "warmup_samples": samples[:2],
        "timed_samples": samples,
        "metrics": capacity.summarize_steps(samples),
        "vocabulary": capacity.validate_vocabulary(
            10, 12, 16, 11, input_embedding_rows=16, output_embedding_rows=16
        ),
        "parameters": {
            "trainable_parameters": 10,
            "frozen_parameters": 90,
            "total_parameters": 100,
            "trainable_parameters_by_dtype": {"torch.float32": 10},
            "frozen_parameters_by_dtype": {"torch.bfloat16": 90},
            "full_gpu_parameters": True,
            "base_parameters_frozen": True,
        },
        "runtime": {"rocm": "CPU evidence fixture", "gpu_name": "CPU evidence fixture"},
        "adapter_weights_changed": True,
        "selected_lora_B_max_absolute_update": 0.01,
        "peak_allocated_bytes": 1000,
        "peak_reserved_bytes": 2000,
        "model_snapshot_sha256": digest,
        "model_snapshot_file_count": 1,
        "model_snapshot_bytes": 10,
        "model_snapshot": snapshot.model_dump(mode="json", by_alias=True),
        "input_ids_sha256": "b" * 64,
        "labels_sha256": "c" * 64,
        "optimizer_state_dtypes": {
            "step": {"torch.float32": 2},
            "exp_avg": {"torch.float32": 10},
            "exp_avg_sq": {"torch.float32": 10},
        },
    }


def test_large_model_catalog_is_operator_only_and_pinned():
    plan = capacity.validate_plan(["qwen3-14b", "qwen3-32b"], 128, 1200, 600, None)
    assert [identity["id"] for identity in plan["models"]] == list(capacity.MODEL_IDS)
    assert all(len(identity["revision"]) == 40 for identity in plan["models"])
    assert not set(capacity.MODEL_IDS).intersection(MODELS)
    assert plan["protocol"]["base_dtype"] == "torch.bfloat16"
    assert plan["protocol"]["trainable_adapter_dtype"] == "torch.float32"
    assert plan["protocol"]["lora_rank"] == 8


@pytest.mark.parametrize("sequence", [128, 256])
def test_causal_batch_has_exactly_thirty_two_supervised_targets_and_masked_prefix(sequence):
    batch = capacity.make_batch([1, 2, 3], [4, 5], 9, 10, sequence)
    prefix = sequence - 32
    assert len(batch["input_ids"]) == len(batch["labels"]) == sequence
    assert batch["attention_mask"] == [1] * sequence
    assert batch["labels"][:prefix] == [-100] * prefix
    assert batch["labels"][prefix:] == batch["input_ids"][prefix:]
    assert batch["input_ids"][-1] == batch["labels"][-1] == 9
    assert sum(token != -100 for token in batch["labels"][1:]) == 32


def test_added_eos_uses_total_tokenizer_length_and_fits_padded_model_embeddings():
    # Added special tokens are absent from .vocab_size but have model embeddings.
    base, total, model_tokens, eos_id = 151643, 151669, 151936, 151645
    with pytest.raises(ValueError, match="EOS"):
        capacity.make_batch([10, 11], [12, 13], eos_id, base, 128)
    encoded = capacity.make_batch([10, 11], [12, 13], eos_id, total, 128)
    assert encoded["input_ids"][-1] == encoded["labels"][-1] == eos_id
    assert sum(token != -100 for token in encoded["labels"][1:]) == 32
    vocabulary = capacity.validate_vocabulary(
        base,
        total,
        model_tokens,
        eos_id,
        input_embedding_rows=model_tokens,
        output_embedding_rows=model_tokens,
    )
    assert vocabulary["tokenizer_base_tokens"] < vocabulary["eos_token_id"]
    assert vocabulary["tokenizer_total_tokens"] < vocabulary["input_embedding_rows"]


@pytest.mark.parametrize(
    "change",
    [
        {"base_tokens": True},
        {"base_tokens": 20},
        {"total_tokens": 32},
        {"eos_id": 12},
        {"input_embedding_rows": 31},
        {"output_embedding_rows": 31},
        {"input_embedding_rows": True},
    ],
)
def test_vocabulary_rejects_out_of_model_tokens_eos_and_mismatched_embedding_rows(change):
    values = {"base_tokens": 10, "total_tokens": 12, "model_tokens": 16, "eos_id": 11}
    with pytest.raises(ValueError):
        capacity.validate_vocabulary(**{**values, **change})


def cached_snapshot(tmp_path, payload):
    cache = tmp_path / "models--Qwen--Qwen3-14B"
    blobs, snapshot_dir = cache / "blobs", cache / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)
    blob = blobs / hashlib.sha256(payload).hexdigest()
    blob.write_bytes(payload)
    (snapshot_dir / "config.json").symlink_to(Path("../../blobs") / blob.name)
    identity = BENCHMARK_MODELS["qwen3-14b"]
    snapshot = capture_vllm_model_snapshot(
        snapshot_dir,
        model_id=identity["repo_id"],
        revision=identity["revision"],
        tokenizer_revision=identity["revision"],
    )
    return snapshot, blob


def test_captured_huggingface_config_symlink_is_read_without_loosening_job_metadata(tmp_path):
    payload = b'{"model_type":"qwen3","vocab_size":151936}\n'
    snapshot, blob = cached_snapshot(tmp_path, payload)
    assert snapshot.files[0].storage == "symlink"
    assert capacity.read_snapshot_config(snapshot)["vocab_size"] == 151936
    with pytest.raises(ValueError, match="symbolic links"):
        capacity.read_json(snapshot.root / "config.json")
    blob.write_bytes(payload.replace(b"151936", b"151937"))
    with pytest.raises(ValueError, match="changed"):
        capacity.read_snapshot_config(snapshot)


def test_snapshot_config_reader_refuses_oversized_or_non_object_verified_json(tmp_path):
    oversized, _ = cached_snapshot(tmp_path / "large", b" " * (capacity.MAX_CHILD_BYTES + 1))
    with pytest.raises(ValueError, match="bounded"):
        capacity.read_snapshot_config(oversized)
    scalar, _ = cached_snapshot(tmp_path / "scalar", b"[151936]\n")
    with pytest.raises(ValueError, match="JSON object"):
        capacity.read_snapshot_config(scalar)


@pytest.mark.parametrize(
    "change",
    [
        {"reference_ids": []},
        {"target_ids": []},
        {"reference_ids": [10]},
        {"target_ids": [True]},
        {"eos_id": -1},
        {"eos_id": True},
        {"vocabulary_size": 0},
        {"sequence_length": 32},
        {"sequence_length": 128.0},
    ],
)
def test_synthetic_batch_rejects_invalid_shapes_and_out_of_vocabulary_targets(change):
    kwargs = {
        "reference_ids": [1, 2],
        "target_ids": [3, 4],
        "eos_id": 9,
        "vocabulary_size": 10,
        "sequence_length": 128,
    }
    with pytest.raises(ValueError):
        capacity.make_batch(**{**kwargs, **change})


@pytest.mark.parametrize(
    "changes",
    [
        {"models": []},
        {"models": ["qwen3-8b"]},
        {"models": ["qwen3-14b", "qwen3-14b"]},
        {"sequence_length": 2048},
        {"walltime_seconds": 9},
        {"walltime_seconds": 3601},
        {"model_timeout_seconds": 0},
        {"model_timeout_seconds": True},
        {"stop_at": "2026-10-01T03:00:00"},
    ],
)
def test_plan_has_finite_size_and_time_bounds(changes):
    kwargs = {
        "models": ["qwen3-14b"],
        "sequence_length": 128,
        "walltime_seconds": 1200,
        "model_timeout_seconds": 600,
        "stop_at": None,
    }
    with pytest.raises(ValueError):
        capacity.validate_plan(**{**kwargs, **changes})


def test_step_rates_count_actual_optimizer_steps_and_supervised_tokens():
    measured = result()["metrics"]
    assert measured["median_step_seconds"] == 1.0
    assert measured["median_optimizer_steps_per_second"] == 1.0
    assert measured["median_supervised_tokens_per_second"] == 32.0
    for key, value in (
        ("seconds", 0),
        ("seconds", float("inf")),
        ("loss", float("nan")),
        ("gradient_norm", -1),
        ("finite_gradients", False),
    ):
        samples = copy.deepcopy(result()["timed_samples"])
        samples[0][key] = value
        with pytest.raises(ValueError):
            capacity.summarize_steps(samples)


@pytest.mark.parametrize(
    "key,value",
    [
        ("adapter_published", True),
        ("quality_evaluated", True),
        ("optimizer_step_calls", 3),
        ("adapter_weights_changed", False),
        ("model_snapshot_sha256", "invalid"),
        ("peak_allocated_bytes", 0),
    ],
)
def test_partial_or_unsubstantiated_capacity_success_is_rejected(key, value):
    evidence = result()
    evidence[key] = value
    with pytest.raises(ValueError):
        capacity.validate_completed(evidence, evidence["identity"])


def test_evidence_needs_actual_base_adapter_dtypes_and_matching_timing_samples():
    evidence = result()
    capacity.validate_completed(evidence, evidence["identity"])
    wrong_dtype = copy.deepcopy(evidence)
    wrong_dtype["parameters"]["trainable_parameters_by_dtype"] = {"torch.bfloat16": 10}
    wrong_metrics = copy.deepcopy(evidence)
    wrong_metrics["metrics"]["median_step_seconds"] = 0.01
    wrong_runtime = copy.deepcopy(evidence)
    wrong_runtime["runtime"]["rocm"] = None
    for changed in (wrong_dtype, wrong_metrics, wrong_runtime):
        with pytest.raises(ValueError):
            capacity.validate_completed(changed, evidence["identity"])


def test_completed_evidence_requires_the_actual_padded_input_and_output_vocabulary():
    evidence = result()
    capacity.validate_completed(evidence, evidence["identity"])
    for key, value in (
        ("input_embedding_rows", None),
        ("output_embedding_rows", 12),
        ("tokenizer_total_tokens", 17),
        ("eos_token_id", 12),
    ):
        changed = copy.deepcopy(evidence)
        changed["vocabulary"][key] = value
        with pytest.raises(ValueError):
            capacity.validate_completed(changed, evidence["identity"])
    missing = copy.deepcopy(evidence)
    del missing["vocabulary"]["output_embedding_rows"]
    with pytest.raises(ValueError):
        capacity.validate_completed(missing, evidence["identity"])


def fixture_worker(tmp_path, *, descendant_marker=None):
    source = tmp_path / "child.py"
    document = result()
    code = "import json,os,sys\nconfig=json.load(open(sys.argv[1]))\n"
    if descendant_marker is not None:
        descendant = (
            "import signal,time\n"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            f"with open({str(descendant_marker)!r},'a') as stream:\n"
            " while True:\n"
            "  stream.write('x'); stream.flush(); time.sleep(0.03)\n"
        )
        code += (
            "import subprocess,time\n"
            f"subprocess.Popen([sys.executable,'-c',{descendant!r}],"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            f"while not os.path.exists({str(descendant_marker)!r}): time.sleep(0.01)\n"
        )
    code += (
        f"document=json.loads({json.dumps(document)!r})\n"
        "document['identity']=config['identity']\n"
        "from amd_inference_opt.vllm_model_snapshot import (\n"
        " snapshot_identity_sha256, VLLMSnapshotFile)\n"
        "snapshot=document['model_snapshot']\n"
        "snapshot.update(model_id=config['identity']['repo_id'],"
        "revision=config['identity']['revision'],tokenizer_revision=config['identity']['revision'])\n"
        "snapshot['snapshot_digest']=snapshot_identity_sha256(model_id=snapshot['model_id'],"
        "revision=snapshot['revision'],tokenizer_revision=snapshot['tokenizer_revision'],"
        "files=[VLLMSnapshotFile.model_validate(row) for row in snapshot['files']])\n"
        "document['model_snapshot_sha256']=snapshot['snapshot_digest']\n"
        "with open(config['child_output'],'w') as stream: json.dump(document,stream)\n"
    )
    source.write_text(code)
    return [sys.executable, str(source)]


def test_complete_small_evidence_is_atomic_bounded_and_contains_no_weights(tmp_path):
    output = tmp_path / "capacity.json"
    evidence = capacity.run_probe(
        ["qwen3-14b", "qwen3-32b"],
        output,
        worker_command=fixture_worker(tmp_path),
    )
    assert evidence["status"] == "succeeded"
    assert len(evidence["models"]) == 2
    assert json.loads(output.read_text()) == evidence
    assert output.stat().st_size < capacity.MAX_EVIDENCE_BYTES
    assert all(row["adapter_published"] is False for row in evidence["models"])
    assert all(row["model_snapshot"]["files"] for row in evidence["models"])
    assert not list(tmp_path.glob("lora-capacity-*"))
    assert not list(tmp_path.glob("*.safetensors"))
    original = output.read_bytes()
    with pytest.raises(ValueError, match="fresh output"):
        capacity.run_probe(["qwen3-14b"], output)
    assert output.read_bytes() == original


def test_full_snapshot_inventory_is_digest_bound_and_can_be_relocated(tmp_path):
    evidence = result()
    original_inventory = copy.deepcopy(evidence["model_snapshot"]["files"])
    evidence["model_snapshot"]["root"] = str(tmp_path / "redownloaded-pinned-model")
    capacity.validate_completed(evidence, evidence["identity"])
    assert evidence["model_snapshot"]["files"] == original_inventory
    corrupted = copy.deepcopy(evidence)
    corrupted["model_snapshot"]["files"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        capacity.validate_completed(corrupted, corrupted["identity"])
    missing = copy.deepcopy(evidence)
    del missing["model_snapshot"]
    with pytest.raises(ValueError):
        capacity.validate_completed(missing, missing["identity"])


def test_expired_absolute_deadline_launches_no_child(tmp_path):
    output = tmp_path / "expired.json"
    expired = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    evidence = capacity.run_probe(
        ["qwen3-14b"],
        output,
        stop_at=expired,
        worker_command=[sys.executable, "-c", "raise RuntimeError('must not execute')"],
    )
    assert evidence["status"] == "incomplete"
    assert evidence["models"][0]["status"] == "skipped_deadline"


def test_real_default_module_child_waits_for_gpu_lease_and_is_killed_on_deadline(
    tmp_path, monkeypatch
):
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("MACFIT_GPU_LOCK", str(lock))
    root = Path(__file__).resolve().parents[2] / "src"
    monkeypatch.setenv("PYTHONPATH", str(root))
    with lock.open("w") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        started = time.monotonic()
        evidence = capacity.run_probe(
            ["qwen3-14b"],
            tmp_path / "lock-wait.json",
            walltime_seconds=10,
            model_timeout_seconds=1,
        )
    assert time.monotonic() - started < 5
    row = evidence["models"][0]
    assert row["status"] == "timed_out"
    # A correctly launched real module published before acquiring the GPU lease.
    assert row["adapter_published"] is False
    assert "runtime" not in row and "timed_samples" not in row


def test_successful_leader_exit_cleans_its_sigterm_ignoring_descendant(tmp_path):
    marker = tmp_path / "heartbeat"
    worker = fixture_worker(tmp_path, descendant_marker=marker)
    evidence = capacity.run_probe(
        ["qwen3-14b"], tmp_path / "descendant.json", worker_command=worker
    )
    assert evidence["status"] == "succeeded"
    size = marker.stat().st_size
    assert size > 0
    time.sleep(0.2)
    assert marker.stat().st_size == size


def boundary_worker(tmp_path, path):
    if path == "succeeded":
        return fixture_worker(tmp_path)
    source = tmp_path / "boundary-child.py"
    code = "import json,sys,time\nconfig=json.load(open(sys.argv[1]))\n"
    if path == "invalid_json":
        code += "with open(config['child_output'],'w') as stream: stream.write('invalid-json')\n"
    source.write_text(code + "time.sleep(30)\n")
    return [sys.executable, str(source)]


@pytest.mark.parametrize("path", ["succeeded", "timed_out", "invalid_json"])
def test_cleanup_attempts_one_process_group_exactly_once_on_each_supervisor_path(
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
    output = tmp_path / "once.json"
    options = {"model_timeout_seconds": 1, "worker_command": boundary_worker(tmp_path, path)}
    if path == "invalid_json":
        with pytest.raises(json.JSONDecodeError):
            capacity.run_probe(["qwen3-14b"], output, **options)
    else:
        evidence = capacity.run_probe(["qwen3-14b"], output, **options)
        assert evidence["models"][0]["status"] == path
    assert len(calls) == 1


@pytest.mark.parametrize("path", ["succeeded", "timed_out"])
def test_cleanup_permission_failure_is_sanitized_and_never_retried(tmp_path, monkeypatch, path):
    original = benchmark.terminate_group
    calls = []

    def forbidden_cleanup(process):
        calls.append(process.pid)
        original(process)
        raise PermissionError("Private process ownership details")

    monkeypatch.setattr(benchmark, "terminate_group", forbidden_cleanup)
    output = tmp_path / "cleanup-failed.json"
    with pytest.raises(PermissionError, match="ownership"):
        capacity.run_probe(
            ["qwen3-14b"],
            output,
            model_timeout_seconds=1,
            worker_command=boundary_worker(tmp_path, path),
        )
    assert len(calls) == 1
    evidence = json.loads(output.read_text())
    assert evidence["status"] == evidence["models"][0]["status"] == "cleanup_failed"
    assert evidence["models"][0]["cleanup_error_type"] == "PermissionError"
    assert "Private process ownership details" not in output.read_text()


def test_original_supervisor_failure_survives_a_single_cleanup_permission_error(
    tmp_path, monkeypatch
):
    original = benchmark.terminate_group
    calls = []

    def forbidden_cleanup(process):
        calls.append(process.pid)
        original(process)
        raise PermissionError("Private process ownership details")

    monkeypatch.setattr(benchmark, "terminate_group", forbidden_cleanup)
    output = tmp_path / "original-failed.json"
    with pytest.raises(json.JSONDecodeError) as error:
        capacity.run_probe(
            ["qwen3-14b"], output, worker_command=boundary_worker(tmp_path, "invalid_json")
        )
    assert isinstance(error.value.__cause__, PermissionError)
    assert error.value.__notes__ == ["Process-group cleanup failed: PermissionError"]
    assert len(calls) == 1
    evidence = json.loads(output.read_text())
    assert evidence["status"] == evidence["models"][0]["status"] == "supervisor_failed"
    assert evidence["models"][0]["cleanup_error_type"] == "PermissionError"
    assert "Private process ownership details" not in output.read_text()


def test_capacity_import_and_validate_only_never_import_accelerator_libraries():
    root = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys\n"
        "from macfit_training import training_capacity as module\n"
        "assert module.main(['--validate-only']) == 0\n"
        "assert not {'torch','transformers','peft'}.intersection(sys.modules)\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
        stdout=subprocess.DEVNULL,
    )


def test_oversized_child_json_and_nonfinite_publication_are_refused(tmp_path):
    path = tmp_path / "child.json"
    path.write_bytes(b" " * (capacity.MAX_CHILD_BYTES + 1))
    with pytest.raises(ValueError, match="bound"):
        capacity._read_child(path)
    with pytest.raises(ValueError):
        capacity.write_atomic(tmp_path / "invalid.json", {"loss": float("nan")})
    assert not (tmp_path / "invalid.json").exists()
