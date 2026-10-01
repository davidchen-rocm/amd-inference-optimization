"""Operator-only, bounded LoRA training capacity checks; no accelerator imports at startup.

Five synthetic optimizer updates measure capacity, not fine-tuning quality. No
adapter or model weights are saved, and the website model registry is unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import canonical_sha256
from .benchmark import BENCHMARK_MODELS, _ProcessGroupCleanup, parse_stop_at, write_atomic
from .export import read_json, real_path
from .hardware_probe import positive_finite

MODULE = "macfit_training.training_capacity"
SCHEMA = "macfit-lora-training-capacity.v1"
MODEL_IDS = ("qwen3-14b", "qwen3-32b")
SEQUENCE_LENGTHS = (128, 256)
SUPERVISED_TOKENS = 32
WARMUP_STEPS = 2
TIMED_STEPS = 3
MAX_EVIDENCE_BYTES = 1024**2 - 1
MAX_CHILD_BYTES = 128 * 1024
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
REFERENCE = (
    "Juniper is a fictional shop. Orders ship within two business days. Unused items "
    "may be returned within thirty calendar days. Support works Monday through Friday. "
)
TARGET = (
    "Juniper orders ship within two business days. Unused items have a thirty-day return "
    "window. These are fictional reference facts for a capacity microbenchmark. "
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def protocol(sequence_length: int) -> dict[str, Any]:
    if type(sequence_length) is not int or sequence_length not in SEQUENCE_LENGTHS:
        raise ValueError("Choose a sequence length of exactly 128 or 256 tokens.")
    return {
        "batch_size": 1,
        "sequence_length": sequence_length,
        "supervised_tokens": SUPERVISED_TOKENS,
        "warmup_optimizer_steps": WARMUP_STEPS,
        "timed_optimizer_steps": TIMED_STEPS,
        "method": "synthetic continuation causal-LM loss",
        "base_dtype": "torch.bfloat16",
        "trainable_adapter_dtype": "torch.float32",
        "autocast_adapter_dtype": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": list(TARGET_MODULES),
        "gradient_checkpointing": True,
        "checkpoint_use_reentrant": False,
        "optimizer": "AdamW",
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 42,
        "reference_text": REFERENCE,
        "target_text": TARGET,
        "reference_text_sha256": hashlib.sha256(REFERENCE.encode()).hexdigest(),
        "target_text_sha256": hashlib.sha256(TARGET.encode()).hexdigest(),
        "padding": "repeat reference to prefix length; repeat target to 31 tokens, append EOS",
        "timing": "synchronized wall time for zero_grad, forward, loss check, backward, "
        "gradient norm check/clip and AdamW step",
    }


def validate_plan(
    models: list[str],
    sequence_length: int,
    walltime_seconds: int,
    model_timeout_seconds: int,
    stop_at: str | None,
) -> dict[str, Any]:
    if (
        not isinstance(models, list)
        or not 1 <= len(models) <= 2
        or any(not isinstance(model, str) or model not in MODEL_IDS for model in models)
        or len(models) != len(set(models))
    ):
        raise ValueError("Choose one or both distinct pinned 14B/32B capacity-probe models.")
    if type(walltime_seconds) is not int or not 10 <= walltime_seconds <= 3600:
        raise ValueError("Walltime must be 10 to 3600 integer seconds including cleanup.")
    if type(model_timeout_seconds) is not int or not 1 <= model_timeout_seconds <= 1800:
        raise ValueError("Per-model timeout must be 1 to 1800 integer seconds.")
    return {
        "models": [copy.deepcopy(BENCHMARK_MODELS[model]) for model in models],
        "protocol": protocol(sequence_length),
        "walltime_seconds": walltime_seconds,
        "model_timeout_seconds": model_timeout_seconds,
        "stop_at": parse_stop_at(stop_at).isoformat() if stop_at else None,
        "source_sha256": SOURCE_SHA256,
    }


def make_batch(
    reference_ids: list[int],
    target_ids: list[int],
    eos_id: int,
    vocabulary_size: int,
    sequence_length: int,
) -> dict[str, list[int]]:
    protocol(sequence_length)
    if type(vocabulary_size) is not int or vocabulary_size <= 1:
        raise ValueError("A valid model vocabulary is required.")
    for ids in (reference_ids, target_ids):
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids)
        ):
            raise ValueError("Fixed public text must encode to nonempty valid token IDs.")
    if type(eos_id) is not int or not 0 <= eos_id < vocabulary_size:
        raise ValueError("A valid EOS token is required.")
    prefix_length = sequence_length - SUPERVISED_TOKENS
    prefix = (reference_ids * math.ceil(prefix_length / len(reference_ids)))[:prefix_length]
    answer_length = SUPERVISED_TOKENS - 1
    target = (target_ids * math.ceil(answer_length / len(target_ids)))[:answer_length] + [eos_id]
    batch = {
        "input_ids": prefix + target,
        "attention_mask": [1] * sequence_length,
        "labels": [-100] * prefix_length + target,
    }
    if sum(token != -100 for token in batch["labels"][1:]) != SUPERVISED_TOKENS:
        raise ValueError("The causal loss must supervise exactly 32 continuation tokens.")
    return batch


def validate_vocabulary(
    base_tokens: int,
    total_tokens: int,
    model_tokens: int,
    eos_id: int,
    *,
    input_embedding_rows: int | None = None,
    output_embedding_rows: int | None = None,
) -> dict[str, int]:
    """Added tokenizer tokens must fit the pinned, possibly padded model vocabulary."""
    if (
        any(type(value) is not int for value in (base_tokens, total_tokens, model_tokens))
        or not 1 < base_tokens <= total_tokens <= model_tokens
        or type(eos_id) is not int
        or not 0 <= eos_id < total_tokens
    ):
        raise ValueError("The full tokenizer vocabulary and EOS must fit the pinned model.")
    metadata = {
        "tokenizer_base_tokens": base_tokens,
        "tokenizer_total_tokens": total_tokens,
        "model_config_tokens": model_tokens,
        "eos_token_id": eos_id,
    }
    for name, rows in (
        ("input_embedding_rows", input_embedding_rows),
        ("output_embedding_rows", output_embedding_rows),
    ):
        if rows is not None:
            if type(rows) is not int or rows != model_tokens:
                raise ValueError("Actual model embedding rows must match its pinned config.")
            metadata[name] = rows
    return metadata


def read_snapshot_config(snapshot: Any) -> dict[str, Any]:
    """Read only the already hash-captured config, including an HF blob symlink."""
    records = [row for row in snapshot.files if row.relative_path == "config.json"]
    if len(records) != 1 or records[0].size_bytes > MAX_CHILD_BYTES:
        raise ValueError("The pinned snapshot needs one bounded config.json inventory record.")
    record = records[0]
    # Model snapshots permit HF file symlinks. Shared job/export metadata retains
    # its stricter no-link policy; resolve only this inventoried model component.
    path = (Path(snapshot.root) / "config.json").resolve(strict=True)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != record.size_bytes:
            raise ValueError("The pinned model config must be the captured regular blob.")
        payload = stream.read(MAX_CHILD_BYTES + 1)
    if len(payload) != record.size_bytes or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise ValueError("The model config changed after its pinned snapshot was captured.")
    config = json.loads(payload)
    if not isinstance(config, dict):
        raise ValueError("The pinned model config must contain a JSON object.")
    return config


def summarize_steps(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import statistics

    if not isinstance(samples, list) or len(samples) != TIMED_STEPS:
        raise ValueError("Exactly three measured optimizer steps are required.")
    durations = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("A measured optimizer step must be an object.")
        durations.append(positive_finite(sample.get("seconds"), "Optimizer step time"))
        for key in ("loss", "gradient_norm"):
            value = sample.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Loss and gradient norms must be finite and nonnegative.")
        if sample.get("finite_gradients") is not True:
            raise ValueError("Every step must verify finite gradients.")
    median = statistics.median(durations)
    return {
        "samples": samples,
        "median_step_seconds": median,
        "minimum_step_seconds": min(durations),
        "maximum_step_seconds": max(durations),
        "median_optimizer_steps_per_second": 1 / median,
        "median_supervised_tokens_per_second": SUPERVISED_TOKENS / median,
    }


def _parameter_counts(model: Any) -> dict[str, Any]:
    frozen, trainable = Counter(), Counter()
    for name, parameter in model.named_parameters():
        if parameter.device.type != "cuda":
            raise RuntimeError("All base and adapter parameters must reside on the GPU.")
        if parameter.requires_grad:
            if "lora_" not in name or str(parameter.dtype) != "torch.float32":
                raise RuntimeError("Only standard FP32 LoRA parameters may be trained.")
            trainable[str(parameter.dtype)] += parameter.numel()
        else:
            if str(parameter.dtype) != "torch.bfloat16":
                raise RuntimeError("Frozen base parameters must remain BF16.")
            frozen[str(parameter.dtype)] += parameter.numel()
    trainable_total, frozen_total = sum(trainable.values()), sum(frozen.values())
    if not 0 < trainable_total < trainable_total + frozen_total:
        raise RuntimeError("The probe must freeze the base and restrict training to LoRA.")
    return {
        "trainable_parameters": trainable_total,
        "frozen_parameters": frozen_total,
        "total_parameters": trainable_total + frozen_total,
        "trainable_parameters_by_dtype": dict(trainable),
        "frozen_parameters_by_dtype": dict(frozen),
        "full_gpu_parameters": True,
        "base_parameters_frozen": True,
    }


def _optimizer_state_dtypes(optimizer: Any, torch: Any) -> dict[str, Any]:
    counters: dict[str, Counter] = {}
    for state in optimizer.state.values():
        for name, tensor in state.items():
            if torch.is_tensor(tensor):
                counters.setdefault(name, Counter())[str(tensor.dtype)] += tensor.numel()
    return {name: dict(counts) for name, counts in counters.items()}


def _measure_step(model: Any, optimizer: Any, parameters: list[Any], batch: dict, torch: Any):
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    observed_loss = float(loss.detach().float().cpu())
    if not math.isfinite(observed_loss) or observed_loss < 0:
        raise RuntimeError("The synthetic causal-LM loss is not finite and nonnegative.")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    observed_norm = float(norm.detach().float().cpu())
    if not math.isfinite(observed_norm):
        raise RuntimeError("The LoRA gradient norm is not finite.")
    optimizer.step()
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    return {
        "seconds": positive_finite(elapsed, "Optimizer step time"),
        "loss": observed_loss,
        "gradient_norm": observed_norm,
        "finite_gradients": True,
    }


def collect_model(identity: dict, settings: dict, document: dict, publish: Any) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from amd_inference_opt.vllm_model_snapshot import capture_vllm_model_snapshot

    if not torch.cuda.is_available() or not torch.version.hip or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The training capacity probe requires a native BF16 ROCm GPU.")
    torch.cuda.set_device(0)
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    document["status"] = "verifying_cached_model"
    publish()
    model_path = snapshot_download(
        identity["repo_id"],
        revision=identity["revision"],
        token=False,
        cache_dir=os.environ.get("MACFIT_MODEL_CACHE"),
        local_files_only=True,
    )
    snapshot = capture_vllm_model_snapshot(
        model_path,
        model_id=identity["repo_id"],
        revision=identity["revision"],
        tokenizer_revision=identity["revision"],
    )
    document["model_snapshot_sha256"] = snapshot.snapshot_digest
    document["model_snapshot_file_count"] = snapshot.file_count
    document["model_snapshot_bytes"] = snapshot.total_bytes
    document["model_snapshot"] = snapshot.model_dump(mode="json", by_alias=True)
    document["status"] = "loading_base"
    publish()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    model_tokens = read_snapshot_config(snapshot).get("vocab_size")
    document["vocabulary"] = validate_vocabulary(
        tokenizer.vocab_size, len(tokenizer), model_tokens, tokenizer.eos_token_id
    )
    encoded = make_batch(
        tokenizer.encode(REFERENCE, add_special_tokens=False),
        tokenizer.encode(TARGET, add_special_tokens=False),
        tokenizer.eos_token_id,
        len(tokenizer),
        settings["sequence_length"],
    )
    document["input_ids_sha256"] = canonical_sha256(encoded["input_ids"])
    document["labels_sha256"] = canonical_sha256(encoded["labels"])
    document["attention_mask_sha256"] = canonical_sha256(encoded["attention_mask"])
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    if model.config.vocab_size != model_tokens:
        raise ValueError("The loaded model vocabulary differs from its pinned config.")
    document["vocabulary"] = validate_vocabulary(
        tokenizer.vocab_size,
        len(tokenizer),
        model.config.vocab_size,
        tokenizer.eos_token_id,
        input_embedding_rows=model.get_input_embeddings().weight.shape[0],
        output_embedding_rows=model.get_output_embeddings().weight.shape[0],
    )
    model = model.to("cuda:0")
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=list(TARGET_MODULES),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
        autocast_adapter_dtype=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    document["parameters"] = _parameter_counts(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    selected = next(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_B" in name
    )
    initial_selected = selected.detach().clone()
    batch = {
        name: torch.tensor([values], dtype=torch.long, device="cuda:0")
        for name, values in encoded.items()
    }
    torch.cuda.reset_peak_memory_stats(0)
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=0.01)
    document["runtime"] = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "rocm": str(torch.version.hip),
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "attention_configuration": "sdpa; selected fused kernel not identified",
        "local_files_only": True,
        "trust_remote_code": False,
        "source_sha256": SOURCE_SHA256,
    }
    document["status"] = "warmup_training"
    document["warmup_samples"] = []
    document["timed_samples"] = []
    publish()
    for _ in range(WARMUP_STEPS):
        document["warmup_samples"].append(_measure_step(model, optimizer, parameters, batch, torch))
        publish()
    document["status"] = "timed_training"
    for _ in range(TIMED_STEPS):
        document["timed_samples"].append(_measure_step(model, optimizer, parameters, batch, torch))
        publish()
    delta = float((selected.detach() - initial_selected).abs().max().float().cpu())
    if not math.isfinite(delta) or delta <= 0:
        raise RuntimeError("The real optimizer steps did not change the selected LoRA weights.")
    document["selected_lora_B_max_absolute_update"] = delta
    document["adapter_weights_changed"] = True
    document["optimizer_state_dtypes"] = _optimizer_state_dtypes(optimizer, torch)
    document["metrics"] = summarize_steps(document["timed_samples"])
    document["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
    document["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
    document["optimizer_step_calls"] = WARMUP_STEPS + TIMED_STEPS
    document["status"] = "succeeded"
    publish()
    optimizer.zero_grad(set_to_none=True)
    del optimizer, model, tokenizer, parameters, selected, initial_selected, batch
    torch.cuda.empty_cache()


def run_child(config: dict[str, Any]) -> int:
    from amd_inference_opt.resource_lock import exclusive_gpu_lock

    identity = config["identity"]
    if identity not in [BENCHMARK_MODELS[model] for model in MODEL_IDS]:
        raise ValueError("The child model must match the fixed pinned capacity registry.")
    settings = protocol(config["sequence_length"])
    output = real_path(Path(config["child_output"]))
    document = {
        "identity": identity,
        "status": "waiting_for_gpu",
        "adapter_published": False,
        "quality_evaluated": False,
        "training_finished_adapter": False,
    }

    def publish() -> None:
        write_atomic(output, document, max_bytes=MAX_CHILD_BYTES)

    publish()
    try:
        lock = Path(os.environ.get("MACFIT_GPU_LOCK", "/tmp/macfit-training-gpu.lock"))
        with exclusive_gpu_lock(lock):
            collect_model(identity, settings, document, publish)
        return 0
    except Exception as error:
        document["status"] = (
            "out_of_memory" if "out of memory" in str(error).lower() else "worker_failed"
        )
        document["error_type"] = type(error).__name__[:80]
        publish()
        return 2


def validate_completed(result: dict[str, Any], identity: dict[str, Any]) -> None:
    from amd_inference_opt.vllm_model_snapshot import VLLMModelSnapshotManifest

    if result.get("identity") != identity or result.get("status") != "succeeded":
        raise ValueError("The child did not complete its pinned model measurement.")
    if any(
        result.get(key) is not False
        for key in ("adapter_published", "quality_evaluated", "training_finished_adapter")
    ):
        raise ValueError("A capacity check cannot claim adapter publication or task quality.")
    if result.get("optimizer_step_calls") != WARMUP_STEPS + TIMED_STEPS:
        raise ValueError("The child must complete all five actual optimizer steps.")
    warmups = result.get("warmup_samples")
    if not isinstance(warmups, list) or len(warmups) != WARMUP_STEPS:
        raise ValueError("Both real warmup optimizer steps must be recorded.")
    # The same finite checks apply to the excluded warmups and timed steps.
    summarize_steps([warmups[0], warmups[1], warmups[0]])
    if result.get("metrics") != summarize_steps(result.get("timed_samples")):
        raise ValueError("The child metrics do not match its actual timing samples.")
    vocabulary = result.get("vocabulary")
    if (
        not isinstance(vocabulary, dict)
        or not {"input_embedding_rows", "output_embedding_rows"} <= set(vocabulary)
        or validate_vocabulary(
            vocabulary.get("tokenizer_base_tokens"),
            vocabulary.get("tokenizer_total_tokens"),
            vocabulary.get("model_config_tokens"),
            vocabulary.get("eos_token_id"),
            input_embedding_rows=vocabulary.get("input_embedding_rows"),
            output_embedding_rows=vocabulary.get("output_embedding_rows"),
        )
        != vocabulary
    ):
        raise ValueError("Actual tokenizer and both model embedding bounds must be recorded.")
    parameters = result.get("parameters")
    if (
        not isinstance(parameters, dict)
        or parameters.get("full_gpu_parameters") is not True
        or parameters.get("base_parameters_frozen") is not True
        or not 0 < parameters.get("trainable_parameters", 0) < parameters.get("total_parameters", 0)
        or parameters.get("trainable_parameters_by_dtype")
        != {"torch.float32": parameters["trainable_parameters"]}
        or parameters.get("frozen_parameters_by_dtype")
        != {"torch.bfloat16": parameters.get("frozen_parameters")}
        or parameters["total_parameters"]
        != parameters["trainable_parameters"] + parameters["frozen_parameters"]
    ):
        raise ValueError("The child must prove a BF16 frozen base and FP32 trainable LoRA.")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("rocm") or not runtime.get("gpu_name"):
        raise ValueError("An actual ROCm runtime identity is required.")
    if result.get("adapter_weights_changed") is not True:
        raise ValueError("The actual optimizer must change the selected LoRA weights.")
    for key in (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "selected_lora_B_max_absolute_update",
    ):
        positive_finite(result.get(key), key)
    for key in ("model_snapshot_sha256", "input_ids_sha256", "labels_sha256"):
        value = result.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Snapshot and synthetic input identities must be SHA-256 values.")
    snapshot = VLLMModelSnapshotManifest.model_validate(result.get("model_snapshot"))
    if (
        snapshot.model_id != identity["repo_id"]
        or snapshot.revision != identity["revision"]
        or snapshot.tokenizer_revision != identity["revision"]
        or snapshot.snapshot_digest != result["model_snapshot_sha256"]
        or snapshot.file_count != result.get("model_snapshot_file_count")
        or snapshot.total_bytes != result.get("model_snapshot_bytes")
    ):
        raise ValueError("The full cached model inventory must match the pinned snapshot identity.")
    states = result.get("optimizer_state_dtypes")
    if not isinstance(states, dict) or not {"step", "exp_avg", "exp_avg_sq"} <= set(states):
        raise ValueError("Actual AdamW state dtypes must be recorded.")


def _read_child(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_CHILD_BYTES:
        raise ValueError("The capacity child evidence exceeded its bound.")
    return read_json(path)


def run_probe(
    models: list[str],
    output: Path,
    *,
    sequence_length: int = 128,
    walltime_seconds: int = 1200,
    model_timeout_seconds: int = 600,
    stop_at: str | None = None,
    worker_command: list[str] | None = None,
) -> dict[str, Any]:
    plan = validate_plan(models, sequence_length, walltime_seconds, model_timeout_seconds, stop_at)
    output = real_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a fresh output filename; existing evidence is not replaced.")
    started = time.monotonic()
    deadline = started + walltime_seconds
    if plan["stop_at"]:
        remaining = (parse_stop_at(plan["stop_at"]) - datetime.now(UTC)).total_seconds()
        deadline = min(deadline, started + max(0, remaining))
    work_deadline = deadline - 8
    document = {
        "schema": SCHEMA,
        "status": "running",
        "started_at": utc_now(),
        "protocol": plan,
        "models": [],
        "limitations": [
            "Five updates of one synthetic batch are a training capacity microbenchmark, "
            "not a completed fine-tune, dataset-quality test or useful adapter.",
            "The website trainer uses rank 16; this rank-8 short-sequence probe is smaller.",
            "Warmup steps update weights but are excluded from timing statistics.",
            "Allocated/reserved peaks include this model, LoRA and optimizer; "
            "they exclude other processes and non-Torch allocations.",
            "No weights, adapters or optimizer checkpoints are saved.",
        ],
    }

    def publish() -> None:
        write_atomic(output, document, max_bytes=MAX_EVIDENCE_BYTES)

    publish()
    try:
        for identity in plan["models"]:
            row = {"identity": identity, "status": "pending"}
            document["models"].append(row)
            if time.monotonic() >= work_deadline:
                row["status"] = "skipped_deadline"
                publish()
                continue
            with tempfile.TemporaryDirectory(
                prefix="lora-capacity-", dir=output.parent
            ) as temporary:
                scratch = Path(temporary)
                config, child_output = scratch / "config.json", scratch / "child.json"
                write_atomic(
                    config,
                    {
                        "identity": identity,
                        "sequence_length": sequence_length,
                        "child_output": str(child_output),
                    },
                    max_bytes=MAX_CHILD_BYTES,
                )
                command = worker_command or [sys.executable, "-m", MODULE, "--child-config"]
                process = subprocess.Popen(
                    [*command, str(config)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                model_deadline = min(work_deadline, time.monotonic() + model_timeout_seconds)
                cleanup = _ProcessGroupCleanup(process)
                timed_out = False

                def clean_process(
                    cleanup: _ProcessGroupCleanup = cleanup, row: dict[str, Any] = row
                ) -> None:
                    try:
                        cleanup()
                    except Exception as cleanup_error:
                        row["cleanup_error_type"] = type(cleanup_error).__name__
                        raise

                try:
                    while process.poll() is None:
                        if child_output.exists():
                            row.update(_read_child(child_output))
                            publish()
                        if time.monotonic() >= model_deadline:
                            timed_out = True
                            clean_process()
                            break
                        time.sleep(0.2)
                    clean_process()
                    if child_output.exists():
                        row.update(_read_child(child_output))
                    row["exit_code"] = process.wait(timeout=5)
                    if timed_out:
                        row["status"] = "timed_out"
                    elif row.get("status") != "succeeded" or row["exit_code"] != 0:
                        row["status"] = "worker_failed"
                    else:
                        validate_completed(row, identity)
                except BaseException as error:
                    row["status"] = (
                        "cleanup_failed"
                        if "cleanup_error_type" in row
                        else "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "supervisor_failed"
                    )
                    try:
                        clean_process()
                    except Exception as cleanup_error:
                        error.add_note(
                            "Process-group cleanup failed: " + type(cleanup_error).__name__
                        )
                        raise error from cleanup_error
                    raise
                finally:
                    try:
                        clean_process()
                    finally:
                        publish()
        document["status"] = (
            "succeeded"
            if all(row["status"] == "succeeded" for row in document["models"])
            else "incomplete"
        )
    except BaseException as error:
        document["status"] = (
            "cleanup_failed"
            if any(row.get("status") == "cleanup_failed" for row in document["models"])
            else "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "supervisor_failed"
        )
        raise
    finally:
        document["finished_at"] = utc_now()
        document["elapsed_seconds"] = time.monotonic() - started
        publish()
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=list(MODEL_IDS))
    parser.add_argument("--sequence-length", type=int, choices=SEQUENCE_LENGTHS, default=128)
    parser.add_argument("--walltime-seconds", type=int, default=1200)
    parser.add_argument("--model-timeout-seconds", type=int, default=600)
    parser.add_argument("--stop-at", help="Absolute ISO-8601 deadline including timezone.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--child-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.child_config:
        return run_child(read_json(args.child_config))

    def cancel(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    try:
        if args.validate_only:
            print(
                json.dumps(
                    validate_plan(
                        args.models,
                        args.sequence_length,
                        args.walltime_seconds,
                        args.model_timeout_seconds,
                        args.stop_at,
                    ),
                    indent=2,
                )
            )
            return 0
        if args.output is None:
            parser.error("--output is required for a real capacity probe.")
        result = run_probe(
            args.models,
            args.output,
            sequence_length=args.sequence_length,
            walltime_seconds=args.walltime_seconds,
            model_timeout_seconds=args.model_timeout_seconds,
            stop_at=args.stop_at,
        )
        return 0 if result["status"] == "succeeded" else 2
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
