"""A small real ROCm LoRA trainer with explicit assistant-only loss and evaluation."""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any

from .artifacts import training_source_identity, write_json
from .data import prepare_training_data, tensor_batch
from .evaluation import comparison, evaluate_model


def load_runtime(job: dict[str, Any], job_dir: Path, emit: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Lazy imports keep config validation and API startup CPU-only."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from amd_inference_opt.vllm_environment import capture_vllm_environment
    from amd_inference_opt.vllm_model_snapshot import capture_vllm_model_snapshot

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("This worker requires an available ROCm GPU.")
    base = job["base_model"]
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    emit("loading_model", message="Loading the pinned base model.")
    cache = os.environ.get("MACFIT_MODEL_CACHE")
    path = snapshot_download(
        base["repo_id"],
        revision=base["revision"],
        cache_dir=cache,
        token=False,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.tiktoken",
            "*.jinja",
            "LICENSE*",
        ],
    )
    emit("verifying_model", message="Hashing model files and runtime packages.")
    snapshot = capture_vllm_model_snapshot(
        path,
        model_id=base["repo_id"],
        revision=base["revision"],
        tokenizer_revision=base["revision"],
    )
    environment = capture_vllm_environment(
        required_distributions=("torch", "transformers", "peft", "safetensors"),
        optional_distributions=("accelerate", "amdsmi", "triton", "pytorch-triton-rocm"),
    )
    # Complete evidence stays in the private job directory, never in public logs.
    write_json(job_dir / "model-snapshot.json", snapshot.model_dump(mode="json"))
    write_json(job_dir / "runtime-environment.json", environment.model_dump(mode="json"))
    training_source = training_source_identity()
    write_json(job_dir / "training-source.json", training_source)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=False,
        local_files_only=True,
    ).to("cuda:0")
    packages = {item.name: item.version for item in environment.distributions}
    provenance = {
        "model_snapshot_sha256": snapshot.snapshot_digest,
        "environment_sha256": environment.identity_sha256,
        "training_framework_sha256": training_source["sha256"],
        "packages": packages,
        "python": environment.python_version,
        "gpu": torch.cuda.get_device_name(0),
        "rocm": torch.version.hip,
        "dtype": "bfloat16",
        "seed": 42,
        "trust_remote_code": False,
        "model_revision": base["revision"],
        "tokenizer_revision": base["revision"],
    }
    return model, tokenizer, provenance


def run_training(job: dict[str, Any], job_dir: Path, emit: Any) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    started = time.monotonic()
    model, tokenizer, provenance = load_runtime(job, job_dir, emit)
    settings = job["training_config"]
    rows = prepare_training_data(tokenizer, job)
    emit("evaluating_base", completed=0, total=len(job["evaluation"]), unit="questions")
    before = evaluate_model(model, tokenizer, job, stage="evaluating_base", emit=emit)
    lora = LoraConfig(
        r=settings["rank"],
        lora_alpha=settings["alpha"],
        lora_dropout=settings["dropout"],
        target_modules=settings["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable, total_parameters = model.get_nb_trainable_parameters()
    if trainable <= 0 or trainable >= total_parameters:
        raise RuntimeError("The model did not create a restricted trainable LoRA adapter.")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings["learning_rate"],
        weight_decay=0.01,
    )
    accumulation = settings["gradient_accumulation_steps"]
    total_steps = min(
        settings["max_steps"], settings["epochs"] * math.ceil(len(rows) / accumulation)
    )
    rng = random.Random(settings["seed"])
    steps, loss_sum, loss_tokens = 0, 0.0, 0
    emit("training", completed=0, total=total_steps, unit="optimizer_steps")
    model.train()
    for _epoch in range(settings["epochs"]):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for offset in range(0, len(order), accumulation):
            group = order[offset : offset + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                encoded = rows[index]
                batch = tensor_batch(encoded, model.device)
                loss = model(**batch).loss
                measured = float(loss.detach().float().cpu())
                if not math.isfinite(measured):
                    raise RuntimeError("Training loss is not finite; adapter was not published.")
                (loss / len(group)).backward()
                count = sum(label != -100 for label in encoded["labels"][1:])
                loss_sum += measured * count
                loss_tokens += count
                del batch, loss
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                settings["max_grad_norm"],
                error_if_nonfinite=True,
            )
            if not torch.isfinite(norm):
                raise RuntimeError("Training gradients are not finite.")
            optimizer.step()
            steps += 1
            emit("training", completed=steps, total=total_steps, unit="optimizer_steps")
            if steps >= total_steps:
                break
        if steps >= total_steps:
            break
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    emit("evaluating_adapter", completed=0, total=len(job["evaluation"]), unit="questions")
    after = evaluate_model(model, tokenizer, job, stage="evaluating_adapter", emit=emit)
    emit("saving_adapter", message="Saving adapter weights and measured evaluation results.")
    adapter_dir = job_dir / "adapter"
    adapter_dir.mkdir(exist_ok=False)
    # PEFT must reference a portable HF model identity, not the worker's cache path.
    model.peft_config["default"].base_model_name_or_path = job["base_model"]["repo_id"]
    model.peft_config["default"].revision = job["base_model"]["revision"]
    model.save_pretrained(adapter_dir, safe_serialization=True)
    torch.cuda.synchronize()
    result = {
        "method": "lora_sft",
        "base_model": job["base_model"],
        "training": {
            "steps": steps,
            "loss": loss_sum / loss_tokens,
            "supervised_tokens": loss_tokens,
            "assistant_only_loss": True,
            "trainable_parameters": trainable,
            "total_parameters": total_parameters,
            "elapsed_seconds": time.monotonic() - started,
            "epochs_requested": settings["epochs"],
            "rows": len(rows),
        },
        "evaluation": comparison(before, after),
        "provenance": provenance,
        "artifact_format": "peft_lora_adapter",
        "standalone_model": False,
    }
    del model, tokenizer
    torch.cuda.empty_cache()
    return result
