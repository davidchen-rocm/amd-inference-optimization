"""Bounded CPU portability check of a verified LoRA job and its merged export.

Merged weights remain private and regeneratable; only compact JSON evidence is
intended for backup. No GPU calls, model downloads or website submissions occur.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import canonical_sha256, describe_artifact
from .benchmark import _ProcessGroupCleanup, parse_stop_at, write_atomic
from .config import validate_job_input
from .data import evaluation_messages, prompt_ids, tensor_batch
from .experiments import parse_strict_json
from .export import export_merged, extract_adapter, read_json, real_path, verified_job

MODULE = "macfit_training.export_verification"
MAX_EVIDENCE_BYTES = 1024**2 - 1
MAX_CHILD_BYTES = 128 * 1024
QUESTION_COUNT = 6
MAX_NEW_TOKENS = 96
CPU_THREADS = 4
SOURCE_KEYS = (
    "base_model",
    "source_input_sha256",
    "source_adapter_sha256",
    "source_model_snapshot_sha256",
    "source_result_sha256",
    "source_artifact_manifest_sha256",
    "source_model_snapshot_manifest_sha256",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def source_identity(job_dir: Path, *, verify_bytes: bool) -> tuple[dict, dict]:
    """Bind declared source metadata; the bounded child verifies the actual bytes."""
    if verify_bytes:
        job, result = verified_job(job_dir)
    else:
        job = validate_job_input("training", read_json(job_dir / "input.json"))
        result = read_json(job_dir / "result.json")
    manifest = read_json(job_dir / "artifacts/manifest.json")
    snapshot = read_json(job_dir / "model-snapshot.json")
    declared = result.get("artifacts")
    if not isinstance(declared, list):
        raise ValueError("The completed source needs a declared artifact inventory.")
    adapters = [
        row for row in declared if isinstance(row, dict) and row.get("type") == "lora_adapter"
    ]
    provenance = result.get("provenance")
    if (
        result.get("method") != "lora_sft"
        or result.get("base_model") != job["base_model"]
        or manifest.get("schema") != "macfit-training-artifacts.v1"
        or manifest.get("input_sha256") != canonical_sha256(job)
        or manifest.get("base_model") != job["base_model"]
        or not isinstance(provenance, dict)
        or provenance != manifest.get("provenance")
        or len(adapters) != 1
        or adapters[0].get("name") != "adapter.tar.gz"
    ):
        raise ValueError("The completed source metadata and artifact provenance disagree.")
    digest = provenance.get("model_snapshot_sha256")
    adapter_digest = adapters[0].get("sha256")
    if (
        not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or not isinstance(adapter_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", adapter_digest)
        or snapshot.get("snapshot_digest") != digest
        or snapshot.get("model_id") != job["base_model"]["repo_id"]
        or snapshot.get("revision") != job["base_model"]["revision"]
        or snapshot.get("tokenizer_revision") != job["base_model"]["revision"]
    ):
        raise ValueError("The source must declare the exact pinned model and adapter identities.")
    return job, {
        "base_model": job["base_model"],
        "source_input_sha256": canonical_sha256(job),
        "source_adapter_sha256": adapter_digest,
        "source_model_snapshot_sha256": digest,
        "source_result_sha256": canonical_sha256(result),
        "source_artifact_manifest_sha256": canonical_sha256(manifest),
        "source_model_snapshot_manifest_sha256": canonical_sha256(snapshot),
    }


def json_value(text: str) -> tuple[bool, Any]:
    """Strict typed JSON, including arrays/scalars, without accepting duplicate keys."""
    wrapped = parse_strict_json('{"answer":' + text + "}")
    if wrapped is None or set(wrapped) != {"answer"}:
        return False, None
    return True, wrapped["answer"]


def expected_score(expected: str, prediction: str) -> dict[str, Any]:
    expected_json, expected_value = json_value(expected)
    predicted_json, predicted_value = json_value(prediction)
    if expected_json:
        return {
            "format": "json",
            "strict_json": predicted_json,
            "exact_expected": predicted_json
            and canonical_sha256(predicted_value) == canonical_sha256(expected_value),
        }

    def normalize(value: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()

    return {"format": "text", "exact_expected": normalize(expected) == normalize(prediction)}


def select_questions(job: dict[str, Any], ids: list[str] | None = None) -> list[dict[str, Any]]:
    evaluation = job["evaluation"]
    if len(evaluation) < QUESTION_COUNT:
        raise ValueError("The portability check requires at least six held-out questions.")
    if ids is not None:
        if (
            not isinstance(ids, list)
            or len(ids) != QUESTION_COUNT
            or any(not isinstance(value, str) for value in ids)
            or len(set(ids)) != QUESTION_COUNT
        ):
            raise ValueError("Select exactly six distinct held-out question IDs.")
        by_id = {row["id"]: row for row in evaluation}
        if any(value not in by_id for value in ids):
            raise ValueError("Every selected ID must belong to this job's held-out evaluation.")
        return [dict(by_id[value]) for value in ids]

    # Preserve original order and cover distinct user-provided JSON topics first.
    # No benchmark ID, expected answer or fixture-specific ground truth is hardcoded.
    selected, topics = [], set()
    for row in evaluation:
        parsed, value = json_value(row["expected"])
        topic = value.get("topic") if parsed and isinstance(value, dict) else None
        if isinstance(topic, str) and topic not in topics:
            selected.append(dict(row))
            topics.add(topic)
        if len(selected) == QUESTION_COUNT:
            return selected
    chosen_ids = {row["id"] for row in selected}

    def language(row: dict[str, Any]) -> str:
        parsed, value = json_value(row["expected"])
        label = value.get("language") if parsed and isinstance(value, dict) else None
        if isinstance(label, str):
            return label
        return "zh" if re.search(r"[\u3400-\u9fff]", row["question"]) else "other"

    while len(selected) < QUESTION_COUNT:
        counts = {
            language(row): sum(language(other) == language(row) for other in selected)
            for row in evaluation
        }
        candidates = [row for row in evaluation if row["id"] not in chosen_ids]
        row = min(candidates, key=lambda candidate: counts[language(candidate)])
        selected.append(dict(row))
        chosen_ids.add(row["id"])
    return selected


def verify_export(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check every saved merged/tokenizer file against the generated export inventory."""
    directory = real_path(directory)
    if (directory / "EXPORT_INCOMPLETE.json").exists():
        raise ValueError("The merged export is incomplete.")
    recorded = read_json(directory / "export-manifest.json")
    if recorded != manifest or recorded.get("schema") != "macfit-merged-export.v1":
        raise ValueError("The saved export manifest does not match this completed export.")
    files = recorded.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("The export must contain a registered file inventory.")
    names = [row.get("name") for row in files if isinstance(row, dict)]
    if (
        len(names) != len(files)
        or any(not isinstance(name, str) or Path(name).name != name for name in names)
        or len(set(names)) != len(names)
        or {path.name for path in directory.iterdir()} != {*names, "export-manifest.json"}
    ):
        raise ValueError("The merged export has missing, unexpected or unsafe files.")
    for row in files:
        path = real_path(directory / row["name"])
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Merged files must be private regular files without hard links.")
        if describe_artifact(path, "merged_model_file") != row:
            raise ValueError("A merged model or tokenizer file failed its hash check.")
    return describe_artifact(directory / "export-manifest.json", "export_manifest")


def load_cpu_model(directory: Path) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        directory,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation="sdpa",
    ).eval()
    verify_cpu_parameters(model)
    return model, tokenizer


def verify_cpu_parameters(model: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        if parameter.device.type != "cpu":
            raise RuntimeError("The portability check must keep every parameter on the CPU.")
        dtype = str(parameter.dtype)
        counts[dtype] = counts.get(dtype, 0) + parameter.numel()
    if not counts:
        raise RuntimeError("The portability model must contain actual CPU parameters.")
    return counts


def generate_cpu(model: Any, tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Record actual greedy tokens and finite raw logits at the prompt/output boundaries."""
    import torch

    ids = prompt_ids(tokenizer, messages)
    if len(ids) > 2048:
        raise ValueError("A portability prompt exceeds the trained context bound.")
    batch = tensor_batch({"input_ids": ids, "attention_mask": [1] * len(ids)}, "cpu")
    started = time.monotonic()
    with torch.inference_mode():
        initial = model(**batch, use_cache=False, logits_to_keep=1)
        finite_prompt = bool(torch.isfinite(initial.logits).all().item())
        del initial
        if not finite_prompt:
            raise RuntimeError("The CPU model produced nonfinite prompt logits.")
        complete = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
        generated = complete[len(ids) :]
        final = model(
            input_ids=complete.unsqueeze(0),
            attention_mask=torch.ones_like(complete).unsqueeze(0),
            use_cache=False,
            logits_to_keep=1,
        )
        finite_output = bool(torch.isfinite(final.logits).all().item())
        if not finite_output:
            raise RuntimeError("The CPU model produced nonfinite output-boundary logits.")
        output_ids = generated.cpu().tolist()
        eos = tokenizer.eos_token_id
        return {
            "text": tokenizer.decode(generated, skip_special_tokens=True),
            "input_tokens": len(ids),
            "input_ids_sha256": canonical_sha256(ids),
            "output_tokens": len(output_ids),
            "output_ids": output_ids,
            "output_ids_sha256": canonical_sha256(output_ids),
            "elapsed_seconds": time.monotonic() - started,
            "finite_logits": True,
            "finite_checks": "prompt_last_token_and_generated_sequence_last_token",
            "stopped_by_limit": len(output_ids) >= MAX_NEW_TOKENS and output_ids[-1] != eos,
        }


def compare_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        len(samples) != QUESTION_COUNT
        or len({sample["id"] for sample in samples}) != QUESTION_COUNT
    ):
        raise ValueError("Exactly six distinct output comparisons are required.")
    for sample in samples:
        for stage in ("peft_cpu", "merged_cpu"):
            prediction = sample.get(stage)
            if (
                not isinstance(prediction, dict)
                or not isinstance(prediction.get("text"), str)
                or prediction.get("finite_logits") is not True
                or type(prediction.get("stopped_by_limit")) is not bool
                or not isinstance(prediction.get("output_ids"), list)
            ):
                raise ValueError(
                    "Each branch must contain actual CPU tokens, text and finite checks."
                )
            ids = prediction["output_ids"]
            if (
                not 1 <= len(ids) <= MAX_NEW_TOKENS
                or any(type(token) is not int or token < 0 for token in ids)
                or prediction.get("output_tokens") != len(ids)
                or prediction.get("output_ids_sha256") != canonical_sha256(ids)
                or type(prediction.get("input_tokens")) is not int
                or not 1 <= prediction["input_tokens"] <= 2048
                or not isinstance(prediction.get("input_ids_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", prediction["input_ids_sha256"])
                or prediction.get("finite_checks")
                != "prompt_last_token_and_generated_sequence_last_token"
                or not isinstance(prediction.get("elapsed_seconds"), (float, int))
                or not math.isfinite(prediction["elapsed_seconds"])
                or prediction["elapsed_seconds"] < 0
            ):
                raise ValueError("Raw CPU tokens, timing and their identities must be consistent.")
            score = expected_score(sample["expected"], prediction["text"])
            score["passed"] = bool(score["exact_expected"] and not prediction["stopped_by_limit"])
            if prediction.get("score") != score:
                raise ValueError("Recorded scores must match the actual branch outputs.")
        if sample["peft_cpu"]["input_ids_sha256"] != sample["merged_cpu"]["input_ids_sha256"]:
            raise ValueError("The exported tokenizer changed the selected prompt tokens.")
        sample["same_output_token_ids"] = (
            sample["peft_cpu"]["output_ids"] == sample["merged_cpu"]["output_ids"]
        )
        sample["same_output_text"] = sample["peft_cpu"]["text"] == sample["merged_cpu"]["text"]
    return {
        "questions": len(samples),
        **{
            stage + "_passes": sum(sample[stage]["score"]["passed"] for sample in samples)
            for stage in ("peft_cpu", "merged_cpu")
        },
        "identical_token_output_count": sum(sample["same_output_token_ids"] for sample in samples),
        "identical_text_output_count": sum(sample["same_output_text"] for sample in samples),
        "changed_case_ids": [
            sample["id"] for sample in samples if not sample["same_output_token_ids"]
        ],
    }


def run_child(config: dict[str, Any]) -> int:
    document: dict[str, Any] = {"status": "verifying_source_job", "samples": []}
    evidence = real_path(Path(config["child_output"]))

    def publish() -> None:
        write_atomic(evidence, document, max_bytes=MAX_CHILD_BYTES)

    publish()
    try:
        job_dir, merged_dir = (
            real_path(Path(config["job_dir"])),
            real_path(Path(config["merged_output"])),
        )
        job, identity = source_identity(job_dir, verify_bytes=True)
        if any(identity[key] != config[key] for key in SOURCE_KEYS):
            raise ValueError(
                "The verified source job differs from the requested source identities."
            )
        document.update(identity)
        if config.get("inventory_only"):
            document["status"] = "verifying_final_export_inventory"
            publish()
            manifest = config["export_manifest"]
            document["export_manifest_file"] = verify_export(merged_dir, manifest)
            document["export_manifest_sha256"] = canonical_sha256(manifest)
            document["status"] = "succeeded"
            publish()
            return 0
        rows = select_questions(job, config.get("question_ids"))
        document.update(
            selected_questions_sha256=canonical_sha256(rows),
            selected_question_ids=[row["id"] for row in rows],
        )
        base_override = (
            real_path(Path(config["base_model_dir"])) if config.get("base_model_dir") else None
        )
        document["status"] = "exporting_on_cpu"
        publish()
        # Set bounds before export_merged imports/loads Torch models.
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            os.environ[name] = ""
        import torch
        from peft import PeftModel

        torch.set_num_threads(CPU_THREADS)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        exported = export_merged(job_dir, merged_dir, base_model_dir=base_override)
        if (
            exported.get("source_input_sha256") != document["source_input_sha256"]
            or exported.get("source_adapter_sha256") != document["source_adapter_sha256"]
            or exported.get("source_model_snapshot_sha256")
            != document["source_model_snapshot_sha256"]
        ):
            raise ValueError("The export source identities changed during verification.")
        document["export_manifest"] = exported
        document["export_manifest_file"] = verify_export(merged_dir, exported)
        base_dir = base_override or real_path(
            Path(read_json(job_dir / "model-snapshot.json")["root"])
        )
        document["samples"] = [
            {key: row[key] for key in ("id", "question", "expected")} for row in rows
        ]
        document["runtime"] = {
            "device": "cpu",
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "base_dtype": "bfloat16",
            "cpu_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "gpu_visibility": {
                name: os.environ[name]
                for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
            },
            "local_files_only": True,
            "trust_remote_code": False,
        }
        for stage in ("peft_cpu", "merged_cpu"):
            document["status"] = "loading_" + stage
            publish()
            model, tokenizer = load_cpu_model(base_dir if stage == "peft_cpu" else merged_dir)
            with tempfile.TemporaryDirectory(
                prefix="cpu-verified-adapter-", dir=evidence.parent
            ) as temporary:
                if stage == "peft_cpu":
                    adapter = Path(temporary)
                    extract_adapter(
                        job_dir / "artifacts/adapter.tar.gz", adapter, job["base_model"]
                    )
                    model = PeftModel.from_pretrained(
                        model,
                        adapter,
                        local_files_only=True,
                        is_trainable=False,
                        torch_device="cpu",
                        device_map={"": "cpu"},
                    ).eval()
                document.setdefault("parameter_dtypes", {})[stage] = verify_cpu_parameters(model)
                document["status"] = "evaluating_" + stage
                publish()
                for row, sample in zip(rows, document["samples"], strict=True):
                    prediction = generate_cpu(model, tokenizer, evaluation_messages(row, job))
                    score = expected_score(row["expected"], prediction["text"])
                    score["passed"] = bool(
                        score["exact_expected"] and not prediction["stopped_by_limit"]
                    )
                    prediction["score"] = score
                    sample[stage] = prediction
                    publish()
            del model, tokenizer
            gc.collect()
        document["metrics"] = compare_samples(document["samples"])
        document["status"] = "succeeded"
        publish()
        return 0
    except Exception as error:
        document["status"] = "worker_failed"
        document["error_type"] = type(error).__name__[:80]
        publish()
        return 2


def validate_plan(
    job_dir: Path,
    merged_output: Path,
    output: Path,
    *,
    base_model_dir: Path | None = None,
    question_ids: list[str] | None = None,
    walltime_seconds: int = 600,
    stop_at: str | None = None,
) -> dict[str, Any]:
    if type(walltime_seconds) is not int or not 10 <= walltime_seconds <= 600:
        raise ValueError("CPU portability walltime must be 10 to 600 seconds including cleanup.")
    job_dir, merged_output, output = real_path(job_dir), real_path(merged_output), real_path(output)
    base_model_dir = real_path(base_model_dir) if base_model_dir else None
    if not job_dir.is_dir() or not merged_output.parent.is_dir() or not output.parent.is_dir():
        raise ValueError("The source job and both output parent directories must exist.")
    if merged_output.exists() or output.exists():
        raise ValueError("Both evidence and merged export destinations must be fresh.")
    if (
        job_dir == merged_output
        or job_dir in merged_output.parents
        or merged_output == output
        or merged_output in output.parents
        or job_dir in output.parents
    ):
        raise ValueError("Merged weights, evidence and original jobs must be separate.")
    # Only bounded JSON is read here. Potentially large artifact/snapshot byte checks
    # run in the supervised child so they share the hard walltime and cleanup budget.
    job, identity = source_identity(job_dir, verify_bytes=False)
    rows = select_questions(job, question_ids)
    return {
        "job_dir": str(job_dir),
        "merged_output": str(merged_output),
        "output": str(output),
        "base_model_dir": str(base_model_dir) if base_model_dir else None,
        "question_ids": [row["id"] for row in rows],
        "selected_questions": rows,
        "selected_questions_sha256": canonical_sha256(rows),
        **identity,
        "walltime_seconds": walltime_seconds,
        "stop_at": parse_stop_at(stop_at).isoformat() if stop_at else None,
    }


def validate_completed(result: dict[str, Any], plan: dict[str, Any]) -> None:
    if (
        result.get("status") != "succeeded"
        or result.get("source_input_sha256") != plan["source_input_sha256"]
        or result.get("selected_questions_sha256") != plan["selected_questions_sha256"]
        or result.get("selected_question_ids") != plan["question_ids"]
    ):
        raise ValueError(
            "The child did not verify the requested source input and selected questions."
        )
    if any(result.get(key) != plan[key] for key in SOURCE_KEYS):
        raise ValueError("The completed comparison is not bound to the original source artifacts.")
    manifest = result.get("export_manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "macfit-merged-export.v1"
        or manifest.get("format") != "transformers_safetensors"
        or manifest.get("dtype") != "bfloat16"
        or any(manifest.get(key) != plan[key] for key in SOURCE_KEYS[:4])
    ):
        raise ValueError("The merged export must preserve the original pinned source identities.")
    samples = result.get("samples")
    if not isinstance(samples, list) or [
        {key: sample.get(key) for key in ("id", "question", "expected")}
        for sample in samples
        if isinstance(sample, dict)
    ] != [
        {key: row[key] for key in ("id", "question", "expected")}
        for row in plan["selected_questions"]
    ]:
        raise ValueError("Actual comparison questions and ground truth must match the source job.")
    if not isinstance(samples, list) or result.get("metrics") != compare_samples(samples):
        raise ValueError("A successful portability check needs all six actual output comparisons.")
    runtime = result.get("runtime", {})
    if (
        runtime.get("device") != "cpu"
        or manifest.get("device") != "cpu"
        or runtime.get("cpu_threads") != CPU_THREADS
        or runtime.get("cpu_interop_threads") != 1
        or runtime.get("local_files_only") is not True
        or runtime.get("trust_remote_code") is not False
        or runtime.get("gpu_visibility")
        != {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}
    ):
        raise ValueError("Both export and verification must use the CPU.")
    counts = result.get("parameter_dtypes")
    if (
        not isinstance(counts, dict)
        or set(counts) != {"peft_cpu", "merged_cpu"}
        or any(
            not isinstance(stage, dict)
            or not stage
            or any(
                not isinstance(dtype, str) or type(count) is not int or count <= 0
                for dtype, count in stage.items()
            )
            for stage in counts.values()
        )
    ):
        raise ValueError("The actual parameter dtype inventories are required for both branches.")


def validate_inventory(result: dict, comparison: dict, plan: dict) -> None:
    if (
        result.get("status") != "succeeded"
        or any(result.get(key) != plan[key] for key in SOURCE_KEYS)
        or result.get("export_manifest_sha256") != canonical_sha256(comparison["export_manifest"])
        or result.get("export_manifest_file") != comparison.get("export_manifest_file")
    ):
        raise ValueError(
            "The final export inventory check did not confirm the completed comparison."
        )


def run_verification(
    job_dir: Path,
    merged_output: Path,
    output: Path,
    *,
    base_model_dir: Path | None = None,
    question_ids: list[str] | None = None,
    walltime_seconds: int = 600,
    stop_at: str | None = None,
    worker_command: list[str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    plan = validate_plan(
        job_dir,
        merged_output,
        output,
        base_model_dir=base_model_dir,
        question_ids=question_ids,
        walltime_seconds=walltime_seconds,
        stop_at=stop_at,
    )
    deadline = started + walltime_seconds
    if plan["stop_at"]:
        remaining = (parse_stop_at(plan["stop_at"]) - datetime.now(UTC)).total_seconds()
        deadline = min(deadline, time.monotonic() + max(0, remaining))
    output = real_path(output)
    document = {
        "schema": "macfit-cpu-export-verification.v1",
        "status": "running",
        "started_at": utc_now(),
        "protocol": {
            "device": "cpu",
            "base_dtype": "bfloat16",
            "cpu_threads": CPU_THREADS,
            "cpu_interop_threads": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "questions": QUESTION_COUNT,
            "greedy": True,
            "enable_thinking": False,
            "walltime_seconds": walltime_seconds,
            "stop_at": plan["stop_at"],
            "question_ids": plan["question_ids"],
            "source_sha256": describe_artifact(Path(__file__), "source")["sha256"],
            "strict_scoring_helpers_source_sha256": describe_artifact(
                Path(parse_strict_json.__code__.co_filename), "source"
            )["sha256"],
            "export_helpers_source_sha256": describe_artifact(
                Path(export_merged.__code__.co_filename), "source"
            )["sha256"],
            "prompt_helpers_source_sha256": describe_artifact(
                Path(evaluation_messages.__code__.co_filename), "source"
            )["sha256"],
            "process_supervisor_helpers_source_sha256": describe_artifact(
                Path(_ProcessGroupCleanup.__call__.__code__.co_filename), "source"
            )["sha256"],
        },
        "source": {key: plan[key] for key in SOURCE_KEYS},
        "limitations": [
            "Six questions are a portability diagnostic, not a general accuracy benchmark.",
            "BF16 merge rounding and CPU arithmetic can change tokens; differences are preserved.",
            "Finite raw logits are checked only at prompt and generated-sequence boundaries.",
            "Merged weights remain private outside backup and can be regenerated "
            "from the adapter and pinned base.",
            "This exports Transformers safetensors, not GGUF, MLX, a Mac app or hosted model.",
        ],
    }

    def publish() -> None:
        write_atomic(output, document, max_bytes=MAX_EVIDENCE_BYTES)

    publish()
    if time.monotonic() >= deadline - 8:
        document.update(status="skipped_deadline", finished_at=utc_now())
        publish()
        return document

    def supervise(config_value: dict, scratch: Path, key: str) -> bool:
        config, child_output = scratch / (key + "-config.json"), scratch / (key + ".json")
        write_atomic(config, {**config_value, "child_output": str(child_output)})
        command = worker_command or [sys.executable, "-m", MODULE, "--child-config"]
        environment = dict(os.environ)
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            environment[name] = ""
        environment.update(OMP_NUM_THREADS=str(CPU_THREADS), MKL_NUM_THREADS=str(CPU_THREADS))
        process = subprocess.Popen(
            [*command, str(config)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=environment,
        )
        cleanup = _ProcessGroupCleanup(process)
        phase = document.setdefault("phases", {}).setdefault(key, {})

        def clean_process() -> None:
            try:
                cleanup()
            except Exception as error:
                # Never persist exception text, process arguments or credentials.
                phase["cleanup_error_type"] = type(error).__name__[:80]
                document["cleanup_error_type"] = phase["cleanup_error_type"]
                raise

        def read_child() -> None:
            if child_output.exists():
                if child_output.stat().st_size > MAX_CHILD_BYTES:
                    raise ValueError("CPU child evidence exceeded its bound.")
                document[key] = read_json(child_output)

        timed_out = False
        try:
            while process.poll() is None:
                read_child()
                publish()
                if time.monotonic() >= deadline - 8:
                    timed_out = True
                    break
                time.sleep(0.2)
            # Reap the whole group before accepting any final evidence or starting
            # the independent inventory verifier, even if its leader already exited.
            clean_process()
            read_child()
            phase["exit_code"] = process.wait(timeout=1)
            if timed_out:
                document["status"] = "timed_out"
                return False
            if document.get(key, {}).get("status") != "succeeded" or phase["exit_code"] != 0:
                document["status"] = "worker_failed"
                return False
            return True
        except BaseException as error:
            try:
                clean_process()
            except Exception as cleanup_error:
                error.add_note("Process-group cleanup failed: " + type(cleanup_error).__name__)
                raise error from cleanup_error
            raise
        finally:
            # _ProcessGroupCleanup marks attempted before signalling: no reused
            # process-group ID is retried after success or failure.
            clean_process()
            publish()

    try:
        with tempfile.TemporaryDirectory(
            prefix="cpu-export-check-", dir=output.parent
        ) as temporary:
            scratch = Path(temporary)
            if supervise(plan, scratch, "result"):
                validate_completed(document["result"], plan)
                if time.monotonic() >= deadline - 8:
                    document["status"] = "timed_out"
                else:
                    document["status"] = "verifying_final_export_inventory"
                    publish()
                    inventory_config = {
                        **plan,
                        "inventory_only": True,
                        "export_manifest": document["result"]["export_manifest"],
                    }
                    if supervise(inventory_config, scratch, "post_exit_validation"):
                        validate_inventory(
                            document["post_exit_validation"], document["result"], plan
                        )
                        document["status"] = "succeeded"
    except BaseException as error:
        document["status"] = (
            "cleanup_failed"
            if document.get("cleanup_error_type")
            else "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "supervisor_failed"
        )
        document["error_type"] = type(error).__name__[:80]
        raise
    finally:
        document["finished_at"] = utc_now()
        document["elapsed_seconds"] = time.monotonic() - started
        publish()
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-model-dir", type=Path)
    parser.add_argument("--question-ids", nargs=QUESTION_COUNT)
    parser.add_argument("--walltime-seconds", type=int, default=600)
    parser.add_argument("--stop-at")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--child-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.child_config:
        return run_child(read_json(args.child_config))
    if args.job_dir is None or args.merged_output is None or args.output is None:
        parser.error("--job-dir, --merged-output and --output are required.")

    def cancel(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, cancel)
    try:
        options = {
            "base_model_dir": args.base_model_dir,
            "question_ids": args.question_ids,
            "walltime_seconds": args.walltime_seconds,
            "stop_at": args.stop_at,
        }
        if args.validate_only:
            print(
                json.dumps(
                    validate_plan(args.job_dir, args.merged_output, args.output, **options),
                    indent=2,
                )
            )
            return 0
        result = run_verification(args.job_dir, args.merged_output, args.output, **options)
        print(json.dumps({"schema": result["schema"], "status": result["status"]}))
        return 0 if result["status"] == "succeeded" else 2
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
