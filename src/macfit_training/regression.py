"""Bounded local LoRA capability checks with independent, generic instructions.

This operator-only CLI never submits website jobs or downloads models. It reads
completed, hash-verified training artifacts and a verified local base snapshot.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
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
from .experiments import parse_strict_json
from .export import extract_adapter, read_json, real_path, verified_job

MODULE = "macfit_training.regression"
MAX_EVIDENCE_BYTES = 1024**2
MAX_CHILD_BYTES = 64 * 1024
SYSTEM = (
    "Follow the user's task. Return only the requested answer or JSON object. "
    "Do not add explanations, Markdown or code fences. "
    "遵守用户要求，只返回所要求的答案或 JSON 对象，不要添加解释或代码块。"
)
CASES = (
    {
        "id": "add-en",
        "family": "arithmetic",
        "language": "en",
        "question": "Compute 17 + 25. Return only the integer.",
        "expected": "42",
        "format": "text",
    },
    {
        "id": "multiply-zh",
        "family": "arithmetic",
        "language": "zh",
        "question": "计算 9 乘以 7，只输出整数。",
        "expected": "63",
        "format": "text",
    },
    {
        "id": "subtract-en",
        "family": "arithmetic",
        "language": "en",
        "question": "Compute 100 - 37. Return only the integer.",
        "expected": "63",
        "format": "text",
    },
    {
        "id": "divide-zh",
        "family": "arithmetic",
        "language": "zh",
        "question": "12 本书平均分给 3 人，每人得到几本？只输出整数。",
        "expected": "4",
        "format": "text",
    },
    {
        "id": "reverse-en",
        "family": "string",
        "language": "en",
        "question": "Reverse the exact ASCII string stressed. Return only the reversed string.",
        "expected": "desserts",
        "format": "text",
    },
    {
        "id": "reverse-zh",
        "family": "string",
        "language": "zh",
        "question": "将 ASCII 字符串 abcdef 的字符顺序反转，只输出反转后的字符串。",
        "expected": "fedcba",
        "format": "text",
    },
    {
        "id": "vowels-en",
        "family": "string",
        "language": "en",
        "question": "Extract only the lowercase vowels from education in their original order. "
        "Return only those letters.",
        "expected": "euaio",
        "format": "text",
    },
    {
        "id": "json-add-en",
        "family": "json",
        "language": "en",
        "question": "Return a JSON object with exactly two keys: total is the integer sum of "
        "3 and 4; is_even is a boolean stating whether that total is even.",
        "expected": '{"total":7,"is_even":false}',
        "format": "json",
    },
    {
        "id": "json-sort-zh",
        "family": "json",
        "language": "zh",
        "question": "把整数列表 [3,1,2] 从小到大排序，只输出一个 JSON 对象，"
        "唯一的键是 sorted，值为排序后的整数列表。",
        "expected": '{"sorted":[1,2,3]}',
        "format": "json",
    },
    {
        "id": "order-zh",
        "family": "reasoning",
        "language": "zh",
        "question": "甲的年龄比乙大，乙的年龄比丙大。谁的年龄最小？只输出甲、乙或丙中的一个字。",
        "expected": "丙",
        "format": "text",
    },
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def protocol() -> dict[str, Any]:
    return {
        "schema": "macfit-lora-regression-protocol.v1",
        "system": SYSTEM,
        "cases": list(CASES),
        "cases_sha256": canonical_sha256(CASES),
        "system_sha256": canonical_sha256(SYSTEM),
        "source_sha256": describe_artifact(Path(__file__), "source")["sha256"],
        "strict_scoring_helpers_source_sha256": describe_artifact(
            Path(parse_strict_json.__code__.co_filename), "source"
        )["sha256"],
        "decoding": {"mode": "greedy", "enable_thinking": False, "max_new_tokens": 64},
        "normalization": "NFKC and surrounding whitespace for text; "
        "typed JSON equality for objects.",
        "limitations": [
            "Ten hand-specified examples are a tiny diagnostic, "
            "not a general capability benchmark.",
            "Independent generic instructions differ from the policy task used in SFT.",
            "Exact-output checks reject additional explanations even when their answer is correct.",
            "A pass-rate difference does not establish broad improvement or degradation.",
        ],
    }


def answer_passes(case: dict[str, Any], answer: Any) -> bool:
    if not isinstance(answer, str):
        return False
    if case["format"] == "json":
        predicted = parse_strict_json(answer)
        expected = parse_strict_json(case["expected"])
        return predicted is not None and canonical_sha256(predicted) == canonical_sha256(expected)
    return unicodedata.normalize("NFKC", answer).strip() == case["expected"]


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if {sample.get("id") for sample in samples} != {case["id"] for case in CASES}:
        raise ValueError("Every regression case must appear exactly once.")
    if len(samples) != len(CASES) or any(
        any(stage not in sample for stage in ("base", "adapter")) for sample in samples
    ):
        raise ValueError("Both model outputs are required for every regression case.")
    before = sum(sample["base"]["passed"] is True for sample in samples)
    after = sum(sample["adapter"]["passed"] is True for sample in samples)
    return {
        "cases": len(samples),
        "base_passes": before,
        "adapter_passes": after,
        "base_pass_fraction": before / len(samples),
        "adapter_pass_fraction": after / len(samples),
        "pass_fraction_change": (after - before) / len(samples),
        "regressed_case_ids": [
            sample["id"]
            for sample in samples
            if sample["base"]["passed"] and not sample["adapter"]["passed"]
        ],
        "improved_case_ids": [
            sample["id"]
            for sample in samples
            if not sample["base"]["passed"] and sample["adapter"]["passed"]
        ],
    }


def verified_model(job_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Any, Path]:
    """Reuse completed artifact checks and independently verify cached model bytes."""
    from amd_inference_opt.vllm_model_snapshot import (
        VLLMModelSnapshotManifest,
        verify_vllm_model_snapshot,
    )

    job, result = verified_job(job_dir)
    manifest = read_json(job_dir / "artifacts/manifest.json")
    if manifest.get("provenance") != result.get("provenance"):
        raise ValueError("The result provenance does not match the artifact manifest.")
    metadata = read_json(job_dir / "model-snapshot.json")
    if "schema_name" in metadata:
        metadata["schema"] = metadata.pop("schema_name")
    snapshot = VLLMModelSnapshotManifest.model_validate(metadata)
    base = job["base_model"]
    if (
        snapshot.model_id != base["repo_id"]
        or snapshot.revision != base["revision"]
        or snapshot.tokenizer_revision != base["revision"]
        or snapshot.snapshot_digest != result.get("provenance", {}).get("model_snapshot_sha256")
    ):
        raise ValueError("The snapshot does not match the completed job's pinned base model.")
    directory = real_path(snapshot.root)
    verify_vllm_model_snapshot(snapshot.model_copy(update={"root": directory}))
    return job, result, snapshot, directory


def load_base(base_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.version.hip or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Regression inference requires a native BF16 ROCm GPU.")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    tokenizer = AutoTokenizer.from_pretrained(
        base_dir, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            base_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        .to("cuda:0")
        .eval()
    )
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("All model weights must reside on the GPU.")
    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "gpu_name": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "attention": "sdpa",
        "full_gpu_weights": True,
        "seed": 42,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    return model, tokenizer, runtime


def run_child(config: dict[str, Any]) -> int:
    from amd_inference_opt.resource_lock import exclusive_gpu_lock

    job_dir, output = real_path(Path(config["job"])), real_path(Path(config["child_output"]))
    document: dict[str, Any] = {"status": "waiting_for_gpu", "samples": []}

    def publish() -> None:
        write_atomic(output, document, max_bytes=MAX_CHILD_BYTES)

    publish()
    try:
        lock = Path(os.environ.get("MACFIT_GPU_LOCK", "/tmp/macfit-training-gpu.lock"))
        with exclusive_gpu_lock(lock):
            document["status"] = "verifying_artifacts"
            publish()
            job, result, snapshot, base_dir = verified_model(job_dir)
            document["base_model"] = job["base_model"]
            document["input_sha256"] = canonical_sha256(job)
            document["model_snapshot_sha256"] = snapshot.snapshot_digest
            document["adapter_sha256"] = next(
                row["sha256"] for row in result["artifacts"] if row["type"] == "lora_adapter"
            )
            document["status"] = "loading_base"
            publish()
            with tempfile.TemporaryDirectory(
                prefix="verified-adapter-", dir=output.parent
            ) as temporary:
                adapter_dir = Path(temporary)
                extract_adapter(
                    job_dir / "artifacts/adapter.tar.gz", adapter_dir, job["base_model"]
                )
                model, tokenizer, runtime = load_base(base_dir)
                document["runtime"] = runtime
                document["samples"] = [dict(case) for case in CASES]
                from .evaluation import generate_text

                for stage in ("base", "adapter"):
                    if stage == "adapter":
                        from peft import PeftModel

                        model = PeftModel.from_pretrained(
                            model,
                            adapter_dir,
                            local_files_only=True,
                            is_trainable=False,
                            torch_device="cuda:0",
                            device_map={"": "cuda:0"},
                        ).eval()
                    document["status"] = "evaluating_" + stage
                    publish()
                    for sample in document["samples"]:
                        prediction = generate_text(
                            model,
                            tokenizer,
                            [
                                {"role": "system", "content": SYSTEM},
                                {"role": "user", "content": sample["question"]},
                            ],
                            max_input_tokens=1024,
                            max_new_tokens=64,
                        )
                        prediction["passed"] = bool(
                            not prediction["stopped_by_limit"]
                            and answer_passes(sample, prediction["text"])
                        )
                        sample[stage] = prediction
                        publish()
                document["metrics"] = summarize(document["samples"])
                document["status"] = "succeeded"
                publish()
                del model, tokenizer
                import torch

                torch.cuda.empty_cache()
        return 0
    except Exception as error:
        document["status"] = "worker_failed"
        document["error_type"] = type(error).__name__
        publish()
        print(f"Regression worker failed: {type(error).__name__}", file=sys.stderr)
        return 2


def validate_plan(
    jobs: list[Path], walltime_seconds: int, job_timeout_seconds: int, stop_at: str | None
) -> dict[str, Any]:
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 8:
        raise ValueError("Choose one to eight completed local job directories.")
    paths = [real_path(Path(job)) for job in jobs]
    if len(set(paths)) != len(paths) or any(not path.is_dir() for path in paths):
        raise ValueError("Job directories must exist and be distinct, without symbolic links.")
    if type(walltime_seconds) is not int or not 10 <= walltime_seconds <= 21600:
        raise ValueError("Walltime must be 10 to 21600 seconds, including cleanup.")
    if type(job_timeout_seconds) is not int or not 1 <= job_timeout_seconds <= 3600:
        raise ValueError("Per-job timeout must be 1 to 3600 seconds.")
    return {
        "jobs": [str(path) for path in paths],
        "walltime_seconds": walltime_seconds,
        "job_timeout_seconds": job_timeout_seconds,
        "stop_at": parse_stop_at(stop_at).isoformat() if stop_at else None,
    }


def read_child(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_CHILD_BYTES:
        raise ValueError("Child regression evidence exceeded its size bound.")
    result = read_json(path)
    if len(json.dumps(result, ensure_ascii=False).encode()) > MAX_CHILD_BYTES:
        raise ValueError("Child regression evidence exceeded its size bound.")
    return result


def validate_completed_child(result: dict[str, Any]) -> None:
    samples = result.get("samples")
    if not isinstance(samples, list) or result.get("metrics") != summarize(samples):
        raise ValueError("A completed child needs all actual outputs and matching metrics.")
    expected = {case["id"]: case for case in CASES}
    for sample in samples:
        case = expected[sample["id"]]
        if any(sample.get(key) != value for key, value in case.items()):
            raise ValueError("The child changed a regression question or expected answer.")
        for stage in ("base", "adapter"):
            prediction = sample[stage]
            if (
                not isinstance(prediction, dict)
                or not isinstance(prediction.get("text"), str)
                or type(prediction.get("stopped_by_limit")) is not bool
                or prediction.get("passed")
                is not (
                    not prediction["stopped_by_limit"] and answer_passes(case, prediction["text"])
                )
            ):
                raise ValueError("The child pass flag does not match its actual output.")


def run_regressions(
    jobs: list[Path],
    output: Path,
    *,
    walltime_seconds: int = 1800,
    job_timeout_seconds: int = 600,
    stop_at: str | None = None,
    worker_command: list[str] | None = None,
) -> dict[str, Any]:
    plan = validate_plan(jobs, walltime_seconds, job_timeout_seconds, stop_at)
    output = real_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a fresh output filename; evidence is never overwritten.")
    started = time.monotonic()
    deadline = started + walltime_seconds
    if plan["stop_at"]:
        remaining = (parse_stop_at(plan["stop_at"]) - datetime.now(UTC)).total_seconds()
        deadline = min(deadline, started + max(0, remaining))
    # Reserve time for graceful termination and reaping before the absolute deadline.
    work_deadline = deadline - 8
    document = {
        "schema": "macfit-lora-regression.v1",
        "status": "running",
        "started_at": utc_now(),
        "protocol": protocol(),
        "jobs": [],
        "walltime_seconds": walltime_seconds,
        "stop_at": plan["stop_at"],
    }

    def publish() -> None:
        write_atomic(output, document, max_bytes=MAX_EVIDENCE_BYTES)

    publish()
    try:
        for index, job_path in enumerate(plan["jobs"]):
            row = {"source_job": Path(job_path).name, "index": index, "status": "pending"}
            document["jobs"].append(row)
            if time.monotonic() >= work_deadline:
                row["status"] = "skipped_deadline"
                publish()
                continue
            with tempfile.TemporaryDirectory(
                prefix="lora-regression-", dir=output.parent
            ) as temporary:
                scratch = Path(temporary)
                config, child_output = scratch / "config.json", scratch / "child.json"
                write_atomic(config, {"job": job_path, "child_output": str(child_output)})
                command = worker_command or [sys.executable, "-m", MODULE, "--child-config"]
                stderr_path = output.with_name(f"{output.stem}-{index}.stderr.log")
                with stderr_path.open("xb") as stderr:
                    os.chmod(stderr_path, 0o600)
                    process = subprocess.Popen(
                        [*command, str(config)],
                        stdout=subprocess.DEVNULL,
                        stderr=stderr,
                        start_new_session=True,
                    )
                    cleanup = _ProcessGroupCleanup(process)

                    def clean_process(
                        cleanup_attempt: Any = cleanup, evidence_row: dict = row
                    ) -> None:
                        try:
                            cleanup_attempt()
                        except Exception as cleanup_error:
                            evidence_row["cleanup_error_type"] = type(cleanup_error).__name__[:80]
                            evidence_row["status"] = "cleanup_failed"
                            raise

                    job_deadline = min(work_deadline, time.monotonic() + job_timeout_seconds)
                    timed_out = False
                    try:
                        while process.poll() is None:
                            if child_output.exists():
                                row.update(read_child(child_output))
                                publish()
                            if time.monotonic() >= job_deadline:
                                timed_out = True
                                break
                            time.sleep(0.2)
                        # Cleanup covers surviving descendants after leader exit,
                        # before accepting final evidence or starting the next job.
                        clean_process()
                        if child_output.exists():
                            row.update(read_child(child_output))
                        row["exit_code"] = process.wait(timeout=5)
                        if timed_out:
                            row["status"] = "timed_out"
                        elif row.get("status") != "succeeded" or row["exit_code"] != 0:
                            row["status"] = "worker_failed"
                        else:
                            try:
                                validate_completed_child(row)
                            except (ValueError, TypeError, KeyError):
                                row["status"] = "invalid_child_evidence"
                                raise
                    except BaseException as error:
                        row["error_type"] = type(error).__name__[:80]
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
                            # The helper marks attempted before signalling, so a
                            # failed or completed process-group ID is never retried.
                            clean_process()
                        finally:
                            publish()
        document["status"] = (
            "succeeded"
            if all(row["status"] == "succeeded" for row in document["jobs"])
            else "incomplete"
        )
    except BaseException as error:
        document["status"] = (
            "cleanup_failed"
            if any(row.get("cleanup_error_type") for row in document["jobs"])
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
    parser.add_argument("--jobs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--walltime-seconds", type=int, default=1800)
    parser.add_argument("--job-timeout-seconds", type=int, default=600)
    parser.add_argument("--stop-at", help="Absolute ISO-8601 stop time including timezone.")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--child-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.child_config:
        return run_child(read_json(args.child_config))
    if args.jobs is None:
        parser.error("--jobs is required.")
    if args.output is None and not args.validate_only:
        parser.error("--output is required for a live regression run.")

    def cancel(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, cancel)
    try:
        if args.validate_only:
            print(
                json.dumps(
                    validate_plan(
                        args.jobs, args.walltime_seconds, args.job_timeout_seconds, args.stop_at
                    ),
                    indent=2,
                )
            )
            return 0
        result = run_regressions(
            args.jobs,
            args.output,
            walltime_seconds=args.walltime_seconds,
            job_timeout_seconds=args.job_timeout_seconds,
            stop_at=args.stop_at,
        )
        return 0 if result["status"] == "succeeded" else 2
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
