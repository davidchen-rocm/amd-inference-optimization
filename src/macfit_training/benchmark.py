"""Bounded, evidence-producing ROCm Transformers baseline; no GPU imports at startup.

Run ``python -m macfit_training.benchmark --help`` for the offline campaign CLI.
The benchmark never changes the training service or accepts website requests.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import selectors
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import MODELS

SCHEMA_VERSION = 2
MAX_EVIDENCE_BYTES = 1024**2 - 1
# Operator-only benchmark models. They are deliberately outside the web-training registry.
BENCHMARK_MODELS = {
    **MODELS,
    "qwen3-14b": {
        "id": "qwen3-14b",
        "name": "Qwen3 14B",
        "repo_id": "Qwen/Qwen3-14B",
        "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    },
    "qwen3-32b": {
        "id": "qwen3-32b",
        "name": "Qwen3 32B",
        "repo_id": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
    },
}
PROMPT_TEXT = (
    "This is a fictional support reference. Juniper sells notebooks. Orders ship within "
    "two business days. Returns are accepted within thirty days. Summarize the reference "
    "accurately and avoid inventing details. "
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parse_stop_at(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("The stop time needs an explicit timezone.")
    return result.astimezone(UTC)


def integer_grid(value: str, maximum: int) -> list[int]:
    result = [int(item) for item in value.split(",")]
    if not result or len(result) > 12 or any(not 1 <= item <= maximum for item in result):
        raise ValueError(f"Choose 1 to 12 comma-separated integers between 1 and {maximum}.")
    if len(set(result)) != len(result):
        raise ValueError("Grid coordinates must be unique.")
    return result


def validate_plan(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the local operator's bounded protocol without importing Torch."""
    plan = copy.deepcopy(value)
    if not plan.get("models") or len(plan["models"]) > 8:
        raise ValueError("Choose one to eight model identities.")
    if len(set(plan["models"])) != len(plan["models"]):
        raise ValueError("Model identities must be unique.")
    if any(item not in BENCHMARK_MODELS for item in plan["models"]):
        raise ValueError("Every model must be in the pinned benchmark registry.")
    for key, maximum in (("batch_sizes", 64), ("prompt_tokens", 32768)):
        values = plan.get(key)
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 12
            or any(type(item) is not int or not 1 <= item <= maximum for item in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Invalid {key} grid.")
    bounds = {
        "decode_steps": (1, 512),
        "warmup_repetitions": (1, 5),
        "repetitions": (2, 10),
        "walltime_seconds": (1, 21600),
        "model_timeout_seconds": (1, 7200),
        "seed": (0, 2**31 - 1),
    }
    for key, (minimum, maximum) in bounds.items():
        if type(plan.get(key)) is not int or not minimum <= plan[key] <= maximum:
            raise ValueError(f"{key} must be an integer between {minimum} and {maximum}.")
    if len(plan["batch_sizes"]) * len(plan["prompt_tokens"]) > 64:
        raise ValueError("At most 64 shape coordinates may be requested per model.")
    if plan.get("stop_at") is not None:
        plan["stop_at"] = parse_stop_at(plan["stop_at"]).isoformat()
    plan["model_identities"] = [copy.deepcopy(BENCHMARK_MODELS[item]) for item in plan["models"]]
    plan["prompt_sha256"] = hashlib.sha256(PROMPT_TEXT.encode()).hexdigest()
    plan["benchmark_source_sha256"] = source_sha256()
    return plan


def write_atomic(path: Path, value: Any, *, max_bytes: int = MAX_EVIDENCE_BYTES) -> None:
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    if len(payload) > max_bytes:
        raise ValueError("Evidence exceeds its bounded JSON size.")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < 2:
        raise ValueError("At least two scored samples are required.")
    for sample in samples:
        for key in ("prefill_seconds", "decode_seconds"):
            if not isinstance(sample[key], (float, int)) or not (
                math.isfinite(sample[key]) and sample[key] > 0
            ):
                raise ValueError("Timing samples must be positive and finite.")
        if sample["finite_logits"] is not True:
            raise ValueError("A sample did not prove finite logits.")
    result: dict[str, Any] = {"samples": samples}
    for phase in ("prefill", "decode"):
        elapsed = [sample[f"{phase}_seconds"] for sample in samples]
        throughput = [sample[f"{phase}_tokens_per_second"] for sample in samples]
        if any(not math.isfinite(item) or item <= 0 for item in throughput):
            raise ValueError("Token rates must be positive and finite.")
        result[phase] = {
            "median_seconds": statistics.median(elapsed),
            "median_tokens_per_second": statistics.median(throughput),
            "minimum_tokens_per_second": min(throughput),
            "maximum_tokens_per_second": max(throughput),
            "coefficient_of_variation": statistics.stdev(throughput) / statistics.fmean(throughput),
        }
    result["correctness"] = {
        "finite_logits": True,
        "logit_checks": "prefill_and_final_decode",
        "greedy_repeats_identical": len({sample["output_ids_sha256"] for sample in samples}) == 1,
        "semantic_quality_evaluated": False,
    }
    result["peak_allocated_bytes"] = max(sample["peak_allocated_bytes"] for sample in samples)
    result["peak_reserved_bytes"] = max(sample["peak_reserved_bytes"] for sample in samples)
    return result


def terminate_group(process: subprocess.Popen, *, grace_seconds: float = 2) -> None:
    # A model loader may have descendants even after its group leader has exited.
    # Do not let a completed Popen hide surviving workers in the same process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.wait(timeout=5)


class _ProcessGroupCleanup:
    """One signal/cleanup attempt for one freshly spawned process group."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.attempted = False

    def __call__(self) -> None:
        if self.attempted:
            return
        # Mark before attempting cleanup: retrying a failed or already completed
        # PGID can target a reused identifier rather than this original process.
        self.attempted = True
        terminate_group(self.process)


def run_campaign(
    plan: dict[str, Any],
    output: Path,
    *,
    worker_command: list[str] | None = None,
) -> dict[str, Any]:
    """Supervise model children and publish partial, atomic recovery evidence."""
    plan = validate_plan(plan)
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a fresh output filename; existing evidence is never overwritten.")
    journal_path = output.with_suffix(".events.jsonl")
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "transformers_bf16_sdpa_baseline",
        "status": "running",
        "started_at": utc_now(),
        "protocol": plan,
        "models": [],
        "events": [],
        "limitations": [
            "Synthetic fixed-length repeated reference tokens; duplicate batch sequences.",
            "Greedy decode intentionally continues past EOS for a fixed compute workload.",
            "Torch allocated/reserved memory excludes non-Torch GPU consumers.",
            "Finite logits and repeatability do not establish answer quality.",
            "The unoptimized Transformers baseline is not a vLLM serving benchmark.",
        ],
    }
    started = time.monotonic()
    global_deadline = started + plan["walltime_seconds"]
    if plan.get("stop_at"):
        remaining = (parse_stop_at(plan["stop_at"]) - datetime.now(UTC)).total_seconds()
        global_deadline = min(global_deadline, started + max(0, remaining))
    write_atomic(output, document)
    with journal_path.open("x", encoding="utf-8") as journal:
        os.chmod(journal_path, 0o600)

        def publish(event: dict[str, Any]) -> None:
            event.setdefault("timestamp", utc_now())
            journal.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            compact = {
                key: value for key, value in event.items() if key not in {"result", "runtime"}
            }
            if event.get("stage") == "case_completed":
                compact["coordinate"] = {
                    key: event["result"][key]
                    for key in (
                        "prompt_tokens_per_sequence",
                        "batch_size",
                        "decode_steps_per_sequence",
                    )
                    if key in event["result"]
                }
            document["events"].append(compact)
            # Full samples stay in the per-model evidence and JSONL journal. A
            # compact event tail must not duplicate them or exhaust backup bounds.
            document["events"] = document["events"][-256:]
            write_atomic(output, document)
            print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)

        for identity in plan["model_identities"]:
            model_result: dict[str, Any] = {"identity": identity, "status": "pending", "cases": []}
            model_output = output.with_name(f"{output.stem}-{identity['id']}.json")
            if model_output.exists() or model_output.is_symlink():
                raise ValueError("A per-model evidence filename already exists.")
            model_index = {
                "identity": identity,
                "status": "pending",
                "cases": [],
                "evidence_file": model_output.name,
            }
            document["models"].append(model_index)

            def publish_model(path=model_output, detailed=model_result, index=model_index) -> None:
                write_atomic(path, detailed)
                index.update({key: value for key, value in detailed.items() if key != "cases"})
                index["cases"] = [
                    {key: value for key, value in case.items() if key != "samples"}
                    for case in detailed["cases"]
                ]
                index["evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

            if time.monotonic() >= global_deadline:
                model_result["status"] = "skipped_deadline"
                publish_model()
                publish({"stage": "skipped_deadline", "model_id": identity["id"]})
                continue
            model_result["status"] = "running"
            publish_model()
            publish({"stage": "model_started", "model_id": identity["id"]})
            with tempfile.TemporaryDirectory(prefix="gpu-benchmark-", dir=output.parent) as scratch:
                config = Path(scratch) / "config.json"
                write_atomic(config, {"plan": plan, "identity": identity})
                # Executing this CLI with ``python -m`` sets __name__ to __main__.
                # A child must import the stable package module, never ``-m __main__``.
                command = worker_command or [
                    sys.executable,
                    "-m",
                    "macfit_training.benchmark",
                    "--child-config",
                ]
                stderr_path = output.with_name(f"{output.stem}.{identity['id']}.stderr.log")
                with stderr_path.open("xb") as stderr:
                    os.chmod(stderr_path, 0o600)
                    process = subprocess.Popen(
                        [*command, str(config)],
                        stdout=subprocess.PIPE,
                        stderr=stderr,
                        start_new_session=True,
                    )
                    cleanup = _ProcessGroupCleanup(process)
                    assert process.stdout is not None
                    model_deadline = min(
                        global_deadline, time.monotonic() + plan["model_timeout_seconds"]
                    )
                    buffer = bytearray()
                    try:
                        with selectors.DefaultSelector() as selector:
                            selector.register(process.stdout, selectors.EVENT_READ)
                            while selector.get_map():
                                if time.monotonic() >= model_deadline:
                                    model_result["status"] = "timed_out"
                                    cleanup()
                                    publish({"stage": "model_timeout", "model_id": identity["id"]})
                                    break
                                for key, _ in selector.select(timeout=0.1):
                                    chunk = os.read(key.fileobj.fileno(), 65536)
                                    if not chunk:
                                        selector.unregister(key.fileobj)
                                        continue
                                    buffer.extend(chunk)
                                    if len(buffer) > 1024**2:
                                        raise ValueError("Worker event framing exceeded its bound.")
                                    while b"\n" in buffer:
                                        raw, _, tail = buffer.partition(b"\n")
                                        buffer = bytearray(tail)
                                        event = json.loads(raw)
                                        if (
                                            not isinstance(event, dict)
                                            or event.get("model_id") != identity["id"]
                                        ):
                                            raise ValueError(
                                                "The worker emitted an invalid event identity."
                                            )
                                        stage = event.get("stage")
                                        if stage == "case_completed":
                                            model_result["cases"].append(event["result"])
                                        elif stage == "runtime_ready":
                                            model_result["runtime"] = event["runtime"]
                                        elif stage in {
                                            "succeeded",
                                            "out_of_memory",
                                            "worker_failed",
                                        }:
                                            model_result["status"] = stage
                                        publish_model()
                                        publish(event)
                            if buffer.strip() and model_result["status"] != "timed_out":
                                raise ValueError("Worker ended with an incomplete event.")
                        return_code = process.wait(timeout=5)
                        model_result["exit_code"] = return_code
                        if model_result["status"] == "running" or (
                            model_result["status"] == "succeeded" and return_code != 0
                        ):
                            model_result["status"] = "worker_failed"
                    except BaseException as error:
                        model_result["status"] = (
                            "interrupted"
                            if isinstance(error, KeyboardInterrupt)
                            else "supervisor_failed"
                        )
                        document["status"] = model_result["status"]
                        document["finished_at"] = utc_now()
                        try:
                            cleanup()
                        except Exception as cleanup_error:
                            model_result["cleanup_error_type"] = type(cleanup_error).__name__
                            error.add_note(
                                "Process-group cleanup failed: " + type(cleanup_error).__name__
                            )
                            raise error from cleanup_error
                        raise
                    finally:
                        # Download/runtime descendants can redirect stdio and outlive
                        # a successful leader. Release the entire lease holder group.
                        try:
                            cleanup()
                        except Exception as cleanup_error:
                            model_result["status"] = "cleanup_failed"
                            model_result["cleanup_error_type"] = type(cleanup_error).__name__
                            document["status"] = "cleanup_failed"
                            document["finished_at"] = utc_now()
                            raise
                        finally:
                            process.stdout.close()
                            publish_model()
                            write_atomic(output, document)
        document["status"] = (
            "succeeded"
            if all(item["status"] == "succeeded" for item in document["models"])
            else "incomplete"
        )
        document["finished_at"] = utc_now()
        document["elapsed_seconds"] = time.monotonic() - started
        publish({"stage": "campaign_finished", "status": document["status"]})
    return document


def _model_worker(config: dict[str, Any]) -> int:
    # The supervisor owns the deadline, including lock wait/download/native GPU calls.
    from amd_inference_opt.resource_lock import exclusive_gpu_lock

    identity, plan = config["identity"], validate_plan(config["plan"])

    def emit(stage: str, **values: Any) -> None:
        print(json.dumps({"stage": stage, "model_id": identity["id"], **values}), flush=True)

    try:
        emit("waiting_for_gpu")
        lock = Path(os.environ.get("MACFIT_GPU_LOCK", "/tmp/macfit-training-gpu.lock"))
        with exclusive_gpu_lock(lock):
            return _benchmark_loaded_model(identity, plan, emit)
    except Exception as error:
        import traceback

        traceback.print_exc()
        stage = "out_of_memory" if "out of memory" in str(error).lower() else "worker_failed"
        emit(stage, error_type=type(error).__name__)
        return 2


def _benchmark_loaded_model(identity: dict[str, str], plan: dict[str, Any], emit: Any) -> int:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from amd_inference_opt.vllm_model_snapshot import capture_vllm_model_snapshot

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("An available ROCm GPU is required; CPU/offload results are forbidden.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This baseline requires native BF16 support.")
    torch.manual_seed(plan["seed"])
    torch.cuda.manual_seed_all(plan["seed"])
    emit("loading_model")
    path = snapshot_download(
        identity["repo_id"],
        revision=identity["revision"],
        cache_dir=os.environ.get("MACFIT_MODEL_CACHE"),
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
    snapshot = capture_vllm_model_snapshot(
        path,
        model_id=identity["repo_id"],
        revision=identity["revision"],
        tokenizer_revision=identity["revision"],
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    loaded_at = time.monotonic()
    model = (
        AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        .to("cuda:0")
        .eval()
    )
    torch.cuda.synchronize()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("The loaded model includes non-GPU parameters.")
    properties = torch.cuda.get_device_properties(0)
    emit(
        "runtime_ready",
        runtime={
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "rocm": torch.version.hip,
            "transformers": importlib.metadata.version("transformers"),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_bytes": properties.total_memory,
            "dtype": "bfloat16",
            "attention": "sdpa",
            "full_gpu_weights": True,
            "model_snapshot_sha256": snapshot.snapshot_digest,
            "benchmark_source_sha256": source_sha256(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "model_load_seconds": time.monotonic() - loaded_at,
            "model_allocated_bytes": torch.cuda.memory_allocated(),
        },
    )
    tokens = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    if not tokens:
        raise RuntimeError("The fixed benchmark prompt tokenized to empty input.")
    maximum_context = getattr(model.config, "max_position_embeddings", 0)
    for prompt_length in plan["prompt_tokens"]:
        if not maximum_context or prompt_length + plan["decode_steps"] > maximum_context:
            raise ValueError("A benchmark coordinate exceeds the model's declared context.")
        ids = (tokens * math.ceil(prompt_length / len(tokens)))[:prompt_length]
        for batch_size in plan["batch_sizes"]:
            emit("case_started", prompt_tokens=prompt_length, batch_size=batch_size)
            torch.cuda.empty_cache()
            for _ in range(plan["warmup_repetitions"]):
                _measure_sample(model, ids, batch_size, plan["decode_steps"], torch)
            samples = [
                _measure_sample(model, ids, batch_size, plan["decode_steps"], torch)
                for _ in range(plan["repetitions"])
            ]
            result = {
                "prompt_tokens_per_sequence": prompt_length,
                "batch_size": batch_size,
                "decode_steps_per_sequence": plan["decode_steps"],
                "scored_repetitions": plan["repetitions"],
                "warmup_repetitions_excluded": plan["warmup_repetitions"],
                **summarize_samples(samples),
            }
            emit("case_completed", result=result)
    emit("succeeded")
    return 0


def _measure_sample(
    model: Any, ids: list[int], batch_size: int, steps: int, torch: Any
) -> dict[str, Any]:
    """Include Python dispatch/synchronization; exclude finite checks from timing."""
    inputs = torch.tensor([ids] * batch_size, dtype=torch.long, device="cuda:0")
    mask = torch.ones_like(inputs)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        torch.cuda.synchronize()
        started = time.monotonic()
        output = model(input_ids=inputs, attention_mask=mask, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        prefill_seconds = time.monotonic() - started
        finite = bool(torch.isfinite(output.logits).all().item())
        next_ids = output.logits[:, -1:].argmax(dim=-1)
        generated = [next_ids]
        cache = output.past_key_values
        del output
        torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(steps):
            mask = torch.cat(
                (mask, torch.ones((batch_size, 1), dtype=mask.dtype, device=mask.device)), dim=1
            )
            output = model(
                input_ids=next_ids,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            next_ids = output.logits[:, -1:].argmax(dim=-1)
            generated.append(next_ids)
            cache = output.past_key_values
        torch.cuda.synchronize()
        decode_seconds = time.monotonic() - started
        # Check both prefill and final decode logits without contaminating phase timing.
        finite = finite and bool(torch.isfinite(output.logits).all().item())
        output_ids = torch.cat(generated, dim=1).cpu().tolist()
        result = {
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "prefill_tokens_per_second": batch_size * len(ids) / prefill_seconds,
            "decode_tokens_per_second": batch_size * steps / decode_seconds,
            "finite_logits": finite,
            "output_ids_sha256": hashlib.sha256(json.dumps(output_ids).encode()).hexdigest(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        del output, cache, next_ids, generated, inputs, mask
    if not finite:
        raise RuntimeError("The model produced non-finite logits.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen3-0-6b,qwen3-4b,qwen3-8b,qwen3-14b,qwen3-32b")
    parser.add_argument("--batch-sizes", default="1,4,16")
    parser.add_argument("--prompt-tokens", default="128,512,2048")
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--walltime-seconds", type=int, default=3600)
    parser.add_argument("--model-timeout-seconds", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stop-at", help="Absolute ISO-8601 time with timezone; includes all work."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--child-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.child_config:
        return _model_worker(json.loads(args.child_config.read_text()))

    def cancel(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, cancel)
    try:
        plan = validate_plan(
            {
                "models": args.models.split(","),
                "batch_sizes": integer_grid(args.batch_sizes, 64),
                "prompt_tokens": integer_grid(args.prompt_tokens, 32768),
                "decode_steps": args.decode_steps,
                "warmup_repetitions": args.warmup_repetitions,
                "repetitions": args.repetitions,
                "walltime_seconds": args.walltime_seconds,
                "model_timeout_seconds": args.model_timeout_seconds,
                "seed": args.seed,
                "stop_at": args.stop_at,
            }
        )
        if args.validate_only:
            print(json.dumps(plan, indent=2))
            return 0
        if args.output is None:
            parser.error("--output is required for a live campaign.")
        result = run_campaign(plan, args.output)
        return 0 if result["status"] == "succeeded" else 2
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
