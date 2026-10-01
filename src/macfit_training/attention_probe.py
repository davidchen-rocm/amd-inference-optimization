"""Operator-only, bounded ROCm Flash Attention preference arms; no eager GPU imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import benchmark
from .service.outputs import open_regular

ARMS = ("aotriton", "ck")
ALLOWED_MODELS = {"qwen3-8b", "qwen3-14b"}
MAX_PROFILE_EVENTS = 65536
MAX_PROFILE_NAMES = 1024
DISPLAY_PROFILE_NAMES = 128
MAX_BACKUP_EVIDENCE_FILES = 64


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_protocol(plan: dict[str, Any]) -> dict[str, Any]:
    result = benchmark.validate_plan(plan)
    if len(result["models"]) != 1 or result["models"][0] not in ALLOWED_MODELS:
        raise ValueError("Choose exactly one pinned Qwen3 8B or 14B model.")
    if not set(result["batch_sizes"]) <= {1, 4} or not set(result["prompt_tokens"]) <= {2048, 8192}:
        raise ValueError("Use batches 1/4 and contexts 2048/8192 for this controlled probe.")
    if (
        result["decode_steps"] != 64
        or result["warmup_repetitions"] != 2
        or result["repetitions"] != 5
        or result["seed"] != 42
    ):
        raise ValueError("The controlled protocol uses decode64, warmup2, scored5 and seed42.")
    if result["walltime_seconds"] > 3600 or result["stop_at"] is None:
        raise ValueError("Supply an absolute cutoff and a campaign budget of at most one hour.")
    return result


def summarize_profile(events: Any) -> dict[str, Any]:
    """Aggregate capped CPU/device names; hashes cover full names, not display truncation."""
    if len(events) > MAX_PROFILE_EVENTS:
        raise ValueError("The profiling event bound was exceeded.")
    kernels: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    for event in events:
        name = str(event.name)
        if len(name) > 8192:
            raise ValueError("A profile name exceeded its bound.")
        device = str(event.device_type).lower().rsplit(".", 1)[-1]
        if device in {"cuda", "hip"}:
            kernels[name] += 1
        elif name.startswith("aten::"):
            operators[name] += 1
        if len(kernels) > MAX_PROFILE_NAMES or len(operators) > MAX_PROFILE_NAMES:
            raise ValueError("The unique profile-name bound was exceeded.")

    def names(value: Counter[str]) -> dict[str, Any]:
        complete = sorted(value)
        return {
            "unique_names": len(complete),
            "event_count": sum(value.values()),
            "full_name_set_sha256": hashlib.sha256(json.dumps(complete).encode()).hexdigest(),
            "display_names": [
                {
                    "name": name.encode("utf-8")[:256].decode("utf-8", errors="ignore"),
                    "count": value[name],
                }
                for name in complete[:DISPLAY_PROFILE_NAMES]
            ],
            "display_truncated": len(complete) > DISPLAY_PROFILE_NAMES
            or any(len(name.encode("utf-8")) > 256 for name in complete),
        }

    flash = any("flash_attention" in name.lower() for name in operators)
    math_op = any("scaled_dot_product_attention_math" in name for name in operators)
    return {
        "status": "captured" if kernels and flash and not math_op else "dispatch_unverified",
        "events": len(events),
        "gpu_kernels": names(kernels),
        "torch_operators": names(operators),
        "flash_operator_observed": flash,
        "math_operator_observed": math_op,
        "timed_samples_include_profiling": False,
    }


def capture_profile(torch: Any, execute: Any) -> dict[str, Any]:
    started, completed = False, False
    try:
        activity = torch.profiler.ProfilerActivity
        if activity.CUDA not in torch.profiler.supported_activities():
            return {"status": "unsupported", "reason": "gpu_profiler_unavailable"}
        with torch.profiler.profile(
            activities=[activity.CPU, activity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
        ) as profile:
            started = True
            execute()
            completed = True
        return summarize_profile(profile.events())
    except Exception as error:
        if started and not completed:
            raise  # An actual forward failed; do not hide it as a profiler limitation.
        return {
            "status": "unsupported",
            "reason": "profiler_failed",
            "error_type": type(error).__name__,
        }


def _worker(config: dict[str, Any], arm: str) -> int:
    from amd_inference_opt.resource_lock import exclusive_gpu_lock

    if arm not in ARMS:
        raise ValueError("Unknown attention preference arm.")
    plan = validate_protocol(config["plan"])
    identity = config["identity"]
    if identity != plan["model_identities"][0]:
        raise ValueError("The worker identity does not match its pinned protocol.")

    def emit(stage: str, **values: Any) -> None:
        print(json.dumps({"stage": stage, "model_id": identity["id"], **values}), flush=True)

    original_measure = benchmark._measure_sample
    runtime: dict[str, Any] = {
        "preference_requested": arm,
        "forced_sdpa_backend": "FLASH_ATTENTION",
        "attention_probe_source_sha256": source_sha256(),
    }
    try:
        emit("waiting_for_gpu")
        lease = os.environ.get("MACFIT_GPU_LOCK")
        if not lease:
            raise ValueError("An explicit shared MACFIT_GPU_LOCK is required.")
        with exclusive_gpu_lock(Path(lease)):
            import torch
            from torch.nn.attention import SDPBackend, sdpa_kernel

            if not torch.cuda.is_available() or not torch.version.hip:
                raise RuntimeError("ROCm GPU runtime unavailable.")
            torch.backends.cuda.preferred_rocm_fa_library(arm)
            preference = torch.backends.cuda.preferred_rocm_fa_library()
            returned = getattr(preference, "name", str(preference)).lower()
            runtime["preference_returned"] = returned
            if returned != arm:
                raise RuntimeError("The requested ROCm preference was not accepted.")
            profiled = False

            def measured(model, ids, batch, steps, accelerator):
                nonlocal profiled
                if not profiled:
                    profiled = True
                    emit("profiling_attention", batch_size=batch, prompt_tokens=len(ids))
                    profile = capture_profile(
                        accelerator, lambda: original_measure(model, ids, batch, 1, accelerator)
                    )
                    profile.update(batch_size=batch, prompt_tokens=len(ids), decode_steps=1)
                    runtime["profile"] = profile
                    emit("runtime_ready", runtime=runtime)
                return original_measure(model, ids, batch, steps, accelerator)

            def enriched(stage, **values):
                if stage == "runtime_ready":
                    runtime.update(values["runtime"])
                    runtime["attention"] = "sdpa_flash_only_preference_arm"
                    values["runtime"] = runtime
                emit(stage, **values)

            # This hook exists only inside the isolated operator child. The extra
            # profiling pass finishes before either warmup and every scored sample.
            benchmark._measure_sample = measured
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                return benchmark._benchmark_loaded_model(identity, plan, enriched)
    except Exception as error:
        import traceback

        traceback.print_exc()
        message = str(error).lower()
        unavailable = isinstance(error, (ImportError, AttributeError)) or any(
            marker in message
            for marker in (
                "not compiled",
                "not been compiled",
                "not supported",
                "unavailable",
                "no available kernel",
                "no viable backend",
                "was not accepted",
            )
        )
        stage = "out_of_memory" if "out of memory" in message else "worker_failed"
        emit(
            stage,
            error_type=type(error).__name__,
            code="attention_unsupported" if unavailable else "attention_probe_failed",
        )
        return 2
    finally:
        benchmark._measure_sample = original_measure


def read_verified_model(directory: Path, index: dict[str, Any]) -> dict[str, Any]:
    filename = index["evidence_file"]
    if Path(filename).name != filename:
        raise ValueError("The model evidence filename must be local.")
    with open_regular(directory / filename, benchmark.MAX_EVIDENCE_BYTES) as stream:
        payload = stream.read(benchmark.MAX_EVIDENCE_BYTES + 1)
    if hashlib.sha256(payload).hexdigest() != index["evidence_sha256"]:
        raise ValueError("The model evidence failed its recorded byte hash.")
    return json.loads(payload)


def comparison(arms: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": "preference_arms",
        "backend_difference_established": False,
        "quality_evaluated": False,
        "cases": [],
    }
    if len(arms) != 2 or any(arm["status"] != "succeeded" for arm in arms):
        result["status"] = "incomplete"
        return result
    profiles = [arm["model"].get("runtime", {}).get("profile", {}) for arm in arms]
    result["profile_dispatch_verified"] = all(
        profile.get("status") == "captured" for profile in profiles
    )
    if result["profile_dispatch_verified"]:
        result["kernel_name_fingerprints_differ"] = (
            profiles[0]["gpu_kernels"]["full_name_set_sha256"]
            != profiles[1]["gpu_kernels"]["full_name_set_sha256"]
        )
    else:
        result["kernel_name_fingerprints_differ"] = None

    def coordinate(case):
        return (
            case["prompt_tokens_per_sequence"],
            case["batch_size"],
            case["decode_steps_per_sequence"],
        )

    baseline = {coordinate(case): case for case in arms[0]["model"]["cases"]}
    treatment = {coordinate(case): case for case in arms[1]["model"]["cases"]}
    if baseline.keys() != treatment.keys():
        raise ValueError("Preference arms completed different coordinates.")
    for key in baseline:
        first, second = baseline[key], treatment[key]
        result["cases"].append(
            {
                "prompt_tokens": key[0],
                "batch_size": key[1],
                "decode_steps": key[2],
                "aotriton_prefill_tokens_per_second": first["prefill"]["median_tokens_per_second"],
                "ck_prefill_tokens_per_second": second["prefill"]["median_tokens_per_second"],
                "aotriton_decode_tokens_per_second": first["decode"]["median_tokens_per_second"],
                "ck_decode_tokens_per_second": second["decode"]["median_tokens_per_second"],
                "greedy_token_hash_sets_equal": (
                    {sample["output_ids_sha256"] for sample in first["samples"]}
                    == {sample["output_ids_sha256"] for sample in second["samples"]}
                ),
            }
        )
    result["status"] = "measured_preferences"
    return result


def run_probe(
    plan: dict[str, Any], output: Path, *, worker_commands: dict[str, list[str]] | None = None
) -> dict[str, Any]:
    plan = validate_protocol(plan)
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a fresh attention-probe evidence filename.")
    # Two arm indexes + two detailed model files + this comparison. Reserve one
    # additional slot for the external runtime evidence selected by deployment.
    if len(list(output.parent.glob("*.json"))) + 5 + 1 > MAX_BACKUP_EVIDENCE_FILES:
        raise ValueError("This probe would exceed the selected backup evidence-file bound.")
    state: dict[str, Any] = {
        "schema": "macfit-attention-preferences-v1",
        "status": "running",
        "protocol": plan,
        "source_sha256": source_sha256(),
        "started_at": benchmark.utc_now(),
        "arms": [],
        "limitations": [
            "Requested library preferences may use another flash implementation.",
            "Different kernel-name sets do not alone establish distinct attention libraries.",
            "Synthetic inference timings and token hashes do not establish answer quality.",
            "The bounded profile covers the first coordinate, outside scored timing samples.",
        ],
    }
    deadline = min(
        time.monotonic() + plan["walltime_seconds"],
        time.monotonic()
        + max(0, (benchmark.parse_stop_at(plan["stop_at"]) - datetime.now(UTC)).total_seconds()),
    )
    benchmark.write_atomic(output, state)
    try:
        for arm in ARMS:
            record: dict[str, Any] = {"preference": arm, "status": "running"}
            state["arms"].append(record)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                record["status"] = "skipped_deadline"
                benchmark.write_atomic(output, state)
                continue
            arm_output = output.with_name(f"{output.stem}-{arm}.json")
            arm_cutoff = min(
                benchmark.parse_stop_at(plan["stop_at"]),
                datetime.now(UTC) + timedelta(seconds=remaining),
            )
            arm_plan = {
                **plan,
                "walltime_seconds": max(1, math.ceil(remaining)),
                "stop_at": arm_cutoff.isoformat(),
            }
            command = (worker_commands or {}).get(arm) or [
                sys.executable,
                "-m",
                "macfit_training.attention_probe",
                "--child-arm",
                arm,
                "--child-config",
            ]
            benchmark.write_atomic(output, state)
            document = benchmark.run_campaign(arm_plan, arm_output, worker_command=command)
            document["benchmark"] = "rocm_flash_attention_preference_arm"
            document["preference_requested"] = arm
            benchmark.write_atomic(arm_output, document)
            record.update(
                status=document["status"],
                evidence_file=arm_output.name,
                evidence_sha256=hashlib.sha256(arm_output.read_bytes()).hexdigest(),
            )
            if document["models"]:
                record["model"] = read_verified_model(output.parent, document["models"][0])
            if any(event.get("code") == "attention_unsupported" for event in document["events"]):
                record["status"] = "unsupported"
            benchmark.write_atomic(output, state)
        state["comparison"] = comparison(state["arms"])
        state["status"] = (
            "succeeded"
            if all(arm["status"] == "succeeded" for arm in state["arms"])
            else "unsupported"
            if all(arm["status"] == "unsupported" for arm in state["arms"])
            else "incomplete"
        )
    except BaseException as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error_type"] = type(error).__name__
        raise
    finally:
        state["finished_at"] = benchmark.utc_now()
        benchmark.write_atomic(output, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(ALLOWED_MODELS), default="qwen3-8b")
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--prompt-tokens", default="2048,8192")
    parser.add_argument("--walltime-seconds", type=int, default=1800)
    parser.add_argument("--model-timeout-seconds", type=int, default=900)
    parser.add_argument("--stop-at")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--child-arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--child-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.child_config:
        return _worker(json.loads(args.child_config.read_text()), args.child_arm)
    try:
        plan = validate_protocol(
            {
                "models": [args.model],
                "batch_sizes": benchmark.integer_grid(args.batch_sizes, 4),
                "prompt_tokens": benchmark.integer_grid(args.prompt_tokens, 8192),
                "decode_steps": 64,
                "warmup_repetitions": 2,
                "repetitions": 5,
                "seed": 42,
                "walltime_seconds": args.walltime_seconds,
                "model_timeout_seconds": args.model_timeout_seconds,
                "stop_at": args.stop_at,
            }
        )
        if args.validate_only:
            print(json.dumps(plan, indent=2))
            return 0
        if args.output is None:
            parser.error("A fresh --output is required.")

        def stop(_signal, _frame):
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, stop)
        result = run_probe(plan, args.output)
        return 0 if result["status"] == "succeeded" else 2
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
