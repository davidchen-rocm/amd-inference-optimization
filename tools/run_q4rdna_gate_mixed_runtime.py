#!/usr/bin/env python3
"""Bracketed real-model A/B for Q4 versus tiered Q2/Q3/Q4 FFN gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any

from amd_inference_opt.command import CommandResult, CommandRunner
from amd_inference_opt.llama_cpp import parse_llama_bench_json, sha256_file
from amd_inference_opt.protocol import BASELINE_UNSET_ENVIRONMENT, amd_runtime_environment
from amd_inference_opt.resource_lock import exclusive_gpu_lock


class MixedRuntimeError(RuntimeError):
    pass


def _write_atomic(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _argv(binary: Path, model: Path, generation_tokens: int, repetitions: int) -> list[str]:
    return [
        str(binary), "-m", str(model), "-p", "0", "-n", str(generation_tokens),
        "-b", "2048", "-ub", "512", "-t", "12", "-r", str(repetitions),
        "-ngl", "999", "-mg", "0", "-dev", "ROCm0", "-o", "json", "-oe", "none",
    ]


def _activation(stderr: str, *, expected_q2: int, expected_q3: int) -> tuple[bool, int, int]:
    loaded = re.search(r"Q4_RDNA: loaded 252 tensors \((\d+) Q2, (\d+) Q3\)", stderr)
    hotspot = re.search(r"12288x4096=(\d+)", stderr)
    q2_count = int(loaded.group(1)) if loaded else -1
    q3_count = int(loaded.group(2)) if loaded else -1
    active = bool(
        loaded
        and hotspot
        and int(hotspot.group(1)) > 0
        and q2_count == expected_q2
        and q3_count == expected_q3
    )
    return active, q2_count, q3_count


def _validate_row(row: dict[str, Any], generation_tokens: int) -> None:
    if not (
        row.get("n_prompt") == 0
        and row.get("n_gen") == generation_tokens
        and row.get("devices") == "ROCm0"
        and row.get("main_gpu") == 0
        and row.get("n_gpu_layers") == 999
    ):
        raise MixedRuntimeError("benchmark output does not match the locked coordinate")


def _command_document(result: CommandResult) -> dict[str, object]:
    document = result.to_dict()
    document["stdout_sha256"] = hashlib.sha256(result.stdout.encode()).hexdigest()
    document["stderr_sha256"] = hashlib.sha256(result.stderr.encode()).hexdigest()
    return document


def _runtime_library_hashes(binary: Path) -> dict[str, str]:
    names = ("libggml-hip.so", "libggml.so", "libllama.so", "libllama-common.so")
    hashes = {}
    for name in names:
        path = binary.parent / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    if "libggml-hip.so" not in hashes:
        raise MixedRuntimeError("runtime HIP library is missing")
    return hashes


def _measurement(
    *,
    label: str,
    samples: tuple[float, ...],
    active: bool,
    q2_count: int,
    q3_count: int,
) -> dict[str, object]:
    average = statistics.fmean(samples)
    cv_percent = (
        statistics.stdev(samples) / average * 100.0
        if len(samples) > 1
        else 0.0
    )
    return {
        "arm_id": label,
        "samples_tokens_per_second": samples,
        "mean_tokens_per_second": average,
        "cv_percent": cv_percent,
        "activation_verified": active,
        "q2_tensor_count": q2_count,
        "q3_tensor_count": q3_count,
    }


def _run_arm(
    *,
    label: str,
    expected_q2: int,
    expected_q3: int,
    binary: Path,
    model: Path,
    sidecar: Path,
    output_root: Path,
    generation_tokens: int,
    repetitions: int,
    warmup_samples: int,
    timeout_seconds: float,
) -> dict[str, object]:
    root = output_root / label
    environment = amd_runtime_environment(
        rocm_library_paths=(str(binary.parent), "/opt/rocm/core-7.14/lib", "/opt/rocm/lib"),
        extra={"LLAMA_Q4_RDNA_SIDECAR": str(sidecar), "LLAMA_Q4_RDNA_TRACE": "1"},
    )
    unset = tuple(name for name in BASELINE_UNSET_ENVIRONMENT if name not in environment)
    result = CommandRunner().run(
        _argv(binary, model, generation_tokens, repetitions + warmup_samples),
        cwd=binary.parent,
        env=environment,
        unset_env=unset,
        timeout_seconds=timeout_seconds,
        stdout_path=root / "benchmark.stdout.json",
        stderr_path=root / "benchmark.stderr.log",
    )
    _write_atomic(root / "command.json", _command_document(result))
    if not result.succeeded:
        raise MixedRuntimeError(
            f"{label} failed: exit={result.exit_code} timeout={result.timed_out}"
        )
    parsed = parse_llama_bench_json(result.stdout)
    if len(parsed.records) != 1:
        raise MixedRuntimeError(f"{label} did not produce one coordinate")
    record = parsed.records[0]
    _validate_row(record.raw, generation_tokens)
    expected_samples = repetitions + warmup_samples
    if record.sample_count != expected_samples:
        raise MixedRuntimeError(
            f"{label} returned {record.sample_count} samples, expected {expected_samples}"
        )
    samples = tuple(record.samples_tokens_per_second[warmup_samples:])
    active, q2_count, q3_count = _activation(
        result.stderr, expected_q2=expected_q2, expected_q3=expected_q3
    )
    measurement = _measurement(
        label=label,
        samples=samples,
        active=active,
        q2_count=q2_count,
        q3_count=q3_count,
    )
    _write_atomic(
        root / "result.json",
        {
            "schema": "gpuopt.q4rdna-gate-mixed-runtime-arm.v1",
            "binary": str(binary),
            "binary_sha256": sha256_file(binary),
            "runtime_library_hashes": _runtime_library_hashes(binary),
            "model": str(model),
            "model_sha256": sha256_file(model),
            "sidecar": str(sidecar),
            "sidecar_sha256": sha256_file(sidecar),
            "generation_tokens": generation_tokens,
            "repetitions": repetitions,
            "warmup_samples_dropped": warmup_samples,
            "request_sha256": result.request_sha256,
            **measurement,
            "raw_row": record.raw,
        },
    )
    return measurement


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--baseline-sidecar", required=True, type=Path)
    parser.add_argument("--candidate-sidecar", required=True, type=Path)
    parser.add_argument("--candidate-q2-count", type=int, default=20)
    parser.add_argument("--candidate-q3-count", type=int, default=8)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--generation-tokens", required=True, type=int)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--gpu-lock", required=True, type=Path)
    args = parser.parse_args()
    if args.repetitions < 3 or args.warmup_samples < 1:
        raise MixedRuntimeError("invalid benchmark repetition policy")
    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    baseline_sidecar = args.baseline_sidecar.resolve(strict=True)
    candidate_sidecar = args.candidate_sidecar.resolve(strict=True)
    output_root = args.output_root.resolve()
    common = {
        "binary": binary,
        "model": model,
        "output_root": output_root,
        "generation_tokens": args.generation_tokens,
        "repetitions": args.repetitions,
        "warmup_samples": args.warmup_samples,
        "timeout_seconds": args.timeout_seconds,
    }
    with exclusive_gpu_lock(args.gpu_lock):
        before = _run_arm(
            label="q4-before", expected_q2=0, expected_q3=0,
            sidecar=baseline_sidecar, **common
        )
        candidate = _run_arm(
            label="q2q3q4-candidate",
            expected_q2=args.candidate_q2_count,
            expected_q3=args.candidate_q3_count,
            sidecar=candidate_sidecar, **common
        )
        after = _run_arm(
            label="q4-after", expected_q2=0, expected_q3=0,
            sidecar=baseline_sidecar, **common
        )
    before_mean = float(before["mean_tokens_per_second"])
    after_mean = float(after["mean_tokens_per_second"])
    baseline_samples = tuple(before["samples_tokens_per_second"]) + tuple(
        after["samples_tokens_per_second"]
    )
    baseline = _measurement(
        label="q4-combined",
        samples=baseline_samples,
        active=bool(before["activation_verified"] and after["activation_verified"]),
        q2_count=0,
        q3_count=0,
    )
    baseline_mean = float(baseline["mean_tokens_per_second"])
    candidate_mean = float(candidate["mean_tokens_per_second"])
    drift = abs(after_mean / before_mean - 1.0) * 100.0
    improvement = (candidate_mean / baseline_mean - 1.0) * 100.0
    if drift > 1.0:
        outcome = "INCONCLUSIVE"
        reasons = ["bracketing baseline drift exceeded 1%"]
    elif not baseline["activation_verified"] or not candidate["activation_verified"]:
        outcome = "REJECT"
        reasons = ["runtime activation did not match the expected mixed tensor counts"]
    elif max(float(baseline["cv_percent"]), float(candidate["cv_percent"])) > 2.0:
        outcome = "INCONCLUSIVE"
        reasons = ["runtime measurement exceeded the 2% CV threshold"]
    elif improvement >= 2.0:
        outcome = "PROMOTE"
        reasons = ["mixed representation passed the real-model performance screen"]
    else:
        outcome = "REJECT"
        reasons = ["mixed representation did not reach the 2% runtime threshold"]
    summary = {
        "schema": "gpuopt.q4rdna-gate-mixed-runtime-screen.v1",
        "hypothesis": (
            "Tiering routed FFN tensors across Q2, Q3, and Q4 reduces model decode "
            "traffic while all non-selected paths remain fixed."
        ),
        "single_variable": {
            "q2_ffn_gate_tensors": args.candidate_q2_count,
            "q3_ffn_gate_tensors": args.candidate_q3_count,
            "remaining_routed_tensors": "Q4",
        },
        "fixed_coordinates": {
            "binary_sha256": sha256_file(binary),
            "runtime_library_hashes": _runtime_library_hashes(binary),
            "model_sha256": sha256_file(model),
            "baseline_sidecar_sha256": sha256_file(baseline_sidecar),
            "candidate_sidecar_sha256": sha256_file(candidate_sidecar),
            "generation_tokens": args.generation_tokens,
            "repetitions_per_arm": args.repetitions,
            "warmup_samples_dropped": args.warmup_samples,
        },
        "baseline_before": before,
        "candidate": candidate,
        "baseline_after": after,
        "combined_baseline": baseline,
        "baseline_drift_percent": drift,
        "improvement_percent": improvement,
        "screen": {"outcome": outcome, "reasons": reasons},
        "sidecar_bytes": {
            "baseline": baseline_sidecar.stat().st_size,
            "candidate": candidate_sidecar.stat().st_size,
            "reduction_percent": (
                1.0 - candidate_sidecar.stat().st_size / baseline_sidecar.stat().st_size
            ) * 100.0,
        },
        "stop_condition": (
            "Do not profile or run quality evaluation unless both tg coordinates "
            "improve by at least 2%."
        ),
    }
    _write_atomic(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
