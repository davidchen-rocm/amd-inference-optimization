#!/usr/bin/env python3
"""Bracketed real-model A/B for Q4-up/Q4-gate versus Q4-up/Q3-gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from amd_inference_opt.command import CommandResult, CommandRunner
from amd_inference_opt.llama_cpp import parse_llama_bench_json, sha256_file
from amd_inference_opt.protocol import BASELINE_UNSET_ENVIRONMENT, amd_runtime_environment
from amd_inference_opt.q4rdna_tuning import (
    GateQ3RuntimeMeasurement,
    MappingScreenOutcome,
    drop_initial_warmup_samples,
    screen_gate_q3_runtime,
)
from amd_inference_opt.resource_lock import exclusive_gpu_lock


class GateQ3RuntimeError(RuntimeError):
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


def _activation(stderr: str, *, q3_gate: bool) -> tuple[bool, int]:
    loaded = re.search(r"Q4_RDNA: loaded 252 tensors \((\d+) Q3\)", stderr)
    hotspot = re.search(r"12288x4096=(\d+)", stderr)
    count = int(loaded.group(1)) if loaded else -1
    expected = 36 if q3_gate else 0
    return bool(loaded and hotspot and int(hotspot.group(1)) > 0 and count == expected), count


def _validate_row(row: dict[str, Any], generation_tokens: int) -> None:
    if not (
        row.get("n_prompt") == 0
        and row.get("n_gen") == generation_tokens
        and row.get("devices") == "ROCm0"
        and row.get("main_gpu") == 0
        and row.get("n_gpu_layers") == 999
    ):
        raise GateQ3RuntimeError("benchmark output does not match the locked coordinate")


def _command_document(result: CommandResult) -> dict[str, object]:
    document = result.to_dict()
    document["stdout_sha256"] = hashlib.sha256(result.stdout.encode()).hexdigest()
    document["stderr_sha256"] = hashlib.sha256(result.stderr.encode()).hexdigest()
    return document


def _run_arm(
    *,
    label: str,
    q3_gate: bool,
    binary: Path,
    model: Path,
    sidecar: Path,
    output_root: Path,
    generation_tokens: int,
    repetitions: int,
    warmup_samples: int,
    timeout_seconds: float,
) -> GateQ3RuntimeMeasurement:
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
        raise GateQ3RuntimeError(
            f"{label} failed: exit={result.exit_code} timeout={result.timed_out}"
        )
    parsed = parse_llama_bench_json(result.stdout)
    if len(parsed.records) != 1:
        raise GateQ3RuntimeError(f"{label} did not produce one coordinate")
    record = parsed.records[0]
    _validate_row(record.raw, generation_tokens)
    expected = repetitions + warmup_samples
    if record.sample_count != expected:
        raise GateQ3RuntimeError(
            f"{label} returned {record.sample_count} samples, expected {expected}"
        )
    active, q3_count = _activation(result.stderr, q3_gate=q3_gate)
    measurement = GateQ3RuntimeMeasurement(
        arm_id=label,
        q3_gate=q3_gate,
        samples_tokens_per_second=drop_initial_warmup_samples(
            record.samples_tokens_per_second, warmup_samples=warmup_samples
        ),
        activation_verified=active,
        q3_tensor_count=q3_count,
    )
    _write_atomic(
        root / "result.json",
        {
            "schema": "gpuopt.q4rdna-gate-q3-runtime-arm.v1",
            "binary": str(binary),
            "binary_sha256": sha256_file(binary),
            "model": str(model),
            "model_sha256": sha256_file(model),
            "sidecar": str(sidecar),
            "sidecar_sha256": sha256_file(sidecar),
            "generation_tokens": generation_tokens,
            "repetitions": repetitions,
            "warmup_samples_dropped": warmup_samples,
            "request_sha256": result.request_sha256,
            **measurement.to_dict(),
            "raw_row": record.raw,
        },
    )
    return measurement


def _combine(
    before: GateQ3RuntimeMeasurement,
    after: GateQ3RuntimeMeasurement,
) -> GateQ3RuntimeMeasurement:
    return GateQ3RuntimeMeasurement(
        arm_id="q4-gate-combined",
        q3_gate=False,
        samples_tokens_per_second=(
            *before.samples_tokens_per_second,
            *after.samples_tokens_per_second,
        ),
        activation_verified=before.activation_verified and after.activation_verified,
        q3_tensor_count=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--baseline-sidecar", required=True, type=Path)
    parser.add_argument("--candidate-sidecar", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--generation-tokens", required=True, type=int)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--gpu-lock", required=True, type=Path)
    args = parser.parse_args()
    if args.repetitions < 3 or args.warmup_samples < 1:
        raise GateQ3RuntimeError("invalid benchmark repetition policy")
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
        before = _run_arm(label="q4-before", q3_gate=False, sidecar=baseline_sidecar, **common)
        candidate = _run_arm(
            label="q3-candidate", q3_gate=True, sidecar=candidate_sidecar, **common
        )
        after = _run_arm(label="q4-after", q3_gate=False, sidecar=baseline_sidecar, **common)
    baseline = _combine(before, after)
    drift = abs(after.mean_tokens_per_second / before.mean_tokens_per_second - 1) * 100
    screen = screen_gate_q3_runtime(baseline, candidate).to_dict()
    if drift > 1.0:
        screen.update(
            outcome=MappingScreenOutcome.INCONCLUSIVE,
            improvement_percent=None,
            reasons=["bracketing baseline drift exceeded 1%"],
        )
    summary = {
        "schema": "gpuopt.q4rdna-gate-q3-runtime-screen.v1",
        "hypothesis": (
            "Replacing only FFN gate Q4 weights with Q3 reduces fused weight "
            "traffic enough to improve decode."
        ),
        "single_variable": "36 ffn_gate tensors: Q4_RDNA 4.25 bpw to Q3_RDNA 3.25 bpw",
        "fixed_coordinates": {
            "binary_sha256": sha256_file(binary),
            "model_sha256": sha256_file(model),
            "baseline_sidecar_sha256": sha256_file(baseline_sidecar),
            "candidate_sidecar_sha256": sha256_file(candidate_sidecar),
            "generation_tokens": args.generation_tokens,
            "repetitions_per_arm": args.repetitions,
            "warmup_samples_dropped": args.warmup_samples,
        },
        "baseline_before": before.to_dict(),
        "candidate": candidate.to_dict(),
        "baseline_after": after.to_dict(),
        "combined_baseline": baseline.to_dict(),
        "baseline_drift_percent": drift,
        "screen": screen,
        "stop_condition": (
            "Reject before profiling/quality if either tg coordinate improves "
            "by less than 2%."
        ),
    }
    _write_atomic(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
