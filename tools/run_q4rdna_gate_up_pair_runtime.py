#!/usr/bin/env python3
"""Bracketed real-model A/B for the Q4_RDNA paired gate/up layout."""

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
    GateUpPairRuntimeMeasurement,
    MappingScreenOutcome,
    drop_initial_warmup_samples,
    screen_gate_up_pair_runtime,
)
from amd_inference_opt.resource_lock import exclusive_gpu_lock


class PairRuntimeError(RuntimeError):
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


def _command_document(result: CommandResult) -> dict[str, object]:
    document = result.to_dict()
    document["stdout_sha256"] = hashlib.sha256(result.stdout.encode()).hexdigest()
    document["stderr_sha256"] = hashlib.sha256(result.stderr.encode()).hexdigest()
    return document


def _argv(
    binary: Path,
    model: Path,
    *,
    generation_tokens: int,
    repetitions: int,
) -> list[str]:
    return [
        str(binary),
        "-m",
        str(model),
        "-p",
        "0",
        "-n",
        str(generation_tokens),
        "-b",
        "2048",
        "-ub",
        "512",
        "-t",
        "12",
        "-r",
        str(repetitions),
        "-ngl",
        "999",
        "-mg",
        "0",
        "-dev",
        "ROCm0",
        "-o",
        "json",
        "-oe",
        "none",
    ]


def _activation(stderr: str, *, paired: bool) -> tuple[bool, int]:
    loaded = "Q4_RDNA: loaded" in stderr and "device 0" in stderr
    pair_match = re.search(r"Q4_RDNA: paired (\d+) gate/up tensors", stderr)
    pair_count = int(pair_match.group(1)) if pair_match else 0
    hit_match = re.search(r"12288x4096=(\d+)", stderr)
    hotspot = hit_match is not None and int(hit_match.group(1)) > 0
    expected = 36 if paired else 0
    return loaded and hotspot and pair_count == expected, pair_count


def _validate_row(row: dict[str, Any], *, generation_tokens: int) -> None:
    valid = (
        row.get("n_prompt") == 0
        and row.get("n_gen") == generation_tokens
        and row.get("devices") == "ROCm0"
        and row.get("main_gpu") == 0
        and row.get("n_gpu_layers") == 999
    )
    if not valid:
        raise PairRuntimeError("benchmark output does not match the locked coordinate")


def _run_arm(
    *,
    label: str,
    paired: bool,
    binary: Path,
    model: Path,
    sidecar: Path,
    hashes: dict[str, str],
    output_root: Path,
    generation_tokens: int,
    repetitions: int,
    warmup_samples: int,
    timeout_seconds: float,
    scope: str,
) -> GateUpPairRuntimeMeasurement:
    root = output_root / label
    root.mkdir(parents=True, exist_ok=True)
    extra = {
        "LLAMA_Q4_RDNA_SIDECAR": str(sidecar),
        "LLAMA_Q4_RDNA_TRACE": "1",
    }
    if scope == "hotspot":
        extra["LLAMA_Q4_RDNA_SCOPE"] = "hotspot"
        extra["LLAMA_Q4_RDNA_COOP"] = "8"
    if paired:
        extra["LLAMA_Q4_RDNA_GATE_PAIR"] = "1"
    environment = amd_runtime_environment(
        rocm_library_paths=(
            str(binary.parent),
            "/opt/rocm/core-7.14/lib",
            "/opt/rocm/lib",
        ),
        extra=extra,
    )
    unset = tuple(name for name in BASELINE_UNSET_ENVIRONMENT if name not in environment)
    argv = _argv(
        binary,
        model,
        generation_tokens=generation_tokens,
        repetitions=repetitions + warmup_samples,
    )
    result = CommandRunner().run(
        argv,
        cwd=binary.parent,
        env=environment,
        unset_env=unset,
        timeout_seconds=timeout_seconds,
        stdout_path=root / "benchmark.stdout.json",
        stderr_path=root / "benchmark.stderr.log",
    )
    _write_atomic(root / "command.json", _command_document(result))
    if not result.succeeded:
        raise PairRuntimeError(
            f"{label} failed: exit={result.exit_code} timeout={result.timed_out}"
        )
    parsed = parse_llama_bench_json(result.stdout)
    if len(parsed.records) != 1:
        raise PairRuntimeError(f"{label} did not produce one coordinate")
    record = parsed.records[0]
    _validate_row(record.raw, generation_tokens=generation_tokens)
    expected_samples = repetitions + warmup_samples
    if record.sample_count != expected_samples:
        raise PairRuntimeError(
            f"{label} returned {record.sample_count} samples, expected {expected_samples}"
        )
    active, pair_count = _activation(result.stderr, paired=paired)
    scored = drop_initial_warmup_samples(
        record.samples_tokens_per_second,
        warmup_samples=warmup_samples,
    )
    measurement = GateUpPairRuntimeMeasurement(
        arm_id=label,
        paired_layout=paired,
        samples_tokens_per_second=scored,
        activation_verified=active,
        paired_tensor_count=pair_count,
    )
    _write_atomic(
        root / "result.json",
        {
            "schema": "gpuopt.q4rdna-gate-up-pair-runtime-arm.v1",
            "binary": str(binary),
            "model": str(model),
            "sidecar": str(sidecar),
            **hashes,
            "generation_tokens": generation_tokens,
            "repetitions": repetitions,
            "warmup_samples_dropped": warmup_samples,
            "environment": dict(sorted(environment.items())),
            "unset_environment": list(unset),
            "source_samples_tokens_per_second": list(
                record.samples_tokens_per_second
            ),
            "request_sha256": result.request_sha256,
            **measurement.to_dict(),
            "raw_row": record.raw,
        },
    )
    return measurement


def _combined(
    before: GateUpPairRuntimeMeasurement,
    after: GateUpPairRuntimeMeasurement,
) -> GateUpPairRuntimeMeasurement:
    return GateUpPairRuntimeMeasurement(
        arm_id="baseline-combined",
        paired_layout=False,
        samples_tokens_per_second=(
            *before.samples_tokens_per_second,
            *after.samples_tokens_per_second,
        ),
        activation_verified=before.activation_verified and after.activation_verified,
        paired_tensor_count=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--scope", choices=("hotspot", "full"), default="hotspot")
    args = parser.parse_args()
    if args.generation_tokens <= 0 or args.repetitions < 3 or args.warmup_samples < 1:
        raise PairRuntimeError("invalid benchmark repetitions or warmup")
    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    sidecar = args.sidecar.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    hashes = {
        "binary_sha256": sha256_file(binary),
        "model_sha256": sha256_file(model),
        "sidecar_sha256": sha256_file(sidecar),
    }
    common = dict(
        binary=binary,
        model=model,
        sidecar=sidecar,
        hashes=hashes,
        output_root=output_root,
        generation_tokens=args.generation_tokens,
        repetitions=args.repetitions,
        warmup_samples=args.warmup_samples,
        timeout_seconds=args.timeout_seconds,
        scope=args.scope,
    )
    lock_path = args.gpu_lock or output_root.parent / "gpu-0.lock"
    with exclusive_gpu_lock(lock_path):
        before = _run_arm(label="baseline-before", paired=False, **common)
        candidate = _run_arm(label="candidate-paired", paired=True, **common)
        after = _run_arm(label="baseline-after", paired=False, **common)
    baseline = _combined(before, after)
    drift = abs(after.mean_tokens_per_second / before.mean_tokens_per_second - 1) * 100
    screen = screen_gate_up_pair_runtime(baseline, candidate).to_dict()
    if drift > 1.0:
        screen.update(
            outcome=MappingScreenOutcome.INCONCLUSIVE,
            improvement_percent=None,
            reasons=["bracketing baseline drift exceeded 1%"],
        )
    summary = {
        "schema": "gpuopt.q4rdna-gate-up-pair-runtime-screen.v1",
        "hypothesis": (
            "A paired gate/up tile layout replaces two byte loads with one 16-bit load "
            "without changing split-K, quantization, or arithmetic order."
        ),
        "single_variable": "LLAMA_Q4_RDNA_GATE_PAIR",
        "fixed_coordinates": {
            **hashes,
            "scope": args.scope,
            "split_waves": 8 if args.scope == "hotspot" else "shape-specific-default",
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
            "Reject without profiling or quality when stable tg128 improvement is below 0.5%."
        ),
    }
    _write_atomic(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
