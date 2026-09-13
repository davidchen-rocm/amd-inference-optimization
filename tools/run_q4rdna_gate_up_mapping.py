#!/usr/bin/env python3
"""Screen Q4_RDNA fused gate/up split widths with one fixed runtime binary."""

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
from amd_inference_opt.q4rdna_tuning import (
    GateUpBenchmarkMeasurement,
    MappingScreenOutcome,
    build_gate_up_performance_model,
    drop_initial_warmup_samples,
    gate_up_mapping_arm,
    screen_gate_up_mapping,
)
from amd_inference_opt.resource_lock import exclusive_gpu_lock


class GateUpRunError(RuntimeError):
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


def _benchmark_argv(
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


def _activation(stderr: str) -> tuple[bool, dict[str, int]]:
    loaded = "Q4_RDNA: loaded" in stderr and "device 0" in stderr
    pattern = re.compile(
        r"hits by rows x columns: 12288x4096=(\d+), 4096x12288=(\d+), "
        r"4096x4096=(\d+), 1024x4096=(\d+), other=(\d+)"
    )
    match = pattern.search(stderr)
    if match is None:
        return False, {}
    names = ("12288x4096", "4096x12288", "4096x4096", "1024x4096", "other")
    counts = dict(zip(names, (int(value) for value in match.groups()), strict=True))
    hotspot_only = counts["12288x4096"] > 0 and all(
        counts[name] == 0 for name in names if name != "12288x4096"
    )
    return loaded and hotspot_only, counts


def _validate_row(row: dict[str, Any], *, generation_tokens: int) -> None:
    checks = {
        "decode coordinate": row.get("n_prompt") == 0
        and row.get("n_gen") == generation_tokens,
        "ROCm device": row.get("devices") == "ROCm0" and row.get("main_gpu") == 0,
        "GPU offload": row.get("n_gpu_layers") == 999,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise GateUpRunError("benchmark coordinate failed: " + ", ".join(failures))


def _run_arm(
    *,
    label: str,
    split_waves: int,
    binary: Path,
    model: Path,
    sidecar: Path,
    binary_sha256: str,
    model_sha256: str,
    sidecar_sha256: str,
    output_root: Path,
    generation_tokens: int,
    repetitions: int,
    warmup_samples: int,
    timeout_seconds: float,
) -> GateUpBenchmarkMeasurement:
    root = output_root / label
    terminal = root / "result.json"
    if terminal.is_file():
        document = json.loads(terminal.read_text(encoding="utf-8"))
        expected = {
            "split_waves": split_waves,
            "binary_sha256": binary_sha256,
            "model_sha256": model_sha256,
            "sidecar_sha256": sidecar_sha256,
            "generation_tokens": generation_tokens,
            "repetitions": repetitions,
            "warmup_samples_dropped": warmup_samples,
        }
        if any(document.get(key) != value for key, value in expected.items()):
            raise GateUpRunError(f"existing arm belongs to another coordinate: {terminal}")
        return GateUpBenchmarkMeasurement(
            arm_id=label,
            split_waves=split_waves,
            samples_tokens_per_second=tuple(document["samples_tokens_per_second"]),
            activation_verified=bool(document["activation_verified"]),
        )

    arm = gate_up_mapping_arm(
        split_waves,
        sidecar_path=sidecar,
        rocm_library_paths=(str(binary.parent), "/opt/rocm/core-7.14/lib"),
    )
    argv = _benchmark_argv(
        binary,
        model,
        generation_tokens=generation_tokens,
        repetitions=repetitions + warmup_samples,
    )
    root.mkdir(parents=True, exist_ok=True)
    result = CommandRunner().run(
        argv,
        cwd=binary.parent,
        env=arm.environment,
        unset_env=arm.unset_environment,
        timeout_seconds=timeout_seconds,
        stdout_path=root / "benchmark.stdout.json",
        stderr_path=root / "benchmark.stderr.log",
    )
    _write_atomic(root / "command.json", _command_document(result))
    if not result.succeeded:
        raise GateUpRunError(
            f"{label} failed: exit={result.exit_code} timeout={result.timed_out}"
        )
    parsed = parse_llama_bench_json(result.stdout)
    if len(parsed.records) != 1:
        raise GateUpRunError(f"{label} produced more than one benchmark coordinate")
    record = parsed.records[0]
    _validate_row(record.raw, generation_tokens=generation_tokens)
    expected_samples = repetitions + warmup_samples
    if record.sample_count != expected_samples:
        raise GateUpRunError(
            f"{label} returned {record.sample_count} samples, expected {expected_samples}"
        )
    activation_verified, shape_hits = _activation(result.stderr)
    source_samples = record.samples_tokens_per_second
    scored_samples = drop_initial_warmup_samples(
        source_samples,
        warmup_samples=warmup_samples,
    )
    measurement = GateUpBenchmarkMeasurement(
        arm_id=label,
        split_waves=split_waves,
        samples_tokens_per_second=scored_samples,
        activation_verified=activation_verified,
    )
    _write_atomic(
        terminal,
        {
            "schema": "gpuopt.q4rdna-gate-up-mapping-arm.v1",
            "label": label,
            "split_waves": split_waves,
            "binary": str(binary),
            "binary_sha256": binary_sha256,
            "model": str(model),
            "model_sha256": model_sha256,
            "sidecar": str(sidecar),
            "sidecar_sha256": sidecar_sha256,
            "generation_tokens": generation_tokens,
            "repetitions": repetitions,
            "warmup_samples_dropped": warmup_samples,
            "source_samples_tokens_per_second": list(source_samples),
            "environment": dict(sorted(arm.environment.items())),
            "unset_environment": list(arm.unset_environment),
            "request_sha256": result.request_sha256,
            "shape_hits": shape_hits,
            **measurement.to_dict(),
            "raw_row": record.raw,
        },
    )
    return measurement


def _combined_baseline(
    before: GateUpBenchmarkMeasurement,
    after: GateUpBenchmarkMeasurement,
) -> GateUpBenchmarkMeasurement:
    return GateUpBenchmarkMeasurement(
        arm_id="gate-up-split-8-combined",
        split_waves=8,
        samples_tokens_per_second=(
            *before.samples_tokens_per_second,
            *after.samples_tokens_per_second,
        ),
        activation_verified=before.activation_verified and after.activation_verified,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--theoretical-bandwidth-gbps", type=float, default=640.0)
    parser.add_argument("--baseline-kernel-us", type=float, default=89.8410385443583)
    parser.add_argument("--hotspot-share-percent", type=float, default=40.597033)
    args = parser.parse_args()
    if (
        args.generation_tokens <= 0
        or args.repetitions < 3
        or args.warmup_samples < 1
    ):
        raise GateUpRunError("generation tokens must be positive and repetitions at least three")

    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    sidecar = args.sidecar.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    binary_sha256 = sha256_file(binary)
    model_sha256 = sha256_file(model)
    sidecar_sha256 = sha256_file(sidecar)
    model_document = build_gate_up_performance_model(
        measured_kernel_us=args.baseline_kernel_us,
        hotspot_gpu_time_share_percent=args.hotspot_share_percent,
        theoretical_bandwidth_gbps=args.theoretical_bandwidth_gbps,
    ).to_dict()
    _write_atomic(output_root / "performance-model.json", model_document)

    lock_path = args.gpu_lock or output_root.parent / "gpu-0.lock"
    with exclusive_gpu_lock(lock_path):
        before = _run_arm(
            label="baseline-split-8-before",
            split_waves=8,
            binary=binary,
            model=model,
            sidecar=sidecar,
            binary_sha256=binary_sha256,
            model_sha256=model_sha256,
            sidecar_sha256=sidecar_sha256,
            output_root=output_root,
            generation_tokens=args.generation_tokens,
            repetitions=args.repetitions,
            warmup_samples=args.warmup_samples,
            timeout_seconds=args.timeout_seconds,
        )
        candidates = {
            split_waves: _run_arm(
                label=f"candidate-split-{split_waves}",
                split_waves=split_waves,
                binary=binary,
                model=model,
                sidecar=sidecar,
                binary_sha256=binary_sha256,
                model_sha256=model_sha256,
                sidecar_sha256=sidecar_sha256,
                output_root=output_root,
                generation_tokens=args.generation_tokens,
                repetitions=args.repetitions,
                warmup_samples=args.warmup_samples,
                timeout_seconds=args.timeout_seconds,
            )
            for split_waves in (4, 2)
        }
        after = _run_arm(
            label="baseline-split-8-after",
            split_waves=8,
            binary=binary,
            model=model,
            sidecar=sidecar,
            binary_sha256=binary_sha256,
            model_sha256=model_sha256,
            sidecar_sha256=sidecar_sha256,
            output_root=output_root,
            generation_tokens=args.generation_tokens,
            repetitions=args.repetitions,
            warmup_samples=args.warmup_samples,
            timeout_seconds=args.timeout_seconds,
        )

    baseline = _combined_baseline(before, after)
    drift_percent = abs(
        after.mean_tokens_per_second / before.mean_tokens_per_second - 1.0
    ) * 100.0
    screens = {
        f"split-{split_waves}": screen_gate_up_mapping(baseline, candidate).to_dict()
        for split_waves, candidate in candidates.items()
    }
    if drift_percent > 1.0:
        for screen in screens.values():
            screen["outcome"] = MappingScreenOutcome.INCONCLUSIVE
            screen["improvement_percent"] = None
            screen["reasons"] = ["split-8 bracketing baseline drift exceeded 1%"]
    promoted = [
        split_waves
        for split_waves in (4, 2)
        if screens[f"split-{split_waves}"]["outcome"] == MappingScreenOutcome.PROMOTE
    ]
    summary = {
        "schema": "gpuopt.q4rdna-gate-up-mapping-screen.v1",
        "hypothesis": (
            "The accepted split-8 mapping may oversupply waves for the fused 12288x4096 "
            "gate/up kernel; reducing only split_waves can lower reduction/LDS overhead."
        ),
        "single_variable": "LLAMA_Q4_RDNA_COOP",
        "fixed_coordinates": {
            "binary_sha256": binary_sha256,
            "model_sha256": model_sha256,
            "sidecar_sha256": sidecar_sha256,
            "scope": "hotspot",
            "generation_tokens": args.generation_tokens,
            "repetitions_per_arm": args.repetitions,
            "same_coordinate_warmup_samples_dropped": args.warmup_samples,
        },
        "performance_model": model_document,
        "baseline_before": before.to_dict(),
        "baseline_after": after.to_dict(),
        "combined_baseline": baseline.to_dict(),
        "baseline_drift_percent": drift_percent,
        "candidates": {
            str(split_waves): measurement.to_dict()
            for split_waves, measurement in candidates.items()
        },
        "screens": screens,
        "promoted_for_profile_and_correctness": promoted,
        "stop_condition": (
            "Stop this mapping direction when no candidate improves stable tg128 by at least 0.5%."
        ),
    }
    _write_atomic(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
