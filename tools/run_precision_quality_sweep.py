#!/usr/bin/env python3
"""Evaluate each precision arm once with the provisional math-100 protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from amd_inference_opt.quality_policy import (
    ProvisionalMath100Policy,
    ProvisionalQualityMeasurement,
)
from amd_inference_opt.resource_lock import exclusive_gpu_lock
from tools.q8_runtime_quality_eval import (
    EvaluationError,
    audit_toolchain,
    canonical_sha256,
    compare_math_correctness,
    evaluate_runtime,
    load_math_items,
    require_regular_file,
    resolve_tools,
    select_greedy_canary,
    select_math_items,
    serialize_multiple_choice,
    sha256_file,
    write_json_atomic,
)

SCHEMA = "gpuopt.precision-quality-sweep.v1"


def _parse_arm(value: str) -> tuple[str, Path]:
    fields = value.split("=", 1)
    if len(fields) != 2 or not all(fields):
        raise argparse.ArgumentTypeError("arm must be NAME=MODEL.gguf")
    return fields[0], Path(fields[1])


def _measurement(result: dict[str, object], protocol_hash: str):
    return ProvisionalQualityMeasurement(
        math_correct=int(result["math_correct"]),
        math_total=int(result["math_total"]),
        perplexity=float(result["perplexity"]),
        greedy_correct=int(result["greedy_correct"]),
        greedy_total=int(result["greedy_total"]),
        protocol_hash=protocol_hash,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--arm", action="append", required=True, type=_parse_arm)
    parser.add_argument("--baseline-arm", default="q8")
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--greedy-context-size", type=int, default=2048)
    args = parser.parse_args()
    if len({name for name, _ in args.arm}) != len(args.arm):
        parser.error("arm names must be unique")
    if args.baseline_arm not in {name for name, _ in args.arm}:
        parser.error("baseline arm is missing")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark = require_regular_file(args.binary, label="llama-bench", executable=True)
    cli, perplexity = resolve_tools(benchmark, label="shared")
    libraries = audit_toolchain(benchmark, cli, perplexity)
    manifest_path = require_regular_file(
        args.fixture_manifest, label="quality fixture manifest"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    math_meta = manifest["math"]
    ppl_meta = manifest["perplexity"]
    math_path = require_regular_file(
        manifest_path.parent / math_meta["path"], label="math fixture"
    )
    ppl_path = require_regular_file(
        manifest_path.parent / ppl_meta["path"], label="perplexity fixture"
    )
    if sha256_file(math_path) != math_meta["sha256"]:
        raise EvaluationError("math fixture hash mismatch")
    if sha256_file(ppl_path) != ppl_meta["sha256"]:
        raise EvaluationError("perplexity fixture hash mismatch")
    source_items = load_math_items(math_path)
    greedy_items, greedy_canary = select_greedy_canary(source_items, 8)
    items, math_selection = select_math_items(source_items, 100)
    policy = ProvisionalMath100Policy()
    if math_selection["selected_ids_sha256"] != policy.protocol.order_sha256:
        raise EvaluationError("math-100 selection differs from the framework policy")
    math_binary = output_root / "mmlu-math-100.bin"
    math_binary.write_bytes(serialize_multiple_choice(items))
    coordinates = {
        "ngl": 999,
        "threads": 12,
        "context_size": args.context_size,
        "greedy_context_size": args.greedy_context_size,
        "batch_size": 512,
        "ubatch_size": 512,
        "ppl_chunks": 32,
        "greedy_count": 8,
        "runtime_args": [],
        "math_selection": math_selection,
        "greedy_canary": greedy_canary,
        "quality_policy_hash": policy.protocol.protocol_hash,
    }
    coordinate_hash = canonical_sha256(coordinates)
    lock_path = args.gpu_lock or output_root.parent / "gpu-0.lock"
    results: dict[str, dict[str, object]] = {}
    with exclusive_gpu_lock(lock_path):
        for name, raw_model in args.arm:
            model = require_regular_file(raw_model, label=f"{name} model")
            model_sha = sha256_file(model)
            terminal = output_root / name / "result.json"
            if terminal.is_file():
                existing = json.loads(terminal.read_text(encoding="utf-8"))
                if (
                    existing.get("model_sha256") != model_sha
                    or existing.get("coordinate_hash") != coordinate_hash
                ):
                    raise EvaluationError(f"existing {name} quality result drifted")
                results[name] = existing["measurement"]
                continue
            measurement = evaluate_runtime(
                name=name,
                benchmark_binary=benchmark,
                cli=cli,
                perplexity=perplexity,
                runtime_libraries=libraries,
                runtime_args=(),
                model=model,
                ppl_file=ppl_path,
                math_binary=math_binary,
                math_total=100,
                greedy_items=greedy_items,
                artifact_dir=output_root,
                ngl=999,
                threads=12,
                context_size=args.context_size,
                batch_size=512,
                ubatch_size=512,
                ppl_chunks=32,
                greedy_count=8,
                timeout_seconds=args.timeout_seconds,
                greedy_context_size=args.greedy_context_size,
            )
            write_json_atomic(
                terminal,
                {
                    "schema": "gpuopt.precision-quality-arm.v1",
                    "arm": name,
                    "model": str(model),
                    "model_sha256": model_sha,
                    "coordinate_hash": coordinate_hash,
                    "quality_policy_hash": policy.protocol.protocol_hash,
                    "measurement": measurement,
                },
            )
            results[name] = measurement

    baseline = results[args.baseline_arm]
    gates = {}
    for name, candidate in results.items():
        if name == args.baseline_arm:
            continue
        quality = policy.evaluate(
            _measurement(baseline, policy.protocol.protocol_hash),
            _measurement(candidate, policy.protocol.protocol_hash),
            protocol=policy.protocol,
        )
        gates[name] = {
            "quality": quality.to_dict(),
            "paired_math": compare_math_correctness(
                str(baseline["math_correctness_bits"]),
                str(candidate["math_correctness_bits"]),
            ),
            "ppl_regression_fraction": (
                float(candidate["perplexity"]) / float(baseline["perplexity"]) - 1
            ),
            "math_accuracy_delta": (
                float(candidate["math_accuracy"]) - float(baseline["math_accuracy"])
            ),
        }
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "baseline_arm": args.baseline_arm,
        "coordinate_hash": coordinate_hash,
        "coordinates": coordinates,
        "quality_policy": policy.protocol.to_dict(),
        "arms": results,
        "gates": gates,
        "production_ready": False,
    }
    write_json_atomic(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
