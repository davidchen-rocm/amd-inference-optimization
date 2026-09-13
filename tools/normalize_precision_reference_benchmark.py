#!/usr/bin/env python3
"""Normalize a completed superset run after fixing the warmup policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from amd_inference_opt.llama_cpp import parse_llama_bench_json
from tools.run_precision_reference_benchmark import _drop_warmup_samples

SCHEMA = "gpuopt.precision-reference-benchmark-normalized.v1"


class NormalizationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _normalize_arm(root: Path, arm: str, warmup_samples: int) -> dict[str, object]:
    arm_root = root / arm
    original = json.loads((arm_root / "result.json").read_text(encoding="utf-8"))
    pp_path = arm_root / "attempt-2-pp512.stdout.json"
    tg_path = arm_root / "attempt-2-decode.stdout.json"
    pp_command_path = arm_root / "attempt-2-pp512.command.json"
    tg_command_path = arm_root / "attempt-2-decode.command.json"
    pp_command = json.loads(pp_command_path.read_text(encoding="utf-8"))
    tg_command = json.loads(tg_command_path.read_text(encoding="utf-8"))
    for label, command in (("pp512", pp_command), ("decode", tg_command)):
        argv = command.get("argv")
        if not isinstance(argv, list) or "-r" not in argv:
            raise NormalizationError(f"{arm} {label} command is not bound")
        if argv[argv.index("-r") + 1] != "12":
            raise NormalizationError(f"{arm} {label} is not the reviewed 12-sample run")
        if command.get("exit_code") != 0 or command.get("timed_out") is not False:
            raise NormalizationError(f"{arm} {label} workload did not complete")
    records = (
        *parse_llama_bench_json(pp_path).records,
        *parse_llama_bench_json(tg_path).records,
    )
    scored = {
        record.test_id: _drop_warmup_samples(
            record, warmup_samples=warmup_samples
        )
        for record in records
    }
    if set(scored) != {"pp512", "tg128", "tg512"}:
        raise NormalizationError(f"{arm} benchmark coordinates are incomplete")
    if any(record.sample_count != 12 - warmup_samples for record in scored.values()):
        raise NormalizationError(f"{arm} scored sample count is incorrect")
    max_cv = max(record.coefficient_of_variation for record in scored.values())
    return {
        "schema": SCHEMA,
        "arm": arm,
        "binary": original["binary"],
        "binary_sha256": original["binary_sha256"],
        "model": original["model"],
        "model_sha256": original["model_sha256"],
        "model_size_bytes": original["model_size_bytes"],
        "protocol": {
            "source_repetitions": 12,
            "same_coordinate_warmup_samples_dropped": warmup_samples,
            "scored_samples": 12 - warmup_samples,
            "maximum_cv_percent": 2.0,
            "prompt_tokens": [512],
            "generation_tokens": [128, 512],
            "batch_size": 2048,
            "ubatch_size": 512,
            "threads": 12,
            "gpu_layers": 999,
            "device": "ROCm0",
        },
        "source_artifacts": {
            "pp512": {"path": str(pp_path), "sha256": _sha256(pp_path)},
            "decode": {"path": str(tg_path), "sha256": _sha256(tg_path)},
            "pp512_command": {
                "path": str(pp_command_path),
                "sha256": _sha256(pp_command_path),
            },
            "decode_command": {
                "path": str(tg_command_path),
                "sha256": _sha256(tg_command_path),
            },
        },
        "max_cv_percent": max_cv * 100,
        "stable": max_cv <= 0.02,
        "records": {key: value.to_dict() for key, value in sorted(scored.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--warmup-samples", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.warmup_samples < 12:
        parser.error("warmup-samples must be between 1 and 11")
    if len(args.arm) != len(set(args.arm)):
        parser.error("arms must be unique")
    root = args.input_root.resolve(strict=True)
    arms = {
        arm: _normalize_arm(root, arm, args.warmup_samples) for arm in args.arm
    }
    summary = {
        "schema": "gpuopt.precision-reference-benchmark-summary.v2",
        "normalization": {
            "reason": "short pp512 and tg128 require same-coordinate clock/graph warmup",
            "source_run_preserved": True,
            "warmup_samples_dropped": args.warmup_samples,
        },
        "arms": arms,
    }
    _write_atomic(args.output.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
