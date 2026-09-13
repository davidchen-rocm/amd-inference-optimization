#!/usr/bin/env python3
"""Deterministic token-output canary for the Q4_RDNA paired gate/up path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from amd_inference_opt.command import CommandRunner
from amd_inference_opt.llama_cpp import sha256_file
from amd_inference_opt.protocol import BASELINE_UNSET_ENVIRONMENT, amd_runtime_environment
from amd_inference_opt.q4rdna_tuning import normalize_llama_cli_generation
from amd_inference_opt.resource_lock import exclusive_gpu_lock


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu-lock", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument(
        "--prompt",
        default="Solve exactly: If 7x + 5 = 40, what is x? Answer briefly.",
    )
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    sidecar = args.sidecar.resolve(strict=True)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    common_extra = {
        "LLAMA_Q4_RDNA_SIDECAR": str(sidecar),
        "LLAMA_Q4_RDNA_TRACE": "1",
    }
    argv = [
        str(binary),
        "-m",
        str(model),
        "-p",
        args.prompt,
        "-n",
        str(args.tokens),
        "-c",
        "512",
        "--seed",
        "20260822",
        "--temp",
        "0",
        "--no-conversation",
        "--no-display-prompt",
        "--simple-io",
        "--single-turn",
        "-ngl",
        "999",
        "-mg",
        "0",
        "-dev",
        "ROCm0",
        "-t",
        "12",
    ]
    arm_results: dict[str, dict[str, object]] = {}
    with exclusive_gpu_lock(args.gpu_lock):
        for label, paired in (("baseline", False), ("candidate-paired", True)):
            extra = dict(common_extra)
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
            unset = tuple(
                name for name in BASELINE_UNSET_ENVIRONMENT if name not in environment
            )
            arm_root = root / label
            result = CommandRunner().run(
                argv,
                cwd=binary.parent,
                env=environment,
                unset_env=unset,
                timeout_seconds=300,
                stdout_path=arm_root / "stdout.txt",
                stderr_path=arm_root / "stderr.log",
            )
            arm_results[label] = {
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "request_sha256": result.request_sha256,
                "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                "generation_sha256": hashlib.sha256(
                    normalize_llama_cli_generation(result.stdout).encode()
                ).hexdigest(),
                "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
                "paired_activation_verified": (
                    "Q4_RDNA: paired 36 gate/up tensors" in result.stderr
                    if paired
                    else "Q4_RDNA: paired" not in result.stderr
                ),
            }
    passed = (
        all(item["exit_code"] == 0 for item in arm_results.values())
        and all(item["paired_activation_verified"] for item in arm_results.values())
        and arm_results["baseline"]["generation_sha256"]
        == arm_results["candidate-paired"]["generation_sha256"]
    )
    summary = {
        "schema": "gpuopt.q4rdna-gate-up-pair-correctness.v1",
        "outcome": "PASS" if passed else "FAIL",
        "coordinates": {
            "binary_sha256": sha256_file(binary),
            "model_sha256": sha256_file(model),
            "sidecar_sha256": sha256_file(sidecar),
            "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
            "seed": 20260822,
            "temperature": 0,
            "tokens": args.tokens,
        },
        "arms": arm_results,
    }
    _write_atomic(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
