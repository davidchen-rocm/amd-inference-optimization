#!/usr/bin/env python3
"""Compile and screen the exact-shape Q4-up/Q2-gate microbenchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from amd_inference_opt.command import CommandResult, CommandRunner
from amd_inference_opt.llama_cpp import sha256_file
from amd_inference_opt.resource_lock import exclusive_gpu_lock


class GateQ2RunError(RuntimeError):
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


def _screen(result: dict[str, object]) -> dict[str, object]:
    required = (
        "max_absolute_error",
        "q4_cv_percent",
        "q2_cv_percent",
        "q2_improvement_percent",
        "fused_weight_byte_reduction_percent",
    )
    values: dict[str, float] = {}
    for name in required:
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GateQ2RunError(f"Q2 result is missing numeric {name}")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise GateQ2RunError(f"Q2 result {name} must be finite")
        values[name] = parsed
    if values["max_absolute_error"] != 0:
        outcome = "REJECT"
        reasons = ["Q2 packing or unpacking changed the controlled mathematical result"]
    elif max(values["q4_cv_percent"], values["q2_cv_percent"]) > 1.0:
        outcome = "INCONCLUSIVE"
        reasons = ["Q2 microbenchmark exceeded the 1% CV threshold"]
    elif values["fused_weight_byte_reduction_percent"] <= 0:
        outcome = "REJECT"
        reasons = ["Q2 candidate did not reduce fused gate and up weight bytes"]
    elif values["q2_improvement_percent"] >= 2.0:
        outcome = "PROMOTE"
        reasons = ["Q2 gate passed the exact-shape performance screen"]
    else:
        outcome = "REJECT"
        reasons = ["Q2 unpack cost consumed the expected byte-traffic benefit"]
    return {
        "baseline_arm_id": "q4-up-q4-gate",
        "candidate_arm_id": "q4-up-q2-gate",
        "outcome": outcome,
        "improvement_percent": (
            None if outcome == "INCONCLUSIVE" else values["q2_improvement_percent"]
        ),
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("benchmarks/q4rdna_gate_q2.hip")
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--hipcc", type=Path, default=Path("/usr/bin/hipcc"))
    parser.add_argument("--gfx-target", default="gfx1201")
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    hipcc = args.hipcc.resolve(strict=True)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    binary = root / "q4rdna_gate_q2"
    runner = CommandRunner()
    build = runner.run(
        [
            str(hipcc),
            "-O3",
            f"--offload-arch={args.gfx_target}",
            str(source),
            "-o",
            str(binary),
        ],
        cwd=source.parent,
        timeout_seconds=args.timeout_seconds,
        stdout_path=root / "build.stdout.log",
        stderr_path=root / "build.stderr.log",
    )
    _write_atomic(root / "build-command.json", _command_document(build))
    if not build.succeeded or not binary.is_file():
        raise GateQ2RunError(f"build failed: exit={build.exit_code}")
    environment = {
        "HIP_VISIBLE_DEVICES": "0",
        "HSA_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/opt/rocm/core-7.14/lib",
    }
    lock_path = args.gpu_lock or root.parent / "gpu-0.lock"
    with exclusive_gpu_lock(lock_path):
        execution = runner.run(
            [str(binary)],
            cwd=root,
            env=environment,
            timeout_seconds=args.timeout_seconds,
            stdout_path=root / "benchmark.stdout.json",
            stderr_path=root / "benchmark.stderr.log",
        )
    _write_atomic(root / "run-command.json", _command_document(execution))
    if not execution.succeeded:
        raise GateQ2RunError(
            f"microbenchmark failed: exit={execution.exit_code} timeout={execution.timed_out}"
        )
    try:
        result = json.loads(execution.stdout)
    except json.JSONDecodeError as error:
        raise GateQ2RunError("microbenchmark did not return JSON") from error
    if result.get("schema") != "gpuopt.q4rdna-gate-q2-microbenchmark.v1":
        raise GateQ2RunError("unexpected microbenchmark schema")
    if result.get("gfx") != args.gfx_target:
        raise GateQ2RunError("microbenchmark ran on an unexpected GPU")
    summary = {
        "schema": "gpuopt.q4rdna-gate-q2-screen.v1",
        "hypothesis": (
            "Changing selected FFN gate tensors from Q4_RDNA 4.25 bpw to "
            "Q2_RDNA 2.25 bpw reduces fused weight bytes enough to offset 2-bit unpack cost."
        ),
        "single_hypothesis": "Q4-up plus Q2-gate representation and kernel",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "result": result,
        "screen": _screen(result),
        "stop_condition": (
            "Do not build a model sidecar unless exact-shape improvement is at least 2%."
        ),
    }
    _write_atomic(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
