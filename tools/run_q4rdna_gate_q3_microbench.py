#!/usr/bin/env python3
"""Compile, run, and Gate the exact-shape Q4-up/Q3-gate microbenchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from amd_inference_opt.command import CommandResult, CommandRunner
from amd_inference_opt.llama_cpp import sha256_file
from amd_inference_opt.q4rdna_tuning import screen_gate_q3_microbenchmark
from amd_inference_opt.resource_lock import exclusive_gpu_lock


class GateQ3RunError(RuntimeError):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("benchmarks/q4rdna_gate_q3.hip")
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
    binary = root / "q4rdna_gate_q3"
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
        raise GateQ3RunError(f"build failed: exit={build.exit_code}")
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
        raise GateQ3RunError(
            f"microbenchmark failed: exit={execution.exit_code} timeout={execution.timed_out}"
        )
    try:
        result = json.loads(execution.stdout)
    except json.JSONDecodeError as error:
        raise GateQ3RunError("microbenchmark did not return JSON") from error
    if result.get("schema") != "gpuopt.q4rdna-gate-q3-microbenchmark.v1":
        raise GateQ3RunError("unexpected microbenchmark schema")
    if result.get("gfx") != args.gfx_target:
        raise GateQ3RunError("microbenchmark ran on an unexpected GPU")
    screen = screen_gate_q3_microbenchmark(result)
    summary = {
        "schema": "gpuopt.q4rdna-gate-q3-screen.v1",
        "hypothesis": (
            "Changing only FFN gate from Q4_RDNA 4.25 bpw to Q3_RDNA 3.25 bpw "
            "reduces fused gate/up bytes enough to offset 3-bit unpack cost."
        ),
        "single_hypothesis": "Q4-up plus Q3-gate representation/kernel",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "result": result,
        "screen": screen.to_dict(),
        "stop_condition": (
            "Do not build a model sidecar unless exact-shape improvement is at least 2%."
        ),
    }
    _write_atomic(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
