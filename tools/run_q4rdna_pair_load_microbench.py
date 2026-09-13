#!/usr/bin/env python3
"""Compile, run, and preserve the Q4_RDNA gate/up paired-load microbenchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from amd_inference_opt.command import CommandResult, CommandRunner
from amd_inference_opt.llama_cpp import sha256_file
from amd_inference_opt.q4rdna_tuning import screen_gate_up_pair_load
from amd_inference_opt.resource_lock import exclusive_gpu_lock


class PairLoadRunError(RuntimeError):
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
        "--source",
        type=Path,
        default=Path("benchmarks/q4rdna_gate_up_pair_load.hip"),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--hipcc", type=Path, default=Path("/usr/bin/hipcc"))
    parser.add_argument("--gfx-target", default="gfx1201")
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    hipcc = args.hipcc.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    binary = output_root / "q4rdna_gate_up_pair_load"
    source_sha256 = sha256_file(source)
    compile_argv = [
        str(hipcc),
        "-O3",
        f"--offload-arch={args.gfx_target}",
        str(source),
        "-o",
        str(binary),
    ]
    runner = CommandRunner()
    build = runner.run(
        compile_argv,
        cwd=source.parent,
        timeout_seconds=args.timeout_seconds,
        stdout_path=output_root / "build.stdout.log",
        stderr_path=output_root / "build.stderr.log",
    )
    _write_atomic(output_root / "build-command.json", _command_document(build))
    if not build.succeeded or not binary.is_file():
        raise PairLoadRunError(
            f"microbenchmark build failed: exit={build.exit_code} timeout={build.timed_out}"
        )
    binary_sha256 = sha256_file(binary)
    environment = {
        "HIP_VISIBLE_DEVICES": "0",
        "HSA_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/opt/rocm/core-7.14/lib",
    }
    lock_path = args.gpu_lock or output_root.parent / "gpu-0.lock"
    with exclusive_gpu_lock(lock_path):
        execution = runner.run(
            [str(binary)],
            cwd=output_root,
            env=environment,
            timeout_seconds=args.timeout_seconds,
            stdout_path=output_root / "benchmark.stdout.json",
            stderr_path=output_root / "benchmark.stderr.log",
        )
    _write_atomic(output_root / "run-command.json", _command_document(execution))
    if not execution.succeeded:
        raise PairLoadRunError(
            f"microbenchmark failed: exit={execution.exit_code} timeout={execution.timed_out}"
        )
    try:
        result = json.loads(execution.stdout)
    except json.JSONDecodeError as error:
        raise PairLoadRunError("microbenchmark did not return JSON") from error
    if result.get("schema") != "gpuopt.q4rdna-gate-up-pair-load.v1":
        raise PairLoadRunError("microbenchmark returned an unknown schema")
    if result.get("gfx") != args.gfx_target:
        raise PairLoadRunError("microbenchmark ran on an unexpected GPU target")
    screen = screen_gate_up_pair_load(result)
    summary = {
        "schema": "gpuopt.q4rdna-gate-up-pair-load-experiment.v1",
        "hypothesis": (
            "Pairing each gate/up quant byte into one aligned uint16 load reduces VMEM "
            "instructions without changing bytes, arithmetic, split width, or output."
        ),
        "single_variable": "gate/up quant byte packing and load width",
        "source": str(source),
        "source_sha256": source_sha256,
        "binary": str(binary),
        "binary_sha256": binary_sha256,
        "compile_request_sha256": build.request_sha256,
        "run_request_sha256": execution.request_sha256,
        "result": result,
        "screen": screen.to_dict(),
        "required_next_validation": (
            [
                "inspect gfx1201 ISA for one uint16 global load",
                "integrate paired gate/up packing without changing other tensors",
                "profile the real fused 12288x4096 kernel",
                "run tg128/tg512 and deterministic correctness",
            ]
            if screen.outcome == "PROMOTE"
            else []
        ),
    }
    _write_atomic(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
