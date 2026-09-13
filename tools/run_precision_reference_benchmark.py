#!/usr/bin/env python3
"""Benchmark stock llama.cpp precision arms with one immutable protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from amd_inference_opt.command import CommandResult, CommandRunner
from amd_inference_opt.llama_cpp import LlamaBenchRecord, parse_llama_bench_json
from amd_inference_opt.resource_lock import exclusive_gpu_lock


class PrecisionBenchmarkError(RuntimeError):
    """The benchmark did not prove a comparable full-GPU run."""


LOCAL_RUNTIME_LIBRARY_PREFIXES = ("libggml", "libllama")
REQUIRED_RUNTIME_LIBRARY_PREFIXES = ("libggml-hip", "libllama.so")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_runtime_libraries(output: str, *, expected_dir: Path) -> dict[str, object]:
    libraries: dict[str, dict[str, object]] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if " => " not in line:
            continue
        name, remainder = line.split(" => ", 1)
        if not name.startswith(LOCAL_RUNTIME_LIBRARY_PREFIXES):
            continue
        rendered_path = remainder.split(" (", 1)[0].strip()
        if rendered_path == "not found":
            raise PrecisionBenchmarkError(f"runtime library is not found: {name}")
        path = Path(rendered_path).resolve()
        if path.parent != expected_dir:
            raise PrecisionBenchmarkError(
                f"runtime library pollution: {name} resolved to {path}, "
                f"expected {expected_dir}"
            )
        if not path.is_file():
            raise PrecisionBenchmarkError(f"runtime library is not a file: {path}")
        libraries[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    for required in REQUIRED_RUNTIME_LIBRARY_PREFIXES:
        if not any(name.startswith(required) for name in libraries):
            raise PrecisionBenchmarkError(
                f"runtime library audit is missing {required}"
            )
    canonical = json.dumps(libraries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "status": "verified",
        "libraries": libraries,
        "closure_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _audit_runtime_libraries(binary: Path) -> dict[str, object]:
    result = subprocess.run(
        ["ldd", str(binary)],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        shell=False,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise PrecisionBenchmarkError(
            f"ldd failed for {binary}: {result.stderr.strip()[-1000:]}"
        )
    return _parse_runtime_libraries(result.stdout, expected_dir=binary.parent.resolve())


def _write_atomic(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    with temporary.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def _request(
    binary: Path,
    model: Path,
    *,
    prompt_tokens: str,
    generation_tokens: str,
    repetitions: int,
) -> list[str]:
    return [
        str(binary),
        "-m",
        str(model),
        "-p",
        prompt_tokens,
        "-n",
        generation_tokens,
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


def _command_document(result: CommandResult) -> dict[str, object]:
    return {
        **result.to_dict(),
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
    }


def _run(
    runner: CommandRunner,
    *,
    argv: list[str],
    binary: Path,
    environment: dict[str, str],
    root: Path,
    name: str,
    timeout_seconds: float,
) -> tuple[CommandResult, tuple[LlamaBenchRecord, ...], tuple[dict[str, Any], ...]]:
    result = runner.run(
        argv,
        cwd=binary.parent,
        env=environment,
        timeout_seconds=timeout_seconds,
        stdout_path=root / f"{name}.stdout.json",
        stderr_path=root / f"{name}.stderr.log",
    )
    _write_atomic(root / f"{name}.command.json", _command_document(result))
    if not result.succeeded:
        raise PrecisionBenchmarkError(
            f"{name} failed: exit={result.exit_code} timeout={result.timed_out}"
        )
    parsed = parse_llama_bench_json(result.stdout)
    return result, parsed.records, parsed.raw


def _validate_offload(row: dict[str, Any], expected_type: str) -> None:
    checks = {
        "model_type": expected_type in str(row.get("model_type", "")),
        "rocm_backend": "ROCm" in str(row.get("backends", "")),
        "device": row.get("devices") == "ROCm0",
        "main_gpu": row.get("main_gpu") == 0,
        "gpu_layers": row.get("n_gpu_layers") == 999,
        "kv_offload": row.get("no_kv_offload") is False,
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise PrecisionBenchmarkError(f"full-GPU precondition failed: {failed}")


def _drop_warmup_samples(
    record: LlamaBenchRecord, *, warmup_samples: int
) -> LlamaBenchRecord:
    samples = record.samples_tokens_per_second[warmup_samples:]
    if not samples:
        raise PrecisionBenchmarkError("warmup consumed every benchmark sample")
    mean = statistics.fmean(samples)
    stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    raw = dict(record.raw)
    raw["gpuopt_warmup_samples_dropped"] = warmup_samples
    raw["gpuopt_scored_samples_ts"] = list(samples)
    return LlamaBenchRecord(
        phase=record.phase,
        prompt_tokens=record.prompt_tokens,
        generation_tokens=record.generation_tokens,
        mean_tokens_per_second=mean,
        stddev_tokens_per_second=stddev,
        samples_tokens_per_second=samples,
        coefficient_of_variation=stddev / mean,
        raw=raw,
    )


def _arm(
    *,
    name: str,
    expected_type: str,
    model: Path,
    binary: Path,
    output_root: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    root = output_root / name
    terminal = root / "result.json"
    binary_sha = _sha256(binary)
    model_sha = _sha256(model)
    runtime_libraries = _audit_runtime_libraries(binary)
    if terminal.is_file():
        existing = json.loads(terminal.read_text(encoding="utf-8"))
        if (
            existing.get("binary_sha256") == binary_sha
            and existing.get("model_sha256") == model_sha
            and (
                existing.get("runtime_libraries") is None
                or existing.get("runtime_libraries") == runtime_libraries
            )
        ):
            return existing
        raise PrecisionBenchmarkError(f"existing result belongs to other inputs: {terminal}")

    environment = {
        "HIP_VISIBLE_DEVICES": "0",
        "HSA_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": f"{binary.parent}:/opt/rocm/core-7.14/lib",
    }
    runner = CommandRunner()
    smoke, smoke_records, smoke_raw = _run(
        runner,
        argv=_request(
            binary, model, prompt_tokens="0", generation_tokens="1", repetitions=1
        ),
        binary=binary,
        environment=environment,
        root=root,
        name="smoke",
        timeout_seconds=timeout_seconds,
    )
    if len(smoke_records) != 1:
        raise PrecisionBenchmarkError("smoke must emit one row")
    _validate_offload(smoke_raw[0], expected_type)

    attempts: list[dict[str, object]] = []
    final_records: dict[str, LlamaBenchRecord] = {}
    warmup_samples = 5
    for attempt, repetitions in ((1, 5), (2, 12)):
        prefix = f"attempt-{attempt}"
        pp, pp_records, _ = _run(
            runner,
            argv=_request(
                binary,
                model,
                prompt_tokens="512",
                generation_tokens="0",
                repetitions=repetitions + warmup_samples,
            ),
            binary=binary,
            environment=environment,
            root=root,
            name=f"{prefix}-pp512",
            timeout_seconds=timeout_seconds,
        )
        tg, tg_records, _ = _run(
            runner,
            argv=_request(
                binary,
                model,
                prompt_tokens="0",
                generation_tokens="128,512",
                repetitions=repetitions + warmup_samples,
            ),
            binary=binary,
            environment=environment,
            root=root,
            name=f"{prefix}-decode",
            timeout_seconds=timeout_seconds,
        )
        by_id = {
            record.test_id: _drop_warmup_samples(
                record, warmup_samples=warmup_samples
            )
            for record in (*pp_records, *tg_records)
        }
        if set(by_id) != {"pp512", "tg128", "tg512"}:
            raise PrecisionBenchmarkError(f"unexpected benchmark coordinates: {sorted(by_id)}")
        if any(record.sample_count != repetitions for record in by_id.values()):
            raise PrecisionBenchmarkError("benchmark sample count drifted")
        max_cv = max(record.coefficient_of_variation for record in by_id.values())
        attempts.append(
            {
                "attempt": attempt,
                "repetitions": repetitions,
                "warmup_samples_dropped": warmup_samples,
                "pp512_request_sha256": pp.request_sha256,
                "decode_request_sha256": tg.request_sha256,
                "max_cv_percent": max_cv * 100,
                "stable": max_cv <= 0.02,
            }
        )
        final_records = by_id
        if max_cv <= 0.02:
            break

    stable = max(
        record.coefficient_of_variation for record in final_records.values()
    ) <= 0.02
    runtime_libraries_after = _audit_runtime_libraries(binary)
    if runtime_libraries_after != runtime_libraries:
        raise PrecisionBenchmarkError(
            "runtime dynamic-library closure changed during the benchmark arm"
        )
    document: dict[str, object] = {
        "schema": "gpuopt.precision-reference-benchmark.v1",
        "arm": name,
        "expected_model_type": expected_type,
        "binary": str(binary),
        "binary_sha256": binary_sha,
        "runtime_libraries": runtime_libraries,
        "model": str(model),
        "model_sha256": model_sha,
        "model_size_bytes": model.stat().st_size,
        "protocol": {
            "prompt_tokens": [512],
            "generation_tokens": [128, 512],
            "batch_size": 2048,
            "ubatch_size": 512,
            "threads": 12,
            "gpu_layers": 999,
            "device": "ROCm0",
            "built_in_warmup": True,
            "additional_same_coordinate_warmup_samples": warmup_samples,
            "initial_repetitions": 5,
            "retry_repetitions": 12,
            "maximum_cv_percent": 2.0,
        },
        "smoke_request_sha256": smoke.request_sha256,
        "attempts": attempts,
        "stable": stable,
        "records": {key: value.to_dict() for key, value in sorted(final_records.items())},
    }
    _write_atomic(terminal, document)
    return document


def _parse_arm(value: str) -> tuple[str, str, Path]:
    fields = value.split("=", 2)
    if len(fields) != 3 or not all(fields):
        raise argparse.ArgumentTypeError("arm must be NAME=EXPECTED_MODEL_TYPE=MODEL.gguf")
    return fields[0], fields[1], Path(fields[2])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--arm", action="append", required=True, type=_parse_arm)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    output_root = args.output_root.resolve()
    if len({name for name, _, _ in args.arm}) != len(args.arm):
        parser.error("arm names must be unique")
    lock_path = args.gpu_lock or output_root.parent / "gpu-0.lock"
    with exclusive_gpu_lock(lock_path):
        arms = {
            name: _arm(
                name=name,
                expected_type=expected,
                model=model.resolve(strict=True),
                binary=binary,
                output_root=output_root,
                timeout_seconds=args.timeout_seconds,
            )
            for name, expected, model in args.arm
        }
    summary = {
        "schema": "gpuopt.precision-reference-benchmark-summary.v1",
        "arms": arms,
    }
    _write_atomic(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
