#!/usr/bin/env python3
"""Run the fixed Qwen3-8B KV-cache benchmark matrix with durable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from amd_inference_opt.command import CommandResult, CommandRunner
from amd_inference_opt.kv_cache import (
    KVCacheArm,
    KVCacheBenchmarkProtocol,
    KVCacheError,
    KVCacheType,
    evaluate_kv_cache_gate,
    parse_kv_cache_benchmark_json,
)
from amd_inference_opt.resource_lock import exclusive_gpu_lock


def _json_bytes(document: object) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(_json_bytes(document))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_document(result: CommandResult) -> dict[str, object]:
    document = result.to_dict()
    document["stdout_sha256"] = hashlib.sha256(result.stdout.encode()).hexdigest()
    document["stderr_sha256"] = hashlib.sha256(result.stderr.encode()).hexdigest()
    return document


def _decode_rows(result: CommandResult, arm: KVCacheArm, repetitions: int) -> list[dict[str, Any]]:
    if not result.succeeded:
        raise RuntimeError(
            f"{arm.cache_type.value} benchmark failed: exit={result.exit_code} "
            f"timeout={result.timed_out} spawn={result.spawn_error}"
        )
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{arm.cache_type.value} emitted invalid JSON: {error}") from error
    if not isinstance(rows, list) or len(rows) != 3:
        raise RuntimeError(f"{arm.cache_type.value} must emit exactly three depth rows")
    expected_depths = {4096, 16384, 28672}
    if {row.get("n_depth") for row in rows if isinstance(row, dict)} != expected_depths:
        raise RuntimeError(f"{arm.cache_type.value} depth coordinates are incomplete")
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("llama-bench row is not an object")
        if row.get("type_k") != arm.type_k.value or row.get("type_v") != arm.type_v.value:
            raise RuntimeError(f"{arm.cache_type.value} cache coordinate drifted")
        if row.get("flash_attn") != 1:
            raise RuntimeError(f"{arm.cache_type.value} did not use flash attention")
        samples = row.get("samples_ts")
        if not isinstance(samples, list) or len(samples) != repetitions:
            raise RuntimeError(
                f"{arm.cache_type.value} row does not contain {repetitions} samples"
            )
    return rows


def _drop_same_coordinate_warmup(
    rows: list[dict[str, Any]],
    *,
    scored_repetitions: int,
    warmup_samples: int = 5,
) -> list[dict[str, Any]]:
    """Exclude the repeatable in-process ramp observed after each model load."""

    normalized: list[dict[str, Any]] = []
    expected = warmup_samples + scored_repetitions
    for row in rows:
        samples = row.get("samples_ts")
        if not isinstance(samples, list) or len(samples) != expected:
            raise RuntimeError(
                f"KV row must contain {expected} source samples before warmup removal"
            )
        scored = [float(value) for value in samples[warmup_samples:]]
        updated = dict(row)
        updated["gpuopt_source_samples_ts"] = list(samples)
        updated["gpuopt_warmup_samples_dropped"] = warmup_samples
        updated["samples_ts"] = scored
        updated["avg_ts"] = statistics.fmean(scored)
        updated["stddev_ts"] = statistics.stdev(scored)
        raw_ns = row.get("samples_ns")
        if isinstance(raw_ns, list) and len(raw_ns) == expected:
            updated["gpuopt_source_samples_ns"] = list(raw_ns)
            updated["samples_ns"] = list(raw_ns[warmup_samples:])
            updated["avg_ns"] = statistics.fmean(updated["samples_ns"])
            updated["stddev_ns"] = statistics.stdev(updated["samples_ns"])
        normalized.append(updated)
    return normalized


def _arm_protocol(
    *,
    binary: Path,
    model: Path,
    arm: KVCacheArm,
    repetitions: int,
    environment: dict[str, str],
    cwd: Path,
    timeout_seconds: float,
    model_quantization: str,
) -> KVCacheBenchmarkProtocol:
    return KVCacheBenchmarkProtocol(
        llama_bench_path=str(binary),
        model_path=str(model),
        arm=arm,
        repetitions=repetitions,
        environment=environment,
        cwd=str(cwd),
        timeout_seconds=timeout_seconds,
        model_quantization=model_quantization,
    )


def _run_arm(
    *,
    binary: Path,
    model: Path,
    arm: KVCacheArm,
    output_root: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    model_quantization: str,
) -> dict[str, object]:
    arm_root = output_root / arm.cache_type.value
    terminal = arm_root / "result.json"
    if terminal.is_file():
        existing = json.loads(terminal.read_text(encoding="utf-8"))
        if (
            existing.get("binary_sha256") == _sha256(binary)
            and existing.get("model_sha256") == _sha256(model)
            and existing.get("model_quantization") == model_quantization
        ):
            return existing
        raise RuntimeError(f"existing {terminal} belongs to different inputs")

    runner = CommandRunner()

    def run_attempt(repetitions: int, attempt: int):
        suffix = "" if attempt == 1 else f"-attempt{attempt}"
        protocol = _arm_protocol(
            binary=binary,
            model=model,
            arm=arm,
            repetitions=repetitions,
            environment=environment,
            cwd=binary.parent,
            timeout_seconds=timeout_seconds,
            model_quantization=model_quantization,
        )
        precondition_argv = list(protocol.argv)
        precondition_argv[precondition_argv.index("-r") + 1] = "1"
        precondition_result = runner.run(
            precondition_argv,
            cwd=binary.parent,
            env=environment,
            timeout_seconds=timeout_seconds,
            stdout_path=arm_root / f"precondition{suffix}.stdout.json",
            stderr_path=arm_root / f"precondition{suffix}.stderr.log",
        )
        _write_atomic(
            arm_root / f"precondition{suffix}.command.json",
            _command_document(precondition_result),
        )
        _decode_rows(precondition_result, arm, 1)
        scored_argv = list(protocol.argv)
        scored_argv[scored_argv.index("-r") + 1] = str(repetitions + 5)
        scored_result = runner.run(
            scored_argv,
            cwd=binary.parent,
            env=environment,
            timeout_seconds=timeout_seconds,
            stdout_path=arm_root / f"benchmark{suffix}.stdout.json",
            stderr_path=arm_root / f"benchmark{suffix}.stderr.log",
        )
        _write_atomic(
            arm_root / f"benchmark{suffix}.command.json",
            _command_document(scored_result),
        )
        source_rows = _decode_rows(scored_result, arm, repetitions + 5)
        rows = _drop_same_coordinate_warmup(
            source_rows,
            scored_repetitions=repetitions,
        )
        parsed = parse_kv_cache_benchmark_json(rows, protocol=protocol)
        return protocol, precondition_result, scored_result, rows, source_rows, parsed

    existing_stdout = arm_root / "benchmark.stdout.json"
    if existing_stdout.is_file():
        protocol = _arm_protocol(
            binary=binary,
            model=model,
            arm=arm,
            repetitions=5,
            environment=environment,
            cwd=binary.parent,
            timeout_seconds=timeout_seconds,
            model_quantization=model_quantization,
        )
        existing_source_rows = json.loads(existing_stdout.read_text(encoding="utf-8"))
        try:
            existing_rows = _drop_same_coordinate_warmup(
                existing_source_rows,
                scored_repetitions=5,
            )
            parse_kv_cache_benchmark_json(existing_rows, protocol=protocol)
        except KVCacheError as error:
            if "CV" not in str(error):
                raise
            gate_path = arm_root / "attempt1-gate.json"
            if not gate_path.exists():
                _write_atomic(
                    gate_path,
                    {
                        "status": "INCONCLUSIVE",
                        "reason": str(error),
                        "next_attempt_repetitions": 12,
                    },
                )
            retry_attempt = 2
            while (arm_root / f"precondition-attempt{retry_attempt}.command.json").exists():
                retry_attempt += 1
            (
                protocol,
                precondition_result,
                scored_result,
                rows,
                source_rows,
                parsed,
            ) = run_attempt(12, retry_attempt)
        else:
            raise RuntimeError(
                f"{existing_stdout} is valid but terminal result.json was not finalized"
            )
    else:
        try:
            (
                protocol,
                precondition_result,
                scored_result,
                rows,
                source_rows,
                parsed,
            ) = run_attempt(5, 1)
        except KVCacheError as error:
            if "CV" not in str(error):
                raise
            _write_atomic(
                arm_root / "attempt1-gate.json",
                {
                    "status": "INCONCLUSIVE",
                    "reason": str(error),
                    "next_attempt_repetitions": 12,
                },
            )
            (
                protocol,
                precondition_result,
                scored_result,
                rows,
                source_rows,
                parsed,
            ) = run_attempt(12, 2)
    document: dict[str, object] = {
        "schema": "gpuopt.kv-cache-arm.v1",
        "arm": arm.cache_type.value,
        "binary": str(binary),
        "binary_sha256": _sha256(binary),
        "model": str(model),
        "model_sha256": _sha256(model),
        "model_quantization": model_quantization,
        "protocol": protocol.to_dict(),
        "same_coordinate_warmup_samples_dropped": 5,
        "records": [asdict(record) for record in parsed.records],
        "raw_rows": rows,
        "source_raw_rows": source_rows,
        "max_cv_percent": parsed.max_cv_percent,
        "precondition_request_sha256": precondition_result.request_sha256,
        "benchmark_request_sha256": scored_result.request_sha256,
        "precondition_duration_seconds": precondition_result.duration_seconds,
        "benchmark_duration_seconds": scored_result.duration_seconds,
    }
    _write_atomic(terminal, document)
    return document


def _parsed(document: dict[str, object]):
    protocol = document.get("protocol")
    repetitions = 5
    if isinstance(protocol, dict):
        repetitions = int(protocol.get("repetitions", 5))
    return parse_kv_cache_benchmark_json(
        document["raw_rows"],
        arm=KVCacheArm(KVCacheType(str(document["arm"]))),
        repetitions=repetitions,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-quantization", default="Q6_K")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--gpu-lock", type=Path)
    args = parser.parse_args()

    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    output_root = args.output_root.resolve()
    environment = {
        "HIP_VISIBLE_DEVICES": "0",
        "HSA_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": f"{binary.parent}:/opt/rocm/core-7.14/lib",
    }
    lock_path = args.gpu_lock or output_root.parent / "gpu-0.lock"
    with exclusive_gpu_lock(lock_path):
        documents = {
            cache_type.value: _run_arm(
                binary=binary,
                model=model,
                arm=KVCacheArm(cache_type),
                output_root=output_root,
                environment=environment,
                timeout_seconds=args.timeout_seconds,
                model_quantization=args.model_quantization,
            )
            for cache_type in KVCacheType
        }
    baseline = _parsed(documents[KVCacheType.F16.value])
    gates = {
        cache_type.value: asdict(
            evaluate_kv_cache_gate(baseline, _parsed(documents[cache_type.value]))
        )
        for cache_type in (KVCacheType.Q8_0, KVCacheType.Q4_0)
    }
    summary = {
        "schema": "gpuopt.kv-cache-campaign.v1",
        "binary_sha256": _sha256(binary),
        "model_sha256": _sha256(model),
        "arms": documents,
        "gates": gates,
    }
    _write_atomic(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
