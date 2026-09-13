#!/usr/bin/env python3
"""Prepare Q5/Q6/Q8 reference arms directly from one frozen BF16 GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from tools.prepare_mixed_bit_models import (
    PreparationError,
    inspect_gguf,
    require_regular_file,
    sha256_file,
)

SCHEMA = "gpuopt.precision-sweep-preparation.v1"
SUPPORTED = {
    "Q5_K_M": {"file_type": 17, "primary_type": "Q5_K"},
    "Q6_K": {"file_type": 18, "primary_type": "Q6_K"},
    "Q8_0": {"file_type": 7, "primary_type": "Q8_0"},
}
HIGH_PRECISION_TYPES = {"F32", "F16", "BF16"}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_record(path: Path, *, label: str) -> dict[str, Any]:
    path = require_regular_file(path, label=label)
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    coordinates = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    current = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if current != coordinates:
        raise PreparationError(f"{label} changed while hashing")
    return {"path": str(path), "size": after.st_size, "sha256": digest}


def run(argv: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            shell=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise PreparationError(f"command timed out: {argv[0]}") from error
    if result.returncode:
        raise PreparationError(
            f"command failed ({result.returncode}): {argv!r}\n"
            + result.stderr.strip()[-4000:]
        )
    return {
        "argv": argv,
        "cwd": str(cwd),
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
    }


def validate_source(path: Path) -> dict[str, Any]:
    inspection = inspect_gguf(path)
    if inspection["file_type"] != 32 or inspection["tensor_type_histogram"].get(
        "BF16", 0
    ) == 0:
        raise PreparationError("precision sweep requires a MOSTLY_BF16 source")
    quantized = sorted(
        set(inspection["tensor_type_histogram"]) - HIGH_PRECISION_TYPES
    )
    if quantized:
        raise PreparationError(
            "precision sweep refuses requantization from tensor types: "
            + ", ".join(quantized)
        )
    return inspection


def validate_candidate(
    path: Path, *, quantization: str, source: dict[str, Any]
) -> dict[str, Any]:
    inspection = inspect_gguf(path)
    expected = SUPPORTED[quantization]
    if inspection["file_type"] != expected["file_type"]:
        raise PreparationError(f"{quantization} output file type is incorrect")
    if inspection["tensor_type_histogram"].get(expected["primary_type"], 0) == 0:
        raise PreparationError(f"{quantization} output has no primary tensors")
    if inspection["tensor_count"] != source["tensor_count"]:
        raise PreparationError(f"{quantization} tensor count differs from source")
    if inspection["architecture"] != source["architecture"]:
        raise PreparationError(f"{quantization} architecture differs from source")
    return inspection


def prepare_precision_sweep(
    *,
    bf16_model: Path,
    calibration: Path,
    llama_bin_dir: Path,
    output_dir: Path,
    quantizations: tuple[str, ...] = ("Q8_0", "Q6_K", "Q5_K_M"),
    ngl: int = 999,
    threads: int = 12,
    context_size: int = 512,
    chunks: int = 16,
    timeout_seconds: int = 14400,
    dry_run: bool = False,
    resume_partials: bool = False,
) -> dict[str, Any]:
    if not quantizations or len(quantizations) != len(set(quantizations)):
        raise PreparationError("quantizations must be a non-empty unique sequence")
    unsupported = sorted(set(quantizations) - set(SUPPORTED))
    if unsupported:
        raise PreparationError("unsupported quantizations: " + ", ".join(unsupported))
    if "Q8_0" not in quantizations:
        raise PreparationError("Q8_0 is required as the imatrix calibration model")
    if ngl < 0 or min(threads, context_size, chunks, timeout_seconds) <= 0:
        raise PreparationError("invalid execution coordinates")

    source = require_regular_file(bf16_model, label="BF16 source")
    calibration = require_regular_file(calibration, label="calibration corpus")
    source_inspection = validate_source(source)
    source_record = {**stable_record(source, label="BF16 source"), "gguf": source_inspection}
    calibration_record = stable_record(calibration, label="calibration corpus")
    bin_dir = llama_bin_dir.resolve()
    quantizer = require_regular_file(
        bin_dir / "llama-quantize", label="llama-quantize", executable=True
    )
    imatrix_binary = require_regular_file(
        bin_dir / "llama-imatrix", label="llama-imatrix", executable=True
    )
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "preparation.json"
    outputs = {value: output_dir / f"{value}.gguf" for value in quantizations}
    imatrix = output_dir / "imatrix.gguf"
    all_outputs = [*outputs.values(), imatrix, manifest_path]
    if len(all_outputs) != len(set(all_outputs)):
        raise PreparationError("output paths must be distinct")
    if source in all_outputs or calibration in all_outputs:
        raise PreparationError("outputs cannot overwrite inputs")

    toolchain = {
        "llama_quantize": stable_record(quantizer, label="llama-quantize"),
        "llama_imatrix": stable_record(imatrix_binary, label="llama-imatrix"),
    }
    partials = {
        name: path.with_name(path.name + ".partial") for name, path in outputs.items()
    }
    imatrix_partial = imatrix.with_name(imatrix.name + ".partial")
    commands: dict[str, dict[str, Any]] = {}
    q8_partial = partials["Q8_0"]
    commands["Q8_0"] = {
        "argv": [
            str(quantizer),
            str(source),
            str(q8_partial),
            "Q8_0",
            str(threads),
        ],
        "cwd": str(bin_dir),
    }
    commands["imatrix"] = {
        "argv": [
            str(imatrix_binary),
            "-m",
            str(q8_partial),
            "-f",
            str(calibration),
            "-o",
            str(imatrix_partial),
            "--output-format",
            "gguf",
            "--no-ppl",
            "-ngl",
            str(ngl),
            "-t",
            str(threads),
            "-c",
            str(context_size),
            "--chunks",
            str(chunks),
        ],
        "cwd": str(bin_dir),
    }
    for quantization in quantizations:
        if quantization == "Q8_0":
            continue
        commands[quantization] = {
            "argv": [
                str(quantizer),
                "--imatrix",
                str(imatrix_partial),
                str(source),
                str(partials[quantization]),
                quantization,
                str(threads),
            ],
            "cwd": str(bin_dir),
        }

    plan = {
        "schema": SCHEMA,
        "status": "planned",
        "source": source_record,
        "calibration": calibration_record,
        "toolchain": toolchain,
        "quantizations": list(quantizations),
        "coordinates": {
            "ngl": ngl,
            "threads": threads,
            "context_size": context_size,
            "chunks": chunks,
        },
        "commands": commands,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "imatrix": str(imatrix),
        "manifest": str(manifest_path),
    }
    plan["plan_sha256"] = canonical_sha256(plan)

    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("plan_sha256") != plan["plan_sha256"]:
            raise PreparationError("existing manifest does not match requested sweep")
        for quantization, path in outputs.items():
            claimed = existing["artifacts"][quantization]
            actual = stable_record(path, label=quantization)
            if (actual["sha256"], actual["size"]) != (
                claimed["sha256"],
                claimed["size"],
            ):
                raise PreparationError(f"existing {quantization} artifact drifted")
            validate_candidate(path, quantization=quantization, source=source_inspection)
        return {**existing, "action": "reuse"}
    existing_outputs = [
        path for path in all_outputs if path.exists() or path.is_symlink()
    ]
    if existing_outputs and not resume_partials:
        raise PreparationError("partial preparation exists without a matching manifest")
    if dry_run:
        return {**plan, "action": "dry_run"}

    if resume_partials:
        output_dir.mkdir(parents=True, exist_ok=True)
        unexpected = [
            path
            for path in output_dir.iterdir()
            if path not in {*partials.values(), imatrix_partial}
        ]
        if unexpected:
            raise PreparationError(
                "resume directory contains unexpected artifacts: "
                + ", ".join(sorted(path.name for path in unexpected))
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    execution: dict[str, dict[str, Any]] = {}
    if q8_partial.is_file():
        validate_candidate(q8_partial, quantization="Q8_0", source=source_inspection)
        execution["Q8_0"] = {"execution": "reused_valid_partial"}
    else:
        execution["Q8_0"] = run(
            commands["Q8_0"]["argv"], cwd=bin_dir, timeout_seconds=timeout_seconds
        )
        validate_candidate(q8_partial, quantization="Q8_0", source=source_inspection)
    if imatrix_partial.is_file():
        imatrix_inspection = inspect_gguf(imatrix_partial)
        execution["imatrix"] = {"execution": "reused_valid_partial"}
    else:
        execution["imatrix"] = run(
            commands["imatrix"]["argv"], cwd=bin_dir, timeout_seconds=timeout_seconds
        )
        imatrix_inspection = inspect_gguf(imatrix_partial)
    if imatrix_inspection["general_type"] != "imatrix":
        raise PreparationError("llama-imatrix did not produce an imatrix GGUF")
    for quantization in quantizations:
        if quantization == "Q8_0":
            continue
        if partials[quantization].is_file():
            execution[quantization] = {"execution": "reused_valid_partial"}
        else:
            execution[quantization] = run(
                commands[quantization]["argv"],
                cwd=bin_dir,
                timeout_seconds=timeout_seconds,
            )
        validate_candidate(
            partials[quantization],
            quantization=quantization,
            source=source_inspection,
        )

    artifacts = {
        quantization: {
            **stable_record(partials[quantization], label=quantization),
            "path": str(outputs[quantization]),
            "gguf": inspect_gguf(partials[quantization]),
        }
        for quantization in quantizations
    }
    imatrix_record = {
        **stable_record(imatrix_partial, label="imatrix"),
        "path": str(imatrix),
        "gguf": imatrix_inspection,
    }
    manifest = {
        **plan,
        "status": "complete",
        "commands": {
            name: {**command, **execution[name]} for name, command in commands.items()
        },
        "artifacts": artifacts,
        "imatrix_artifact": imatrix_record,
        "provenance": {
            "direct_from_bf16": True,
            "q8_used_only_for_imatrix_collection": True,
            "requantization": False,
        },
    }
    manifest_partial = manifest_path.with_name(manifest_path.name + ".partial")
    manifest_partial.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for quantization, final_path in outputs.items():
        os.replace(partials[quantization], final_path)
    os.replace(imatrix_partial, imatrix)
    os.replace(manifest_partial, manifest_path)
    return {**manifest, "action": "create"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--llama-bin-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quantization",
        action="append",
        dest="quantizations",
        choices=sorted(SUPPORTED),
    )
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-partials",
        action="store_true",
        help="resume only hash/format-valid partial artifacts after an interrupted run",
    )
    args = parser.parse_args()
    try:
        result = prepare_precision_sweep(
            bf16_model=args.bf16_model,
            calibration=args.calibration,
            llama_bin_dir=args.llama_bin_dir,
            output_dir=args.output_dir,
            quantizations=tuple(args.quantizations or ("Q8_0", "Q6_K", "Q5_K_M")),
            ngl=args.ngl,
            threads=args.threads,
            context_size=args.context_size,
            chunks=args.chunks,
            timeout_seconds=args.timeout_seconds,
            dry_run=args.dry_run,
            resume_partials=args.resume_partials,
        )
    except PreparationError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
