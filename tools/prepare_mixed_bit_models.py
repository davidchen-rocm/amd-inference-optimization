#!/usr/bin/env python3
"""Prepare imatrix-guided Q5_K_M and Q4_K_M models from one frozen BF16 GGUF."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KIND = "llama_cpp_bf16_imatrix_q5_q4"
BF16_FILE_TYPE = 32
OUTPUT_FILE_TYPES = {"Q5_K_M": 17, "Q4_K_M": 15}
HIGH_PRECISION_TENSOR_TYPES = {"F32", "F16", "BF16"}
GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
    40: "NVFP4",
}


class PreparationError(RuntimeError):
    """The requested preparation is unsafe, incomplete, or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_file(path: Path, *, label: str, executable: bool = False) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise PreparationError(f"{label} must be a regular, non-symlink file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise PreparationError(f"{label} is not executable: {resolved}")
    return resolved


def _file_coordinates(path: Path, *, label: str) -> dict[str, int]:
    path = require_regular_file(path, label=label)
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _stable_file_record(path: Path, *, label: str) -> dict[str, Any]:
    path = require_regular_file(path, label=label)
    before_coordinates = _file_coordinates(path, label=label)
    digest = sha256_file(path)
    after_coordinates = _file_coordinates(path, label=label)
    if before_coordinates != after_coordinates:
        raise PreparationError(f"{label} changed while hashing: {path}")
    return {
        "path": str(path),
        "size": after_coordinates["size"],
        "sha256": digest,
        "stable_coordinates": after_coordinates,
    }


def _read_exact(source: Any, size: int) -> bytes:
    value = source.read(size)
    if len(value) != size:
        raise PreparationError("truncated GGUF header")
    return value


def _read_gguf_string(source: Any) -> str:
    length = struct.unpack("<Q", _read_exact(source, 8))[0]
    if length > 64 * 1024 * 1024:
        raise PreparationError(f"unreasonable GGUF string length: {length}")
    try:
        return _read_exact(source, length).decode("utf-8")
    except UnicodeDecodeError as error:
        raise PreparationError("invalid UTF-8 in GGUF header") from error


def _read_gguf_value(source: Any, value_type: int, *, keep: bool) -> Any:
    scalar_formats = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }
    if value_type in scalar_formats:
        fmt = scalar_formats[value_type]
        raw = _read_exact(source, struct.calcsize(fmt))
        return struct.unpack(fmt, raw)[0] if keep else None
    if value_type == 8:
        value = _read_gguf_string(source)
        return value if keep else None
    if value_type == 9:
        element_type = struct.unpack("<I", _read_exact(source, 4))[0]
        count = struct.unpack("<Q", _read_exact(source, 8))[0]
        if count > 100_000_000:
            raise PreparationError(f"unreasonable GGUF array length: {count}")
        values = [] if keep else None
        for _ in range(count):
            value = _read_gguf_value(source, element_type, keep=keep)
            if keep:
                values.append(value)
        return values
    raise PreparationError(f"unsupported GGUF metadata value type: {value_type}")


def inspect_gguf(path: Path) -> dict[str, Any]:
    """Read model identity and a tensor-type histogram without loading tensor data."""

    path = require_regular_file(path, label="GGUF")
    with path.open("rb") as source:
        if _read_exact(source, 4) != b"GGUF":
            raise PreparationError(f"file does not have GGUF magic: {path}")
        version = struct.unpack("<I", _read_exact(source, 4))[0]
        if version not in {2, 3}:
            raise PreparationError(f"unsupported GGUF version: {version}")
        tensor_count = struct.unpack("<Q", _read_exact(source, 8))[0]
        metadata_count = struct.unpack("<Q", _read_exact(source, 8))[0]
        if tensor_count > 10_000_000 or metadata_count > 10_000_000:
            raise PreparationError("unreasonable GGUF header counts")
        wanted: dict[str, Any] = {}
        wanted_keys = {"general.architecture", "general.file_type", "general.type"}
        for _ in range(metadata_count):
            key = _read_gguf_string(source)
            value_type = struct.unpack("<I", _read_exact(source, 4))[0]
            keep = key in wanted_keys
            value = _read_gguf_value(source, value_type, keep=keep)
            if keep:
                wanted[key] = value

        histogram: dict[str, int] = {}
        for _ in range(tensor_count):
            _read_gguf_string(source)
            dimensions = struct.unpack("<I", _read_exact(source, 4))[0]
            if dimensions > 16:
                raise PreparationError(f"unreasonable GGUF tensor dimensions: {dimensions}")
            _read_exact(source, dimensions * 8)
            tensor_type = struct.unpack("<I", _read_exact(source, 4))[0]
            _read_exact(source, 8)
            type_name = GGML_TYPE_NAMES.get(tensor_type, f"UNKNOWN_{tensor_type}")
            histogram[type_name] = histogram.get(type_name, 0) + 1
    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "architecture": wanted.get("general.architecture"),
        "file_type": wanted.get("general.file_type"),
        "general_type": wanted.get("general.type"),
        "tensor_type_histogram": dict(sorted(histogram.items())),
    }


def _validate_bf16(inspection: dict[str, Any]) -> None:
    if inspection["file_type"] != BF16_FILE_TYPE:
        raise PreparationError(
            "source GGUF is not MOSTLY_BF16 (32); refusing requantization: "
            f"{inspection['file_type']!r}"
        )
    histogram = inspection["tensor_type_histogram"]
    if histogram.get("BF16", 0) == 0:
        raise PreparationError("source GGUF has no BF16 tensors")
    quantized = sorted(set(histogram) - HIGH_PRECISION_TENSOR_TYPES)
    if quantized:
        raise PreparationError(
            f"source GGUF contains non-high-precision tensor types: {quantized}"
        )


def _validate_imatrix(inspection: dict[str, Any]) -> None:
    if inspection["general_type"] != "imatrix":
        raise PreparationError("importance matrix GGUF does not have general.type=imatrix")
    if inspection["tensor_count"] == 0:
        raise PreparationError("importance matrix GGUF contains no tensors")


def _validate_quantized(
    inspection: dict[str, Any], *, quantization: str, source: dict[str, Any]
) -> None:
    expected_file_type = OUTPUT_FILE_TYPES[quantization]
    if inspection["file_type"] != expected_file_type:
        raise PreparationError(
            f"{quantization} output has file type {inspection['file_type']!r}, "
            f"expected {expected_file_type}"
        )
    primary_type = "Q5_K" if quantization == "Q5_K_M" else "Q4_K"
    if inspection["tensor_type_histogram"].get(primary_type, 0) == 0:
        raise PreparationError(f"{quantization} output has no {primary_type} tensors")
    if inspection["tensor_count"] != source["tensor_count"]:
        raise PreparationError(
            f"{quantization} tensor count differs from BF16 source: "
            f"{inspection['tensor_count']} != {source['tensor_count']}"
        )
    if inspection["architecture"] != source["architecture"]:
        raise PreparationError(f"{quantization} architecture differs from BF16 source")


def _run(
    argv: list[str], *, cwd: Path, timeout_seconds: int
) -> dict[str, str]:
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
        raise PreparationError(
            f"command timed out after {timeout_seconds}s: {argv[0]}"
        ) from error
    if result.returncode != 0:
        stderr = result.stderr.strip()[-4000:]
        raise PreparationError(
            f"command failed ({result.returncode}): {argv!r}\n{stderr}"
        )
    return {
        "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
    }


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise PreparationError(f"refusing to overwrite stale partial manifest: {partial}")
    partial.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, path)


def _load_json(path: Path) -> dict[str, Any]:
    path = require_regular_file(path, label="preparation manifest")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"invalid preparation manifest: {path}") from error
    if not isinstance(value, dict):
        raise PreparationError(f"preparation manifest is not an object: {path}")
    return value


def _toolchain(bin_dir: Path) -> dict[str, Any]:
    bin_dir = bin_dir.resolve()
    imatrix = require_regular_file(
        bin_dir / "llama-imatrix", label="llama-imatrix", executable=True
    )
    quantize = require_regular_file(
        bin_dir / "llama-quantize", label="llama-quantize", executable=True
    )
    return {
        "bin_dir": str(bin_dir),
        "llama_imatrix": _stable_file_record(imatrix, label="llama-imatrix"),
        "llama_quantize": _stable_file_record(quantize, label="llama-quantize"),
    }


def _command_plan(
    *,
    source: Path,
    calibration: Path,
    toolchain: dict[str, Any],
    imatrix_partial: Path,
    q5_partial: Path,
    q4_partial: Path,
    ngl: int,
    threads: int,
    context_size: int,
    chunks: int,
) -> dict[str, dict[str, Any]]:
    imatrix_binary = toolchain["llama_imatrix"]["path"]
    quantize_binary = toolchain["llama_quantize"]["path"]
    cwd = toolchain["bin_dir"]
    return {
        "imatrix": {
            "argv": [
                imatrix_binary,
                "-m",
                str(source),
                "-f",
                str(calibration),
                "-o",
                str(imatrix_partial),
                "--output-format",
                "gguf",
                "--no-ppl",
                "--output-frequency",
                "0",
                "--save-frequency",
                "0",
                "-ngl",
                str(ngl),
                "-t",
                str(threads),
                "-c",
                str(context_size),
                "--chunks",
                str(chunks),
            ],
            "cwd": cwd,
        },
        "q5_k_m": {
            "argv": [
                quantize_binary,
                "--imatrix",
                str(imatrix_partial),
                str(source),
                str(q5_partial),
                "Q5_K_M",
                str(threads),
            ],
            "cwd": cwd,
        },
        "q4_k_m": {
            "argv": [
                quantize_binary,
                "--imatrix",
                str(imatrix_partial),
                str(source),
                str(q4_partial),
                "Q4_K_M",
                str(threads),
            ],
            "cwd": cwd,
        },
    }


def _artifact_record(path: Path, *, kind: str, source: dict[str, Any]) -> dict[str, Any]:
    record = _stable_file_record(path, label=kind)
    inspection = inspect_gguf(path)
    if kind == "imatrix":
        _validate_imatrix(inspection)
    else:
        _validate_quantized(inspection, quantization=kind, source=source)
    return {**record, "gguf": inspection}


def _artifact_records(
    *, imatrix: Path, q5: Path, q4: Path, source: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        "imatrix": _artifact_record(imatrix, kind="imatrix", source=source),
        "q5_k_m": _artifact_record(q5, kind="Q5_K_M", source=source),
        "q4_k_m": _artifact_record(q4, kind="Q4_K_M", source=source),
    }


def _expected_coordinates(
    *,
    source_record: dict[str, Any],
    calibration_record: dict[str, Any],
    toolchain: dict[str, Any],
    commands: dict[str, dict[str, Any]],
    artifacts: dict[str, Path],
    manifest_path: Path,
    coordinates: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "source_path": source_record["path"],
        "source_sha256": source_record["sha256"],
        "calibration_path": calibration_record["path"],
        "calibration_sha256": calibration_record["sha256"],
        "imatrix_binary_sha256": toolchain["llama_imatrix"]["sha256"],
        "quantize_binary_sha256": toolchain["llama_quantize"]["sha256"],
        "commands": {name: command["argv"] for name, command in commands.items()},
        "artifact_paths": {name: str(path) for name, path in artifacts.items()},
        "manifest_path": str(manifest_path),
        "coordinates": coordinates,
    }


def _manifest_coordinates(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "kind": manifest.get("kind"),
        "source_path": manifest.get("source", {}).get("path"),
        "source_sha256": manifest.get("source", {}).get("sha256"),
        "calibration_path": manifest.get("calibration", {}).get("path"),
        "calibration_sha256": manifest.get("calibration", {}).get("sha256"),
        "imatrix_binary_sha256": manifest.get("toolchain", {})
        .get("llama_imatrix", {})
        .get("sha256"),
        "quantize_binary_sha256": manifest.get("toolchain", {})
        .get("llama_quantize", {})
        .get("sha256"),
        "commands": {
            name: manifest.get("commands", {}).get(name, {}).get("argv")
            for name in ("imatrix", "q5_k_m", "q4_k_m")
        },
        "artifact_paths": {
            name: manifest.get("artifacts", {}).get(name, {}).get("path")
            for name in ("imatrix", "q5_k_m", "q4_k_m")
        },
        "manifest_path": manifest.get("manifest_path"),
        "coordinates": manifest.get("coordinates"),
    }


def _validate_existing_manifest(
    manifest: dict[str, Any],
    *,
    expected: dict[str, Any],
    artifact_paths: dict[str, Path],
    source: dict[str, Any],
) -> dict[str, Any]:
    actual = _manifest_coordinates(manifest)
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise PreparationError(
            f"existing preparation does not match {', '.join(mismatches)}"
        )
    current = _artifact_records(
        imatrix=artifact_paths["imatrix"],
        q5=artifact_paths["q5_k_m"],
        q4=artifact_paths["q4_k_m"],
        source=source,
    )
    for name, record in current.items():
        claimed = manifest.get("artifacts", {}).get(name, {})
        if record["size"] != claimed.get("size") or record["sha256"] != claimed.get(
            "sha256"
        ):
            raise PreparationError(f"existing {name} does not match preparation manifest")
    return manifest


def _make_manifest(
    *,
    source_record: dict[str, Any],
    calibration_record: dict[str, Any],
    toolchain: dict[str, Any],
    coordinates: dict[str, int],
    commands: dict[str, dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
    manifest_path: Path,
    mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "complete",
        "source": source_record,
        "calibration": calibration_record,
        "toolchain": toolchain,
        "coordinates": coordinates,
        "commands": commands,
        "provenance": {
            "mode": mode,
            "original_commands_known": mode == "created",
            "no_requantization": True,
            "source_validation": "MOSTLY_BF16 metadata and high-precision tensor histogram",
        },
        "artifacts": artifacts,
        "manifest_path": str(manifest_path),
    }


def prepare_mixed_bit_models(
    bf16_model: Path,
    calibration: Path,
    llama_bin_dir: Path,
    output_dir: Path,
    *,
    imatrix_path: Path | None = None,
    q5_path: Path | None = None,
    q4_path: Path | None = None,
    manifest_path: Path | None = None,
    ngl: int = 999,
    threads: int = 12,
    context_size: int = 512,
    chunks: int = 100,
    dry_run: bool = False,
    adopt_existing: bool = False,
    timeout_seconds: int = 14400,
) -> dict[str, Any]:
    """Create, reuse, or adopt an atomic imatrix/Q5_K_M/Q4_K_M preparation."""

    if ngl < 0 or threads <= 0 or context_size < 32 or chunks <= 0:
        raise PreparationError("invalid imatrix execution coordinates")
    if timeout_seconds <= 0:
        raise PreparationError("timeout_seconds must be positive")
    bf16_model = require_regular_file(bf16_model, label="BF16 source")
    calibration = require_regular_file(calibration, label="calibration data")
    output_dir = output_dir.resolve()
    imatrix_path = (imatrix_path or output_dir / "imatrix.gguf").resolve()
    q5_path = (q5_path or output_dir / "Q5_K_M.gguf").resolve()
    q4_path = (q4_path or output_dir / "Q4_K_M.gguf").resolve()
    manifest_path = (manifest_path or output_dir / "preparation.json").resolve()
    artifacts = {"imatrix": imatrix_path, "q5_k_m": q5_path, "q4_k_m": q4_path}
    all_paths = [*artifacts.values(), manifest_path]
    if len(all_paths) != len(set(all_paths)):
        raise PreparationError("artifact and manifest paths must be distinct")
    if any(path.suffix.lower() != ".gguf" for path in artifacts.values()):
        raise PreparationError("imatrix and quantized outputs must use .gguf suffixes")
    forbidden_inputs = {bf16_model, calibration}
    if any(path in forbidden_inputs for path in all_paths):
        raise PreparationError("outputs must not overwrite source or calibration inputs")

    source_record = _stable_file_record(bf16_model, label="BF16 source")
    source_inspection = inspect_gguf(bf16_model)
    _validate_bf16(source_inspection)
    source_record["gguf"] = source_inspection
    calibration_record = _stable_file_record(calibration, label="calibration data")
    if calibration_record["size"] == 0:
        raise PreparationError("calibration data must not be empty")
    toolchain = _toolchain(llama_bin_dir)
    partials = {
        name: path.with_name(path.name + ".partial") for name, path in artifacts.items()
    }
    commands = _command_plan(
        source=bf16_model,
        calibration=calibration,
        toolchain=toolchain,
        imatrix_partial=partials["imatrix"],
        q5_partial=partials["q5_k_m"],
        q4_partial=partials["q4_k_m"],
        ngl=ngl,
        threads=threads,
        context_size=context_size,
        chunks=chunks,
    )
    coordinates = {
        "ngl": ngl,
        "threads": threads,
        "context_size": context_size,
        "chunks": chunks,
    }
    expected = _expected_coordinates(
        source_record=source_record,
        calibration_record=calibration_record,
        toolchain=toolchain,
        commands=commands,
        artifacts=artifacts,
        manifest_path=manifest_path,
        coordinates=coordinates,
    )

    artifact_exists = {
        name: path.exists() or path.is_symlink() for name, path in artifacts.items()
    }
    manifest_exists = manifest_path.exists() or manifest_path.is_symlink()
    if manifest_exists:
        if not all(artifact_exists.values()):
            raise PreparationError("manifest exists without all three prepared artifacts")
        existing = _validate_existing_manifest(
            _load_json(manifest_path),
            expected=expected,
            artifact_paths=artifacts,
            source=source_inspection,
        )
        return {**existing, "action": "reuse"}
    if any(artifact_exists.values()) and not all(artifact_exists.values()):
        raise PreparationError("prepared artifacts must either all exist or all be absent")
    if all(artifact_exists.values()) and not adopt_existing:
        raise PreparationError(
            "artifacts exist without a manifest; pass --adopt-existing to validate them"
        )

    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "source": source_record,
        "calibration": calibration_record,
        "toolchain": toolchain,
        "coordinates": coordinates,
        "commands": commands,
        "artifact_paths": {name: str(path) for name, path in artifacts.items()},
        "manifest_path": str(manifest_path),
    }
    if all(artifact_exists.values()):
        adopted_artifacts = _artifact_records(
            imatrix=imatrix_path,
            q5=q5_path,
            q4=q4_path,
            source=source_inspection,
        )
        if dry_run:
            return {**plan, "action": "dry_run_adopt", "artifacts": adopted_artifacts}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = manifest_path.with_name(manifest_path.name + ".lock")
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if manifest_path.exists() or manifest_path.is_symlink():
                raise PreparationError("manifest appeared while acquiring preparation lock")
            adopted_artifacts = _artifact_records(
                imatrix=imatrix_path,
                q5=q5_path,
                q4=q4_path,
                source=source_inspection,
            )
            manifest = _make_manifest(
                source_record=source_record,
                calibration_record=calibration_record,
                toolchain=toolchain,
                coordinates=coordinates,
                commands=commands,
                artifacts=adopted_artifacts,
                manifest_path=manifest_path,
                mode="adopt_existing",
            )
            _write_json_atomic(manifest_path, manifest)
            return {**manifest, "action": "adopt"}

    if adopt_existing:
        raise PreparationError("--adopt-existing requires all three existing artifacts")
    if dry_run:
        return {**plan, "action": "dry_run"}

    for path in artifacts.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_name(manifest_path.name + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(path.exists() or path.is_symlink() for path in all_paths):
            raise PreparationError("output appeared while acquiring preparation lock")
        stale = [path for path in partials.values() if path.exists() or path.is_symlink()]
        if stale:
            raise PreparationError(f"refusing to overwrite stale partial output: {stale[0]}")

        execution: dict[str, dict[str, str]] = {}
        for name in ("imatrix", "q5_k_m", "q4_k_m"):
            command = commands[name]
            execution[name] = _run(
                command["argv"],
                cwd=Path(command["cwd"]),
                timeout_seconds=timeout_seconds,
            )
            if name == "imatrix":
                _validate_imatrix(inspect_gguf(partials[name]))

        prepared_artifacts = _artifact_records(
            imatrix=partials["imatrix"],
            q5=partials["q5_k_m"],
            q4=partials["q4_k_m"],
            source=source_inspection,
        )
        for name, final_path in artifacts.items():
            prepared_artifacts[name]["path"] = str(final_path)
        current_source_coordinates = _file_coordinates(
            bf16_model, label="BF16 source after preparation"
        )
        if current_source_coordinates != source_record["stable_coordinates"]:
            raise PreparationError("BF16 source changed during preparation")
        current_calibration_coordinates = _file_coordinates(
            calibration, label="calibration data after preparation"
        )
        if current_calibration_coordinates != calibration_record["stable_coordinates"]:
            raise PreparationError("calibration data changed during preparation")
        recorded_commands = {
            name: {**command, **execution[name]} for name, command in commands.items()
        }
        manifest = _make_manifest(
            source_record=source_record,
            calibration_record=calibration_record,
            toolchain=toolchain,
            coordinates=coordinates,
            commands=recorded_commands,
            artifacts=prepared_artifacts,
            manifest_path=manifest_path,
            mode="created",
        )
        manifest_partial = manifest_path.with_name(manifest_path.name + ".partial")
        if manifest_partial.exists() or manifest_partial.is_symlink():
            raise PreparationError(
                f"refusing to overwrite stale partial manifest: {manifest_partial}"
            )
        manifest_partial.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name, final_path in artifacts.items():
            if final_path.exists() or final_path.is_symlink():
                raise PreparationError(f"refusing to overwrite output: {final_path}")
            os.replace(partials[name], final_path)
        os.replace(manifest_partial, manifest_path)
        return {**manifest, "action": "create"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--llama-bin-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imatrix-output", type=Path)
    parser.add_argument("--q5-output", type=Path)
    parser.add_argument("--q4-output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--adopt-existing", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    try:
        result = prepare_mixed_bit_models(
            args.bf16_model,
            args.calibration,
            args.llama_bin_dir,
            args.output_dir,
            imatrix_path=args.imatrix_output,
            q5_path=args.q5_output,
            q4_path=args.q4_output,
            manifest_path=args.manifest,
            ngl=args.ngl,
            threads=args.threads,
            context_size=args.context_size,
            chunks=args.chunks,
            dry_run=args.dry_run,
            adopt_existing=args.adopt_existing,
            timeout_seconds=args.timeout_seconds,
        )
    except PreparationError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
