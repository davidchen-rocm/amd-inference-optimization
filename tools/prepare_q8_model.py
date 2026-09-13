#!/usr/bin/env python3
"""Prepare a reproducible Q8_0 GGUF from a local Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KIND = "llama_cpp_hf_to_q8_0"
INDEX_NAME = "model.safetensors.index.json"
MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


class PreparationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run(
    argv: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            capture_output=True,
            shell=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise PreparationError(
            f"command timed out after {timeout_seconds}s: {argv!r}"
        ) from error
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if len(stderr) > 4000:
            stderr = stderr[-4000:]
        raise PreparationError(
            f"command failed ({result.returncode}): {argv!r}\n{stderr}"
        )
    return result


def require_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{label} must be a regular, non-symlink file: {path}")


def require_executable(path: Path, *, label: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise PreparationError(f"{label} must be an executable file: {path}")


def _model_input_paths(model_dir: Path) -> list[Path]:
    index_path = model_dir / INDEX_NAME
    require_regular_file(index_path, label="safetensors index")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"invalid safetensors index: {index_path}") from error
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PreparationError(f"safetensors index has no weight_map: {index_path}")

    relative_names = {INDEX_NAME}
    for filename in weight_map.values():
        if not isinstance(filename, str) or not filename:
            raise PreparationError("safetensors weight_map contains an invalid filename")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise PreparationError(f"unsafe safetensors shard path: {filename!r}")
        relative_names.add(relative.as_posix())
    for filename in MODEL_METADATA_FILES:
        if (model_dir / filename).exists():
            relative_names.add(filename)
    if "config.json" not in relative_names:
        raise PreparationError(f"missing model config: {model_dir / 'config.json'}")

    paths: list[Path] = []
    for relative_name in sorted(relative_names):
        path = model_dir / relative_name
        require_regular_file(path, label="model input")
        try:
            path.resolve().relative_to(model_dir)
        except ValueError as error:
            raise PreparationError(f"model input escapes source directory: {path}") from error
        paths.append(path)
    return paths


def build_input_manifest(model_dir: Path) -> dict[str, Any]:
    try:
        index = json.loads((model_dir / INDEX_NAME).read_text(encoding="utf-8"))
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"invalid model metadata in {model_dir}") from error
    files = []
    for path in _model_input_paths(model_dir):
        files.append(
            {
                "path": path.relative_to(model_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    content = {
        "schema_version": SCHEMA_VERSION,
        "files": files,
        "total_size": sum(item["size"] for item in files),
        "tensor_count": len(index["weight_map"]),
        "model_type": config.get("model_type"),
        "torch_dtype": config.get("torch_dtype"),
    }
    return {**content, "sha256": canonical_sha256(content)}


def _converter_identity(llama_repo: Path, *, timeout_seconds: int) -> dict[str, Any]:
    if not (llama_repo / "CMakeLists.txt").is_file():
        raise PreparationError(f"not a llama.cpp checkout: {llama_repo}")
    converter = llama_repo / "convert_hf_to_gguf.py"
    require_regular_file(converter, label="converter")
    commit = run(
        ["git", "rev-parse", "HEAD"],
        cwd=llama_repo,
        timeout_seconds=timeout_seconds,
    ).stdout.strip()
    if not commit:
        raise PreparationError(f"cannot resolve llama.cpp commit: {llama_repo}")
    dirty = run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=llama_repo,
        timeout_seconds=timeout_seconds,
    ).stdout.strip()
    if dirty:
        raise PreparationError(f"llama.cpp checkout has tracked changes: {llama_repo}")
    return {
        "repo": str(llama_repo),
        "commit": commit,
        "script": str(converter),
        "script_sha256": sha256_file(converter),
    }


def _conversion_argv(
    *,
    python_executable: Path,
    converter: Path,
    model_dir: Path,
    destination: Path,
    dry_run: bool,
) -> list[str]:
    argv = [
        str(python_executable),
        str(converter),
        str(model_dir),
        "--outfile",
        str(destination),
        "--outtype",
        "q8_0",
    ]
    if dry_run:
        argv.append("--dry-run")
    return argv


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"invalid preparation manifest: {path}") from error
    if not isinstance(value, dict):
        raise PreparationError(f"preparation manifest must be an object: {path}")
    return value


def _read_exact(source: Any, size: int) -> bytes:
    value = source.read(size)
    if len(value) != size:
        raise PreparationError("truncated GGUF metadata")
    return value


def _read_gguf_string(source: Any) -> str:
    length = struct.unpack("<Q", _read_exact(source, 8))[0]
    if length > 64 * 1024 * 1024:
        raise PreparationError(f"unreasonable GGUF string length: {length}")
    try:
        return _read_exact(source, length).decode("utf-8")
    except UnicodeDecodeError as error:
        raise PreparationError("invalid UTF-8 in GGUF metadata") from error


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
            item = _read_gguf_value(source, element_type, keep=keep)
            if keep:
                values.append(item)
        return values
    raise PreparationError(f"unsupported GGUF metadata value type: {value_type}")


def inspect_q8_gguf(path: Path) -> dict[str, Any]:
    require_regular_file(path, label="Q8 output")
    with path.open("rb") as source:
        if _read_exact(source, 4) != b"GGUF":
            raise PreparationError(f"output does not have GGUF magic: {path}")
        version = struct.unpack("<I", _read_exact(source, 4))[0]
        tensor_count = struct.unpack("<Q", _read_exact(source, 8))[0]
        metadata_count = struct.unpack("<Q", _read_exact(source, 8))[0]
        wanted: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _read_gguf_string(source)
            value_type = struct.unpack("<I", _read_exact(source, 4))[0]
            keep = key in {"general.architecture", "general.file_type"}
            value = _read_gguf_value(source, value_type, keep=keep)
            if keep:
                wanted[key] = value
            if len(wanted) == 2:
                break
    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "architecture": wanted.get("general.architecture"),
        "file_type": wanted.get("general.file_type"),
    }


def _validate_q8_identity(inspection: dict[str, Any], input_manifest: dict[str, Any]) -> None:
    if inspection["version"] not in {2, 3}:
        raise PreparationError(f"unsupported GGUF version: {inspection['version']}")
    if inspection["tensor_count"] != input_manifest["tensor_count"]:
        raise PreparationError(
            "GGUF tensor count does not match source index: "
            f"{inspection['tensor_count']} != {input_manifest['tensor_count']}"
        )
    if inspection["architecture"] != input_manifest["model_type"]:
        raise PreparationError(
            "GGUF architecture does not match source config: "
            f"{inspection['architecture']!r} != {input_manifest['model_type']!r}"
        )
    if inspection["file_type"] != 7:
        raise PreparationError(
            f"GGUF general.file_type is not MOSTLY_Q8_0 (7): {inspection['file_type']!r}"
        )


def _validate_existing(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    output: Path,
    model_dir: Path,
    input_manifest: dict[str, Any],
    converter: dict[str, Any],
    argv: list[str],
) -> dict[str, Any]:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "source_model_dir": str(model_dir),
        "input_manifest_sha256": input_manifest["sha256"],
        "converter_commit": converter["commit"],
        "converter_script_sha256": converter["script_sha256"],
        "argv": argv,
        "output_path": str(output),
    }
    actual = {
        "schema_version": manifest.get("schema_version"),
        "kind": manifest.get("kind"),
        "source_model_dir": manifest.get("source_model_dir"),
        "input_manifest_sha256": manifest.get("input_manifest", {}).get("sha256"),
        "converter_commit": manifest.get("converter", {}).get("commit"),
        "converter_script_sha256": manifest.get("converter", {}).get("script_sha256"),
        "argv": manifest.get("conversion", {}).get("argv"),
        "output_path": manifest.get("output", {}).get("path"),
    }
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise PreparationError(
            f"existing preparation does not match {', '.join(mismatches)}: {manifest_path}"
        )
    require_regular_file(output, label="Q8 output")
    output_record = manifest.get("output", {})
    if output.stat().st_size != output_record.get("size"):
        raise PreparationError(f"existing Q8 output size does not match manifest: {output}")
    if sha256_file(output) != output_record.get("sha256"):
        raise PreparationError(f"existing Q8 output hash does not match manifest: {output}")
    return manifest


def prepare(
    model_dir: Path,
    llama_repo: Path,
    output: Path,
    *,
    python_executable: Path = Path(sys.executable),
    manifest_path: Path | None = None,
    dry_run: bool = False,
    adopt_existing: bool = False,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    if timeout_seconds < 1:
        raise PreparationError("timeout_seconds must be positive")
    model_dir = model_dir.resolve()
    llama_repo = llama_repo.resolve()
    output = output.resolve()
    python_executable = Path(os.path.abspath(python_executable))
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else output.with_name(output.name + ".preparation.json")
    )
    partial = output.with_name(output.name + ".partial")

    if not model_dir.is_dir():
        raise PreparationError(f"model directory does not exist: {model_dir}")
    require_executable(python_executable, label="Python executable")
    if output.suffix.lower() != ".gguf":
        raise PreparationError(f"Q8 output must use the .gguf suffix: {output}")
    if output == manifest_path:
        raise PreparationError("output and manifest paths must be different")
    try:
        output.relative_to(model_dir)
    except ValueError:
        pass
    else:
        raise PreparationError("Q8 output must not be inside the source model directory")
    try:
        manifest_path.relative_to(model_dir)
    except ValueError:
        pass
    else:
        raise PreparationError("preparation manifest must not be inside the source model directory")

    input_manifest = build_input_manifest(model_dir)
    converter = _converter_identity(llama_repo, timeout_seconds=timeout_seconds)
    final_argv = _conversion_argv(
        python_executable=python_executable,
        converter=Path(converter["script"]),
        model_dir=model_dir,
        destination=partial,
        dry_run=False,
    )

    output_exists = output.exists() or output.is_symlink()
    manifest_exists = manifest_path.exists() or manifest_path.is_symlink()
    if manifest_exists and not output_exists:
        raise PreparationError(
            "Q8 output and preparation manifest must either both exist or both be absent"
        )
    if output_exists and manifest_exists:
        existing = _validate_existing(
            manifest=_load_json(manifest_path),
            manifest_path=manifest_path,
            output=output,
            model_dir=model_dir,
            input_manifest=input_manifest,
            converter=converter,
            argv=final_argv,
        )
        return {**existing, "action": "reuse"}

    dry_argv = _conversion_argv(
        python_executable=python_executable,
        converter=Path(converter["script"]),
        model_dir=model_dir,
        destination=output,
        dry_run=True,
    )
    if output_exists:
        if not adopt_existing:
            raise PreparationError(
                "Q8 output exists without a manifest; pass --adopt-existing to validate and "
                "record it without reconverting"
            )
        if output.is_symlink():
            raise PreparationError(f"refusing to adopt symlinked Q8 output: {output}")
        result = run(dry_argv, cwd=llama_repo, timeout_seconds=timeout_seconds)
        inspection = inspect_q8_gguf(output)
        _validate_q8_identity(inspection, input_manifest)
        validation = {
            "converter_dry_run_argv": dry_argv,
            "converter_dry_run_stdout_sha256": hashlib.sha256(
                result.stdout.encode()
            ).hexdigest(),
            "converter_dry_run_stderr_sha256": hashlib.sha256(
                result.stderr.encode()
            ).hexdigest(),
            "gguf": inspection,
        }
        if dry_run:
            return {
                "schema_version": SCHEMA_VERSION,
                "kind": KIND,
                "action": "dry_run_adopt",
                "source_model_dir": str(model_dir),
                "input_manifest": input_manifest,
                "converter": converter,
                "conversion": {
                    "argv": final_argv,
                    "cwd": str(llama_repo),
                    "outtype": "q8_0",
                },
                "adoption_validation": validation,
                "output": {"path": str(output), "manifest_path": str(manifest_path)},
            }

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = manifest_path.with_name(manifest_path.name + ".lock")
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if manifest_path.exists() or manifest_path.is_symlink():
                raise PreparationError(
                    f"preparation manifest appeared while acquiring lock: {manifest_path}"
                )
            before = output.stat()
            output_sha256 = sha256_file(output)
            after = output.stat()
            stable_coordinates = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            if stable_coordinates != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise PreparationError(f"Q8 output changed while hashing: {output}")
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "kind": KIND,
                "status": "complete",
                "source_model_dir": str(model_dir),
                "input_manifest": input_manifest,
                "converter": converter,
                "conversion": {
                    "argv": final_argv,
                    "cwd": str(llama_repo),
                    "outtype": "q8_0",
                },
                "provenance": {
                    "mode": "adopt_existing",
                    "original_conversion_argv_known": False,
                    "validation": validation,
                },
                "output": {
                    "path": str(output),
                    "size": after.st_size,
                    "sha256": output_sha256,
                    "manifest_path": str(manifest_path),
                },
            }
            manifest_partial = manifest_path.with_name(manifest_path.name + ".partial")
            if manifest_partial.exists() or manifest_partial.is_symlink():
                raise PreparationError(
                    f"refusing to overwrite stale partial manifest: {manifest_partial}"
                )
            manifest_partial.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(manifest_partial, manifest_path)
            return {**manifest, "action": "adopt"}

    if adopt_existing:
        raise PreparationError("--adopt-existing requires an existing Q8 output")
    if dry_run:
        result = run(dry_argv, cwd=llama_repo, timeout_seconds=timeout_seconds)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "action": "dry_run",
            "source_model_dir": str(model_dir),
            "input_manifest": input_manifest,
            "converter": converter,
            "conversion": {
                "argv": final_argv,
                "dry_run_argv": dry_argv,
                "cwd": str(llama_repo),
                "outtype": "q8_0",
                "dry_run_stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                "dry_run_stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
            },
            "output": {"path": str(output), "manifest_path": str(manifest_path)},
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_name(manifest_path.name + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists() or manifest_path.exists():
            raise PreparationError("Q8 output appeared while acquiring preparation lock")
        if partial.exists() or partial.is_symlink():
            raise PreparationError(f"refusing to overwrite stale partial output: {partial}")
        run(final_argv, cwd=llama_repo, timeout_seconds=timeout_seconds)
        require_regular_file(partial, label="converter output")
        output_size = partial.stat().st_size
        if output_size == 0:
            raise PreparationError(f"converter produced an empty output: {partial}")
        inspection = inspect_q8_gguf(partial)
        _validate_q8_identity(inspection, input_manifest)
        output_sha256 = sha256_file(partial)

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "status": "complete",
            "source_model_dir": str(model_dir),
            "input_manifest": input_manifest,
            "converter": converter,
            "conversion": {
                "argv": final_argv,
                "cwd": str(llama_repo),
                "outtype": "q8_0",
            },
            "provenance": {"mode": "created", "gguf": inspection},
            "output": {
                "path": str(output),
                "size": output_size,
                "sha256": output_sha256,
                "manifest_path": str(manifest_path),
            },
        }
        manifest_partial = manifest_path.with_name(manifest_path.name + ".partial")
        if manifest_partial.exists() or manifest_partial.is_symlink():
            raise PreparationError(
                f"refusing to overwrite stale partial manifest: {manifest_partial}"
            )
        manifest_partial.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise PreparationError(f"refusing to overwrite Q8 output: {output}")
        os.replace(partial, output)
        os.replace(manifest_partial, manifest_path)
        return {**manifest, "action": "create"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--llama-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python",
        dest="python_executable",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--manifest", dest="manifest_path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--adopt-existing", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    try:
        result = prepare(
            args.model_dir,
            args.llama_repo,
            args.output,
            python_executable=args.python_executable,
            manifest_path=args.manifest_path,
            dry_run=args.dry_run,
            adopt_existing=args.adopt_existing,
            timeout_seconds=args.timeout_seconds,
        )
    except PreparationError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
