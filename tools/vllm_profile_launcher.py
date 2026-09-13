#!/usr/bin/env python3
"""Verify an offline vLLM profile coordinate, write evidence, then ``execve``.

This file is intentionally stdlib-only.  It runs under the exact Python that
will be profiled and does not import vLLM, torch, or the framework before the
profiler starts.  The immutable coordinate/manifests are verified before any
GPU code is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

_FORBIDDEN_GPU_ENV = (
    "HIP_VISIBLE_DEVICES",
    "HSA_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
)
_FORBIDDEN_PYTHON_ENV = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP")
_VLLM_CLI_MODULE = "vllm.entrypoints.cli.main"
_VLLM_OFFLINE_PREFIX = ("-I", "-m", _VLLM_CLI_MODULE, "bench", "throughput")


class LauncherError(RuntimeError):
    pass


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as error:
        raise LauncherError(f"cannot stat file: {path}") from error
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise LauncherError(f"file must be regular and not a symlink: {path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        after = path.stat(follow_symlinks=False)
    except OSError as error:
        raise LauncherError(f"cannot hash file: {path}") from error
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_id != after_id or size != after.st_size:
        raise LauncherError(f"file changed while hashing: {path}")
    return digest.hexdigest(), size


def _load_json(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    digest, size = _sha256_file(path)
    if size > 64 * 1024 * 1024:
        raise LauncherError(f"manifest is unexpectedly large: {path}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise LauncherError(f"manifest SHA-256 differs: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LauncherError(f"invalid manifest JSON: {path}") from error
    if not isinstance(value, dict):
        raise LauncherError(f"manifest must be a JSON object: {path}")
    return value


def _environment_identity(manifest: dict[str, Any]) -> str:
    distributions = manifest.get("distributions")
    required = manifest.get("required_distributions")
    if not isinstance(distributions, list) or not isinstance(required, list) or not required:
        raise LauncherError("environment manifest has no required distributions")
    summarized: list[dict[str, Any]] = []
    for item in sorted(distributions, key=lambda value: str(value.get("name", "")).lower()):
        if not isinstance(item, dict):
            raise LauncherError("invalid environment distribution entry")
        summarized.append(
            {
                "name": str(item.get("name", "")).lower(),
                "version": item.get("version"),
                "status": item.get("status"),
                "file_count": item.get("file_count"),
                "total_bytes": item.get("total_bytes"),
                "files_sha256": item.get("files_sha256"),
            }
        )
    return _canonical_sha256(
        {
            "schema": "gpuopt.vllm-environment-identity.v1",
            "python_executable": manifest.get("python_executable"),
            "python_executable_resolved": manifest.get("python_executable_resolved"),
            "python_executable_sha256": manifest.get("python_executable_sha256"),
            "python_version": manifest.get("python_version"),
            "python_prefix": manifest.get("python_prefix"),
            "python_base_prefix": manifest.get("python_base_prefix"),
            "pyvenv_cfg_sha256": manifest.get("pyvenv_cfg_sha256"),
            "probe_sha256": manifest.get("probe_sha256"),
            "framework_source_sha256": manifest.get("framework_source_sha256"),
            "required_distributions": sorted(str(item).lower() for item in required),
            "distributions": summarized,
        }
    )


def _framework_source_identity() -> str:
    spec = importlib.util.find_spec("amd_inference_opt")
    locations = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
    if len(locations) != 1:
        raise LauncherError("cannot resolve one installed framework source root")
    try:
        root = Path(locations[0]).resolve(strict=True)
        paths = sorted(
            path
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    except OSError as error:
        raise LauncherError("cannot inventory installed framework source") from error
    if not paths or len(paths) > 4096:
        raise LauncherError("framework source inventory is empty or unbounded")
    digest = hashlib.sha256()
    for path in paths:
        try:
            if path.is_symlink():
                raise LauncherError(f"framework source cannot be a symlink: {path}")
            before = path.stat(follow_symlinks=False)
            file_sha, size = _sha256_file(path)
            after = path.stat(follow_symlinks=False)
        except OSError as error:
            raise LauncherError(f"cannot inspect framework source: {path}") from error
        before_id = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_id = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_id != after_id:
            raise LauncherError(f"framework source changed while hashed: {path}")
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_environment_manifest(
    path: Path,
    *,
    expected_identity: str,
) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("schema") != "gpuopt.vllm-environment-manifest.v1":
        raise LauncherError("unexpected environment manifest schema")
    if manifest.get("identity_sha256") != expected_identity:
        raise LauncherError("environment manifest identity differs from the coordinate")
    if _environment_identity(manifest) != expected_identity:
        raise LauncherError("environment manifest identity is internally inconsistent")
    lexical_executable = Path(os.path.abspath(sys.executable))
    executable = Path("/proc/self/exe").resolve(strict=True)
    executable_sha, _ = _sha256_file(executable)
    if str(lexical_executable) != manifest.get("python_executable"):
        raise LauncherError("running lexical Python differs from the environment manifest")
    if str(executable) != manifest.get("python_executable_resolved"):
        raise LauncherError("running resolved Python differs from the environment manifest")
    if executable_sha != manifest.get("python_executable_sha256"):
        raise LauncherError("running Python differs from the environment manifest")
    if sys.version.splitlines()[0] != manifest.get("python_version"):
        raise LauncherError("running Python version differs from the environment manifest")
    prefix = Path(sys.prefix).resolve(strict=True)
    base_prefix = Path(sys.base_prefix).resolve(strict=True)
    if str(prefix) != manifest.get("python_prefix"):
        raise LauncherError("running Python prefix differs from the environment manifest")
    if str(base_prefix) != manifest.get("python_base_prefix"):
        raise LauncherError("running Python base prefix differs from the environment manifest")
    pyvenv_cfg = prefix / "pyvenv.cfg"
    observed_pyvenv_sha = _sha256_file(pyvenv_cfg)[0] if pyvenv_cfg.is_file() else None
    if observed_pyvenv_sha != manifest.get("pyvenv_cfg_sha256"):
        raise LauncherError("pyvenv.cfg differs from the environment manifest")
    if _framework_source_identity() != manifest.get("framework_source_sha256"):
        raise LauncherError("framework source differs from the environment manifest")

    required = {str(item).lower() for item in manifest["required_distributions"]}
    present: set[str] = set()
    for item in manifest["distributions"]:
        if not isinstance(item, dict) or item.get("status") != "present":
            continue
        name = str(item.get("name", ""))
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise LauncherError(f"manifest distribution is no longer installed: {name}") from error
        if distribution.version != item.get("version"):
            raise LauncherError(f"distribution version changed: {name}")
        files = item.get("files")
        if not isinstance(files, list) or len(files) != item.get("file_count"):
            raise LauncherError(f"distribution file manifest is invalid: {name}")
        observed_summary = hashlib.sha256()
        observed_total = 0
        for file_item in files:
            if not isinstance(file_item, dict):
                raise LauncherError(f"invalid distribution file entry: {name}")
            relative_text = str(file_item.get("relative_path", ""))
            portable = PurePosixPath(relative_text)
            if portable.is_absolute() or relative_text in {"", "."}:
                raise LauncherError(f"unsafe distribution path: {relative_text}")
            if portable.parts and portable.parts[0] == "@prefix":
                file_path = Path(sys.prefix).joinpath(*portable.parts[1:])
            else:
                file_path = Path(distribution.locate_file(relative_text))
            file_sha, file_size = _sha256_file(file_path.resolve(strict=True))
            if file_sha != file_item.get("sha256") or file_size != file_item.get("size_bytes"):
                raise LauncherError(f"distribution file changed: {name}/{relative_text}")
            observed_total += file_size
            observed_summary.update(relative_text.encode("utf-8"))
            observed_summary.update(b"\0")
            observed_summary.update(file_sha.encode("ascii"))
            observed_summary.update(b"\0")
            observed_summary.update(str(file_size).encode("ascii"))
            observed_summary.update(b"\n")
        if observed_total != item.get("total_bytes"):
            raise LauncherError(f"distribution byte count changed: {name}")
        if observed_summary.hexdigest() != item.get("files_sha256"):
            raise LauncherError(f"distribution file identity changed: {name}")
        present.add(name.lower())
    if not required.issubset(present):
        raise LauncherError("required distributions are absent from the current environment")
    return manifest


def _snapshot_identity(manifest: dict[str, Any]) -> str:
    return _canonical_sha256(
        {
            "schema": "gpuopt.vllm-model-snapshot-identity.v1",
            "model_id": manifest.get("model_id"),
            "revision": manifest.get("revision"),
            "tokenizer_revision": manifest.get("tokenizer_revision"),
            "files": [
                {
                    "relative_path": item.get("relative_path"),
                    "sha256": item.get("sha256"),
                    "size_bytes": item.get("size_bytes"),
                }
                for item in manifest.get("files", [])
                if isinstance(item, dict)
            ],
        }
    )


def _verify_model_manifest(
    path: Path,
    *,
    expected_file_sha256: str,
    coordinate: dict[str, Any],
) -> dict[str, Any]:
    manifest = _load_json(path, expected_file_sha256)
    if manifest.get("schema") != "gpuopt.vllm-model-snapshot.v1":
        raise LauncherError("unexpected model snapshot manifest schema")
    expected_model = coordinate.get("model")
    if not isinstance(expected_model, dict):
        raise LauncherError("profile coordinate lacks model identity")
    for name in ("model_id", "revision", "tokenizer_revision", "snapshot_digest"):
        if manifest.get(name) != expected_model.get(name):
            raise LauncherError(f"model manifest {name} differs from profile coordinate")
    if _snapshot_identity(manifest) != manifest.get("snapshot_digest"):
        raise LauncherError("model snapshot digest is internally inconsistent")
    root = Path(str(manifest.get("root", "")))
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise LauncherError("model snapshot root is not a real absolute directory")
    if str(root.resolve()) != expected_model.get("local_path"):
        raise LauncherError("model snapshot root differs from profile coordinate")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise LauncherError("model snapshot manifest has no files")
    observed_paths: set[str] = set()
    observed_total = 0
    for item in files:
        if not isinstance(item, dict):
            raise LauncherError("invalid model snapshot file entry")
        relative_text = str(item.get("relative_path", ""))
        portable = PurePosixPath(relative_text)
        if portable.is_absolute() or ".." in portable.parts or relative_text in {"", "."}:
            raise LauncherError(f"unsafe model snapshot path: {relative_text}")
        lexical = root.joinpath(*portable.parts)
        if item.get("storage") == "regular" and lexical.is_symlink():
            raise LauncherError(f"model file unexpectedly became a symlink: {relative_text}")
        if item.get("storage") == "symlink" and not lexical.is_symlink():
            raise LauncherError(f"model symlink unexpectedly changed: {relative_text}")
        target = lexical.resolve(strict=True)
        if not target.is_file():
            raise LauncherError(f"model snapshot entry is not a file: {relative_text}")
        digest, size = _sha256_file(target)
        if digest != item.get("sha256") or size != item.get("size_bytes"):
            raise LauncherError(f"model snapshot file changed: {relative_text}")
        observed_paths.add(relative_text)
        observed_total += size
    actual_paths: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        if any((base / name).is_symlink() for name in directory_names):
            raise LauncherError("model snapshot contains a directory symlink")
        for name in file_names:
            actual_paths.add((base / name).relative_to(root).as_posix())
    if actual_paths != observed_paths:
        raise LauncherError("model snapshot file inventory changed")
    if len(files) != manifest.get("file_count") or observed_total != manifest.get("total_bytes"):
        raise LauncherError("model snapshot counts changed")
    return manifest


def _option_values(argv: list[str], name: str) -> list[str | None]:
    values: list[str | None] = []
    for index, item in enumerate(argv):
        if item == name:
            values.append(argv[index + 1] if index + 1 < len(argv) else None)
        elif item.startswith(f"{name}="):
            values.append(item.split("=", 1)[1])
    return values


def _verify_offline_argv(argv: list[str], coordinate: dict[str, Any]) -> None:
    if not argv or not Path(argv[0]).is_absolute() or any("\x00" in item for item in argv):
        raise LauncherError("offline vLLM argv must use an absolute NUL-free executable")
    if tuple(argv[1:6]) != _VLLM_OFFLINE_PREFIX:
        raise LauncherError(
            "offline workload must be the isolated official vLLM throughput module"
        )
    if _canonical_sha256(argv) != coordinate.get("offline_argv_sha256"):
        raise LauncherError("offline argv differs from the profile coordinate")
    expected = coordinate.get("workload")
    model = coordinate.get("model")
    if not isinstance(expected, dict) or not isinstance(model, dict):
        raise LauncherError("profile coordinate lacks workload/model fields")
    checks = {
        "--backend": "vllm",
        "--dataset-name": "random",
        "--model": str(model.get("local_path")),
        "--tensor-parallel-size": str(expected.get("tensor_parallel_size")),
        "--dtype": str(expected.get("dtype")),
        "--input-len": str(expected.get("input_tokens")),
        "--output-len": str(expected.get("output_tokens")),
        "--num-prompts": str(expected.get("num_prompts")),
        "--seed": str(expected.get("seed")),
        "--output-json": str(expected.get("completion_output")),
    }
    for name, value in checks.items():
        if _option_values(argv, name) != [value]:
            raise LauncherError(f"offline argv must bind {name} exactly to {value!r}")
    quantization = expected.get("quantization")
    expected_quantization = [] if quantization is None else [str(quantization)]
    if _option_values(argv, "--quantization") != expected_quantization:
        raise LauncherError("offline argv quantization differs from the profile coordinate")
    if any(
        _option_values(argv, name)
        for name in (
            "--random-input-len",
            "--random-output-len",
            "--dataset-path",
            "--max-concurrency",
            "--num-warmups",
        )
    ):
        raise LauncherError("offline argv contains a conflicting workload flag")
    completion = Path(str(expected.get("completion_output", "")))
    coordinate_path = Path(str(coordinate.get("coordinate_path", "")))
    if (
        not completion.is_absolute()
        or not coordinate_path.is_absolute()
        or completion.parent != coordinate_path.parent
        or completion.exists()
        or completion.is_symlink()
    ):
        raise LauncherError("offline completion output is not fresh and attempt-scoped")


def _process_start_ticks() -> int:
    try:
        raw = Path("/proc/self/stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        return int(raw[closing + 1 :].strip().split()[19])
    except (OSError, IndexError, ValueError) as error:
        raise LauncherError("cannot read process start ticks") from error


def _cgroup_identity() -> dict[str, str]:
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise LauncherError("cannot read cgroup membership") from error
    values = [line.split(":", 2)[2] for line in lines if line.startswith("0::")]
    if len(values) != 1 or not values[0].startswith("/"):
        raise LauncherError("profile launcher requires one cgroup v2 binding")
    return {"kind": "cgroup_v2", "binding_id": f"cgroup:{values[0]}"}


def _write_once_atomic(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser()
    if not destination.is_absolute():
        raise LauncherError("profile sidecar path must be absolute")
    destination = destination.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise LauncherError("refusing to replace an existing profile sidecar")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _parse() -> tuple[argparse.Namespace, list[str]]:
    try:
        delimiter = sys.argv.index("--")
    except ValueError as error:
        raise LauncherError("launcher arguments must end with `-- <offline argv>`") from error
    metadata_argv = sys.argv[1:delimiter]
    offline_argv = sys.argv[delimiter + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuopt-environment-manifest", type=Path, required=True)
    parser.add_argument("--gpuopt-environment-manifest-sha256", required=True)
    parser.add_argument("--gpuopt-model-manifest", type=Path, required=True)
    parser.add_argument("--gpuopt-model-manifest-sha256", required=True)
    parser.add_argument("--gpuopt-coordinate", type=Path, required=True)
    parser.add_argument("--gpuopt-coordinate-sha256", required=True)
    parser.add_argument("--gpuopt-sidecar", type=Path, required=True)
    arguments = parser.parse_args(metadata_argv)
    if not offline_argv:
        raise LauncherError("offline vLLM argv is empty")
    for name, value in vars(arguments).items():
        if name.endswith("sha256") and re.fullmatch(r"[0-9a-f]{64}", str(value)) is None:
            raise LauncherError(f"{name} must be a lowercase SHA-256 digest")
    return arguments, offline_argv


def main() -> None:
    arguments, offline_argv = _parse()
    coordinate = _load_json(arguments.gpuopt_coordinate, arguments.gpuopt_coordinate_sha256)
    if coordinate.get("schema") != "gpuopt.vllm-profile-coordinate.v1":
        raise LauncherError("unexpected profile coordinate schema")
    environment = _verify_environment_manifest(
        arguments.gpuopt_environment_manifest,
        expected_identity=arguments.gpuopt_environment_manifest_sha256,
    )
    if coordinate.get("environment_manifest_sha256") != environment.get("identity_sha256"):
        raise LauncherError("profile coordinate environment identity differs")
    _verify_model_manifest(
        arguments.gpuopt_model_manifest,
        expected_file_sha256=arguments.gpuopt_model_manifest_sha256,
        coordinate=coordinate,
    )
    _verify_offline_argv(offline_argv, coordinate)
    launcher_sha, _ = _sha256_file(Path(__file__).resolve(strict=True))
    if launcher_sha != coordinate.get("profile_launcher_sha256"):
        raise LauncherError("profile launcher bytes differ from the coordinate")
    executable_argv = Path(offline_argv[0])
    if not executable_argv.is_absolute():
        raise LauncherError("offline executable must be absolute")
    executable = executable_argv.resolve(strict=True)
    executable_sha, _ = _sha256_file(executable)
    if executable_sha != environment.get("python_executable_sha256"):
        raise LauncherError("offline executable differs from the environment manifest")
    expected_environment = coordinate.get("required_environment")
    if not isinstance(expected_environment, dict) or not expected_environment:
        raise LauncherError("profile coordinate has no required environment")
    selected_environment = {
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        **{name: os.environ.get(name) for name in _FORBIDDEN_GPU_ENV},
        **{name: os.environ.get(name) for name in _FORBIDDEN_PYTHON_ENV},
    }
    declared_matches = all(
        os.environ.get(name) == value for name, value in expected_environment.items()
    )
    unset_absent = all(
        os.environ.get(name) is None
        for name in (*_FORBIDDEN_GPU_ENV, *_FORBIDDEN_PYTHON_ENV)
    )
    if not declared_matches or not unset_absent:
        raise LauncherError("profile process environment differs from the coordinate")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    if not boot_id:
        raise LauncherError("host boot_id is empty")
    model = coordinate["model"]
    device = coordinate.get("device")
    if not isinstance(device, dict):
        raise LauncherError("profile coordinate lacks device identity")
    sidecar = {
        "schema": "gpuopt.vllm-profile-launch.v1",
        "pid": os.getpid(),
        "boot_id": boot_id,
        "start_ticks": _process_start_ticks(),
        "native_executable_sha256": executable_sha,
        "environment_manifest_sha256": environment["identity_sha256"],
        "profile_launcher_sha256": launcher_sha,
        "profile_coordinate_sha256": arguments.gpuopt_coordinate_sha256,
        "model_manifest_sha256": arguments.gpuopt_model_manifest_sha256,
        "model_snapshot_digest": model["snapshot_digest"],
        "model_revision": model["revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "serving_protocol_sha256": coordinate["serving_protocol_sha256"],
        "device_uuid": device["device_uuid"],
        "pci_bdf": device["pci_bdf"],
        "partition_id": device["partition_id"],
        "selected_environment": selected_environment,
        "declared_environment_matches": declared_matches,
        "unset_environment_absent": unset_absent,
        "cgroup": _cgroup_identity(),
        "offline_argv": offline_argv,
        "offline_argv_sha256": _canonical_sha256(offline_argv),
        "throughput_result_path": coordinate["workload"]["completion_output"],
        "cwd": str(Path.cwd().resolve()),
    }
    _write_once_atomic(arguments.gpuopt_sidecar, sidecar)
    os.execve(str(executable_argv), offline_argv, os.environ.copy())


if __name__ == "__main__":
    try:
        main()
    except LauncherError as error:
        print(f"vllm profile launcher refused execution: {error}", file=sys.stderr)
        raise SystemExit(2) from error
