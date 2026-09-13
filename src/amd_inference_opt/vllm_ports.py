"""Concrete, fail-closed execution ports for the vLLM + MI300X V0.

The workflow coordinator intentionally owns only state transitions.  This
module supplies the process/file/MCP boundaries used by a CLI on the rented
MI300X host.  Commands are argv-only, every benchmark sample is read from the
official ``--save-result`` JSON file, and profiler artifacts are copied out of
the ROCm Issue Agent case store before the MCP process can disappear.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from .command import CommandRunner, validate_argv
from .models import (
    ArtifactRef,
    BaselineResult,
    BenchmarkResult,
    EnvironmentFingerprint,
    ExperimentResult,
    MetricSeries,
    QualityResult,
    RunIdentity,
    RunStatus,
)
from .rocm_mcp import (
    MCPServerConfig,
    MCPToolCall,
    RocmIssueAgentClient,
    approval_request,
)
from .store import ExperimentStore, StoreError
from .vllm_adapter import parse_bench_serve_json
from .vllm_environment import VLLMEnvironmentManifest
from .vllm_model_snapshot import (
    VLLMModelSnapshotManifest,
    verify_vllm_model_snapshot,
)
from .vllm_models import (
    MI300XDeviceBinding,
    VLLMCampaignConfig,
    VLLMExperimentSpec,
    VLLMModelCoordinate,
    VLLMProfileExecutionPermit,
    VLLMProfileResultEvidence,
    VLLMProfileRuntimeEvidence,
    VLLMServingProtocol,
    canonical_sha256,
)
from .vllm_workflow import build_server_spec

_CONFLICTING_GPU_SELECTORS = (
    "HIP_VISIBLE_DEVICES",
    "HSA_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
)
_CONFLICTING_PYTHON_ENV = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP")
_UNSET_EXECUTION_ENV = (*_CONFLICTING_GPU_SELECTORS, *_CONFLICTING_PYTHON_ENV)
_RESULT_FLAGS = ("--result-dir", "--result-filename")
_PROFILE_SIDECAR_FLAG = "--gpuopt-sidecar"
_VLLM_CLI_MODULE = "vllm.entrypoints.cli.main"
_VLLM_OFFLINE_PREFIX = ("-I", "-m", _VLLM_CLI_MODULE, "bench", "throughput")
_MAX_JSON_BYTES = 64 * 1024 * 1024


class VLLMPortError(RuntimeError):
    """A concrete vLLM execution boundary could not prove its result."""


class VLLMCommandPort(Protocol):
    """Structural subset of :class:`~amd_inference_opt.command.CommandRunner`."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        unset_env: Sequence[str] = (),
        timeout_seconds: float | None = None,
        stdout_path: str | Path | None = None,
        stderr_path: str | Path | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class VLLMEnvironmentCapture:
    """Fresh pre/post telemetry plus the normalized comparable fingerprint."""

    fingerprint: EnvironmentFingerprint
    before_artifacts: tuple[ArtifactRef, ...]
    after_artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not self.before_artifacts or not self.after_artifacts:
            raise VLLMPortError(
                "MI300X benchmark capture requires raw pre and post telemetry artifacts"
            )


class VLLMEnvironmentWindowPort(Protocol):
    """Capture AMD SMI clocks/temp/power/throttle/partition around a benchmark."""

    def begin(
        self,
        config: VLLMCampaignConfig,
        *,
        phase: str,
        server_request_hash: str,
    ) -> Any: ...

    def finish(
        self,
        config: VLLMCampaignConfig,
        *,
        phase: str,
        server_request_hash: str,
        before: Any,
    ) -> VLLMEnvironmentCapture: ...


@dataclass(frozen=True)
class ProfileContainerObservation:
    """Previously observed outer-container identity for the profile process."""

    binding_kind: str
    binding_id: str
    repo_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.binding_kind != "cgroup_v2":
            raise VLLMPortError("MI300X V0 requires a cgroup_v2 container binding")
        if not self.binding_id.strip() or not self.repo_digests:
            raise VLLMPortError("container observation is missing binding or RepoDigest evidence")


@dataclass(frozen=True)
class MaterializedVLLMProfileCommand:
    """One exact, approval-ready offline profile command and its fresh outputs."""

    attempt_id: str
    argv: tuple[str, ...]
    coordinate_path: Path
    coordinate_sha256: str
    sidecar_path: Path
    environment_manifest_path: Path
    environment_manifest_sha256: str
    model_manifest_path: Path
    model_manifest_sha256: str
    throughput_result_path: Path


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise VLLMPortError(f"refusing to replace immutable materialization: {destination}")
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


def _require_argv_value(argv: Sequence[str], name: str, expected: str) -> None:
    if _option_values(argv, name) != [expected]:
        raise VLLMPortError(f"offline profile argv must bind {name} exactly to {expected!r}")


def materialize_vllm_profile_command(
    *,
    python_executable: str | Path,
    launcher_path: str | Path,
    environment_manifest_path: str | Path,
    model_manifest_path: str | Path,
    model: VLLMModelCoordinate,
    device: MI300XDeviceBinding,
    serving: VLLMServingProtocol,
    offline_argv: Sequence[str],
    output_root: str | Path,
    attempt_id: str,
) -> MaterializedVLLMProfileCommand:
    """Create the immutable coordinate and unique sidecar path for one attempt.

    This function must run *before* the framework creates the exact MCP approval
    request.  A retry receives another ``attempt_id`` and therefore cannot read
    or overwrite an earlier launch sidecar.
    """

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", attempt_id) is None:
        raise VLLMPortError("profile attempt_id contains unsafe characters")
    python = Path(python_executable).expanduser()
    launcher = Path(launcher_path).expanduser()
    environment_path = Path(environment_manifest_path).expanduser()
    model_path = Path(model_manifest_path).expanduser()
    for name, path in {
        "python_executable": python,
        "launcher_path": launcher,
        "environment_manifest_path": environment_path,
        "model_manifest_path": model_path,
    }.items():
        if not path.is_absolute():
            raise VLLMPortError(f"{name} must be absolute")
    python_lexical = python.absolute()
    python = python.resolve(strict=True)
    launcher = launcher.resolve(strict=True)
    environment_path = environment_path.resolve(strict=True)
    model_path = model_path.resolve(strict=True)
    python_sha, _ = _sha256_file(python)
    launcher_sha, _ = _sha256_file(launcher)
    environment_file_sha, _ = _sha256_file(environment_path, max_bytes=_MAX_JSON_BYTES)
    model_file_sha, _ = _sha256_file(model_path, max_bytes=_MAX_JSON_BYTES)
    del environment_file_sha  # Identity is the manifest's canonical package digest.
    try:
        environment_manifest = VLLMEnvironmentManifest.model_validate_json(
            environment_path.read_text(encoding="utf-8")
        )
        model_manifest = VLLMModelSnapshotManifest.model_validate_json(
            model_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as error:
        raise VLLMPortError("profile manifests are invalid") from error
    if environment_manifest.python_executable_sha256 != python_sha:
        raise VLLMPortError("profile Python differs from the environment manifest")
    if (
        environment_manifest.python_executable != python_lexical
        or environment_manifest.python_executable_resolved != python
    ):
        raise VLLMPortError(
            "profile Python lexical/resolved paths differ from the environment manifest"
        )
    if (
        model_manifest.model_id != model.model_id
        or model_manifest.root != model.local_path
        or model.snapshot_manifest_path.resolve(strict=True) != model_path
        or model_manifest.revision != model.revision
        or model_manifest.tokenizer_revision != model.tokenizer_revision
        or model_manifest.snapshot_digest != model.snapshot_digest
    ):
        raise VLLMPortError("profile model manifest differs from the model coordinate")
    verify_vllm_model_snapshot(model_manifest)
    offline = validate_argv(offline_argv)
    offline_python = Path(offline[0]).expanduser()
    if (
        not offline_python.is_absolute()
        or offline_python.absolute() != python_lexical
        or offline_python.resolve(strict=True) != python
    ):
        raise VLLMPortError(
            "offline profile must execute the hash-bound lexical venv Python"
        )
    if tuple(offline[1:6]) != _VLLM_OFFLINE_PREFIX:
        raise VLLMPortError(
            "offline profile must execute `python -I -m "
            f"{_VLLM_CLI_MODULE} bench throughput`"
        )
    bindings = {
        "--backend": "vllm",
        "--dataset-name": "random",
        "--model": str(model.local_path),
        "--tensor-parallel-size": str(serving.tensor_parallel_size),
        "--dtype": serving.dtype,
        "--input-len": str(serving.input_tokens),
        "--output-len": str(serving.output_tokens),
        "--num-prompts": str(serving.num_prompts),
        "--seed": str(serving.seed),
    }
    for name, value in bindings.items():
        _require_argv_value(offline, name, value)
    quantization = _option_values(offline, "--quantization")
    expected_quantization = [] if serving.quantization is None else [serving.quantization]
    if quantization != expected_quantization:
        raise VLLMPortError("offline profile quantization differs from serving")
    if _option_values(offline, "--output-json"):
        raise VLLMPortError("offline profile output JSON is attempt-owned")
    conflicting_shape_flags = [
        name
        for name in (
            "--random-input-len",
            "--random-output-len",
            "--dataset-path",
            "--max-concurrency",
            "--num-warmups",
        )
        if _option_values(offline, name)
    ]
    if conflicting_shape_flags:
        raise VLLMPortError(
            "offline profile contains a conflicting workload flag: "
            + ", ".join(conflicting_shape_flags)
        )
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        raise VLLMPortError("profile materialization output_root must be absolute")
    root = root.resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise VLLMPortError("profile materialization output_root is unsafe")
    attempt_root = root / attempt_id
    try:
        attempt_root.mkdir(mode=0o700)
    except FileExistsError as error:
        raise VLLMPortError("profile attempt materialization already exists") from error
    coordinate_path = attempt_root / "profile-coordinate.json"
    sidecar_path = attempt_root / "profile-launch-sidecar.json"
    throughput_result_path = attempt_root / "vllm-throughput-result.json"
    offline = (*offline, "--output-json", str(throughput_result_path))
    coordinate = {
        "schema": "gpuopt.vllm-profile-coordinate.v1",
        "coordinate_path": str(coordinate_path),
        "model": model.model_dump(mode="json"),
        "device": {
            "gfx_target": device.gfx_target,
            "device_uuid": device.device_uuid,
            "pci_bdf": device.pci_bdf,
            "partition_id": device.partition_id,
            "compute_partition": device.compute_partition,
            "memory_partition": device.memory_partition,
        },
        "workload": {
            "tensor_parallel_size": serving.tensor_parallel_size,
            "dtype": serving.dtype,
            "quantization": serving.quantization,
            "input_tokens": serving.input_tokens,
            "output_tokens": serving.output_tokens,
            "num_prompts": serving.num_prompts,
            "online_serving_concurrency_not_applied": serving.concurrency,
            "seed": serving.seed,
            "completion_output": str(throughput_result_path),
        },
        "serving_protocol_sha256": serving.coordinate_sha256,
        "profile_launcher_sha256": launcher_sha,
        "environment_manifest_sha256": environment_manifest.identity_sha256,
        "required_environment": {
            "ROCR_VISIBLE_DEVICES": device.device_uuid,
            "PYTHONNOUSERSITE": "1",
        },
        "offline_argv_sha256": canonical_sha256(list(offline)),
    }
    _write_json_exclusive(coordinate_path, coordinate)
    coordinate_sha, _ = _sha256_file(coordinate_path, max_bytes=_MAX_JSON_BYTES)
    argv = (
        str(python_lexical),
        "-I",
        str(launcher),
        "--gpuopt-environment-manifest",
        str(environment_path),
        "--gpuopt-environment-manifest-sha256",
        environment_manifest.identity_sha256,
        "--gpuopt-model-manifest",
        str(model_path),
        "--gpuopt-model-manifest-sha256",
        model_file_sha,
        "--gpuopt-coordinate",
        str(coordinate_path),
        "--gpuopt-coordinate-sha256",
        coordinate_sha,
        _PROFILE_SIDECAR_FLAG,
        str(sidecar_path),
        "--",
        *offline,
    )
    return MaterializedVLLMProfileCommand(
        attempt_id=attempt_id,
        argv=argv,
        coordinate_path=coordinate_path,
        coordinate_sha256=coordinate_sha,
        sidecar_path=sidecar_path,
        environment_manifest_path=environment_path,
        environment_manifest_sha256=environment_manifest.identity_sha256,
        model_manifest_path=model_path,
        model_manifest_sha256=model_file_sha,
        throughput_result_path=throughput_result_path,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise VLLMPortError("port evidence is not JSON serializable by contract")


def _sha256_file(path: Path, *, max_bytes: int | None = None) -> tuple[str, int]:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as error:
        raise VLLMPortError(f"cannot stat evidence file: {path}") from error
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise VLLMPortError(f"evidence must be a regular non-symlink file: {path}")
    if max_bytes is not None and before.st_size > max_bytes:
        raise VLLMPortError(f"evidence file exceeds its byte budget: {path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        after = path.stat(follow_symlinks=False)
    except OSError as error:
        raise VLLMPortError(f"cannot hash evidence file: {path}") from error
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or size != after.st_size:
        raise VLLMPortError(f"evidence changed while it was being hashed: {path}")
    return digest.hexdigest(), size


def _absolute_executable_identity(argv: Sequence[str]) -> dict[str, Any]:
    frozen = validate_argv(argv)
    executable = Path(frozen[0])
    if not executable.is_absolute():
        raise VLLMPortError("vLLM/quality executable must use an absolute path")
    resolved = executable.resolve(strict=True)
    executable_sha, executable_size = _sha256_file(resolved)
    identity: dict[str, Any] = {
        "argv0": str(executable),
        "resolved_executable": str(resolved),
        "executable_sha256": executable_sha,
        "executable_size": executable_size,
    }
    # A Python interpreter plus a module/script has a second code-loading
    # coordinate.  Parse only a deliberately small, deterministic flag subset;
    # unsupported interpreter modes fail closed instead of becoming unrecorded.
    if re.fullmatch(r"python(?:3(?:\.\d+)?)?", resolved.name) is not None:
        cursor = 1
        interpreter_flags: list[str] = []
        while cursor < len(frozen) and frozen[cursor] in {
            "-I",
            "-B",
            "-E",
            "-s",
            "-S",
            "-u",
        }:
            interpreter_flags.append(frozen[cursor])
            cursor += 1
        identity["python_flags"] = interpreter_flags
        if cursor + 1 < len(frozen) and frozen[cursor] == "-m":
            module = frozen[cursor + 1]
            if not module or module.startswith("-"):
                raise VLLMPortError("Python module invocation is invalid")
            if "-I" not in interpreter_flags:
                raise VLLMPortError("Python module invocation must use isolated mode (-I)")
            identity["python_module"] = module
        elif cursor < len(frozen) and Path(frozen[cursor]).is_absolute():
            script = Path(frozen[cursor]).resolve(strict=True)
            script_sha, script_size = _sha256_file(script)
            identity.update(
                {
                    "script": str(script),
                    "script_sha256": script_sha,
                    "script_size": script_size,
                }
            )
        else:
            raise VLLMPortError(
                "Python command must bind an isolated module or absolute script"
            )
    return identity


def _benchmark_launcher_identity(
    argv: Sequence[str],
    *,
    runtime_executable: str,
    runtime_executable_sha256: str,
) -> dict[str, Any]:
    """Bind bench serve to the isolated module in the pinned venv Python.

    A generated ``vllm`` console script can bind its own bytes and shebang but
    still inherits ``PYTHONPATH``, ``PYTHONHOME`` and user-site imports.  The
    ``-I -m`` contract removes that unresolved code-loading boundary.
    """

    frozen = validate_argv(argv)
    expected_python = Path(runtime_executable).expanduser()
    if not expected_python.is_absolute():
        raise VLLMPortError("runtime Python coordinate must be absolute")
    if Path(frozen[0]).expanduser().absolute() != expected_python.absolute():
        raise VLLMPortError("vLLM benchmark must use the pinned lexical venv Python")
    expected_prefix = ("-I", "-m", _VLLM_CLI_MODULE, "bench", "serve")
    if tuple(frozen[1:6]) != expected_prefix:
        raise VLLMPortError(
            "vLLM benchmark must execute isolated official "
            f"`python -I -m {_VLLM_CLI_MODULE} bench serve`"
        )
    identity = _absolute_executable_identity(frozen)
    if identity["executable_sha256"] != runtime_executable_sha256:
        raise VLLMPortError("vLLM benchmark Python differs from the pinned runtime")
    identity.update(
        {
            "launcher_kind": "isolated_python_module",
            "python_module": _VLLM_CLI_MODULE,
            "isolated_mode": True,
        }
    )
    return identity


def _option_values(argv: Sequence[str], name: str) -> list[str | None]:
    values: list[str | None] = []
    for index, item in enumerate(argv):
        if item == name:
            values.append(argv[index + 1] if index + 1 < len(argv) else None)
        elif item.startswith(f"{name}="):
            values.append(item.split("=", 1)[1])
    return values


def materialize_bench_serve_argv(
    benchmark_argv: Sequence[str],
    *,
    result_dir: str | Path,
    result_filename: str,
) -> tuple[str, ...]:
    """Add one coordinator-owned, crash-isolated result file to a bench argv.

    Workload flags stay immutable.  Only result plumbing is appended.  Fixed
    user-supplied result paths are rejected because a retry could otherwise
    accept a stale JSON file from a partially completed attempt.
    """

    frozen = validate_argv(benchmark_argv)
    if not any(
        frozen[index : index + 2] == ("bench", "serve") for index in range(max(0, len(frozen) - 1))
    ):
        raise VLLMPortError("benchmark argv must execute `vllm bench serve`")
    # Newer pinned images support native request warmups.  This materializer
    # only owns output paths; it must not impose a hard-coded 0.10.1 CLI on
    # an explicitly configured command from the workload image's help.
    warmups = _option_values(frozen, "--num-warmups")
    if warmups and (
        len(warmups) != 1
        or warmups[0] is None
        or re.fullmatch(r"[0-9]+", warmups[0]) is None
    ):
        raise VLLMPortError("--num-warmups must occur once with a nonnegative integer")
    for name in _RESULT_FLAGS:
        if _option_values(frozen, name):
            raise VLLMPortError(f"{name} is coordinator-owned and cannot be preconfigured")
    save_result_count = sum(item == "--save-result" for item in frozen)
    if save_result_count > 1:
        raise VLLMPortError("benchmark argv contains duplicate --save-result")
    destination = Path(result_dir).expanduser().resolve()
    if not destination.is_dir() or destination.is_symlink():
        raise VLLMPortError("result_dir must be an existing regular directory")
    if Path(result_filename).name != result_filename or result_filename in {"", ".", ".."}:
        raise VLLMPortError("result_filename must be one safe filename")
    additions: list[str] = []
    if save_result_count == 0:
        additions.append("--save-result")
    if "--save-detailed" not in frozen:
        additions.append("--save-detailed")
    if "--disable-tqdm" not in frozen:
        additions.append("--disable-tqdm")
    additions.extend(["--result-dir", str(destination), "--result-filename", result_filename])
    return (*frozen, *additions)


class CommandVLLMQualityPort:
    """Run exact external quality argv and enforce protocol/model provenance."""

    _IDENTITY_PATH = "state/vllm-quality-execution-identity.json"

    def __init__(
        self,
        store: ExperimentStore,
        runner: VLLMCommandPort | None = None,
    ) -> None:
        self.store = store
        self.runner = runner or CommandRunner()

    def evaluate_baseline(self, config: VLLMCampaignConfig) -> QualityResult:
        return self._evaluate(config, label="baseline", establish_identity=True)

    def evaluate(self, config: VLLMCampaignConfig, spec: VLLMExperimentSpec) -> QualityResult:
        return self._evaluate(config, label=spec.id, establish_identity=False)

    def _evaluate(
        self,
        config: VLLMCampaignConfig,
        *,
        label: str,
        establish_identity: bool,
    ) -> QualityResult:
        protocol = config.quality_protocol
        executable_before = _absolute_executable_identity(protocol.argv)
        identity = {
            "schema": "gpuopt.vllm-quality-execution-identity.v1",
            "argv": protocol.argv,
            "cwd": str(protocol.cwd),
            "environment": dict(sorted(protocol.env.items())),
            "unset_environment": list(_UNSET_EXECUTION_ENV),
            "timeout_seconds": protocol.timeout_seconds,
            "protocol_sha256": protocol.coordinate_sha256,
            "model_snapshot_digest": config.model.snapshot_digest,
            "runtime_environment_manifest_sha256": (
                config.task.runtime.environment_manifest_sha256
            ),
            "executable": executable_before,
        }
        identity_sha = canonical_sha256(identity)
        if establish_identity:
            try:
                existing = self.store.artifact_ref(config.task.id, self._IDENTITY_PATH)
                if existing is None:
                    self.store.save_immutable_json(
                        config.task.id,
                        self._IDENTITY_PATH,
                        {**identity, "identity_sha256": identity_sha},
                        producer="vllm-quality-port",
                    )
                else:
                    saved = self.store.load_json(config.task.id, self._IDENTITY_PATH)
                    if saved.get("identity_sha256") != identity_sha:
                        raise VLLMPortError("baseline quality executable identity changed")
            except StoreError as error:
                raise VLLMPortError(str(error)) from error
        else:
            try:
                saved = self.store.load_json(config.task.id, self._IDENTITY_PATH)
            except StoreError as error:
                raise VLLMPortError(
                    "candidate quality cannot run before baseline executable identity exists"
                ) from error
            if saved.get("identity_sha256") != identity_sha:
                raise VLLMPortError("quality executable/script differs from baseline")

        output = protocol.output_path
        before = _file_generation(output)
        result = self.runner.run(
            protocol.argv,
            cwd=protocol.cwd,
            env=protocol.env,
            unset_env=_UNSET_EXECUTION_ENV,
            timeout_seconds=protocol.timeout_seconds,
        )
        command_ref = self.store.save_evidence_json(
            config.task.id,
            f"vllm/quality/{label}-command",
            _jsonable(result),
            producer="vllm-quality-port",
        )
        if not bool(getattr(result, "succeeded", False)):
            raise VLLMPortError("external quality command failed")
        if _absolute_executable_identity(protocol.argv) != executable_before:
            raise VLLMPortError("quality executable/script changed during execution")
        after = _file_generation(output)
        if after is None or after == before:
            raise VLLMPortError("quality command did not freshly replace its output JSON")
        output_sha, output_size = _sha256_file(output, max_bytes=16 * 1024 * 1024)
        try:
            payload_bytes = output.read_bytes()
            if (
                len(payload_bytes) != output_size
                or hashlib.sha256(payload_bytes).hexdigest() != output_sha
            ):
                raise ValueError("quality output changed while reading")

            def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for name, value in pairs:
                    if name in result:
                        raise ValueError(f"duplicate JSON key: {name}")
                    result[name] = value
                return result

            def reject_constant(value: str) -> None:
                raise ValueError(f"non-finite JSON number: {value}")

            payload = json.loads(
                payload_bytes.decode("utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
            quality = QualityResult.model_validate(payload)
            if (
                (quality.perplexity is not None and not math.isfinite(quality.perplexity))
                or any(not math.isfinite(value) for value in quality.accuracies.values())
            ):
                raise ValueError("quality metrics must be finite")
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as error:
            raise VLLMPortError("quality output is not a strict QualityResult") from error
        output_ref = self.store.import_evidence(
            config.task.id,
            f"vllm/quality/{label}-result",
            output,
            producer="external-vllm-quality",
            media_type="application/json",
        )
        if output_ref.sha256 != output_sha or output_ref.size != output_size:
            raise VLLMPortError("stored quality output differs from the observed file")
        if quality.coordinate_hash != protocol.coordinate_sha256:
            raise VLLMPortError("quality result is not bound to the configured protocol")
        if quality.representation_hash != config.model.snapshot_digest:
            raise VLLMPortError("quality result is not bound to the verified model snapshot")
        return quality.model_copy(
            update={"artifacts": [*quality.artifacts, command_ref, output_ref]}
        )


def _file_generation(path: Path) -> tuple[int, int, int, int] | None:
    try:
        info = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise VLLMPortError(f"output path must be a regular non-symlink file: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


class CommandVLLMBenchmarkPort:
    """Run official vLLM result-file samples and normalize all six Gate metrics."""

    _IDENTITY_PATH = "state/vllm-benchmark-execution-identity.json"

    def __init__(
        self,
        store: ExperimentStore,
        environment: VLLMEnvironmentWindowPort,
        quality: CommandVLLMQualityPort,
        *,
        model_snapshot_manifest: VLLMModelSnapshotManifest | str | Path,
        runner: VLLMCommandPort | None = None,
    ) -> None:
        self.store = store
        self.environment = environment
        self.quality = quality
        self.runner = runner or CommandRunner()
        if isinstance(model_snapshot_manifest, VLLMModelSnapshotManifest):
            self.model_manifest = model_snapshot_manifest
        else:
            try:
                self.model_manifest = VLLMModelSnapshotManifest.model_validate_json(
                    Path(model_snapshot_manifest).read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, ValidationError) as error:
                raise VLLMPortError("invalid vLLM model snapshot manifest") from error

    def capture_baseline(self, config: VLLMCampaignConfig) -> BaselineResult:
        server = build_server_spec(config, self.store)
        benchmark, environment, run_identity, artifacts = self._capture(
            config,
            label="baseline",
            benchmark_argv=config.serving.benchmark_argv,
            server_request_hash=server.request_hash,
            establish_identity=True,
        )
        quality = self.quality.evaluate_baseline(config)
        return BaselineResult(
            environment=environment,
            run_identity=run_identity,
            build_status=RunStatus.REUSED,
            smoke_passed=True,
            benchmark=benchmark,
            quality=quality,
            artifacts=[*artifacts, *quality.artifacts],
        )

    def run_experiment(
        self, config: VLLMCampaignConfig, spec: VLLMExperimentSpec
    ) -> ExperimentResult:
        server = build_server_spec(config, self.store, experiment=spec)
        benchmark, environment, run_identity, artifacts = self._capture(
            config,
            label=spec.id,
            benchmark_argv=spec.benchmark_argv,
            server_request_hash=server.request_hash,
            establish_identity=False,
        )
        return ExperimentResult(
            experiment_id=spec.id,
            environment=environment,
            run_identity=run_identity,
            build_status=RunStatus.REUSED,
            smoke_passed=True,
            e2e=benchmark,
            artifacts=artifacts,
        )

    def _capture(
        self,
        config: VLLMCampaignConfig,
        *,
        label: str,
        benchmark_argv: Sequence[str],
        server_request_hash: str,
        establish_identity: bool,
    ) -> tuple[BenchmarkResult, EnvironmentFingerprint, RunIdentity, list[ArtifactRef]]:
        if config.task.environment.require_stable_telemetry is not True:
            raise VLLMPortError(
                "vLLM MI300X tasks must set environment.require_stable_telemetry=true"
            )
        if (
            config.serving.server_env.get("ROCR_VISIBLE_DEVICES")
            != config.device.device_uuid
            or config.serving.server_env.get("PYTHONNOUSERSITE") != "1"
            or any(name in config.serving.server_env for name in _UNSET_EXECUTION_ENV)
        ):
            raise VLLMPortError(
                "vLLM benchmark environment lacks isolated Python/device coordinates"
            )
        self._verify_model_manifest(config)
        executable = _benchmark_launcher_identity(
            benchmark_argv,
            runtime_executable=config.task.runtime.executable,
            runtime_executable_sha256=config.task.runtime.executable_sha256,
        )
        execution_identity = {
            "schema": "gpuopt.vllm-benchmark-execution-identity.v1",
            "base_argv": list(validate_argv(benchmark_argv)),
            "cwd": str(config.serving.cwd),
            "environment": dict(sorted(config.serving.server_env.items())),
            "unset_environment": list(_UNSET_EXECUTION_ENV),
            "executable": executable,
            "model_snapshot_digest": config.model.snapshot_digest,
            "protocol_sha256": config.serving.coordinate_sha256,
        }
        execution_sha = canonical_sha256(execution_identity)
        self._bind_benchmark_identity(
            config,
            execution_identity,
            execution_sha,
            establish=establish_identity,
        )

        task_dir = self.store.task_dir(config.task.id)
        attempt_dir = task_dir / "workspaces" / "vllm-bench" / label / f"attempt-{uuid.uuid4().hex}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        artifacts: list[ArtifactRef] = []
        values: dict[str, list[float]] = {}
        units: dict[str, str] = {}
        before = self.environment.begin(
            config,
            phase=label,
            server_request_hash=server_request_hash,
        )
        capture: VLLMEnvironmentCapture | None = None
        pending_error: BaseException | None = None
        try:
            total = config.serving.warmup_runs + config.serving.sample_count
            for index in range(total):
                filename = f"run-{index + 1:04d}.json"
                output = attempt_dir / filename
                argv = materialize_bench_serve_argv(
                    benchmark_argv,
                    result_dir=attempt_dir,
                    result_filename=filename,
                )
                command = self.runner.run(
                    argv,
                    cwd=config.serving.cwd,
                    env=config.serving.server_env,
                    unset_env=_UNSET_EXECUTION_ENV,
                    timeout_seconds=config.serving.timeout_seconds,
                )
                artifacts.append(
                    self.store.save_evidence_json(
                        config.task.id,
                        f"vllm/benchmark/{label}-run-{index + 1:04d}-command",
                        _jsonable(command),
                        producer="vllm-benchmark-port",
                    )
                )
                if not bool(getattr(command, "succeeded", False)):
                    raise VLLMPortError(f"vLLM benchmark run {index + 1} failed")
                if (
                    _benchmark_launcher_identity(
                        benchmark_argv,
                        runtime_executable=config.task.runtime.executable,
                        runtime_executable_sha256=config.task.runtime.executable_sha256,
                    )
                    != executable
                ):
                    raise VLLMPortError("vLLM benchmark executable changed during sampling")
                _sha256_file(output, max_bytes=_MAX_JSON_BYTES)
                parsed = parse_bench_serve_json(
                    output,
                    expected_num_prompts=config.serving.num_prompts,
                    require_e2el=("mean_e2el_ms" in config.serving.required_metrics),
                )
                if len(parsed.raw) != 1:
                    raise VLLMPortError("each repeated vLLM result file must contain one run")
                artifacts.append(
                    self.store.import_evidence(
                        config.task.id,
                        f"vllm/benchmark/{label}-run-{index + 1:04d}-result",
                        output,
                        producer="vllm-bench-serve",
                        media_type="application/json",
                    )
                )
                if index < config.serving.warmup_runs:
                    continue
                for name, metric in parsed.canonical_metrics.items():
                    values.setdefault(name, []).append(metric.value)
                    units[name] = metric.unit
        except BaseException as error:
            pending_error = error
        finally:
            try:
                capture = self.environment.finish(
                    config,
                    phase=label,
                    server_request_hash=server_request_hash,
                    before=before,
                )
            except BaseException as telemetry_error:
                if pending_error is None:
                    pending_error = telemetry_error
        if pending_error is not None:
            raise pending_error
        assert capture is not None
        self._validate_environment_capture(config, capture, server_request_hash)
        artifacts.extend(capture.before_artifacts)
        artifacts.extend(capture.after_artifacts)
        artifacts.append(
            self.store.save_evidence_json(
                config.task.id,
                f"vllm/benchmark/{label}-environment",
                capture.fingerprint,
                producer="vllm-mi300x-environment-port",
            )
        )
        missing = set(config.serving.required_metrics) - set(values)
        if missing:
            raise VLLMPortError(
                "vLLM result JSON lacks canonical metrics: " + ", ".join(sorted(missing))
            )
        if any(len(samples) != config.serving.sample_count for samples in values.values()):
            raise VLLMPortError("canonical metric sample count differs from sample_count")
        benchmark = BenchmarkResult(
            status=RunStatus.SUCCEEDED,
            metrics={
                name: MetricSeries(unit=units[name], samples=samples)
                for name, samples in values.items()
            },
        )
        runtime = config.task.runtime
        environment_hash = canonical_sha256(config.environment_coordinates)
        run_identity = RunIdentity(
            protocol_hash=config.serving.coordinate_sha256,
            binary_sha256=str(runtime.executable_sha256),
            source_snapshot_sha256=str(config.serving.engine_config["launcher_sha256"]),
            model_sha256=config.model.snapshot_digest,
            runtime_libraries_hash=str(runtime.environment_manifest_sha256),
            environment_hash=environment_hash,
            sidecar_sha256=runtime.image_digest,
            command_hashes={
                "benchmark_base_argv": canonical_sha256(list(benchmark_argv)),
                "benchmark_executable": executable["executable_sha256"],
                "benchmark_execution_identity": execution_sha,
            },
        )
        return benchmark, capture.fingerprint, run_identity, artifacts

    def _verify_model_manifest(self, config: VLLMCampaignConfig) -> None:
        manifest = self.model_manifest
        if (
            manifest.root != config.model.local_path
            or manifest.model_id != config.model.model_id
            or manifest.revision != config.model.revision
            or manifest.tokenizer_revision != config.model.tokenizer_revision
            or manifest.snapshot_digest != config.model.snapshot_digest
        ):
            raise VLLMPortError("model snapshot manifest differs from campaign coordinates")
        verify_vllm_model_snapshot(manifest)
        path = "state/vllm-model-snapshot.json"
        try:
            reference = self.store.artifact_ref(config.task.id, path)
            if reference is None:
                self.store.save_immutable_json(
                    config.task.id,
                    path,
                    manifest,
                    producer="vllm-model-snapshot",
                )
            else:
                saved = self.store.load_json(config.task.id, path, VLLMModelSnapshotManifest)
                if saved.snapshot_digest != manifest.snapshot_digest:
                    raise VLLMPortError("persisted model snapshot manifest differs")
                if not self.store.verify_artifact(config.task.id, reference):
                    raise VLLMPortError("persisted model snapshot manifest is corrupt")
        except StoreError as error:
            raise VLLMPortError(str(error)) from error

    def _bind_benchmark_identity(
        self,
        config: VLLMCampaignConfig,
        identity: Mapping[str, Any],
        digest: str,
        *,
        establish: bool,
    ) -> None:
        try:
            reference = self.store.artifact_ref(config.task.id, self._IDENTITY_PATH)
            if establish and reference is None:
                self.store.save_immutable_json(
                    config.task.id,
                    self._IDENTITY_PATH,
                    {**identity, "identity_sha256": digest},
                    producer="vllm-benchmark-port",
                )
                return
            saved = self.store.load_json(config.task.id, self._IDENTITY_PATH)
        except StoreError as error:
            raise VLLMPortError(
                "candidate benchmark cannot run before baseline execution identity exists"
            ) from error
        if saved.get("identity_sha256") != digest:
            raise VLLMPortError("benchmark executable/argv identity differs from baseline")

    def _validate_environment_capture(
        self,
        config: VLLMCampaignConfig,
        capture: VLLMEnvironmentCapture,
        server_request_hash: str,
    ) -> None:
        fingerprint = capture.fingerprint
        problems: list[str] = []
        if fingerprint.source != "observed":
            problems.append("source is not observed")
        if fingerprint.capture_id is None or fingerprint.captured_at is None:
            problems.append("fresh capture id/timestamp is absent")
        for name, expected in config.environment_coordinates.items():
            if fingerprint.values.get(name) != expected:
                problems.append(f"{name} differs")
        if fingerprint.values.get("server_request_hash") != server_request_hash:
            problems.append("managed server request hash differs")
        for reference in (*capture.before_artifacts, *capture.after_artifacts):
            if not self.store.verify_artifact(config.task.id, reference):
                problems.append(f"telemetry artifact failed integrity: {reference.path}")
        # ``None`` is retained as explicit missing evidence.  The mandatory
        # stable-telemetry Gate turns it into INCONCLUSIVE instead of fabricating
        # stability here.
        if problems:
            raise VLLMPortError("invalid MI300X environment capture: " + "; ".join(problems))


VLLMMCPClientFactory = Callable[
    [MCPServerConfig, Mapping[str, Any]],
    AbstractAsyncContextManager[RocmIssueAgentClient],
]


@dataclass(frozen=True)
class _ImportedAgentArtifacts:
    references: tuple[ArtifactRef, ...]
    by_source_path: Mapping[str, ArtifactRef]
    retention_refs: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]


class VLLMRocmProfilePort:
    """Execute an exact permit and evacuate verified ROCm case artifacts."""

    def __init__(
        self,
        config: VLLMCampaignConfig,
        store: ExperimentStore,
        *,
        resolved_environment: Mapping[str, str],
        artifact_roots: Sequence[str | Path],
        container_observation: ProfileContainerObservation | None = None,
        client_factory: VLLMMCPClientFactory | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.environment = dict(resolved_environment)
        self.artifact_roots = tuple(Path(path).expanduser().resolve() for path in artifact_roots)
        self.container_observation = container_observation
        self.client_factory = client_factory or self._default_client

    @staticmethod
    def _default_client(
        server: MCPServerConfig, context: Mapping[str, Any]
    ) -> RocmIssueAgentClient:
        return RocmIssueAgentClient(server, approval_context=context)

    def preflight(self, permit: VLLMProfileExecutionPermit) -> None:
        """Reject unavailable or changed contracts before consuming an approval."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.preflight_async(permit))
        raise VLLMPortError("await preflight_async() when running inside an event loop")

    async def preflight_async(self, permit: VLLMProfileExecutionPermit) -> None:
        server = self._server_for_permit(permit)
        async with self.client_factory(server, permit.execution_context) as client:
            await client.check_profile_contract(
                permit.arguments,
                expected_sha256=permit.execution_context.get("profile_contract_sha256"),
            )

    def _server_for_permit(self, permit: VLLMProfileExecutionPermit) -> MCPServerConfig:
        if (
            self.environment.get("ROCR_VISIBLE_DEVICES")
            != self.config.device.device_uuid
            or self.environment.get("PYTHONNOUSERSITE") != "1"
            or any(name in self.environment for name in _UNSET_EXECUTION_ENV)
        ):
            raise VLLMPortError(
                "resolved MCP environment lacks isolated Python/device coordinates"
            )
        exact = approval_request(
            permit.tool,
            permit.arguments,
            execution_context=permit.execution_context,
        )
        if exact.request_sha256 != permit.request_hash:
            raise VLLMPortError("profile permit hash does not bind its exact MCP request")
        environment_sha = canonical_sha256(dict(sorted(self.environment.items())))
        if permit.execution_context.get("mcp_env_sha256") != environment_sha:
            raise VLLMPortError("resolved MCP environment differs from the approved context")
        if permit.execution_context.get("config_identity_sha256") != self.config.identity_sha256:
            raise VLLMPortError("profile permit is bound to another immutable campaign")
        if not permit.execution_context.get("profile_contract_sha256"):
            raise VLLMPortError(
                "profile permit lacks capability negotiation; request a new approval"
            )
        server = MCPServerConfig.from_domain(self.config.task.mcp)
        return MCPServerConfig(
            command=server.command,
            args=server.args,
            env=self.environment,
            cwd=server.cwd,
        )

    async def execute_async(self, permit: VLLMProfileExecutionPermit) -> VLLMProfileResultEvidence:
        server = self._server_for_permit(permit)
        async with self.client_factory(server, permit.execution_context) as client:
            call: MCPToolCall = await client.call_tool(
                permit.tool,
                permit.arguments,
                approval_sha256=permit.request_hash,
            )
        envelope = call.to_dict()
        raw_ref = self.store.save_evidence_json(
            self.config.task.id,
            f"vllm/profile/{permit.request_id}-mcp-envelope",
            envelope,
            producer="rocm-issue-agent-mcp",
        )
        structured, kernel, kernels = self._validate_profile_call(call, permit)
        imported = self._import_agent_artifacts(structured, permit)
        retention_ref = self.store.save_evidence_json(
            self.config.task.id,
            f"vllm/profile/{permit.request_id}-artifact-retention",
            {
                "schema": "gpuopt.vllm-rocm-artifact-retention.v1",
                "request_id": permit.request_id,
                "case_id": structured.get("case_id"),
                "retention_refs": list(imported.retention_refs),
                "warnings": list(imported.warnings),
            },
            producer="vllm-profile-port",
        )
        if imported.retention_refs or imported.warnings:
            raise VLLMPortError(
                "ROCm raw artifacts were not fully evacuated; profile is INCONCLUSIVE; "
                f"see {retention_ref.path}"
            )
        raw_context_ref = next(
            (
                reference
                for path, reference in imported.by_source_path.items()
                if path.endswith("raw/observation/run-context.json")
            ),
            None,
        )
        if raw_context_ref is None:
            raise VLLMPortError("ROCm artifact manifest lacks raw run-context evidence")
        raw_context = self.store.load_json(self.config.task.id, raw_context_ref.path)
        if not isinstance(raw_context, Mapping):
            raise VLLMPortError("ROCm run context is not a JSON object")
        sidecar_path = self._profile_sidecar_path(permit)
        sidecar = self._load_launcher_sidecar(sidecar_path)
        sidecar_ref = self.store.import_evidence(
            self.config.task.id,
            f"vllm/profile/{permit.request_id}-launcher-sidecar",
            sidecar_path,
            producer="vllm-profile-launcher",
            media_type="application/json",
        )
        runtime = self._runtime_evidence(sidecar, permit)
        completion, completion_ref = self._load_profile_completion(permit)
        enriched = self._enrich_run_context(raw_context, runtime, call, permit)
        execution = enriched.get("vllm_execution")
        if not isinstance(execution, dict):  # defensive: created by _enrich_run_context
            raise VLLMPortError("enriched run context lacks vLLM execution evidence")
        execution["offline_vllm_completion"] = {
            "artifact": completion_ref.model_dump(mode="json"),
            "metrics": completion,
        }
        execution_ref = self.store.save_evidence_json(
            self.config.task.id,
            f"vllm/profile/{permit.request_id}-run-context",
            enriched,
            producer="vllm-profile-port",
        )
        mcp_call = dict(envelope)
        mcp_call["framework_evidence"] = {
            "mcp_envelope": raw_ref.model_dump(mode="json"),
            "launcher_sidecar": sidecar_ref.model_dump(mode="json"),
            "offline_vllm_completion": completion_ref.model_dump(mode="json"),
            "offline_vllm_metrics": completion,
            "artifact_retention": retention_ref.model_dump(mode="json"),
            "imported_raw_artifacts": [
                reference.model_dump(mode="json") for reference in imported.references
            ],
        }
        return VLLMProfileResultEvidence(
            mcp_call=mcp_call,
            execution_environment_artifact=execution_ref,
            runtime_evidence=runtime,
            run_context_hash=str(structured["run_context_hash"]),
            image_digest=self.config.task.runtime.image_digest,
            model_snapshot_digest=self.config.model.snapshot_digest,
            model_revision=self.config.model.revision,
            tokenizer_revision=self.config.model.tokenizer_revision,
            serving_protocol_sha256=self.config.serving.coordinate_sha256,
            profile_command_sha256=str(self.config.profile.command_sha256),
            profile_launcher_sha256=str(self.config.profile.launcher_sha256),
            environment_manifest_sha256=str(self.config.task.runtime.environment_manifest_sha256),
            device_uuid=self.config.device.device_uuid,
            pci_bdf=self.config.device.pci_bdf,
            partition_id=self.config.device.partition_id,
            engine_kernel_count=len(kernels),
        )

    def execute(self, permit: VLLMProfileExecutionPermit) -> VLLMProfileResultEvidence:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute_async(permit))
        raise VLLMPortError("execute() cannot be called inside a running event loop")

    def _validate_profile_call(
        self,
        call: MCPToolCall,
        permit: VLLMProfileExecutionPermit,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Any]]:
        if call.tool_name != permit.tool or call.arguments != permit.arguments:
            raise VLLMPortError("ROCm MCP response differs from the approved call")
        if call.contract_error is not None:
            raise VLLMPortError(call.contract_error)
        structured = call.structured_content
        if not isinstance(structured, Mapping):
            raise VLLMPortError("ROCm profile result has no structured object")
        kernel = structured.get("kernel_evidence")
        if not isinstance(kernel, Mapping):
            raise VLLMPortError("ROCm profile result lacks kernel_evidence")
        workload = kernel.get("workload")
        profiler = kernel.get("profiler")
        kernels = kernel.get("kernels")
        problems: list[str] = []
        if structured.get("status") != "completed":
            problems.append("top-level status is not completed")
        if kernel.get("status") != "completed":
            problems.append("kernel status is not completed")
        if kernel.get("preset") != self.config.profile.preset:
            problems.append("profile preset differs")
        if kernel.get("workload_succeeded") is not True:
            problems.append("workload_succeeded is not true")
        if kernel.get("aggregate_timing_complete") is not True:
            problems.append("aggregate kernel timing is incomplete")
        if not isinstance(workload, Mapping) or (
            workload.get("status") != "completed" or workload.get("exit_code") != 0
        ):
            problems.append("workload status/exit code is invalid")
        if not isinstance(profiler, Mapping) or profiler.get("status") != "completed":
            problems.append("profiler status is not completed")
        if not isinstance(kernels, list) or not kernels:
            problems.append("no kernel rows were returned")
            kernels = []
        elif any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("name"), str)
            or not str(item["name"]).strip()
            for item in kernels
        ):
            problems.append("kernel rows contain invalid symbols")
        run_hash = structured.get("run_context_hash")
        if not isinstance(run_hash, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", run_hash) is None:
            problems.append("run_context_hash is missing or invalid")
        if kernel.get("run_context_hash") != run_hash:
            problems.append("kernel and top-level run_context_hash differ")
        if problems:
            raise VLLMPortError("invalid ROCm profile evidence: " + "; ".join(problems))
        return structured, kernel, kernels

    def _import_agent_artifacts(
        self,
        structured: Mapping[str, Any],
        permit: VLLMProfileExecutionPermit,
    ) -> _ImportedAgentArtifacts:
        manifest = structured.get("artifact_manifest")
        case_id = structured.get("case_id")
        if (
            not isinstance(case_id, str)
            or re.fullmatch(r"CASE-[A-Za-z0-9-]+", case_id) is None
            or not isinstance(manifest, list)
            or not manifest
        ):
            raise VLLMPortError("ROCm profile lacks a valid case artifact manifest")
        case_dirs = self._case_directories(case_id)
        imported: list[ArtifactRef] = []
        by_source: dict[str, ArtifactRef] = {}
        retention: list[dict[str, Any]] = []
        warnings: list[str] = []
        seen: set[str] = set()
        used_bytes = 0
        used_files = 0
        ordered = sorted(
            manifest,
            key=lambda item: (
                0
                if isinstance(item, Mapping)
                and str(item.get("path", "")).endswith("raw/observation/run-context.json")
                else 1,
                str(item.get("path", "")) if isinstance(item, Mapping) else "",
            ),
        )
        for index, raw in enumerate(ordered):
            entry = self._manifest_entry(raw)
            source_name = entry["path"]
            if source_name in seen:
                raise VLLMPortError("ROCm artifact manifest contains duplicate paths")
            seen.add(source_name)
            size = entry["size_bytes"]
            if (
                used_files + 1 > self.config.profile.max_trace_files
                or used_bytes + size > self.config.profile.max_trace_bytes
            ):
                retention.append({**entry, "reason": "framework_import_budget_exceeded"})
                continue
            source = self._resolve_manifest_source(case_dirs, entry)
            if source is None:
                retention.append({**entry, "reason": "case_artifact_not_locally_accessible"})
                continue
            observed_sha, observed_size = _sha256_file(
                source, max_bytes=self.config.profile.max_trace_bytes
            )
            if observed_sha != entry["sha256"] or observed_size != size:
                raise VLLMPortError(
                    f"ROCm case artifact failed SHA/size verification: {source_name}"
                )
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(source_name).name)
            reference = self.store.import_evidence(
                self.config.task.id,
                f"vllm/profile/{permit.request_id}/raw/{index + 1:04d}-{safe_name}",
                source,
                producer="rocm-issue-agent-case",
                media_type=entry["content_type"],
            )
            if reference.sha256 != observed_sha or reference.size != observed_size:
                raise VLLMPortError("ExperimentStore copy differs from ROCm case artifact")
            imported.append(reference)
            by_source[source_name] = reference
            used_files += 1
            used_bytes += size
        if not case_dirs:
            warnings.append("no configured ROCm Issue Agent case root contains this case")
        return _ImportedAgentArtifacts(
            references=tuple(imported),
            by_source_path=by_source,
            retention_refs=tuple(retention),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _manifest_entry(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise VLLMPortError("ROCm artifact manifest entry must be an object")
        path = raw.get("path")
        digest = raw.get("sha256")
        size = raw.get("size_bytes")
        content_type = raw.get("content_type")
        ownership = raw.get("ownership_scope")
        retention = raw.get("retention_policy")
        if not isinstance(path, str):
            raise VLLMPortError("ROCm artifact path is invalid")
        portable = PurePosixPath(path)
        if portable.is_absolute() or ".." in portable.parts or path in {"", "."}:
            raise VLLMPortError("ROCm artifact path is unsafe")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise VLLMPortError("ROCm artifact SHA-256 is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise VLLMPortError("ROCm artifact size is invalid")
        metadata = (content_type, ownership, retention)
        if not all(isinstance(item, str) and item.strip() for item in metadata):
            raise VLLMPortError("ROCm artifact metadata is incomplete")
        return {
            "path": portable.as_posix(),
            "sha256": digest,
            "size_bytes": size,
            "content_type": content_type,
            "ownership_scope": ownership,
            "retention_policy": retention,
        }

    def _case_directories(self, case_id: str) -> tuple[Path, ...]:
        selected: list[Path] = []
        for root in self.artifact_roots:
            if root.is_symlink() or not root.is_dir():
                continue
            candidate = root if root.name == case_id else root / case_id
            try:
                candidate = candidate.resolve(strict=True)
            except OSError:
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            if candidate.name != case_id:
                continue
            selected.append(candidate)
        return tuple(selected)

    def _resolve_manifest_source(
        self, case_dirs: Sequence[Path], entry: Mapping[str, Any]
    ) -> Path | None:
        portable = PurePosixPath(str(entry["path"]))
        for case_dir in case_dirs:
            lexical = case_dir.joinpath(*portable.parts)
            source = self._safe_case_file(case_dir, lexical)
            if source is not None:
                return source
            # Privacy may replace a hostname path component.  Recover only by
            # unique content identity inside this exact case directory.
            if any(part.startswith("[") and part.endswith("]") for part in portable.parts):
                matches: list[Path] = []
                inspected = 0
                scan_complete = True
                for candidate in case_dir.rglob("*"):
                    inspected += 1
                    if inspected > 20_000:
                        scan_complete = False
                        break
                    safe = self._safe_case_file(case_dir, candidate)
                    if safe is None:
                        continue
                    try:
                        if safe.stat().st_size != entry["size_bytes"]:
                            continue
                    except OSError:
                        continue
                    digest, _ = _sha256_file(safe)
                    if digest == entry["sha256"]:
                        matches.append(safe)
                if scan_complete and len(matches) == 1:
                    return matches[0]
        return None

    @staticmethod
    def _safe_case_file(case_dir: Path, lexical: Path) -> Path | None:
        try:
            relative = lexical.relative_to(case_dir)
        except ValueError:
            return None
        current = case_dir
        try:
            for part in relative.parts:
                current = current / part
                info = current.lstat()
                if stat.S_ISLNK(info.st_mode):
                    return None
            resolved = lexical.resolve(strict=True)
        except OSError:
            return None
        if not resolved.is_relative_to(case_dir) or not resolved.is_file():
            return None
        return resolved

    @staticmethod
    def _profile_sidecar_path(permit: VLLMProfileExecutionPermit) -> Path:
        command = permit.arguments.get("command")
        if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
            raise VLLMPortError("approved profile command is invalid")
        values = _option_values(command, _PROFILE_SIDECAR_FLAG)
        if len(values) != 1 or values[0] is None:
            raise VLLMPortError("profile command must bind one --gpuopt-sidecar path")
        path = Path(values[0])
        if not path.is_absolute():
            raise VLLMPortError("profile sidecar path must be absolute")
        return path

    @staticmethod
    def _load_launcher_sidecar(path: Path) -> Mapping[str, Any]:
        _sha256_file(path, max_bytes=8 * 1024 * 1024)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise VLLMPortError("invalid profile launcher sidecar JSON") from error
        if not isinstance(payload, Mapping) or payload.get("schema") != (
            "gpuopt.vllm-profile-launch.v1"
        ):
            raise VLLMPortError("unexpected profile launcher sidecar schema")
        return payload

    def _approved_profile_command(
        self, permit: VLLMProfileExecutionPermit
    ) -> tuple[list[str], list[str], str, Mapping[str, Any]]:
        """Return both the approved launch argv and ROCm's observed argv form.

        The approval intentionally binds the lexical virtual-environment Python
        path so Python still discovers ``pyvenv.cfg``.  ROCm Issue Agent records
        argv[0] after resolving that path, however, so comparison must reproduce
        only that documented normalization without changing the command that was
        approved or executed.
        """

        command = permit.arguments.get("command")
        cwd = permit.arguments.get("cwd")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise VLLMPortError("approved profile command is invalid")
        if len(command) < 3 or command[1] != "-I":
            raise VLLMPortError("profile launcher must run under isolated Python")
        if not isinstance(cwd, str) or not Path(cwd).is_absolute():
            raise VLLMPortError("approved profile cwd is invalid")
        materialized = permit.execution_context.get("materialized_profile")
        if not isinstance(materialized, Mapping):
            raise VLLMPortError("profile permit lacks materialized command coordinates")
        expected_materialized_hash = canonical_sha256(command)
        delimiters = [index for index, item in enumerate(command) if item == "--"]
        if len(delimiters) != 1 or delimiters[0] + 1 >= len(command):
            raise VLLMPortError("profile command lacks one exact offline argv delimiter")
        offline_argv = command[delimiters[0] + 1 :]
        throughput_result_path = materialized.get("throughput_result_path")
        expected_offline_argv = [
            *validate_argv(self.config.profile.profile_argv),
            "--output-json",
            throughput_result_path,
        ]
        checks = {
            "materialized argv": materialized.get("argv") == command,
            "materialized command SHA-256": (
                permit.execution_context.get("materialized_profile_command_sha256")
                == expected_materialized_hash
            ),
            "base profile command SHA-256": (
                permit.execution_context.get("profile_command_sha256")
                == self.config.profile.command_sha256
            ),
            "profile launcher SHA-256": (
                permit.execution_context.get("profile_launcher_sha256")
                == self.config.profile.launcher_sha256
            ),
            "environment manifest SHA-256": (
                permit.execution_context.get("environment_manifest_sha256")
                == self.config.task.runtime.environment_manifest_sha256
            ),
            "serving protocol SHA-256": (
                permit.execution_context.get("serving_protocol_sha256")
                == self.config.serving.coordinate_sha256
            ),
            "model snapshot digest": (
                permit.execution_context.get("model_snapshot_digest")
                == self.config.model.snapshot_digest
            ),
            "coordinate path": (
                _option_values(command, "--gpuopt-coordinate")
                == [materialized.get("coordinate_path")]
            ),
            "coordinate SHA-256": (
                _option_values(command, "--gpuopt-coordinate-sha256")
                == [materialized.get("coordinate_sha256")]
            ),
            "sidecar path": (
                _option_values(command, _PROFILE_SIDECAR_FLAG)
                == [materialized.get("sidecar_path")]
            ),
            "environment manifest path": (
                _option_values(command, "--gpuopt-environment-manifest")
                == [materialized.get("environment_manifest_path")]
            ),
            "environment manifest identity": (
                _option_values(command, "--gpuopt-environment-manifest-sha256")
                == [materialized.get("environment_manifest_sha256")]
            ),
            "model manifest path": (
                _option_values(command, "--gpuopt-model-manifest")
                == [materialized.get("model_manifest_path")]
            ),
            "model manifest file SHA-256": (
                _option_values(command, "--gpuopt-model-manifest-sha256")
                == [materialized.get("model_manifest_sha256")]
            ),
            "throughput completion output": (
                _option_values(command, "--output-json")
                == [throughput_result_path]
            ),
            "attempt-owned official offline argv": offline_argv == expected_offline_argv,
        }
        failures = [name for name, matches in checks.items() if not matches]
        if failures:
            raise VLLMPortError(
                "profile permit materialization differs: " + ", ".join(failures)
            )
        executable = command[0]
        if "/" in executable or "\\" in executable:
            executable_path = Path(executable).expanduser()
            if not executable_path.is_absolute():
                executable_path = Path(cwd) / executable_path
            observed_executable = str(executable_path.resolve())
        else:
            observed_executable = shutil.which(executable) or executable
        return command, [observed_executable, *command[1:]], cwd, materialized

    def _load_profile_completion(
        self,
        permit: VLLMProfileExecutionPermit,
    ) -> tuple[dict[str, int | float], ArtifactRef]:
        """Prove the profiled process completed the official vLLM workload.

        ROCm's workload exit status alone only proves that *some* approved
        process exited successfully.  The attempt-owned ``bench throughput``
        result additionally proves that the pinned vLLM CLI loaded the model
        and completed the configured number of prompts.
        """

        _, _, _, materialized = self._approved_profile_command(permit)
        raw_path = materialized.get("throughput_result_path")
        if not isinstance(raw_path, str):
            raise VLLMPortError("profile permit lacks its throughput result path")
        path = Path(raw_path)
        if not path.is_absolute():
            raise VLLMPortError("profile throughput result path must be absolute")
        digest, size = _sha256_file(path, max_bytes=8 * 1024 * 1024)
        try:
            payload_bytes = path.read_bytes()
        except OSError as error:
            raise VLLMPortError("cannot read profile throughput completion") from error
        if len(payload_bytes) != size or hashlib.sha256(payload_bytes).hexdigest() != digest:
            raise VLLMPortError("profile throughput completion changed while reading")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError(f"duplicate JSON key: {name}")
                result[name] = value
            return result

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON number: {value}")

        try:
            payload = json.loads(
                payload_bytes.decode("utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise VLLMPortError("profile throughput completion is not strict JSON") from error
        if not isinstance(payload, Mapping):
            raise VLLMPortError("profile throughput completion must be one JSON object")

        integers: dict[str, int] = {}
        for name in ("num_requests", "total_num_tokens"):
            value = payload.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise VLLMPortError(
                    f"profile throughput completion has invalid {name}"
                )
            integers[name] = value
        if integers["num_requests"] != self.config.serving.num_prompts:
            raise VLLMPortError(
                "profile throughput completion did not finish the configured prompts"
            )

        numbers: dict[str, float] = {}
        for name in ("elapsed_time", "requests_per_second", "tokens_per_second"):
            value = payload.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise VLLMPortError(
                    f"profile throughput completion has invalid {name}"
                )
            numbers[name] = float(value)
        expected_request_rate = integers["num_requests"] / numbers["elapsed_time"]
        expected_token_rate = integers["total_num_tokens"] / numbers["elapsed_time"]
        if not math.isclose(
            numbers["requests_per_second"],
            expected_request_rate,
            rel_tol=5e-4,
            abs_tol=1e-6,
        ):
            raise VLLMPortError("profile request throughput is internally inconsistent")
        if not math.isclose(
            numbers["tokens_per_second"],
            expected_token_rate,
            rel_tol=5e-4,
            abs_tol=1e-6,
        ):
            raise VLLMPortError("profile token throughput is internally inconsistent")

        try:
            reference = self.store.import_evidence(
                self.config.task.id,
                f"vllm/profile/{permit.request_id}-throughput-completion",
                path,
                producer="vllm-bench-throughput",
                media_type="application/json",
            )
        except StoreError as error:
            raise VLLMPortError("cannot preserve profile throughput completion") from error
        if reference.sha256 != digest or reference.size != size:
            raise VLLMPortError(
                "stored profile throughput completion differs from the observed file"
            )
        normalized: dict[str, int | float] = {**integers, **numbers}
        return normalized, reference

    def _runtime_evidence(
        self,
        sidecar: Mapping[str, Any],
        permit: VLLMProfileExecutionPermit,
    ) -> VLLMProfileRuntimeEvidence:
        runtime = self.config.task.runtime
        profile_argv, _, _, materialized = self._approved_profile_command(permit)
        coordinate_hashes = _option_values(profile_argv, "--gpuopt-coordinate-sha256")
        model_manifest_hashes = _option_values(profile_argv, "--gpuopt-model-manifest-sha256")
        if len(coordinate_hashes) != 1 or len(model_manifest_hashes) != 1:
            raise VLLMPortError("profile argv must bind coordinate and model-manifest SHA-256 once")
        expected = {
            "native_executable_sha256": runtime.executable_sha256,
            "environment_manifest_sha256": runtime.environment_manifest_sha256,
            "profile_launcher_sha256": self.config.profile.launcher_sha256,
            "profile_coordinate_sha256": coordinate_hashes[0],
            "model_manifest_sha256": model_manifest_hashes[0],
            "model_snapshot_digest": self.config.model.snapshot_digest,
            "model_revision": self.config.model.revision,
            "tokenizer_revision": self.config.model.tokenizer_revision,
            "serving_protocol_sha256": self.config.serving.coordinate_sha256,
            "device_uuid": self.config.device.device_uuid,
            "pci_bdf": self.config.device.pci_bdf,
            "partition_id": self.config.device.partition_id,
            "throughput_result_path": materialized.get("throughput_result_path"),
        }
        materialized_coordinate = materialized.get("coordinate_sha256")
        materialized_model_manifest = materialized.get("model_manifest_sha256")
        if coordinate_hashes[0] != materialized_coordinate:
            raise VLLMPortError("approved coordinate SHA-256 differs from materialization")
        if model_manifest_hashes[0] != materialized_model_manifest:
            raise VLLMPortError("approved model-manifest SHA-256 differs from materialization")
        mismatches = [name for name, value in expected.items() if sidecar.get(name) != value]
        selected_env = sidecar.get("selected_environment")
        if not isinstance(selected_env, Mapping):
            mismatches.append("selected_environment")
            selected_env = {}
        if selected_env.get("ROCR_VISIBLE_DEVICES") != self.config.device.device_uuid:
            mismatches.append("ROCR_VISIBLE_DEVICES")
        if selected_env.get("PYTHONNOUSERSITE") != "1":
            mismatches.append("PYTHONNOUSERSITE")
        if any(selected_env.get(name) is not None for name in _UNSET_EXECUTION_ENV):
            mismatches.append("conflicting execution environment")
        if sidecar.get("declared_environment_matches") is not True:
            mismatches.append("declared_environment_matches")
        if sidecar.get("unset_environment_absent") is not True:
            mismatches.append("unset_environment_absent")
        delimiters = [index for index, item in enumerate(profile_argv) if item == "--"]
        offline_argv = profile_argv[delimiters[0] + 1 :] if len(delimiters) == 1 else []
        if sidecar.get("offline_argv") != offline_argv:
            mismatches.append("offline_argv")
        if sidecar.get("offline_argv_sha256") != canonical_sha256(offline_argv):
            mismatches.append("offline_argv_sha256")
        if mismatches:
            raise VLLMPortError(
                "profile launcher coordinates differ: " + ", ".join(sorted(set(mismatches)))
            )
        try:
            pid = int(sidecar["pid"])
            start_ticks = int(sidecar["start_ticks"])
            boot_id = str(sidecar["boot_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise VLLMPortError("profile sidecar process identity is invalid") from error
        binding = sidecar.get("cgroup")
        binding_kind: str | None = None
        binding_id: str | None = None
        repo_digests: tuple[str, ...] = ()
        image_matches: bool | None = None
        if runtime.image_digest is not None:
            observation = self.container_observation
            if observation is None or not isinstance(binding, Mapping):
                raise VLLMPortError("profile lacks observed same-container evidence")
            binding_kind = observation.binding_kind
            binding_id = str(binding.get("binding_id", ""))
            if binding_id != observation.binding_id:
                raise VLLMPortError("profile cgroup differs from the baseline container")
            repo_digests = observation.repo_digests
            suffix = f"sha256:{runtime.image_digest}"
            image_matches = any(
                value == suffix or value.endswith(f"@{suffix}") for value in repo_digests
            )
            if not image_matches:
                raise VLLMPortError("observed container RepoDigest differs from config")
        return VLLMProfileRuntimeEvidence(
            verification="VERIFIED",
            observed_pid=pid,
            observed_boot_id=boot_id,
            observed_start_ticks=start_ticks,
            native_executable_sha256=str(runtime.executable_sha256),
            environment_manifest_sha256=str(runtime.environment_manifest_sha256),
            profile_launcher_sha256=str(self.config.profile.launcher_sha256),
            native_executable_matches=True,
            declared_environment_matches=True,
            unset_environment_absent=True,
            rocr_visible_devices=self.config.device.device_uuid,
            image_digest_matches=image_matches,
            container_binding_kind=binding_kind,
            process_binding_id=binding_id,
            container_binding_id=(
                self.container_observation.binding_id
                if self.container_observation is not None
                else None
            ),
            container_repo_digests=list(repo_digests),
        )

    def _enrich_run_context(
        self,
        raw_context: Mapping[str, Any],
        runtime: VLLMProfileRuntimeEvidence,
        call: MCPToolCall,
        permit: VLLMProfileExecutionPermit,
    ) -> dict[str, Any]:
        _, observed_argv, approved_cwd, _ = self._approved_profile_command(permit)
        context = dict(raw_context)
        if context.get("schema") != "rocm.run-context.v1":
            raise VLLMPortError("unexpected ROCm run-context schema")
        run_hash = call.structured_content.get("run_context_hash")
        if context.get("run_context_hash") != run_hash:
            raise VLLMPortError("raw ROCm run-context hash differs from MCP evidence")
        hash_payload = {
            key: value
            for key, value in context.items()
            if key not in {"collected_at", "run_context_hash", "vllm_execution"}
        }
        if f"sha256:{canonical_sha256(hash_payload)}" != run_hash:
            raise VLLMPortError("raw ROCm run-context canonical hash is invalid")
        if context.get("argv") != observed_argv:
            raise VLLMPortError("raw ROCm run-context argv differs from approved argv")
        if context.get("cwd") != str(Path(approved_cwd).resolve()):
            raise VLLMPortError("raw ROCm run-context cwd differs")
        environment = context.get("environment")
        if not isinstance(environment, Mapping) or (
            environment.get("ROCR_VISIBLE_DEVICES") != self.config.device.device_uuid
            or environment.get("PYTHONNOUSERSITE") != "1"
            or any(name in environment for name in _UNSET_EXECUTION_ENV)
        ):
            raise VLLMPortError("raw ROCm run-context device isolation differs")
        gpus = context.get("gpus")
        if not isinstance(gpus, list) or not any(
            isinstance(gpu, Mapping)
            and gpu.get("gfx_architecture") == self.config.device.gfx_target
            and gpu.get("pci_bdf") == self.config.device.pci_bdf
            and gpu.get("partition_id") == self.config.device.partition_id
            and gpu.get("accelerator_partition") == self.config.device.compute_partition
            and gpu.get("memory_partition") == self.config.device.memory_partition
            for gpu in gpus
        ):
            raise VLLMPortError("raw ROCm run-context MI300X partition differs")
        context["vllm_execution"] = {
            "image_digest": self.config.task.runtime.image_digest,
            "model_snapshot_digest": self.config.model.snapshot_digest,
            "model_revision": self.config.model.revision,
            "tokenizer_revision": self.config.model.tokenizer_revision,
            "serving_protocol_sha256": self.config.serving.coordinate_sha256,
            "profile_command_sha256": self.config.profile.command_sha256,
            "profile_launcher_sha256": self.config.profile.launcher_sha256,
            "environment_manifest_sha256": (self.config.task.runtime.environment_manifest_sha256),
            "device_uuid": self.config.device.device_uuid,
            "pci_bdf": self.config.device.pci_bdf,
            "partition_id": self.config.device.partition_id,
            "runtime_evidence": runtime.model_dump(mode="json"),
        }
        return context


__all__ = [
    "CommandVLLMBenchmarkPort",
    "CommandVLLMQualityPort",
    "MaterializedVLLMProfileCommand",
    "ProfileContainerObservation",
    "VLLMEnvironmentCapture",
    "VLLMEnvironmentWindowPort",
    "VLLMPortError",
    "VLLMRocmProfilePort",
    "materialize_bench_serve_argv",
    "materialize_vllm_profile_command",
]
