"""Reproducible llama.cpp decode protocol and Q4 runtime coordinates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .command import StageCommand, argv_sha256, validate_argv


class ProtocolError(ValueError):
    """A benchmark or runtime coordinate is ambiguous or invalid."""


Q4_RDNA_ENVIRONMENT = (
    "LLAMA_Q4_RDNA_SIDECAR",
    "LLAMA_Q4_RDNA_MAPPING",
    "LLAMA_Q4_RDNA_SCOPE",
    "LLAMA_Q4_RDNA_COOP",
    "LLAMA_Q4_RDNA_SMALL_ROWS",
    "LLAMA_Q4_RDNA_GATE_PAIR",
    "LLAMA_Q4_RDNA_TRACE",
)
SMITHY_ENVIRONMENT = ("SMITHY_CONFIG", "SMITHY_MODEL")
BASELINE_UNSET_ENVIRONMENT = Q4_RDNA_ENVIRONMENT + SMITHY_ENVIRONMENT


def _canonical_sha256(document: object) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def amd_runtime_environment(
    *,
    device_id: int = 0,
    rocm_library_paths: Sequence[str] = (
        "/opt/rocm/core-7.14/lib",
        "/opt/rocm/lib",
    ),
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the explicit AMD runtime environment shared by every comparison."""

    if device_id < 0:
        raise ProtocolError("device_id must be non-negative")
    if not rocm_library_paths or any(not path for path in rocm_library_paths):
        raise ProtocolError("rocm_library_paths must not be empty")
    environment = {
        "HSA_VISIBLE_DEVICES": str(device_id),
        "HIP_VISIBLE_DEVICES": str(device_id),
        "ROCR_VISIBLE_DEVICES": str(device_id),
        "LD_LIBRARY_PATH": ":".join(rocm_library_paths),
    }
    environment.update(extra or {})
    return environment


@dataclass(frozen=True)
class RuntimeCoordinate:
    """Set/unset delta that selects baseline, old mapping, or split-K."""

    variant: Literal["baseline", "old", "split"]
    env: Mapping[str, str]
    unset_env: tuple[str, ...]

    @property
    def coordinate_hash(self) -> str:
        return _canonical_sha256(
            {
                "variant": self.variant,
                "env": dict(sorted(self.env.items())),
                "unset_env": sorted(self.unset_env),
            }
        )


def q4_runtime_coordinate(
    variant: Literal["baseline", "old", "split"],
    *,
    sidecar_path: str | Path | None = None,
    device_id: int = 0,
    rocm_library_paths: Sequence[str] = (
        "/opt/rocm/core-7.14/lib",
        "/opt/rocm/lib",
    ),
) -> RuntimeCoordinate:
    base = amd_runtime_environment(
        device_id=device_id,
        rocm_library_paths=rocm_library_paths,
    )
    if variant == "baseline":
        return RuntimeCoordinate(
            variant=variant,
            env=base,
            unset_env=BASELINE_UNSET_ENVIRONMENT,
        )
    if sidecar_path is None:
        raise ProtocolError(f"{variant} coordinate requires sidecar_path")
    sidecar = Path(sidecar_path).resolve()
    base["LLAMA_Q4_RDNA_SIDECAR"] = str(sidecar)
    if variant == "old":
        base["LLAMA_Q4_RDNA_MAPPING"] = "old"
        unset = tuple(
            name
            for name in BASELINE_UNSET_ENVIRONMENT
            if name not in {"LLAMA_Q4_RDNA_SIDECAR", "LLAMA_Q4_RDNA_MAPPING"}
        )
    elif variant == "split":
        unset = tuple(
            name for name in BASELINE_UNSET_ENVIRONMENT if name != "LLAMA_Q4_RDNA_SIDECAR"
        )
    else:  # pragma: no cover - guarded by the Literal for typed callers
        raise ProtocolError(f"unsupported runtime variant: {variant}")
    return RuntimeCoordinate(variant=variant, env=base, unset_env=unset)


@dataclass(frozen=True)
class DecodeBenchmarkProtocol:
    """The exact llama.cpp live-run benchmark coordinate.

    llama-bench has a single built-in warmup toggle. ``warmup_runs`` is retained
    for task-schema compatibility: zero disables that warmup and any positive
    value enables it. The Q4 live protocol itself pins the value to one.
    """

    llama_bench_path: str
    model_path: str
    generation_tokens: tuple[int, ...] = (128, 512)
    prompt_tokens: tuple[int, ...] = (0,)
    batch_size: int = 2048
    ubatch_size: int = 512
    threads: int = 12
    repetitions: int = 3
    warmup_runs: int = 1
    gpu_layers: int = 999
    device_id: int = 0
    device_selector: str = "ROCm0"
    timeout_seconds: float = 600
    extra_args: tuple[str, ...] = ()
    cwd: str = "."
    environment: Mapping[str, str] = field(default_factory=dict)
    unset_environment: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.generation_tokens or any(value <= 0 for value in self.generation_tokens):
            raise ProtocolError("generation_tokens must contain positive values")
        if not self.prompt_tokens or any(value < 0 for value in self.prompt_tokens):
            raise ProtocolError("prompt_tokens must contain non-negative values")
        if self.batch_size < 1 or self.ubatch_size < 1:
            raise ProtocolError("batch sizes must be positive")
        if self.ubatch_size > self.batch_size:
            raise ProtocolError("ubatch_size cannot exceed batch_size")
        if self.threads < 1 or self.repetitions < 1:
            raise ProtocolError("threads and repetitions must be positive")
        if self.warmup_runs < 0:
            raise ProtocolError("warmup_runs must be non-negative")
        if self.gpu_layers < 0 or self.device_id < 0 or self.timeout_seconds <= 0:
            raise ProtocolError("GPU settings and timeout are invalid")
        if not self.device_selector:
            raise ProtocolError("device_selector must not be empty")
        if self.extra_args:
            validate_argv(self.extra_args)
        overlap = set(self.environment) & set(self.unset_environment)
        if overlap:
            raise ProtocolError(
                "protocol environment cannot set and unset the same names: "
                + ", ".join(sorted(overlap))
            )

    @property
    def argv(self) -> tuple[str, ...]:
        argv = (
            str(Path(self.llama_bench_path).resolve()),
            "-m",
            str(Path(self.model_path).resolve()),
            "-p",
            ",".join(str(value) for value in self.prompt_tokens),
            "-n",
            ",".join(str(value) for value in self.generation_tokens),
            "-b",
            str(self.batch_size),
            "-ub",
            str(self.ubatch_size),
            "-t",
            str(self.threads),
            "-r",
            str(self.repetitions),
            "-ngl",
            str(self.gpu_layers),
            "-mg",
            "0",
            "-dev",
            self.device_selector,
            "-o",
            "json",
            "-oe",
            "none",
        )
        if self.warmup_runs == 0:
            argv += ("--no-warmup",)
        return argv + tuple(self.extra_args)

    @property
    def argv_hash(self) -> str:
        return argv_sha256(self.argv)

    @property
    def protocol_hash(self) -> str:
        # Executable/model identities and runtime deltas intentionally do not
        # participate. Baseline and source-patch candidates must share this
        # semantic hash while full argv and content hashes remain independently
        # bound by RunIdentity and execution approvals.
        return _canonical_sha256(
            {
                "schema_version": 2,
                "generation_tokens": self.generation_tokens,
                "prompt_tokens": self.prompt_tokens,
                "batch_size": self.batch_size,
                "ubatch_size": self.ubatch_size,
                "threads": self.threads,
                "repetitions": self.repetitions,
                "warmup_runs": self.warmup_runs,
                "gpu_layers": self.gpu_layers,
                "visible_device": self.device_id,
                "device_selector": self.device_selector,
                "extra_args": self.extra_args,
            }
        )

    def command(self, *, name: str = "e2e") -> StageCommand:
        return StageCommand(
            name=name,
            argv=self.argv,
            cwd=self.cwd,
            env=self.environment,
            unset_env=self.unset_environment,
            timeout_seconds=self.timeout_seconds,
        )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "argv": list(self.argv),
                "argv_hash": self.argv_hash,
                "protocol_hash": self.protocol_hash,
            }
        )
        return result


@dataclass(frozen=True)
class KernelTimingProfileProtocol:
    """Short, canonical Level-2 trace coordinate.

    Kernel-timing traces can be orders of magnitude larger than the benchmark
    JSON they support.  The profiler therefore runs one warmed-up tg128
    repetition while the authoritative E2E protocol remains the independent
    tg128/tg512, three-sample comparison.
    """

    llama_bench_path: str
    model_path: str
    batch_size: int = 2048
    ubatch_size: int = 512
    threads: int = 12
    gpu_layers: int = 999
    device_id: int = 0
    device_selector: str = "ROCm0"
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.ubatch_size < 1:
            raise ProtocolError("batch sizes must be positive")
        if self.ubatch_size > self.batch_size:
            raise ProtocolError("ubatch_size cannot exceed batch_size")
        if self.threads < 1 or self.gpu_layers < 0 or self.device_id < 0:
            raise ProtocolError("profile compute coordinates are invalid")
        if not self.device_selector:
            raise ProtocolError("device_selector must not be empty")
        if self.extra_args:
            validate_argv(self.extra_args)

    @classmethod
    def from_e2e(cls, protocol: DecodeBenchmarkProtocol) -> KernelTimingProfileProtocol:
        """Copy compute coordinates without copying E2E sample breadth."""

        return cls(
            llama_bench_path=protocol.llama_bench_path,
            model_path=protocol.model_path,
            batch_size=protocol.batch_size,
            ubatch_size=protocol.ubatch_size,
            threads=protocol.threads,
            gpu_layers=protocol.gpu_layers,
            device_id=protocol.device_id,
            device_selector=protocol.device_selector,
            extra_args=protocol.extra_args,
        )

    @property
    def argv(self) -> tuple[str, ...]:
        # Absence of --no-warmup intentionally retains llama-bench's warmup.
        return (
            str(Path(self.llama_bench_path).resolve()),
            "-m",
            str(Path(self.model_path).resolve()),
            "-p",
            "0",
            "-n",
            "128",
            "-b",
            str(self.batch_size),
            "-ub",
            str(self.ubatch_size),
            "-t",
            str(self.threads),
            "-r",
            "1",
            "-ngl",
            str(self.gpu_layers),
            "-mg",
            "0",
            "-dev",
            self.device_selector,
            "-o",
            "json",
            "-oe",
            "none",
            *self.extra_args,
        )

    @property
    def command_hash(self) -> str:
        return argv_sha256(self.argv)

    @property
    def details(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "kernel-timing-short",
            "preset": "kernel-timing",
            "prompt_tokens": [0],
            "generation_tokens": [128],
            "repetitions": 1,
            "warmup": True,
            "batch_size": self.batch_size,
            "ubatch_size": self.ubatch_size,
            "threads": self.threads,
            "gpu_layers": self.gpu_layers,
            "visible_device": self.device_id,
            "device_selector": self.device_selector,
            "profile_command_hash": self.command_hash,
        }

    @property
    def protocol_hash(self) -> str:
        return _canonical_sha256(self.details)


__all__ = [
    "BASELINE_UNSET_ENVIRONMENT",
    "DecodeBenchmarkProtocol",
    "KernelTimingProfileProtocol",
    "ProtocolError",
    "Q4_RDNA_ENVIRONMENT",
    "RuntimeCoordinate",
    "SMITHY_ENVIRONMENT",
    "amd_runtime_environment",
    "q4_runtime_coordinate",
]
