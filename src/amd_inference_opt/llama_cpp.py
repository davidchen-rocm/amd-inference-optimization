"""llama.cpp target, build, and benchmark adapter for the MVP."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .command import CommandResult, CommandRunner, validate_argv
from .protocol import DecodeBenchmarkProtocol


class LlamaCppError(RuntimeError):
    """Raised when target inspection or llama-bench output is invalid."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LlamaTargetInspection:
    source_dir: str
    repository_root: str
    commit: str
    dirty: bool
    model_path: str
    model_size_bytes: int
    model_sha256: str
    expected_commit_matches: bool | None
    expected_model_sha256_matches: bool | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaBuildResult:
    build_dir: str
    configure: CommandResult
    build: CommandResult | None
    llama_bench_path: str
    llama_cli_path: str

    @property
    def succeeded(self) -> bool:
        return self.configure.succeeded and self.build is not None and self.build.succeeded

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaBenchRecord:
    phase: str
    prompt_tokens: int
    generation_tokens: int
    mean_tokens_per_second: float
    stddev_tokens_per_second: float
    samples_tokens_per_second: tuple[float, ...]
    coefficient_of_variation: float
    raw: dict[str, Any]

    @property
    def sample_count(self) -> int:
        return len(self.samples_tokens_per_second)

    @property
    def test_id(self) -> str:
        if self.phase == "decode":
            return f"tg{self.generation_tokens}"
        if self.phase == "prefill":
            return f"pp{self.prompt_tokens}"
        return f"pp{self.prompt_tokens}+tg{self.generation_tokens}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaBenchmarkResult:
    records: tuple[LlamaBenchRecord, ...]
    raw: tuple[dict[str, Any], ...]

    def by_test_id(self) -> dict[str, LlamaBenchRecord]:
        return {record.test_id: record for record in self.records}

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaBenchmarkRun:
    command: CommandResult
    benchmark: LlamaBenchmarkResult | None

    @property
    def succeeded(self) -> bool:
        return self.command.succeeded and self.benchmark is not None


@dataclass(frozen=True)
class LlamaProtocolRun:
    protocol: dict[str, object]
    binary_sha256: str
    command: CommandResult
    benchmark: LlamaBenchmarkResult | None

    @property
    def succeeded(self) -> bool:
        return self.command.succeeded and self.benchmark is not None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _number(value: object, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LlamaCppError(f"llama-bench field {field!r} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result < 0 if nonnegative else result <= 0):
        condition = "non-negative" if nonnegative else "positive"
        raise LlamaCppError(f"llama-bench field {field!r} must be finite and {condition}")
    return result


def parse_llama_bench_json(
    payload: str | bytes | Path | list[object] | dict[str, object],
) -> LlamaBenchmarkResult:
    """Parse llama-bench JSON without discarding any source fields."""

    if isinstance(payload, Path):
        decoded: object = json.loads(payload.read_text(encoding="utf-8"))
    elif isinstance(payload, bytes):
        decoded = json.loads(payload.decode("utf-8"))
    elif isinstance(payload, str):
        decoded = json.loads(payload)
    else:
        decoded = payload
    if isinstance(decoded, dict):
        rows: list[object] = [decoded]
    elif isinstance(decoded, list):
        rows = decoded
    else:
        raise LlamaCppError("llama-bench JSON must be an object or array")
    if not rows:
        raise LlamaCppError("llama-bench JSON contains no benchmark rows")

    raw_rows: list[dict[str, Any]] = []
    records: list[LlamaBenchRecord] = []
    for index, untyped in enumerate(rows):
        if not isinstance(untyped, dict):
            raise LlamaCppError(f"llama-bench row {index} must be an object")
        row = dict(untyped)
        try:
            n_prompt = int(row.get("n_prompt", 0))
            n_gen = int(row.get("n_gen", 0))
        except (TypeError, ValueError) as error:
            raise LlamaCppError(f"llama-bench row {index} has invalid token counts") from error
        if n_prompt < 0 or n_gen < 0 or (n_prompt == 0 and n_gen == 0):
            raise LlamaCppError(f"llama-bench row {index} has no measured tokens")
        mean = _number(row.get("avg_ts"), "avg_ts")

        raw_samples = row.get("samples_ts")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise LlamaCppError(f"llama-bench row {index} has no samples_ts")
        samples = tuple(_number(item, "samples_ts") for item in raw_samples)
        if "stddev_ts" in row:
            stddev = _number(row["stddev_ts"], "stddev_ts", nonnegative=True)
        else:
            stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
        phase = "decode" if n_prompt == 0 else "prefill" if n_gen == 0 else "mixed"
        raw_rows.append(row)
        records.append(
            LlamaBenchRecord(
                phase=phase,
                prompt_tokens=n_prompt,
                generation_tokens=n_gen,
                mean_tokens_per_second=mean,
                stddev_tokens_per_second=stddev,
                samples_tokens_per_second=samples,
                coefficient_of_variation=stddev / mean,
                raw=row,
            )
        )
    return LlamaBenchmarkResult(records=tuple(records), raw=tuple(raw_rows))


class LlamaCppAdapter:
    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self.commands = command_runner or CommandRunner()

    def inspect_target(
        self,
        source_dir: str | Path,
        model_path: str | Path,
        *,
        expected_commit: str | None = None,
        expected_model_sha256: str | None = None,
    ) -> LlamaTargetInspection:
        source = Path(source_dir).resolve()
        model = Path(model_path).resolve()
        if not source.is_dir() or not (source / "CMakeLists.txt").is_file():
            raise LlamaCppError(f"not a llama.cpp source directory: {source}")
        if not model.is_file():
            raise LlamaCppError(f"model file does not exist: {model}")
        if model.suffix.lower() != ".gguf":
            raise LlamaCppError(f"MVP requires a GGUF model: {model}")

        root_result = self.commands.run(["git", "-C", str(source), "rev-parse", "--show-toplevel"])
        commit_result = self.commands.run(["git", "-C", str(source), "rev-parse", "HEAD"])
        status_result = self.commands.run(["git", "-C", str(source), "status", "--porcelain"])
        for result in (root_result, commit_result, status_result):
            if not result.succeeded:
                message = result.stderr.strip() or "failed to inspect llama.cpp repository"
                raise LlamaCppError(message)
        root = root_result.stdout.strip()
        commit = commit_result.stdout.strip()
        model_digest = sha256_file(model)
        return LlamaTargetInspection(
            source_dir=str(source),
            repository_root=root,
            commit=commit,
            dirty=bool(status_result.stdout.strip()),
            model_path=str(model),
            model_size_bytes=model.stat().st_size,
            model_sha256=model_digest,
            expected_commit_matches=(
                commit.startswith(expected_commit) if expected_commit else None
            ),
            expected_model_sha256_matches=(
                model_digest.lower() == expected_model_sha256.lower()
                if expected_model_sha256
                else None
            ),
        )

    def build(
        self,
        source_dir: str | Path,
        build_dir: str | Path,
        *,
        cmake_flags: Sequence[str],
        jobs: int = 1,
        targets: Sequence[str] = ("llama-bench", "llama-cli"),
        env: Mapping[str, str] | None = None,
        configure_timeout_seconds: float = 600,
        build_timeout_seconds: float = 3600,
        artifact_dir: str | Path | None = None,
    ) -> LlamaBuildResult:
        source = Path(source_dir).resolve()
        build = Path(build_dir).resolve()
        if not source.is_dir():
            raise LlamaCppError(f"source directory does not exist: {source}")
        if jobs < 1:
            raise LlamaCppError("jobs must be at least one")
        frozen_flags = validate_argv(cmake_flags) if cmake_flags else ()
        frozen_targets = validate_argv(targets) if targets else ()
        build.parent.mkdir(parents=True, exist_ok=True)
        logs = Path(artifact_dir).resolve() if artifact_dir else None

        configure = self.commands.run(
            ["cmake", "-S", str(source), "-B", str(build), *frozen_flags],
            cwd=source,
            env=env,
            timeout_seconds=configure_timeout_seconds,
            stdout_path=logs / "configure.stdout" if logs else None,
            stderr_path=logs / "configure.stderr" if logs else None,
        )
        build_result: CommandResult | None = None
        if configure.succeeded:
            argv = ["cmake", "--build", str(build), "--parallel", str(jobs)]
            if frozen_targets:
                argv.extend(["--target", *frozen_targets])
            build_result = self.commands.run(
                argv,
                cwd=source,
                env=env,
                timeout_seconds=build_timeout_seconds,
                stdout_path=logs / "build.stdout" if logs else None,
                stderr_path=logs / "build.stderr" if logs else None,
            )
        return LlamaBuildResult(
            build_dir=str(build),
            configure=configure,
            build=build_result,
            llama_bench_path=str(build / "bin" / "llama-bench"),
            llama_cli_path=str(build / "bin" / "llama-cli"),
        )

    @staticmethod
    def benchmark_argv(
        llama_bench_path: str | Path,
        model_path: str | Path,
        *,
        generation_tokens: Sequence[int] = (128,),
        prompt_tokens: Sequence[int] = (0,),
        repetitions: int = 3,
        gpu_layers: int = 999,
        batch_size: int | None = None,
        ubatch_size: int | None = None,
        threads: int | None = None,
        main_gpu: int | None = None,
        warmup: bool = True,
        extra_args: Sequence[str] = (),
    ) -> tuple[str, ...]:
        if repetitions < 1:
            raise LlamaCppError("repetitions must be at least one")
        if batch_size is not None and batch_size < 1:
            raise LlamaCppError("batch_size must be positive")
        if ubatch_size is not None and ubatch_size < 1:
            raise LlamaCppError("ubatch_size must be positive")
        if batch_size is not None and ubatch_size is not None and ubatch_size > batch_size:
            raise LlamaCppError("ubatch_size cannot exceed batch_size")
        if threads is not None and threads < 1:
            raise LlamaCppError("threads must be positive")
        if main_gpu is not None and main_gpu < 0:
            raise LlamaCppError("main_gpu must be non-negative")
        if not generation_tokens or any(value < 0 for value in generation_tokens):
            raise LlamaCppError("generation_tokens must be a non-empty non-negative list")
        if not prompt_tokens or any(value < 0 for value in prompt_tokens):
            raise LlamaCppError("prompt_tokens must be a non-empty non-negative list")
        no_generation = all(value == 0 for value in generation_tokens)
        no_prompt = all(value == 0 for value in prompt_tokens)
        if no_generation and no_prompt:
            raise LlamaCppError("benchmark must measure prompt or generation tokens")
        extras = validate_argv(extra_args) if extra_args else ()
        argv = [
            str(Path(llama_bench_path).resolve()),
            "-m",
            str(Path(model_path).resolve()),
            "-p",
            ",".join(str(value) for value in prompt_tokens),
            "-n",
            ",".join(str(value) for value in generation_tokens),
            "-r",
            str(repetitions),
            "-ngl",
            str(gpu_layers),
            "-o",
            "json",
            "-oe",
            "none",
        ]
        if batch_size is not None:
            argv.extend(["-b", str(batch_size)])
        if ubatch_size is not None:
            argv.extend(["-ub", str(ubatch_size)])
        if threads is not None:
            argv.extend(["-t", str(threads)])
        if main_gpu is not None:
            argv.extend(["-mg", str(main_gpu)])
        if not warmup:
            argv.append("--no-warmup")
        argv.extend(extras)
        return tuple(argv)

    def run_benchmark(
        self,
        llama_bench_path: str | Path,
        model_path: str | Path,
        *,
        generation_tokens: Sequence[int] = (128,),
        prompt_tokens: Sequence[int] = (0,),
        repetitions: int = 3,
        gpu_layers: int = 999,
        batch_size: int | None = None,
        ubatch_size: int | None = None,
        threads: int | None = None,
        main_gpu: int | None = None,
        warmup: bool = True,
        extra_args: Sequence[str] = (),
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        unset_env: Sequence[str] = (),
        timeout_seconds: float = 1800,
        artifact_dir: str | Path | None = None,
    ) -> LlamaBenchmarkRun:
        argv = self.benchmark_argv(
            llama_bench_path,
            model_path,
            generation_tokens=generation_tokens,
            prompt_tokens=prompt_tokens,
            repetitions=repetitions,
            gpu_layers=gpu_layers,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            threads=threads,
            main_gpu=main_gpu,
            warmup=warmup,
            extra_args=extra_args,
        )
        logs = Path(artifact_dir).resolve() if artifact_dir else None
        command = self.commands.run(
            argv,
            cwd=cwd or Path(llama_bench_path).resolve().parent,
            env=env,
            unset_env=unset_env,
            timeout_seconds=timeout_seconds,
            stdout_path=logs / "llama-bench.json" if logs else None,
            stderr_path=logs / "llama-bench.stderr" if logs else None,
        )
        benchmark = parse_llama_bench_json(command.stdout) if command.succeeded else None
        return LlamaBenchmarkRun(command=command, benchmark=benchmark)

    def run_protocol(
        self,
        protocol: DecodeBenchmarkProtocol,
        *,
        artifact_dir: str | Path | None = None,
    ) -> LlamaProtocolRun:
        """Run a frozen decode protocol and bind results to the exact binary hash."""

        binary = Path(protocol.llama_bench_path).resolve()
        if not binary.is_file():
            raise LlamaCppError(f"llama-bench binary does not exist: {binary}")
        command_spec = protocol.command()
        logs = Path(artifact_dir).resolve() if artifact_dir else None
        command = self.commands.run(
            command_spec.argv,
            cwd=command_spec.cwd,
            env=command_spec.env,
            unset_env=command_spec.unset_env,
            timeout_seconds=command_spec.timeout_seconds,
            stdout_path=logs / "llama-bench.json" if logs else None,
            stderr_path=logs / "llama-bench.stderr" if logs else None,
        )
        benchmark: LlamaBenchmarkResult | None = None
        if command.succeeded:
            try:
                benchmark = parse_llama_bench_json(command.stdout)
            except (json.JSONDecodeError, LlamaCppError):
                benchmark = None
        return LlamaProtocolRun(
            protocol=protocol.to_dict(),
            binary_sha256=sha256_file(binary),
            command=command,
            benchmark=benchmark,
        )
