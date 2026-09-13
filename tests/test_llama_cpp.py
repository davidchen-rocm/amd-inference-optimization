from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from amd_inference_opt.command import CommandResult
from amd_inference_opt.llama_cpp import (
    LlamaCppAdapter,
    LlamaCppError,
    parse_llama_bench_json,
    sha256_file,
)


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_parses_decode_records_and_preserves_raw_fields() -> None:
    payload = [
        {
            "build_commit": "a7a6d0d",
            "n_prompt": 0,
            "n_gen": 128,
            "avg_ts": 100.0,
            "stddev_ts": 2.0,
            "samples_ts": [98.0, 100.0, 102.0],
            "future_field": {"kept": True},
        },
        {
            "n_prompt": 0,
            "n_gen": 512,
            "avg_ts": 96.0,
            "stddev_ts": 0.0,
            "samples_ts": [96.0, 96.0, 96.0],
        },
    ]

    result = parse_llama_bench_json(json.dumps(payload))

    assert set(result.by_test_id()) == {"tg128", "tg512"}
    assert result.by_test_id()["tg128"].phase == "decode"
    assert result.by_test_id()["tg128"].coefficient_of_variation == pytest.approx(0.02)
    assert result.raw[0]["future_field"] == {"kept": True}


def test_parser_rejects_missing_samples_or_invalid_numbers() -> None:
    with pytest.raises(LlamaCppError, match="samples_ts"):
        parse_llama_bench_json({"n_prompt": 0, "n_gen": 1, "avg_ts": 2})
    with pytest.raises(LlamaCppError, match="positive"):
        parse_llama_bench_json(
            {"n_prompt": 0, "n_gen": 1, "avg_ts": 0, "samples_ts": [2]}
        )


def test_builds_decode_only_llama_bench_argv(tmp_path: Path) -> None:
    argv = LlamaCppAdapter.benchmark_argv(
        tmp_path / "llama-bench",
        tmp_path / "model.gguf",
        generation_tokens=(128, 512),
        prompt_tokens=(0,),
        repetitions=3,
        warmup=False,
    )

    assert argv[0] == str((tmp_path / "llama-bench").resolve())
    assert argv[argv.index("-n") + 1] == "128,512"
    assert argv[argv.index("-p") + 1] == "0"
    assert argv[argv.index("-o") + 1] == "json"
    assert "--no-warmup" in argv


def test_inspects_git_commit_dirty_state_and_model_hash(tmp_path: Path) -> None:
    source = tmp_path / "llama.cpp"
    source.mkdir()
    (source / "CMakeLists.txt").write_text("project(llama)\n", encoding="utf-8")
    _run_git(source, "init")
    _run_git(source, "config", "user.email", "tests@example.invalid")
    _run_git(source, "config", "user.name", "Tests")
    _run_git(source, "add", "CMakeLists.txt")
    _run_git(source, "commit", "-m", "initial")
    commit = _run_git(source, "rev-parse", "HEAD")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-test")

    inspection = LlamaCppAdapter().inspect_target(
        source,
        model,
        expected_commit=commit[:8],
        expected_model_sha256=sha256_file(model),
    )

    assert inspection.commit == commit
    assert not inspection.dirty
    assert inspection.expected_commit_matches is True
    assert inspection.expected_model_sha256_matches is True


class _BenchmarkCommandRunner:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.argv: tuple[str, ...] | None = None

    def run(self, argv: object, **_: object) -> CommandResult:
        self.argv = tuple(argv)  # type: ignore[arg-type]
        return CommandResult(
            argv=self.argv,
            cwd="/tmp",
            started_at="now",
            duration_seconds=1,
            exit_code=0,
            stdout=self.stdout,
            stderr="",
        )


def test_run_benchmark_parses_only_successful_stdout(tmp_path: Path) -> None:
    raw = json.dumps(
        [{"n_prompt": 0, "n_gen": 8, "avg_ts": 4.0, "samples_ts": [4.0]}]
    )
    commands = _BenchmarkCommandRunner(raw)
    adapter = LlamaCppAdapter(commands)  # type: ignore[arg-type]

    run = adapter.run_benchmark(
        tmp_path / "llama-bench",
        tmp_path / "model.gguf",
        generation_tokens=(8,),
        cwd=tmp_path,
    )

    assert run.succeeded
    assert run.benchmark is not None
    assert run.benchmark.records[0].test_id == "tg8"

