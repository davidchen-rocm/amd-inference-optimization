from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from amd_inference_opt.command import CommandRunner, InvalidCommand, validate_argv


def test_rejects_command_string_and_invalid_argv() -> None:
    with pytest.raises(InvalidCommand, match="argv sequence"):
        validate_argv("echo unsafe")
    with pytest.raises(InvalidCommand, match="empty"):
        validate_argv([])
    with pytest.raises(InvalidCommand, match="NUL"):
        validate_argv(["echo", "bad\x00value"])


def test_metacharacters_are_passed_as_literal_argv(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    literal = f"$(touch {marker})"
    result = CommandRunner().run(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", literal],
        cwd=tmp_path,
    )

    assert result.succeeded
    assert result.stdout.strip() == literal
    assert not marker.exists()


def test_captures_output_environment_and_exit_code(tmp_path: Path) -> None:
    stdout_path = tmp_path / "logs" / "out.txt"
    stderr_path = tmp_path / "logs" / "err.txt"
    script = (
        "import os,sys; print(os.environ['GPUOPT_TEST']); "
        "print('bad', file=sys.stderr); sys.exit(7)"
    )
    result = CommandRunner().run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={"GPUOPT_TEST": "present"},
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    assert not result.succeeded
    assert result.exit_code == 7
    assert stdout_path.read_text(encoding="utf-8") == "present\n"
    assert stderr_path.read_text(encoding="utf-8") == "bad\n"


def test_timeout_is_structured_result(tmp_path: Path) -> None:
    result = CommandRunner().run(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=tmp_path,
        timeout_seconds=0.05,
    )

    assert result.timed_out
    assert result.exit_code is None
    assert not result.succeeded
    assert "timed out" in result.stderr


def test_command_runner_can_remove_inherited_environment(tmp_path: Path) -> None:
    key = "GPUOPT_TEST_INHERITED"
    os.environ[key] = "old"
    try:
        result = CommandRunner().run(
            [sys.executable, "-c", f"import os; print(os.getenv('{key}', 'unset'))"],
            cwd=tmp_path,
            unset_env=[key],
        )
    finally:
        os.environ.pop(key, None)

    assert result.succeeded
    assert result.stdout == "unset\n"
