"""Small, auditable subprocess adapter.

The optimization workflow intentionally deals in argv arrays.  It never accepts a
shell command string, and this module is the only place that starts ordinary
build/benchmark processes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .process_group import (
    ProcessGroupError,
    ProcessGroupLease,
    read_process,
    terminate_uncaptured_child,
)


class InvalidCommand(ValueError):
    """Raised before execution when a command is not a safe argv array."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze an argv sequence.

    A plain string is rejected even though it is technically a Sequence; accepting
    it is a common source of accidental shell-like APIs.
    """

    if isinstance(argv, (str, bytes)):
        raise InvalidCommand("command must be an argv sequence, not a string")
    frozen = tuple(argv)
    if not frozen:
        raise InvalidCommand("argv must not be empty")
    for index, item in enumerate(frozen):
        if not isinstance(item, str):
            raise InvalidCommand(f"argv[{index}] must be a string")
        if not item:
            raise InvalidCommand(f"argv[{index}] must not be empty")
        if "\x00" in item:
            raise InvalidCommand(f"argv[{index}] contains a NUL byte")
    return frozen


def argv_sha256(argv: Sequence[str]) -> str:
    """Hash only argv so baseline and candidate command identity can be compared."""

    return _canonical_sha256(list(validate_argv(argv)))


def command_request_sha256(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str] | None = None,
    unset_env: Sequence[str] = (),
    timeout_seconds: float | None = None,
) -> str:
    """Hash the complete declared command request, including its environment delta."""

    return _canonical_sha256(
        {
            "argv": list(validate_argv(argv)),
            "cwd": str(Path(cwd).resolve()),
            "env": dict(sorted((env or {}).items())),
            "unset_env": sorted(unset_env),
            "timeout_seconds": timeout_seconds,
        }
    )


@dataclass(frozen=True)
class StageCommand:
    """Typed command declaration for one build, benchmark, or quality stage."""

    name: str
    argv: tuple[str, ...]
    cwd: str = "."
    env: Mapping[str, str] = field(default_factory=dict)
    unset_env: tuple[str, ...] = ()
    timeout_seconds: float = 1800

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name or self.name in {".", ".."}:
            raise InvalidCommand(f"invalid stage command name: {self.name!r}")
        object.__setattr__(self, "argv", validate_argv(self.argv))
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise InvalidCommand("timeout_seconds must be finite and greater than zero")
        _validated_environment(self.env, self.unset_env)

    def resolved_cwd(self, root: str | Path) -> Path:
        path = Path(self.cwd)
        return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()

    @property
    def argv_sha256(self) -> str:
        return argv_sha256(self.argv)

    def request_sha256(self, root: str | Path) -> str:
        return command_request_sha256(
            self.argv,
            cwd=self.resolved_cwd(root),
            env=self.env,
            unset_env=self.unset_env,
            timeout_seconds=self.timeout_seconds,
        )


def _validated_environment(
    env: Mapping[str, str] | None,
    unset_env: Sequence[str],
) -> None:
    for key in unset_env:
        if not isinstance(key, str) or not key or "\x00" in key or "=" in key:
            raise InvalidCommand(f"invalid environment key to unset: {key!r}")
    if env:
        for key, value in env.items():
            if not isinstance(key, str) or not key or "\x00" in key or "=" in key:
                raise InvalidCommand(f"invalid environment key: {key!r}")
            if not isinstance(value, str) or "\x00" in value:
                raise InvalidCommand(f"invalid environment value for {key!r}")
    overlap = set(env or {}) & set(unset_env)
    if overlap:
        raise InvalidCommand(
            "environment variables cannot be both set and unset: "
            + ", ".join(sorted(overlap))
        )


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    cwd: str
    started_at: str
    duration_seconds: float
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    spawn_error: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    unset_environment: tuple[str, ...] = ()
    timeout_seconds: float | None = None
    argv_sha256: str | None = None
    request_sha256: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.spawn_error is None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CommandRunner:
    """Run an argv command without invoking a shell and capture all evidence."""

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
    ) -> CommandResult:
        frozen = validate_argv(argv)
        selected_cwd = Path(cwd or Path.cwd()).resolve()
        if not selected_cwd.is_dir():
            raise InvalidCommand(f"cwd is not a directory: {selected_cwd}")
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise InvalidCommand("timeout_seconds must be finite and greater than zero")

        _validated_environment(env, unset_env)
        selected_env = dict(env or {})
        selected_unset = tuple(unset_env)
        process_env = os.environ.copy()
        for key in selected_unset:
            process_env.pop(key, None)
        process_env.update(selected_env)

        started_at = datetime.now(UTC).isoformat()
        start = time.monotonic()
        stdout = ""
        stderr = ""
        exit_code: int | None = None
        timed_out = False
        spawn_error: str | None = None
        process: subprocess.Popen[str] | None = None
        start_ticks: int | None = None
        lease: ProcessGroupLease | None = None
        try:
            process = subprocess.Popen(
                list(frozen),
                cwd=selected_cwd,
                env=process_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                start_new_session=True,
            )
            identity = read_process(process.pid)
            start_ticks = identity.start_ticks if identity is not None else None
            if start_ticks is not None:
                lease = ProcessGroupLease(process.pid, start_ticks)
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                exit_code = process.returncode
                if lease is not None:
                    try:
                        if lease.has_live_members():
                            spawn_error = "command exited with running descendants"
                            lease.terminate(grace_seconds=5)
                    except ProcessGroupError as cleanup_error:
                        spawn_error = f"command descendant cleanup unverified: {cleanup_error}"
            except subprocess.TimeoutExpired as error:
                timed_out = True
                try:
                    self._terminate_process_group(process, start_ticks=start_ticks, lease=lease)
                except ProcessGroupError as cleanup_error:
                    spawn_error = str(cleanup_error)
                try:
                    final_stdout, final_stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired as drain_error:
                    # A worker that escaped the session may retain the pipes.
                    # Preserve partial evidence instead of hanging indefinitely.
                    final_stdout = _as_text(drain_error.stdout)
                    final_stderr = _as_text(drain_error.stderr)
                    spawn_error = spawn_error or (
                        "descendant output pipes remained open after cleanup"
                    )
                stdout = final_stdout or _as_text(error.stdout)
                stderr = final_stderr or _as_text(error.stderr)
                if stderr and not stderr.endswith("\n"):
                    stderr += "\n"
                stderr += f"command timed out after {timeout_seconds} seconds\n"
        except subprocess.TimeoutExpired as error:
            # Kept as a defensive fallback for alternative Popen implementations.
            timed_out = True
            stdout = _as_text(error.stdout)
            stderr = _as_text(error.stderr)
            if stderr and not stderr.endswith("\n"):
                stderr += "\n"
            stderr += f"command timed out after {timeout_seconds} seconds\n"
        except OSError as error:
            spawn_error = f"{type(error).__name__}: {error}"
            stderr = spawn_error + "\n"
            if process is not None:
                try:
                    self._terminate_process_group(process, start_ticks=start_ticks, lease=lease)
                except (ProcessGroupError, OSError, subprocess.TimeoutExpired) as cleanup_error:
                    stderr += f"managed command cleanup failed: {cleanup_error}\n"
        except BaseException as error:
            if process is not None:
                try:
                    self._terminate_process_group(process, start_ticks=start_ticks, lease=lease)
                except (ProcessGroupError, OSError, subprocess.TimeoutExpired) as cleanup_error:
                    error.add_note(f"managed command cleanup failed: {cleanup_error}")
            raise
        finally:
            if process is not None:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

        result = CommandResult(
            argv=frozen,
            cwd=str(selected_cwd),
            started_at=started_at,
            duration_seconds=time.monotonic() - start,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            spawn_error=spawn_error,
            environment=selected_env,
            unset_environment=selected_unset,
            timeout_seconds=timeout_seconds,
            argv_sha256=argv_sha256(frozen),
            request_sha256=command_request_sha256(
                frozen,
                cwd=selected_cwd,
                env=selected_env,
                unset_env=selected_unset,
                timeout_seconds=timeout_seconds,
            ),
        )
        self._write_capture(stdout_path, result.stdout)
        self._write_capture(stderr_path, result.stderr)
        return result

    @staticmethod
    def _terminate_process_group(
        process: subprocess.Popen[str], *, start_ticks: int | None = None,
        lease: ProcessGroupLease | None = None,
    ) -> None:
        if lease is None:
            terminate_uncaptured_child(process)
            return
        lease.terminate(grace_seconds=5)
        process.wait(timeout=5)

    @staticmethod
    def _write_capture(path: str | Path | None, content: str) -> None:
        if path is None:
            return
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
