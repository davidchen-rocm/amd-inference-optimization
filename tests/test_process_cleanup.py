from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from amd_inference_opt.command import CommandRunner, InvalidCommand
from amd_inference_opt.process_group import ProcessGroupError, ProcessGroupLease, read_process
from amd_inference_opt.vllm_adapter import (
    LocalProcessBackend,
    VLLMServerConflictError,
    VLLMServerStartError,
)


def _worker_script(marker: Path, *, separate_group: bool = False) -> str:
    child = (
        "import os, signal, time; from pathlib import Path; "
        + ("os.setpgrp(); " if separate_group else "")
        +
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    return (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "print('leader started', flush=True); time.sleep(60)"
    )


def _wait_for_worker(marker: Path) -> int:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if marker.exists() and marker.read_text():
            return int(marker.read_text())
        time.sleep(0.01)
    pytest.fail("worker did not become ready")


def _is_executing(pid: int) -> bool:
    identity = read_process(pid)
    return identity is not None and identity.state not in {"Z", "X"}


def _cleanup(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@pytest.mark.parametrize("separate_group", [False, True])
def test_vllm_stop_kills_worker_even_when_leader_exits_on_term(
    tmp_path: Path, separate_group: bool,
) -> None:
    marker = tmp_path / "worker.pid"
    backend = LocalProcessBackend()
    process = backend.spawn(
        [sys.executable, "-c", _worker_script(marker, separate_group=separate_group)],
        cwd=str(tmp_path), env={}, unset_env=(), stdout_path=None, stderr_path=None,
    )
    try:
        worker = _wait_for_worker(marker)
        assert backend.terminate_exact(
            pid=process.pid, boot_id=process.boot_id, start_ticks=process.start_ticks,
            timeout_seconds=0.15,
        )
        assert not _is_executing(process.pid)
        assert not _is_executing(worker)
    finally:
        _cleanup(process.pid)


def test_command_timeout_does_not_wait_forever_for_worker_pipes(tmp_path: Path) -> None:
    marker = tmp_path / "worker.pid"
    try:
        result = CommandRunner().run(
            [sys.executable, "-c", _worker_script(marker)],
            timeout_seconds=0.5,
        )
        worker = _wait_for_worker(marker)
        assert result.timed_out and not result.succeeded
        assert "leader started" in result.stdout
        assert result.spawn_error is None
        assert result.duration_seconds < 12
        assert not _is_executing(worker)
    finally:
        if marker.exists() and _is_executing(int(marker.read_text())):
            os.kill(int(marker.read_text()), signal.SIGKILL)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, OSError])
def test_command_failure_cleans_up_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException],
) -> None:
    marker = tmp_path / "worker.pid"

    def interrupt(*args: object, **kwargs: object) -> None:
        _wait_for_worker(marker)
        raise error_type("simulated capture failure")

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupt)
    try:
        if error_type is KeyboardInterrupt:
            with pytest.raises(KeyboardInterrupt):
                CommandRunner().run([sys.executable, "-c", _worker_script(marker)])
        else:
            result = CommandRunner().run([sys.executable, "-c", _worker_script(marker)])
            assert not result.succeeded
            assert "simulated capture failure" in (result.spawn_error or "")
        assert not _is_executing(int(marker.read_text()))
    finally:
        if marker.exists() and _is_executing(int(marker.read_text())):
            os.kill(int(marker.read_text()), signal.SIGKILL)


def test_session_lease_rejects_reused_pid_without_signaling(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               start_new_session=True)
    try:
        identity = read_process(process.pid)
        assert identity is not None
        with pytest.raises(ProcessGroupError, match="identity"):
            ProcessGroupLease(process.pid, identity.start_ticks + 1)
        assert _is_executing(process.pid)
    finally:
        _cleanup(process.pid)
        process.wait(timeout=5)


def test_startup_identity_failure_rolls_back_workers(tmp_path: Path) -> None:
    marker = tmp_path / "worker.pid"

    class FailingBackend(LocalProcessBackend):
        def start_ticks(self, pid: int) -> int | None:
            _wait_for_worker(marker)
            raise RuntimeError("identity capture failed")

    try:
        with pytest.raises(RuntimeError, match="identity capture failed"):
            FailingBackend().spawn(
                [sys.executable, "-c", _worker_script(marker)],
                cwd=str(tmp_path), env={}, unset_env=(), stdout_path=None, stderr_path=None,
            )
        assert not _is_executing(int(marker.read_text()))
    finally:
        if marker.exists() and _is_executing(int(marker.read_text())):
            os.kill(int(marker.read_text()), signal.SIGKILL)


def test_failure_before_lease_capture_stops_owned_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amd_inference_opt import vllm_adapter

    marker = tmp_path / "worker.pid"

    def unavailable(*args: object) -> None:
        _wait_for_worker(marker)
        return None

    monkeypatch.setattr(vllm_adapter, "_proc_start_ticks", unavailable)
    try:
        with pytest.raises(VLLMServerStartError, match="capture the spawned session"):
            LocalProcessBackend().spawn(
                [sys.executable, "-c", _worker_script(marker)],
                cwd=str(tmp_path), env={}, unset_env=(), stdout_path=None, stderr_path=None,
            )
        assert not _is_executing(int(marker.read_text()))
    finally:
        if marker.exists() and _is_executing(int(marker.read_text())):
            os.kill(int(marker.read_text()), signal.SIGKILL)


def test_normal_exit_cannot_claim_success_with_background_worker(tmp_path: Path) -> None:
    marker = tmp_path / "worker.pid"
    child = (
        "import os,time; from pathlib import Path; "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "time.sleep(0.3)"
    )
    try:
        result = CommandRunner().run([sys.executable, "-c", parent], timeout_seconds=5)
        assert not result.succeeded
        assert "cleanup unverified" in (result.spawn_error or "")
    finally:
        if marker.exists() and _is_executing(int(marker.read_text())):
            os.kill(int(marker.read_text()), signal.SIGKILL)


def test_new_server_cannot_replace_an_unverified_stale_session(tmp_path: Path) -> None:
    from test_vllm_adapter import _adapter, _spec

    adapter, processes, _, _ = _adapter(tmp_path)
    spec = _spec(tmp_path)
    adapter.start_or_resume(spec)
    processes.alive.clear()
    with pytest.raises(VLLMServerConflictError, match="session cleanup is unverified"):
        adapter.start_or_resume(spec)
    assert len(processes.spawn_calls) == 1


@pytest.mark.parametrize("timeout", [float("inf"), float("nan"), 0, -1])
def test_invalid_timeout_fails_before_spawn(timeout: float) -> None:
    with pytest.raises(InvalidCommand, match="finite"):
        CommandRunner().run([sys.executable, "-c", "raise AssertionError"],
                            timeout_seconds=timeout)
