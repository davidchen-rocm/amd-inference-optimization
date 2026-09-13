"""Durable, single-owner execution for long-running quality evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .command import StageCommand, command_request_sha256
from .models import (
    ProcessIdentity,
    QualityExecution,
    QualityExecutionSpec,
    QualityExecutionState,
)
from .store import ExperimentStore, StoreError

TERMINAL_QUALITY_STATES = frozenset(
    {
        QualityExecutionState.COMPLETED,
        QualityExecutionState.FAILED,
        QualityExecutionState.TIMED_OUT,
        QualityExecutionState.ORPHANED,
    }
)


class QualityExecutionError(RuntimeError):
    """A durable quality attempt cannot safely be started or resumed."""


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:  # pragma: no cover - Linux is required by ROCm
        raise QualityExecutionError("cannot read Linux boot_id") from exc


def process_identity(pid: int) -> ProcessIdentity:
    """Read a PID plus Linux boot/start identity, rejecting a vanished process."""

    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        # comm is parenthesized and may contain spaces or ')'; fields after its
        # final ')' begin at proc stat field 3. starttime is field 22.
        fields_after_comm = stat_line[stat_line.rfind(")") + 2 :].split()
        start_ticks = int(fields_after_comm[19])
    except (OSError, ValueError, IndexError) as exc:
        raise QualityExecutionError(f"process {pid} is not inspectable") from exc
    return ProcessIdentity(pid=pid, boot_id=_boot_id(), start_ticks=start_ticks)


def process_matches(identity: ProcessIdentity | None) -> bool:
    if identity is None:
        return False
    try:
        current = process_identity(identity.pid)
        stat_line = Path(f"/proc/{identity.pid}/stat").read_text(encoding="ascii")
        state = stat_line[stat_line.rfind(")") + 2 :].split()[0]
        return current == identity and state != "Z"
    except (QualityExecutionError, OSError, IndexError):
        return False


def _terminate_process_group(identity: ProcessIdentity | None) -> None:
    """Stop only the process group whose leader still has the recorded identity."""

    if not process_matches(identity):
        return
    assert identity is not None
    try:
        os.killpg(identity.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not process_matches(identity):
            return
        time.sleep(0.05)
    if process_matches(identity):
        try:
            os.killpg(identity.pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def _attempt_path(experiment_id: str, attempt_id: str) -> str:
    return f"experiments/{experiment_id}/quality/attempts/{attempt_id}.json"


def _pointer_path(experiment_id: str) -> str:
    return f"experiments/{experiment_id}/quality/active-execution.json"


def _save_execution(store: ExperimentStore, execution: QualityExecution) -> None:
    store.save_json(
        execution.task_id,
        _attempt_path(execution.experiment_id, execution.attempt_id),
        execution,
        producer="quality-execution",
    )


def _updated(
    execution: QualityExecution,
    **changes: object,
) -> QualityExecution:
    changes.setdefault("updated_at", datetime.now(UTC))
    payload = execution.model_dump(mode="python")
    payload.update(changes)
    return QualityExecution.model_validate(payload)


def finalize_quality_execution(
    store: ExperimentStore,
    task_id: str,
    experiment_id: str,
    attempt_id: str,
    state: Literal[
        QualityExecutionState.COMPLETED,
        QualityExecutionState.FAILED,
        QualityExecutionState.TIMED_OUT,
        QualityExecutionState.ORPHANED,
    ],
    *,
    exit_code: int | None = None,
    failure_reason: str | None = None,
    expected_worker: ProcessIdentity | None = None,
    bind_artifacts: bool = False,
) -> QualityExecution:
    """Finalize once; repeated terminal calls return the original terminal record."""

    if state not in TERMINAL_QUALITY_STATES:
        raise QualityExecutionError(f"cannot finalize quality execution as {state}")
    with store.task_lock(task_id):
        path = _attempt_path(experiment_id, attempt_id)
        execution = store.load_json(task_id, path, QualityExecution)
        if execution.state in TERMINAL_QUALITY_STATES:
            return execution
        if expected_worker is not None and execution.worker != expected_worker:
            raise QualityExecutionError("quality worker identity changed before finalize")
        output_artifact = execution.output_artifact
        stdout_artifact = execution.stdout_artifact
        stderr_artifact = execution.stderr_artifact
        if bind_artifacts:
            stdout_artifact = store.register_existing_artifact(
                task_id,
                execution.spec.stdout_path,
                producer="quality-evaluator",
                media_type="text/plain",
            )
            stderr_artifact = store.register_existing_artifact(
                task_id,
                execution.spec.stderr_path,
                producer="quality-evaluator",
                media_type="text/plain",
            )
            output_path = store.task_dir(task_id) / execution.spec.output_path
            if output_path.is_file():
                output_artifact = store.register_existing_artifact(
                    task_id,
                    execution.spec.output_path,
                    producer="quality-evaluator",
                    media_type="application/json",
                )
        execution = _updated(
            execution,
            state=state,
            exit_code=exit_code,
            failure_reason=failure_reason,
            terminal_at=datetime.now(UTC),
            heartbeat_at=datetime.now(UTC),
            output_artifact=output_artifact,
            stdout_artifact=stdout_artifact,
            stderr_artifact=stderr_artifact,
        )
        _save_execution(store, execution)
        return execution


class QualityExecutionManager:
    """Atomically start or reuse the one exact active quality request."""

    def __init__(
        self,
        store: ExperimentStore,
        *,
        launcher: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self.store = store
        self.launcher = launcher

    def start_or_resume(
        self,
        task_id: str,
        experiment_id: str,
        command: StageCommand,
        *,
        output_path: str,
        spec_coordinates: Mapping[str, object],
        retry_orphaned: bool = False,
        retry_failed: bool = False,
    ) -> QualityExecution:
        cwd = command.resolved_cwd(command.cwd)
        request_hash = command_request_sha256(
            command.argv,
            cwd=cwd,
            env=command.env,
            unset_env=command.unset_env,
            timeout_seconds=command.timeout_seconds,
        )
        quality_root = f"experiments/{experiment_id}/quality"
        spec = QualityExecutionSpec(
            argv=list(command.argv),
            cwd=cwd,
            env=dict(command.env),
            unset_env=list(command.unset_env),
            timeout_seconds=command.timeout_seconds,
            output_path=output_path,
            stdout_path=f"{quality_root}/quality.stdout",
            stderr_path=f"{quality_root}/quality.stderr",
        )
        spec_hash = _canonical_hash(
            {
                "request_hash": request_hash,
                "execution_spec": spec.model_dump(mode="json"),
                "coordinates": dict(spec_coordinates),
            }
        )
        with self.store.task_lock(task_id):
            current = self._load_current(task_id, experiment_id)
            if current is not None:
                exact = (
                    current.request_hash == request_hash
                    and current.spec_hash == spec_hash
                )
                if current.state == QualityExecutionState.RUNNING:
                    if process_matches(current.worker):
                        if not exact:
                            raise QualityExecutionError(
                                "a different quality request is already active"
                            )
                        return current
                    _terminate_process_group(current.evaluator)
                    current = _updated(
                        current,
                        state=QualityExecutionState.ORPHANED,
                        failure_reason="recorded worker is no longer alive",
                        terminal_at=datetime.now(UTC),
                    )
                    _save_execution(self.store, current)
                    if not retry_orphaned:
                        return current
                elif current.state == QualityExecutionState.STARTED:
                    current = _updated(
                        current,
                        state=QualityExecutionState.ORPHANED,
                        failure_reason="starter exited before recording its worker",
                        terminal_at=datetime.now(UTC),
                    )
                    _save_execution(self.store, current)
                    if not retry_orphaned:
                        return current
                elif current.state == QualityExecutionState.ORPHANED and exact:
                    if not retry_orphaned:
                        return current
                elif exact and current.state == QualityExecutionState.COMPLETED:
                    return current
                elif exact and current.state in {
                    QualityExecutionState.FAILED,
                    QualityExecutionState.TIMED_OUT,
                }:
                    if not retry_failed:
                        return current
                elif (
                    not exact
                    and retry_failed
                    and current.state
                    in {QualityExecutionState.FAILED, QualityExecutionState.TIMED_OUT}
                ):
                    pass
                elif not exact:
                    raise QualityExecutionError(
                        "this experiment already has a different quality request"
                    )
            execution = self._new_attempt(
                task_id, experiment_id, request_hash, spec_hash, spec
            )
            _save_execution(self.store, execution)
            self.store.save_json(
                task_id,
                _pointer_path(experiment_id),
                {
                    "attempt_id": execution.attempt_id,
                    "attempt_path": _attempt_path(experiment_id, execution.attempt_id),
                    "request_hash": request_hash,
                    "spec_hash": spec_hash,
                },
                producer="quality-execution",
            )
            worker_command = [
                sys.executable,
                "-m",
                "amd_inference_opt.quality_worker",
                "--store",
                str(self.store.root),
                "--task",
                task_id,
                "--experiment",
                experiment_id,
                "--attempt",
                execution.attempt_id,
            ]
            try:
                worker = self.launcher(
                    worker_command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                worker_identity = process_identity(worker.pid)
            except (OSError, QualityExecutionError) as exc:
                failed = _updated(
                    execution,
                    state=QualityExecutionState.FAILED,
                    failure_reason=f"quality worker launch failed: {exc}",
                    terminal_at=datetime.now(UTC),
                )
                _save_execution(self.store, failed)
                return failed
            running = _updated(
                execution,
                state=QualityExecutionState.RUNNING,
                worker=worker_identity,
                heartbeat_at=datetime.now(UTC),
            )
            _save_execution(self.store, running)
            return running

    def _load_current(
        self, task_id: str, experiment_id: str
    ) -> QualityExecution | None:
        try:
            pointer = self.store.load_json(task_id, _pointer_path(experiment_id))
        except StoreError:
            return None
        path = pointer.get("attempt_path") if isinstance(pointer, dict) else None
        if not isinstance(path, str):
            raise QualityExecutionError("quality execution pointer is invalid")
        execution = self.store.load_json(task_id, path, QualityExecution)
        if execution.task_id != task_id or execution.experiment_id != experiment_id:
            raise QualityExecutionError("quality execution pointer has wrong ownership")
        return execution

    def _new_attempt(
        self,
        task_id: str,
        experiment_id: str,
        request_hash: str,
        spec_hash: str,
        spec: QualityExecutionSpec,
    ) -> QualityExecution:
        attempts = (
            self.store.task_dir(task_id)
            / "experiments"
            / experiment_id
            / "quality"
            / "attempts"
        )
        existing = [
            int(path.stem.removeprefix("attempt-"))
            for path in attempts.glob("attempt-*.json")
            if path.stem.removeprefix("attempt-").isdigit()
        ]
        attempt_id = f"attempt-{max(existing, default=0) + 1:06d}"
        return QualityExecution(
            task_id=task_id,
            experiment_id=experiment_id,
            attempt_id=attempt_id,
            request_hash=request_hash,
            spec_hash=spec_hash,
            state=QualityExecutionState.STARTED,
            spec=spec,
        )


def run_quality_worker(
    store_root: str | Path,
    task_id: str,
    experiment_id: str,
    attempt_id: str,
    *,
    poll_seconds: float = 0.1,
    heartbeat_seconds: float = 1.0,
) -> int:
    """Worker entry point; evaluator output streams go directly to stable files."""

    store = ExperimentStore(store_root)
    worker_identity = process_identity(os.getpid())
    attempt_path = _attempt_path(experiment_id, attempt_id)
    with store.task_lock(task_id):
        execution = store.load_json(task_id, attempt_path, QualityExecution)
        if execution.state in TERMINAL_QUALITY_STATES:
            return 0
        if execution.state != QualityExecutionState.RUNNING:
            raise QualityExecutionError(
                f"worker found attempt in unexpected state {execution.state}"
            )
        if execution.worker != worker_identity:
            raise QualityExecutionError("worker PID identity does not own this attempt")
        spec = execution.spec

    stdout = store.task_dir(task_id) / spec.stdout_path
    stderr = store.task_dir(task_id) / spec.stderr_path
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    process_env = os.environ.copy()
    for name in spec.unset_env:
        process_env.pop(name, None)
    process_env.update(spec.env)
    evaluator: subprocess.Popen[bytes] | None = None
    evaluator_identity: ProcessIdentity | None = None
    start = time.monotonic()
    last_heartbeat = 0.0
    timed_out = False
    output_path = store.task_dir(task_id) / spec.output_path
    initial_output = (
        _file_identity(output_path) if output_path.is_file() else None
    )
    try:
        with stdout.open("wb") as stdout_file, stderr.open("wb") as stderr_file:
            evaluator = subprocess.Popen(
                spec.argv,
                cwd=spec.cwd,
                env=process_env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                shell=False,
                start_new_session=True,
            )
            evaluator_identity = process_identity(evaluator.pid)
            _heartbeat(
                store,
                execution,
                worker_identity,
                evaluator=evaluator_identity,
            )
            while evaluator.poll() is None:
                now = time.monotonic()
                if now - start >= spec.timeout_seconds:
                    timed_out = True
                    _terminate_process_group(evaluator_identity)
                    evaluator.wait(timeout=5)
                    break
                if now - last_heartbeat >= heartbeat_seconds:
                    execution = _heartbeat(
                        store,
                        execution,
                        worker_identity,
                        evaluator=evaluator_identity,
                    )
                    last_heartbeat = now
                time.sleep(poll_seconds)
        exit_code = evaluator.returncode
        output = output_path
        if timed_out:
            final_state = QualityExecutionState.TIMED_OUT
            reason = f"quality command timed out after {spec.timeout_seconds} seconds"
        elif (
            exit_code == 0
            and output.is_file()
            and _file_identity(output) != initial_output
        ):
            final_state = QualityExecutionState.COMPLETED
            reason = None
        elif exit_code == 0:
            final_state = QualityExecutionState.FAILED
            reason = (
                "quality command exited successfully without producing a fresh "
                "declared output"
            )
        else:
            final_state = QualityExecutionState.FAILED
            reason = f"quality command exited with code {exit_code}"
        finalize_quality_execution(
            store,
            task_id,
            experiment_id,
            attempt_id,
            final_state,
            exit_code=exit_code,
            failure_reason=reason,
            expected_worker=worker_identity,
            bind_artifacts=True,
        )
        return 0
    except BaseException as exc:
        if evaluator is not None and evaluator.poll() is None:
            _terminate_process_group(evaluator_identity)
        finalize_quality_execution(
            store,
            task_id,
            experiment_id,
            attempt_id,
            QualityExecutionState.FAILED,
            exit_code=evaluator.returncode if evaluator is not None else None,
            failure_reason=f"quality worker failed: {type(exc).__name__}: {exc}",
            expected_worker=worker_identity,
            bind_artifacts=True,
        )
        return 1


def _heartbeat(
    store: ExperimentStore,
    execution: QualityExecution,
    worker: ProcessIdentity,
    *,
    evaluator: ProcessIdentity,
) -> QualityExecution:
    with store.task_lock(execution.task_id):
        current = store.load_json(
            execution.task_id,
            _attempt_path(execution.experiment_id, execution.attempt_id),
            QualityExecution,
        )
        if current.state != QualityExecutionState.RUNNING or current.worker != worker:
            raise QualityExecutionError("quality worker lost ownership")
        current = _updated(
            current,
            evaluator=evaluator,
            heartbeat_at=datetime.now(UTC),
        )
        _save_execution(store, current)
        return current


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


__all__ = [
    "QualityExecutionError",
    "QualityExecutionManager",
    "TERMINAL_QUALITY_STATES",
    "finalize_quality_execution",
    "process_identity",
    "process_matches",
    "run_quality_worker",
]
