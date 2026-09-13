from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

from amd_inference_opt.command import StageCommand
from amd_inference_opt.models import (
    MCPConfig,
    ModelTarget,
    OptimizationTask,
    QualityExecutionState,
    RuntimeTarget,
)
from amd_inference_opt.quality_execution import (
    QualityExecutionManager,
    finalize_quality_execution,
    process_matches,
)
from amd_inference_opt.store import ExperimentStore


def _store(tmp_path: Path, task_id: str = "quality-task") -> ExperimentStore:
    store = ExperimentStore(tmp_path / "store")
    store.create_task(
        OptimizationTask(
            id=task_id,
            model=ModelTarget(path=tmp_path / "model.gguf"),
            runtime=RuntimeTarget(repo_path=tmp_path, base_commit="fixture"),
            mcp=MCPConfig(command=["unused-mcp"]),
        )
    )
    return store


def _wait_terminal(
    manager: QualityExecutionManager,
    task_id: str,
    experiment_id: str,
    command: StageCommand,
    output_path: str,
    *,
    timeout: float = 5,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        execution = manager.start_or_resume(
            task_id,
            experiment_id,
            command,
            output_path=output_path,
            spec_coordinates={"fixture": 1},
        )
        if execution.state not in {
            QualityExecutionState.STARTED,
            QualityExecutionState.RUNNING,
        }:
            return execution
        time.sleep(0.05)
    raise AssertionError("quality worker did not reach a terminal state")


def test_task_lock_keeps_one_inode(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lock = store.task_dir("quality-task") / ".task.lock"
    inode = lock.stat().st_ino

    with store.task_lock("quality-task"):
        assert lock.stat().st_ino == inode
    with store.task_lock("quality-task"):
        assert lock.stat().st_ino == inode


def test_start_or_resume_reuses_active_attempt_and_binds_direct_files(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    experiment_id = "split-k"
    output_relative = f"experiments/{experiment_id}/quality/result.json"
    output = store.task_dir("quality-task") / output_relative
    payload = {"status": "complete", "value": 7}
    code = (
        "import json,pathlib,sys,time;"
        "print('quality-out', flush=True);"
        "print('quality-err', file=sys.stderr, flush=True);"
        "time.sleep(0.3);"
        f"pathlib.Path({str(output)!r}).write_text(json.dumps({payload!r}))"
    )
    command = StageCommand(
        name="quality",
        argv=(sys.executable, "-c", code),
        cwd=str(tmp_path),
        timeout_seconds=3,
    )
    manager = QualityExecutionManager(store)

    first = manager.start_or_resume(
        "quality-task",
        experiment_id,
        command,
        output_path=output_relative,
        spec_coordinates={"fixture": 1},
    )
    second = manager.start_or_resume(
        "quality-task",
        experiment_id,
        command,
        output_path=output_relative,
        spec_coordinates={"fixture": 1},
    )

    assert first.state == QualityExecutionState.RUNNING
    assert second.attempt_id == first.attempt_id
    assert second.worker == first.worker
    completed = _wait_terminal(
        manager, "quality-task", experiment_id, command, output_relative
    )
    assert completed.state == QualityExecutionState.COMPLETED
    assert completed.exit_code == 0
    assert completed.output_artifact is not None
    assert completed.stdout_artifact is not None
    assert completed.stderr_artifact is not None
    assert store.verify_artifact("quality-task", completed.output_artifact)
    assert store.verify_artifact("quality-task", completed.stdout_artifact)
    assert store.verify_artifact("quality-task", completed.stderr_artifact)
    assert json.loads(output.read_text())["value"] == 7
    assert "quality-out" in (
        store.task_dir("quality-task") / completed.spec.stdout_path
    ).read_text()
    assert "quality-err" in (
        store.task_dir("quality-task") / completed.spec.stderr_path
    ).read_text()

    repeated = finalize_quality_execution(
        store,
        "quality-task",
        experiment_id,
        completed.attempt_id,
        QualityExecutionState.FAILED,
        failure_reason="must not replace completion",
    )
    assert repeated == completed


def test_timeout_kills_evaluator_process_group(tmp_path: Path) -> None:
    store = _store(tmp_path)
    experiment_id = "timeout"
    quality_dir = store.task_dir("quality-task") / "experiments/timeout/quality"
    child_pid_path = quality_dir / "child.pid"
    output_relative = "experiments/timeout/quality/result.json"
    code = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid));"
        "time.sleep(30)"
    )
    command = StageCommand(
        name="quality",
        argv=(sys.executable, "-c", code),
        cwd=str(tmp_path),
        timeout_seconds=0.25,
    )
    manager = QualityExecutionManager(store)
    manager.start_or_resume(
        "quality-task",
        experiment_id,
        command,
        output_path=output_relative,
        spec_coordinates={"fixture": 1},
    )
    terminal = _wait_terminal(
        manager, "quality-task", experiment_id, command, output_relative
    )

    assert terminal.state == QualityExecutionState.TIMED_OUT
    assert terminal.stdout_artifact is not None
    assert terminal.stderr_artifact is not None
    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and Path(f"/proc/{child_pid}/stat").exists():
        stat = Path(f"/proc/{child_pid}/stat").read_text()
        if stat[stat.rfind(")") + 2 :].split()[0] == "Z":
            break
        time.sleep(0.05)
    if Path(f"/proc/{child_pid}/stat").exists():
        stat = Path(f"/proc/{child_pid}/stat").read_text()
        assert stat[stat.rfind(")") + 2 :].split()[0] == "Z"


def test_dead_worker_becomes_orphaned_and_requires_explicit_retry(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    experiment_id = "orphan"
    output_relative = "experiments/orphan/quality/result.json"
    command = StageCommand(
        name="quality",
        argv=(sys.executable, "-c", "import time;time.sleep(30)"),
        cwd=str(tmp_path),
        timeout_seconds=30,
    )
    manager = QualityExecutionManager(store)
    running = manager.start_or_resume(
        "quality-task",
        experiment_id,
        command,
        output_path=output_relative,
        spec_coordinates={"fixture": 1},
    )
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = manager.start_or_resume(
            "quality-task",
            experiment_id,
            command,
            output_path=output_relative,
            spec_coordinates={"fixture": 1},
        )
        if current.evaluator is not None:
            running = current
            break
        time.sleep(0.05)
    assert running.worker is not None
    assert running.evaluator is not None
    os.kill(running.worker.pid, signal.SIGKILL)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and process_matches(running.worker):
        time.sleep(0.05)

    orphaned = manager.start_or_resume(
        "quality-task",
        experiment_id,
        command,
        output_path=output_relative,
        spec_coordinates={"fixture": 1},
    )
    assert orphaned.state == QualityExecutionState.ORPHANED
    assert orphaned.attempt_id == running.attempt_id
    again = manager.start_or_resume(
        "quality-task",
        experiment_id,
        command,
        output_path=output_relative,
        spec_coordinates={"fixture": 1},
    )
    assert again.attempt_id == running.attempt_id

    retried = manager.start_or_resume(
        "quality-task",
        experiment_id,
        command,
        output_path=output_relative,
        spec_coordinates={"fixture": 1},
        retry_orphaned=True,
    )
    assert retried.state == QualityExecutionState.RUNNING
    assert retried.attempt_id != running.attempt_id
    assert retried.worker is not None
    os.kill(retried.worker.pid, signal.SIGKILL)
    if retried.evaluator is not None:
        try:
            os.killpg(retried.evaluator.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_failed_exact_request_requires_explicit_new_attempt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    experiment_id = "failed-retry"
    output_relative = "experiments/failed-retry/quality/result.json"
    command = StageCommand(
        name="quality",
        argv=(sys.executable, "-c", "raise SystemExit(7)"),
        cwd=str(tmp_path),
        timeout_seconds=3,
    )
    manager = QualityExecutionManager(store)
    manager.start_or_resume(
        "quality-task",
        experiment_id,
        command,
        output_path=output_relative,
        spec_coordinates={"fixture": 1},
    )
    failed = _wait_terminal(
        manager, "quality-task", experiment_id, command, output_relative
    )

    same = manager.start_or_resume(
        "quality-task",
        experiment_id,
        command,
        output_path=output_relative,
        spec_coordinates={"fixture": 1},
    )
    assert same.attempt_id == failed.attempt_id
    retried = manager.start_or_resume(
        "quality-task",
        experiment_id,
        command,
        output_path=output_relative,
        spec_coordinates={"fixture": 1},
        retry_failed=True,
    )
    assert retried.state == QualityExecutionState.RUNNING
    assert retried.attempt_id != failed.attempt_id
    _wait_terminal(manager, "quality-task", experiment_id, command, output_relative)
