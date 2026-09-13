import json
from pathlib import Path

import pytest

from amd_inference_opt.models import MCPConfig, ModelTarget, OptimizationTask, RuntimeTarget
from amd_inference_opt.store import ExperimentStore, StoreError
from amd_inference_opt.workflow import WorkflowEngine


def make_task(tmp_path: Path) -> OptimizationTask:
    return OptimizationTask(
        id="store-task",
        model=ModelTarget(path=tmp_path / "model.gguf"),
        runtime=RuntimeTarget(repo_path=tmp_path / "llama.cpp", base_commit="abc"),
        mcp=MCPConfig(command=["rocm-agent-mcp"]),
    )


def test_store_round_trips_records_and_verifies_manifest(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "store")
    task = make_task(tmp_path)
    store.create_task(task)
    workflow_ref = store.save_workflow(WorkflowEngine.new(task.id))
    log_ref = store.save_text(task.id, "artifacts/build.log", "ok\n", producer="runner")

    assert store.load_task(task.id) == task
    assert store.load_workflow(task.id).current_stage == "CREATE_TASK"
    assert store.verify_artifact(task.id, workflow_ref)
    assert store.verify_artifact(task.id, log_ref)
    manifest = json.loads(
        (store.task_dir(task.id) / "artifacts/manifest.json").read_text(encoding="utf-8")
    )
    assert "artifacts/build.log" in manifest["artifacts"]


def test_store_refuses_overwrite_creation_and_path_escape(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "store")
    task = make_task(tmp_path)
    store.create_task(task)

    with pytest.raises(StoreError, match="already exists"):
        store.create_task(task)
    with pytest.raises(StoreError, match="task-relative"):
        store.save_text(task.id, "../escape", "no", producer="test")


def test_import_artifact_hashes_content_and_events_are_jsonl(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "store")
    task = make_task(tmp_path)
    store.create_task(task)
    source = tmp_path / "trace.json"
    source.write_text('{"kernel": "gemv"}\n', encoding="utf-8")

    ref = store.import_artifact(
        task.id, "artifacts/trace.json", source, producer="rocm_issue_agent"
    )
    store.append_event(task.id, "profile_complete", {"artifact": ref.path})

    assert store.verify_artifact(task.id, ref)
    event = json.loads(
        (store.task_dir(task.id) / "events/events.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert event["event"] == "profile_complete"


def test_evidence_writes_are_versioned_immutable_and_manifest_addressable(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / "store")
    task = make_task(tmp_path)
    store.create_task(task)

    first = store.save_evidence_json(
        task.id, "kernel/profile.json", {"attempt": 1}, producer="test"
    )
    second = store.save_evidence_json(
        task.id, "kernel/profile.json", {"attempt": 2}, producer="test"
    )

    assert first.path.endswith("kernel/profile/v000001.json")
    assert second.path.endswith("kernel/profile/v000002.json")
    assert first.sha256 != second.sha256
    assert store.load_json(task.id, first.path)["attempt"] == 1
    assert store.artifact_ref(task.id, first.path) == first
    with pytest.raises(StoreError, match="immutable artifact already exists"):
        store.save_immutable_bytes(
            task.id, first.path, b"replacement", producer="test"
        )


def test_task_lock_is_added_lazily_without_rewriting_existing_records(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / "store")
    task = make_task(tmp_path)
    store.create_task(task)
    lock = store.task_dir(task.id) / ".task.lock"
    lock.unlink()
    task_path = store.task_dir(task.id) / "task.json"
    original = task_path.read_bytes()
    original_mtime = task_path.stat().st_mtime_ns

    with store.task_lock(task.id):
        assert lock.is_file()

    assert task_path.read_bytes() == original
    assert task_path.stat().st_mtime_ns == original_mtime
