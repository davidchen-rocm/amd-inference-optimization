from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from amd_inference_opt.cli import app
from amd_inference_opt.models import ApprovalRequest
from amd_inference_opt.rocm_mcp import approval_request
from amd_inference_opt.store import ExperimentStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_replay_command_runs_without_live_tools(tmp_path: Path) -> None:
    output = tmp_path / "recorded-run"
    result = runner.invoke(
        app,
        [
            "replay",
            str(PROJECT_ROOT / "examples" / "q4-rdna-recorded.yaml"),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    assert response["evidence_kind"] == "recorded_import"
    assert response["experiment_decisions"] == {
        "q4_rdna_old_mapping": "REJECT",
        "q4_rdna_split_k": "ACCEPT",
    }
    assert (output / "final-report" / "final-config.json").is_file()
    assert not (output / "final-report" / "final.patch").exists()


def test_task_create_and_status(tmp_path: Path) -> None:
    config = {
        "schema_version": 1,
        "id": "unit-task",
        "model": {"path": "/models/qwen.gguf", "architecture": "qwen"},
        "runtime": {"repo_path": "/src/llama.cpp", "base_commit": "abc123"},
        "mcp": {"command": ["/opt/rocm-agent-mcp"]},
    }
    config_path = tmp_path / "task.yaml"
    config_path.write_text(json.dumps(config))
    store_root = tmp_path / "store"

    created = runner.invoke(
        app,
        ["task", "create", "--config", str(config_path), "--store", str(store_root)],
    )
    assert created.exit_code == 0, created.output
    assert json.loads(created.stdout)["stage"] == "CREATE_TASK"

    status = runner.invoke(app, ["task", "status", "unit-task", "--store", str(store_root)])
    assert status.exit_code == 0, status.output
    payload = json.loads(status.stdout)
    assert payload["task"]["id"] == "unit-task"
    assert payload["workflow"]["current_stage"] == "CREATE_TASK"

    advanced = runner.invoke(app, ["advance", "unit-task", "--store", str(store_root)])
    assert advanced.exit_code == 0, advanced.output
    advance_payload = json.loads(advanced.stdout)
    assert advance_payload["completed_stages"] == ["CREATE_TASK"]
    assert advance_payload["stage"] == "INSPECT_TARGET"
    assert advance_payload["paused_for"] == "evidence"


def test_approve_is_exact_and_one_time(tmp_path: Path) -> None:
    config = {
        "schema_version": 1,
        "id": "approval-task",
        "model": {"path": "/models/qwen.gguf"},
        "runtime": {"repo_path": "/src/llama.cpp", "base_commit": "abc123"},
        "mcp": {"command": ["/opt/rocm-agent-mcp"]},
    }
    config_path = tmp_path / "task.yaml"
    config_path.write_text(json.dumps(config))
    store_root = tmp_path / "store"
    created = runner.invoke(
        app,
        ["task", "create", "--config", str(config_path), "--store", str(store_root)],
    )
    assert created.exit_code == 0, created.output

    arguments = {"command": ["/bin/true"], "preset": "kernel-timing"}
    request_hash = approval_request("rocm_profile_workload", arguments).request_sha256
    request = ApprovalRequest(
        id="request-1",
        task_id="approval-task",
        tool="rocm_profile_workload",
        arguments=arguments,
        request_hash=request_hash,
    )
    ExperimentStore(store_root).save_json(
        "approval-task",
        "state/approvals/request-1.request.json",
        request,
        producer="workflow",
    )

    approved = runner.invoke(
        app,
        ["approve", "approval-task", "request-1", "--store", str(store_root)],
    )
    assert approved.exit_code == 0, approved.output
    assert json.loads(approved.stdout)["request_hash"] == request_hash

    replayed = runner.invoke(
        app,
        ["approve", "approval-task", "request-1", "--store", str(store_root)],
    )
    assert replayed.exit_code == 2
    assert "approval already exists" in replayed.stderr


def test_live_example_initializes_without_executing_workload(tmp_path: Path) -> None:
    store = tmp_path / "store"
    result = runner.invoke(
        app,
        [
            "run-live",
            str(PROJECT_ROOT / "examples" / "q4-rdna-live.yaml"),
            "--store",
            str(store),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["task_id"] == "q4-rdna-live"
    assert payload["stage"] == "CREATE_TASK"
    assert payload["execution_started"] is False
    assert payload["approval_required_before_executing_mcp"] is True

    resumed = runner.invoke(
        app,
        [
            "run-live",
            str(PROJECT_ROOT / "examples" / "q4-rdna-live.yaml"),
            "--store",
            str(store),
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    assert json.loads(resumed.stdout)["task_id"] == "q4-rdna-live"
