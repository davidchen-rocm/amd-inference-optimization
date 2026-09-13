from __future__ import annotations

import json
from pathlib import Path

import pytest

from amd_inference_opt.replay import ReplayError, run_recorded_replay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "examples" / "q4-rdna-recorded.yaml"


def test_q4_rdna_replay_reproduces_recorded_decisions(tmp_path: Path) -> None:
    result = run_recorded_replay(CONFIG, tmp_path / "run")

    assert result.final_decision == "ACCEPT"
    assert result.experiment_decisions == {
        "q4_rdna_old_mapping": "REJECT",
        "q4_rdna_split_k": "ACCEPT",
    }

    experiments = {item["id"]: item["gate"] for item in result.report["experiments"]}
    old = experiments["q4_rdna_old_mapping"]
    split = experiments["q4_rdna_split_k"]
    assert old["improvement_fraction"]["128"] == pytest.approx(-0.36893565)
    assert old["improvement_fraction"]["512"] == pytest.approx(-0.36457271)
    assert split["improvement_fraction"]["128"] == pytest.approx(0.15324869)
    assert split["improvement_fraction"]["512"] == pytest.approx(0.15330595)
    assert split["quality"]["perplexity_regression_fraction"] == pytest.approx(0.01441598)
    assert split["quality"]["math_accuracy_drop_points"] == pytest.approx(13 / 848)


def test_replay_materializes_auditable_recorded_store(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    run_recorded_replay(CONFIG, destination)

    expected = [
        "task.json",
        "state.json",
        "events.jsonl",
        "baseline/decomposition.json",
        "execution-map.json",
        "mcp/recorded-evidence.json",
        "analysis/bottleneck.json",
        "analysis/limit-estimate.json",
        "experiments/q4_rdna_old_mapping/decision.json",
        "experiments/q4_rdna_split_k/decision.json",
        "final-report/report.json",
        "final-report/report.md",
        "final-report/final-config.json",
        "artifact-manifest.json",
    ]
    assert all((destination / relative).is_file() for relative in expected)

    report = json.loads((destination / "final-report" / "report.json").read_text())
    state = json.loads((destination / "state.json").read_text())
    events = [json.loads(line) for line in (destination / "events.jsonl").read_text().splitlines()]
    execution_map = json.loads((destination / "execution-map.json").read_text())
    manifest = json.loads((destination / "artifact-manifest.json").read_text())

    assert report["evidence_kind"] == "recorded_import"
    assert state["source"] == "recorded_import"
    assert state["workflow_status"] == "ACCEPTED"
    assert events[0]["stage"] == "CREATE_TASK"
    assert [event["decision"] for event in events if event["stage"] == "DECIDE"] == [
        "REJECT",
        "ACCEPT",
    ]
    assert execution_map["coverage"] == "dominant decode GEMV path only"
    assert "MemUnitBusy" in execution_map["unsupported_counters"]
    assert "not a live MCP observation" in (destination / "final-report" / "report.md").read_text()
    final_config = json.loads(
        (destination / "final-report" / "final-config.json").read_text()
    )
    assert final_config["unset_environment"] == ["LLAMA_Q4_RDNA_MAPPING"]
    manifest_paths = {artifact["path"] for artifact in manifest["artifacts"]}
    assert "final-report/final-config.json" in manifest_paths
    assert "analysis/limit-estimate.json" in manifest_paths


def test_replay_refuses_to_overwrite_a_nonempty_store(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    (destination / "owned-by-user.txt").write_text("keep")

    with pytest.raises(ReplayError, match="refusing to overwrite"):
        run_recorded_replay(CONFIG, destination)


def test_replay_returns_inconclusive_for_insufficient_samples(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    fixture = json.loads((PROJECT_ROOT / "fixtures/q4-rdna-recorded/replay.json").read_text())
    for experiment in fixture["experiments"]:
        experiment.pop("expected_decision")
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture))
    config["fixture"] = str(fixture_path)
    config["stability"]["minimum_samples"] = 4
    config_path = tmp_path / "unstable.json"
    config_path.write_text(json.dumps(config))

    result = run_recorded_replay(config_path, tmp_path / "run")

    assert result.final_decision == "INCONCLUSIVE"
    assert result.experiment_decisions == {
        "q4_rdna_old_mapping": "INCONCLUSIVE",
        "q4_rdna_split_k": "INCONCLUSIVE",
    }
