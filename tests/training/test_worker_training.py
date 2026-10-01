import json

import pytest

from macfit_training.generation import GenerationError
from macfit_training.service.outputs import verify_outputs
from macfit_training.service.settings import Settings
from macfit_training.worker import EventWriter, execute_job, read_input


def generation_request():
    return {
        "model_id": "qwen3-0-6b",
        "task": "answers",
        "goal": "Answer support questions clearly.",
        "language": "en",
        "source": "",
        "purpose": "preview",
        "target_count": 3,
        "seeds": [],
    }


def test_worker_publishes_result_only_after_all_artifacts_and_refuses_duplicate_run(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MACFIT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    (tmp_path / "input.json").write_text(json.dumps(generation_request()))

    def run(job, directory, emit):
        assert not (directory / "result.json").exists()
        assert len(job["base_model"]["revision"]) == 40
        emit("generating_examples", completed=3, total=3, unit="examples")
        return {"examples": [], "provenance": {"method": "test"}, "base_model": job["base_model"]}

    assert execute_job(tmp_path, "generation", runners={"generation": run}) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert len(result["artifacts"]) == 3
    assert all(
        (tmp_path / "artifacts" / record["name"]).is_file() for record in result["artifacts"]
    )
    public_result, verified_artifacts = verify_outputs(
        tmp_path, Settings(data_dir=tmp_path, gateway_secret="test-secret-" * 4)
    )
    assert public_result["examples"] == []
    assert verified_artifacts == result["artifacts"]
    assert (
        json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1])["stage"] == "succeeded"
    )
    with pytest.raises(ValueError, match="already been used"):
        execute_job(tmp_path, "generation", runners={"generation": run})


def test_invalid_generation_fails_retryably_without_success_output(tmp_path, monkeypatch):
    monkeypatch.setenv("MACFIT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    raw = json.dumps(generation_request())
    (tmp_path / "input.json").write_text(raw)

    def fail(*_args):
        raise GenerationError("Invalid generated JSON; please retry.")

    assert execute_job(tmp_path, "generation", runners={"generation": fail}) == 2
    assert not (tmp_path / "result.json").exists()
    error = json.loads((tmp_path / "error.json").read_text())
    assert error["code"] == "generation_invalid" and error["retryable"] is True
    assert (tmp_path / "input.json").read_text() == raw


def test_invalid_input_never_invokes_gpu_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("MACFIT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    (tmp_path / "input.json").write_text(
        json.dumps({**generation_request(), "command": "forbidden"})
    )

    def run(*_args):
        pytest.fail("Invalid input must never reach GPU execution")

    assert execute_job(tmp_path, "generation", runners={"generation": run}) == 2
    assert json.loads((tmp_path / "error.json").read_text())["code"] == "invalid_input"


def test_symlink_input_and_invented_progress_are_rejected(tmp_path):
    source = tmp_path / "source"
    source.write_text("{}")
    (tmp_path / "input.json").symlink_to(source)
    with pytest.raises(OSError):
        read_input(tmp_path / "input.json")
    events = EventWriter(tmp_path / "events.jsonl")
    try:
        with pytest.raises(ValueError):
            events("training", completed=3, total=1, unit="steps")
    finally:
        events.close()
