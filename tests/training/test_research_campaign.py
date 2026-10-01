from __future__ import annotations

import fcntl
import json
import secrets
import signal
import time

import pytest

import macfit_training.campaign as campaign
from macfit_training.artifacts import describe_artifact, write_json
from macfit_training.campaign import RECIPES, instant, recipe_identity, run_campaign
from macfit_training.config import validate_job_input
from macfit_training.experiments import build_input
from macfit_training.robustness import build_input as build_robustness_input
from macfit_training.service.settings import Settings
from macfit_training.service.store import JobStore, ServiceError


def prepare_job(store, campaign_id, recipe=RECIPES[0], *, complete=False, variant="baseline"):
    model, preset = recipe
    project, request = recipe_identity(campaign_id, model, preset)
    builder = build_input if variant == "baseline" else build_robustness_input
    fixture = builder(model, preset)
    job, created = store.create(
        "macfit-research-" + campaign_id,
        request,
        project,
        "training",
        validate_job_input("training", fixture),
    )
    assert created
    path = store.directory(job["id"])
    if complete:
        evaluation = {
            "samples": [
                {
                    **{k: r[k] for k in ("id", "question", "expected")},
                    "before": r["expected"],
                    "after": r["expected"],
                }
                for r in fixture["evaluation"]
            ]
        }
        evaluation_path = path / "artifacts/evaluation.json"
        write_json(evaluation_path, evaluation)
        store.finish(
            job["id"],
            "succeeded",
            result={"training": {"steps": 25}},
            artifacts=[describe_artifact(evaluation_path, "evaluation")],
        )
    return job, path


def test_expired_window_cannot_submit_a_research_job(tmp_path):
    state = run_campaign(tmp_path, "research-test", time.time() - 1)
    assert state["status"] == "deadline_closed"
    assert state["runs"] == []
    store = JobStore(Settings(tmp_path, secrets.token_hex(32)))
    assert store.queue_size() == 0
    assert store.list("macfit-research-research-test") == []


def test_recovered_completed_campaign_scores_real_saved_files_without_requeue(tmp_path):
    campaign_id = "research-test"
    owner = "macfit-research-" + campaign_id
    store = JobStore(Settings(tmp_path, secrets.token_hex(32)))
    snapshots = []
    for recipe in RECIPES:
        _, path = prepare_job(store, campaign_id, recipe, complete=True)
        snapshots.append((path / "input.json", (path / "input.json").stat().st_mtime_ns))
    first = run_campaign(tmp_path, campaign_id, time.time() - 1)
    second = run_campaign(tmp_path, campaign_id, time.time() - 1)
    assert first["status"] == second["status"] == "succeeded"
    assert len(store.list(owner)) == len(RECIPES)
    assert store.queue_size() == 0
    assert [r["job_id"] for r in first["runs"]] == [r["job_id"] for r in second["runs"]]
    for path, mtime in snapshots:
        assert path.stat().st_mtime_ns == mtime
    for model, preset in RECIPES:
        name = f"policy-json-{campaign_id}-{model}-{preset}.json"
        path = tmp_path / "evidence" / name
        result = json.loads(path.read_text())
        assert result["metrics"]["after"]["samples"] == 20
        row = next(r for r in second["runs"] if r["recipe"] == f"{model}-{preset}")
        assert row["assessment_file"] == name


def test_campaign_lease_rejects_another_live_coordinator(tmp_path):
    directory = tmp_path / "campaigns"
    directory.mkdir()
    with (directory / "research-test.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="running coordinator"):
            run_campaign(tmp_path, "research-test", time.time() + 3600)


def test_campaign_identity_rejects_paths_and_is_repeatable():
    for campaign_id in ("../escape", "", "A", "a" * 81):
        with pytest.raises(ValueError):
            recipe_identity(campaign_id, *RECIPES[0])
    assert recipe_identity("a", *RECIPES[0]) == recipe_identity("a", *RECIPES[0])
    assert recipe_identity("a", *RECIPES[0]) != recipe_identity("a", *RECIPES[1])
    with pytest.raises(ValueError, match="timezone"):
        instant("2026-10-01T00:00:00")


def test_persistently_failed_publish_cancels_queued_job_and_preserves_original_error(
    tmp_path, monkeypatch
):
    def fail_write(*_args, **_kwargs):
        raise OSError("Simulated disk failure with private diagnostic text")

    monkeypatch.setattr(campaign, "write_json", fail_write)
    with pytest.raises(OSError, match="Simulated disk failure"):
        run_campaign(tmp_path, "publish-failed", time.time() + 7200)
    store = JobStore(Settings(tmp_path, secrets.token_hex(32)))
    jobs = store.list("macfit-research-publish-failed")
    assert len(jobs) == 1
    assert jobs[0]["status"] == "cancelled"
    assert store.queue_size() == 0


def test_exception_after_claim_cancels_running_job_and_saves_sanitized_failure(
    tmp_path, monkeypatch
):
    store = JobStore(Settings(tmp_path, secrets.token_hex(32)))
    original = campaign.write_json
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            assert store.claim()["status"] == "running"
            raise OSError("private credential text must not be included in evidence")
        return original(*args, **kwargs)

    previous_handler = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(campaign, "write_json", fail_once)
    with pytest.raises(OSError, match="private credential"):
        run_campaign(tmp_path, "claim-failed", time.time() + 7200)
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    state = json.loads((tmp_path / "evidence/claim-failed.json").read_text())
    assert state["status"] == "failed"
    assert state["failure"] == {"code": "coordinator_failed", "exception_type": "OSError"}
    assert "private credential" not in json.dumps(state)
    jobs = store.list("macfit-research-claim-failed")
    assert len(jobs) == 1 and jobs[0]["status"] == "cancelling"
    assert state["runs"][0]["status"] == "cancelling"


@pytest.mark.parametrize("running", [False, True])
def test_existing_active_job_is_cancelled_at_expired_deadline_without_requeue(tmp_path, running):
    store = JobStore(Settings(tmp_path, secrets.token_hex(32)))
    job, _ = prepare_job(store, "expired-active")
    if running:
        assert store.claim()["id"] == job["id"]
    state = run_campaign(tmp_path, "expired-active", time.time() - 1)
    assert state["status"] == "deadline_closed"
    assert len(store.list("macfit-research-expired-active")) == 1
    assert store.get(job["id"])["status"] == ("cancelling" if running else "cancelled")


def test_stop_handler_cancels_active_job_and_is_restored(tmp_path, monkeypatch):
    previous_handler = signal.getsignal(signal.SIGTERM)

    def stop_on_poll(_seconds):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setattr(campaign.time, "sleep", stop_on_poll)
    state = run_campaign(tmp_path, "signal-stop", time.time() + 7200)
    assert state["status"] == "interrupted"
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    store = JobStore(Settings(tmp_path, secrets.token_hex(32)))
    jobs = store.list("macfit-research-signal-stop")
    assert len(jobs) == 1 and jobs[0]["status"] == "cancelled"


@pytest.mark.parametrize("tamper", ["content", "symlink", "oversize"])
def test_saved_evaluation_requires_registered_integrity_and_bounded_regular_file(tmp_path, tamper):
    store = JobStore(Settings(tmp_path, secrets.token_hex(32)))
    job, path = prepare_job(store, "tampered-result", complete=True)
    evaluation = path / "artifacts/evaluation.json"
    if tamper == "content":
        evaluation.write_text("{}")
    elif tamper == "symlink":
        target = tmp_path / "unexpected-evaluation.json"
        target.write_bytes(evaluation.read_bytes())
        evaluation.unlink()
        evaluation.symlink_to(target)
    else:
        with evaluation.open("r+b") as stream:
            stream.truncate(8 * 1024**2 + 1)
    with pytest.raises(ServiceError) as error:
        run_campaign(tmp_path, "tampered-result", time.time() - 1)
    assert error.value.code == "artifact_unavailable"
    state = json.loads((tmp_path / "evidence/tampered-result.json").read_text())
    assert state["status"] == "failed"
    assert state["failure"]["exception_type"] == "ServiceError"
    assert "assessment_file" not in state["runs"][0]
    assert store.get(job["id"])["status"] == "succeeded"


def test_campaigns_preserve_separate_assessments_in_same_service_data(tmp_path):
    store = JobStore(Settings(tmp_path, secrets.token_hex(32)))
    snapshots = {}
    for campaign_id in ("first-campaign", "second-campaign"):
        for recipe in RECIPES:
            prepare_job(store, campaign_id, recipe, complete=True)
        state = run_campaign(tmp_path, campaign_id, time.time() - 1)
        assert state["status"] == "succeeded"
        for row in state["runs"]:
            path = tmp_path / "evidence" / row["assessment_file"]
            assert campaign_id in path.name
            assert path not in snapshots
            snapshots[path] = path.read_bytes()
    assert len(snapshots) == 12
    assert all(path.read_bytes() == snapshot for path, snapshot in snapshots.items())


def test_robustness_recipes_use_three_standard_models_without_modifying_baseline():
    assert len(campaign.RECIPES) == 6
    recipes, builder, assessor = campaign.variant_protocol("robustness")
    assert recipes == (
        ("qwen3-0-6b", "standard"),
        ("qwen3-4b", "standard"),
        ("qwen3-8b", "standard"),
    )
    assert len(builder("qwen3-4b", "standard")["training"]) == 360
    assert assessor is campaign.assess_robustness
    assert campaign.variant_protocol("baseline") == (
        campaign.RECIPES,
        campaign.build_input,
        campaign.assess_evaluation,
    )


def test_recovered_robustness_campaign_scores_its_own_bound_inputs_without_requeue(tmp_path):
    campaign_id = "adaptive-robustness"
    store = JobStore(Settings(tmp_path, secrets.token_hex(32)))
    for recipe in campaign.ROBUSTNESS_RECIPES:
        prepare_job(store, campaign_id, recipe, complete=True, variant="robustness")
    first = run_campaign(tmp_path, campaign_id, time.time() - 1, variant="robustness")
    second = run_campaign(tmp_path, campaign_id, time.time() - 1, variant="robustness")
    assert first["status"] == second["status"] == "succeeded"
    assert first["benchmark_variant"] == "robustness"
    assert first["adaptive_diagnostic"] is True
    assert len(store.list("macfit-research-" + campaign_id)) == 3
    assert store.queue_size() == 0
    assert [row["job_id"] for row in first["runs"]] == [row["job_id"] for row in second["runs"]]
    for row in second["runs"]:
        assessment = json.loads((tmp_path / "evidence" / row["assessment_file"]).read_text())
        assert assessment["adaptive_diagnostic"] is True
        assert assessment["benchmark"]["training_rows"] == 360
        assert assessment["metrics"]["after"]["structured_exact_rate"] == 1.0
        assert row["fixture_sha256"] == campaign.canonical_sha256(
            build_robustness_input(row["model_id"], row["preset"])
        )


def test_legacy_campaign_without_variant_field_remains_baseline(tmp_path):
    campaign_id = "legacy-baseline"
    project, _ = recipe_identity(campaign_id, *RECIPES[0])
    write_json(
        tmp_path / "evidence" / (campaign_id + ".json"),
        {
            "schema": "macfit-lora-campaign-v1",
            "campaign_id": campaign_id,
            "project_id": project,
            "runs": [],
            "limitations": [],
        },
    )
    state = run_campaign(tmp_path, campaign_id, time.time() - 1)
    assert state["benchmark_variant"] == "baseline"
    assert state["adaptive_diagnostic"] is False


@pytest.mark.parametrize("legacy", [False, True])
def test_cross_variant_resume_rejects_before_submitting_jobs(tmp_path, legacy):
    campaign_id = "different-variant"
    project, _ = recipe_identity(campaign_id, *RECIPES[0])
    state = {
        "schema": "macfit-lora-campaign-v1",
        "campaign_id": campaign_id,
        "project_id": project,
        "runs": [],
        "limitations": [],
    }
    if not legacy:
        state["benchmark_variant"] = "baseline"
    path = tmp_path / "evidence" / (campaign_id + ".json")
    write_json(path, state)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="across benchmark variants"):
        run_campaign(tmp_path, campaign_id, time.time() + 7200, variant="robustness")
    assert path.read_bytes() == before
    store = JobStore(Settings(tmp_path, secrets.token_hex(32)))
    assert store.list("macfit-research-" + campaign_id) == []


def test_unknown_variant_is_rejected_without_campaign_side_effects(tmp_path):
    with pytest.raises(ValueError, match="benchmark variant"):
        run_campaign(tmp_path, "invalid-variant", time.time() + 7200, variant="module.path")
    assert not (tmp_path / "campaigns").exists()


@pytest.mark.parametrize("tamper", ["content", "symlink", "oversize"])
def test_robustness_still_requires_registered_evaluation_artifact_integrity(tmp_path, tamper):
    store = JobStore(Settings(tmp_path, secrets.token_hex(32)))
    job, path = prepare_job(
        store, "robust-tamper", campaign.ROBUSTNESS_RECIPES[0], complete=True, variant="robustness"
    )
    evaluation = path / "artifacts/evaluation.json"
    if tamper == "content":
        evaluation.write_text("{}")
    elif tamper == "symlink":
        target = tmp_path / "changed-evaluation.json"
        target.write_bytes(evaluation.read_bytes())
        evaluation.unlink()
        evaluation.symlink_to(target)
    else:
        with evaluation.open("r+b") as stream:
            stream.truncate(8 * 1024**2 + 1)
    with pytest.raises(ServiceError) as error:
        run_campaign(tmp_path, "robust-tamper", time.time() - 1, variant="robustness")
    assert error.value.code == "artifact_unavailable"
    state = json.loads((tmp_path / "evidence/robust-tamper.json").read_text())
    assert state["status"] == "failed"
    assert state["benchmark_variant"] == "robustness"
    assert store.get(job["id"])["status"] == "succeeded"
