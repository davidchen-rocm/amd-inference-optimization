"""Trusted local administrator campaign, supervised by the existing training queue.

No HTTP route, Firebase bypass, remote shell, or GPU import is exposed here.
The caller must already have filesystem access to the private service data.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import signal
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import canonical_sha256, write_json
from .config import validate_job_input
from .experiments import assess_evaluation, build_input
from .robustness import assess_evaluation as assess_robustness
from .robustness import build_input as build_robustness_input
from .service.outputs import artifact_handle
from .service.settings import Settings
from .service.store import ACTIVE, JobStore, ServiceError

RECIPES = (
    ("qwen3-0-6b", "quick"),
    ("qwen3-0-6b", "standard"),
    ("qwen3-4b", "quick"),
    ("qwen3-4b", "standard"),
    ("qwen3-8b", "quick"),
    ("qwen3-8b", "standard"),
)
ROBUSTNESS_RECIPES = tuple(recipe for recipe in RECIPES if recipe[1] == "standard")
VARIANTS = {
    "baseline": (RECIPES, build_input, assess_evaluation),
    "robustness": (ROBUSTNESS_RECIPES, build_robustness_input, assess_robustness),
}


def variant_protocol(variant: str):
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError("Choose the baseline or robustness benchmark variant.")
    return VARIANTS[variant]


def instant(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("The deadline requires a timezone.")
    return parsed.timestamp()


def recipe_identity(campaign_id: str, model: str, preset: str) -> tuple[str, str]:
    if (
        not campaign_id
        or len(campaign_id) > 80
        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in campaign_id)
    ):
        raise ValueError("Choose a short lowercase campaign identifier.")
    if (model, preset) not in RECIPES:
        raise ValueError("Unsupported campaign recipe.")
    project = str(uuid.uuid5(uuid.NAMESPACE_URL, "macfit:research:" + campaign_id))
    request = str(uuid.uuid5(uuid.UUID(project), model + ":" + preset))
    return project, request


def _run_campaign(
    data: Path,
    campaign_id: str,
    deadline: float,
    *,
    poll_seconds: float = 5,
    variant: str = "baseline",
) -> dict:
    recipes, input_builder, evaluator = variant_protocol(variant)
    if poll_seconds <= 0 or not 0 < deadline:
        raise ValueError("Invalid campaign time bounds.")
    project, _ = recipe_identity(campaign_id, *RECIPES[0])
    owner = "macfit-research-" + campaign_id
    # This secret is never persisted or used by an HTTP server. The campaign uses
    # the operator's existing Unix permissions, and preserves normal queue limits.
    settings = Settings(
        data,
        secrets.token_hex(32),
        gpu_deadline=deadline,
        stop_accepting_at=deadline - 45 * 60,
    )
    store = JobStore(settings)
    output = store.root / "evidence" / (campaign_id + ".json")
    prior = json.loads(output.read_text()) if output.exists() else None
    if prior and (prior.get("campaign_id") != campaign_id or prior.get("project_id") != project):
        raise ValueError("Existing evidence belongs to a different campaign.")
    if prior and prior.get("benchmark_variant", "baseline") != variant:
        raise ValueError("Cannot resume a campaign across benchmark variants; choose a new ID.")
    state = prior or {
        "schema": "macfit-lora-campaign-v1",
        "campaign_id": campaign_id,
        "project_id": project,
        "started_at": datetime.now(UTC).isoformat(),
        "runs": [],
        "limitations": [
            "Synthetic reference facts and one fixed seed; not general model quality.",
            "Local administrator research jobs are not user-account submissions.",
        ],
    }
    state["benchmark_variant"] = variant
    state["adaptive_diagnostic"] = variant == "robustness"
    if variant == "robustness":
        adaptive_note = (
            "Designed after baseline errors; the original held-out evaluation is reused."
        )
        if adaptive_note not in state["limitations"]:
            state["limitations"].append(adaptive_note)
    state["status"] = "running"
    cancelled = False
    active_id = None

    def stop(_signum, _frame):
        nonlocal cancelled
        cancelled = True

    prior_handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}

    def publish():
        state["updated_at"] = datetime.now(UTC).isoformat()
        write_json(output, state)

    try:
        for model, preset in recipes:
            name = model + "-" + preset
            project, request = recipe_identity(campaign_id, model, preset)
            existing = store.list(owner, request_id=request)
            if not existing and (cancelled or not settings.accepts_new()):
                state["status"] = "deadline_closed" if not cancelled else "interrupted"
                break
            fixture = input_builder(model, preset)
            validated = validate_job_input("training", fixture)
            try:
                job, _ = store.create(owner, request, project, "training", validated)
            except ServiceError as error:
                state["status"] = "admission_refused"
                state["admission_error"] = error.code
                break
            # Capture the job before processing prior evidence so an unexpected
            # coordinator failure cannot leave a newly queued GPU job behind.
            active_id = job["id"]
            row = next((r for r in state["runs"] if r["recipe"] == name), None)
            if row is None:
                row = {
                    "recipe": name,
                    "job_id": job["id"],
                    "fixture_sha256": canonical_sha256(fixture),
                    "model_id": model,
                    "preset": preset,
                }
                state["runs"].append(row)
            elif row["job_id"] != job["id"]:
                raise ValueError("Saved campaign evidence references a different job.")
            while True:
                job = store.get(active_id, owner)
                row.update(status=job["status"], stage=job["stage"], progress=job["progress"])
                publish()
                if job["status"] not in ACTIVE:
                    break
                if cancelled or time.time() >= deadline - 30:
                    store.cancel(active_id, owner)
                    row["status"] = "cancellation_requested"
                    state["status"] = "interrupted" if cancelled else "deadline_closed"
                    state["finished_at"] = datetime.now(UTC).isoformat()
                    publish()
                    return state
                time.sleep(poll_seconds)
            active_id = None
            if job["status"] == "succeeded":
                declared = [
                    artifact
                    for artifact in job["artifacts"]
                    if artifact["name"] == "evaluation.json" and artifact["type"] == "evaluation"
                ]
                if (
                    len(declared) != 1
                    or type(declared[0]["size_bytes"]) is not int
                    or not 0 < declared[0]["size_bytes"] <= settings.max_result_bytes
                ):
                    raise ValueError("A completed job needs one bounded registered evaluation.")
                evaluation_settings = replace(
                    settings, max_artifact_bytes=settings.max_result_bytes
                )
                with artifact_handle(
                    store.directory(job["id"]), declared[0], evaluation_settings
                ) as handle:
                    content = handle.read(settings.max_result_bytes + 1)
                if (
                    len(content) > settings.max_result_bytes
                    or hashlib.sha256(content).hexdigest() != declared[0]["sha256"]
                ):
                    raise ValueError("Evaluation content changed during its verified read.")
                evaluation = json.loads(content)
                assessment = evaluator(evaluation, fixture)
                assessment_name = "policy-json-" + campaign_id + "-" + name + ".json"
                assessment_path = store.root / "evidence" / assessment_name
                write_json(assessment_path, assessment)
                row.update(
                    metrics=assessment["metrics"],
                    training=job["result"]["training"],
                    assessment_file=assessment_name,
                    assessment_sha256=canonical_sha256(assessment),
                    artifacts=[
                        {k: a[k] for k in ("id", "name", "sha256", "size_bytes")}
                        for a in job["artifacts"]
                    ],
                )
            else:
                row["error"] = job["error"]
            publish()
        else:
            state["status"] = (
                "succeeded"
                if all(r["status"] == "succeeded" for r in state["runs"])
                else "completed_with_failures"
            )
        state["finished_at"] = datetime.now(UTC).isoformat()
        publish()
        return state
    except BaseException as error:
        state["status"] = "failed"
        state["failure"] = {
            "code": "coordinator_failed",
            "exception_type": type(error).__name__[:80],
        }
        state["finished_at"] = datetime.now(UTC).isoformat()
        if active_id:
            try:
                stopped = store.cancel(active_id, owner)
                row = next((r for r in state["runs"] if r["job_id"] == active_id), None)
                if row is not None:
                    row.update(status=stopped["status"], stage=stopped["stage"])
            except Exception as cleanup_error:
                state["failure"]["cancellation_exception_type"] = type(cleanup_error).__name__[:80]
        try:
            publish()
        except Exception:
            # A full/unwritable filesystem cannot publish diagnostics. Cancellation
            # still happens above, and the original error remains the process result.
            pass
        raise
    finally:
        if active_id and cancelled:
            try:
                store.cancel(active_id, owner)
            except Exception:
                pass  # Never mask an earlier coordinator failure or handler restoration.
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def run_campaign(
    data: Path,
    campaign_id: str,
    deadline: float,
    *,
    poll_seconds: float = 5,
    variant: str = "baseline",
):
    """A host-local campaign lease prevents duplicate coordinators, without holding the GPU."""
    recipe_identity(campaign_id, *RECIPES[0])
    variant_protocol(variant)
    directory = Path(data).resolve() / "campaigns"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(directory / (campaign_id + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This campaign already has a running coordinator.") from error
        return _run_campaign(
            data, campaign_id, deadline, poll_seconds=poll_seconds, variant=variant
        )
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="baseline")
    args = parser.parse_args(argv)
    result = run_campaign(
        args.data_dir, args.campaign_id, instant(args.stop_at), variant=args.variant
    )
    print(json.dumps({"campaign_id": result["campaign_id"], "status": result["status"]}))
    return 0 if result["status"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
