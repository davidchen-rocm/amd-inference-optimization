from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from amd_inference_opt.experiment_bundle import (
    BundleError,
    ExperimentAttemptIndex,
    ExperimentCatalog,
    ModelInputProvenanceV1,
    load_experiment_bundle,
    model_input_provenance_path,
    refresh_experiment_bundle,
    refresh_task_bundles,
)
from amd_inference_opt.models import (
    GPUTarget,
    MCPConfig,
    ModelTarget,
    OptimizationTask,
    QualityConstraints,
    RuntimeTarget,
)
from amd_inference_opt.store import ExperimentStore


def _make_store(tmp_path: Path) -> tuple[ExperimentStore, OptimizationTask]:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"packed-model")
    task = OptimizationTask(
        id="bundle-task",
        model=ModelTarget(
            path=model,
            sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            architecture="qwen",
            quantization="Q6_K",
        ),
        runtime=RuntimeTarget(
            repo_path=tmp_path / "llama.cpp",
            base_commit="abc123",
        ),
        gpu=GPUTarget(gfx_target="gfx1201", name="Radeon RX 9070 XT"),
        quality=QualityConstraints(
            max_ppl_regression_percent=2.0,
            max_accuracy_drop_percentage_points=2.0,
            accuracy_metric="math_accuracy",
        ),
        mcp=MCPConfig(command=["rocm-agent-mcp"]),
        metadata={"model_packed_bytes": str(len(b"packed-model"))},
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    return store, task


def _seed_completed_legacy_experiment(
    store: ExperimentStore,
    task: OptimizationTask,
    experiment_id: str = "legacy-split-k",
) -> Path:
    experiment = store.task_dir(task.id) / "experiments" / experiment_id
    experiment.mkdir()
    patch = experiment.parent.parent / "legacy-source.patch"
    patch.write_text("diff --git a/kernel.cu b/kernel.cu\n", encoding="utf-8")
    store.save_json(
        task.id,
        f"experiments/{experiment_id}/spec.json",
        {
            "hypothesis_id": "hypothesis-split-k",
            "change": {
                "kind": "source_patch",
                "description": "increase decode wave supply",
                "patch_path": str(patch),
                "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            },
        },
        producer="legacy-workflow",
    )
    store.save_json(
        task.id,
        "artifacts/baseline.json",
        {
            "benchmark": {
                "metrics": {
                    "tg128": {"unit": "tokens/s", "samples": [99.0, 100.0, 101.0]},
                    "tg512": {"unit": "tokens/s", "samples": [100.0, 100.0, 100.0]},
                }
            },
            "quality": {
                "status": "SUCCEEDED",
                "correctness_passed": True,
                "perplexity": 2.0,
                "accuracies": {"math_accuracy": 0.8},
            },
        },
        producer="legacy-workflow",
    )
    store.save_json(
        task.id,
        f"experiments/{experiment_id}/e2e-result.json",
        {
            "metrics": {
                "tg128": {"unit": "tokens/s", "samples": [110.0, 111.0, 112.0]},
                "tg512": {"unit": "tokens/s", "samples": [104.0, 105.0, 106.0]},
            }
        },
        producer="legacy-runner",
    )
    store.save_json(
        task.id,
        f"experiments/{experiment_id}/quality-result.json",
        {
            "status": "SUCCEEDED",
            "correctness_passed": True,
            "perplexity": 2.02,
            "accuracies": {"math_accuracy": 0.79},
        },
        producer="legacy-quality",
    )
    store.save_json(
        task.id,
        f"experiments/{experiment_id}/gate-decision.json",
        {
            "outcome": "ACCEPT",
            "reasons": ["performance and quality gates passed"],
            "metric_improvements_percent": {"tg128": 11.0, "tg512": 5.0},
        },
        producer="legacy-gate",
    )

    # Old runners wrote these files directly, before every output was registered.
    runner = experiment / "runner"
    runner.mkdir()
    (runner / "runner-result.json").write_text(
        json.dumps({"status": "SUCCEEDED"}), encoding="utf-8"
    )
    (runner / "stdout.log").write_text("legacy runner output\n", encoding="utf-8")

    # Neither an old in-experiment worktree nor the new task-level workspace may
    # become bundle evidence.
    old_worktree = runner / "candidate-worktree"
    old_worktree.mkdir()
    (old_worktree / "candidate.o").write_bytes(b"large build output")
    workspace = store.task_dir(task.id) / "workspaces" / experiment_id
    workspace.mkdir()
    (workspace / "candidate.o").write_bytes(b"task workspace output")
    return experiment


def test_refresh_backfills_legacy_outputs_and_builds_complete_summary(tmp_path: Path) -> None:
    store, task = _make_store(tmp_path)
    experiment = _seed_completed_legacy_experiment(store, task)

    manifests = refresh_task_bundles(store, task.id)

    assert len(manifests) == 1
    summary, manifest = load_experiment_bundle(store, task.id, "legacy-split-k")
    assert summary.status == "ACCEPT"
    assert summary.strategy == "source_patch"
    assert summary.hypothesis_id == "hypothesis-split-k"
    assert summary.change_summary == "increase decode wave supply"
    assert summary.model.quantization == "Q6_K"
    assert summary.model.packed_bytes == len(b"packed-model")

    metrics = {metric.name: metric for metric in summary.metrics}
    assert metrics["tg128"].baseline_mean == pytest.approx(100.0)
    assert metrics["tg128"].candidate_mean == pytest.approx(111.0)
    assert metrics["tg128"].delta_percent == pytest.approx(11.0)
    assert metrics["tg128"].baseline_sample_count == 3
    assert metrics["tg128"].candidate_sample_count == 3
    assert metrics["tg512"].delta_percent == pytest.approx(5.0)

    assert summary.quality is not None
    assert summary.quality.status == "SUCCEEDED"
    assert summary.quality.correctness_passed is True
    assert summary.quality.perplexity_regression_percent == pytest.approx(1.0)
    assert summary.quality.baseline_accuracy == pytest.approx(0.8)
    assert summary.quality.candidate_accuracy == pytest.approx(0.79)
    assert summary.quality.accuracy_drop_percentage_points == pytest.approx(1.0)
    assert summary.decision is not None
    assert summary.decision.outcome == "ACCEPT"
    assert summary.decision.final is True
    assert summary.decision.reasons == ["performance and quality gates passed"]
    assert summary.selected_attempts == {"runner": "initial"}

    paths = {entry.artifact.path for entry in manifest.artifacts}
    assert "experiments/legacy-split-k/runner/runner-result.json" in paths
    assert "experiments/legacy-split-k/runner/stdout.log" in paths
    assert not any("candidate-worktree" in path for path in paths)
    assert not any(path.startswith("workspaces/") for path in paths)
    assert not store.artifact_ref(
        task.id, "experiments/legacy-split-k/runner/candidate-worktree/candidate.o"
    )
    assert not store.artifact_ref(
        task.id, "workspaces/legacy-split-k/candidate.o"
    )
    assert experiment.joinpath("summary.json").is_file()
    assert experiment.joinpath("attempts/index.json").is_file()
    assert experiment.joinpath("manifest.json").is_file()


def test_bundle_manifest_is_hash_bound_and_refresh_is_idempotent(tmp_path: Path) -> None:
    store, task = _make_store(tmp_path)
    experiment = _seed_completed_legacy_experiment(store, task)
    first = refresh_experiment_bundle(store, task.id, "legacy-split-k")
    generated_paths = [
        experiment / "summary.json",
        experiment / "attempts/index.json",
        experiment / "manifest.json",
    ]
    first_bytes = {path: path.read_bytes() for path in generated_paths}
    first_mtimes = {path: path.stat().st_mtime_ns for path in generated_paths}

    second = refresh_experiment_bundle(store, task.id, "legacy-split-k")

    assert second == first
    assert {path: path.read_bytes() for path in generated_paths} == first_bytes
    assert {path: path.stat().st_mtime_ns for path in generated_paths} == first_mtimes
    assert hashlib.sha256((experiment / "summary.json").read_bytes()).hexdigest() == (
        second.summary.sha256
    )
    for entry in second.artifacts:
        assert store.verify_artifact(task.id, entry.artifact)
    manifest_ref = store.artifact_ref(
        task.id, "experiments/legacy-split-k/manifest.json"
    )
    assert manifest_ref is not None
    assert store.verify_artifact(task.id, manifest_ref)


def test_catalog_contains_experiment_metric_and_quality_rows(tmp_path: Path) -> None:
    store, task = _make_store(tmp_path)
    _seed_completed_legacy_experiment(store, task)
    refresh_experiment_bundle(store, task.id, "legacy-split-k")

    rows = ExperimentCatalog(store.root).rows(task_id=task.id)

    assert len(rows) == 1
    assert rows[0]["task_id"] == task.id
    assert rows[0]["experiment_id"] == "legacy-split-k"
    assert rows[0]["status"] == "ACCEPT"
    assert rows[0]["quantization"] == "Q6_K"
    assert rows[0]["strategy"] == "source_patch"
    with sqlite3.connect(store.root / "catalog.sqlite3") as connection:
        metric = connection.execute(
            """
            SELECT baseline_mean, candidate_mean, delta_percent
            FROM metrics
            WHERE task_id=? AND experiment_id=? AND name='tg128'
            """,
            (task.id, "legacy-split-k"),
        ).fetchone()
        quality = connection.execute(
            """
            SELECT baseline_perplexity, candidate_perplexity, baseline_accuracy,
                   candidate_accuracy
            FROM quality_summary
            WHERE task_id=? AND experiment_id=?
            """,
            (task.id, "legacy-split-k"),
        ).fetchone()
    assert metric == pytest.approx((100.0, 111.0, 11.0))
    assert quality == pytest.approx((2.0, 2.02, 0.8, 0.79))


def test_load_rejects_tampered_summary(tmp_path: Path) -> None:
    store, task = _make_store(tmp_path)
    experiment = _seed_completed_legacy_experiment(store, task)
    refresh_experiment_bundle(store, task.id, "legacy-split-k")
    summary_path = experiment / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "REJECT"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(BundleError, match="summary is not hash-bound"):
        load_experiment_bundle(store, task.id, "legacy-split-k")


def test_new_task_has_dedicated_workspaces_directory(tmp_path: Path) -> None:
    store, task = _make_store(tmp_path)

    workspaces = store.task_dir(task.id) / "workspaces"

    assert workspaces.is_dir()
    assert workspaces.parent == store.task_dir(task.id)


def test_attempt_current_selection_is_scoped_by_kind_and_strips_json_suffix(
    tmp_path: Path,
) -> None:
    store, task = _make_store(tmp_path)
    _seed_completed_legacy_experiment(store, task)
    prefix = "experiments/legacy-split-k"
    for attempt_id, outcome in (
        ("attempt-0001", "INCONCLUSIVE"),
        ("attempt-0002", "ACCEPT"),
    ):
        root = f"{prefix}/e2e-reruns/{attempt_id}"
        store.save_json(
            task.id,
            f"{root}/performance-pre-gate.json",
            {"outcome": outcome},
            producer="test",
        )
        store.save_text(
            task.id,
            f"{root}/run.log",
            attempt_id,
            producer="test",
        )

    pair_root = f"{prefix}/extended-pair-verification/attempt-0001"
    store.save_json(
        task.id,
        f"{pair_root}/baseline-result.json",
        {
            "benchmark": {
                "metrics": {"tg128": {"unit": "tokens/s", "samples": [100.0]}}
            }
        },
        producer="test",
    )
    store.save_json(
        task.id,
        f"{pair_root}/candidate-e2e-result.json",
        {"metrics": {"tg128": {"unit": "tokens/s", "samples": [110.0]}}},
        producer="test",
    )
    store.save_json(
        task.id,
        f"{pair_root}/performance-pre-gate.json",
        {"outcome": "ACCEPT", "metric_improvements_percent": {"tg128": 10.0}},
        producer="test",
    )
    store.save_text(
        task.id,
        f"{pair_root}/run.log",
        "paired",
        producer="test",
    )
    quality_path = f"{prefix}/quality/attempts/attempt-0001.json"
    store.save_json(
        task.id,
        quality_path,
        {"attempt_id": "attempt-0001", "state": "COMPLETED"},
        producer="test",
    )

    manifest = refresh_experiment_bundle(store, task.id, "legacy-split-k")
    attempts = store.load_json(
        task.id,
        f"{prefix}/attempts/index.json",
        ExperimentAttemptIndex,
    )

    assert attempts.selected["e2e-rerun"] == "attempt-0002"
    assert attempts.selected["extended-pair"] == "attempt-0001"
    assert attempts.selected["quality"] == "attempt-0001"
    assert all(not attempt.attempt_id.endswith(".json") for attempt in attempts.attempts)
    quality = next(attempt for attempt in attempts.attempts if attempt.kind == "quality")
    assert [artifact.path for artifact in quality.evidence] == [quality_path]

    entries = {entry.artifact.path: entry for entry in manifest.artifacts}
    assert entries[f"{prefix}/e2e-reruns/attempt-0001/run.log"].current is False
    assert entries[f"{prefix}/e2e-reruns/attempt-0002/run.log"].current is True
    assert entries[f"{pair_root}/run.log"].current is True
    assert entries[quality_path].attempt_id == "attempt-0001"
    assert entries[quality_path].current is True


def test_distinct_candidate_size_requires_hash_bound_model_provenance(
    tmp_path: Path,
) -> None:
    store, task = _make_store(tmp_path)
    _seed_completed_legacy_experiment(store, task)
    experiment_id = "legacy-split-k"
    candidate = tmp_path / "candidate.gguf"
    candidate_sha256 = "c" * 64
    store.save_json(
        task.id,
        f"experiments/{experiment_id}/spec.json",
        {
            "hypothesis_id": "candidate-model",
            "change": {
                "kind": "runtime_config",
                "description": "use a packed candidate",
                "candidate_model_path": str(candidate),
                "candidate_model_sha256": candidate_sha256,
                "candidate_model_quantization": "Q5_K_M",
            },
        },
        producer="test",
    )

    refresh_experiment_bundle(store, task.id, experiment_id)
    summary, _ = load_experiment_bundle(store, task.id, experiment_id)
    assert summary.model.sha256 == candidate_sha256
    assert summary.model.packed_bytes is None
    assert summary.model.provenance_artifact is None

    provenance = ModelInputProvenanceV1(
        model_path=str(candidate),
        model_sha256=candidate_sha256,
        packed_bytes=123_456,
        effective_bpw=5.25,
        quantization="Q5_K_M",
        architecture="qwen",
        source_model_path=str(task.model.path),
        source_model_sha256=task.model.sha256,
    )
    provenance_ref = store.save_json(
        task.id,
        model_input_provenance_path(experiment_id),
        provenance,
        producer="test",
    )

    refresh_experiment_bundle(store, task.id, experiment_id)
    summary, manifest = load_experiment_bundle(store, task.id, experiment_id)
    assert summary.model.packed_bytes == 123_456
    assert summary.model.effective_bpw == pytest.approx(5.25)
    assert summary.model.provenance_artifact == provenance_ref
    assert provenance_ref.path in {entry.artifact.path for entry in manifest.artifacts}


def test_candidate_model_provenance_must_match_spec_coordinates(tmp_path: Path) -> None:
    store, task = _make_store(tmp_path)
    _seed_completed_legacy_experiment(store, task)
    experiment_id = "legacy-split-k"
    candidate = tmp_path / "candidate.gguf"
    store.save_json(
        task.id,
        f"experiments/{experiment_id}/spec.json",
        {
            "change": {
                "kind": "runtime_config",
                "description": "use a packed candidate",
                "candidate_model_path": str(candidate),
                "candidate_model_sha256": "c" * 64,
                "candidate_model_quantization": "Q5_K_M",
            }
        },
        producer="test",
    )
    store.save_json(
        task.id,
        model_input_provenance_path(experiment_id),
        ModelInputProvenanceV1(
            model_path=str(candidate),
            model_sha256="d" * 64,
            packed_bytes=123_456,
            quantization="Q5_K_M",
        ),
        producer="test",
    )

    with pytest.raises(BundleError, match="SHA-256 differs"):
        refresh_experiment_bundle(store, task.id, experiment_id)
