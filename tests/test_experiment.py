from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import amd_inference_opt.experiment as experiment_module
from amd_inference_opt.experiment import (
    ExperimentExecutionError,
    execute_active_experiment,
    freeze_experiment_inputs,
    materialize_q8_source_experiment,
)
from amd_inference_opt.experiment_bundle import (
    ModelInputProvenanceV1,
    model_input_provenance_path,
)
from amd_inference_opt.models import (
    BaselineResult,
    BenchmarkProtocol,
    BenchmarkResult,
    ChangeSet,
    EnvironmentFingerprint,
    ExperimentSpec,
    MCPConfig,
    MetricSeries,
    ModelTarget,
    OptimizationObjective,
    OptimizationTask,
    QualityConstraints,
    RunStatus,
    RuntimeTarget,
    TaskBudgets,
    WorkflowRecord,
    WorkflowStage,
)
from amd_inference_opt.store import ExperimentStore


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_executes_runtime_config_through_deterministic_accept_gate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Tests")
    (repo / "README").write_text("fixture\n")
    _git(repo, "add", "README")
    _git(repo, "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"baseline model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    task = OptimizationTask(
        id="accept-task",
        model=ModelTarget(
            path=model,
            sha256=model_sha256,
            quantization="Q6_K",
        ),
        runtime=RuntimeTarget(repo_path=repo, base_commit=commit),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
        benchmark=BenchmarkProtocol(sample_count=3),
        objective=OptimizationObjective(minimum_improvement_percent=10),
        quality=QualityConstraints(
            max_ppl_regression_percent=1.5,
            max_accuracy_drop_percentage_points=2,
        ),
    )
    environment = EnvironmentFingerprint(
        values={
            "gpu_gfx": "gfx1201",
            "gpu_device": "0",
            "rocm_version": "7.0",
            "runtime_base_commit": commit,
            "model_sha256": model_sha256,
            "build_flags_hash": "flags",
        }
    )
    baseline = BaselineResult(
        environment=environment,
        build_status=RunStatus.SUCCEEDED,
        smoke_passed=True,
        benchmark=BenchmarkResult(
            status=RunStatus.SUCCEEDED,
            metrics={
                "tokens_per_second": MetricSeries(
                    unit="tokens/s", samples=[99, 100, 101]
                )
            },
        ),
    )
    bench = json.dumps(
        [
            {
                "n_prompt": 0,
                "n_gen": 128,
                "avg_ts": 115,
                "samples_ts": [114, 115, 116],
            }
        ]
    )
    quality = json.dumps(
        {
            "baseline": {
                "status": "SUCCEEDED",
                "correctness_passed": True,
                "perplexity": 3.43498,
                "accuracies": {"math_accuracy": 512 / 848},
                "coordinate_hash": "protocol",
            },
            "candidate": {
                "status": "SUCCEEDED",
                "correctness_passed": True,
                "perplexity": 3.48450,
                "accuracies": {"math_accuracy": 499 / 848},
                "coordinate_hash": "protocol",
            },
        }
    )
    spec = ExperimentSpec(
        id="split-k",
        task_id=task.id,
        hypothesis_id="hypothesis",
        change=ChangeSet(
            kind="runtime_config", description="mapping", env={"MAPPING": "split-k"}
        ),
        commands={
            "smoke": [sys.executable, "-c", "print('ok')"],
            "e2e": [sys.executable, "-c", f"print({bench!r})"],
            "quality": [sys.executable, "-c", f"print({quality!r})"],
        },
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    store.save_json(task.id, "artifacts/baseline.json", baseline, producer="test")
    spec_artifact = store.save_json(
        task.id, f"experiments/{spec.id}/spec.json", spec, producer="test"
    )
    store.save_json(
        task.id,
        "state/active-experiment.json",
        {"experiment_id": spec.id, "spec_path": spec_artifact.path},
        producer="test",
    )
    record = WorkflowRecord(
        task_id=task.id,
        current_stage=WorkflowStage.PATCH_AND_BUILD,
        experiment_count=1,
    )
    store.save_workflow(record)

    result = execute_active_experiment(task, record, store)

    assert result["decision"].outcome == "ACCEPT"
    assert result["workflow_status"] == "ACCEPTED"
    assert (store.task_dir(task.id) / "reports" / "final-config.json").is_file()
    assert not (store.task_dir(task.id) / "reports" / "final.patch").exists()
    assert store.artifact_ref(task.id, spec_artifact.path) == spec_artifact
    provenance = store.load_json(
        task.id,
        model_input_provenance_path(spec.id),
        ModelInputProvenanceV1,
    )
    assert provenance.model_sha256 == model_sha256
    assert provenance.packed_bytes == len(b"baseline model")
    assert provenance.quantization == "Q6_K"


def test_build_failure_is_rejected_without_running_later_steps(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Tests")
    (repo / "file").write_text("old\n")
    _git(repo, "add", "file")
    _git(repo, "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    patch = tmp_path / "change.patch"
    patch.write_text(
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new\n"
    )
    model = tmp_path / "model.gguf"
    model.write_bytes(b"source experiment model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    task = OptimizationTask(
        id="reject-task",
        model=ModelTarget(
            path=model,
            sha256=model_sha256,
            quantization="Q6_K",
        ),
        runtime=RuntimeTarget(repo_path=repo, base_commit=commit),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
        budgets=TaskBudgets(max_experiments=1),
    )
    environment = EnvironmentFingerprint(
        values={
            "gpu_gfx": "gfx1201",
            "gpu_device": "0",
            "rocm_version": "unknown",
            "runtime_base_commit": commit,
            "model_sha256": model_sha256,
            "build_flags_hash": "flags",
        }
    )
    baseline = BaselineResult(
        environment=environment,
        build_status=RunStatus.SUCCEEDED,
        smoke_passed=True,
        benchmark=BenchmarkResult(
            status=RunStatus.SUCCEEDED,
            metrics={
                "tokens_per_second": MetricSeries(
                    unit="tokens/s", samples=[99, 100, 101, 100, 100]
                )
            },
        ),
    )
    spec = ExperimentSpec(
        id="bad-build",
        task_id=task.id,
        hypothesis_id="hypothesis",
        change=ChangeSet(
            kind="source_patch", description="bad build", patch_path=patch
        ),
        commands={
            "build": [sys.executable, "-c", "raise SystemExit(2)"],
            "smoke": [sys.executable, "-c", "raise AssertionError('must not run')"],
        },
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    store.save_json(task.id, "artifacts/baseline.json", baseline, producer="test")
    artifact = store.save_json(
        task.id, f"experiments/{spec.id}/spec.json", spec, producer="test"
    )
    store.save_json(
        task.id,
        "state/active-experiment.json",
        {"experiment_id": spec.id, "spec_path": artifact.path},
        producer="test",
    )
    record = WorkflowRecord(
        task_id=task.id,
        current_stage=WorkflowStage.PATCH_AND_BUILD,
        experiment_count=1,
    )
    store.save_workflow(record)

    result = execute_active_experiment(task, record, store)

    assert result["decision"].outcome == "REJECT"
    assert result["workflow_status"] == "REJECTED"
    assert not (store.task_dir(task.id) / "reports" / "final.patch").exists()
    assert store.artifact_ref(task.id, artifact.path) == artifact
    assert store.verify_artifact(task.id, artifact)
    pointer = store.load_json(task.id, "state/active-experiment.json")
    assert pointer["create_experiment_spec_path"] == artifact.path
    assert pointer["spec_path"].endswith("/execution-spec.json")


def test_q8_source_experiment_commands_use_only_worktree_binary(tmp_path: Path) -> None:
    source = tmp_path / "llama.cpp"
    source.mkdir()
    patch = tmp_path / "candidate.patch"
    patch.write_text("fixture\n", encoding="utf-8")
    task = OptimizationTask(
        id="q8-source-materialization",
        campaign_kind="llama_cpp_q8",
        model=ModelTarget(
            path=tmp_path / "model.gguf",
            sha256="a" * 64,
            quantization="Q8_0",
        ),
        runtime=RuntimeTarget(
            repo_path=source,
            base_commit="abc",
            prepared_binary_path=tmp_path / "baseline" / "llama-bench",
            prepared_binary_sha256="b" * 64,
            build_flags=["-DGGML_HIP=ON", "-DAMDGPU_TARGETS=gfx1201"],
        ),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
        metadata={"cmake_path": "/usr/bin/cmake", "build_jobs": "4"},
    )
    spec = ExperimentSpec(
        id="candidate",
        task_id=task.id,
        hypothesis_id="hypothesis",
        change=ChangeSet(
            kind="source_patch",
            description="Q8 kernel change",
            patch_path=patch,
            patch_sha256="c" * 64,
        ),
        commands={
            "e2e": [str(task.runtime.prepared_binary_path), "--wrong-baseline-binary"]
        },
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)

    resolved = materialize_q8_source_experiment(task, spec, store)

    assert resolved.worktree_path is not None
    candidate_binary = resolved.worktree_path / "build-gpuopt/bin/llama-bench"
    assert resolved.stage_commands["e2e"].argv[0] == str(candidate_binary)
    assert resolved.stage_commands["smoke"].argv[0] == str(candidate_binary)
    assert resolved.stage_commands["build"].argv[2] == str(
        resolved.worktree_path / "build-gpuopt"
    )
    assert str(task.runtime.prepared_binary_path) not in resolved.stage_commands["e2e"].argv


def test_q8_model_experiment_reuses_binary_and_binds_candidate_model(
    tmp_path: Path,
) -> None:
    source = tmp_path / "llama.cpp"
    source.mkdir()
    baseline_model = tmp_path / "q8.gguf"
    baseline_model.write_bytes(b"q8")
    candidate_model = tmp_path / "q6.gguf"
    candidate_model.write_bytes(b"q6")
    binary = tmp_path / "build/bin/llama-bench"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")
    candidate_sha = hashlib.sha256(candidate_model.read_bytes()).hexdigest()
    task = OptimizationTask(
        id="q8-model-materialization",
        campaign_kind="llama_cpp_q8",
        model=ModelTarget(
            path=baseline_model,
            sha256="a" * 64,
            quantization="Q8_0",
        ),
        runtime=RuntimeTarget(
            repo_path=source,
            base_commit="abc",
            prepared_binary_path=binary,
            prepared_binary_sha256="b" * 64,
        ),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    spec = ExperimentSpec(
        id="candidate-q6",
        task_id=task.id,
        hypothesis_id="hypothesis",
        change=ChangeSet(
            kind="runtime_config",
            description="Q6 representation",
            candidate_model_path=candidate_model,
            candidate_model_sha256=candidate_sha,
            candidate_model_quantization="Q6_K",
        ),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)

    resolved = materialize_q8_source_experiment(task, spec, store)

    assert "build" not in resolved.stage_commands
    assert resolved.stage_commands["e2e"].argv[0] == str(binary.resolve())
    model_index = resolved.stage_commands["e2e"].argv.index("-m") + 1
    assert resolved.stage_commands["e2e"].argv[model_index] == str(
        candidate_model.resolve()
    )


def test_candidate_model_is_hash_bound_once_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "llama.cpp"
    source.mkdir()
    baseline_model = tmp_path / "q8.gguf"
    baseline_model.write_bytes(b"baseline")
    candidate_model = tmp_path / "q5.gguf"
    candidate_model.write_bytes(b"candidate packed bytes")
    candidate_sha256 = hashlib.sha256(candidate_model.read_bytes()).hexdigest()
    binary = tmp_path / "build/bin/llama-bench"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")
    task = OptimizationTask(
        id="candidate-provenance",
        campaign_kind="llama_cpp_q8",
        model=ModelTarget(
            path=baseline_model,
            sha256=hashlib.sha256(baseline_model.read_bytes()).hexdigest(),
            architecture="qwen",
            quantization="Q8_0",
        ),
        runtime=RuntimeTarget(
            repo_path=source,
            base_commit="abc",
            prepared_binary_path=binary,
            prepared_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        ),
        mcp=MCPConfig(command=["/opt/rocm-agent-mcp"]),
    )
    spec = ExperimentSpec(
        id="candidate-q5",
        task_id=task.id,
        hypothesis_id="hypothesis",
        change=ChangeSet(
            kind="runtime_config",
            description="Q5 representation",
            candidate_model_path=candidate_model,
            candidate_model_sha256=candidate_sha256,
            candidate_model_quantization="Q5_K_M",
        ),
    )
    store = ExperimentStore(tmp_path / "store")
    store.create_task(task)
    spec_ref = store.save_json(
        task.id,
        f"experiments/{spec.id}/spec.json",
        spec,
        producer="test",
    )
    original_reader = experiment_module._read_model_input_once
    reads: list[Path] = []

    def counted_reader(path: Path) -> tuple[str, int]:
        reads.append(path)
        return original_reader(path)

    monkeypatch.setattr(experiment_module, "_read_model_input_once", counted_reader)

    frozen = freeze_experiment_inputs(task, spec, store)
    materialized = materialize_q8_source_experiment(task, frozen, store)

    assert frozen == spec
    assert materialized.stage_commands["e2e"].argv
    assert reads == [candidate_model.resolve()]
    assert store.artifact_ref(task.id, spec_ref.path) == spec_ref
    provenance_path = model_input_provenance_path(spec.id)
    provenance_ref = store.artifact_ref(task.id, provenance_path)
    assert provenance_ref is not None
    assert store.verify_artifact(task.id, provenance_ref)
    provenance = store.load_json(
        task.id,
        provenance_path,
        ModelInputProvenanceV1,
    )
    assert provenance.model_path == str(candidate_model.resolve())
    assert provenance.model_sha256 == candidate_sha256
    assert provenance.packed_bytes == len(b"candidate packed bytes")
    assert provenance.quantization == "Q5_K_M"
    assert provenance.architecture == "qwen"

    candidate_model.write_bytes(b"candidate changed")
    with pytest.raises(ExperimentExecutionError, match="SHA-256 does not match"):
        freeze_experiment_inputs(task, spec, store)
    assert store.artifact_ref(task.id, provenance_path) == provenance_ref
