from pathlib import Path

import pytest
from pydantic import ValidationError

from amd_inference_opt.models import (
    ArtifactRef,
    CampaignKind,
    ChangeSet,
    MCPConfig,
    MetricSeries,
    ModelTarget,
    OptimizationTask,
    RuntimeTarget,
    StageCommandSpec,
    VLLMServingWorkloadConfig,
)


def make_task(tmp_path: Path, **overrides: object) -> OptimizationTask:
    values = {
        "id": "task-1",
        "model": ModelTarget(path=tmp_path / "model.gguf", sha256="a" * 64),
        "runtime": RuntimeTarget(repo_path=tmp_path / "llama.cpp", base_commit="abc123"),
        "mcp": MCPConfig(command=["/opt/rocm-agent-mcp"]),
    }
    values.update(overrides)
    return OptimizationTask(**values)


def test_task_round_trip_is_strict_and_versioned(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    restored = OptimizationTask.model_validate_json(task.model_dump_json())

    assert restored == task
    assert restored.schema_version == 1
    assert restored.campaign_kind == CampaignKind.GENERIC
    assert restored.gpu.gfx_target == "gfx1201"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        OptimizationTask.model_validate({**task.model_dump(), "surprise": True})


def test_campaign_kind_is_typed_and_legacy_q4_metadata_is_migrated(tmp_path: Path) -> None:
    q8 = make_task(tmp_path, campaign_kind="llama_cpp_q8")
    legacy_q4 = make_task(tmp_path, metadata={"live_q4rdna": "true"})

    assert q8.campaign_kind == CampaignKind.LLAMA_CPP_Q8
    assert legacy_q4.campaign_kind == CampaignKind.Q4_RDNA


def test_hashes_and_persisted_paths_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="SHA-256"):
        ModelTarget(path=tmp_path / "model.gguf", sha256="not-a-hash")
    with pytest.raises(ValidationError, match="task-relative"):
        ArtifactRef(path="../escape", sha256="0" * 64, size=0, producer="test")


def test_change_set_enforces_kind_specific_input(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="patch_path"):
        ChangeSet(kind="source_patch", description="missing")
    source = ChangeSet(
        kind="source_patch", description="kernel change", patch_path=tmp_path / "x.patch"
    )
    config = ChangeSet(kind="runtime_config", description="mapping", env={"MAP": "old"})
    unset = ChangeSet(
        kind="runtime_config", description="default mapping", unset_env=["MAP"]
    )

    assert source.require_rebuild is True
    assert config.require_rebuild is False
    assert unset.unset_env == ["MAP"]

    model_change = ChangeSet(
        kind="runtime_config",
        description="compare a smaller representation",
        candidate_model_path=tmp_path / "candidate.gguf",
        candidate_model_sha256="b" * 64,
        candidate_model_quantization="Q6_K",
    )
    assert model_change.require_rebuild is False
    with pytest.raises(ValidationError, match="path, SHA-256, and quantization"):
        ChangeSet(
            kind="runtime_config",
            description="incomplete model identity",
            candidate_model_path=tmp_path / "candidate.gguf",
        )


def test_decode_defaults_and_optional_telemetry_match_mvp(tmp_path: Path) -> None:
    task = make_task(tmp_path)

    assert task.workload.prompt_tokens == 0
    assert task.environment.require_stable_telemetry is False


def test_online_serving_workload_records_real_request_shape() -> None:
    workload = VLLMServingWorkloadConfig(
        input_tokens=1024,
        output_tokens=256,
        num_prompts=100,
        concurrency=8,
        seed=20260823,
    )

    assert workload.kind == "online_serving"
    assert workload.concurrency == 8
    with pytest.raises(ValidationError, match="at least concurrency"):
        VLLMServingWorkloadConfig(
            input_tokens=1024,
            output_tokens=256,
            num_prompts=4,
            concurrency=8,
        )


def test_metric_series_computes_statistics_and_rejects_nan() -> None:
    series = MetricSeries(unit="tokens/s", samples=[99.0, 100.0, 101.0])

    assert series.mean == pytest.approx(100.0)
    assert series.cv_percent == pytest.approx(1.0)
    with pytest.raises(ValidationError, match="finite"):
        MetricSeries(unit="tokens/s", samples=[float("nan")])


def test_stage_command_spec_carries_exact_process_coordinates() -> None:
    command = StageCommandSpec(
        argv=["./llama-bench", "-n", "128"],
        cwd=Path("build/bin"),
        env={"HSA_VISIBLE_DEVICES": "0"},
        unset_env=["LLAMA_Q4_RDNA_MAPPING"],
        timeout_seconds=900,
    )

    assert command.cwd == Path("build/bin")
    assert command.unset_env == ["LLAMA_Q4_RDNA_MAPPING"]
    with pytest.raises(ValidationError, match="within the worktree"):
        StageCommandSpec(argv=["true"], cwd=Path("../outside"))
    with pytest.raises(ValidationError, match="both set and unset"):
        StageCommandSpec(argv=["true"], env={"X": "1"}, unset_env=["X"])
