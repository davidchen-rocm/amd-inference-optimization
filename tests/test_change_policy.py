from pathlib import Path

import pytest

from amd_inference_opt.change_policy import ChangePolicyError, validate_experiment_change
from amd_inference_opt.models import (
    ChangeKind,
    ChangePolicy,
    ChangeSet,
    ExperimentSpec,
    MCPConfig,
    ModelTarget,
    OptimizationTask,
    RuntimeTarget,
)


def task(policy: ChangePolicy) -> OptimizationTask:
    return OptimizationTask(
        id="policy-test",
        model=ModelTarget(path=Path("model.gguf"), quantization="Q6_K"),
        runtime=RuntimeTarget(repo_path=Path("repo"), base_commit="abc"),
        mcp=MCPConfig(command=["mcp"]),
        change_policy=policy,
    )


def spec(change: ChangeSet) -> ExperimentSpec:
    return ExperimentSpec(
        id="exp", task_id="policy-test", hypothesis_id="hyp", change=change
    )


def test_locked_model_rejects_representation_change() -> None:
    selected = task(
        ChangePolicy(
            allowed_change_kinds=[ChangeKind.RUNTIME_CONFIG],
            locked_coordinates=["model_sha256", "quantization"],
        )
    )
    change = ChangeSet(
        kind=ChangeKind.RUNTIME_CONFIG,
        description="switch model",
        candidate_model_path=Path("q4.gguf"),
        candidate_model_sha256="a" * 64,
        candidate_model_quantization="Q4_K_M",
    )
    with pytest.raises(ChangePolicyError, match="locked coordinates"):
        validate_experiment_change(selected, spec(change))


def test_kv_runtime_args_must_be_allow_listed() -> None:
    selected = task(
        ChangePolicy(
            allowed_change_kinds=[ChangeKind.RUNTIME_CONFIG],
            allowed_runtime_args=["-ctk", "-ctv", "-fa"],
        )
    )
    allowed = ChangeSet(
        kind=ChangeKind.RUNTIME_CONFIG,
        description="Q8 KV",
        runtime_args=["-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on"],
    )
    validate_experiment_change(selected, spec(allowed))
    forbidden = allowed.model_copy(
        update={"runtime_args": ["-ctk", "q8_0", "--override-tensor", "CPU"]}
    )
    with pytest.raises(ChangePolicyError, match="outside"):
        validate_experiment_change(selected, spec(forbidden))


def test_locked_binary_rejects_source_patch() -> None:
    selected = task(
        ChangePolicy(
            allowed_change_kinds=[ChangeKind.SOURCE_PATCH],
            locked_coordinates=["binary_sha256"],
        )
    )
    change = ChangeSet(
        kind=ChangeKind.SOURCE_PATCH,
        description="patch",
        patch_path=Path("patch.diff"),
    )
    with pytest.raises(ChangePolicyError, match="binary_sha256"):
        validate_experiment_change(selected, spec(change))


def test_vllm_boolean_runtime_switch_uses_declared_zero_arity() -> None:
    selected = task(
        ChangePolicy(
            allowed_change_kinds=[ChangeKind.RUNTIME_CONFIG],
            allowed_runtime_args=["--enforce-eager", "--max-num-seqs"],
            runtime_arg_arity={"--enforce-eager": 0},
        )
    )
    change = ChangeSet(
        kind=ChangeKind.RUNTIME_CONFIG,
        description="disable graph capture for a strict A/B",
        runtime_args=["--enforce-eager", "--max-num-seqs", "64"],
    )
    validate_experiment_change(selected, spec(change))


def test_runtime_argument_arity_rejects_missing_or_duplicate_values() -> None:
    selected = task(
        ChangePolicy(
            allowed_change_kinds=[ChangeKind.RUNTIME_CONFIG],
            allowed_runtime_args=["--enforce-eager", "--max-num-seqs"],
            runtime_arg_arity={"--enforce-eager": 0},
        )
    )
    missing = ChangeSet(
        kind=ChangeKind.RUNTIME_CONFIG,
        description="missing value",
        runtime_args=["--max-num-seqs"],
    )
    with pytest.raises(ChangePolicyError, match="requires a value"):
        validate_experiment_change(selected, spec(missing))

    duplicate = ChangeSet(
        kind=ChangeKind.RUNTIME_CONFIG,
        description="duplicate switch",
        runtime_args=["--enforce-eager", "--enforce-eager"],
    )
    with pytest.raises(ChangePolicyError, match="cannot repeat"):
        validate_experiment_change(selected, spec(duplicate))


def test_runtime_argument_arity_must_reference_an_allow_listed_flag() -> None:
    with pytest.raises(ValueError, match="must also be allow-listed"):
        ChangePolicy(
            allowed_change_kinds=[ChangeKind.RUNTIME_CONFIG],
            allowed_runtime_args=["--max-num-seqs"],
            runtime_arg_arity={"--enforce-eager": 0},
        )
