import json
import sys
from pathlib import Path

import pytest

from amd_inference_opt.models import RunStatus
from amd_inference_opt.quality import (
    Q4ThreeWayQualityProtocol,
    Q8RuntimeQualityProtocol,
    QualityAdapterError,
    normalize_q4_threeway_quality,
    normalize_q8_runtime_quality,
)


def _raw_quality(*, status: str = "complete") -> dict[str, object]:
    return {
        "status": status,
        "protocol": {"hash": "a" * 64},
        "q4_k_m": {
            "perplexity": {"perplexity": 3.43498},
            "math": {"correct": 512, "total": 848, "accuracy": 512 / 848},
        },
        "q4_rdna": {
            "perplexity": {"perplexity": 3.4844986109449088},
            "math": {"correct": 499, "total": 848, "accuracy": 499 / 848},
        },
    }


def test_threeway_protocol_uses_isolated_python_dependency_path_and_new_output(
    tmp_path: Path,
) -> None:
    root = tmp_path / "math-rule-loop"
    script = root / "tools/q4rdna_threeway_quality_eval.py"
    dependencies = root / "tools/kernel-anvil-deps"
    hf_model = root / "models/hf"
    q4_model = root / "models/model.gguf"
    script.parent.mkdir(parents=True)
    dependencies.mkdir(parents=True)
    hf_model.mkdir(parents=True)
    script.write_text("print('unused')\n", encoding="utf-8")
    q4_model.write_bytes(b"GGUF")
    output = root / "artifacts/task-owned/quality.json"

    protocol = Q4ThreeWayQualityProtocol(
        math_rule_loop_dir=str(root),
        python_path=sys.executable,
        hf_model_path=str(hf_model),
        q4_k_model_path=str(q4_model),
        output_path=str(output),
    )
    command = protocol.command()

    assert command.cwd == str(root.resolve())
    assert command.env == {"PYTHONPATH": str(dependencies.resolve())}
    assert command.timeout_seconds == 14400
    assert command.argv[command.argv.index("--ppl-tokens") + 1] == "16384"
    assert command.argv[command.argv.index("--output") + 1] == str(output.resolve())
    assert "--full-math" in command.argv


def test_normalizes_complete_threeway_result_to_gate_quality_models(tmp_path: Path) -> None:
    raw_path = tmp_path / "quality.json"
    raw_path.write_text(json.dumps(_raw_quality()), encoding="utf-8")

    pair = normalize_q4_threeway_quality(
        raw_path,
        baseline_representation_hash="q4-k",
        candidate_representation_hash="q4-rdna",
    )

    assert pair.baseline.status == RunStatus.SUCCEEDED
    assert pair.candidate.status == RunStatus.SUCCEEDED
    assert pair.baseline.coordinate_hash == pair.candidate.coordinate_hash == "a" * 64
    assert pair.baseline.perplexity == pytest.approx(3.43498)
    assert pair.candidate.accuracies["math_accuracy"] == pytest.approx(499 / 848)
    assert pair.to_command_output()["candidate"]["representation_hash"] == "q4-rdna"


def test_rejects_partial_or_internally_inconsistent_quality_result() -> None:
    with pytest.raises(QualityAdapterError, match="not 'complete'"):
        normalize_q4_threeway_quality(_raw_quality(status="running"))
    inconsistent = _raw_quality()
    inconsistent["q4_rdna"]["math"]["accuracy"] = 0.1  # type: ignore[index]
    with pytest.raises(QualityAdapterError, match="does not match"):
        normalize_q4_threeway_quality(inconsistent)


def test_q8_runtime_protocol_resolves_only_declared_coordinates(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    model = tmp_path / "model.gguf"
    for path in (baseline, candidate, model):
        path.write_bytes(b"fixture")
    output = tmp_path / "result.json"
    protocol = Q8RuntimeQualityProtocol(
        command_template=(
            sys.executable,
            "quality.py",
            "--baseline",
            "{baseline_binary}",
            "--candidate",
            "{candidate_binary}",
            "--model",
            "{model}",
            "--output",
            "{output}",
        ),
        baseline_binary=str(baseline),
        candidate_binary=str(candidate),
        model_path=str(model),
        output_path=str(output),
        cwd=str(tmp_path),
        environment={"DEVICE": "0"},
    )

    command = protocol.command()

    assert command.argv[command.argv.index("--candidate") + 1] == str(candidate.resolve())
    assert command.argv[command.argv.index("--output") + 1] == str(output.resolve())
    assert command.env == {"DEVICE": "0"}


def test_q8_quality_requires_same_model_and_greedy_tokens() -> None:
    payload = {
        "schema": "gpuopt.q8-runtime-quality.v1",
        "status": "complete",
        "protocol_hash": "a" * 64,
        "model_sha256": "b" * 64,
        "baseline": {
            "perplexity": 3.1,
            "math_accuracy": 0.6,
            "greedy_tokens_sha256": "c" * 64,
        },
        "candidate": {
            "perplexity": 3.102,
            "math_accuracy": 0.6,
            "greedy_tokens_sha256": "c" * 64,
        },
    }

    pair = normalize_q8_runtime_quality(payload, expected_model_sha256="b" * 64)

    assert pair.baseline.representation_hash == pair.candidate.representation_hash == "b" * 64
    assert pair.candidate.correctness_passed is True
    payload["candidate"]["greedy_tokens_sha256"] = "d" * 64  # type: ignore[index]
    pair = normalize_q8_runtime_quality(payload, expected_model_sha256="b" * 64)
    assert pair.candidate.correctness_passed is False
    with pytest.raises(QualityAdapterError, match="different model"):
        normalize_q8_runtime_quality(payload, expected_model_sha256="e" * 64)


def test_q8_quality_uses_ground_truth_greedy_accuracy_for_new_artifacts() -> None:
    payload = {
        "schema": "gpuopt.q8-runtime-quality.v1",
        "status": "complete",
        "protocol_hash": "a" * 64,
        "model_sha256": "b" * 64,
        "baseline": {
            "perplexity": 3.1,
            "math_accuracy": 0.6,
            "greedy_tokens_sha256": "c" * 64,
            "greedy_correct": 6,
            "greedy_total": 8,
            "greedy_accuracy": 0.75,
        },
        "candidate": {
            "perplexity": 3.102,
            "math_accuracy": 0.6,
            "greedy_tokens_sha256": "d" * 64,
            "greedy_correct": 6,
            "greedy_total": 8,
            "greedy_accuracy": 0.75,
        },
    }

    pair = normalize_q8_runtime_quality(payload, expected_model_sha256="b" * 64)

    assert pair.candidate.correctness_passed is True
    assert pair.baseline.accuracies["greedy_accuracy"] == 0.75
    payload["candidate"]["greedy_correct"] = 5  # type: ignore[index]
    payload["candidate"]["greedy_accuracy"] = 0.625  # type: ignore[index]
    pair = normalize_q8_runtime_quality(payload, expected_model_sha256="b" * 64)
    assert pair.candidate.correctness_passed is False

    del payload["candidate"]["greedy_accuracy"]  # type: ignore[index]
    with pytest.raises(QualityAdapterError, match="greedy correctness metrics"):
        normalize_q8_runtime_quality(payload, expected_model_sha256="b" * 64)


def test_q8_quality_preserves_generic_suite_accuracies() -> None:
    arm = {
        "perplexity": 3.1,
        "math_accuracy": 0.6,
        "greedy_tokens_sha256": "c" * 64,
        "greedy_correct": 8,
        "greedy_total": 8,
        "greedy_accuracy": 1.0,
        "accuracies": {
            "math_accuracy": 0.6,
            "general_accuracy": 0.7,
            "arc_challenge_accuracy": 0.68,
        },
    }
    payload = {
        "schema": "gpuopt.q8-runtime-quality.v1",
        "status": "complete",
        "protocol_hash": "a" * 64,
        "model_sha256": "b" * 64,
        "baseline": arm,
        "candidate": {**arm, "general_accuracy": 0.7},
    }

    pair = normalize_q8_runtime_quality(payload, expected_model_sha256="b" * 64)

    assert pair.candidate.accuracies["general_accuracy"] == 0.7
    assert pair.candidate.accuracies["arc_challenge_accuracy"] == 0.68


def test_q8_quality_binds_distinct_baseline_and_candidate_representations() -> None:
    payload = {
        "schema": "gpuopt.q8-runtime-quality.v1",
        "status": "complete",
        "protocol_hash": "a" * 64,
        "baseline_model_sha256": "b" * 64,
        "candidate_model_sha256": "c" * 64,
        "baseline": {
            "perplexity": 3.1,
            "math_accuracy": 0.6,
            "greedy_tokens_sha256": "d" * 64,
        },
        "candidate": {
            "perplexity": 3.101,
            "math_accuracy": 0.6,
            "greedy_tokens_sha256": "d" * 64,
        },
    }

    pair = normalize_q8_runtime_quality(
        payload,
        expected_model_sha256="b" * 64,
        expected_candidate_model_sha256="c" * 64,
    )

    assert pair.baseline.representation_hash == "b" * 64
    assert pair.candidate.representation_hash == "c" * 64
    assert pair.candidate.correctness_passed is True
