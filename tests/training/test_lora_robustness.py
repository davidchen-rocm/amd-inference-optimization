"""Adaptive split disclosure, independently specified boundary truth and artifact checks."""

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

import macfit_training.robustness as robustness
from macfit_training.artifacts import canonical_sha256
from macfit_training.config import normalized_question, validate_job_input
from macfit_training.experiments import (
    POLICIES,
    schema_valid,
)
from macfit_training.experiments import (
    assess_evaluation as assess_baseline,
)
from macfit_training.experiments import (
    build_input as baseline_input,
)
from macfit_training.robustness import (
    ROBUST_GOAL,
    _shipping,
    assess_evaluation,
    build_input,
    manifest,
    materialize,
)


def perfect_evaluation(fixture):
    return {
        "samples": [
            {
                "id": row["id"],
                "question": row["question"],
                "expected": row["expected"],
                "before": row["expected"],
                "after": row["expected"],
            }
            for row in fixture["evaluation"]
        ]
    }


def test_360_balanced_unique_rows_preserve_the_exact_original_evaluation():
    fixture = build_input()
    assert len(fixture["training"]) == 360
    assert fixture["evaluation"] == baseline_input()["evaluation"]
    questions = [normalized_question(row["messages"][-2]["content"]) for row in fixture["training"]]
    assert len(set(questions)) == 360
    held_out = {normalized_question(row["question"]) for row in fixture["evaluation"]}
    assert set(questions).isdisjoint(held_out)
    assert all(test not in question for test in held_out for question in questions)
    answers = [json.loads(row["messages"][-1]["content"]) for row in fixture["training"]]
    assert Counter(answer["language"] for answer in answers) == {"en": 180, "zh": 180}
    assert Counter(answer["topic"] for answer in answers) == {
        "shipping": 80,
        "return_eligible": 100,
        "support_open": 120,
        "student_discount": 30,
        "warranty": 30,
    }
    assert all(schema_valid(answer) for answer in answers)


@pytest.mark.parametrize("model", ["qwen3-0-6b", "qwen3-1-7b", "qwen3-4b", "qwen3-8b"])
def test_inputs_remain_inside_pinned_standard_worker_contract(model):
    validated = validate_job_input("training", build_input(model))
    assert validated["preset"] == "standard"
    assert len(validated["training"]) < 500
    settings = validated["training_config"]
    assert settings["epochs"] * (360 // settings["gradient_accumulation_steps"]) == 270
    assert settings["walltime_seconds"] == 3600
    assert validated["goal"] == ROBUST_GOAL


@pytest.mark.parametrize("language", ["en", "zh"])
def test_all_policy_labels_have_correct_inclusive_and_exclusive_boundaries(language):
    fixture = build_input()
    # Expected decisions are specified independently of the label helper.
    return_truth = [True, False, True, False, False, True, False, True, False, True]
    support_truth = [True, False, False, True, True, False, False, False, True, True, True, False]
    for policy in POLICIES:
        prefix = f"robust-{policy['shop'].lower()}-{language}-"
        rows = [row for row in fixture["training"] if row["id"].startswith(prefix)]
        labels = [json.loads(row["messages"][-1]["content"]) for row in rows]
        assert [label["value"] for label in labels[:8]] == [policy["shipping"]] * 8
        assert [label["value"] for label in labels[8:18]] == return_truth
        assert [label["value"] for label in labels[18:30]] == support_truth
        assert all(
            label["status"] == "unknown" and label["value"] is None and label["unit"] is None
            for label in labels[30:]
        )


def test_claimed_shipping_values_never_supply_training_labels():
    policy = copy.deepcopy(POLICIES[0])
    policy["shipping"] = 17
    for language in ("en", "zh"):
        assert all(json.loads(answer)["value"] == 17 for _, answer in _shipping(policy, language))


def test_manifest_discloses_adaptive_evaluation_reuse_and_absence_of_human_review():
    fixture, identity = build_input(), manifest()
    assert identity["adaptive_diagnostic"] is True
    assert identity["human_review_performed"] is False
    assert "does not represent independent human review" in identity["approval_note"]
    assert identity["evaluation_reused_from"] == "fictional-policy-json.v1"
    assert identity["training_sha256"] == canonical_sha256(fixture["training"])
    assert identity["evaluation_sha256"] == canonical_sha256(baseline_input()["evaluation"])
    assert identity["reference_sha256"] == canonical_sha256(fixture["source"])
    assert identity["goal_sha256"] == canonical_sha256(fixture["goal"])


def test_adaptive_assessor_binds_its_own_input_and_actual_evaluation():
    fixture = build_input()
    evaluation = perfect_evaluation(fixture)
    result = assess_evaluation(evaluation, fixture)
    assert result["adaptive_diagnostic"] is True
    assert result["metrics"]["after"]["structured_exact_rate"] == 1.0
    assert all(group["after"]["samples"] == 5 for group in result["groups"].values())
    assert result["training_sha256"] == canonical_sha256(fixture["training"])
    assert result["evaluation_artifact_sha256"] == canonical_sha256(evaluation)
    assert len(result["assessment_source_sha256"]) == 64
    helper_file = Path(robustness.score_text.__code__.co_filename)
    assert (
        result["strict_scoring_helpers_source_sha256"]
        == hashlib.sha256(helper_file.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="not this version"):
        assess_baseline(evaluation, fixture)


def test_helper_source_identity_changes_independently_of_assessor_source(tmp_path, monkeypatch):
    fixture = build_input()
    evaluation = perfect_evaluation(fixture)
    helper_source = tmp_path / "strict_helpers.py"
    helper_source.write_text("helper implementation version one\n")
    original = robustness.score_text

    def helper(*args, **kwargs):
        return original(*args, **kwargs)

    helper.__code__ = helper.__code__.replace(co_filename=str(helper_source))
    monkeypatch.setattr(robustness, "score_text", helper)
    first = assess_evaluation(evaluation, fixture)
    helper_source.write_text("helper implementation version two\n")
    second = assess_evaluation(evaluation, fixture)
    assert first["assessment_source_sha256"] == second["assessment_source_sha256"]
    assert (
        first["strict_scoring_helpers_source_sha256"]
        != second["strict_scoring_helpers_source_sha256"]
    )


@pytest.mark.parametrize("field", ["training", "evaluation", "source", "goal"])
def test_changed_adaptive_dataset_reference_goal_or_ground_truth_is_rejected(field):
    fixture = build_input()
    evaluation = perfect_evaluation(fixture)
    if field == "training":
        fixture[field][0]["messages"][-1]["content"] = "{}"
    elif field == "evaluation":
        fixture[field][0]["expected"] = "{}"
    else:
        fixture[field] += " additional text"
    with pytest.raises(ValueError, match="not this version"):
        assess_evaluation(evaluation, fixture)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "question", "expected", "missing", "id", "output"]
)
def test_adaptive_assessor_rejects_altered_or_incomplete_worker_artifacts(mutation):
    fixture = build_input()
    evaluation = perfect_evaluation(fixture)
    if mutation == "duplicate":
        evaluation["samples"][-1] = copy.deepcopy(evaluation["samples"][0])
    elif mutation == "question":
        evaluation["samples"][0]["question"] += " changed"
    elif mutation == "expected":
        evaluation["samples"][0]["expected"] = "{}"
    elif mutation == "missing":
        evaluation["samples"].pop()
    elif mutation == "id":
        evaluation["samples"][0]["id"] = "not-in-heldout"
    else:
        evaluation["samples"][0]["after"] = None
    with pytest.raises(ValueError):
        assess_evaluation(evaluation, fixture)


def test_strict_adaptive_scores_do_not_credit_code_fences_or_boundary_errors():
    fixture = build_input()
    evaluation = perfect_evaluation(fixture)
    evaluation["samples"][0]["after"] = "```json\n" + evaluation["samples"][0]["after"] + "\n```"
    support = next(sample for sample in evaluation["samples"] if sample["id"] == "eval-cedar-3")
    wrong = json.loads(support["after"])
    assert wrong["value"] is False
    wrong["value"] = True
    support["after"] = json.dumps(wrong)
    result = assess_evaluation(evaluation, fixture)
    assert result["metrics"]["after"]["structured_exact_rate"] == 0.9
    assert result["groups"]["support_boundary"]["after"]["grounded_fact_rate"] == 0.8
    assert result["groups"]["shipping_false_premise_and_format"]["after"]["strict_json_rate"] == 0.8


def test_materialization_is_reproducible_and_selection_is_pinned(tmp_path):
    paths = materialize(tmp_path)
    before = {path.name: path.read_bytes() for path in paths}
    assert len(paths) == 4
    materialize(tmp_path)
    assert before == {path.name: path.read_bytes() for path in paths}
    for models in (["arbitrary/model"], ["qwen3-4b", "qwen3-4b"], []):
        output = tmp_path / "not-created"
        with pytest.raises(ValueError, match="distinct model"):
            materialize(output, models)
        assert not output.exists()
