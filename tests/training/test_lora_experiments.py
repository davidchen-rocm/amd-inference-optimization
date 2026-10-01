"""CPU-only checks of split integrity and strict assessment failure modes."""

import copy
import json
from collections import Counter

import pytest

from macfit_training.artifacts import canonical_sha256
from macfit_training.config import normalized_question, validate_job_input
from macfit_training.experiments import (
    POLICIES,
    assess_evaluation,
    benchmark_manifest,
    build_input,
    main,
    materialize,
    parse_strict_json,
    schema_valid,
    score_text,
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


def test_fixture_split_balance_and_manifest_identity():
    fixture = build_input()
    assert len(fixture["training"]) == 100
    assert len(fixture["evaluation"]) == 20
    train_questions = {
        normalized_question(row["messages"][-2]["content"]) for row in fixture["training"]
    }
    test_questions = {normalized_question(row["question"]) for row in fixture["evaluation"]}
    assert len(train_questions) == 100 and len(test_questions) == 20
    assert train_questions.isdisjoint(test_questions)
    training_languages = Counter(
        json.loads(row["messages"][-1]["content"])["language"] for row in fixture["training"]
    )
    evaluation_languages = Counter(
        json.loads(row["expected"])["language"] for row in fixture["evaluation"]
    )
    assert training_languages == {"en": 50, "zh": 50}
    assert evaluation_languages == {"en": 10, "zh": 10}
    manifest = benchmark_manifest()
    assert Counter(manifest["evaluation_groups"].values()) == {
        "shipping_false_premise_and_format": 5,
        "return_boundary": 5,
        "support_boundary": 5,
        "unknown_policy": 5,
    }
    assert manifest["training_sha256"] == canonical_sha256(fixture["training"])
    assert manifest["evaluation_sha256"] == canonical_sha256(fixture["evaluation"])


@pytest.mark.parametrize("model", ["qwen3-0-6b", "qwen3-1-7b", "qwen3-4b", "qwen3-8b"])
@pytest.mark.parametrize("preset,steps", [("quick", 25), ("standard", 75)])
def test_all_job_inputs_are_bounded_and_use_the_same_held_out_split(model, preset, steps):
    value = build_input(model, preset)
    validated = validate_job_input("training", value)
    actual_steps = min(
        validated["training_config"]["max_steps"],
        validated["training_config"]["epochs"] * 25,
    )
    assert actual_steps == steps
    assert value["evaluation"] == build_input()["evaluation"]
    assert value["training"] == build_input()["training"]
    assert all(schema_valid(json.loads(row["expected"])) for row in value["evaluation"])


def test_boundary_ground_truth_is_inclusive_for_returns_exclusive_for_support_close():
    value = build_input()
    for index, policy in enumerate(POLICIES):
        returned = next(
            row for row in value["evaluation"] if row["id"] == f"eval-{policy['shop'].lower()}-2"
        )
        support = next(
            row for row in value["evaluation"] if row["id"] == f"eval-{policy['shop'].lower()}-3"
        )
        assert json.loads(returned["expected"])["value"] is (index % 2 == 0)
        assert json.loads(support["expected"])["value"] is (index % 2 == 0)


def test_perfect_assessment_accepts_semantic_key_order_and_whitespace():
    fixture = build_input()
    evaluation = perfect_evaluation(fixture)
    row = evaluation["samples"][0]
    row["after"] = json.dumps(dict(reversed(list(json.loads(row["expected"]).items()))), indent=2)
    result = assess_evaluation(evaluation, fixture)
    assert result["metrics"]["after"]["structured_exact_rate"] == 1.0
    assert result["metrics"]["before"]["grounded_fact_rate"] == 1.0
    assert result["metrics"]["after"]["known_claim_on_unknown_policy_count"] == 0
    assert all(group["after"]["samples"] == 5 for group in result["groups"].values())


@pytest.mark.parametrize(
    "bad",
    [
        '{"value":1,"value":2}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '```json\n{"value":1}\n```',
        '{"value":1} extra prose',
        '[{"value":1}]',
        "true",
    ],
)
def test_non_json_and_ambiguous_outputs_are_rejected(bad):
    assert parse_strict_json(bad) is None


@pytest.mark.parametrize(
    "field,bad",
    [
        ("shop", []),
        ("shop", {}),
        ("language", []),
        ("topic", []),
        ("topic", {}),
        ("value", True),
        ("value", 2.0),
    ],
)
def test_malformed_fields_are_invalid_instead_of_crashing(field, bad):
    expected = build_input()["evaluation"][0]["expected"]
    predicted = json.loads(expected)
    predicted[field] = bad
    score = score_text(json.dumps(predicted), expected)
    assert score["strict_json"] is True
    assert score["schema_valid"] is False
    assert score["structured_exact"] is False


def test_valid_json_wrong_facts_do_not_count_as_correct():
    expected = build_input()["evaluation"][0]["expected"]
    predicted = json.loads(expected)
    predicted["value"] = 99
    result = score_text(json.dumps(predicted), expected)
    assert result["schema_valid"] is True
    assert result["grounded_fact"] is False
    assert result["structured_exact"] is False


def test_language_error_is_separate_from_fact_correctness():
    expected = build_input()["evaluation"][0]["expected"]
    predicted = json.loads(expected)
    predicted["language"] = "zh"
    result = score_text(json.dumps(predicted), expected)
    assert result["grounded_fact"] is True
    assert result["language_route"] is False
    assert result["structured_exact"] is False


def test_extra_schema_fields_are_rejected():
    expected = build_input()["evaluation"][0]["expected"]
    predicted = json.loads(expected)
    predicted["explanation"] = "These are fictional policies."
    result = score_text(json.dumps(predicted), expected)
    assert result["strict_json"] is True
    assert result["schema_valid"] is False


def test_unknown_policy_known_claim_is_counted_and_not_credited():
    fixture = build_input()
    evaluation = perfect_evaluation(fixture)
    unknown = next(
        row for row in evaluation["samples"] if json.loads(row["expected"])["status"] == "unknown"
    )
    claimed = json.loads(unknown["expected"])
    claimed.update(status="known", value=20, unit="percent")
    unknown["after"] = json.dumps(claimed)
    result = assess_evaluation(evaluation, fixture)
    assert result["metrics"]["after"]["known_claim_on_unknown_policy_count"] == 1
    assert result["metrics"]["after"]["structured_exact_rate"] == 0.95
    assert result["groups"]["unknown_policy"]["after"]["grounded_fact_rate"] == 0.8


@pytest.mark.parametrize("mutation", ["duplicate", "question", "expected", "missing", "unexpected"])
def test_modified_or_partial_evaluation_is_not_scored(mutation):
    fixture = build_input()
    evaluation = perfect_evaluation(fixture)
    if mutation == "duplicate":
        evaluation["samples"][-1] = copy.deepcopy(evaluation["samples"][0])
    elif mutation == "question":
        evaluation["samples"][0]["question"] += " altered"
    elif mutation == "expected":
        evaluation["samples"][0]["expected"] = "{}"
    elif mutation == "missing":
        evaluation["samples"].pop()
    else:
        evaluation["samples"][0]["id"] = "invented"
    with pytest.raises(ValueError):
        assess_evaluation(evaluation, fixture)


def test_changed_training_fixture_is_not_scored_as_this_benchmark():
    fixture = build_input()
    evaluation = perfect_evaluation(fixture)
    fixture["training"][0]["messages"][-1]["content"] += " "
    with pytest.raises(ValueError, match="not this version"):
        assess_evaluation(evaluation, fixture)


def test_materialization_is_deterministic_and_has_no_gpu_dependency(tmp_path):
    paths = materialize(tmp_path)
    before = {path.name: path.read_bytes() for path in paths}
    assert len(paths) == 7
    materialize(tmp_path)
    assert before == {path.name: path.read_bytes() for path in paths}
    assert json.loads((tmp_path / "qwen3-4b-standard.json").read_text())["preset"] == "standard"
    assert json.loads((tmp_path / "qwen3-8b-standard.json").read_text())["model_id"] == "qwen3-8b"


def test_materialize_selected_registry_models_only(tmp_path):
    paths = materialize(tmp_path, ["qwen3-1-7b", "qwen3-8b"])
    assert {path.name for path in paths} == {
        "qwen3-1-7b-quick.json",
        "qwen3-1-7b-standard.json",
        "qwen3-8b-quick.json",
        "qwen3-8b-standard.json",
        "benchmark.json",
    }


@pytest.mark.parametrize(
    "models",
    [
        [],
        ["qwen3-8b", "qwen3-8b"],
        ["arbitrary/checkpoint"],
        ["qwen3-4b", []],
        "qwen3-8b",
    ],
)
def test_invalid_model_selectors_have_no_file_side_effect(tmp_path, models):
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="distinct model IDs"):
        materialize(output, models)
    assert not output.exists()


def test_cli_model_selector_generates_the_requested_model(tmp_path, capsys):
    assert main(["materialize", "--output", str(tmp_path), "--models", "qwen3-8b"]) == 0
    assert json.loads(capsys.readouterr().out)["benchmark"] == "fictional-policy-json.v1"
    assert len(list(tmp_path.glob("*.json"))) == 3


@pytest.mark.parametrize("models", [["qwen3-8b", "qwen3-8b"], ["arbitrary/checkpoint"]])
def test_cli_rejects_duplicates_and_unpinned_models(tmp_path, models):
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as error:
        main(["materialize", "--output", str(output), "--models", *models])
    assert error.value.code == 2
    assert not output.exists()
