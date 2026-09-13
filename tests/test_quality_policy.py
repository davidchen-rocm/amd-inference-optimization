from __future__ import annotations

import json
from pathlib import Path

import pytest

from amd_inference_opt.quality_policy import (
    PROVISIONAL_MATH_100_ANSWERS_SHA256,
    PROVISIONAL_MATH_100_IDS_SHA256,
    PROVISIONAL_MATH_100_ORDER_SHA256,
    PROVISIONAL_MATH_100_QUOTAS,
    ProvisionalMath100Policy,
    ProvisionalQualityMeasurement,
    QualityEvidenceOutcome,
    QualityPolicy,
    QualityPolicyError,
    evaluate_provisional_quality,
)

ROOT = Path(__file__).resolve().parents[1]


def _math_items() -> list[dict[str, object]]:
    fixture = ROOT / "fixtures/q8-runtime-quality/mmlu-math.jsonl"
    return [json.loads(line) for line in fixture.read_text().splitlines() if line]


def test_provisional_policy_freezes_quota_ids_order_and_answers() -> None:
    policy = ProvisionalMath100Policy()
    assert isinstance(policy, QualityPolicy)

    selection = policy.select(_math_items())
    repeated = policy.select(list(reversed(_math_items())))
    protocol = selection.protocol

    assert protocol.protocol_id == "provisional-math-100.v1"
    assert protocol.seed == 20260815
    assert protocol.quotas == PROVISIONAL_MATH_100_QUOTAS
    assert protocol.selected_total == 100
    assert protocol.ids_sha256 == PROVISIONAL_MATH_100_IDS_SHA256
    assert protocol.order_sha256 == PROVISIONAL_MATH_100_ORDER_SHA256
    assert protocol.answers_sha256 == PROVISIONAL_MATH_100_ANSWERS_SHA256
    assert selection.ids == repeated.ids
    assert selection.answers == repeated.answers
    assert protocol.protocol_hash == repeated.protocol.protocol_hash


def test_provisional_policy_rejects_fixture_or_answer_drift() -> None:
    policy = ProvisionalMath100Policy()
    items = _math_items()

    with pytest.raises(QualityPolicyError, match="848-item"):
        policy.select(items[:-1])

    selected = policy.select(items)
    selected_id = selected.ids[0]
    drifted = [dict(item) for item in items]
    changed = next(item for item in drifted if item["id"] == selected_id)
    changed["answer_index"] = (int(changed["answer_index"]) + 1) % 4
    with pytest.raises(QualityPolicyError, match="answers_sha256"):
        policy.select(drifted)


def test_provisional_gate_is_evidence_only_and_maps_pass_to_experimental() -> None:
    protocol = ProvisionalMath100Policy().select(_math_items()).protocol
    baseline = ProvisionalQualityMeasurement(
        math_correct=62,
        math_total=100,
        perplexity=2.0,
        greedy_correct=6,
        greedy_total=8,
        protocol_hash=protocol.protocol_hash,
    )
    candidate = ProvisionalQualityMeasurement(
        math_correct=60,
        math_total=100,
        perplexity=2.01,
        greedy_correct=6,
        greedy_total=8,
        protocol_hash=protocol.protocol_hash,
    )

    evidence = evaluate_provisional_quality(baseline, candidate, protocol=protocol)

    assert evidence.outcome == QualityEvidenceOutcome.PASS
    assert evidence.passed
    assert evidence.evidence_only is True
    assert evidence.campaign_disposition == "experimental"
    assert evidence.measurements["math_correct_drop"] == 2
    assert evidence.measurements["perplexity_regression_fraction"] == pytest.approx(0.005)
    assert set(evidence.to_dict()) == {
        "schema",
        "protocol_id",
        "protocol_hash",
        "outcome",
        "evidence_only",
        "campaign_disposition",
        "checks",
        "measurements",
        "reasons",
    }


@pytest.mark.parametrize(
    ("candidate", "failed_check"),
    [
        (
            {
                "math_correct": 59,
                "math_total": 100,
                "perplexity": 2.0,
                "greedy_correct": 6,
                "greedy_total": 8,
            },
            "math_correct_drop_at_most_2",
        ),
        (
            {
                "math_correct": 62,
                "math_total": 100,
                "perplexity": 2.0101,
                "greedy_correct": 6,
                "greedy_total": 8,
            },
            "perplexity_regression_at_most_0_5_percent",
        ),
        (
            {
                "math_correct": 62,
                "math_total": 100,
                "perplexity": 2.0,
                "greedy_correct": 5,
                "greedy_total": 8,
            },
            "greedy_accuracy_not_lower",
        ),
    ],
)
def test_each_provisional_quality_budget_can_fail(
    candidate: dict[str, object], failed_check: str
) -> None:
    baseline = {
        "math_correct": 62,
        "math_total": 100,
        "perplexity": 2.0,
        "greedy_correct": 6,
        "greedy_total": 8,
    }

    evidence = evaluate_provisional_quality(baseline, candidate)

    assert evidence.outcome == QualityEvidenceOutcome.FAIL
    assert not evidence.checks[failed_check]
    assert evidence.campaign_disposition == "experimental"


def test_quality_measurements_must_bind_to_supplied_protocol() -> None:
    bound = {
        "math_correct": 62,
        "math_total": 100,
        "perplexity": 2.0,
        "greedy_correct": 6,
        "greedy_total": 8,
        "protocol_hash": "0" * 64,
    }

    with pytest.raises(QualityPolicyError, match="protocol hash does not match"):
        evaluate_provisional_quality(bound, bound)
