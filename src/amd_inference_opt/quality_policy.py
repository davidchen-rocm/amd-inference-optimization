"""Pure quality-policy evidence for the provisional 100-question math subset.

The provisional policy is intentionally not an acceptance authority.  A passing
result is useful evidence, but campaign orchestration must still label it
``experimental`` until an authoritative quality protocol is selected.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "PROVISIONAL_MATH_100_ANSWERS_SHA256",
    "PROVISIONAL_MATH_100_IDS_SHA256",
    "PROVISIONAL_MATH_100_ORDER_SHA256",
    "PROVISIONAL_MATH_100_QUOTAS",
    "ProvisionalMath100Policy",
    "ProvisionalMath100Protocol",
    "ProvisionalMath100Selection",
    "ProvisionalQualityEvidence",
    "ProvisionalQualityMeasurement",
    "QualityEvidenceOutcome",
    "QualityPolicy",
    "QualityPolicyError",
    "evaluate_provisional_quality",
]


PROVISIONAL_MATH_100_QUOTAS: Mapping[str, int] = {
    "abstract_algebra": 12,
    "college_mathematics": 12,
    "high_school_mathematics": 32,
    "elementary_mathematics": 44,
}
PROVISIONAL_MATH_100_IDS_SHA256 = (
    "5da1fc7d2c551b081bd21f38326482190a42fa44edaef88a701a7a4e961417d5"
)
PROVISIONAL_MATH_100_ORDER_SHA256 = (
    "11670b5040127d9713742b935a743fe932edbe3cc83f53202962e22929365841"
)
PROVISIONAL_MATH_100_ANSWERS_SHA256 = (
    "9531d710030067deb85355e29e3adc62cfd38b6396eac6c4ed862c397a2b446b"
)
_PROVISIONAL_MATH_100_ID_ANSWERS_SHA256 = (
    "1860661ee9b8b2365c6162b123753ecd4b190573de810521effd882fc61221a9"
)


class QualityPolicyError(ValueError):
    """A policy definition, source fixture, or quality measurement is invalid."""


class QualityEvidenceOutcome(StrEnum):
    """Non-authoritative outcome recorded by a provisional policy."""

    PASS = "PASS"
    FAIL = "FAIL"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank(item_id: str, seed: int) -> bytes:
    return hashlib.sha256(
        str(seed).encode("ascii") + b"\0" + item_id.encode("utf-8")
    ).digest()


@dataclass(frozen=True)
class ProvisionalMath100Protocol:
    """Identity of the frozen, stratified 100-question protocol."""

    protocol_id: str
    seed: int
    quotas: Mapping[str, int]
    selection_algorithm: str
    rank_input: str
    source_total: int
    selected_total: int
    ids_sha256: str
    order_sha256: str
    answers_sha256: str
    id_answers_sha256: str

    @property
    def semantic_hash(self) -> str:
        return _canonical_sha256(asdict(self))

    @property
    def protocol_hash(self) -> str:
        return self.semantic_hash

    def to_dict(self) -> dict[str, object]:
        document = asdict(self)
        document["semantic_hash"] = self.semantic_hash
        document["protocol_hash"] = self.protocol_hash
        return document


@dataclass(frozen=True)
class ProvisionalMath100Selection:
    """The ordered questions and their independently hash-bound protocol."""

    items: tuple[Mapping[str, Any], ...]
    protocol: ProvisionalMath100Protocol

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(str(item["id"]) for item in self.items)

    @property
    def answers(self) -> tuple[str, ...]:
        return tuple(_answer_letter(item, index) for index, item in enumerate(self.items))

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol.to_dict(),
            "items": [dict(item) for item in self.items],
        }


@dataclass(frozen=True)
class ProvisionalQualityMeasurement:
    """Minimum evidence needed to evaluate one model/runtime arm."""

    math_correct: int
    math_total: int
    perplexity: float
    greedy_correct: int
    greedy_total: int
    protocol_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("math_correct", self.math_correct),
            ("math_total", self.math_total),
            ("greedy_correct", self.greedy_correct),
            ("greedy_total", self.greedy_total),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise QualityPolicyError(f"{field_name} must be an integer")
        if self.math_total != 100 or not 0 <= self.math_correct <= self.math_total:
            raise QualityPolicyError("provisional math evidence must contain 0..100 correct")
        if self.greedy_total <= 0 or not 0 <= self.greedy_correct <= self.greedy_total:
            raise QualityPolicyError("greedy correctness counts are invalid")
        if (
            isinstance(self.perplexity, bool)
            or not isinstance(self.perplexity, (int, float))
            or not math.isfinite(float(self.perplexity))
            or self.perplexity <= 0
        ):
            raise QualityPolicyError("perplexity must be finite and positive")
        if self.protocol_hash is not None and (
            len(self.protocol_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.protocol_hash)
        ):
            raise QualityPolicyError("protocol_hash must be a lowercase SHA-256")

    @classmethod
    def from_value(
        cls,
        value: ProvisionalQualityMeasurement | Mapping[str, object],
    ) -> ProvisionalQualityMeasurement:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise QualityPolicyError("quality measurement must be an object")
        try:
            return cls(
                math_correct=value["math_correct"],  # type: ignore[arg-type]
                math_total=value["math_total"],  # type: ignore[arg-type]
                perplexity=value["perplexity"],  # type: ignore[arg-type]
                greedy_correct=value["greedy_correct"],  # type: ignore[arg-type]
                greedy_total=value["greedy_total"],  # type: ignore[arg-type]
                protocol_hash=value.get("protocol_hash"),  # type: ignore[arg-type]
            )
        except KeyError as error:
            raise QualityPolicyError(
                f"quality measurement is missing {error.args[0]}"
            ) from error


@dataclass(frozen=True)
class ProvisionalQualityEvidence:
    """Evidence-only quality comparison; never an accept/reject decision."""

    schema: str
    protocol_id: str
    protocol_hash: str | None
    outcome: QualityEvidenceOutcome
    evidence_only: bool
    campaign_disposition: str
    checks: Mapping[str, bool]
    measurements: Mapping[str, float | int]
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.outcome == QualityEvidenceOutcome.PASS

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@runtime_checkable
class QualityPolicy(Protocol):
    """Minimal pure-policy interface consumed by campaign adapters."""

    protocol_id: str
    authoritative: bool
    campaign_disposition: str

    @property
    def protocol(self) -> ProvisionalMath100Protocol: ...

    def select(self, items: Sequence[Mapping[str, Any]]) -> ProvisionalMath100Selection: ...

    def evaluate(
        self,
        baseline: ProvisionalQualityMeasurement | Mapping[str, object],
        candidate: ProvisionalQualityMeasurement | Mapping[str, object],
        *,
        protocol: ProvisionalMath100Protocol | None = None,
    ) -> ProvisionalQualityEvidence: ...


def _answer_letter(item: Mapping[str, Any], index: int) -> str:
    answer = item.get("answer")
    if isinstance(answer, str) and answer in "ABCD" and len(answer) == 1:
        return answer
    answer_index = item.get("answer_index")
    if (
        isinstance(answer_index, int)
        and not isinstance(answer_index, bool)
        and 0 <= answer_index < 4
    ):
        return "ABCD"[answer_index]
    raise QualityPolicyError(f"math item {index} has no valid answer")


@dataclass(frozen=True)
class ProvisionalMath100Policy:
    """Frozen evidence policy for ``provisional-math-100.v1``."""

    protocol_id: str = "provisional-math-100.v1"
    seed: int = 20260815
    authoritative: bool = False
    campaign_disposition: str = "experimental"
    max_math_correct_drop: int = 2
    max_perplexity_regression_fraction: float = 0.005

    def __post_init__(self) -> None:
        if self.protocol_id != "provisional-math-100.v1" or self.seed != 20260815:
            raise QualityPolicyError("the provisional policy identity is fixed")
        if self.authoritative is not False or self.campaign_disposition != "experimental":
            raise QualityPolicyError("provisional outcomes must remain experimental evidence")
        if self.max_math_correct_drop != 2:
            raise QualityPolicyError("provisional math drop budget is fixed at two questions")
        if self.max_perplexity_regression_fraction != 0.005:
            raise QualityPolicyError("provisional PPL budget is fixed at 0.5%")

    @property
    def protocol(self) -> ProvisionalMath100Protocol:
        """Return the immutable protocol identity even before fixture materialization."""

        return ProvisionalMath100Protocol(
            protocol_id=self.protocol_id,
            seed=self.seed,
            quotas=dict(PROVISIONAL_MATH_100_QUOTAS),
            selection_algorithm="stratified_sha256_rank_v1",
            rank_input="utf8(str(seed) + NUL + item_id)",
            source_total=848,
            selected_total=100,
            ids_sha256=PROVISIONAL_MATH_100_IDS_SHA256,
            order_sha256=PROVISIONAL_MATH_100_ORDER_SHA256,
            answers_sha256=PROVISIONAL_MATH_100_ANSWERS_SHA256,
            id_answers_sha256=_PROVISIONAL_MATH_100_ID_ANSWERS_SHA256,
        )

    def select(self, items: Sequence[Mapping[str, Any]]) -> ProvisionalMath100Selection:
        """Select and identity-check the reviewed subset independent of input order."""

        source = tuple(items)
        if len(source) != 848:
            raise QualityPolicyError("provisional policy requires the complete 848-item fixture")
        ids: list[str] = []
        normalized: list[Mapping[str, Any]] = []
        for index, item in enumerate(source):
            if not isinstance(item, Mapping):
                raise QualityPolicyError(f"math item {index} must be an object")
            item_id = item.get("id")
            subject = item.get("subject")
            if not isinstance(item_id, str) or not item_id:
                raise QualityPolicyError(f"math item {index} has no valid id")
            if subject not in PROVISIONAL_MATH_100_QUOTAS:
                raise QualityPolicyError(f"math item {index} has an unsupported subject")
            _answer_letter(item, index)
            ids.append(item_id)
            normalized.append(item)
        if len(ids) != len(set(ids)):
            raise QualityPolicyError("math item IDs must be unique")

        selected: list[Mapping[str, Any]] = []
        counts = Counter(str(item["subject"]) for item in normalized)
        for subject, quota in PROVISIONAL_MATH_100_QUOTAS.items():
            if counts[subject] < quota:
                raise QualityPolicyError(
                    f"math fixture has {counts[subject]} {subject} items; needs {quota}"
                )
            available = [item for item in normalized if item["subject"] == subject]
            available.sort(
                key=lambda item: (_rank(str(item["id"]), self.seed), str(item["id"]))
            )
            selected.extend(available[:quota])
        selected.sort(
            key=lambda item: (_rank(str(item["id"]), self.seed), str(item["id"]))
        )
        selected_ids = [str(item["id"]) for item in selected]
        answers = [_answer_letter(item, index) for index, item in enumerate(selected)]
        pairs = [
            {"id": item_id, "answer": answer}
            for item_id, answer in zip(selected_ids, answers, strict=True)
        ]
        identities = {
            "ids_sha256": _canonical_sha256(sorted(selected_ids)),
            "order_sha256": _canonical_sha256(selected_ids),
            "answers_sha256": _canonical_sha256(answers),
            "id_answers_sha256": _canonical_sha256(pairs),
        }
        expected = {
            "ids_sha256": PROVISIONAL_MATH_100_IDS_SHA256,
            "order_sha256": PROVISIONAL_MATH_100_ORDER_SHA256,
            "answers_sha256": PROVISIONAL_MATH_100_ANSWERS_SHA256,
            "id_answers_sha256": _PROVISIONAL_MATH_100_ID_ANSWERS_SHA256,
        }
        drift = [name for name, digest in identities.items() if digest != expected[name]]
        if drift:
            raise QualityPolicyError(
                "provisional math fixture identity drifted: " + ", ".join(drift)
            )
        return ProvisionalMath100Selection(items=tuple(selected), protocol=self.protocol)

    def evaluate(
        self,
        baseline: ProvisionalQualityMeasurement | Mapping[str, object],
        candidate: ProvisionalQualityMeasurement | Mapping[str, object],
        *,
        protocol: ProvisionalMath100Protocol | None = None,
    ) -> ProvisionalQualityEvidence:
        return evaluate_provisional_quality(
            baseline,
            candidate,
            policy=self,
            protocol=protocol,
        )


def evaluate_provisional_quality(
    baseline: ProvisionalQualityMeasurement | Mapping[str, object],
    candidate: ProvisionalQualityMeasurement | Mapping[str, object],
    *,
    policy: ProvisionalMath100Policy | None = None,
    protocol: ProvisionalMath100Protocol | None = None,
) -> ProvisionalQualityEvidence:
    """Apply the fixed evidence gate without promoting it to campaign acceptance."""

    resolved_policy = policy or ProvisionalMath100Policy()
    baseline_measurement = ProvisionalQualityMeasurement.from_value(baseline)
    candidate_measurement = ProvisionalQualityMeasurement.from_value(candidate)
    if baseline_measurement.greedy_total != candidate_measurement.greedy_total:
        raise QualityPolicyError("baseline and candidate greedy totals must match")
    resolved_protocol = protocol or resolved_policy.protocol
    if resolved_protocol != resolved_policy.protocol:
        raise QualityPolicyError("quality protocol does not match the provisional policy")
    protocol_hash = resolved_protocol.protocol_hash
    for label, measurement in (
        ("baseline", baseline_measurement),
        ("candidate", candidate_measurement),
    ):
        if measurement.protocol_hash is not None:
            if measurement.protocol_hash != protocol_hash:
                raise QualityPolicyError(f"{label} protocol hash does not match")

    math_drop = baseline_measurement.math_correct - candidate_measurement.math_correct
    ppl_regression = (
        candidate_measurement.perplexity / baseline_measurement.perplexity - 1
    )
    checks = {
        "math_correct_drop_at_most_2": math_drop <= 2,
        "perplexity_regression_at_most_0_5_percent": (
            ppl_regression <= 0.005 or math.isclose(ppl_regression, 0.005, abs_tol=1e-12)
        ),
        "greedy_accuracy_not_lower": (
            candidate_measurement.greedy_correct
            * baseline_measurement.greedy_total
            >= baseline_measurement.greedy_correct * candidate_measurement.greedy_total
        ),
    }
    reasons: list[str] = []
    if not checks["math_correct_drop_at_most_2"]:
        reasons.append("candidate math score drops by more than two questions")
    if not checks["perplexity_regression_at_most_0_5_percent"]:
        reasons.append("candidate perplexity regresses by more than 0.5%")
    if not checks["greedy_accuracy_not_lower"]:
        reasons.append("candidate greedy accuracy declines")
    if not reasons:
        reasons.append("all provisional quality evidence checks pass")
    return ProvisionalQualityEvidence(
        schema="gpuopt.provisional-quality-evidence.v1",
        protocol_id=resolved_policy.protocol_id,
        protocol_hash=protocol_hash,
        outcome=(
            QualityEvidenceOutcome.PASS
            if all(checks.values())
            else QualityEvidenceOutcome.FAIL
        ),
        evidence_only=True,
        campaign_disposition=resolved_policy.campaign_disposition,
        checks=checks,
        measurements={
            "baseline_math_correct": baseline_measurement.math_correct,
            "candidate_math_correct": candidate_measurement.math_correct,
            "math_correct_drop": math_drop,
            "baseline_perplexity": float(baseline_measurement.perplexity),
            "candidate_perplexity": float(candidate_measurement.perplexity),
            "perplexity_regression_fraction": ppl_regression,
            "baseline_greedy_correct": baseline_measurement.greedy_correct,
            "candidate_greedy_correct": candidate_measurement.greedy_correct,
            "greedy_total": baseline_measurement.greedy_total,
        },
        reasons=tuple(reasons),
    )
