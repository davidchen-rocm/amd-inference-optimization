"""Versioned, offline evaluation suites for llama.cpp quality comparisons.

Suite preparation is deliberately separate from evaluation.  A preparation run
normalizes upstream datasets, deterministically selects cases, and atomically
publishes a manifest plus a JSONL fixture.  Experiments consume only that frozen
fixture and verify its digest before constructing llama-perplexity input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EVALUATION_SUITE_SCHEMA = "gpuopt.evaluation-suite-manifest.v1"
QUALITY_PLAN_SCHEMA = "gpuopt.quality-plan.v1"
QUALITY_SUITE_RESULT_SCHEMA = "gpuopt.quality-suite-result.v1"
GENERAL_100_SUITE_ID = "general-100.v1"
GENERAL_100_SEED = "gpuopt-general-100-v1"
GENERAL_100_QUOTA = 25
PROVISIONAL_MATH_100_POLICY = "provisional-math-100.v1"
BALANCED_200_POLICY = "balanced-200.v1"
QualityPolicyId: TypeAlias = Literal[
    "provisional-math-100.v1",
    "balanced-200.v1",
]
QualitySuiteId: TypeAlias = Literal["math-100.v1", "general-100.v1"]
QUALITY_POLICY_SUITES: dict[str, tuple[QualitySuiteId, ...]] = {
    PROVISIONAL_MATH_100_POLICY: ("math-100.v1",),
    BALANCED_200_POLICY: ("math-100.v1", "general-100.v1"),
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FINAL_RESULT_PATTERN = re.compile(
    r"Final result:\s*([0-9]+(?:\.[0-9]+)?)\s*\+/-\s*"
    r"([0-9]+(?:\.[0-9]+)?)"
)
_PROGRESS_PATTERN = re.compile(
    r"^\s*([0-9]+)[\t ]+([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE
)


class EvaluationSuiteError(ValueError):
    """A suite cannot be prepared, verified, serialized, or parsed safely."""


def quality_policy_suites(policy: QualityPolicyId) -> tuple[QualitySuiteId, ...]:
    """Resolve a stable policy name to its ordered suite composition."""

    return QUALITY_POLICY_SUITES[policy]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class DatasetSourceV1(_StrictModel):
    source_id: str
    dataset: str
    config: str | None = None
    split: str
    revision: str
    license: str

    @field_validator("source_id", "dataset", "split", "revision", "license")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dataset source fields must not be empty")
        return value


class SuiteFileV1(_StrictModel):
    path: str
    sha256: str
    bytes: Annotated[int, Field(ge=0)]
    media_type: Literal["application/jsonl"] = "application/jsonl"

    @field_validator("path")
    @classmethod
    def relative_file(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or value in {"", "."} or ".." in candidate.parts:
            raise ValueError("suite file path must be a safe relative path")
        return value

    @field_validator("sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return value


class SelectionEvidenceV1(_StrictModel):
    method: Literal["sha256_rank_v1"] = "sha256_rank_v1"
    seed: str
    rank_input: Literal["utf8(seed + NUL + group + NUL + case_id)"] = (
        "utf8(seed + NUL + group + NUL + case_id)"
    )
    selected_ids_sha256: str

    @field_validator("seed")
    @classmethod
    def seed_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("selection seed must not be empty")
        return value

    @field_validator("selected_ids_sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("selected_ids_sha256 must be a lowercase SHA-256 digest")
        return value


class EvaluationCaseV1(_StrictModel):
    case_id: str
    group: str
    question: str
    choices: list[str]
    answer_index: Annotated[int, Field(ge=0)]
    source_record_id: str

    @field_validator("case_id", "group", "question", "source_record_id")
    @classmethod
    def required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("case text fields must not be empty")
        return value

    @field_validator("choices")
    @classmethod
    def valid_choices(cls, value: list[str]) -> list[str]:
        if not 2 <= len(value) <= 26:
            raise ValueError("multiple-choice cases require 2 through 26 choices")
        if any(not choice.strip() for choice in value):
            raise ValueError("choices must not be empty")
        return value

    @model_validator(mode="after")
    def answer_exists(self) -> EvaluationCaseV1:
        if self.answer_index >= len(self.choices):
            raise ValueError("answer_index is outside the choices list")
        return self


class EvaluationSuiteManifestV1(_StrictModel):
    schema_version: Literal["gpuopt.evaluation-suite-manifest.v1"] = EVALUATION_SUITE_SCHEMA
    suite_id: str
    kind: Literal["multiple_choice"] = "multiple_choice"
    description: str
    total_cases: Annotated[int, Field(gt=0)]
    group_counts: dict[str, Annotated[int, Field(gt=0)]]
    sources: list[DatasetSourceV1]
    selection: SelectionEvidenceV1
    files: list[SuiteFileV1]
    limitations: list[str] = Field(default_factory=list)

    @field_validator("suite_id", "description")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("suite fields must not be empty")
        return value

    @model_validator(mode="after")
    def internally_consistent(self) -> EvaluationSuiteManifestV1:
        if sum(self.group_counts.values()) != self.total_cases:
            raise ValueError("group_counts must sum to total_cases")
        if len({source.source_id for source in self.sources}) != len(self.sources):
            raise ValueError("suite source_id values must be unique")
        if len({file.path for file in self.files}) != len(self.files):
            raise ValueError("suite file paths must be unique")
        if not self.sources or not self.files:
            raise ValueError("suite manifest requires sources and files")
        return self

class QualitySuiteCoordinateV1(_StrictModel):
    suite_id: str
    manifest_path: str
    manifest_sha256: str
    selected_ids_sha256: str

    @field_validator("suite_id", "manifest_path")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("quality suite coordinate fields must not be empty")
        return value

    @field_validator("manifest_sha256", "selected_ids_sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("quality coordinate hashes must be lowercase SHA-256 digests")
        return value


class QualityPlanV1(_StrictModel):
    schema_version: Literal["gpuopt.quality-plan.v1"] = QUALITY_PLAN_SCHEMA
    suites: list[QualitySuiteCoordinateV1]
    scorer: Literal["llama-perplexity-multiple-choice-v1"] = (
        "llama-perplexity-multiple-choice-v1"
    )
    scorer_protocol_sha256: str
    include_perplexity: bool = True
    include_greedy_canary: bool = True

    @field_validator("scorer_protocol_sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("scorer protocol hash must be a lowercase SHA-256 digest")
        return value

    @field_validator("suites")
    @classmethod
    def unique_suites(
        cls, value: list[QualitySuiteCoordinateV1]
    ) -> list[QualitySuiteCoordinateV1]:
        if not value or len({suite.suite_id for suite in value}) != len(value):
            raise ValueError("quality plan requires unique suites")
        return value

    @property
    def protocol_sha256(self) -> str:
        # The filesystem location is operational, not a semantic evaluation
        # coordinate.  Relocating an identical frozen fixture must not change
        # the baseline/candidate comparison protocol.
        return canonical_sha256(
            {
                "schema": QUALITY_PLAN_SCHEMA,
                "suites": [
                    {
                        "suite_id": suite.suite_id,
                        "manifest_sha256": suite.manifest_sha256,
                        "selected_ids_sha256": suite.selected_ids_sha256,
                    }
                    for suite in self.suites
                ],
                "scorer": self.scorer,
                "scorer_protocol_sha256": self.scorer_protocol_sha256,
                "include_perplexity": self.include_perplexity,
                "include_greedy_canary": self.include_greedy_canary,
            }
        )


class ItemResultV1(_StrictModel):
    case_id: str
    correct: bool


class QualitySuiteResultV1(_StrictModel):
    schema_version: Literal["gpuopt.quality-suite-result.v1"] = QUALITY_SUITE_RESULT_SCHEMA
    suite_id: str
    variant: Literal["baseline", "candidate"]
    status: Literal["COMPLETED", "FAILED", "INCONCLUSIVE"]
    manifest_sha256: str
    protocol_sha256: str
    total: Annotated[int, Field(ge=0)] = 0
    correct: Annotated[int, Field(ge=0)] = 0
    accuracy: Annotated[float, Field(ge=0, le=1)] | None = None
    uncertainty_percent: Annotated[float, Field(ge=0)] | None = None
    items: list[ItemResultV1] = Field(default_factory=list)
    failure_reason: str | None = None

    @field_validator("manifest_sha256", "protocol_sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("quality result hashes must be lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def result_consistency(self) -> QualitySuiteResultV1:
        if self.status == "COMPLETED":
            if self.total <= 0 or len(self.items) != self.total:
                raise ValueError("completed result requires one item result per case")
            if self.correct != sum(item.correct for item in self.items):
                raise ValueError("correct count does not match item results")
            expected = self.correct / self.total
            if self.accuracy is None or abs(self.accuracy - expected) > 1e-12:
                raise ValueError("accuracy does not match correct/total")
            if self.failure_reason is not None:
                raise ValueError("completed result cannot have a failure reason")
        else:
            if not self.failure_reason or self.total or self.correct or self.items:
                raise ValueError("non-completed result requires only a failure reason")
            if self.accuracy is not None or self.uncertainty_percent is not None:
                raise ValueError("non-completed result cannot contain metrics")
        return self


GENERAL_100_SOURCES: tuple[DatasetSourceV1, ...] = (
    DatasetSourceV1(
        source_id="mmlu_general",
        dataset="cais/mmlu",
        config="all",
        split="test",
        revision="c30699e8356da336a370243923dbaf21066bb9fe",
        license="MIT",
    ),
    DatasetSourceV1(
        source_id="arc_challenge",
        dataset="allenai/ai2_arc",
        config="ARC-Challenge",
        split="test",
        revision="210d026faf9955653af8916fad021475a3f00453",
        license="CC-BY-SA-4.0",
    ),
    DatasetSourceV1(
        source_id="hellaswag",
        dataset="Rowan/hellaswag",
        split="validation",
        revision="218ec52e09a7e7462a5400043bb9a69a41d06b76",
        license="MIT",
    ),
    DatasetSourceV1(
        source_id="winogrande",
        dataset="allenai/winogrande",
        config="winogrande_xl",
        split="validation",
        revision="01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        license="Apache-2.0",
    ),
)


class GeneralDatasetLoader(Protocol):
    def __call__(self, source: DatasetSourceV1) -> Iterable[Mapping[str, Any]]: ...


def _source_record_id(source: DatasetSourceV1, row: Mapping[str, Any]) -> str:
    for name in ("id", "question_id", "ind"):
        value = row.get(name)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    return canonical_sha256(dict(row))[:24]


def _answer_from_labels(labels: Sequence[object], answer: object) -> int:
    rendered = [str(label) for label in labels]
    try:
        return rendered.index(str(answer))
    except ValueError as exc:
        raise EvaluationSuiteError(f"answer {answer!r} is absent from choice labels") from exc


def _normalize_general_row(
    source: DatasetSourceV1, row: Mapping[str, Any]
) -> EvaluationCaseV1:
    group = source.source_id
    # HellaSwag's ``ind`` is not globally unique in the validation split.  Bind
    # its identity to the complete upstream record so distinct cases can never
    # collide while identical repeated rows still deduplicate deterministically.
    source_record_id = (
        canonical_sha256(dict(row))[:24]
        if group == "hellaswag"
        else _source_record_id(source, row)
    )
    if group == "mmlu_general":
        question = row.get("question")
        choices = row.get("choices")
        answer = row.get("answer")
        if not isinstance(answer, int) or isinstance(answer, bool):
            raise EvaluationSuiteError("MMLU answer must be an integer choice index")
        answer_index = answer
    elif group == "arc_challenge":
        question = row.get("question")
        raw_choices = row.get("choices")
        if not isinstance(raw_choices, Mapping):
            raise EvaluationSuiteError("ARC choices must be an object")
        choices = raw_choices.get("text")
        labels = raw_choices.get("label")
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise EvaluationSuiteError("ARC choice labels must be a list")
        answer_index = _answer_from_labels(labels, row.get("answerKey"))
    elif group == "hellaswag":
        question = row.get("ctx")
        choices = row.get("endings")
        try:
            answer_index = int(row.get("label"))
        except (TypeError, ValueError) as exc:
            raise EvaluationSuiteError("HellaSwag label must be an integer") from exc
    elif group == "winogrande":
        question = row.get("sentence")
        choices = [row.get("option1"), row.get("option2")]
        try:
            answer_index = int(row.get("answer")) - 1
        except (TypeError, ValueError) as exc:
            raise EvaluationSuiteError("WinoGrande answer must be 1 or 2") from exc
    else:  # pragma: no cover - DatasetSourceV1 is public, so guard dynamic callers
        raise EvaluationSuiteError(f"unsupported general suite source: {group}")
    if not isinstance(question, str):
        raise EvaluationSuiteError(f"{group} question must be text")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        raise EvaluationSuiteError(f"{group} choices must be a list")
    rendered_choices = [str(choice) if choice is not None else "" for choice in choices]
    case_id = f"{group}:{source_record_id}"
    try:
        return EvaluationCaseV1(
            case_id=case_id,
            group=group,
            question=question,
            choices=rendered_choices,
            answer_index=answer_index,
            source_record_id=source_record_id,
        )
    except ValueError as exc:
        raise EvaluationSuiteError(f"invalid {case_id}: {exc}") from exc


def _rank(case: EvaluationCaseV1, seed: str) -> tuple[bytes, str]:
    material = f"{seed}\0{case.group}\0{case.case_id}".encode()
    return hashlib.sha256(material).digest(), case.case_id


def _validate_general_sources(sources: Sequence[DatasetSourceV1]) -> None:
    source_ids = [source.source_id for source in sources]
    expected = {source.source_id for source in GENERAL_100_SOURCES}
    if len(source_ids) != len(set(source_ids)):
        raise EvaluationSuiteError("general-100 source_id values must be unique")
    if set(source_ids) != expected:
        raise EvaluationSuiteError(
            "general-100 requires exactly these sources: " + ", ".join(sorted(expected))
        )


def select_general_100(
    records: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    sources: Sequence[DatasetSourceV1] = GENERAL_100_SOURCES,
    seed: str = GENERAL_100_SEED,
) -> list[EvaluationCaseV1]:
    """Normalize and select exactly 25 deterministically ranked cases per group."""

    _validate_general_sources(sources)
    selected: list[EvaluationCaseV1] = []
    all_ids: set[str] = set()
    for source in sources:
        raw_rows = records.get(source.source_id)
        if raw_rows is None:
            raise EvaluationSuiteError(f"loader omitted source {source.source_id}")
        unique_candidates: dict[str, EvaluationCaseV1] = {}
        for row in raw_rows:
            case = _normalize_general_row(source, row)
            existing = unique_candidates.get(case.case_id)
            if existing is not None:
                if existing != case:
                    raise EvaluationSuiteError(
                        f"conflicting duplicate case id: {case.case_id}"
                    )
                # Some upstream aggregate configurations repeat an identical record.
                # Treating it as one case keeps selection independent of row ordering.
                continue
            unique_candidates[case.case_id] = case
        candidates = list(unique_candidates.values())
        overlap = all_ids.intersection(unique_candidates)
        if overlap:
            raise EvaluationSuiteError(f"duplicate case id: {sorted(overlap)[0]}")
        all_ids.update(unique_candidates)
        if len(candidates) < GENERAL_100_QUOTA:
            raise EvaluationSuiteError(
                f"{source.source_id} has {len(candidates)} cases; "
                f"{GENERAL_100_QUOTA} are required"
            )
        selected.extend(
            sorted(candidates, key=lambda case: _rank(case, seed))[:GENERAL_100_QUOTA]
        )
    selected.sort(key=lambda case: _rank(case, seed))
    if len(selected) != 4 * GENERAL_100_QUOTA:  # defensive against future source edits
        raise EvaluationSuiteError("general-100 selection must contain exactly 100 cases")
    return selected


def huggingface_dataset_loader(
    source: DatasetSourceV1,
) -> Iterable[Mapping[str, Any]]:
    """Load one preparation-only source through the optional ``datasets`` package."""

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise EvaluationSuiteError(
            "preparing evaluation suites requires the optional 'datasets' package; "
            "install the project's eval-prep dependencies or pass a dataset loader"
        ) from exc
    positional = (source.dataset, source.config) if source.config is not None else (source.dataset,)
    dataset = load_dataset(*positional, split=source.split, revision=source.revision)
    return (dict(row) for row in dataset)


def _jsonl_bytes(cases: Sequence[EvaluationCaseV1]) -> bytes:
    return b"".join(
        _canonical_json_bytes(case.model_dump(mode="json")) + b"\n" for case in cases
    )


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def prepare_general_100(
    output_dir: str | Path,
    *,
    loader: GeneralDatasetLoader = huggingface_dataset_loader,
    sources: Sequence[DatasetSourceV1] = GENERAL_100_SOURCES,
    seed: str = GENERAL_100_SEED,
) -> EvaluationSuiteManifestV1:
    """Prepare ``general-100.v1`` once and atomically publish its directory.

    An already valid suite is returned without loading upstream data.  A present
    but invalid or partial target is never overwritten implicitly.
    """

    destination = Path(output_dir).resolve()
    _validate_general_sources(sources)
    if destination.exists():
        if _manifest_path(destination).is_file():
            manifest, _ = load_frozen_suite(destination)
            if manifest.suite_id != GENERAL_100_SUITE_ID:
                raise EvaluationSuiteError("output directory contains a different suite")
            return manifest
        raise EvaluationSuiteError(
            f"evaluation suite output already exists without a manifest: {destination}"
        )
    records = {source.source_id: list(loader(source)) for source in sources}
    cases = select_general_100(records, sources=sources, seed=seed)
    cases_payload = _jsonl_bytes(cases)
    cases_sha256 = hashlib.sha256(cases_payload).hexdigest()
    group_counts = dict(sorted(Counter(case.group for case in cases).items()))
    selected_ids = [case.case_id for case in cases]
    manifest = EvaluationSuiteManifestV1(
        suite_id=GENERAL_100_SUITE_ID,
        description=(
            "Deterministic 100-case general multiple-choice smoke suite: 25 each "
            "from MMLU, ARC-Challenge, HellaSwag, and WinoGrande."
        ),
        total_cases=len(cases),
        group_counts=group_counts,
        sources=list(sources),
        selection=SelectionEvidenceV1(
            seed=seed,
            selected_ids_sha256=canonical_sha256(selected_ids),
        ),
        files=[
            SuiteFileV1(
                path="cases.jsonl",
                sha256=cases_sha256,
                bytes=len(cases_payload),
            )
        ],
        limitations=[
            "This compact suite is an optimization regression screen, not a leaderboard.",
            "Each 25-case category is reported separately but is too small for a hard gate.",
        ],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        (staging / "cases.jsonl").write_bytes(cases_payload)
        (staging / "manifest.json").write_bytes(
            json.dumps(
                manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def load_frozen_suite(
    suite_dir: str | Path,
) -> tuple[EvaluationSuiteManifestV1, list[EvaluationCaseV1]]:
    """Load a suite without network access and verify every declared byte."""

    root = Path(suite_dir).resolve()
    try:
        raw_manifest = json.loads(_manifest_path(root).read_text(encoding="utf-8"))
        manifest = EvaluationSuiteManifestV1.model_validate(raw_manifest)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise EvaluationSuiteError(f"invalid suite manifest in {root}: {exc}") from exc
    file_by_path = {item.path: item for item in manifest.files}
    case_meta = file_by_path.get("cases.jsonl")
    if case_meta is None:
        raise EvaluationSuiteError("suite manifest does not declare cases.jsonl")
    case_path = root / case_meta.path
    try:
        payload = case_path.read_bytes()
    except OSError as exc:
        raise EvaluationSuiteError(f"cannot read frozen cases: {case_path}") from exc
    if len(payload) != case_meta.bytes or hashlib.sha256(payload).hexdigest() != case_meta.sha256:
        raise EvaluationSuiteError("frozen cases bytes do not match their manifest")
    cases: list[EvaluationCaseV1] = []
    try:
        for line in payload.splitlines():
            if line.strip():
                cases.append(EvaluationCaseV1.model_validate_json(line))
    except ValueError as exc:
        raise EvaluationSuiteError(f"invalid frozen case: {exc}") from exc
    if len(cases) != manifest.total_cases:
        raise EvaluationSuiteError("frozen case count does not match the manifest")
    if len({case.case_id for case in cases}) != len(cases):
        raise EvaluationSuiteError("frozen suite contains duplicate case ids")
    group_counts = dict(sorted(Counter(case.group for case in cases).items()))
    if group_counts != manifest.group_counts:
        raise EvaluationSuiteError("frozen group counts do not match the manifest")
    if canonical_sha256([case.case_id for case in cases]) != (
        manifest.selection.selected_ids_sha256
    ):
        raise EvaluationSuiteError("frozen case selection does not match the manifest")
    return manifest, cases


def _choice_label(index: int) -> str:
    if not 0 <= index < 26:
        raise EvaluationSuiteError("llama multiple-choice supports at most 26 choices")
    return chr(ord("A") + index)


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > 0xFFFFFFFF:
        raise EvaluationSuiteError("multiple-choice string exceeds uint32 size")
    return struct.pack("<I", len(encoded)) + encoded


def serialize_llama_multiple_choice(
    cases: Sequence[EvaluationCaseV1 | Mapping[str, Any]],
) -> bytes:
    """Serialize variable-choice cases in llama-perplexity's binary format."""

    validated = [
        case if isinstance(case, EvaluationCaseV1) else EvaluationCaseV1.model_validate(case)
        for case in cases
    ]
    if not validated:
        raise EvaluationSuiteError("cannot serialize an empty multiple-choice suite")
    tasks: list[bytes] = []
    for case in validated:
        labels = [_choice_label(index) for index in range(len(case.choices))]
        choices_text = "\n".join(
            f"{label}. {choice}" for label, choice in zip(labels, case.choices, strict=True)
        )
        question = f"Question: {case.question}\n{choices_text}\nAnswer:"
        task = bytearray(_pack_string(question))
        task.extend(struct.pack("<I", len(labels)))
        for label in labels:
            task.extend(_pack_string(label))
        task.extend(
            struct.pack(
                f"<{len(labels)}i",
                *(int(index == case.answer_index) for index in range(len(labels))),
            )
        )
        task.extend(struct.pack("<I", 0))  # required empty mc2 answer set
        tasks.append(bytes(task))
    header_size = 4 + 4 * len(tasks)
    positions: list[int] = []
    position = header_size
    for task in tasks:
        if position > 0xFFFFFFFF:
            raise EvaluationSuiteError("multiple-choice fixture exceeds uint32 offsets")
        positions.append(position)
        position += len(task)
    return (
        struct.pack("<I", len(tasks))
        + struct.pack(f"<{len(positions)}I", *positions)
        + b"".join(tasks)
    )


def parse_llama_multiple_choice_result(
    stdout: str,
    stderr: str,
    *,
    cases: Sequence[EvaluationCaseV1],
    suite_id: str,
    variant: Literal["baseline", "candidate"],
    manifest_sha256: str,
    protocol_sha256: str,
) -> QualitySuiteResultV1:
    """Recover exact per-case correctness from llama.cpp cumulative progress."""

    if not cases:
        raise EvaluationSuiteError("cannot parse output for an empty suite")
    if len({case.case_id for case in cases}) != len(cases):
        raise EvaluationSuiteError("cannot parse a suite with duplicate case ids")
    combined = f"{stdout}\n{stderr}"
    final_matches = _FINAL_RESULT_PATTERN.findall(combined)
    if len(final_matches) != 1:
        raise EvaluationSuiteError("output must contain exactly one final result")
    final_percent, uncertainty = (float(value) for value in final_matches[0])
    progress = _PROGRESS_PATTERN.findall(combined)
    if len(progress) != len(cases):
        raise EvaluationSuiteError(
            f"output contains {len(progress)} progress rows; expected {len(cases)}"
        )
    item_results: list[ItemResultV1] = []
    previous_correct = 0
    for expected_index, ((raw_index, raw_percent), case) in enumerate(
        zip(progress, cases, strict=True), 1
    ):
        index = int(raw_index)
        percent = float(raw_percent)
        if index != expected_index or not math.isfinite(percent) or not 0 <= percent <= 100:
            raise EvaluationSuiteError("progress rows must be ordered and within [0, 100]")
        cumulative = int(math.floor(percent * index / 100.0 + 0.5))
        rendered = 100.0 * cumulative / index
        if abs(percent - rendered) > 1e-6 or cumulative - previous_correct not in {0, 1}:
            raise EvaluationSuiteError(
                f"progress row {index} cannot represent a cumulative correct count"
            )
        item_results.append(
            ItemResultV1(case_id=case.case_id, correct=cumulative > previous_correct)
        )
        previous_correct = cumulative
    expected_final = 100.0 * previous_correct / len(cases)
    if not math.isfinite(final_percent) or abs(final_percent - expected_final) > 1e-4:
        raise EvaluationSuiteError("final result does not match progress evidence")
    if not math.isfinite(uncertainty) or uncertainty < 0:
        raise EvaluationSuiteError("final uncertainty is invalid")
    return QualitySuiteResultV1(
        suite_id=suite_id,
        variant=variant,
        status="COMPLETED",
        manifest_sha256=manifest_sha256,
        protocol_sha256=protocol_sha256,
        total=len(cases),
        correct=previous_correct,
        accuracy=previous_correct / len(cases),
        uncertainty_percent=uncertainty,
        items=item_results,
    )


def quality_suite_coordinate(
    suite_dir: str | Path,
) -> QualitySuiteCoordinateV1:
    manifest, _ = load_frozen_suite(suite_dir)
    return QualitySuiteCoordinateV1(
        suite_id=manifest.suite_id,
        manifest_path=str(_manifest_path(Path(suite_dir).resolve())),
        manifest_sha256=sha256_file(_manifest_path(Path(suite_dir).resolve())),
        selected_ids_sha256=manifest.selection.selected_ids_sha256,
    )


def multiple_choice_scorer_protocol_sha256() -> str:
    """Identity of the exact prompt and binary serialization implemented above."""

    return canonical_sha256(
        {
            "schema": "gpuopt.llama-perplexity-multiple-choice-protocol.v1",
            "byte_order": "little-endian",
            "prompt": "Question: {question}\\n{A..Z}. {choice}\\nAnswer:",
            "answer_encoding": "choice-label continuation",
            "mc2": "empty",
            "progress_parser": "cumulative-accuracy-v1",
        }
    )
