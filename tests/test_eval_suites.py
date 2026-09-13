from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from amd_inference_opt.eval_suites import (
    GENERAL_100_SOURCES,
    DatasetSourceV1,
    EvaluationCaseV1,
    EvaluationSuiteError,
    EvaluationSuiteManifestV1,
    ItemResultV1,
    QualityPlanV1,
    QualitySuiteCoordinateV1,
    QualitySuiteResultV1,
    load_frozen_suite,
    multiple_choice_scorer_protocol_sha256,
    parse_llama_multiple_choice_result,
    prepare_general_100,
    quality_suite_coordinate,
    serialize_llama_multiple_choice,
)


def _rows(source: DatasetSourceV1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(30):
        if source.source_id == "mmlu_general":
            rows.append(
                {
                    "id": f"mmlu-{index:03d}",
                    "question": f"MMLU question {index}?",
                    "choices": [f"choice {choice}" for choice in range(4)],
                    "answer": index % 4,
                }
            )
        elif source.source_id == "arc_challenge":
            rows.append(
                {
                    "id": f"arc-{index:03d}",
                    "question": f"ARC question {index}?",
                    "choices": {
                        "label": ["A", "B", "C", "D"],
                        "text": [f"arc choice {choice}" for choice in range(4)],
                    },
                    "answerKey": "ABCD"[index % 4],
                }
            )
        elif source.source_id == "hellaswag":
            rows.append(
                {
                    "ind": index,
                    "ctx": f"HellaSwag context {index}",
                    "endings": [f"ending {choice}" for choice in range(4)],
                    "label": str(index % 4),
                }
            )
        else:
            rows.append(
                {
                    "id": f"wino-{index:03d}",
                    "sentence": f"WinoGrande sentence {index} _.",
                    "option1": "Alice",
                    "option2": "Bob",
                    "answer": str(index % 2 + 1),
                }
            )
    return rows


def _loader(source: DatasetSourceV1):
    return _rows(source)


def _reversed_loader(source: DatasetSourceV1):
    return reversed(_rows(source))


def _case(
    case_id: str, choices: list[str], answer_index: int
) -> EvaluationCaseV1:
    return EvaluationCaseV1(
        case_id=case_id,
        group="fixture",
        question=f"Question {case_id}?",
        choices=choices,
        answer_index=answer_index,
        source_record_id=case_id,
    )


def _unpack_string(payload: bytes, position: int) -> tuple[str, int]:
    length = struct.unpack_from("<I", payload, position)[0]
    position += 4
    return payload[position : position + length].decode(), position + length


def test_prepare_general_100_is_deterministic_and_offline_consumable(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first/general-100.v1"
    second_dir = tmp_path / "second/general-100.v1"
    first = prepare_general_100(first_dir, loader=_loader)
    second = prepare_general_100(second_dir, loader=_reversed_loader)

    assert first == second
    assert first.total_cases == 100
    assert first.group_counts == {
        "arc_challenge": 25,
        "hellaswag": 25,
        "mmlu_general": 25,
        "winogrande": 25,
    }
    assert (first_dir / "cases.jsonl").read_bytes() == (
        second_dir / "cases.jsonl"
    ).read_bytes()
    assert (first_dir / "manifest.json").read_bytes() == (
        second_dir / "manifest.json"
    ).read_bytes()

    loaded_manifest, cases = load_frozen_suite(first_dir)
    assert loaded_manifest == first
    assert len(cases) == 100
    assert all(len(case.choices) in {2, 4} for case in cases)


def test_existing_valid_suite_is_reused_without_calling_loader(tmp_path: Path) -> None:
    output = tmp_path / "general-100.v1"
    original = prepare_general_100(output, loader=_loader)

    def forbidden_loader(source: DatasetSourceV1):
        raise AssertionError(f"loader called for {source.source_id}")

    assert prepare_general_100(output, loader=forbidden_loader) == original


def test_partial_output_and_tampered_fixture_are_rejected(tmp_path: Path) -> None:
    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises(EvaluationSuiteError, match="without a manifest"):
        prepare_general_100(partial, loader=_loader)

    output = tmp_path / "suite"
    prepare_general_100(output, loader=_loader)
    with (output / "cases.jsonl").open("ab") as target:
        target.write(b"{}\n")
    with pytest.raises(EvaluationSuiteError, match="bytes do not match"):
        load_frozen_suite(output)
    with pytest.raises(EvaluationSuiteError, match="bytes do not match"):
        prepare_general_100(output, loader=_loader)


def test_manifest_records_revision_license_file_hash_and_coordinate(
    tmp_path: Path,
) -> None:
    output = tmp_path / "suite"
    manifest = prepare_general_100(output, loader=_loader)
    source = {item.source_id: item for item in manifest.sources}

    assert source["mmlu_general"].revision == (
        "c30699e8356da336a370243923dbaf21066bb9fe"
    )
    assert source["arc_challenge"].license == "CC-BY-SA-4.0"
    cases = output / "cases.jsonl"
    assert manifest.files[0].sha256 == hashlib.sha256(cases.read_bytes()).hexdigest()
    coordinate = quality_suite_coordinate(output)
    assert coordinate.manifest_sha256 == hashlib.sha256(
        (output / "manifest.json").read_bytes()
    ).hexdigest()
    assert coordinate.selected_ids_sha256 == manifest.selection.selected_ids_sha256


def test_variable_choice_serializer_matches_llama_binary_layout() -> None:
    cases = [
        _case("two", ["yes", "no"], 1),
        _case("five", ["a", "b", "c", "d", "e"], 4),
    ]
    payload = serialize_llama_multiple_choice(cases)
    task_count = struct.unpack_from("<I", payload)[0]
    offsets = struct.unpack_from("<2I", payload, 4)

    assert task_count == 2
    for expected_count, expected_answer, offset in zip(
        (2, 5), (1, 4), offsets, strict=True
    ):
        prompt, position = _unpack_string(payload, offset)
        choice_count = struct.unpack_from("<I", payload, position)[0]
        position += 4
        labels: list[str] = []
        for _ in range(choice_count):
            label, position = _unpack_string(payload, position)
            labels.append(label)
        correct = struct.unpack_from(f"<{choice_count}i", payload, position)
        position += 4 * choice_count
        mc2_count = struct.unpack_from("<I", payload, position)[0]
        assert choice_count == expected_count
        assert labels == [chr(ord("A") + index) for index in range(expected_count)]
        assert correct == tuple(int(index == expected_answer) for index in range(expected_count))
        assert mc2_count == 0
        assert prompt.endswith("\nAnswer:")


def test_parse_llama_result_recovers_per_item_correctness() -> None:
    cases = [
        _case("one", ["a", "b"], 0),
        _case("two", ["a", "b", "c"], 1),
        _case("three", ["a", "b", "c", "d", "e"], 4),
    ]
    stderr = "1 100.000000\n2 50.000000\n3 66.666667\nFinal result: 66.6667 +/- 2.5000\n"
    result = parse_llama_multiple_choice_result(
        "",
        stderr,
        cases=cases,
        suite_id="fixture.v1",
        variant="candidate",
        manifest_sha256="a" * 64,
        protocol_sha256="b" * 64,
    )

    assert result.status == "COMPLETED"
    assert result.correct == 2
    assert result.accuracy == pytest.approx(2 / 3)
    assert [item.correct for item in result.items] == [True, False, True]
    with pytest.raises(EvaluationSuiteError, match="final result"):
        parse_llama_multiple_choice_result(
            "",
            stderr.replace("66.6667", "100.0000"),
            cases=cases,
            suite_id="fixture.v1",
            variant="candidate",
            manifest_sha256="a" * 64,
            protocol_sha256="b" * 64,
        )


def test_quality_plan_and_result_are_strict_and_hash_bound() -> None:
    coordinate = QualitySuiteCoordinateV1(
        suite_id="fixture.v1",
        manifest_path="/frozen/manifest.json",
        manifest_sha256="a" * 64,
        selected_ids_sha256="b" * 64,
    )
    plan = QualityPlanV1(
        suites=[coordinate],
        scorer_protocol_sha256=multiple_choice_scorer_protocol_sha256(),
    )
    assert len(plan.protocol_sha256) == 64
    relocated = plan.model_copy(
        update={
            "suites": [coordinate.model_copy(update={"manifest_path": "/elsewhere/manifest.json"})]
        }
    )
    assert relocated.protocol_sha256 == plan.protocol_sha256
    result = QualitySuiteResultV1(
        suite_id="fixture.v1",
        variant="baseline",
        status="COMPLETED",
        manifest_sha256="a" * 64,
        protocol_sha256=plan.protocol_sha256,
        total=1,
        correct=1,
        accuracy=1,
        uncertainty_percent=0,
        items=[ItemResultV1(case_id="case", correct=True)],
    )
    assert result.accuracy == 1
    with pytest.raises(ValidationError):
        EvaluationSuiteManifestV1.model_validate(
            {
                "schema_version": "gpuopt.evaluation-suite-manifest.v1",
                "suite_id": "fixture",
                "kind": "multiple_choice",
                "description": "fixture",
                "total_cases": 1,
                "group_counts": {"fixture": 1},
                "sources": [GENERAL_100_SOURCES[0].model_dump()],
                "selection": {
                    "method": "sha256_rank_v1",
                    "seed": "seed",
                    "rank_input": "utf8(seed + NUL + group + NUL + case_id)",
                    "selected_ids_sha256": "c" * 64,
                },
                "files": [
                    {
                        "path": "cases.jsonl",
                        "sha256": "d" * 64,
                        "bytes": 1,
                        "media_type": "application/jsonl",
                    }
                ],
                "unexpected": True,
            }
        )


def test_invalid_source_row_and_insufficient_group_are_clear(tmp_path: Path) -> None:
    def insufficient(source: DatasetSourceV1):
        return _rows(source)[:24]

    with pytest.raises(EvaluationSuiteError, match="25 are required"):
        prepare_general_100(tmp_path / "insufficient", loader=insufficient)

    def invalid_arc(source: DatasetSourceV1):
        rows = _rows(source)
        if source.source_id == "arc_challenge":
            rows[0]["answerKey"] = "Z"
        return rows

    with pytest.raises(EvaluationSuiteError, match="absent from choice labels"):
        prepare_general_100(tmp_path / "invalid", loader=invalid_arc)

    with pytest.raises(EvaluationSuiteError, match="requires exactly"):
        prepare_general_100(
            tmp_path / "missing-source",
            loader=_loader,
            sources=GENERAL_100_SOURCES[:-1],
        )


def test_identical_upstream_duplicates_are_deduplicated(tmp_path: Path) -> None:
    def duplicated(source: DatasetSourceV1):
        rows = _rows(source)
        if source.source_id == "mmlu_general":
            rows.append(dict(rows[0]))
        return rows

    manifest = prepare_general_100(tmp_path / "deduplicated", loader=duplicated)
    assert manifest.total_cases == 100


def test_hellaswag_reused_ind_values_do_not_collide(tmp_path: Path) -> None:
    def reused_index(source: DatasetSourceV1):
        rows = _rows(source)
        if source.source_id == "hellaswag":
            rows[1]["ind"] = rows[0]["ind"]
        return rows

    manifest = prepare_general_100(tmp_path / "reused-ind", loader=reused_index)
    assert manifest.total_cases == 100


def test_serializer_rejects_empty_suite() -> None:
    with pytest.raises(EvaluationSuiteError, match="empty"):
        serialize_llama_multiple_choice([])


def test_manifest_json_is_plain_versioned_data(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    prepare_general_100(output, loader=_loader)
    raw = json.loads((output / "manifest.json").read_text())
    assert raw["schema_version"] == "gpuopt.evaluation-suite-manifest.v1"
    assert raw["suite_id"] == "general-100.v1"
