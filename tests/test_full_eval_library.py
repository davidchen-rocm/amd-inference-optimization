from __future__ import annotations

from pathlib import Path

import pytest

from amd_inference_opt.full_eval_library import (
    SOURCE_COORDINATES,
    FullEvaluationError,
    load_english_full,
    prepare_english_full,
)


def _records() -> dict[str, list[dict[str, object]]]:
    return {
        "mmlu_pro": [{"question_id": 1, "question": "q", "answer_index": 0}],
        "arc_challenge": [{"id": "arc-1", "question": "q"}],
        "hellaswag": [{"ind": 1, "ctx": "context", "label": "0"}],
        "winogrande": [{"sentence": "_ won", "answer": "1"}],
        "math_500": [{"unique_id": "math-1", "problem": "1+1", "answer": "2"}],
        "gsm8k": [{"question": "1+1", "answer": "2"}],
        "ifeval": [{"key": 1000, "prompt": "Return JSON"}],
        "livecodebench": [{"question_id": "code-1", "question_content": "solve"}],
        "longbench_en": [
            {
                "benchmark_task": "qasper",
                "_id": "long-1",
                "input": "question",
                "context": "context",
                "answers": ["answer"],
                "language": "en",
            }
        ],
    }


def test_prepare_full_library_is_hash_bound_and_reusable(tmp_path: Path) -> None:
    output = tmp_path / "english-full.v1"
    first = prepare_english_full(output, records_by_source=_records())
    second = prepare_english_full(output, records_by_source={})

    assert first == second
    assert first.total_records == 9
    assert {source.source_id for source in first.sources} == set(_records())
    assert all(source.records == 1 for source in first.sources)
    assert load_english_full(output) == first


def test_full_library_rejects_tampering_and_missing_sources(tmp_path: Path) -> None:
    output = tmp_path / "english-full.v1"
    manifest = prepare_english_full(output, records_by_source=_records())
    first = output / manifest.sources[0].path
    with first.open("ab") as target:
        target.write(b"{}\n")
    with pytest.raises(FullEvaluationError, match="hash/size mismatch"):
        load_english_full(output)

    incomplete = _records()
    incomplete.pop("ifeval")
    with pytest.raises(FullEvaluationError, match="omitted source ifeval"):
        prepare_english_full(tmp_path / "incomplete", records_by_source=incomplete)


def test_full_library_pins_every_upstream_revision_and_excludes_chinese() -> None:
    assert all(coordinate["revision"] != "main" for coordinate in SOURCE_COORDINATES)
    assert {coordinate["source_id"] for coordinate in SOURCE_COORDINATES} == set(_records())
    longbench = next(
        coordinate
        for coordinate in SOURCE_COORDINATES
        if coordinate["source_id"] == "longbench_en"
    )
    assert "English" in str(longbench["config"])
