from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "vllm_mixed500_eval.py"
    spec = importlib.util.spec_from_file_location("vllm_mixed500_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalizes_all_four_sources() -> None:
    module = _module()
    cases = [
        module._normalize(
            "mmlu_pro",
            {
                "gpuopt_case_id": "m:1",
                "question": "q",
                "options": ["x", "y"],
                "answer": "B",
                "category": "math",
            },
        ),
        module._normalize(
            "arc_challenge",
            {
                "gpuopt_case_id": "a:1",
                "question": "q",
                "choices": {"label": ["1", "2"], "text": ["x", "y"]},
                "answerKey": "2",
            },
        ),
        module._normalize(
            "hellaswag",
            {
                "gpuopt_case_id": "h:1",
                "ctx": "q",
                "endings": ["x", "y"],
                "label": "0",
                "activity_label": "activity",
            },
        ),
        module._normalize(
            "winogrande",
            {
                "gpuopt_case_id": "w:1",
                "sentence": "q",
                "option1": "x",
                "option2": "y",
                "answer": "2",
            },
        ),
    ]
    assert [case["answer"] for case in cases] == ["B", "B", "A", "B"]
    assert [case["source"] for case in cases] == list(module.QUOTAS)


def test_selection_is_exact_and_deterministic(tmp_path: Path) -> None:
    module = _module()
    data = tmp_path / "data"
    data.mkdir()
    rows = {
        "mmlu_pro": lambda i: {
            "gpuopt_case_id": f"m:{i}",
            "question": "q",
            "options": ["x", "y"],
            "answer": "A",
            "category": "math",
        },
        "arc_challenge": lambda i: {
            "gpuopt_case_id": f"a:{i}",
            "question": "q",
            "choices": {"label": ["A", "B"], "text": ["x", "y"]},
            "answerKey": "A",
        },
        "hellaswag": lambda i: {
            "gpuopt_case_id": f"h:{i}",
            "ctx": "q",
            "endings": ["x", "y"],
            "label": "0",
        },
        "winogrande": lambda i: {
            "gpuopt_case_id": f"w:{i}",
            "sentence": "q",
            "option1": "x",
            "option2": "y",
            "answer": "1",
        },
    }
    sources = []
    for source_id, make in rows.items():
        path = data / f"{source_id}.jsonl"
        path.write_text("".join(json.dumps(make(i)) + "\n" for i in range(130)))
        sources.append(
            {
                "source_id": source_id,
                "records": 130,
                "sha256": module._sha256_file(path),
            }
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": sources}))
    first, hashes = module._load_selection(tmp_path)
    second, _ = module._load_selection(tmp_path)
    assert len(first) == 500
    assert first == second
    assert set(hashes) == set(module.QUOTAS)
    actual_counts = {
        source: sum(case["source"] == source for case in first)
        for source in module.QUOTAS
    }
    assert actual_counts == module.QUOTAS


def test_prompt_has_only_indexed_choices() -> None:
    module = _module()
    prompt = module._prompt({"question": "q", "options": ["x", "y"]})
    assert "A. x" in prompt
    assert "B. y" in prompt
    assert "exactly one letter" in prompt


def test_unicode_line_separator_inside_json_does_not_split_record(tmp_path: Path) -> None:
    module = _module()
    data = tmp_path / "data"
    data.mkdir()
    sources = []
    makers = {
        "mmlu_pro": lambda i: {
            "gpuopt_case_id": f"m:{i}",
            "question": "q\u2028continued",
            "options": ["x", "y"],
            "answer": "A",
            "category": "math",
        },
        "arc_challenge": lambda i: {
            "gpuopt_case_id": f"a:{i}",
            "question": "q",
            "choices": {"label": ["A", "B"], "text": ["x", "y"]},
            "answerKey": "A",
        },
        "hellaswag": lambda i: {
            "gpuopt_case_id": f"h:{i}",
            "ctx": "q",
            "endings": ["x", "y"],
            "label": "0",
        },
        "winogrande": lambda i: {
            "gpuopt_case_id": f"w:{i}",
            "sentence": "q",
            "option1": "x",
            "option2": "y",
            "answer": "1",
        },
    }
    for source_id, make in makers.items():
        path = data / f"{source_id}.jsonl"
        path.write_text(
            "".join(
                json.dumps(make(i), ensure_ascii=False) + "\n" for i in range(125)
            )
        )
        sources.append(
            {
                "source_id": source_id,
                "records": 125,
                "sha256": module._sha256_file(path),
            }
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": sources}))
    cases, _ = module._load_selection(tmp_path)
    assert len(cases) == 500
    assert any("\u2028" in case["question"] for case in cases)
