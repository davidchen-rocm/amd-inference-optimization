#!/usr/bin/env python3
"""Freeze locally cached MATH-500 and MMLU math data for the Q8 runtime evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SUBJECTS = (
    "abstract_algebra",
    "college_mathematics",
    "high_school_mathematics",
    "elementary_mathematics",
)
SCHEMA = "gpuopt.q8-runtime-quality-fixture.v1"
SEED = 20260815


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # This dependency deliberately stays out of the framework runtime. The preparation
    # command is expected to use math-rule-loop's existing lighteval environment.
    from datasets import disable_progress_bar, load_dataset  # type: ignore[import-not-found]

    disable_progress_bar()
    output_dir = Path(args.output_dir).resolve()
    math_path = output_dir / "mmlu-math.jsonl"
    ppl_path = output_dir / "math500-ppl.txt"
    manifest_path = output_dir / "manifest.json"

    math_lines: list[str] = []
    subject_counts: dict[str, int] = {}
    for subject in SUBJECTS:
        dataset = load_dataset("cais/mmlu", subject, split="test").shuffle(seed=args.seed)
        subject_counts[subject] = len(dataset)
        for index, example in enumerate(dataset):
            record = {
                "id": f"{subject}-{index:04d}",
                "subject": subject,
                "question": example["question"],
                "choices": list(example["choices"]),
                "answer_index": int(example["answer"]),
            }
            math_lines.append(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
    write_text_atomic(math_path, "\n".join(math_lines) + "\n")

    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ppl_text = "\n\n".join(
        f"Problem: {row['problem']}\nSolution: {row['solution']}" for row in math500
    )
    write_text_atomic(ppl_path, ppl_text + "\n")

    manifest = {
        "schema": SCHEMA,
        "seed": args.seed,
        "sources": {
            "math_accuracy": "cais/mmlu test split, four math subjects",
            "perplexity": "HuggingFaceH4/MATH-500 test problem+solution concatenation",
        },
        "math": {
            "path": math_path.name,
            "sha256": sha256_file(math_path),
            "total": len(math_lines),
            "subjects": subject_counts,
        },
        "perplexity": {
            "path": ppl_path.name,
            "sha256": sha256_file(ppl_path),
            "examples": len(math500),
            "bytes": ppl_path.stat().st_size,
        },
        "limitations": [
            (
                "MMLU uses llama.cpp's normalized continuation log-prob scorer, "
                "not an official leaderboard harness."
            ),
            "Perplexity covers a fixed MATH-500-derived corpus, not WikiText-2.",
        ],
    }
    write_text_atomic(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
