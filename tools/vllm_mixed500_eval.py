#!/usr/bin/env python3
"""Evaluate a deterministic 500-case mixed English suite through vLLM."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import string
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "gpuopt.vllm-mixed-general-500.v1"
CHECKPOINT_SCHEMA = "gpuopt.vllm-mixed-general-500-checkpoint.v1"
SEED = "gpuopt-mixed-general-500-v1"
QUOTAS = {
    "mmlu_pro": 125,
    "arc_challenge": 125,
    "hellaswag": 125,
    "winogrande": 125,
}
FILES = {source_id: f"data/{source_id}.jsonl" for source_id in QUOTAS}
LETTERS = string.ascii_uppercase[:10]


class EvaluationError(RuntimeError):
    """The suite, server response, checkpoint, or output is invalid."""


def _canonical_sha256(value: object) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object, *, replace: bool) -> None:
    if path.is_symlink() or (path.exists() and not replace):
        raise EvaluationError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _rank(source_id: str, case_id: str) -> bytes:
    return hashlib.sha256(f"{SEED}\0{source_id}\0{case_id}".encode()).digest()


def _normalize(source_id: str, row: dict[str, Any]) -> dict[str, Any]:
    case_id = row.get("gpuopt_case_id")
    if not isinstance(case_id, str) or not case_id:
        raise EvaluationError(f"{source_id} row has no gpuopt_case_id")
    if source_id == "mmlu_pro":
        options = row.get("options")
        answer = row.get("answer")
        question = row.get("question")
        group = row.get("category")
    elif source_id == "arc_challenge":
        choices = row.get("choices")
        if not isinstance(choices, dict):
            raise EvaluationError("ARC row has no choices")
        labels, options = choices.get("label"), choices.get("text")
        answer_key = row.get("answerKey")
        if not isinstance(labels, list) or answer_key not in labels:
            raise EvaluationError("ARC answer key does not match its choices")
        answer = LETTERS[labels.index(answer_key)]
        question = row.get("question")
        group = "arc_challenge"
    elif source_id == "hellaswag":
        options = row.get("endings")
        label = row.get("label")
        if not isinstance(label, str) or not label.isdigit():
            raise EvaluationError("HellaSwag row has an invalid label")
        answer = LETTERS[int(label)]
        question = row.get("ctx")
        group = row.get("activity_label") or "hellaswag"
    elif source_id == "winogrande":
        options = [row.get("option1"), row.get("option2")]
        label = row.get("answer")
        if label not in {"1", "2"}:
            raise EvaluationError("WinoGrande row has an invalid answer")
        answer = LETTERS[int(label) - 1]
        question = row.get("sentence")
        group = "winogrande"
    else:
        raise EvaluationError(f"unsupported source: {source_id}")
    if (
        not isinstance(question, str)
        or not question
        or not isinstance(options, list)
        or not 2 <= len(options) <= len(LETTERS)
        or any(not isinstance(option, str) or not option for option in options)
        or answer not in LETTERS[: len(options)]
    ):
        raise EvaluationError(f"invalid normalized {source_id} case: {case_id}")
    return {
        "id": case_id,
        "source": source_id,
        "group": group if isinstance(group, str) and group else source_id,
        "question": question,
        "options": options,
        "answer": answer,
    }


def _load_selection(suite_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    manifest_path = suite_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise EvaluationError("english-full manifest has no sources")
    by_id = {source.get("source_id"): source for source in sources}
    selected: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for source_id, quota in QUOTAS.items():
        source = by_id.get(source_id)
        if not isinstance(source, dict):
            raise EvaluationError(f"manifest has no {source_id} source")
        path = suite_dir / FILES[source_id]
        actual = _sha256_file(path)
        if actual != source.get("sha256"):
            raise EvaluationError(f"{source_id} source SHA-256 drifted")
        with path.open(encoding="utf-8") as input_file:
            rows = [json.loads(line) for line in input_file]
        if len(rows) != source.get("records"):
            raise EvaluationError(f"{source_id} source record count drifted")
        cases = [_normalize(source_id, row) for row in rows]
        if len({case["id"] for case in cases}) != len(cases):
            raise EvaluationError(f"{source_id} case ids are not unique")
        cases.sort(key=lambda case: (_rank(source_id, case["id"]), case["id"]))
        selected.extend(cases[:quota])
        source_hashes[source_id] = actual
    selected.sort(key=lambda case: (_rank(case["source"], case["id"]), case["id"]))
    if len(selected) != 500:
        raise EvaluationError("mixed suite selection must contain exactly 500 cases")
    return selected, source_hashes


def _prompt(case: dict[str, Any]) -> str:
    options = "\n".join(
        f"{LETTERS[index]}. {option}" for index, option in enumerate(case["options"])
    )
    return (
        "Choose the correct answer. Return exactly one letter and nothing else.\n"
        f"Question: {case['question']}\n{options}\nAnswer:"
    )


def _request_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise EvaluationError(f"request failed: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationError("server response is not a JSON object")
    return value


def _token_ids(base_url: str, model: str, timeout: float) -> list[int]:
    token_ids: set[int] = set()
    for letter in LETTERS:
        for text in (letter, " " + letter):
            payload = _request_json(
                base_url.rstrip("/") + "/tokenize",
                {"model": model, "prompt": text},
                timeout,
            )
            tokens = payload.get("tokens")
            if not isinstance(tokens, list) or len(tokens) != 1 or not isinstance(tokens[0], int):
                raise EvaluationError(f"{text!r} is not one tokenizer token")
            token_ids.add(tokens[0])
    if len(token_ids) != 20:
        raise EvaluationError("answer token ids are not distinct")
    return sorted(token_ids)


def _evaluate_case(
    *,
    base_url: str,
    model: str,
    case: dict[str, Any],
    allowed_token_ids: list[int],
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    payload = _request_json(
        base_url.rstrip("/") + "/v1/completions",
        {
            "model": model,
            "prompt": _prompt(case),
            "max_tokens": 1,
            "temperature": 0,
            "seed": 42,
            "allowed_token_ids": allowed_token_ids,
        },
        timeout,
    )
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise EvaluationError(f"{case['id']} returned invalid choices")
    text = choices[0].get("text")
    prediction = text.strip() if isinstance(text, str) else ""
    if prediction not in LETTERS[: len(case["options"])]:
        raise EvaluationError(f"{case['id']} returned invalid answer {prediction!r}")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "id": case["id"],
        "source": case["source"],
        "group": case["group"],
        "answer": case["answer"],
        "prediction": prediction,
        "correct": prediction == case["answer"],
        "latency_seconds": time.monotonic() - started,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "response_id": payload.get("id"),
    }


def _save_checkpoint(
    path: Path, protocol_sha256: str, results: dict[str, dict[str, Any]]
) -> None:
    _atomic_write_json(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "protocol_sha256": protocol_sha256,
            "completed": len(results),
            "results": results,
        },
        replace=path.exists(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 32:
        raise EvaluationError("concurrency must be between 1 and 32")
    if args.output.exists() or args.output.is_symlink():
        raise EvaluationError(f"refusing to overwrite output: {args.output}")

    cases, source_hashes = _load_selection(args.suite_dir)
    allowed_token_ids = _token_ids(args.base_url, args.model, args.timeout_seconds)
    protocol = {
        "id": "mixed-general-500.v1-vllm-letter-generation",
        "seed": SEED,
        "quotas": QUOTAS,
        "source_sha256": source_hashes,
        "selected_ids_sha256": _canonical_sha256([case["id"] for case in cases]),
        "prompt_template": "single-letter-mixed-choice.v1",
        "model": args.model,
        "temperature": 0,
        "max_tokens": 1,
        "allowed_token_ids": allowed_token_ids,
    }
    protocol_sha256 = _canonical_sha256(protocol)
    results: dict[str, dict[str, Any]] = {}
    if args.checkpoint.exists():
        saved = json.loads(args.checkpoint.read_text())
        if (
            saved.get("schema") != CHECKPOINT_SCHEMA
            or saved.get("protocol_sha256") != protocol_sha256
            or not isinstance(saved.get("results"), dict)
        ):
            raise EvaluationError("checkpoint does not match the current protocol")
        results = saved["results"]

    pending = [case for case in cases if case["id"] not in results]
    started_at = datetime.now(UTC)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _evaluate_case,
                base_url=args.base_url,
                model=args.model,
                case=case,
                allowed_token_ids=allowed_token_ids,
                timeout=args.timeout_seconds,
            ): case
            for case in pending
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results[result["id"]] = result
            _save_checkpoint(args.checkpoint, protocol_sha256, results)
            print(json.dumps({"completed": len(results), "total": 500}), flush=True)

    ordered = [results[case["id"]] for case in cases]
    by_source = {}
    for source_id, total in QUOTAS.items():
        correct = sum(r["correct"] for r in ordered if r["source"] == source_id)
        by_source[source_id] = {"correct": correct, "total": total, "accuracy": correct / total}
    group_counts = Counter(result["group"] for result in ordered)
    by_group = {
        group: {
            "correct": sum(r["correct"] for r in ordered if r["group"] == group),
            "total": total,
        }
        for group, total in sorted(group_counts.items())
    }
    correct = sum(result["correct"] for result in ordered)
    document = {
        "schema": SCHEMA,
        "status": "complete",
        "model": args.model,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "protocol": {**protocol, "protocol_sha256": protocol_sha256},
        "score": {"correct": correct, "total": 500, "accuracy": correct / 500},
        "sources": by_source,
        "groups": by_group,
        "predictions_sha256": _canonical_sha256(
            [{"id": result["id"], "prediction": result["prediction"]} for result in ordered]
        ),
        "results": ordered,
        "limitations": [
            "This mixed gate covers four compatible multiple-choice scorers.",
            "IFEval, LiveCodeBench, and LongBench require separate task-specific scorers.",
        ],
    }
    _atomic_write_json(args.output, document, replace=False)
    print(json.dumps({"output": str(args.output), **document["score"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
