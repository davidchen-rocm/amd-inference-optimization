#!/usr/bin/env python3
"""Run the frozen provisional Math-100 suite against a vLLM completions API.

This is deliberately a small evidence tool, not an authoritative evaluation
harness.  It constrains generation to the four answer-letter token IDs so a
reasoning model cannot spend the one-token budget on hidden reasoning text.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "gpuopt.vllm-provisional-math100.v1"
SEED = 20260815
QUOTAS = {
    "abstract_algebra": 12,
    "college_mathematics": 12,
    "high_school_mathematics": 32,
    "elementary_mathematics": 44,
}
EXPECTED_IDS_SHA256 = (
    "5da1fc7d2c551b081bd21f38326482190a42fa44edaef88a701a7a4e961417d5"
)
EXPECTED_ORDER_SHA256 = (
    "11670b5040127d9713742b935a743fe932edbe3cc83f53202962e22929365841"
)
EXPECTED_ANSWERS_SHA256 = (
    "9531d710030067deb85355e29e3adc62cfd38b6396eac6c4ed862c397a2b446b"
)
ANSWER_PATTERN = re.compile(r"^[ABCD]$")


class EvaluationError(RuntimeError):
    """The input, server response, or output is not safe to accept."""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(item_id: str) -> bytes:
    return hashlib.sha256(f"{SEED}\0{item_id}".encode()).digest()


def _load_selection(fixture: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("math", {}).get("sha256")
    actual = _sha256_file(fixture)
    if expected != actual:
        raise EvaluationError("math fixture SHA-256 does not match its manifest")
    items = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
    if len(items) != 848:
        raise EvaluationError("the provisional protocol requires all 848 source items")
    ids = [item.get("id") for item in items]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise EvaluationError("every math item needs a non-empty string id")
    if len(ids) != len(set(ids)):
        raise EvaluationError("math item ids are not unique")

    selected: list[dict[str, Any]] = []
    counts = Counter(item.get("subject") for item in items)
    for subject, quota in QUOTAS.items():
        if counts[subject] < quota:
            raise EvaluationError(f"fixture does not contain {quota} {subject} items")
        subject_items = [item for item in items if item.get("subject") == subject]
        subject_items.sort(key=lambda item: (_rank(item["id"]), item["id"]))
        selected.extend(subject_items[:quota])
    selected.sort(key=lambda item: (_rank(item["id"]), item["id"]))

    selected_ids = [item["id"] for item in selected]
    answers = ["ABCD"[item["answer_index"]] for item in selected]
    identities = {
        "ids": _canonical_sha256(sorted(selected_ids)),
        "order": _canonical_sha256(selected_ids),
        "answers": _canonical_sha256(answers),
    }
    expected_identities = {
        "ids": EXPECTED_IDS_SHA256,
        "order": EXPECTED_ORDER_SHA256,
        "answers": EXPECTED_ANSWERS_SHA256,
    }
    if identities != expected_identities:
        raise EvaluationError("the frozen Math-100 selection identity drifted")
    return selected, actual


def _prompt(item: dict[str, Any]) -> str:
    choices = "\n".join(
        f"{'ABCD'[index]}. {choice}" for index, choice in enumerate(item["choices"])
    )
    return (
        "Choose the correct answer. Return exactly one letter: A, B, C, or D.\n"
        f"Question: {item['question']}\n{choices}\nAnswer:"
    )


def _evaluate_item(
    *,
    base_url: str,
    model: str,
    item: dict[str, Any],
    allowed_token_ids: list[int],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": _prompt(item),
        "max_tokens": 1,
        "temperature": 0,
        "allowed_token_ids": allowed_token_ids,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise EvaluationError(f"request for {item['id']} failed: {error}") from error
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise EvaluationError(f"request for {item['id']} returned invalid choices")
    text = choices[0].get("text")
    prediction = text.strip() if isinstance(text, str) else ""
    if not ANSWER_PATTERN.fullmatch(prediction):
        raise EvaluationError(
            f"request for {item['id']} did not return exactly one answer letter"
        )
    answer = "ABCD"[item["answer_index"]]
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "id": item["id"],
        "subject": item["subject"],
        "answer": answer,
        "prediction": prediction,
        "correct": prediction == answer,
        "latency_seconds": time.monotonic() - started,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "response_id": payload.get("id"),
        "system_fingerprint": payload.get("system_fingerprint"),
    }


def _atomic_write_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise EvaluationError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--allowed-token-id", type=int, action="append", required=True)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 32:
        raise EvaluationError("concurrency must be between 1 and 32")
    if len(set(args.allowed_token_id)) != len(args.allowed_token_id):
        raise EvaluationError("allowed token ids must be unique")

    selected, fixture_sha256 = _load_selection(args.fixture, args.manifest)
    protocol = {
        "id": "provisional-math-100.v1-vllm-letter-generation",
        "authoritative": False,
        "evidence_only": True,
        "seed": SEED,
        "quotas": QUOTAS,
        "source_fixture_sha256": fixture_sha256,
        "selected_ids_sha256": EXPECTED_ORDER_SHA256,
        "prompt_template": "single-letter-four-choice.v1",
        "max_tokens": 1,
        "temperature": 0,
        "allowed_token_ids": sorted(args.allowed_token_id),
    }
    protocol["protocol_sha256"] = _canonical_sha256(protocol)
    started_at = datetime.now(UTC)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                _evaluate_item,
                base_url=args.base_url,
                model=args.model,
                item=item,
                allowed_token_ids=args.allowed_token_id,
                timeout_seconds=args.timeout_seconds,
            )
            for item in selected
        ]
        results = [future.result() for future in futures]
    results_by_id = {result["id"]: result for result in results}
    ordered_results = [results_by_id[item["id"]] for item in selected]
    correct = sum(bool(result["correct"]) for result in ordered_results)
    document = {
        "schema": SCHEMA,
        "status": "complete",
        "model": args.model,
        "server_base_url": args.base_url,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "protocol": protocol,
        "score": {"correct": correct, "total": 100, "accuracy": correct / 100},
        "subjects": {
            subject: {
                "correct": sum(
                    result["correct"]
                    for result in ordered_results
                    if result["subject"] == subject
                ),
                "total": quota,
            }
            for subject, quota in QUOTAS.items()
        },
        "predictions_sha256": _canonical_sha256(
            [{"id": result["id"], "prediction": result["prediction"]} for result in ordered_results]
        ),
        "results": ordered_results,
        "limitations": [
            "This provisional 100-question subset is evidence-only, not an "
            "authoritative acceptance gate.",
            "Answers are one-token constrained letter generations, not normalized "
            "option log-probabilities.",
        ],
    }
    _atomic_write_json(args.output, document)
    print(json.dumps({"output": str(args.output), **document["score"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
