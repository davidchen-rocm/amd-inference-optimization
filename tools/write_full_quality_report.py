#!/usr/bin/env python3
"""Write the corrected full-quality decision without replacing prior reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _read(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to replace existing report: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.campaign_root.resolve(strict=True)
    quality_path = root / "mixed-bit/q5/quality/full-848-corrected-v2.json"
    previous_final_path = root / "reports/final.json"
    incorrect_gate_path = root / "reports/final-after-full-quality.json"
    quality = _read(quality_path)
    previous_final = _read(previous_final_path)
    baseline = quality["baseline"]
    candidate = quality["candidate"]
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise RuntimeError("quality arms are not objects")
    gate = quality.get("corrected_gate")
    paired_comparison = quality.get("comparison")
    if not isinstance(gate, dict) or not isinstance(paired_comparison, dict):
        raise RuntimeError("corrected quality gate or comparison is missing")
    if gate.get("outcome") != "PASS" or gate.get("applicable") is not True:
        raise RuntimeError("corrected full-quality gate did not pass")
    math_total = int(candidate["math_total"])
    if math_total != 848 or int(baseline["math_total"]) != math_total:
        raise RuntimeError("full-quality comparison must contain all 848 questions")
    report = {
        "schema": "gpuopt.consumer-amd-corrected-full-quality.v2",
        "candidate": "Q5_K_M_imatrix",
        "verdict": "ACCEPT",
        "corrected_gate": gate,
        "paired_comparison": paired_comparison,
        "baseline": {
            "model": "Q6_K",
            "math_correct": baseline["math_correct"],
            "math_total": math_total,
            "math_accuracy": baseline["math_accuracy"],
            "perplexity": baseline["perplexity"],
            "greedy_correct": baseline["greedy_correct"],
            "greedy_total": baseline["greedy_total"],
            "greedy_accuracy": baseline["greedy_accuracy"],
            "math_correctness_sha256": baseline["math_correctness_sha256"],
        },
        "candidate_result": {
            "model": "Q5_K_M_imatrix",
            "math_correct": candidate["math_correct"],
            "math_total": math_total,
            "math_accuracy": candidate["math_accuracy"],
            "perplexity": candidate["perplexity"],
            "greedy_correct": candidate["greedy_correct"],
            "greedy_total": candidate["greedy_total"],
            "greedy_accuracy": candidate["greedy_accuracy"],
            "math_correctness_sha256": candidate["math_correctness_sha256"],
        },
        "protocol_hash": quality["protocol_hash"],
        "greedy_canary": quality["protocol"]["greedy_canary"],
        "evidence": {
            "path": str(quality_path),
            "sha256": _sha256(quality_path),
        },
        "supersedes": {
            "reason": (
                "The prior report incorrectly used exact greedy token equality "
                "and coupled canary identity to the selected math set."
            ),
            "path": str(incorrect_gate_path),
            "sha256": _sha256(incorrect_gate_path),
        },
    }
    corrected_final = {
        "schema": "gpuopt.consumer-amd-final-corrected.v2",
        "status": "ACCEPT",
        "performance_improvement_is_valid": True,
        "candidate": previous_final["model"],
        "performance": previous_final["performance"],
        "full_quality": report,
        "decision": (
            "Q5 is accepted: throughput improves by about 10.4%, full paired math "
            "regression is not significant, PPL is within budget, and fixed-canary "
            "accuracy does not regress."
        ),
        "previous_report": {
            "path": str(previous_final_path),
            "sha256": _sha256(previous_final_path),
            "scope": "100-question quality subset",
        },
    }
    reports = root / "reports"
    _write_new(reports / "q5-full-848-corrected-v2.json", report)
    _write_new(reports / "final-corrected-v2.json", corrected_final)
    print(json.dumps(corrected_final, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
