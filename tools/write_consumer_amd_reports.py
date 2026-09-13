#!/usr/bin/env python3
"""Validate the consumer-AMD campaign evidence and write compact reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from amd_inference_opt.kv_cache import QWEN3_8B_KV_BYTES, KVCacheType


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _benchmark(path: Path) -> dict[int, dict[str, float]]:
    rows = _read(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"benchmark is not an array: {path}")
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        generation = int(row["n_gen"])
        mean = float(row["avg_ts"])
        stddev = float(row["stddev_ts"])
        result[generation] = {
            "tokens_per_second": mean,
            "stddev_tokens_per_second": stddev,
            "cv_percent": stddev / mean * 100,
            "sample_count": len(row["samples_ts"]),
        }
    if set(result) != {128, 512}:
        raise RuntimeError(f"benchmark does not contain tg128 and tg512: {path}")
    return result


def _improvements(
    baseline: dict[int, dict[str, float]], candidate: dict[int, dict[str, float]]
) -> dict[str, float]:
    return {
        f"tg{generation}": (
            candidate[generation]["tokens_per_second"]
            / baseline[generation]["tokens_per_second"]
            - 1
        )
        * 100
        for generation in (128, 512)
    }


def _quality(path: Path, *, max_math_drop: int = 2) -> dict[str, object]:
    document = _read(path)
    baseline = document["baseline"]
    candidate = document["candidate"]
    ppl_regression = (
        float(candidate["perplexity"]) / float(baseline["perplexity"]) - 1
    ) * 100
    math_delta = int(candidate["math_correct"]) - int(baseline["math_correct"])
    checks = {
        "status_complete": document.get("status") == "complete",
        "greedy_tokens_equal": document.get("greedy_tokens_equal") is True,
        "perplexity_regression_at_most_0_5_percent": ppl_regression <= 0.5,
        "math_drop_at_most_2_of_100": math_delta >= -max_math_drop,
        "math_total_is_100": int(candidate["math_total"]) == 100,
    }
    return {
        "status": "ACCEPT" if all(checks.values()) else "REJECT",
        "checks": checks,
        "baseline": {
            "perplexity": baseline["perplexity"],
            "math_correct": baseline["math_correct"],
            "math_total": baseline["math_total"],
        },
        "candidate": {
            "perplexity": candidate["perplexity"],
            "math_correct": candidate["math_correct"],
            "math_total": candidate["math_total"],
        },
        "perplexity_regression_percent": ppl_regression,
        "math_correct_delta": math_delta,
        "protocol_hash": document["protocol_hash"],
        "evidence": {"path": str(path), "sha256": _sha256(path)},
    }


def _shape_gate(
    name: str,
    path: Path,
    baseline: dict[int, dict[str, float]],
) -> dict[str, object]:
    candidate = _benchmark(path)
    improvements = _improvements(baseline, candidate)
    checks = {
        "cv_at_most_2_percent": all(
            candidate[generation]["cv_percent"] <= 2 for generation in (128, 512)
        ),
        "tg128_improves_at_least_1_percent": improvements["tg128"] >= 1,
        "tg512_improves_at_least_1_percent": improvements["tg512"] >= 1,
    }
    return {
        "experiment": name,
        "status": "ACCEPT" if all(checks.values()) else "REJECT",
        "checks": checks,
        "metrics": candidate,
        "improvements_percent": improvements,
        "evidence": {"path": str(path), "sha256": _sha256(path)},
        "profile_run": False,
        "quality_run": False,
    }


def write_reports(root: Path) -> dict[str, object]:
    q6_path = root / "mixed-bit/q6/bench-warmed-attempt3.json"
    q5_path = root / "mixed-bit/q5/bench-warmed-attempt3.json"
    q4_path = root / "mixed-bit/q4/bench-warmed-attempt3.json"
    q6 = _benchmark(q6_path)
    q5 = _benchmark(q5_path)
    q4 = _benchmark(q4_path)
    q5_quality = _quality(root / "mixed-bit/q5/quality/result-attempt2.json")
    q4_quality = _quality(root / "mixed-bit/q4/quality/result.json")
    q5_improvements = _improvements(q6, q5)
    q4_improvements = _improvements(q6, q4)
    def stable(values: dict[int, dict[str, float]]) -> bool:
        return all(value["cv_percent"] <= 2 for value in values.values())

    q5_accepted = (
        stable(q5)
        and min(q5_improvements.values()) >= 10
        and q5_quality["status"] == "ACCEPT"
    )
    q4_accepted = (
        stable(q4)
        and min(q4_improvements.values()) >= 10
        and q4_quality["status"] == "ACCEPT"
    )
    mixed = {
        "schema": "gpuopt.consumer-amd-mixed-bit-report.v1",
        "baseline": {"arm": "Q6_K", "metrics": q6},
        "candidates": {
            "Q5_K_M_imatrix": {
                "status": "ACCEPT"
                if q5_accepted
                else "REJECT",
                "metrics": q5,
                "improvements_percent": q5_improvements,
                "quality": q5_quality,
                "model_sha256": "fd96f1387d8d3465385338a636a27b6427381304059de23e234f72c466ab183a",
            },
            "Q4_K_M_imatrix": {
                "status": "ACCEPT"
                if q4_accepted
                else "REJECT",
                "metrics": q4,
                "improvements_percent": q4_improvements,
                "quality": q4_quality,
                "model_sha256": "5a4e54d654371cf89413d6cb26136ced63e2ec5a874cbabc4169522e871d357b",
            },
        },
        "winner": "Q5_K_M_imatrix",
        "evidence": {
            "baseline": {"path": str(q6_path), "sha256": _sha256(q6_path)},
            "q5_benchmark": {"path": str(q5_path), "sha256": _sha256(q5_path)},
            "q4_benchmark": {"path": str(q4_path), "sha256": _sha256(q4_path)},
            "q5_profile": {
                "path": str(root / "mixed-bit/q5/profile/evidence.json"),
                "sha256": _sha256(root / "mixed-bit/q5/profile/evidence.json"),
            },
        },
    }
    if mixed["candidates"]["Q5_K_M_imatrix"]["status"] != "ACCEPT":
        raise RuntimeError("expected Q5 mixed-bit winner did not pass deterministic gates")
    if mixed["candidates"]["Q4_K_M_imatrix"]["status"] != "REJECT":
        raise RuntimeError("expected Q4 quality rejection is missing")

    shape = {
        "schema": "gpuopt.consumer-amd-shape-kernel-report.v1",
        "baseline": {"arm": "stock_Q6_K", "metrics": q6},
        "experiments": [
            _shape_gate("N12288_K4096_small_k_rpb8", root / "shape-kernel/a/bench.json", q6),
            _shape_gate("N4096_K4096_small_k_rpb8", root / "shape-kernel/b/bench.json", q6),
        ],
        "winner": None,
    }
    if any(item["status"] != "REJECT" for item in shape["experiments"]):
        raise RuntimeError("a shape kernel unexpectedly passed")

    kv_path = root / "kv-cache/summary.json"
    kv = _read(kv_path)
    if any(gate["passed"] for gate in kv["gates"].values()):
        raise RuntimeError("a KV-cache arm unexpectedly passed")
    kv_report = {
        "schema": "gpuopt.consumer-amd-kv-cache-report.v1",
        "status": "NO_ACCEPTED_CANDIDATE",
        "gates": kv["gates"],
        "cache_bytes": {
            arm: {
                str(depth): QWEN3_8B_KV_BYTES[KVCacheType(arm)][depth]
                for depth in (4096, 16384, 28672)
            }
            for arm in kv["arms"]
        },
        "evidence": {"path": str(kv_path), "sha256": _sha256(kv_path)},
    }

    final = {
        "schema": "gpuopt.consumer-amd-final-report.v1",
        "status": "ACCEPT",
        "gpu": "AMD Radeon RX 9070 XT",
        "architecture": "gfx1201",
        "model": {
            "path": (
                "/workspace/math-rule-loop/models/qwen3-8b-mixed/"
                "Qwen3-8B-Q5_K_M-imatrix.gguf"
            ),
            "sha256": "fd96f1387d8d3465385338a636a27b6427381304059de23e234f72c466ab183a",
            "quantization": "Q5_K_M with independent imatrix calibration",
        },
        "runtime": {
            "commit": "a7a6d0d269c896218b6c78e0933bd6a17519d3f6",
            "source_patch": None,
            "kv_cache": "f16",
            "flash_attention": "runtime default for short decode",
        },
        "performance": {
            "baseline": q6,
            "candidate": q5,
            "improvements_percent": q5_improvements,
        },
        "quality": q5_quality,
        "accepted_components": ["mixed-bit Q5_K_M imatrix model"],
        "rejected_components": [
            "mixed-bit Q4_K_M: 100-question math score dropped by 3",
            "Q6 shape-specific N12288/K4096 small-k kernel: slower",
            "Q6 shape-specific N4096/K4096 small-k kernel: slower and unstable",
            "Q8_0 KV cache: slower at all measured depths",
            "Q4_0 KV cache: slower at all measured depths",
        ],
        "framework_findings": [
            "full-workload preconditioning is required; llama-bench built-in "
            "warmup was insufficient",
            "CV-driven sampling must support a five-to-12-sample retry",
            "GPU resource locking must span campaigns, not only tasks",
            "quality runtime arguments must be arm-specific and included in the protocol hash",
            "performance pre-gates correctly avoid profiler and quality cost "
            "for rejected candidates",
        ],
        "report_inputs": {
            "mixed_bit": "reports/mixed-bit.json",
            "shape_kernel": "reports/shape-kernel.json",
            "kv_cache": "reports/kv-cache.json",
        },
    }
    reports = root / "reports"
    _write(reports / "mixed-bit.json", mixed)
    _write(reports / "shape-kernel.json", shape)
    _write(reports / "kv-cache.json", kv_report)
    _write(reports / "final.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    final = write_reports(args.root.resolve(strict=True))
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
