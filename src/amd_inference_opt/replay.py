"""Deterministic replay of the recorded Q4_RDNA optimization case.

Replay deliberately consumes a compact, provenance-labelled import.  It does not
call MCP, rerun a workload, or represent historical evidence as a live result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ReplayError(ValueError):
    """Raised when a replay config or fixture violates the recorded contract."""


@dataclass(frozen=True)
class ReplayResult:
    task_id: str
    output_dir: Path
    final_decision: str
    experiment_decisions: dict[str, str]
    report: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read JSON document {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _artifact(path: Path, producer: str) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
        "producer": producer,
    }


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReplayError(f"{label} must be a finite number")
    return float(value)


def _series(summary: dict[str, Any], label: str, minimum_samples: int) -> dict[str, Any]:
    raw = summary.get("samples_tokens_per_second")
    if not isinstance(raw, list):
        raise ReplayError(f"{label}.samples_tokens_per_second must be a list")
    samples = [_require_number(item, f"{label}.samples_tokens_per_second") for item in raw]
    if any(item <= 0 for item in samples):
        raise ReplayError(f"{label} samples must be positive")
    mean = statistics.fmean(samples) if samples else 0.0
    stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "samples": samples,
        "sample_count": len(samples),
        "mean": mean,
        "sample_stddev": stddev,
        "coefficient_of_variation": stddev / mean if mean else None,
        "enough_samples": len(samples) >= minimum_samples,
    }


def _quality(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_ppl = _require_number(baseline.get("perplexity"), "baseline perplexity")
    candidate_ppl = _require_number(candidate.get("perplexity"), "candidate perplexity")
    baseline_correct = int(baseline.get("math_correct", -1))
    candidate_correct = int(candidate.get("math_correct", -1))
    baseline_total = int(baseline.get("math_total", 0))
    candidate_total = int(candidate.get("math_total", 0))
    if baseline_total <= 0 or candidate_total != baseline_total:
        raise ReplayError("quality math totals must be positive and identical")
    baseline_accuracy = baseline_correct / baseline_total
    candidate_accuracy = candidate_correct / candidate_total
    return {
        "baseline_perplexity": baseline_ppl,
        "candidate_perplexity": candidate_ppl,
        "perplexity_regression_fraction": (candidate_ppl / baseline_ppl) - 1.0,
        "baseline_math": {
            "correct": baseline_correct,
            "total": baseline_total,
            "accuracy": baseline_accuracy,
        },
        "candidate_math": {
            "correct": candidate_correct,
            "total": candidate_total,
            "accuracy": candidate_accuracy,
        },
        "math_accuracy_drop_points": baseline_accuracy - candidate_accuracy,
    }


def _gate_experiment(
    experiment: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    stability = config["stability"]
    minimum_samples = int(stability["minimum_samples"])
    maximum_cv = _require_number(stability["maximum_coefficient_of_variation"], "maximum CV")
    required_improvement = _require_number(
        config["objective"]["minimum_improvement_fraction"], "minimum improvement"
    )
    quality_budget = config["quality_budget"]

    baseline_series: dict[str, Any] = {}
    candidate_series: dict[str, Any] = {}
    improvements: dict[str, float] = {}
    comparability_errors: list[str] = []

    lengths = sorted(baseline["benchmarks"], key=int)
    if sorted(experiment["benchmarks"], key=int) != lengths:
        comparability_errors.append("candidate generation lengths differ from baseline")
    for length in lengths:
        base = _series(baseline["benchmarks"][length], f"baseline[{length}]", minimum_samples)
        baseline_series[length] = base
        if length not in experiment["benchmarks"]:
            continue
        candidate = _series(
            experiment["benchmarks"][length], f"{experiment['id']}[{length}]", minimum_samples
        )
        candidate_series[length] = candidate
        improvements[length] = (candidate["mean"] / base["mean"]) - 1.0
        for role, series in (("baseline", base), ("candidate", candidate)):
            if not series["enough_samples"]:
                comparability_errors.append(f"{role} tg{length} has too few samples")
            if (
                series["coefficient_of_variation"] is None
                or series["coefficient_of_variation"] > maximum_cv
            ):
                comparability_errors.append(f"{role} tg{length} exceeds CV limit")

    quality = _quality(experiment["quality"], baseline["quality"])
    quality_checks = {
        "perplexity_within_budget": quality["perplexity_regression_fraction"]
        <= _require_number(
            quality_budget["maximum_perplexity_regression_fraction"],
            "maximum perplexity regression",
        ),
        "math_within_budget": quality["math_accuracy_drop_points"]
        <= _require_number(
            quality_budget["maximum_math_accuracy_drop_points"], "maximum math accuracy drop"
        ),
    }
    performance_checks = {
        f"tg{length}_meets_minimum_improvement": improvement >= required_improvement
        for length, improvement in improvements.items()
    }

    if comparability_errors:
        decision = "INCONCLUSIVE"
        reason = "benchmark stability or comparability gate failed"
    elif not all(performance_checks.values()):
        decision = "REJECT"
        reason = "repeatable decode performance did not meet the objective"
    elif not all(quality_checks.values()):
        decision = "REJECT"
        reason = "quality budget was exceeded"
    else:
        decision = "ACCEPT"
        reason = "performance, stability, and quality gates passed"

    return {
        "schema_version": 1,
        "experiment_id": experiment["id"],
        "decision": decision,
        "reason": reason,
        "environment_comparable": not comparability_errors,
        "comparability_errors": comparability_errors,
        "baseline_series": baseline_series,
        "candidate_series": candidate_series,
        "improvement_fraction": improvements,
        "performance_checks": performance_checks,
        "quality": quality,
        "quality_checks": quality_checks,
        "thresholds": {
            "minimum_improvement_fraction": required_improvement,
            "minimum_samples": minimum_samples,
            "maximum_coefficient_of_variation": maximum_cv,
            **quality_budget,
        },
    }


def _execution_map(fixture: dict[str, Any]) -> dict[str, Any]:
    evidence = fixture["kernel_evidence"]
    entries = []
    for dispatch in evidence["dispatches"]:
        entries.append(
            {
                "phase": "decode",
                "layer": "transformer layer",
                "operator": "FFN gate/up/down projection",
                "tensor": {
                    "shape": dispatch["shape"],
                    "dtype": "Q4_K_M or Q4_RDNA",
                    "quantization": "recorded",
                },
                "runtime": {
                    "implementation": "llama.cpp ROCm quantized matvec dispatch",
                    "backend": "ROCm",
                    "source_location": "unavailable in recorded import",
                },
                "kernel": {
                    "name": "quantized GEMV (normalized recorded dispatch)",
                    "shape": dispatch["shape"],
                    "median_duration_us": {
                        "q4_k_m": dispatch["q4_k_us"],
                        "q4_rdna_old": dispatch["old_us"],
                        "q4_rdna_split_k": dispatch["split_k_us"],
                    },
                    "launched_waves": {
                        "q4_rdna_old": dispatch["old_waves"],
                        "q4_rdna_split_k": dispatch["split_k_waves"],
                    },
                    "vgpr": {
                        "q4_rdna_old": dispatch["old_vgpr"],
                        "q4_rdna_split_k": dispatch["split_k_vgpr"],
                    },
                },
                "hardware_behavior": (
                    "memory-latency-bound after split-K; old mapping has insufficient "
                    "device-wide wave supply"
                ),
                "evidence_refs": [evidence["case_reference"]],
            }
        )
    return {
        "schema_version": 1,
        "coverage": "dominant decode GEMV path only",
        "entries": entries,
        "unsupported_counters": evidence["unsupported_counters"],
        "caveat": evidence["caveat"],
    }


def _events(
    experiments: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    stages = [
        "CREATE_TASK",
        "INSPECT_TARGET",
        "CAPTURE_BASELINE",
        "DECOMPOSE_E2E",
        "BUILD_EXECUTION_MAP",
        "DISCOVER_HOTSPOTS",
        "CLASSIFY_BOTTLENECK",
        "ANALYZE_LIMIT",
    ]
    events = [
        {"sequence": index + 1, "stage": stage, "source": "recorded_import"}
        for index, stage in enumerate(stages)
    ]
    sequence = len(events)
    for experiment in experiments:
        for stage in (
            "GENERATE_HYPOTHESIS",
            "CREATE_EXPERIMENT",
            "PATCH_AND_BUILD",
            "MICROBENCH",
            "E2E_VALIDATION",
            "QUALITY_VALIDATION",
            "DECIDE",
        ):
            sequence += 1
            event = {
                "sequence": sequence,
                "stage": stage,
                "experiment_id": experiment["id"],
                "source": "recorded_import",
            }
            if stage == "DECIDE":
                event["decision"] = decisions[experiment["id"]]["decision"]
            events.append(event)
    return events


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Q4_RDNA recorded replay",
        "",
        "> This report replays a normalized historical import. It is not a live MCP observation.",
        "",
        (
            f"Target: {report['target']['model']} on {report['target']['gpu']} "
            f"({report['target']['gfx_target']})"
        ),
        "",
        "## Experiments",
        "",
        (
            "| Experiment | tg128 vs baseline | tg512 vs baseline | PPL regression "
            "| Math drop | Decision |"
        ),
        "|---|---:|---:|---:|---:|---|",
    ]
    for experiment in report["experiments"]:
        decision = experiment["gate"]
        improvement = decision["improvement_fraction"]
        quality = decision["quality"]
        lines.append(
            f"| {experiment['id']} | {improvement['128']:+.2%} | {improvement['512']:+.2%} "
            f"| {quality['perplexity_regression_fraction']:+.2%} "
            f"| {quality['math_accuracy_drop_points']:+.2%} | {decision['decision']} |"
        )
    lines.extend(
        [
            "",
            (
                "The old row mapping is rejected because its stable decode throughput "
                "regresses. The split-K wave32 mapping is accepted under this task's "
                "explicit 10% performance, 1.5% PPL, and 2 percentage-point math budgets."
            ),
            "",
            (
                "The acceptance is scoped to this recorded task and does not claim "
                "lossless quality or a general drop-in replacement."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_recorded_replay(
    config_path: str | Path, output_dir: str | Path, *, allow_existing_empty: bool = True
) -> ReplayResult:
    """Replay one compact recorded case and materialize an auditable file store."""

    config_file = Path(config_path).resolve()
    config = _read_json(config_file)
    if int(config.get("schema_version", 0)) != 1:
        raise ReplayError("only replay config schema_version 1 is supported")
    fixture_ref = config.get("fixture")
    if not isinstance(fixture_ref, str) or not fixture_ref:
        raise ReplayError("replay config must name a fixture")
    fixture_file = (config_file.parent / fixture_ref).resolve()
    fixture = _read_json(fixture_file)
    if int(fixture.get("schema_version", 0)) != 1:
        raise ReplayError("only replay fixture schema_version 1 is supported")
    if fixture.get("recording", {}).get("kind") != "recorded_import":
        raise ReplayError("fixture must be explicitly labelled recorded_import")
    policy = config.get("provenance_policy", {})
    if (
        policy.get("require_recorded_import_label") is not True
        or policy.get("allow_live_mcp_claims") is not False
    ):
        raise ReplayError("recorded replay must forbid live MCP claims")

    task_id = config.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ReplayError("task_id must be a non-empty string")
    destination = Path(output_dir).resolve()
    if destination.exists():
        if not allow_existing_empty or any(destination.iterdir()):
            raise ReplayError(f"refusing to overwrite non-empty replay directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    baseline = fixture["baseline"]
    experiments = fixture.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ReplayError("fixture must contain at least one experiment")
    decisions = {
        experiment["id"]: _gate_experiment(experiment, baseline, config)
        for experiment in experiments
    }
    for experiment in experiments:
        expected = experiment.get("expected_decision")
        actual = decisions[experiment["id"]]["decision"]
        if expected is not None and expected != actual:
            raise ReplayError(
                f"recorded decision mismatch for {experiment['id']}: expected {expected}, "
                f"got {actual}"
            )

    execution_map = _execution_map(fixture)
    hypotheses = [
        {"experiment_id": experiment["id"], **experiment["hypothesis"]}
        for experiment in experiments
    ]
    report_experiments = [
        {
            "id": experiment["id"],
            "change": experiment["change"],
            "hypothesis": experiment["hypothesis"],
            "gate": decisions[experiment["id"]],
        }
        for experiment in experiments
    ]
    final_decision = decisions[experiments[-1]["id"]]["decision"]
    report = {
        "schema_version": 1,
        "task_id": task_id,
        "mode": "recorded_replay",
        "evidence_kind": "recorded_import",
        "target": fixture["target"],
        "objective": config["objective"],
        "quality_budget": config["quality_budget"],
        "stability": config["stability"],
        "analysis": fixture["analysis"],
        "experiments": report_experiments,
        "final_decision": final_decision,
        "limitations": [
            "No workload or profiler was executed during replay.",
            (
                "Recorded artifacts are normalized summaries with source hashes, "
                "not copied raw traces."
            ),
            "ACCEPT means only that this task's explicit performance and quality budgets passed.",
        ],
    }

    _atomic_json(
        destination / "task.json",
        {**config, "fixture": str(fixture_file), "mode": "recorded_replay"},
    )
    _atomic_json(destination / "inspection" / "target.json", fixture["target"])
    _atomic_json(destination / "baseline" / "result.json", baseline)
    _atomic_json(
        destination / "baseline" / "decomposition.json",
        {
            "prefill": {"status": "not_requested"},
            "decode": {"status": "measured", "benchmarks": baseline["benchmarks"]},
            "runtime_http_scheduler_overhead": {"status": "unavailable"},
            "source": "recorded_import",
        },
    )
    _atomic_json(destination / "execution-map.json", execution_map)
    _atomic_json(destination / "mcp" / "recorded-evidence.json", fixture["kernel_evidence"])
    _atomic_json(
        destination / "analysis" / "bottleneck.json",
        fixture["analysis"]["bottleneck_assessment"],
    )
    _atomic_json(
        destination / "analysis" / "limit-estimate.json",
        fixture["analysis"]["limit_estimate"],
    )
    _atomic_json(destination / "hypotheses.json", hypotheses)
    for experiment in experiments:
        experiment_dir = destination / "experiments" / experiment["id"]
        _atomic_json(
            experiment_dir / "spec.json",
            {
                "id": experiment["id"],
                "change": experiment["change"],
                "hypothesis": experiment["hypothesis"],
            },
        )
        _atomic_json(experiment_dir / "e2e.json", experiment["benchmarks"])
        _atomic_json(experiment_dir / "quality.json", experiment["quality"])
        _atomic_json(experiment_dir / "decision.json", decisions[experiment["id"]])
    events = _events(experiments, decisions)
    _write_text(
        destination / "events.jsonl",
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
    )
    _atomic_json(destination / "final-report" / "report.json", report)
    _write_text(destination / "final-report" / "report.md", _markdown(report))
    _atomic_json(
        destination / "final-report" / "final-config.json",
        {
            "schema_version": 1,
            "change_kind": "runtime_config",
            "mapping": "split_k_wave32_default",
            "environment": experiments[-1]["change"]["environment"],
            "unset_environment": experiments[-1]["change"].get(
                "unset_environment", []
            ),
            "sidecar_sha256": fixture["representation"]["sidecar_sha256"],
            "sidecar_unchanged_between_old_and_split": fixture["representation"][
                "unchanged_between_experiments"
            ],
            "binary_identity": "identical_between_old_and_split_in_recorded_ablation",
            "binary_sha256": None,
            "source": "recorded_import",
            "note": (
                "No final.patch is emitted because the accepted experiment is a runtime "
                "configuration ablation."
            ),
        },
    )

    artifacts = []
    for relative in (
        Path("baseline/decomposition.json"),
        Path("execution-map.json"),
        Path("analysis/bottleneck.json"),
        Path("analysis/limit-estimate.json"),
        Path("events.jsonl"),
        Path("final-report/report.json"),
        Path("final-report/report.md"),
        Path("final-report/final-config.json"),
    ):
        artifact_path = destination / relative
        item = _artifact(artifact_path, "recorded-replay")
        item["path"] = str(relative)
        artifacts.append(item)
    _atomic_json(
        destination / "artifact-manifest.json", {"schema_version": 1, "artifacts": artifacts}
    )
    _atomic_json(
        destination / "state.json",
        {
            "schema_version": 1,
            "task_id": task_id,
            "state": "DECIDE",
            "workflow_status": {
                "ACCEPT": "ACCEPTED",
                "REJECT": "REJECTED",
                "INCONCLUSIVE": "INCONCLUSIVE",
            }[final_decision],
            "terminal_decision": final_decision,
            "updated_at": datetime.now(UTC).isoformat(),
            "source": "recorded_import",
        },
    )

    return ReplayResult(
        task_id=task_id,
        output_dir=destination,
        final_decision=final_decision,
        experiment_decisions={key: value["decision"] for key, value in decisions.items()},
        report=report,
    )


__all__ = ["ReplayError", "ReplayResult", "run_recorded_replay"]
