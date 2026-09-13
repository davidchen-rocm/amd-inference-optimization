from datetime import UTC, datetime
from pathlib import Path

import pytest

from amd_inference_opt.gates import GateEngine
from amd_inference_opt.models import (
    AccuracyRequirement,
    BaselineResult,
    BenchmarkProtocol,
    BenchmarkResult,
    DecisionOutcome,
    EnvironmentFingerprint,
    EnvironmentRequirements,
    ExperimentResult,
    MCPConfig,
    MetricSeries,
    ModelTarget,
    OptimizationObjective,
    OptimizationTask,
    PerformanceMetricRequirement,
    QualityConstraints,
    QualityResult,
    RunIdentity,
    RunStatus,
    RuntimeTarget,
)

ENV = {
    "gpu_gfx": "gfx1201",
    "gpu_device": "0",
    "rocm_version": "7.0",
    "runtime_base_commit": "abc",
    "model_sha256": "a" * 64,
    "build_flags_hash": "flags",
}


def task(tmp_path: Path) -> OptimizationTask:
    return OptimizationTask(
        id="gate-task",
        model=ModelTarget(path=tmp_path / "model.gguf", sha256="a" * 64),
        runtime=RuntimeTarget(repo_path=tmp_path / "runtime", base_commit="abc"),
        mcp=MCPConfig(command=["agent"]),
        benchmark=BenchmarkProtocol(sample_count=3),
        environment=EnvironmentRequirements(max_sample_cv_percent=2),
        objective=OptimizationObjective(minimum_improvement_percent=10),
        quality=QualityConstraints(
            max_ppl_regression_percent=1.5,
            max_accuracy_drop_percentage_points=2,
        ),
    )


def quality(ppl: float, accuracy: float) -> QualityResult:
    return QualityResult(
        status=RunStatus.SUCCEEDED,
        correctness_passed=True,
        perplexity=ppl,
        accuracies={"math_accuracy": accuracy},
        coordinate_hash="coordinate",
    )


def baseline() -> BaselineResult:
    return BaselineResult(
        environment=EnvironmentFingerprint(values=ENV, telemetry_stable=True),
        build_status=RunStatus.SUCCEEDED,
        smoke_passed=True,
        benchmark=BenchmarkResult(
            status=RunStatus.SUCCEEDED,
            metrics={"tokens_per_second": MetricSeries(unit="tokens/s", samples=[99, 100, 101])},
        ),
        quality=quality(3.43498, 512 / 848),
    )


def candidate(
    *, samples: list[float] | None = None, result_quality: QualityResult | None = None
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="exp",
        environment=EnvironmentFingerprint(values=ENV, telemetry_stable=True),
        build_status=RunStatus.SUCCEEDED,
        smoke_passed=True,
        e2e=BenchmarkResult(
            status=RunStatus.SUCCEEDED,
            metrics={
                "tokens_per_second": MetricSeries(
                    unit="tokens/s", samples=samples or [114.5, 115, 115.5]
                )
            },
        ),
        quality=result_quality or quality(3.48450, 499 / 848),
    )


def test_gate_accepts_recorded_q4_budget(tmp_path: Path) -> None:
    decision = GateEngine().evaluate(task(tmp_path), baseline(), candidate())

    assert decision.outcome == DecisionOutcome.ACCEPT
    assert decision.improvement_percent == pytest.approx(15)


def test_gate_rejects_performance_or_quality_regression(tmp_path: Path) -> None:
    slow = GateEngine().evaluate(task(tmp_path), baseline(), candidate(samples=[90, 90, 90]))
    bad_quality = GateEngine().evaluate(
        task(tmp_path), baseline(), candidate(result_quality=quality(3.7, 0.5))
    )

    assert slow.outcome == DecisionOutcome.REJECT
    assert bad_quality.outcome == DecisionOutcome.REJECT


def test_gate_requires_each_configured_quality_suite(tmp_path: Path) -> None:
    configured = task(tmp_path)
    configured.quality = QualityConstraints(
        max_ppl_regression_percent=0.5,
        accuracy_requirements=[
            AccuracyRequirement(
                metric="math_accuracy", max_drop_percentage_points=2
            ),
            AccuracyRequirement(
                metric="general_accuracy", max_drop_percentage_points=2
            ),
        ],
    )
    base = baseline()
    assert base.quality is not None
    base.quality.accuracies["general_accuracy"] = 0.70
    result_quality = quality(3.44, 0.60)
    result_quality.accuracies["general_accuracy"] = 0.67

    decision = GateEngine().evaluate(
        configured,
        base,
        candidate(result_quality=result_quality),
    )

    assert decision.outcome == DecisionOutcome.REJECT
    assert "general_accuracy" in decision.reasons[0]
    assert any(
        check.name == "accuracy.general_accuracy" and check.passed is False
        for check in decision.checks
    )


def test_gate_marks_environment_drift_and_noise_inconclusive(tmp_path: Path) -> None:
    drifted = candidate()
    drifted.environment.values["rocm_version"] = "different"  # type: ignore[union-attr]
    drift = GateEngine().evaluate(task(tmp_path), baseline(), drifted)
    noisy = GateEngine().evaluate(
        task(tmp_path), baseline(), candidate(samples=[100, 120, 130])
    )

    assert drift.outcome == DecisionOutcome.INCONCLUSIVE
    assert drift.rerun_from_stage == "CAPTURE_BASELINE"
    assert noisy.outcome == DecisionOutcome.INCONCLUSIVE


def test_candidate_build_and_correctness_failures_are_rejected(tmp_path: Path) -> None:
    build_failed = candidate()
    build_failed.build_status = RunStatus.FAILED
    incorrect = candidate(result_quality=quality(3.44, 512 / 848))
    incorrect.quality.correctness_passed = False  # type: ignore[union-attr]

    assert GateEngine().evaluate(task(tmp_path), baseline(), build_failed).outcome == "REJECT"
    assert GateEngine().evaluate(task(tmp_path), baseline(), incorrect).outcome == "REJECT"


def test_reused_quality_requires_same_recorded_representation(tmp_path: Path) -> None:
    reused = quality(3.48450, 499 / 848)
    reused.reused_from = "previous-experiment"
    reused.representation_hash = "candidate-representation"
    reused.reused_from_representation_hash = "different-representation"

    decision = GateEngine().evaluate(
        task(tmp_path), baseline(), candidate(result_quality=reused)
    )

    assert decision.outcome == DecisionOutcome.INCONCLUSIVE
    assert "representation" in decision.reasons[0]


def test_slow_candidate_is_rejected_before_quality_is_available(tmp_path: Path) -> None:
    slow = candidate(samples=[60, 61, 62])
    slow.quality = None

    decision = GateEngine().evaluate(task(tmp_path), baseline(), slow)

    assert decision.outcome == DecisionOutcome.REJECT
    assert "performance" in decision.reasons[0]


def test_each_required_performance_scenario_must_pass(tmp_path: Path) -> None:
    configured = task(tmp_path)
    configured.benchmark.required_metrics = ["tg128_tokens_per_second", "tg512_tokens_per_second"]
    configured.objective = OptimizationObjective(
        minimum_improvement_percent=10,
        metric_requirements=[
            PerformanceMetricRequirement(
                metric="tg128_tokens_per_second", minimum_improvement_percent=10
            ),
            PerformanceMetricRequirement(
                metric="tg512_tokens_per_second", minimum_improvement_percent=10
            ),
        ],
    )
    base = baseline()
    base.benchmark.metrics = {
        "tg128_tokens_per_second": MetricSeries(unit="tokens/s", samples=[99, 100, 101]),
        "tg512_tokens_per_second": MetricSeries(unit="tokens/s", samples=[99, 100, 101]),
    }
    result = candidate()
    assert result.e2e is not None
    result.e2e.metrics = {
        "tg128_tokens_per_second": MetricSeries(
            unit="tokens/s", samples=[114, 115, 116]
        ),
        "tg512_tokens_per_second": MetricSeries(
            unit="tokens/s", samples=[107, 108, 109]
        ),
    }

    decision = GateEngine().evaluate_performance_pre_gate(configured, base, result)

    assert decision.outcome == DecisionOutcome.REJECT
    assert decision.metric_improvements_percent["tg128_tokens_per_second"] == pytest.approx(15)
    assert decision.metric_improvements_percent["tg512_tokens_per_second"] == pytest.approx(8)


def _run_identity(*, binary: str = "b") -> RunIdentity:
    return RunIdentity(
        protocol_hash="1" * 64,
        binary_sha256=binary * 64,
        source_snapshot_sha256="2" * 64,
        model_sha256="a" * 64,
        runtime_libraries_hash="3" * 64,
        environment_hash="4" * 64,
    )


def test_typed_run_identity_and_fresh_environment_are_comparability_gates(
    tmp_path: Path,
) -> None:
    configured = task(tmp_path)
    configured.environment.require_run_identity = True
    configured.environment.require_fresh_capture = True
    base = baseline()
    result = candidate()
    base.run_identity = _run_identity()
    result.run_identity = _run_identity(binary="c")
    captured_at = datetime.now(UTC)
    base.environment.capture_id = "baseline-capture"
    base.environment.captured_at = captured_at
    result.environment.capture_id = "candidate-capture"  # type: ignore[union-attr]
    result.environment.captured_at = captured_at  # type: ignore[union-attr]

    mismatch = GateEngine().evaluate_performance_pre_gate(configured, base, result)
    assert mismatch.outcome == DecisionOutcome.INCONCLUSIVE
    assert "binary_sha256" in mismatch.reasons[0]

    result.run_identity = _run_identity()
    result.environment.capture_id = "baseline-capture"  # type: ignore[union-attr]
    reused = GateEngine().evaluate_performance_pre_gate(configured, base, result)
    assert reused.outcome == DecisionOutcome.INCONCLUSIVE
    assert "reused" in reused.reasons[0]


def test_verified_representation_change_allows_only_model_identity_difference(
    tmp_path: Path,
) -> None:
    configured = task(tmp_path)
    configured.environment.require_run_identity = True
    base = baseline()
    result = candidate()
    base.run_identity = _run_identity()
    candidate_model = "c" * 64
    result.run_identity = _run_identity().model_copy(
        update={"model_sha256": candidate_model}
    )
    result.environment.values = {
        **ENV,
        "model_sha256": candidate_model,
        "representation_change_declared": "true",
        "source_model_sha256": ENV["model_sha256"],
        "candidate_quantization": "Q6_K",
    }

    accepted = GateEngine().evaluate_performance_pre_gate(configured, base, result)
    assert accepted.outcome == DecisionOutcome.ACCEPT
    assert any(
        check.name == "representation_identity" and check.passed
        for check in accepted.checks
    )

    result.environment.values["source_model_sha256"] = "d" * 64
    rejected = GateEngine().evaluate_performance_pre_gate(configured, base, result)
    assert rejected.outcome == DecisionOutcome.INCONCLUSIVE
