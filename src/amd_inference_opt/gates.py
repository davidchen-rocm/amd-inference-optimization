"""Deterministic performance, correctness, quality, and comparability gates."""

from __future__ import annotations

from .models import (
    BaselineResult,
    CampaignKind,
    DecisionOutcome,
    ExperimentResult,
    GateCheck,
    GateDecision,
    MetricSeries,
    OptimizationTask,
    RunStatus,
    WorkflowStage,
)


class GateEngine:
    """Produce the only authoritative experiment outcome."""

    def evaluate(
        self,
        task: OptimizationTask,
        baseline: BaselineResult,
        candidate: ExperimentResult,
    ) -> GateDecision:
        performance = self.evaluate_performance_pre_gate(task, baseline, candidate)
        if performance.outcome != DecisionOutcome.ACCEPT:
            return performance

        checks = list(performance.checks)
        improvements = dict(performance.metric_improvements_percent)
        improvement = performance.improvement_percent

        correctness_decision = self._check_correctness(task, candidate, checks)
        if correctness_decision is not None:
            outcome, reason = correctness_decision
            if outcome == DecisionOutcome.REJECT:
                return GateDecision(
                    outcome=outcome,
                    checks=checks,
                    reasons=[reason],
                    improvement_percent=improvement,
                    metric_improvements_percent=improvements,
                )
            decision = self._inconclusive(
                checks, reason, WorkflowStage.QUALITY_VALIDATION
            )
            decision.improvement_percent = improvement
            decision.metric_improvements_percent = improvements
            return decision

        quality_decision = self._check_quality(task, baseline, candidate, checks)
        if quality_decision is not None:
            outcome, reason = quality_decision
            if outcome == DecisionOutcome.INCONCLUSIVE:
                decision = self._inconclusive(
                    checks, reason, WorkflowStage.QUALITY_VALIDATION
                )
                decision.improvement_percent = improvement
                decision.metric_improvements_percent = improvements
                return decision
            return GateDecision(
                outcome=outcome,
                checks=checks,
                reasons=[reason],
                improvement_percent=improvement,
                metric_improvements_percent=improvements,
            )

        return GateDecision(
            outcome=DecisionOutcome.ACCEPT,
            checks=checks,
            reasons=["all required performance, correctness, and quality gates passed"],
            improvement_percent=improvement,
            metric_improvements_percent=improvements,
        )

    def evaluate_performance_pre_gate(
        self,
        task: OptimizationTask,
        baseline: BaselineResult,
        candidate: ExperimentResult,
    ) -> GateDecision:
        """Evaluate comparability, build, stability and performance without quality."""

        checks: list[GateCheck] = []

        environment_error = self._check_environment(task, baseline, candidate, checks)
        if environment_error:
            return self._inconclusive(checks, environment_error, WorkflowStage.CAPTURE_BASELINE)

        build_decision = self._check_build_and_smoke(baseline, candidate, checks)
        if build_decision is not None:
            outcome, reason = build_decision
            if outcome == DecisionOutcome.REJECT:
                return GateDecision(outcome=outcome, checks=checks, reasons=[reason])
            return self._inconclusive(checks, reason, WorkflowStage.PATCH_AND_BUILD)

        benchmark_error = self._check_benchmarks(task, baseline, candidate, checks)
        if benchmark_error:
            return self._inconclusive(checks, benchmark_error, WorkflowStage.E2E_VALIDATION)

        improvements: dict[str, float] = {}
        failures: list[str] = []
        for requirement in task.objective.performance_requirements():
            baseline_metric = baseline.benchmark.metrics[requirement.metric]
            candidate_metric = candidate.e2e.metrics[requirement.metric]  # type: ignore[union-attr]
            ratio = candidate_metric.mean / baseline_metric.mean
            improvement = (ratio - 1) * 100 if requirement.maximize else (1 - ratio) * 100
            improvements[requirement.metric] = improvement
            passed = improvement >= requirement.minimum_improvement_percent
            checks.append(
                GateCheck(
                    name=f"performance:{requirement.metric}",
                    passed=passed,
                    detail=(
                        f"{improvement:.4f}% improvement; required >= "
                        f"{requirement.minimum_improvement_percent:.4f}%"
                    ),
                )
            )
            if not passed:
                failures.append(requirement.metric)

        primary_improvement = improvements.get(task.objective.primary_metric)
        if primary_improvement is None and improvements:
            primary_improvement = next(iter(improvements.values()))
        if failures:
            return GateDecision(
                outcome=DecisionOutcome.REJECT,
                checks=checks,
                reasons=[
                    "required performance objectives were not met: "
                    + ", ".join(failures)
                ],
                improvement_percent=primary_improvement,
                metric_improvements_percent=improvements,
            )
        return GateDecision(
            outcome=DecisionOutcome.ACCEPT,
            checks=checks,
            reasons=["all required performance pre-gates passed"],
            improvement_percent=primary_improvement,
            metric_improvements_percent=improvements,
        )

    @staticmethod
    def _inconclusive(
        checks: list[GateCheck],
        reason: str,
        rerun_from: WorkflowStage,
    ) -> GateDecision:
        return GateDecision(
            outcome=DecisionOutcome.INCONCLUSIVE,
            checks=checks,
            reasons=[reason],
            rerun_from_stage=rerun_from,
        )

    @staticmethod
    def _check_environment(
        task: OptimizationTask,
        baseline: BaselineResult,
        candidate: ExperimentResult,
        checks: list[GateCheck],
    ) -> str | None:
        if candidate.environment is None:
            checks.append(GateCheck(name="environment", passed=None, detail="candidate missing"))
            return "candidate environment fingerprint is missing"

        representation_change = False
        baseline_model = baseline.environment.values.get("model_sha256")
        candidate_model = candidate.environment.values.get("model_sha256")
        if candidate.environment.values.get("representation_change_declared") == "true":
            representation_change = bool(
                baseline_model
                and candidate_model
                and baseline_model != candidate_model
                and candidate.environment.values.get("source_model_sha256")
                == baseline_model
                and candidate.run_identity is not None
                and candidate.run_identity.model_sha256 == candidate_model
            )
            checks.append(
                GateCheck(
                    name="representation_identity",
                    passed=representation_change,
                    detail=(
                        f"verified {baseline_model} -> {candidate_model}"
                        if representation_change
                        else "declared representation change is not identity-bound"
                    ),
                )
            )

        mismatches: list[str] = []
        for field in task.environment.required_match_fields:
            baseline_value = baseline.environment.values.get(field)
            candidate_value = candidate.environment.values.get(field)
            if baseline_value is None or candidate_value is None:
                mismatches.append(f"{field}=missing")
            elif baseline_value != candidate_value and not (
                field == "model_sha256" and representation_change
            ):
                mismatches.append(f"{field}: {baseline_value!r} != {candidate_value!r}")

        if task.environment.require_stable_telemetry:
            if baseline.environment.telemetry_stable is not True:
                mismatches.append("baseline telemetry is absent or unstable")
            if candidate.environment.telemetry_stable is not True:
                mismatches.append("candidate telemetry is absent or unstable")
            if task.campaign_kind is CampaignKind.VLLM_MI300X:
                tolerance = task.environment.max_telemetry_clock_drift_percent
                for field in (
                    "telemetry_gfxclk_mean_mhz",
                    "telemetry_uclk_mean_mhz",
                ):
                    baseline_text = baseline.environment.values.get(field)
                    candidate_text = candidate.environment.values.get(field)
                    try:
                        baseline_clock = float(baseline_text or "")
                        candidate_clock = float(candidate_text or "")
                    except ValueError:
                        mismatches.append(f"{field}=missing-or-invalid")
                        continue
                    denominator = max(abs(baseline_clock), abs(candidate_clock))
                    if denominator <= 0:
                        mismatches.append(f"{field}=non-positive")
                        continue
                    drift = abs(candidate_clock - baseline_clock) / denominator * 100
                    if drift > tolerance:
                        mismatches.append(
                            f"{field} drift {drift:.3f}% exceeds {tolerance:.3f}%"
                        )

        if task.environment.require_fresh_capture:
            baseline_capture = baseline.environment.capture_id
            candidate_capture = candidate.environment.capture_id
            if (
                baseline.environment.source != "observed"
                or baseline_capture is None
                or baseline.environment.captured_at is None
            ):
                mismatches.append("baseline environment was not freshly observed")
            if (
                candidate.environment.source != "observed"
                or candidate_capture is None
                or candidate.environment.captured_at is None
            ):
                mismatches.append("candidate environment was not freshly observed")
            if baseline_capture is not None and baseline_capture == candidate_capture:
                mismatches.append("candidate reused the baseline environment capture")

        identity_required = (
            task.environment.require_run_identity
            or baseline.run_identity is not None
            or candidate.run_identity is not None
        )
        if identity_required:
            if baseline.run_identity is None:
                mismatches.append("baseline run identity is missing")
            if candidate.run_identity is None:
                mismatches.append("candidate run identity is missing")
            if baseline.run_identity is not None and candidate.run_identity is not None:
                baseline_identity = baseline.run_identity.model_dump(mode="json")
                candidate_identity = candidate.run_identity.model_dump(mode="json")
                for field in task.environment.required_run_match_fields:
                    baseline_value = baseline_identity.get(field)
                    candidate_value = candidate_identity.get(field)
                    if baseline_value is None or candidate_value is None:
                        mismatches.append(f"run_identity.{field}=missing")
                    elif baseline_value != candidate_value and not (
                        field == "model_sha256" and representation_change
                    ):
                        mismatches.append(
                            f"run_identity.{field}: {baseline_value!r} != "
                            f"{candidate_value!r}"
                        )
                configured_protocol = task.benchmark.protocol_hash
                if configured_protocol is not None:
                    if baseline.run_identity.protocol_hash != configured_protocol:
                        mismatches.append(
                            "baseline protocol hash differs from configured protocol"
                        )
                    if candidate.run_identity.protocol_hash != configured_protocol:
                        mismatches.append(
                            "candidate protocol hash differs from configured protocol"
                        )

        checks.append(
            GateCheck(
                name="environment",
                passed=not mismatches,
                detail="comparable" if not mismatches else "; ".join(mismatches),
            )
        )
        return "environment is not comparable: " + "; ".join(mismatches) if mismatches else None

    @staticmethod
    def _check_build_and_smoke(
        baseline: BaselineResult,
        candidate: ExperimentResult,
        checks: list[GateCheck],
    ) -> tuple[DecisionOutcome, str] | None:
        baseline_valid = (
            baseline.build_status in {RunStatus.SUCCEEDED, RunStatus.REUSED}
            and baseline.smoke_passed
        )
        if not baseline_valid:
            checks.append(GateCheck(name="baseline_build_smoke", passed=None, detail="invalid"))
            return DecisionOutcome.INCONCLUSIVE, "baseline build or smoke test is invalid"

        checks.append(GateCheck(name="baseline_build_smoke", passed=True, detail="passed"))
        if candidate.build_status in {RunStatus.FAILED, RunStatus.TIMED_OUT}:
            checks.append(
                GateCheck(
                    name="candidate_build",
                    passed=False,
                    detail=f"candidate build {candidate.build_status}",
                )
            )
            return DecisionOutcome.REJECT, "candidate build failed against a valid baseline"
        if candidate.build_status not in {RunStatus.SUCCEEDED, RunStatus.REUSED}:
            checks.append(
                GateCheck(name="candidate_build", passed=None, detail=str(candidate.build_status))
            )
            return DecisionOutcome.INCONCLUSIVE, "candidate build did not produce a usable binary"
        checks.append(
            GateCheck(name="candidate_build", passed=True, detail=str(candidate.build_status))
        )

        if candidate.smoke_passed is None:
            checks.append(GateCheck(name="smoke", passed=None, detail="missing"))
            return DecisionOutcome.INCONCLUSIVE, "candidate smoke test is missing"
        checks.append(
            GateCheck(
                name="smoke",
                passed=candidate.smoke_passed,
                detail="passed" if candidate.smoke_passed else "failed",
            )
        )
        if not candidate.smoke_passed:
            return DecisionOutcome.REJECT, "candidate failed deterministic smoke validation"
        return None

    @staticmethod
    def _check_correctness(
        task: OptimizationTask,
        candidate: ExperimentResult,
        checks: list[GateCheck],
    ) -> tuple[DecisionOutcome, str] | None:
        if not task.quality.require_correctness:
            checks.append(GateCheck(name="correctness", passed=True, detail="not required"))
            return None
        if candidate.quality is None or candidate.quality.status != RunStatus.SUCCEEDED:
            checks.append(GateCheck(name="correctness", passed=None, detail="result missing"))
            return DecisionOutcome.INCONCLUSIVE, "required correctness result is missing"
        if candidate.quality.correctness_passed is None:
            checks.append(GateCheck(name="correctness", passed=None, detail="verdict missing"))
            return DecisionOutcome.INCONCLUSIVE, "correctness verdict is missing"
        checks.append(
            GateCheck(
                name="correctness",
                passed=candidate.quality.correctness_passed,
                detail="passed" if candidate.quality.correctness_passed else "failed",
            )
        )
        if not candidate.quality.correctness_passed:
            return DecisionOutcome.REJECT, "candidate correctness validation failed"
        return None

    @staticmethod
    def _series_problem(
        name: str,
        series: MetricSeries,
        required_samples: int,
        max_cv_percent: float,
    ) -> str | None:
        if len(series.samples) < required_samples:
            return f"{name} has {len(series.samples)} samples; {required_samples} required"
        if series.mean <= 0:
            return f"{name} mean must be positive"
        if series.cv_percent > max_cv_percent:
            return f"{name} CV {series.cv_percent:.4f}% exceeds {max_cv_percent:.4f}%"
        return None

    def _check_benchmarks(
        self,
        task: OptimizationTask,
        baseline: BaselineResult,
        candidate: ExperimentResult,
        checks: list[GateCheck],
    ) -> str | None:
        if baseline.benchmark.status != RunStatus.SUCCEEDED:
            checks.append(
                GateCheck(name="baseline_benchmark", passed=None, detail="not successful")
            )
            return "baseline benchmark is not successful"
        if candidate.e2e is None or candidate.e2e.status != RunStatus.SUCCEEDED:
            checks.append(
                GateCheck(name="candidate_benchmark", passed=None, detail="not successful")
            )
            return "candidate benchmark is missing or not successful"
        metric_names = list(task.benchmark.required_metrics)
        for requirement in task.objective.performance_requirements():
            if requirement.metric not in metric_names:
                metric_names.append(requirement.metric)

        missing_metrics = [
            name
            for name in metric_names
            if name not in baseline.benchmark.metrics or name not in candidate.e2e.metrics
        ]
        if missing_metrics:
            detail = ", ".join(missing_metrics)
            checks.append(
                GateCheck(name="benchmark_metric", passed=None, detail=detail + " missing")
            )
            return f"required metrics are missing: {detail}"

        problems = []
        for metric_name in metric_names:
            for label, series in (
                ("baseline", baseline.benchmark.metrics[metric_name]),
                ("candidate", candidate.e2e.metrics[metric_name]),
            ):
                problem = self._series_problem(
                    f"{label}.{metric_name}",
                    series,
                    task.benchmark.sample_count,
                    task.environment.max_sample_cv_percent,
                )
                if problem:
                    problems.append(problem)
        checks.append(
            GateCheck(
                name="benchmark_stability",
                passed=not problems,
                detail="stable" if not problems else "; ".join(problems),
            )
        )
        return "; ".join(problems) if problems else None

    @staticmethod
    def _check_quality(
        task: OptimizationTask,
        baseline: BaselineResult,
        candidate: ExperimentResult,
        checks: list[GateCheck],
    ) -> tuple[DecisionOutcome, str] | None:
        if not task.quality.require_quality_evaluation and not task.quality.require_correctness:
            checks.append(GateCheck(name="quality", passed=True, detail="not required"))
            return None
        if candidate.quality is None or candidate.quality.status != RunStatus.SUCCEEDED:
            checks.append(GateCheck(name="quality", passed=None, detail="candidate result missing"))
            return DecisionOutcome.INCONCLUSIVE, "required candidate quality result is missing"

        candidate_quality = candidate.quality
        baseline_quality = baseline.quality
        if not task.quality.require_quality_evaluation:
            return None
        if baseline_quality is None or baseline_quality.status != RunStatus.SUCCEEDED:
            checks.append(GateCheck(name="quality_baseline", passed=None, detail="missing"))
            return DecisionOutcome.INCONCLUSIVE, "quality baseline is missing"

        if candidate_quality.reused_from is not None:
            representation_match = (
                candidate_quality.representation_hash is not None
                and candidate_quality.representation_hash
                == candidate_quality.reused_from_representation_hash
            )
            checks.append(
                GateCheck(
                    name="quality_reuse_representation",
                    passed=representation_match,
                    detail="match" if representation_match else "missing or mismatched",
                )
            )
            if not representation_match:
                return (
                    DecisionOutcome.INCONCLUSIVE,
                    "reused quality result has a different or unrecorded representation",
                )

        if (
            baseline_quality.coordinate_hash is not None
            and candidate_quality.coordinate_hash is not None
            and baseline_quality.coordinate_hash != candidate_quality.coordinate_hash
        ):
            checks.append(GateCheck(name="quality_coordinates", passed=False, detail="mismatch"))
            return DecisionOutcome.INCONCLUSIVE, "quality evaluation coordinates do not match"
        if (
            baseline_quality.coordinate_hash is None
            or candidate_quality.coordinate_hash is None
        ):
            checks.append(GateCheck(name="quality_coordinates", passed=None, detail="missing"))
            return DecisionOutcome.INCONCLUSIVE, "quality evaluation coordinates are missing"

        if task.quality.max_ppl_regression_percent is not None:
            if baseline_quality.perplexity is None or candidate_quality.perplexity is None:
                checks.append(GateCheck(name="perplexity", passed=None, detail="missing"))
                return DecisionOutcome.INCONCLUSIVE, "perplexity values are missing"
            regression = (
                candidate_quality.perplexity / baseline_quality.perplexity - 1
            ) * 100
            passed = regression <= task.quality.max_ppl_regression_percent
            checks.append(
                GateCheck(
                    name="perplexity",
                    passed=passed,
                    detail=(
                        f"{regression:.4f}% regression; allowed "
                        f"<= {task.quality.max_ppl_regression_percent:.4f}%"
                    ),
                )
            )
            if not passed:
                return DecisionOutcome.REJECT, "perplexity regression exceeds the quality budget"

        for requirement in task.quality.resolved_accuracy_requirements():
            metric = requirement.metric
            if (
                metric not in baseline_quality.accuracies
                or metric not in candidate_quality.accuracies
            ):
                checks.append(
                    GateCheck(
                        name=f"accuracy.{metric}",
                        passed=None,
                        detail=f"{metric} missing",
                    )
                )
                return DecisionOutcome.INCONCLUSIVE, "required accuracy metric is missing"
            drop = (
                baseline_quality.accuracies[metric] - candidate_quality.accuracies[metric]
            ) * 100
            passed = drop <= requirement.max_drop_percentage_points
            checks.append(
                GateCheck(
                    name=f"accuracy.{metric}",
                    passed=passed,
                    detail=(
                        f"{drop:.4f} percentage-point drop; allowed <= "
                        f"{requirement.max_drop_percentage_points:.4f}"
                    ),
                )
            )
            if not passed:
                return (
                    DecisionOutcome.REJECT,
                    f"{metric} drop exceeds the quality budget",
                )
        return None
