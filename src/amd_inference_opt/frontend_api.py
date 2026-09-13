"""Stable, read-only projections for the local control-plane UI.

The UI deliberately consumes these DTOs rather than persisted workflow models.  This
keeps the browser contract stable while campaign and experiment schemas evolve.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, field_validator

from .eval_suites import (
    BALANCED_200_POLICY,
    PROVISIONAL_MATH_100_POLICY,
    QualityPolicyId,
    QualitySuiteId,
    quality_policy_suites,
)
from .experiment_bundle import (
    BundleArtifactEntry,
    ExperimentBundleManifest,
    ExperimentBundleSummary,
)
from .models import ArtifactRef, OptimizationTask, StrictModel, utc_now
from .optimization_map import (
    CapabilityAvailability,
    CapabilityRunStatus,
    OptimizationEdgeState,
    OptimizationEdgeView,
    OptimizationEvidenceLink,
    OptimizationMapV1,
    OptimizationNodeView,
    load_default_topology,
)

FRONTEND_API_SCHEMA = "gpuopt.control-plane.v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_PREVIEW_MEDIA_TYPES = {
    "application/json",
    "application/x-yaml",
    "text/markdown",
    "text/plain",
    "text/x-diff",
}


class SourceViewV1(StrictModel):
    id: str
    label: str
    kind: Literal["experiment_store", "legacy_report"]

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("source id must contain only letters, digits, '-' and '_'")
        return value


class ControlPlaneMetaV1(StrictModel):
    schema_name: Literal["gpuopt.control-plane-meta.v1"] = Field(
        default="gpuopt.control-plane-meta.v1", alias="schema"
    )
    api_version: Literal["v1"] = "v1"
    read_only: Literal[True] = True
    refresh_seconds: int = 5
    sources: list[SourceViewV1] = Field(default_factory=list)


class EvidenceLinkV1(StrictModel):
    path: str
    sha256: str | None = None


class MetricViewV1(StrictModel):
    name: str
    unit: str
    baseline: float | None = None
    candidate: float | None = None
    delta_percent: float | None = None
    sample_count: int | None = None
    cv_percent: float | None = None
    evidence: list[EvidenceLinkV1] = Field(default_factory=list)


class AccuracyComparisonViewV1(StrictModel):
    baseline_percent: float | None = None
    candidate_percent: float | None = None
    delta_points: float | None = None
    candidate_correct: int | None = Field(default=None, ge=0)
    candidate_total: int | None = Field(default=None, ge=1)


class PerplexityComparisonViewV1(StrictModel):
    baseline: float | None = Field(default=None, gt=0)
    candidate: float | None = Field(default=None, gt=0)
    delta_percent: float | None = None


class QualitySummaryViewV1(StrictModel):
    policy: QualityPolicyId
    suite_ids: list[QualitySuiteId]
    status: str | None = None
    math: AccuracyComparisonViewV1 = Field(default_factory=AccuracyComparisonViewV1)
    general: AccuracyComparisonViewV1 = Field(default_factory=AccuracyComparisonViewV1)
    perplexity: PerplexityComparisonViewV1 = Field(
        default_factory=PerplexityComparisonViewV1
    )


class ArtifactViewV1(StrictModel):
    path: str
    sha256: str
    size: int = Field(ge=0)
    media_type: str
    producer: str
    preview_available: bool
    integrity: Literal["verified", "missing", "mismatch", "unchecked"]


class EventViewV1(StrictModel):
    timestamp: datetime
    event: str
    summary: str


class CapabilityViewV1(StrictModel):
    id: str
    title: str
    category: Literal["model", "kernel", "runtime", "report"]
    status: Literal[
        "NOT_STARTED",
        "RUNNING",
        "COMPLETE",
        "ACCEPT",
        "REJECT",
        "MATERIAL_BENEFIT",
        "NO_MATERIAL_EFFECT",
        "HARMFUL",
        "OPPORTUNITY_FOUND",
        "NO_ACTION",
        "EVIDENCE_CLOSED",
        "INCONCLUSIVE",
    ]
    summary: str = ""
    experiment_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceLinkV1] = Field(default_factory=list)


class EvidenceProviderViewV1(StrictModel):
    id: str
    name: str
    integration: str
    availability: Literal["AVAILABLE", "UNAVAILABLE", "UNKNOWN"]
    capability_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceLinkV1] = Field(default_factory=list)
    detail: str = ""


class ExperimentViewV1(StrictModel):
    id: str
    projection: Literal["bundle", "legacy"] = "legacy"
    strategy: str | None = None
    status: str
    decision: str | None = None
    hypothesis: str | None = None
    change_summary: str | None = None
    baseline_metrics: list[MetricViewV1] = Field(default_factory=list)
    candidate_metrics: list[MetricViewV1] = Field(default_factory=list)
    quality: dict[str, Any] | None = None
    quality_summary: QualitySummaryViewV1 | None = None
    gate_reasons: list[str] = Field(default_factory=list)
    evidence: list[EvidenceLinkV1] = Field(default_factory=list)
    artifact_count: int = Field(default=0, ge=0)
    source_digest: str | None = None


class WorkflowStageViewV1(StrictModel):
    name: str
    status: Literal["PENDING", "RUNNING", "COMPLETE", "SKIPPED"]
    completed_at: datetime | None = None
    evidence: list[EvidenceLinkV1] = Field(default_factory=list)


class RunSummaryV1(StrictModel):
    source_id: str
    run_id: str
    kind: Literal["workflow", "campaign", "legacy_report"]
    model_name: str | None = None
    model_sha256: str | None = None
    architecture: str | None = None
    quantization: str | None = None
    runtime: str | None = None
    runtime_commit: str | None = None
    gpu: str | None = None
    gfx: str | None = None
    stage: str
    status: str
    updated_at: datetime | None = None
    experiment_count: int = 0
    artifact_count: int = 0
    accepted: int = 0
    rejected: int = 0
    inconclusive: int = 0


class RunDetailV1(StrictModel):
    schema_name: Literal["gpuopt.run-detail.v1"] = Field(
        default="gpuopt.run-detail.v1", alias="schema"
    )
    generated_at: datetime = Field(default_factory=utc_now)
    snapshot_revision: int | None = None
    summary: RunSummaryV1
    workflow_stages: list[WorkflowStageViewV1] = Field(default_factory=list)
    capabilities: list[CapabilityViewV1] = Field(default_factory=list)
    evidence_providers: list[EvidenceProviderViewV1] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    experiments: list[ExperimentViewV1] = Field(default_factory=list)
    metrics: list[MetricViewV1] = Field(default_factory=list)
    quality: dict[str, Any] | None = None
    quality_summary: QualitySummaryViewV1 | None = None
    recent_events: list[EventViewV1] = Field(default_factory=list)
    artifacts: list[ArtifactViewV1] = Field(default_factory=list)
    final_report: EvidenceLinkV1 | None = None
    warnings: list[str] = Field(default_factory=list)


class RunListV1(StrictModel):
    schema_name: Literal["gpuopt.run-list.v1"] = Field(
        default="gpuopt.run-list.v1", alias="schema"
    )
    generated_at: datetime = Field(default_factory=utc_now)
    total: int
    items: list[RunSummaryV1]
    warnings: list[str] = Field(default_factory=list)


class ArtifactPreviewV1(StrictModel):
    schema_name: Literal["gpuopt.artifact-preview.v1"] = Field(
        default="gpuopt.artifact-preview.v1", alias="schema"
    )
    artifact: ArtifactViewV1
    content: str


class ApiErrorV1(StrictModel):
    schema_name: Literal["gpuopt.api-error.v1"] = Field(
        default="gpuopt.api-error.v1", alias="schema"
    )
    code: str
    message: str


class FrontendReadError(RuntimeError):
    """A source or requested projection cannot be read safely."""


def _finite_number(value: object) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _accuracy_fraction(value: dict[str, Any], metric: str) -> float | None:
    accuracies = value.get("accuracies")
    raw = accuracies.get(metric) if isinstance(accuracies, dict) else value.get(metric)
    number = _finite_number(raw)
    if number is not None and 0 <= number <= 1:
        return number
    prefix = metric.removesuffix("_accuracy")
    correct = value.get(f"{prefix}_correct")
    total = value.get(f"{prefix}_total")
    if (
        isinstance(correct, int)
        and not isinstance(correct, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        and total > 0
        and 0 <= correct <= total
    ):
        return correct / total
    return None


def _candidate_counts(value: dict[str, Any], metric: str) -> tuple[int | None, int | None]:
    prefix = metric.removesuffix("_accuracy")
    correct = value.get(f"{prefix}_correct")
    total = value.get(f"{prefix}_total")
    if (
        isinstance(correct, int)
        and not isinstance(correct, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        and total > 0
        and 0 <= correct <= total
    ):
        return correct, total
    return None, None


def project_quality_summary(quality: dict[str, Any] | None) -> QualitySummaryViewV1 | None:
    """Normalize legacy, paired, and bundle quality records for the UI."""

    if not isinstance(quality, dict):
        return None
    baseline = quality.get("baseline")
    candidate = quality.get("candidate")
    baseline = baseline if isinstance(baseline, dict) else {}
    candidate = candidate if isinstance(candidate, dict) else quality
    if isinstance(quality.get("baseline_accuracies"), dict):
        baseline = {
            **baseline,
            "accuracies": quality["baseline_accuracies"],
            "perplexity": quality.get("baseline_perplexity"),
        }
    if isinstance(quality.get("candidate_accuracies"), dict):
        candidate = {
            **candidate,
            "accuracies": quality["candidate_accuracies"],
            "perplexity": quality.get("candidate_perplexity"),
        }

    def comparison(metric: str) -> AccuracyComparisonViewV1:
        baseline_value = _accuracy_fraction(baseline, metric)
        candidate_value = _accuracy_fraction(candidate, metric)
        if (
            metric == "math_accuracy"
            and candidate_value is None
            and quality.get("accuracy_metric") == "math_accuracy"
        ):
            candidate_value = _finite_number(quality.get("candidate_accuracy"))
            baseline_value = _finite_number(quality.get("baseline_accuracy"))
        correct, total = _candidate_counts(candidate, metric)
        return AccuracyComparisonViewV1(
            baseline_percent=(baseline_value * 100 if baseline_value is not None else None),
            candidate_percent=(
                candidate_value * 100 if candidate_value is not None else None
            ),
            delta_points=(
                (candidate_value - baseline_value) * 100
                if candidate_value is not None and baseline_value is not None
                else None
            ),
            candidate_correct=correct,
            candidate_total=total,
        )

    math_view = comparison("math_accuracy")
    general_view = comparison("general_accuracy")
    baseline_ppl = _finite_number(
        quality.get("baseline_perplexity", baseline.get("perplexity"))
    )
    candidate_ppl = _finite_number(
        quality.get("candidate_perplexity", candidate.get("perplexity"))
    )
    if baseline_ppl is not None and baseline_ppl <= 0:
        baseline_ppl = None
    if candidate_ppl is not None and candidate_ppl <= 0:
        candidate_ppl = None
    ppl_delta = _finite_number(quality.get("perplexity_regression_percent"))
    if ppl_delta is None and baseline_ppl is not None and candidate_ppl is not None:
        ppl_delta = (candidate_ppl / baseline_ppl - 1) * 100
    raw_policy = quality.get("quality_policy", quality.get("policy"))
    policy: QualityPolicyId = (
        BALANCED_200_POLICY
        if raw_policy == BALANCED_200_POLICY
        or general_view.baseline_percent is not None
        or general_view.candidate_percent is not None
        else PROVISIONAL_MATH_100_POLICY
    )
    status = quality.get("status", candidate.get("status"))
    return QualitySummaryViewV1(
        policy=policy,
        suite_ids=list(quality_policy_suites(policy)),
        status=str(status) if status is not None else None,
        math=math_view,
        general=general_view,
        perplexity=PerplexityComparisonViewV1(
            baseline=baseline_ppl,
            candidate=candidate_ppl,
            delta_percent=ppl_delta,
        ),
    )


class ReadOnlyStoreSource:
    """Read an existing ExperimentStore root without constructing a mutating store."""

    def __init__(self, source_id: str, root: str | Path, *, label: str | None = None) -> None:
        if not _SAFE_ID.fullmatch(source_id):
            raise FrontendReadError("unsafe source id")
        candidate = Path(root).expanduser()
        if candidate.is_symlink() or not candidate.is_dir():
            raise FrontendReadError("store root must be an existing non-symlink directory")
        self.id = source_id
        self.label = label or source_id
        self.root = candidate.resolve(strict=True)

    @property
    def view(self) -> SourceViewV1:
        return SourceViewV1(id=self.id, label=self.label, kind="experiment_store")

    @staticmethod
    def _safe_identifier(value: str, label: str) -> None:
        if not _SAFE_ID.fullmatch(value):
            raise FrontendReadError(f"unsafe {label}")

    def _run_dir(self, run_id: str) -> Path:
        self._safe_identifier(run_id, "run id")
        path = self.root / run_id
        if path.is_symlink() or not path.is_dir():
            raise FrontendReadError("run not found")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self.root):
            raise FrontendReadError("run escapes source root")
        return resolved

    @staticmethod
    def _load_json(path: Path) -> Any:
        if path.is_symlink() or not path.is_file():
            raise FrontendReadError("record not found")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FrontendReadError("invalid JSON record") from error

    @staticmethod
    def _optional_json(path: Path) -> Any | None:
        if not path.is_file() or path.is_symlink():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def run_ids(self) -> list[str]:
        result: list[str] = []
        for path in sorted(self.root.iterdir(), key=lambda item: item.name):
            if (
                path.is_dir()
                and not path.is_symlink()
                and _SAFE_ID.fullmatch(path.name)
                and (path / "task.json").is_file()
            ):
                result.append(path.name)
        return result

    @staticmethod
    def _artifact_manifest(run_dir: Path) -> dict[str, ArtifactRef]:
        raw = ReadOnlyStoreSource._optional_json(run_dir / "artifacts/manifest.json")
        if not isinstance(raw, dict) or not isinstance(raw.get("artifacts"), dict):
            return {}
        result: dict[str, ArtifactRef] = {}
        for path, value in raw["artifacts"].items():
            try:
                result[path] = ArtifactRef.model_validate(value)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _candidate_counts(campaign: dict[str, Any] | None) -> tuple[int, int, int]:
        values = [] if not isinstance(campaign, dict) else campaign.get("candidates", [])
        rendered = [str(item.get("disposition", "")) for item in values if isinstance(item, dict)]
        return (
            sum(value in {"ACCEPT", "ACCEPTED", "EXPERIMENTAL_ACCEPTED"} for value in rendered),
            sum(value in {"REJECT", "REJECTED"} for value in rendered),
            sum(value == "INCONCLUSIVE" for value in rendered),
        )

    @staticmethod
    def _experiment_counts(directory: Path) -> tuple[int, int, int]:
        root = directory / "experiments"
        outcomes: list[str] = []
        if not root.is_dir() or root.is_symlink():
            return 0, 0, 0
        for experiment in root.iterdir():
            if not experiment.is_dir() or experiment.is_symlink():
                continue
            gate = ReadOnlyStoreSource._optional_json(experiment / "gate-decision.json")
            if gate is None:
                gate = ReadOnlyStoreSource._optional_json(
                    experiment / "performance-pre-gate.json"
                )
            if isinstance(gate, dict):
                outcomes.append(str(gate.get("outcome", "")))
        return (
            sum(value == "ACCEPT" for value in outcomes),
            sum(value == "REJECT" for value in outcomes),
            sum(value == "INCONCLUSIVE" for value in outcomes),
        )

    def summary(self, run_id: str) -> RunSummaryV1:
        directory = self._run_dir(run_id)
        task = self.task(run_id)
        campaign = self._optional_json(directory / "state/campaign.json")
        if campaign is None:
            campaign = self._optional_json(directory / "state/gfx1201-campaign.json")
        workflow = self._optional_json(directory / "state/workflow.json")
        if workflow is None:
            workflow = self._optional_json(directory / "state/vllm-workflow.json")
        state = campaign if isinstance(campaign, dict) else workflow
        state = state if isinstance(state, dict) else {}
        accepted, rejected, inconclusive = self._candidate_counts(campaign)
        experiment_dir = directory / "experiments"
        experiment_count = sum(
            1 for item in experiment_dir.iterdir() if item.is_dir() and not item.is_symlink()
        ) if experiment_dir.is_dir() else 0
        if not isinstance(campaign, dict) or not campaign.get("candidates"):
            accepted, rejected, inconclusive = self._experiment_counts(directory)
        return RunSummaryV1(
            source_id=self.id,
            run_id=run_id,
            kind="campaign" if isinstance(campaign, dict) else "workflow",
            model_name=Path(task.model.path).name,
            model_sha256=task.model.sha256,
            architecture=task.model.architecture,
            quantization=task.model.quantization,
            runtime=task.runtime.name,
            runtime_commit=task.runtime.base_commit,
            gpu=task.gpu.name,
            gfx=task.gpu.gfx_target,
            stage=str(state.get("current_stage", "UNKNOWN")),
            status=str(state.get("status", "UNKNOWN")),
            updated_at=state.get("updated_at"),
            experiment_count=experiment_count,
            artifact_count=len(self._artifact_manifest(directory)),
            accepted=accepted,
            rejected=rejected,
            inconclusive=inconclusive,
        )

    def task(self, run_id: str) -> OptimizationTask:
        """Return the validated task record without exposing a writable store."""

        directory = self._run_dir(run_id)
        return OptimizationTask.model_validate(self._load_json(directory / "task.json"))

    @staticmethod
    def _reject_symlink_components(base: Path, portable: PurePosixPath) -> Path:
        current = base
        for part in portable.parts:
            current = current / part
            if current.is_symlink():
                raise FrontendReadError("artifact path contains a symbolic link")
        return current

    @staticmethod
    def _artifact_view(directory: Path, artifact: ArtifactRef) -> ArtifactViewV1:
        portable = PurePosixPath(artifact.path)
        try:
            lexical = ReadOnlyStoreSource._reject_symlink_components(directory, portable)
            path = lexical.resolve(strict=False)
        except FrontendReadError:
            path = directory.parent
        integrity: Literal["verified", "missing", "mismatch", "unchecked"] = "unchecked"
        if not path.is_relative_to(directory) or path.is_symlink() or not path.is_file():
            integrity = "missing"
        elif path.stat().st_size != artifact.size:
            integrity = "mismatch"
        preview = artifact.media_type in _PREVIEW_MEDIA_TYPES and artifact.size <= 512 * 1024
        return ArtifactViewV1(
            path=artifact.path,
            sha256=artifact.sha256,
            size=artifact.size,
            media_type=artifact.media_type,
            producer=artifact.producer,
            preview_available=preview and integrity != "missing",
            integrity=integrity,
        )

    @staticmethod
    def _events(directory: Path, *, limit: int = 100) -> list[EventViewV1]:
        path = directory / "events/events.jsonl"
        if path.is_symlink() or not path.is_file():
            return []
        events: list[EventViewV1] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines[-limit:]:
            try:
                value = json.loads(line)
                payload = value.get("payload", {})
                summary = str(payload.get("summary", value.get("event", "")))
                events.append(
                    EventViewV1(
                        timestamp=value["timestamp"],
                        event=str(value["event"]),
                        summary=summary[:240],
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return events

    @staticmethod
    def _metric_views(
        baseline: dict[str, Any],
        candidate: dict[str, Any],
        gate: dict[str, Any],
        *,
        baseline_path: str,
        candidate_path: str,
    ) -> list[MetricViewV1]:
        baseline_metrics = baseline.get("benchmark", {}).get("metrics", {})
        candidate_metrics = candidate.get("metrics", {})
        improvements = gate.get("metric_improvements_percent", {})
        if not isinstance(baseline_metrics, dict) or not isinstance(candidate_metrics, dict):
            return []
        values: list[MetricViewV1] = []
        for name in sorted(set(baseline_metrics) & set(candidate_metrics)):
            raw_baseline = baseline_metrics[name]
            raw_candidate = candidate_metrics[name]
            if not isinstance(raw_baseline, dict) or not isinstance(raw_candidate, dict):
                continue
            baseline_samples = raw_baseline.get("samples")
            candidate_samples = raw_candidate.get("samples")
            if not (
                isinstance(baseline_samples, list)
                and baseline_samples
                and isinstance(candidate_samples, list)
                and candidate_samples
                and all(isinstance(value, (int, float)) for value in baseline_samples)
                and all(isinstance(value, (int, float)) for value in candidate_samples)
            ):
                continue
            baseline_values = [float(value) for value in baseline_samples]
            candidate_values = [float(value) for value in candidate_samples]
            candidate_mean = statistics.fmean(candidate_values)
            cv = (
                statistics.stdev(candidate_values) / candidate_mean * 100
                if len(candidate_values) > 1 and candidate_mean
                else 0.0
            )
            delta = improvements.get(name) if isinstance(improvements, dict) else None
            values.append(
                MetricViewV1(
                    name=name,
                    unit=str(raw_candidate.get("unit", raw_baseline.get("unit", ""))),
                    baseline=statistics.fmean(baseline_values),
                    candidate=candidate_mean,
                    delta_percent=float(delta) if isinstance(delta, (int, float)) else None,
                    sample_count=len(candidate_values),
                    cv_percent=cv,
                    evidence=[
                        EvidenceLinkV1(path=baseline_path),
                        EvidenceLinkV1(path=candidate_path),
                    ],
                )
            )
        return values

    @staticmethod
    def _verified_registered_bytes(
        directory: Path,
        artifact_path: str,
        manifest: dict[str, ArtifactRef],
    ) -> tuple[bytes, ArtifactRef]:
        artifact = manifest.get(artifact_path)
        if artifact is None or artifact.path != artifact_path:
            raise FrontendReadError(f"bundle artifact is not registered: {artifact_path}")
        portable = PurePosixPath(artifact_path)
        lexical = ReadOnlyStoreSource._reject_symlink_components(directory, portable)
        try:
            path = lexical.resolve(strict=True)
        except OSError as error:
            raise FrontendReadError(f"bundle artifact is missing: {artifact_path}") from error
        if not path.is_relative_to(directory) or path.is_symlink() or not path.is_file():
            raise FrontendReadError(f"unsafe bundle artifact: {artifact_path}")
        try:
            data = path.read_bytes()
        except OSError as error:
            raise FrontendReadError(f"bundle artifact is unreadable: {artifact_path}") from error
        if len(data) != artifact.size or hashlib.sha256(data).hexdigest() != artifact.sha256:
            raise FrontendReadError(f"bundle artifact integrity failed: {artifact_path}")
        return data, artifact

    @staticmethod
    def _bundle_source_digest(entries: list[BundleArtifactEntry]) -> str:
        coordinates = [
            {
                "path": entry.artifact.path,
                "sha256": entry.artifact.sha256,
                "size": entry.artifact.size,
                "role": entry.role.value,
                "attempt_id": entry.attempt_id,
                "current": entry.current,
            }
            for entry in entries
        ]
        payload = json.dumps(coordinates, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _bundle_experiment(
        directory: Path,
        experiment_id: str,
        task_manifest: dict[str, ArtifactRef],
    ) -> ExperimentViewV1 | None:
        prefix = f"experiments/{experiment_id}/"
        summary_path = f"{prefix}summary.json"
        manifest_path = f"{prefix}manifest.json"
        summary_registered = summary_path in task_manifest
        manifest_registered = manifest_path in task_manifest
        if not summary_registered and not manifest_registered:
            return None
        if not summary_registered or not manifest_registered:
            raise FrontendReadError("bundle summary and manifest must both be registered")

        manifest_data, _ = ReadOnlyStoreSource._verified_registered_bytes(
            directory, manifest_path, task_manifest
        )
        summary_data, summary_ref = ReadOnlyStoreSource._verified_registered_bytes(
            directory, summary_path, task_manifest
        )
        try:
            bundle_manifest = ExperimentBundleManifest.model_validate_json(manifest_data)
            summary = ExperimentBundleSummary.model_validate_json(summary_data)
        except ValueError as error:
            raise FrontendReadError("bundle schema validation failed") from error

        run_id = directory.name
        if summary.task_id != run_id or bundle_manifest.task_id != run_id:
            raise FrontendReadError("bundle task id mismatch")
        if (
            summary.experiment_id != experiment_id
            or bundle_manifest.experiment_id != experiment_id
        ):
            raise FrontendReadError("bundle experiment id mismatch")
        if bundle_manifest.summary != summary_ref or bundle_manifest.summary.path != summary_path:
            raise FrontendReadError("bundle summary reference mismatch")
        expected_attempts_path = f"{prefix}attempts/index.json"
        if bundle_manifest.attempts_index.path != expected_attempts_path:
            raise FrontendReadError("bundle attempt index path mismatch")

        seen: set[str] = set()
        for entry in bundle_manifest.artifacts:
            path = entry.artifact.path
            if path in seen:
                raise FrontendReadError("bundle contains duplicate artifact references")
            seen.add(path)
            if task_manifest.get(path) != entry.artifact:
                raise FrontendReadError(f"bundle artifact metadata is not registered: {path}")
        if summary_path not in seen or expected_attempts_path not in seen:
            raise FrontendReadError("bundle commit artifacts are incomplete")

        source_entries = [
            entry
            for entry in bundle_manifest.artifacts
            if entry.artifact.path not in {summary_path, expected_attempts_path}
        ]
        source_digest = ReadOnlyStoreSource._bundle_source_digest(source_entries)
        if (
            summary.source_digest != bundle_manifest.source_digest
            or summary.source_digest != source_digest
        ):
            raise FrontendReadError("bundle source digest mismatch")
        if summary.artifact_count != len(source_entries):
            raise FrontendReadError("bundle artifact count mismatch")

        bundle_links = {
            entry.artifact.path: EvidenceLinkV1(
                path=entry.artifact.path,
                sha256=entry.artifact.sha256,
            )
            for entry in bundle_manifest.artifacts
        }
        manifest_ref = task_manifest[manifest_path]
        bundle_links[manifest_path] = EvidenceLinkV1(
            path=manifest_ref.path,
            sha256=manifest_ref.sha256,
        )
        metric_evidence = [
            EvidenceLinkV1(path=summary_ref.path, sha256=summary_ref.sha256),
            EvidenceLinkV1(path=manifest_ref.path, sha256=manifest_ref.sha256),
        ]
        metrics = [
            MetricViewV1(
                name=metric.name,
                unit=metric.unit,
                baseline=metric.baseline_mean,
                candidate=metric.candidate_mean,
                delta_percent=metric.delta_percent,
                sample_count=metric.candidate_sample_count,
                cv_percent=metric.candidate_cv_percent,
                evidence=metric_evidence,
            )
            for metric in summary.metrics
        ]
        decision = summary.decision
        raw_quality = (
            summary.quality.model_dump(mode="json")
            if summary.quality is not None
            else None
        )
        return ExperimentViewV1(
            id=experiment_id,
            projection="bundle",
            strategy=summary.strategy,
            status=summary.status,
            decision=decision.outcome if decision is not None else None,
            hypothesis=summary.hypothesis_id,
            change_summary=summary.change_summary,
            candidate_metrics=metrics,
            quality=raw_quality,
            quality_summary=project_quality_summary(raw_quality),
            gate_reasons=decision.reasons if decision is not None else [],
            evidence=[bundle_links[path] for path in sorted(bundle_links)],
            artifact_count=summary.artifact_count,
            source_digest=summary.source_digest,
        )

    @staticmethod
    def _experiments(
        directory: Path,
        manifest: dict[str, ArtifactRef],
    ) -> tuple[list[ExperimentViewV1], list[str]]:
        root = directory / "experiments"
        baseline_path = "artifacts/baseline.json"
        baseline = ReadOnlyStoreSource._optional_json(directory / baseline_path)
        baseline = baseline if isinstance(baseline, dict) else {}
        if not root.is_dir() or root.is_symlink():
            return [], []
        values: list[ExperimentViewV1] = []
        warnings: list[str] = []
        for experiment in sorted(root.iterdir(), key=lambda item: item.name):
            if (
                not experiment.is_dir()
                or experiment.is_symlink()
                or not _SAFE_ID.fullmatch(experiment.name)
            ):
                continue
            try:
                bundle = ReadOnlyStoreSource._bundle_experiment(
                    directory, experiment.name, manifest
                )
            except FrontendReadError as error:
                warnings.append(
                    f"{experiment.name}: invalid experiment bundle; using legacy evidence ({error})"
                )
                bundle = None
            if bundle is not None:
                values.append(bundle)
                continue
            spec = ReadOnlyStoreSource._optional_json(experiment / "spec.json")
            e2e = ReadOnlyStoreSource._optional_json(experiment / "e2e-result.json")
            gate = ReadOnlyStoreSource._optional_json(experiment / "gate-decision.json")
            if gate is None:
                gate = ReadOnlyStoreSource._optional_json(
                    experiment / "performance-pre-gate.json"
                )
            quality = ReadOnlyStoreSource._optional_json(experiment / "quality-result.json")
            result = ReadOnlyStoreSource._optional_json(experiment / "result.json")
            spec = spec if isinstance(spec, dict) else {}
            e2e = e2e if isinstance(e2e, dict) else {}
            gate = gate if isinstance(gate, dict) else {}
            quality = quality if isinstance(quality, dict) else None
            result = result if isinstance(result, dict) else {}
            change = spec.get("change", {})
            change = change if isinstance(change, dict) else {}
            status = str(
                gate.get("outcome")
                or result.get("failure_reason")
                or e2e.get("status")
                or "INCOMPLETE"
            )
            prefix = f"experiments/{experiment.name}/"
            links = [
                EvidenceLinkV1(path=artifact.path, sha256=artifact.sha256)
                for path, artifact in sorted(manifest.items())
                if path.startswith(prefix)
                and path.rsplit("/", 1)[-1]
                in {
                    "spec.json",
                    "e2e-result.json",
                    "gate-decision.json",
                    "performance-pre-gate.json",
                    "quality-result.json",
                    "result.json",
                }
            ]
            metrics = ReadOnlyStoreSource._metric_views(
                baseline,
                e2e,
                gate,
                baseline_path=baseline_path,
                candidate_path=f"{prefix}e2e-result.json",
            )
            values.append(
                ExperimentViewV1(
                    id=experiment.name,
                    projection="legacy",
                    strategy=str(change.get("kind")) if change.get("kind") else None,
                    status=status,
                    decision=str(gate.get("outcome")) if gate.get("outcome") else None,
                    hypothesis=(
                        str(spec.get("hypothesis_id"))
                        if spec.get("hypothesis_id")
                        else None
                    ),
                    change_summary=(
                        str(change.get("description"))
                        if change.get("description")
                        else None
                    ),
                    candidate_metrics=metrics,
                    quality=quality,
                    quality_summary=project_quality_summary(quality),
                    gate_reasons=[str(value) for value in gate.get("reasons", [])],
                    evidence=links,
                    artifact_count=len(links),
                )
            )
        return values, warnings

    @staticmethod
    def _stages(state: dict[str, Any] | None) -> list[WorkflowStageViewV1]:
        if not isinstance(state, dict):
            return []
        completions = state.get("completions", [])
        result: list[WorkflowStageViewV1] = []
        for item in completions if isinstance(completions, list) else []:
            if not isinstance(item, dict):
                continue
            evidence: list[EvidenceLinkV1] = []
            raw_evidence = item.get("evidence", item.get("evidence_artifacts", {}))
            if isinstance(raw_evidence, list):
                evidence = [
                    EvidenceLinkV1(path=str(value.get("path")), sha256=value.get("sha256"))
                    for value in raw_evidence
                    if isinstance(value, dict) and value.get("path")
                ]
            elif isinstance(raw_evidence, dict):
                evidence = [
                    EvidenceLinkV1(path=str(value.get("path")), sha256=value.get("sha256"))
                    for value in raw_evidence.values()
                    if isinstance(value, dict) and value.get("path")
                ]
            result.append(
                WorkflowStageViewV1(
                    name=str(item.get("stage", "UNKNOWN")),
                    status="COMPLETE",
                    completed_at=item.get("completed_at"),
                    evidence=evidence,
                )
            )
        current = str(state.get("current_stage", "UNKNOWN"))
        if current not in {item.name for item in result} and current != "COMPLETE":
            result.append(WorkflowStageViewV1(name=current, status="RUNNING"))
        return result

    @staticmethod
    def _capabilities(
        directory: Path,
        campaign: dict[str, Any] | None,
    ) -> list[CapabilityViewV1]:
        definitions = (
            ("mixed_precision", "Mixed Precision", "model", "mixed-bit"),
            ("hip_graph_ab", "HIP Graph A/B", "runtime", "hip-graph"),
            (
                "memory_reuse_audit",
                "Memory Reuse Audit",
                "runtime",
                "memory-audit",
            ),
            ("mfma_mmq_evidence", "MFMA/MMQ Evidence", "kernel", "mfma-mmq"),
            (
                "gfx1201_final_report",
                "gfx1201 Final Report",
                "report",
                "gfx1201-final-report",
            ),
        )
        completed: dict[str, dict[str, Any]] = {}
        if isinstance(campaign, dict):
            for completion in campaign.get("completions", []):
                if not isinstance(completion, dict):
                    continue
                result = completion.get("result")
                if isinstance(result, dict) and isinstance(result.get("capability"), str):
                    completed[result["capability"]] = result
        active_capability = (
            {
                "RUN_MIXED_BIT": "mixed_precision",
                "RUN_HIP_GRAPH_AB": "hip_graph_ab",
                "RUN_MEMORY_AUDIT": "memory_reuse_audit",
                "RUN_MFMA_MMQ_EVIDENCE": "mfma_mmq_evidence",
                "BUILD_FINAL_REPORT": "gfx1201_final_report",
            }.get(str(campaign.get("current_stage")))
            if isinstance(campaign, dict)
            else None
        )
        supported_statuses = {
            "ACCEPT",
            "REJECT",
            "MATERIAL_BENEFIT",
            "NO_MATERIAL_EFFECT",
            "HARMFUL",
            "INCONCLUSIVE",
            "OPPORTUNITY_FOUND",
            "NO_ACTION",
            "EVIDENCE_CLOSED",
            "COMPLETE",
        }
        values: list[CapabilityViewV1] = []
        for identifier, title, category, stem in definitions:
            matches = [
                path
                for path in (directory / "reports").glob(f"*{stem}*")
                if path.is_file() and not path.is_symlink()
            ] if (directory / "reports").is_dir() else []
            result = completed.get(identifier)
            outcome = str(result.get("outcome", "")) if result else ""
            status = (
                outcome
                if outcome in supported_statuses
                else "RUNNING"
                if identifier == active_capability
                else "COMPLETE"
                if matches
                else "NOT_STARTED"
            )
            evidence = []
            if result and isinstance(result.get("evidence"), list):
                evidence = [
                    EvidenceLinkV1(path=str(item["path"]), sha256=item.get("sha256"))
                    for item in result["evidence"]
                    if isinstance(item, dict) and item.get("path")
                ]
            values.append(
                CapabilityViewV1(
                    id=identifier,
                    title=title,
                    category=category,  # type: ignore[arg-type]
                    status=status,  # type: ignore[arg-type]
                    summary=(
                        str(result.get("summary", ""))
                        if result
                        else "Persisted report available"
                        if matches
                        else "No result recorded"
                    ),
                    experiment_ids=(
                        [str(value) for value in result.get("experiment_ids", [])]
                        if result
                        else []
                    ),
                    evidence=evidence,
                )
            )
        return values

    @staticmethod
    def _evidence_providers(
        directory: Path,
        manifest: dict[str, ArtifactRef],
    ) -> list[EvidenceProviderViewV1]:
        breakdown = ReadOnlyStoreSource._optional_json(
            directory / "reports/session-breakdown.json"
        )
        collection = (
            breakdown.get("information_collection", {})
            if isinstance(breakdown, dict)
            else {}
        )
        catalogue = collection.get("provider_catalogue", [])
        if not isinstance(catalogue, list):
            return []
        snapshot_ref = collection.get("capability_snapshot")
        snapshot = None
        if isinstance(snapshot_ref, dict) and snapshot_ref.get("path"):
            snapshot = ReadOnlyStoreSource._optional_json(
                directory / str(snapshot_ref["path"])
            )
        probe_rows = snapshot.get("providers", []) if isinstance(snapshot, dict) else []
        probe_by_id = {
            str(item.get("provider_id")): item
            for item in probe_rows
            if isinstance(item, dict)
        }
        provider_markers = {
            "magpie": ("magpie",),
            "tracelens": ("tracelens",),
            "intellikit": ("intellikit",),
            "rocm-issue-agent": ("mcp", "kernel-evidence"),
            "raw-rocprofv3": ("raw-profiler", "raw-rocprof"),
            "gpuopt-local-health": ("runtime-health", "platform"),
            "gpuopt-orchestration-audit": ("workflow", "campaign", "quality/execution"),
        }
        result: list[EvidenceProviderViewV1] = []
        for raw in catalogue:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            identifier = str(raw["id"])
            probe = probe_by_id.get(identifier, {})
            availability = str(probe.get("availability", "UNKNOWN"))
            if availability not in {"AVAILABLE", "UNAVAILABLE", "UNKNOWN"}:
                availability = "UNKNOWN"
            markers = provider_markers.get(identifier, ())
            links = [
                EvidenceLinkV1(path=item.path, sha256=item.sha256)
                for item in manifest.values()
                if any(marker in item.path.lower() for marker in markers)
            ]
            result.append(
                EvidenceProviderViewV1(
                    id=identifier,
                    name=str(raw.get("name", identifier)),
                    integration=str(raw.get("integration", "UNKNOWN")),
                    availability=availability,  # type: ignore[arg-type]
                    capability_ids=[
                        str(item.get("id"))
                        for item in raw.get("capabilities", [])
                        if isinstance(item, dict) and item.get("id")
                    ],
                    evidence=links,
                    detail=str(probe.get("detail", "No capability snapshot recorded")),
                )
            )
        return result

    def detail(self, run_id: str) -> RunDetailV1:
        directory = self._run_dir(run_id)
        campaign = self._optional_json(directory / "state/gfx1201-campaign.json")
        if campaign is None:
            campaign = self._optional_json(directory / "state/campaign.json")
        workflow = self._optional_json(directory / "state/workflow.json")
        if workflow is None:
            workflow = self._optional_json(directory / "state/vllm-workflow.json")
        state = campaign if isinstance(campaign, dict) else workflow
        manifest = self._artifact_manifest(directory)
        artifacts = [self._artifact_view(directory, item) for _, item in sorted(manifest.items())]
        final = next(
            (
                EvidenceLinkV1(path=manifest[path].path, sha256=manifest[path].sha256)
                for path in (
                    "reports/gfx1201-final-report.json",
                    "reports/session-breakdown.json",
                )
                if path in manifest
            ),
            None,
        )
        candidates = campaign.get("candidates", []) if isinstance(campaign, dict) else []
        experiments, bundle_warnings = self._experiments(directory, manifest)
        selected_experiment = next(
            (item for item in experiments if item.status == "ACCEPT"),
            experiments[-1] if experiments else None,
        )
        return RunDetailV1(
            snapshot_revision=campaign.get("revision") if isinstance(campaign, dict) else None,
            summary=self.summary(run_id),
            workflow_stages=self._stages(state),
            capabilities=self._capabilities(directory, campaign),
            evidence_providers=self._evidence_providers(directory, manifest),
            candidates=candidates if isinstance(candidates, list) else [],
            experiments=experiments,
            metrics=(selected_experiment.candidate_metrics if selected_experiment else []),
            quality=selected_experiment.quality if selected_experiment else None,
            quality_summary=(
                selected_experiment.quality_summary if selected_experiment else None
            ),
            recent_events=self._events(directory),
            artifacts=artifacts,
            final_report=final,
            warnings=bundle_warnings,
        )

    def experiment(self, run_id: str, experiment_id: str) -> ExperimentViewV1:
        """Return one verified bundle projection, or its legacy fallback."""

        self._safe_identifier(experiment_id, "experiment id")
        directory = self._run_dir(run_id)
        experiments, _ = self._experiments(directory, self._artifact_manifest(directory))
        for experiment in experiments:
            if experiment.id == experiment_id:
                return experiment
        raise FrontendReadError("experiment not found")

    def optimization_map(self, run_id: str) -> OptimizationMapV1:
        """Overlay one persisted run on the config-driven framework topology."""

        directory = self._run_dir(run_id)
        detail = self.detail(run_id)
        topology = load_default_topology()
        manifest = self._artifact_manifest(directory)
        by_id = {node.id: node for node in topology.nodes}
        updates: dict[str, dict[str, Any]] = {}

        priority = {
            CapabilityRunStatus.NOT_STARTED: 0,
            CapabilityRunStatus.UNAVAILABLE: 1,
            CapabilityRunStatus.REJECT: 2,
            CapabilityRunStatus.INCONCLUSIVE: 3,
            CapabilityRunStatus.COMPLETE: 4,
            CapabilityRunStatus.NO_ACTION: 5,
            CapabilityRunStatus.OPPORTUNITY_FOUND: 6,
            CapabilityRunStatus.RUNNING: 7,
            CapabilityRunStatus.ACCEPT: 8,
        }

        def links(paths: list[str]) -> list[OptimizationEvidenceLink]:
            values = []
            for path in paths:
                artifact = manifest.get(path)
                values.append(
                    OptimizationEvidenceLink(
                        path=path,
                        sha256=artifact.sha256 if artifact else None,
                    )
                )
            return values

        def update(
            node_id: str,
            status: CapabilityRunStatus,
            summary: str,
            *,
            experiment_ids: list[str] | None = None,
            evidence: list[OptimizationEvidenceLink] | None = None,
        ) -> None:
            if node_id not in by_id:
                return
            current = updates.get(node_id)
            if current is None or priority[status] >= priority[current["status"]]:
                updates[node_id] = {
                    "status": status,
                    "summary": summary,
                    "experiment_ids": list(experiment_ids or []),
                    "evidence": list(evidence or []),
                }
            else:
                current["experiment_ids"] = sorted(
                    set(current["experiment_ids"]) | set(experiment_ids or [])
                )
                known_paths = {item.path for item in current["evidence"]}
                current["evidence"].extend(
                    item for item in evidence or [] if item.path not in known_paths
                )

        artifact_rules = (
            ("artifacts/inspection.json", "target-inspection", "Target was inspected"),
            ("artifacts/baseline.json", "shared-baseline", "Baseline was captured"),
            (
                "artifacts/execution-map.json",
                "execution-map",
                "Execution map was initialized",
            ),
        )
        for path, node_id, summary in artifact_rules:
            if path in manifest:
                update(
                    node_id,
                    CapabilityRunStatus.COMPLETE,
                    summary,
                    evidence=links([path]),
                )
        evidence_artifact_nodes = (
            (
                "evidence-capabilities",
                "evidence-provider-registry",
                "Evidence provider capability snapshot is persisted",
            ),
            (
                "runtime-health",
                "runtime-health-evidence",
                "Runtime health evidence is persisted",
            ),
            (
                "kernel-shape-manifest",
                "kernel-shape-manifest",
                "Shape-aware kernel variants are persisted",
            ),
            (
                "magpie-evidence",
                "magpie-evidence-import",
                "Normalized Magpie evidence is persisted",
            ),
            (
                "tracelens-evidence",
                "tracelens-evidence-import",
                "Normalized TraceLens evidence is persisted",
            ),
            (
                "intellikit-evidence",
                "intellikit-evidence-import",
                "IntelliKit analysis evidence is persisted",
            ),
        )
        for marker, node_id, summary in evidence_artifact_nodes:
            selected = [path for path in manifest if marker in path]
            if selected:
                update(
                    node_id,
                    CapabilityRunStatus.COMPLETE,
                    summary,
                    evidence=links(selected),
                )
        profile_paths = [
            path
            for path in manifest
            if "kernel-evidence" in path
            or "/profiles/" in path
            or "raw-profiler" in path and path.endswith("outcome.json")
        ]
        if profile_paths:
            update(
                "rocm-kernel-profile",
                CapabilityRunStatus.COMPLETE,
                "Kernel or trace evidence is persisted",
                evidence=links(profile_paths),
            )
        isa_paths = [
            path
            for path in manifest
            if any(word in path.lower() for word in ("isa", "code-object", "resource"))
        ]
        if isa_paths:
            update(
                "isa-resource-evidence",
                CapabilityRunStatus.COMPLETE,
                "ISA or code-object resource evidence is persisted",
                evidence=links(isa_paths),
            )

        capability_nodes = {
            "mixed_precision": [
                "tensor-sensitivity",
                "precision-assignment",
                "weight-packing",
                "mixed-precision",
            ],
            "hip_graph_ab": ["hip-graph-ab"],
            "memory_reuse_audit": ["buffer-memory-audit", "memory-copy-audit"],
            "mfma_mmq_evidence": ["mfma-mmq-evidence"],
            "gfx1201_final_report": ["gfx1201-final-report"],
        }
        status_values = {item.value for item in CapabilityRunStatus}
        for capability in detail.capabilities:
            status = (
                CapabilityRunStatus(capability.status)
                if capability.status in status_values
                else CapabilityRunStatus.COMPLETE
                if capability.status in {"MATERIAL_BENEFIT", "EVIDENCE_CLOSED"}
                else CapabilityRunStatus.REJECT
                if capability.status == "HARMFUL"
                else CapabilityRunStatus.COMPLETE
                if capability.status == "NO_MATERIAL_EFFECT"
                else CapabilityRunStatus.NOT_STARTED
            )
            for node_id in capability_nodes.get(capability.id, []):
                update(
                    node_id,
                    status,
                    capability.summary,
                    experiment_ids=capability.experiment_ids,
                    evidence=[
                        OptimizationEvidenceLink(path=item.path, sha256=item.sha256)
                        for item in capability.evidence
                    ],
                )

        disposition_status = {
            "EXPERIMENTAL_ACCEPTED": CapabilityRunStatus.ACCEPT,
            "ACCEPTED": CapabilityRunStatus.ACCEPT,
            "ACCEPT": CapabilityRunStatus.ACCEPT,
            "REJECTED": CapabilityRunStatus.REJECT,
            "REJECT": CapabilityRunStatus.REJECT,
            "INCONCLUSIVE": CapabilityRunStatus.INCONCLUSIVE,
            "RUNNING": CapabilityRunStatus.RUNNING,
            "PLANNED": CapabilityRunStatus.NOT_STARTED,
        }
        strategy_nodes = {
            "mixed_bit": ["mixed-precision"],
            "shape_kernel": ["shape-specific-kernel"],
            "kv_cache": ["kv-cache-quantization"],
        }
        for candidate in detail.candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("id", "unknown"))
            status = disposition_status.get(
                str(candidate.get("disposition", "")), CapabilityRunStatus.NOT_STARTED
            )
            candidate_links = [
                OptimizationEvidenceLink(
                    path=str(item["path"]), sha256=item.get("sha256")
                )
                for item in candidate.get("result_artifacts", [])
                if isinstance(item, dict) and item.get("path")
            ]
            for node_id in strategy_nodes.get(str(candidate.get("strategy", "")), []):
                update(
                    node_id,
                    status,
                    f"Candidate {candidate_id}: {status.value}",
                    experiment_ids=[candidate_id],
                    evidence=candidate_links,
                )

        for experiment in detail.experiments:
            status = disposition_status.get(
                experiment.status, CapabilityRunStatus.COMPLETE
            )
            text = " ".join(
                value
                for value in (
                    experiment.id,
                    experiment.strategy or "",
                    experiment.hypothesis or "",
                    experiment.change_summary or "",
                )
            ).lower()
            node_ids: set[str] = set()
            if "q4rdna" in text or "q4_rdna" in text:
                node_ids.update(("q4rdna-hybrid", "weight-layout-fused-dequant"))
            if "split-k" in text or "split_k" in text:
                node_ids.add("split-k")
            if any(word in text for word in ("small-k", "small_k", "shape-specific")):
                node_ids.add("shape-specific-kernel")
            if any(word in text for word in ("vector", "vdr", "dword", "vgpr")):
                node_ids.add("vector-load-vgpr")
            if any(word in text for word in ("fused-dual", "gate-up", "gate_up")):
                node_ids.add("gate-up-fusion")
            if "hip-graph" in text or "hip_graph" in text:
                node_ids.add("hip-graph-ab")
            if "kv-cache" in text or "kv_cache" in text:
                node_ids.add("kv-cache-quantization")
            if "memory" in text or "buffer" in text:
                node_ids.add("buffer-memory-audit")
            if experiment.strategy == "mixed_bit":
                node_ids.add("mixed-precision")
            elif experiment.strategy == "shape_kernel":
                node_ids.add("shape-specific-kernel")
            elif experiment.strategy == "kv_cache":
                node_ids.add("kv-cache-quantization")
            experiment_links = [
                OptimizationEvidenceLink(path=item.path, sha256=item.sha256)
                for item in experiment.evidence
            ]
            for node_id in node_ids:
                update(
                    node_id,
                    status,
                    f"Experiment {experiment.id}: {experiment.status}",
                    experiment_ids=[experiment.id],
                    evidence=experiment_links,
                )
            if experiment.candidate_metrics:
                update(
                    "performance-gate",
                    status,
                    f"Performance gate from {experiment.id}: {experiment.status}",
                    experiment_ids=[experiment.id],
                    evidence=experiment_links,
                )
            if experiment.quality is not None:
                update(
                    "quality-gate",
                    status,
                    f"Quality evidence from {experiment.id}: {experiment.status}",
                    experiment_ids=[experiment.id],
                    evidence=experiment_links,
                )
            if experiment.status in {"ACCEPT", "REJECT", "INCONCLUSIVE"}:
                update(
                    "decision-gate",
                    status,
                    f"Decision from {experiment.id}: {experiment.status}",
                    experiment_ids=[experiment.id],
                    evidence=experiment_links,
                )

        current_stage_nodes = {
            "INSPECT_TARGET": "target-inspection",
            "CAPTURE_BASELINE": "shared-baseline",
            "BUILD_EXECUTION_MAP": "execution-map",
            "DISCOVER_HOTSPOTS": "rocm-kernel-profile",
            "CLASSIFY_BOTTLENECK": "execution-map",
            "ANALYZE_LIMIT": "execution-map",
            "E2E_VALIDATION": "performance-gate",
            "QUALITY_VALIDATION": "quality-gate",
            "DECIDE": "decision-gate",
            "RUN_MIXED_BIT": "mixed-precision",
            "RUN_HIP_GRAPH_AB": "hip-graph-ab",
            "RUN_MEMORY_AUDIT": "buffer-memory-audit",
            "RUN_MFMA_MMQ_EVIDENCE": "mfma-mmq-evidence",
            "BUILD_FINAL_REPORT": "gfx1201-final-report",
        }
        active_node = current_stage_nodes.get(detail.summary.stage)
        if active_node and detail.summary.status in {"ACTIVE", "RUNNING"}:
            update(
                active_node,
                CapabilityRunStatus.RUNNING,
                f"Current workflow stage: {detail.summary.stage}",
            )

        nodes: list[OptimizationNodeView] = []
        for template in topology.nodes:
            state = updates.get(template.id)
            unavailable = template.availability == CapabilityAvailability.NOT_IMPLEMENTED
            nodes.append(
                OptimizationNodeView(
                    id=template.id,
                    title=template.title,
                    layer=template.layer,
                    availability=template.availability,
                    run_status=(
                        state["status"]
                        if state
                        else CapabilityRunStatus.UNAVAILABLE
                        if unavailable
                        else CapabilityRunStatus.NOT_STARTED
                    ),
                    summary=template.summary,
                    status_summary=(
                        state["summary"]
                        if state
                        else "Not implemented in the current framework"
                        if unavailable
                        else "No evidence recorded for this run"
                    ),
                    inputs=template.inputs,
                    outputs=template.outputs,
                    tags=template.tags,
                    experiment_ids=state["experiment_ids"] if state else [],
                    evidence=state["evidence"] if state else [],
                )
            )
        node_status = {node.id: node.run_status for node in nodes}
        successful = {
            CapabilityRunStatus.COMPLETE,
            CapabilityRunStatus.ACCEPT,
            CapabilityRunStatus.NO_ACTION,
            CapabilityRunStatus.OPPORTUNITY_FOUND,
        }
        failed = {
            CapabilityRunStatus.REJECT,
            CapabilityRunStatus.INCONCLUSIVE,
            CapabilityRunStatus.UNAVAILABLE,
        }
        edges: list[OptimizationEdgeView] = []
        for edge in topology.edges:
            source_status = node_status[edge.source]
            target_status = node_status[edge.target]
            state = (
                OptimizationEdgeState.ACTIVE
                if target_status == CapabilityRunStatus.RUNNING
                else OptimizationEdgeState.COMPLETE
                if target_status in successful
                else OptimizationEdgeState.BLOCKED
                if source_status in failed
                else OptimizationEdgeState.READY
                if source_status in successful
                else OptimizationEdgeState.INACTIVE
            )
            edges.append(
                OptimizationEdgeView(
                    source=edge.source,
                    target=edge.target,
                    relation=edge.relation,
                    label=edge.label,
                    state=state,
                )
            )
        return OptimizationMapV1(
            source_id=self.id,
            run_id=run_id,
            nodes=nodes,
            edges=edges,
        )

    def preview(self, run_id: str, artifact_path: str) -> ArtifactPreviewV1:
        directory = self._run_dir(run_id)
        portable = PurePosixPath(artifact_path)
        if portable.is_absolute() or ".." in portable.parts or artifact_path in {"", "."}:
            raise FrontendReadError("unsafe artifact path")
        manifest = self._artifact_manifest(directory)
        artifact = manifest.get(str(portable))
        if artifact is None:
            raise FrontendReadError("artifact is not registered")
        view = self._artifact_view(directory, artifact)
        if not view.preview_available:
            raise FrontendReadError("artifact is not previewable")
        lexical = self._reject_symlink_components(directory, portable)
        path = lexical.resolve(strict=True)
        if not path.is_relative_to(directory):
            raise FrontendReadError("artifact escapes run directory")
        data = path.read_bytes()
        if len(data) != artifact.size or hashlib.sha256(data).hexdigest() != artifact.sha256:
            raise FrontendReadError("artifact integrity check failed")
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise FrontendReadError("artifact is not UTF-8 text") from error
        return ArtifactPreviewV1(
            artifact=view.model_copy(update={"integrity": "verified"}),
            content=content,
        )


class ControlPlaneReader:
    def __init__(self, sources: list[ReadOnlyStoreSource]) -> None:
        if not sources:
            raise FrontendReadError("at least one source is required")
        if len({source.id for source in sources}) != len(sources):
            raise FrontendReadError("source ids must be unique")
        self.sources = {source.id: source for source in sources}

    def meta(self) -> ControlPlaneMetaV1:
        return ControlPlaneMetaV1(sources=[source.view for source in self.sources.values()])

    def schema(self) -> dict[str, Any]:
        models = (
            ControlPlaneMetaV1,
            RunListV1,
            RunDetailV1,
            ExperimentViewV1,
            QualitySummaryViewV1,
            ArtifactPreviewV1,
            ApiErrorV1,
            OptimizationMapV1,
        )
        return {
            "schema": "gpuopt.control-plane-schema.v1",
            "api_version": "v1",
            "models": {model.__name__: model.model_json_schema() for model in models},
        }

    def runs(self) -> RunListV1:
        items: list[RunSummaryV1] = []
        warnings: list[str] = []
        for source in self.sources.values():
            for run_id in source.run_ids():
                try:
                    items.append(source.summary(run_id))
                except (FrontendReadError, ValueError) as error:
                    warnings.append(f"{source.id}/{run_id}: {error}")
        items.sort(key=lambda item: (item.updated_at or datetime.min, item.run_id), reverse=True)
        return RunListV1(total=len(items), items=items, warnings=warnings)

    def source(self, source_id: str) -> ReadOnlyStoreSource:
        try:
            return self.sources[source_id]
        except KeyError as error:
            raise FrontendReadError("source not found") from error


__all__ = [
    "ApiErrorV1",
    "ArtifactPreviewV1",
    "ArtifactViewV1",
    "CapabilityViewV1",
    "EvidenceProviderViewV1",
    "ControlPlaneMetaV1",
    "ControlPlaneReader",
    "EventViewV1",
    "ExperimentViewV1",
    "FrontendReadError",
    "MetricViewV1",
    "OptimizationMapV1",
    "QualitySummaryViewV1",
    "ReadOnlyStoreSource",
    "RunDetailV1",
    "RunListV1",
    "RunSummaryV1",
    "project_quality_summary",
]
