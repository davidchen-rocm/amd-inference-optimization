"""Stable, read-only campaign summary inspired by HyperLoom session breakdowns."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .architecture import ArchitectureFamily, ArchitectureProfile, architecture_profile
from .campaign_models import CampaignRecord
from .control_policy import ActionSpec, OptimizationLedger, action_catalogue
from .evidence_catalog import EVIDENCE_PROVIDERS, EvidenceProviderSpec
from .models import ArtifactRef, StrictModel, utc_now
from .platform_probe import LocalPlatformEvidence
from .store import ExperimentStore, StoreError

SESSION_BREAKDOWN_PATH = "reports/session-breakdown.json"
SESSION_BREAKDOWN_MARKDOWN_PATH = "reports/session-breakdown.md"


class SessionIdentity(StrictModel):
    campaign_id: str
    task_id: str
    created_at_utc: datetime
    updated_at_utc: datetime
    status: str
    stage: str
    revision: int = Field(ge=0)


class WorkloadBreakdown(StrictModel):
    model_name: str
    model_sha256: str | None = None
    quantization: str | None = None
    runtime: str
    runtime_commit: str
    workload_kind: str
    objective: str
    gfx_target: str
    device_id: int = Field(ge=0)


class CandidateBreakdown(StrictModel):
    id: str
    strategy: str
    disposition: str
    fingerprint: str | None = None
    selected: bool
    reasons: list[str] = Field(default_factory=list)
    evidence: list[ArtifactRef] = Field(default_factory=list)


class ControlPlaneBreakdown(StrictModel):
    action_catalogue: list[ActionSpec]
    max_validation_attempts_per_fingerprint: int
    profile_gain_watermark_percent: float


class InformationCollectionBreakdown(StrictModel):
    """Provider boundary and immutable evidence currently visible to the run."""

    provider_catalogue: list[EvidenceProviderSpec] = Field(default_factory=list)
    capability_snapshot: ArtifactRef | None = None
    runtime_health: ArtifactRef | None = None
    kernel_shape_manifests: list[ArtifactRef] = Field(default_factory=list)
    external_evidence: list[ArtifactRef] = Field(default_factory=list)


class SessionBreakdownV1(StrictModel):
    schema_name: Literal["gpuopt.session-breakdown.v1"] = Field(
        default="gpuopt.session-breakdown.v1", alias="schema"
    )
    exported_at_utc: datetime = Field(default_factory=utc_now)
    session: SessionIdentity
    workload: WorkloadBreakdown
    architecture: ArchitectureProfile
    platform: LocalPlatformEvidence | None = None
    control_plane: ControlPlaneBreakdown
    information_collection: InformationCollectionBreakdown = Field(
        default_factory=InformationCollectionBreakdown
    )
    baselines: list[ArtifactRef] = Field(default_factory=list)
    candidates: list[CandidateBreakdown] = Field(default_factory=list)
    optimization_ledger: OptimizationLedger
    final: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    source_files: dict[str, str | list[str]] = Field(default_factory=dict)


def build_session_breakdown(
    record: CampaignRecord,
    *,
    platform: LocalPlatformEvidence | None = None,
) -> SessionBreakdownV1:
    task = record.config.task
    architecture = architecture_profile(
        task.gpu.gfx_target,
        board_type=task.gpu.board_type,
    )
    warnings = [
        "quality policy provisional-math-100.v1 is experimental, not production qualification"
    ]
    if architecture.family == ArchitectureFamily.UNKNOWN:
        warnings.append(
            f"architecture {architecture.gfx_target} has no active local optimization profile"
        )
    if platform is None:
        warnings.append("local platform evidence is unavailable")
    elif platform.status != "ok":
        warnings.extend(platform.warnings)

    candidates = [
        CandidateBreakdown(
            id=candidate.id,
            strategy=candidate.strategy.value,
            disposition=candidate.disposition.value,
            fingerprint=candidate.content_fingerprint,
            selected=candidate.selected_as_winner,
            reasons=candidate.reasons,
            evidence=candidate.result_artifacts,
        )
        for candidate in record.candidates
    ]
    final = {
        "status": record.status.value,
        "selected_candidate_ids": record.selected_candidate_ids,
        "compatibility": (
            record.compatibility.model_dump(mode="json")
            if record.compatibility is not None
            else None
        ),
        "combination": (
            record.final_combination.model_dump(mode="json")
            if record.final_combination is not None
            else None
        ),
    }
    return SessionBreakdownV1(
        session=SessionIdentity(
            campaign_id=record.campaign_id,
            task_id=record.task_id,
            created_at_utc=record.config.task.created_at,
            updated_at_utc=record.updated_at,
            status=record.status.value,
            stage=record.current_stage.value,
            revision=record.revision,
        ),
        workload=WorkloadBreakdown(
            model_name=Path(task.model.path).name,
            model_sha256=task.model.sha256,
            quantization=task.model.quantization,
            runtime=task.runtime.name,
            runtime_commit=task.runtime.base_commit,
            workload_kind=task.workload.kind,
            objective=task.objective.primary_metric,
            gfx_target=task.gpu.gfx_target,
            device_id=task.gpu.device_id,
        ),
        architecture=architecture,
        platform=platform,
        control_plane=ControlPlaneBreakdown(
            action_catalogue=list(action_catalogue()),
            max_validation_attempts_per_fingerprint=(
                record.config.control_policy.max_validation_attempts_per_fingerprint
            ),
            profile_gain_watermark_percent=(
                record.config.control_policy.profile_refresh.gain_watermark_percent
            ),
        ),
        information_collection=InformationCollectionBreakdown(
            provider_catalogue=list(EVIDENCE_PROVIDERS)
        ),
        baselines=[item.artifact for item in record.shared_baselines],
        candidates=candidates,
        optimization_ledger=record.optimization_ledger,
        final=final,
        warnings=sorted(set(warnings)),
        source_files={
            "task": "task.json",
            "campaign": "state/campaign.json",
            "artifact_manifest": "artifacts/manifest.json",
            "events": "events/events.jsonl",
            "platform": (
                record.platform_evidence.path
                if record.platform_evidence is not None
                else ""
            ),
        },
    )


def render_session_breakdown_markdown(report: SessionBreakdownV1) -> str:
    lines = [
        "# Optimization Session Breakdown",
        "",
        f"- Campaign: `{report.session.campaign_id}`",
        f"- Status: `{report.session.status}`",
        f"- Stage: `{report.session.stage}`",
        f"- Model: `{report.workload.model_name}`",
        f"- Runtime: `{report.workload.runtime}`",
        f"- GPU target: `{report.workload.gfx_target}`",
        "",
        "## Candidate Attempts",
        "",
        "| Candidate | Strategy | Disposition | Selected |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| `{item.id}` | {item.strategy} | {item.disposition} | "
        f"{'yes' if item.selected else 'no'} |"
        for item in report.candidates
    )
    if not report.candidates:
        lines.append("| — | — | — | — |")
    lines.extend(("", "## Accepted Optimization Stack", ""))
    if report.optimization_ledger.accepted_stack:
        lines.extend(
            f"- `{item.action_name}`: `{item.candidate_id}` (`{item.fingerprint[:12]}`)"
            for item in report.optimization_ledger.accepted_stack
        )
    else:
        lines.append("No compatible accepted stack has been selected.")
    if report.warnings:
        lines.extend(("", "## Warnings", ""))
        lines.extend(f"- {warning}" for warning in report.warnings)
    return "\n".join(lines) + "\n"


def persist_session_breakdown(
    store: ExperimentStore,
    record: CampaignRecord,
    *,
    platform: LocalPlatformEvidence | None = None,
) -> tuple[ArtifactRef, ArtifactRef]:
    report = build_session_breakdown(record, platform=platform)
    try:
        manifest = store.load_json(record.task_id, "artifacts/manifest.json")
    except StoreError:  # the report remains valid for an empty/new artifact store
        manifest = {}
    raw_artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    artifacts: list[ArtifactRef] = []
    if isinstance(raw_artifacts, dict):
        for raw in raw_artifacts.values():
            try:
                artifacts.append(ArtifactRef.model_validate(raw))
            except (TypeError, ValueError):
                continue

    def latest(prefix: str) -> ArtifactRef | None:
        matches = sorted(
            (item for item in artifacts if item.path.startswith(prefix)),
            key=lambda item: item.path,
        )
        return matches[-1] if matches else None

    report.information_collection = InformationCollectionBreakdown(
        provider_catalogue=list(EVIDENCE_PROVIDERS),
        capability_snapshot=latest("artifacts/evidence/campaign/evidence-capabilities/"),
        runtime_health=latest("artifacts/evidence/campaign/runtime-health/"),
        kernel_shape_manifests=[
            item for item in artifacts if "kernel-shape-manifest" in item.path
        ],
        external_evidence=[
            item
            for item in artifacts
            if any(
                marker in item.path
                for marker in (
                    "magpie-evidence",
                    "tracelens-evidence",
                    "intellikit-evidence",
                )
            )
        ],
    )
    json_ref = store.save_json(
        record.task_id,
        SESSION_BREAKDOWN_PATH,
        report,
        producer="session-breakdown",
    )
    markdown_ref = store.save_text(
        record.task_id,
        SESSION_BREAKDOWN_MARKDOWN_PATH,
        render_session_breakdown_markdown(report),
        producer="session-breakdown",
        media_type="text/markdown",
    )
    return json_ref, markdown_ref


__all__ = [
    "SESSION_BREAKDOWN_MARKDOWN_PATH",
    "SESSION_BREAKDOWN_PATH",
    "SessionBreakdownV1",
    "build_session_breakdown",
    "persist_session_breakdown",
    "render_session_breakdown_markdown",
]
