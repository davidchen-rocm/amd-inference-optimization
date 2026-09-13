"""Machine-readable and Markdown gfx1201 capability closure report."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import ArtifactRef, StrictModel, utc_now
from .store import ExperimentStore


class CapabilityReportStatus(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    COMPLETE = "COMPLETE"
    MATERIAL_BENEFIT = "MATERIAL_BENEFIT"
    NO_MATERIAL_EFFECT = "NO_MATERIAL_EFFECT"
    HARMFUL = "HARMFUL"
    OPPORTUNITY_FOUND = "OPPORTUNITY_FOUND"
    NO_ACTION = "NO_ACTION"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNAVAILABLE = "UNAVAILABLE"


class CapabilityReportEntry(StrictModel):
    id: str
    title: str
    section: Literal["model", "kernel", "runtime"]
    status: CapabilityReportStatus
    summary: str
    experiment_ids: list[str] = Field(min_length=1)
    evidence: list[ArtifactRef] = Field(min_length=1)
    benchmarks: list[ArtifactRef] = Field(min_length=1)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("id", "title", "summary")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("capability report text fields cannot be empty")
        return value


REQUIRED_CAPABILITIES = {
    "mixed_precision",
    "hip_graph_ab",
    "memory_reuse_audit",
    "mfma_mmq_evidence",
}


class Gfx1201FinalReport(StrictModel):
    schema_name: Literal["gpuopt.gfx1201-final-report.v1"] = Field(
        default="gpuopt.gfx1201-final-report.v1", alias="schema"
    )
    title: Literal["gfx1201-final-report"] = "gfx1201-final-report"
    task_id: str
    campaign_id: str
    model_name: str
    model_sha256: str
    quantization: str
    runtime_commit: str
    gpu: str
    gfx_target: Literal["gfx1201"] = "gfx1201"
    generated_at: datetime = Field(default_factory=utc_now)
    model: list[CapabilityReportEntry]
    kernel: list[CapabilityReportEntry]
    runtime: list[CapabilityReportEntry]
    phase_complete: Literal[True] = True

    @model_validator(mode="after")
    def required_capabilities_are_linked(self) -> Gfx1201FinalReport:
        entries = [*self.model, *self.kernel, *self.runtime]
        ids = [entry.id for entry in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("final report capability ids must be unique")
        missing = REQUIRED_CAPABILITIES - set(ids)
        if missing:
            raise ValueError(
                "final report is missing capabilities: " + ", ".join(sorted(missing))
            )
        if any(entry.section != "model" for entry in self.model):
            raise ValueError("model entries must use section=model")
        if any(entry.section != "kernel" for entry in self.kernel):
            raise ValueError("kernel entries must use section=kernel")
        if any(entry.section != "runtime" for entry in self.runtime):
            raise ValueError("runtime entries must use section=runtime")
        return self


def render_gfx1201_markdown(report: Gfx1201FinalReport) -> str:
    lines = [
        "# gfx1201 Final Capability Report",
        "",
        f"- Model: `{report.model_name}` (`{report.model_sha256}`)",
        f"- Quantization: `{report.quantization}`",
        f"- Runtime commit: `{report.runtime_commit}`",
        f"- GPU: `{report.gpu}` / `{report.gfx_target}`",
        f"- Campaign: `{report.campaign_id}`",
        "",
    ]
    for title, entries in (
        ("Model", report.model),
        ("Kernel", report.kernel),
        ("Runtime", report.runtime),
    ):
        lines.extend((f"## {title}", ""))
        for entry in entries:
            lines.extend(
                (
                    f"### {entry.title}: {entry.status.value}",
                    "",
                    entry.summary,
                    "",
                    "Experiments: " + ", ".join(f"`{item}`" for item in entry.experiment_ids),
                    "",
                    "Evidence:",
                    "",
                )
            )
            lines.extend(
                f"- `{artifact.path}` — `{artifact.sha256}`" for artifact in entry.evidence
            )
            lines.extend(("", "Benchmarks:", ""))
            lines.extend(
                f"- `{artifact.path}` — `{artifact.sha256}`" for artifact in entry.benchmarks
            )
            if entry.limitations:
                lines.extend(("", "Limitations:", ""))
                lines.extend(f"- {limitation}" for limitation in entry.limitations)
            lines.append("")
    lines.extend(("## Phase Status", "", "gfx1201 feature expansion is complete.", ""))
    return "\n".join(lines)


def persist_gfx1201_final_report(
    report: Gfx1201FinalReport,
    store: ExperimentStore,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Persist immutable machine-readable and human-readable final reports."""

    linked = {
        artifact.path: artifact
        for entry in [*report.model, *report.kernel, *report.runtime]
        for artifact in [*entry.evidence, *entry.benchmarks, *entry.artifacts]
    }
    for path, artifact in linked.items():
        registered = store.artifact_ref(report.task_id, path)
        if registered is None or registered.sha256 != artifact.sha256:
            raise ValueError(f"final report references unregistered artifact: {path}")
        if not store.verify_artifact(report.task_id, registered):
            raise ValueError(f"final report artifact integrity failed: {path}")
    encoded = (
        json.dumps(
            report.model_dump(mode="json", by_alias=True),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    json_ref = store.save_immutable_bytes(
        report.task_id,
        "reports/gfx1201-final-report.json",
        encoded,
        producer="gfx1201-final-report",
        media_type="application/json",
    )
    markdown_ref = store.save_immutable_bytes(
        report.task_id,
        "reports/gfx1201-final-report.md",
        render_gfx1201_markdown(report).encode(),
        producer="gfx1201-final-report",
        media_type="text/markdown",
    )
    return json_ref, markdown_ref


__all__ = [
    "CapabilityReportEntry",
    "CapabilityReportStatus",
    "Gfx1201FinalReport",
    "REQUIRED_CAPABILITIES",
    "persist_gfx1201_final_report",
    "render_gfx1201_markdown",
]
