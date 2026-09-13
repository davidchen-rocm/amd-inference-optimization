"""Evidence-oriented action catalogue and optimization attempt ledger.

This adopts HyperLoom's useful control-plane idea—actions declare lanes, risk,
side effects, and evidence—without importing its agent runtime or replacing the
existing gpuopt state machine.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .models import ArtifactRef, StrictModel, utc_now


class ActionFamily(StrEnum):
    PREPARE = "prepare"
    MODEL = "model"
    KERNEL = "kernel"
    RUNTIME = "runtime"
    EVIDENCE = "evidence"
    VALIDATION = "validation"
    REPORT = "report"


class ActionSpec(StrictModel):
    name: str
    family: ActionFamily
    description: str
    expected_gain_percent: tuple[float, float]
    accuracy_risk: float = Field(ge=0, le=1)
    crash_risk: float = Field(ge=0, le=1)
    typical_runtime_minutes: float = Field(gt=0)
    lease_ttl_seconds: int = Field(ge=1)
    required_lanes: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    coordinator_owned: bool = False

    @field_validator(
        "name",
        "description",
    )
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("action text cannot be empty")
        return value

    @field_validator("required_lanes", "side_effects", "required_evidence")
    @classmethod
    def unique_values(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values) or len(values) != len(set(values)):
            raise ValueError("action lists must contain unique non-empty values")
        return values

    @model_validator(mode="after")
    def gain_range_is_ordered(self) -> ActionSpec:
        if self.expected_gain_percent[0] > self.expected_gain_percent[1]:
            raise ValueError("expected gain range must be ordered")
        return self


_ACTION_CATALOGUE = {
    item.name: item
    for item in (
        ActionSpec(
            name="baseline",
            family=ActionFamily.PREPARE,
            description="Capture an immutable same-coordinate performance and quality baseline.",
            expected_gain_percent=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.05,
            typical_runtime_minutes=8,
            lease_ttl_seconds=3600,
            required_lanes=["gpu", "benchmark"],
            side_effects=["executes_workload", "writes_evidence"],
            required_evidence=["run_identity", "benchmark", "quality"],
            coordinator_owned=True,
        ),
        ActionSpec(
            name="profile",
            family=ActionFamily.EVIDENCE,
            description="Collect the shallowest profiling evidence needed for a decision.",
            expected_gain_percent=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.08,
            typical_runtime_minutes=5,
            lease_ttl_seconds=3600,
            required_lanes=["gpu", "profiler"],
            side_effects=["executes_workload", "writes_evidence"],
            required_evidence=["capability", "kernel_timing", "coverage"],
            coordinator_owned=True,
        ),
        ActionSpec(
            name="mixed_bit",
            family=ActionFamily.MODEL,
            description="Evaluate a hash-bound mixed-precision tensor assignment.",
            expected_gain_percent=(2.0, 25.0),
            accuracy_risk=0.35,
            crash_risk=0.05,
            typical_runtime_minutes=30,
            lease_ttl_seconds=14400,
            required_lanes=["model_build", "gpu", "benchmark"],
            side_effects=["writes_model", "executes_workload"],
            required_evidence=["tensor_assignment", "effective_bpw", "performance", "quality"],
        ),
        ActionSpec(
            name="shape_kernel",
            family=ActionFamily.KERNEL,
            description="Test one exact-shape source patch with launch and resource evidence.",
            expected_gain_percent=(0.0, 20.0),
            accuracy_risk=0.1,
            crash_risk=0.2,
            typical_runtime_minutes=25,
            lease_ttl_seconds=10800,
            required_lanes=["workspace_mutation", "build", "gpu", "benchmark"],
            side_effects=["patches_source", "builds_runtime", "executes_workload"],
            required_evidence=["patch", "correctness", "performance", "kernel_mapping"],
        ),
        ActionSpec(
            name="kv_cache",
            family=ActionFamily.RUNTIME,
            description="Evaluate a typed KV-cache precision arm at fixed context depths.",
            expected_gain_percent=(-5.0, 15.0),
            accuracy_risk=0.2,
            crash_risk=0.08,
            typical_runtime_minutes=20,
            lease_ttl_seconds=7200,
            required_lanes=["gpu", "benchmark"],
            side_effects=["executes_workload"],
            required_evidence=["cache_identity", "capacity", "performance", "quality"],
        ),
        ActionSpec(
            name="hip_graph_ab",
            family=ActionFamily.RUNTIME,
            description="Run paired HIP Graph OFF/ON measurements at identical coordinates.",
            expected_gain_percent=(-3.0, 8.0),
            accuracy_risk=0.0,
            crash_risk=0.05,
            typical_runtime_minutes=15,
            lease_ttl_seconds=5400,
            required_lanes=["gpu", "benchmark", "profiler"],
            side_effects=["executes_workload"],
            required_evidence=["paired_performance", "runtime_trace"],
        ),
        ActionSpec(
            name="memory_audit",
            family=ActionFamily.EVIDENCE,
            description=(
                "Classify observable buffer, allocation, and memory-copy reuse opportunities."
            ),
            expected_gain_percent=(0.0, 10.0),
            accuracy_risk=0.0,
            crash_risk=0.05,
            typical_runtime_minutes=15,
            lease_ttl_seconds=5400,
            required_lanes=["gpu", "profiler"],
            side_effects=["executes_workload", "writes_evidence"],
            required_evidence=["runtime_trace", "memory_operations", "coverage"],
            coordinator_owned=True,
        ),
        ActionSpec(
            name="mfma_mmq_evidence",
            family=ActionFamily.EVIDENCE,
            description=(
                "Close selected-kernel, ISA, resource, and bottleneck evidence for hot shapes."
            ),
            expected_gain_percent=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.05,
            typical_runtime_minutes=20,
            lease_ttl_seconds=7200,
            required_lanes=["gpu", "profiler"],
            side_effects=["executes_workload", "writes_evidence"],
            required_evidence=["kernel_mapping", "isa", "resources", "classification"],
            coordinator_owned=True,
        ),
        ActionSpec(
            name="final_validation",
            family=ActionFamily.VALIDATION,
            description="Re-run the compatible accepted stack against the frozen baseline.",
            expected_gain_percent=(0.0, 100.0),
            accuracy_risk=0.0,
            crash_risk=0.05,
            typical_runtime_minutes=30,
            lease_ttl_seconds=14400,
            required_lanes=["gpu", "benchmark"],
            side_effects=["executes_workload", "writes_evidence"],
            required_evidence=["stack_identity", "performance", "quality", "gate"],
            coordinator_owned=True,
        ),
        ActionSpec(
            name="report",
            family=ActionFamily.REPORT,
            description="Project immutable campaign evidence into a stable session breakdown.",
            expected_gain_percent=(0.0, 0.0),
            accuracy_risk=0.0,
            crash_risk=0.0,
            typical_runtime_minutes=1,
            lease_ttl_seconds=60,
            side_effects=["writes_report"],
            required_evidence=["campaign_state", "artifact_manifest"],
            coordinator_owned=True,
        ),
    )
}
ACTION_CATALOGUE: Mapping[str, ActionSpec] = MappingProxyType(_ACTION_CATALOGUE)


def action_catalogue() -> tuple[ActionSpec, ...]:
    return tuple(ACTION_CATALOGUE[name] for name in sorted(ACTION_CATALOGUE))


def content_fingerprint(action_name: str, configuration: Mapping[str, Any]) -> str:
    """Hash semantic action content so labels cannot bypass experiment dedup."""

    if action_name not in ACTION_CATALOGUE:
        raise ValueError(f"unknown optimization action: {action_name}")
    try:
        encoded = json.dumps(
            {"action": action_name, "configuration": configuration},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("candidate configuration must be canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


class AttemptDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class OptimizationAttempt(StrictModel):
    candidate_id: str
    action_name: str
    fingerprint: str
    disposition: AttemptDisposition
    validation_attempt: int = Field(default=1, ge=1, le=3)
    evidence: list[ArtifactRef] = Field(default_factory=list)
    recorded_at: datetime = Field(default_factory=utc_now)

    @field_validator("fingerprint")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("fingerprint must be a lowercase SHA-256 digest")
        return value

    @field_validator("action_name")
    @classmethod
    def known_action(cls, value: str) -> str:
        if value not in ACTION_CATALOGUE:
            raise ValueError(f"unknown optimization action: {value}")
        return value


class OptimizationStackEntry(StrictModel):
    candidate_id: str
    action_name: str
    fingerprint: str
    evidence: list[ArtifactRef] = Field(default_factory=list)


class OptimizationLedger(StrictModel):
    attempts: list[OptimizationAttempt] = Field(default_factory=list)
    accepted_stack: list[OptimizationStackEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_content(self) -> OptimizationLedger:
        fingerprints = [attempt.fingerprint for attempt in self.attempts]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("optimization ledger cannot repeat candidate content")
        accepted = [entry.fingerprint for entry in self.accepted_stack]
        if len(accepted) != len(set(accepted)):
            raise ValueError("optimization stack cannot repeat candidate content")
        known = {
            attempt.fingerprint
            for attempt in self.attempts
            if attempt.disposition == AttemptDisposition.ACCEPTED
        }
        if any(fingerprint not in known for fingerprint in accepted):
            raise ValueError("optimization stack entries require an accepted attempt")
        return self


def record_optimization_attempt(
    ledger: OptimizationLedger,
    *,
    candidate_id: str,
    action_name: str,
    fingerprint: str,
    disposition: AttemptDisposition,
    evidence: Sequence[ArtifactRef] = (),
) -> OptimizationLedger:
    if any(attempt.fingerprint == fingerprint for attempt in ledger.attempts):
        raise ValueError("candidate content has already been evaluated")
    attempt = OptimizationAttempt(
        candidate_id=candidate_id,
        action_name=action_name,
        fingerprint=fingerprint,
        disposition=disposition,
        evidence=list(evidence),
    )
    return ledger.model_copy(update={"attempts": [*ledger.attempts, attempt]})


def select_optimization_stack(
    ledger: OptimizationLedger,
    candidate_ids: Sequence[str],
) -> OptimizationLedger:
    selected = []
    for candidate_id in candidate_ids:
        attempt = next(
            (
                item
                for item in ledger.attempts
                if item.candidate_id == candidate_id
                and item.disposition == AttemptDisposition.ACCEPTED
            ),
            None,
        )
        if attempt is None:
            raise ValueError(f"selected candidate is not accepted: {candidate_id}")
        selected.append(
            OptimizationStackEntry(
                candidate_id=attempt.candidate_id,
                action_name=attempt.action_name,
                fingerprint=attempt.fingerprint,
                evidence=attempt.evidence,
            )
        )
    return ledger.model_copy(update={"accepted_stack": selected})


class ProfileRefreshPolicy(StrictModel):
    gain_watermark_percent: float = Field(default=10.0, gt=0)

    def should_refresh(
        self,
        *,
        baseline_profiled: bool,
        cumulative_gain_percent: float,
        last_profile_gain_percent: float | None,
    ) -> tuple[bool, str]:
        for value in (cumulative_gain_percent, last_profile_gain_percent):
            if value is not None and not math.isfinite(value):
                raise ValueError("profile gain coordinates must be finite")
        if not baseline_profiled:
            return True, "baseline profile is required"
        if last_profile_gain_percent is None:
            return True, "last profile watermark is unknown"
        delta = cumulative_gain_percent - last_profile_gain_percent
        if delta >= self.gain_watermark_percent:
            return True, f"validated gain advanced by {delta:.4f}%"
        return False, f"validated gain advanced by only {delta:.4f}%"


class CampaignControlPolicy(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )
    schema_name: Literal["gpuopt.campaign-control-policy.v1"] = Field(
        default="gpuopt.campaign-control-policy.v1", alias="schema"
    )
    max_validation_attempts_per_fingerprint: Literal[3] = 3
    profile_refresh: ProfileRefreshPolicy = Field(default_factory=ProfileRefreshPolicy)


__all__ = [
    "ACTION_CATALOGUE",
    "ActionFamily",
    "ActionSpec",
    "AttemptDisposition",
    "CampaignControlPolicy",
    "OptimizationAttempt",
    "OptimizationLedger",
    "OptimizationStackEntry",
    "ProfileRefreshPolicy",
    "action_catalogue",
    "content_fingerprint",
    "record_optimization_attempt",
    "select_optimization_stack",
]
