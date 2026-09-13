"""Canonical, rebuildable views over one experiment's scattered evidence.

The bundle is a projection, not a second source of truth.  Existing hash-bound
artifacts remain authoritative; ``summary.json`` is the human-facing entry point and
``manifest.json`` is published last as the bundle commit marker.  ``catalog.sqlite3``
is a disposable query cache rebuilt from those JSON bundles.
"""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import sqlite3
import statistics
from collections.abc import Iterable
from contextlib import nullcontext
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, field_validator

from .models import ArtifactRef, OptimizationTask, StrictModel
from .store import ExperimentStore, StoreError

BUNDLE_SUMMARY_SCHEMA = "gpuopt.experiment-summary.v1"
BUNDLE_MANIFEST_SCHEMA = "gpuopt.experiment-manifest.v1"
BUNDLE_ATTEMPTS_SCHEMA = "gpuopt.experiment-attempt-index.v1"
MODEL_INPUT_PROVENANCE_SCHEMA = "gpuopt.model-input-provenance.v1"
CATALOG_SCHEMA_VERSION = 1
CATALOG_FILENAME = "catalog.sqlite3"

_DERIVED_BUNDLE_FILES = {"summary.json", "manifest.json", "attempts/index.json"}
_COLLECT_ROOTS = ("runner", "quality", "e2e-reruns", "extended-pair-verification")
_MAX_COLLECTED_FILES = 2048
_MAX_COLLECTED_FILE_BYTES = 512 * 1024 * 1024
_MAX_COLLECTED_TOTAL_BYTES = 512 * 1024 * 1024


class BundleError(RuntimeError):
    """The requested bundle is unsafe, incomplete, or internally inconsistent."""


class BundleArtifactRole(StrEnum):
    SPEC = "spec"
    CHANGE = "change"
    BUILD = "build"
    BENCHMARK = "benchmark"
    QUALITY = "quality"
    PROFILE = "profile"
    DECISION = "decision"
    ENVIRONMENT = "environment"
    LOG = "log"
    ATTEMPT = "attempt"
    BASELINE = "baseline"
    SUMMARY = "summary"
    OTHER = "other"


class BundleArtifactEntry(StrictModel):
    artifact: ArtifactRef
    role: BundleArtifactRole
    stage: str | None = None
    attempt_id: str | None = None
    ownership: Literal["experiment", "shared", "external"] = "experiment"
    storage: Literal["task-local", "external-case-store"] = "task-local"
    current: bool = True


class BundleMetricSummary(StrictModel):
    name: str
    unit: str
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    delta_percent: float | None = None
    baseline_sample_count: int | None = Field(default=None, ge=0)
    candidate_sample_count: int | None = Field(default=None, ge=0)
    baseline_cv_percent: float | None = Field(default=None, ge=0)
    candidate_cv_percent: float | None = Field(default=None, ge=0)


class BundleQualitySummary(StrictModel):
    status: str
    correctness_passed: bool | None = None
    baseline_perplexity: float | None = None
    candidate_perplexity: float | None = None
    perplexity_regression_percent: float | None = None
    accuracy_metric: str | None = None
    baseline_accuracy: float | None = None
    candidate_accuracy: float | None = None
    accuracy_drop_percentage_points: float | None = None
    baseline_accuracies: dict[str, float] = Field(default_factory=dict)
    candidate_accuracies: dict[str, float] = Field(default_factory=dict)


class BundleDecisionSummary(StrictModel):
    outcome: str
    final: bool
    reasons: list[str] = Field(default_factory=list)
    rerun_from_stage: str | None = None


class BundleModelSummary(StrictModel):
    name: str
    path: str
    sha256: str | None = None
    architecture: str | None = None
    quantization: str | None = None
    packed_bytes: int | None = Field(default=None, ge=0)
    effective_bpw: float | None = Field(default=None, gt=0)
    provenance_artifact: ArtifactRef | None = None


class ModelInputProvenanceV1(StrictModel):
    """Hash-bound coordinates and size accounting for one model input."""

    schema_name: Literal["gpuopt.model-input-provenance.v1"] = Field(
        default=MODEL_INPUT_PROVENANCE_SCHEMA, alias="schema"
    )
    model_path: str = Field(min_length=1)
    model_sha256: str
    packed_bytes: int = Field(gt=0)
    effective_bpw: float | None = Field(default=None, gt=0)
    quantization: str | None = None
    architecture: str | None = None
    source_model_path: str | None = None
    source_model_sha256: str | None = None

    @field_validator("model_path", "source_model_path")
    @classmethod
    def valid_path(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or "\x00" in value):
            raise ValueError("model provenance paths must be non-empty and NUL-free")
        return value

    @field_validator("model_sha256", "source_model_sha256")
    @classmethod
    def valid_sha256(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("model provenance SHA-256 must be a lowercase digest")
        return value

    @field_validator("effective_bpw")
    @classmethod
    def finite_effective_bpw(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("effective_bpw must be finite")
        return value


class BundleAttemptSummary(StrictModel):
    kind: Literal["runner", "profile", "e2e-rerun", "quality", "extended-pair"]
    attempt_id: str
    status: str
    selected: bool = False
    evidence: list[ArtifactRef] = Field(default_factory=list)


class ExperimentAttemptIndex(StrictModel):
    schema_name: Literal["gpuopt.experiment-attempt-index.v1"] = Field(
        default=BUNDLE_ATTEMPTS_SCHEMA, alias="schema"
    )
    task_id: str
    experiment_id: str
    attempts: list[BundleAttemptSummary] = Field(default_factory=list)
    selected: dict[str, str] = Field(default_factory=dict)


class ExperimentBundleSummary(StrictModel):
    schema_name: Literal["gpuopt.experiment-summary.v1"] = Field(
        default=BUNDLE_SUMMARY_SCHEMA, alias="schema"
    )
    task_id: str
    experiment_id: str
    status: str
    model: BundleModelSummary
    runtime: str
    runtime_commit: str
    gpu: str | None = None
    gfx: str
    strategy: str | None = None
    hypothesis_id: str | None = None
    change_summary: str | None = None
    metrics: list[BundleMetricSummary] = Field(default_factory=list)
    quality: BundleQualitySummary | None = None
    decision: BundleDecisionSummary | None = None
    selected_attempts: dict[str, str] = Field(default_factory=dict)
    artifact_count: int = Field(ge=0)
    source_digest: str
    updated_at: datetime | None = None

    @field_validator("source_digest")
    @classmethod
    def valid_source_digest(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("source_digest must be a lowercase SHA-256 digest")
        return value


class ExperimentBundleManifest(StrictModel):
    schema_name: Literal["gpuopt.experiment-manifest.v1"] = Field(
        default=BUNDLE_MANIFEST_SCHEMA, alias="schema"
    )
    task_id: str
    experiment_id: str
    source_digest: str
    summary: ArtifactRef
    attempts_index: ArtifactRef
    artifacts: list[BundleArtifactEntry]
    published_at: datetime


def _safe_experiment_id(experiment_id: str) -> None:
    portable = PurePosixPath(experiment_id)
    if (
        not experiment_id
        or "\x00" in experiment_id
        or portable.is_absolute()
        or len(portable.parts) != 1
        or experiment_id in {".", ".."}
    ):
        raise BundleError("unsafe experiment id")


def _optional_json(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _media_type(path: Path) -> str:
    detected = mimetypes.guess_type(path.name)[0]
    if detected is not None:
        return detected
    if path.suffix in {".stdout", ".stderr", ".log"}:
        return "text/plain"
    return "application/octet-stream"


def _collect_unregistered_outputs(
    store: ExperimentStore,
    task_id: str,
    experiment_id: str,
) -> None:
    """Register bounded runner outputs in place, without copying worktrees or builds."""

    task_directory = store.task_dir(task_id)
    experiment_directory = task_directory / "experiments" / experiment_id
    known = _task_manifest(store, task_id)
    count = 0
    total_bytes = 0
    for root_name in _COLLECT_ROOTS:
        root = experiment_directory / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if count >= _MAX_COLLECTED_FILES:
                raise BundleError("experiment output collector exceeded its file limit")
            if path.is_symlink() or not path.is_file():
                continue
            relative_parts = path.relative_to(experiment_directory).parts[:-1]
            if any(
                part.endswith("-worktree")
                or part in {"build", "build-gpuopt"}
                or part.startswith("build-")
                for part in relative_parts
            ):
                continue
            size = path.stat().st_size
            if size > _MAX_COLLECTED_FILE_BYTES:
                continue
            total_bytes += size
            if total_bytes > _MAX_COLLECTED_TOTAL_BYTES:
                raise BundleError("experiment output collector exceeded its byte limit")
            relative = str(path.relative_to(task_directory).as_posix())
            if relative not in known:
                known[relative] = store.register_existing_artifact(
                    task_id,
                    relative,
                    producer="experiment-artifact-collector",
                    media_type=_media_type(path),
                )
            elif not store.verify_artifact(task_id, known[relative]):
                raise BundleError(f"registered experiment output changed: {relative}")
            count += 1


def _collect_declared_patch(
    store: ExperimentStore,
    task_id: str,
    experiment_id: str,
) -> None:
    """Freeze an older source-patch input that pre-dates immutable patch import."""

    spec_path = f"experiments/{experiment_id}/spec.json"
    spec = _verified_json(store, task_id, spec_path)
    change = spec.get("change")
    if not isinstance(change, dict) or str(change.get("kind")) != "source_patch":
        return
    declared = change.get("patch_sha256")
    raw_source = change.get("patch_path")
    if declared is not None and (not isinstance(declared, str) or len(declared) != 64):
        raise BundleError("source experiment has no valid patch SHA-256")
    destination = f"experiments/{experiment_id}/change/patch.diff"
    existing = store.artifact_ref(task_id, destination)
    if existing is not None:
        if (
            (declared is not None and existing.sha256 != declared)
            or not store.verify_artifact(task_id, existing)
        ):
            raise BundleError("frozen experiment patch failed integrity verification")
        return
    if not isinstance(raw_source, str):
        raise BundleError("source experiment patch bytes are unavailable")
    source = Path(raw_source).expanduser()
    if source.is_symlink() or not source.is_file():
        raise BundleError("source experiment patch bytes are unavailable")
    if source.stat().st_size > _MAX_COLLECTED_FILE_BYTES:
        raise BundleError("source experiment patch exceeds the artifact size limit")
    patch_bytes = source.read_bytes()
    digest = hashlib.sha256(patch_bytes).hexdigest()
    if declared is not None and digest != declared:
        raise BundleError("source experiment patch differs from its declared SHA-256")
    store.save_immutable_bytes(
        task_id,
        destination,
        patch_bytes,
        producer="experiment-input-freezer",
        media_type="text/x-diff",
    )


def _task_manifest(store: ExperimentStore, task_id: str) -> dict[str, ArtifactRef]:
    try:
        raw = store.load_json(task_id, "artifacts/manifest.json")
    except StoreError:
        return {}
    values = raw.get("artifacts", {}) if isinstance(raw, dict) else {}
    result: dict[str, ArtifactRef] = {}
    if isinstance(values, dict):
        for path, value in values.items():
            try:
                result[str(path)] = ArtifactRef.model_validate(value)
            except (TypeError, ValueError):
                continue
    return result


def _verified_ref(
    store: ExperimentStore,
    task_id: str,
    path: str,
    *,
    expected_sha256: str | None = None,
) -> ArtifactRef:
    reference = store.artifact_ref(task_id, path)
    if reference is None:
        raise BundleError(f"bundle source is not registered: {path}")
    if expected_sha256 is not None and reference.sha256 != expected_sha256:
        raise BundleError(f"bundle source pointer hash differs from manifest: {path}")
    if not store.verify_artifact(task_id, reference):
        raise BundleError(f"bundle source failed integrity verification: {path}")
    return reference


def _verified_json(
    store: ExperimentStore,
    task_id: str,
    path: str,
) -> dict[str, Any]:
    _verified_ref(store, task_id, path)
    value = store.load_json(task_id, path)
    if not isinstance(value, dict):
        raise BundleError(f"bundle source is not a JSON object: {path}")
    return value


def _pointer_path(
    store: ExperimentStore,
    task_id: str,
    pointer_path: str,
) -> str | None:
    if store.artifact_ref(task_id, pointer_path) is None:
        return None
    pointer = _verified_json(store, task_id, pointer_path)
    path = pointer.get("path")
    digest = pointer.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        return None
    _verified_ref(store, task_id, path, expected_sha256=digest)
    return path


def _latest_terminal_attempt_path(root: Path, filename: str) -> str | None:
    if root.is_symlink() or not root.is_dir():
        return None
    candidates = sorted(
        directory
        for directory in root.iterdir()
        if directory.is_dir()
        and not directory.is_symlink()
        and (directory / filename).is_file()
        and not (directory / filename).is_symlink()
    )
    if not candidates:
        return None
    return str((candidates[-1] / filename).as_posix())


def _selected_evidence_paths(
    store: ExperimentStore,
    task: OptimizationTask,
    experiment_id: str,
) -> dict[str, str]:
    """Resolve the same A/B coordinates used by the live Gate."""

    task_directory = store.task_dir(task.id)
    experiment_prefix = f"experiments/{experiment_id}/"
    selection = {
        "spec": f"{experiment_prefix}spec.json",
        "baseline": "artifacts/baseline.json",
        "candidate_e2e": f"{experiment_prefix}e2e-result.json",
        "candidate_result": f"{experiment_prefix}result.json",
        "pre_gate": f"{experiment_prefix}performance-pre-gate.json",
        "quality": f"{experiment_prefix}quality-result.json",
    }
    active_path = "state/active-execution.json"
    if store.artifact_ref(task.id, active_path) is not None:
        active = _verified_json(store, task.id, active_path)
    else:
        active = None
    if isinstance(active, dict) and active.get("experiment_id") == experiment_id:
        selection["active_pointer"] = active_path
        for key, selected_key in (
            ("e2e_path", "candidate_e2e"),
            ("result_path", "candidate_result"),
            ("pre_gate_path", "pre_gate"),
        ):
            value = active.get(key)
            if isinstance(value, str) and value.startswith(experiment_prefix):
                selection[selected_key] = value
        baseline_path = _pointer_path(
            store, task.id, "state/performance-baseline-current.json"
        )
        if baseline_path is not None and (
            baseline_path == "artifacts/baseline.json"
            or baseline_path.startswith(experiment_prefix)
        ):
            selection["baseline"] = baseline_path
            selection["performance_baseline_pointer"] = (
                "state/performance-baseline-current.json"
            )
    else:
        experiment_directory = task_directory / "experiments" / experiment_id
        pair_root = experiment_directory / "extended-pair-verification"
        pair_e2e = _latest_terminal_attempt_path(pair_root, "candidate-e2e-result.json")
        if pair_e2e is not None:
            pair_relative = Path(pair_e2e).relative_to(task_directory)
            attempt_directory = pair_relative.parent
            selection.update(
                {
                    "candidate_e2e": str(pair_relative.as_posix()),
                    "candidate_result": str(
                        (attempt_directory / "candidate-result.json").as_posix()
                    ),
                    "pre_gate": str(
                        (attempt_directory / "performance-pre-gate.json").as_posix()
                    ),
                    "baseline": str(
                        (attempt_directory / "baseline-result.json").as_posix()
                    ),
                }
            )
        else:
            rerun_root = experiment_directory / "e2e-reruns"
            rerun_e2e = _latest_terminal_attempt_path(rerun_root, "e2e-result.json")
            if rerun_e2e is not None:
                rerun_relative = Path(rerun_e2e).relative_to(task_directory)
                attempt_directory = rerun_relative.parent
                selection.update(
                    {
                        "candidate_e2e": str(rerun_relative.as_posix()),
                        "candidate_result": str(
                            (attempt_directory / "result-pre-quality.json").as_posix()
                        ),
                        "pre_gate": str(
                            (attempt_directory / "performance-pre-gate.json").as_posix()
                        ),
                    }
                )
    final_gate = f"{experiment_prefix}gate-decision.json"
    if store.artifact_ref(task.id, final_gate) is not None:
        selection["decision"] = final_gate
    else:
        selection["decision"] = selection["pre_gate"]
    quality_baseline = _pointer_path(
        store, task.id, "state/quality-baseline-current.json"
    )
    expected_quality_prefix = f"artifacts/evidence/quality-baselines/{experiment_id}/"
    if quality_baseline is not None and quality_baseline.startswith(expected_quality_prefix):
        selection["quality_baseline"] = quality_baseline
        selection["quality_baseline_pointer"] = "state/quality-baseline-current.json"
    else:
        candidates = sorted(
            path
            for path in _task_manifest(store, task.id)
            if path.startswith(expected_quality_prefix)
        )
        if candidates:
            selection["quality_baseline"] = candidates[-1]
    profile_state_path = f"state/profiles/{experiment_id}.json"
    if store.artifact_ref(task.id, profile_state_path) is not None:
        selection["profile_state"] = profile_state_path
    selection["task"] = "task.json"
    for key, path in list(selection.items()):
        optional = key in {"candidate_result", "quality"} or (
            key == "pre_gate" and path != selection["decision"]
        )
        if optional and store.artifact_ref(task.id, path) is None:
            selection.pop(key)
            continue
        _verified_ref(store, task.id, path)
    decision = _verified_json(store, task.id, selection["decision"])
    is_final_gate = selection["decision"].endswith("/gate-decision.json")
    if not is_final_gate and str(decision.get("outcome")) != "REJECT":
        raise BundleError("experiment has no terminal Gate decision")
    return selection


def _artifact_role(path: str) -> BundleArtifactRole:
    name = PurePosixPath(path).name.lower()
    lowered = path.lower()
    if name == "spec.json":
        return BundleArtifactRole.SPEC
    if "patch" in name or name.endswith(".diff"):
        return BundleArtifactRole.CHANGE
    if "gate" in name or "decision" in name:
        return BundleArtifactRole.DECISION
    if "quality" in lowered:
        return BundleArtifactRole.QUALITY
    if "profile" in lowered or "kernel-evidence" in lowered:
        return BundleArtifactRole.PROFILE
    if "environment" in name or "identity" in name:
        return BundleArtifactRole.ENVIRONMENT
    if "build" in name or "configure" in name or "smoke" in name:
        return BundleArtifactRole.BUILD
    if "e2e" in name or "microbenchmark" in name or "benchmark" in name:
        return BundleArtifactRole.BENCHMARK
    if name.endswith((".stdout", ".stderr", ".log")):
        return BundleArtifactRole.LOG
    if "attempt" in lowered:
        return BundleArtifactRole.ATTEMPT
    return BundleArtifactRole.OTHER


def _attempt_id(path: str) -> str | None:
    coordinate = _attempt_coordinate(path)
    return coordinate[1] if coordinate is not None else None


def _attempt_coordinate(path: str) -> tuple[str, str] | None:
    """Return an attempt's namespace and normalized id from its canonical path."""

    parts = PurePosixPath(path).parts
    markers = {
        "e2e-reruns": "e2e-rerun",
        "extended-pair-verification": "extended-pair",
    }
    for marker, kind in markers.items():
        if marker not in parts:
            continue
        index = parts.index(marker) + 1
        if index < len(parts):
            identifier = parts[index].removesuffix(".json")
            if identifier:
                return kind, identifier
    for index in range(len(parts) - 2):
        if parts[index : index + 2] != ("quality", "attempts"):
            continue
        identifier = parts[index + 2].removesuffix(".json")
        if identifier:
            return "quality", identifier
    return None


def model_input_provenance_path(experiment_id: str) -> str:
    """Return the canonical registered artifact path for candidate model provenance."""

    _safe_experiment_id(experiment_id)
    return f"experiments/{experiment_id}/model-input-provenance.json"


def _stage_for_role(role: BundleArtifactRole) -> str | None:
    return {
        BundleArtifactRole.SPEC: "CREATE_EXPERIMENT",
        BundleArtifactRole.CHANGE: "PATCH_AND_BUILD",
        BundleArtifactRole.BUILD: "PATCH_AND_BUILD",
        BundleArtifactRole.BENCHMARK: "E2E_VALIDATION",
        BundleArtifactRole.QUALITY: "QUALITY_VALIDATION",
        BundleArtifactRole.PROFILE: "DISCOVER_HOTSPOTS",
        BundleArtifactRole.DECISION: "DECIDE",
    }.get(role)


def _source_entries(
    store: ExperimentStore,
    task: OptimizationTask,
    experiment_id: str,
    selection: dict[str, str],
) -> list[BundleArtifactEntry]:
    manifest = _task_manifest(store, task.id)
    prefix = f"experiments/{experiment_id}/"
    entries: list[BundleArtifactEntry] = []
    for path, artifact in sorted(manifest.items()):
        if not path.startswith(prefix):
            continue
        inner = path.removeprefix(prefix)
        if inner in _DERIVED_BUNDLE_FILES or "-worktree/" in inner:
            continue
        role = _artifact_role(path)
        entries.append(
            BundleArtifactEntry(
                artifact=artifact,
                role=role,
                stage=_stage_for_role(role),
                attempt_id=_attempt_id(path),
            )
        )
    for key, path in sorted(selection.items()):
        artifact = manifest.get(path)
        if artifact is None:
            raise BundleError(f"selected bundle source is not registered: {path}")
        role = (
            BundleArtifactRole.BASELINE
            if key in {"baseline", "quality_baseline"}
            else BundleArtifactRole.PROFILE
            if key == "profile_state"
            else _artifact_role(path)
        )
        entries.append(
            BundleArtifactEntry(
                artifact=artifact,
                role=role,
                stage=(
                    "CAPTURE_BASELINE"
                    if role == BundleArtifactRole.BASELINE
                    else _stage_for_role(role)
                ),
                attempt_id=_attempt_id(path),
                ownership=(
                    "experiment"
                    if path.startswith(prefix)
                    and key not in {"baseline", "quality_baseline"}
                    else "shared"
                ),
            )
        )
    profile_state = _optional_json(
        store.task_dir(task.id) / "state/profiles" / f"{experiment_id}.json"
    )
    if profile_state is not None:
        attempts = profile_state.get("attempts", [])
        if isinstance(attempts, list):
            succeeded_attempts = [
                raw
                for raw in attempts
                if isinstance(raw, dict) and raw.get("status") == "succeeded"
            ]
            selected_profile = (
                str(succeeded_attempts[-1].get("attempt_id"))
                if succeeded_attempts
                else None
            )
            for raw in attempts:
                if not isinstance(raw, dict) or not isinstance(raw.get("evidence"), str):
                    continue
                artifact = manifest.get(raw["evidence"])
                if artifact is None:
                    continue
                entries.append(
                    BundleArtifactEntry(
                        artifact=artifact,
                        role=BundleArtifactRole.PROFILE,
                        stage="DISCOVER_HOTSPOTS",
                        attempt_id=str(raw.get("attempt_id") or "") or None,
                        ownership="shared",
                        current=str(raw.get("attempt_id")) == selected_profile,
                    )
                )
    unique: dict[str, BundleArtifactEntry] = {}
    for entry in entries:
        unique[entry.artifact.path] = entry
    result = [unique[path] for path in sorted(unique)]
    for entry in result:
        if not store.verify_artifact(task.id, entry.artifact):
            raise BundleError(
                f"bundle source failed integrity verification: {entry.artifact.path}"
            )
    return result


def _source_digest(entries: Iterable[BundleArtifactEntry]) -> str:
    coordinates = [
        {
            "path": entry.artifact.path,
            "sha256": entry.artifact.sha256,
            "size": entry.artifact.size,
            "role": entry.role,
            "attempt_id": entry.attempt_id,
            "current": entry.current,
        }
        for entry in entries
    ]
    payload = json.dumps(coordinates, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _mean_cv(raw: Any) -> tuple[float | None, int | None, float | None, str]:
    if not isinstance(raw, dict):
        return None, None, None, ""
    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples or not all(
        isinstance(value, (int, float)) and math.isfinite(float(value)) for value in samples
    ):
        return None, None, None, str(raw.get("unit", ""))
    values = [float(value) for value in samples]
    mean = statistics.fmean(values)
    cv = statistics.stdev(values) / abs(mean) * 100 if len(values) > 1 and mean else 0.0
    return mean, len(values), cv, str(raw.get("unit", ""))


def _metric_summaries(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    gate: dict[str, Any] | None,
) -> list[BundleMetricSummary]:
    baseline_metrics = (
        baseline.get("benchmark", {}).get("metrics", {})
        if isinstance(baseline, dict)
        else {}
    )
    candidate_metrics = candidate.get("metrics", {}) if isinstance(candidate, dict) else {}
    improvements = (
        gate.get("metric_improvements_percent", {}) if isinstance(gate, dict) else {}
    )
    if not isinstance(baseline_metrics, dict) or not isinstance(candidate_metrics, dict):
        return []
    values: list[BundleMetricSummary] = []
    for name in sorted(set(baseline_metrics) | set(candidate_metrics)):
        baseline_mean, baseline_count, baseline_cv, baseline_unit = _mean_cv(
            baseline_metrics.get(name)
        )
        candidate_mean, candidate_count, candidate_cv, candidate_unit = _mean_cv(
            candidate_metrics.get(name)
        )
        delta = improvements.get(name) if isinstance(improvements, dict) else None
        if delta is None and baseline_mean not in {None, 0} and candidate_mean is not None:
            delta = (candidate_mean - baseline_mean) / abs(baseline_mean) * 100
        values.append(
            BundleMetricSummary(
                name=name,
                unit=candidate_unit or baseline_unit,
                baseline_mean=baseline_mean,
                candidate_mean=candidate_mean,
                delta_percent=float(delta) if isinstance(delta, (int, float)) else None,
                baseline_sample_count=baseline_count,
                candidate_sample_count=candidate_count,
                baseline_cv_percent=baseline_cv,
                candidate_cv_percent=candidate_cv,
            )
        )
    return values


def _quality_summary(
    task: OptimizationTask,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> BundleQualitySummary | None:
    baseline_quality = baseline.get("quality") if isinstance(baseline, dict) else None
    if not isinstance(candidate, dict) and not isinstance(baseline_quality, dict):
        return None
    candidate = candidate if isinstance(candidate, dict) else {}
    baseline_quality = baseline_quality if isinstance(baseline_quality, dict) else {}
    baseline_ppl = baseline_quality.get("perplexity")
    candidate_ppl = candidate.get("perplexity")
    for label, value in (("baseline", baseline_ppl), ("candidate", candidate_ppl)):
        if value is not None and (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise BundleError(f"{label} quality perplexity is invalid")
    ppl_delta = None
    if isinstance(baseline_ppl, (int, float)) and baseline_ppl and isinstance(
        candidate_ppl, (int, float)
    ):
        ppl_delta = (float(candidate_ppl) - float(baseline_ppl)) / float(baseline_ppl) * 100
    baseline_accuracies = baseline_quality.get("accuracies", {})
    candidate_accuracies = candidate.get("accuracies", {})
    baseline_accuracies = baseline_accuracies if isinstance(baseline_accuracies, dict) else {}
    candidate_accuracies = candidate_accuracies if isinstance(candidate_accuracies, dict) else {}
    for label, values in (
        ("baseline", baseline_accuracies),
        ("candidate", candidate_accuracies),
    ):
        for name, value in values.items():
            if (
                not isinstance(name, str)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise BundleError(f"{label} quality accuracy is invalid")
    requirements = task.quality.resolved_accuracy_requirements()
    metric = (
        requirements[0].metric
        if requirements
        else task.quality.accuracy_metric
    )
    baseline_accuracy = baseline_accuracies.get(metric)
    candidate_accuracy = candidate_accuracies.get(metric)
    accuracy_drop = None
    if isinstance(baseline_accuracy, (int, float)) and isinstance(
        candidate_accuracy, (int, float)
    ):
        accuracy_drop = (float(baseline_accuracy) - float(candidate_accuracy)) * 100
    return BundleQualitySummary(
        status=str(candidate.get("status", baseline_quality.get("status", "UNKNOWN"))),
        correctness_passed=candidate.get("correctness_passed"),
        baseline_perplexity=(
            float(baseline_ppl) if isinstance(baseline_ppl, (int, float)) else None
        ),
        candidate_perplexity=(
            float(candidate_ppl) if isinstance(candidate_ppl, (int, float)) else None
        ),
        perplexity_regression_percent=ppl_delta,
        accuracy_metric=metric,
        baseline_accuracy=(
            float(baseline_accuracy) if isinstance(baseline_accuracy, (int, float)) else None
        ),
        candidate_accuracy=(
            float(candidate_accuracy) if isinstance(candidate_accuracy, (int, float)) else None
        ),
        accuracy_drop_percentage_points=accuracy_drop,
        baseline_accuracies={str(key): float(value) for key, value in baseline_accuracies.items()},
        candidate_accuracies={
            str(key): float(value) for key, value in candidate_accuracies.items()
        },
    )


def _artifact_refs_for_prefix(
    entries: Iterable[BundleArtifactEntry], prefix: str
) -> list[ArtifactRef]:
    boundary = prefix.rstrip("/") + "/"
    return [
        entry.artifact
        for entry in entries
        if entry.artifact.path == prefix.rstrip("/")
        or entry.artifact.path.startswith(boundary)
    ]


def _attempt_index(
    store: ExperimentStore,
    task: OptimizationTask,
    experiment_id: str,
    entries: list[BundleArtifactEntry],
) -> ExperimentAttemptIndex:
    attempts: list[BundleAttemptSummary] = []
    selected: dict[str, str] = {}
    runner_result = _optional_json(
        store.task_dir(task.id) / "experiments" / experiment_id / "runner-result.json"
    )
    if runner_result is None:
        runner_result = _optional_json(
            store.task_dir(task.id)
            / "experiments"
            / experiment_id
            / "runner"
            / "runner-result.json"
        )
    if any(entry.artifact.path.endswith("runner-result.json") for entry in entries):
        runner_status = (
            "FAILED"
            if isinstance(runner_result, dict) and runner_result.get("failure_stage")
            else "COMPLETED"
            if isinstance(runner_result, dict)
            else "INCOMPLETE"
        )
        attempts.append(
            BundleAttemptSummary(
                kind="runner",
                attempt_id="initial",
                status=runner_status,
                selected=runner_status != "INCOMPLETE",
                evidence=_artifact_refs_for_prefix(entries, f"experiments/{experiment_id}/runner"),
            )
        )
        if runner_status != "INCOMPLETE":
            selected["runner"] = "initial"
    experiment_directory = store.task_dir(task.id) / "experiments" / experiment_id
    for kind, directory_name in (
        ("e2e-rerun", "e2e-reruns"),
        ("extended-pair", "extended-pair-verification"),
    ):
        root = experiment_directory / directory_name
        if root.is_dir() and not root.is_symlink():
            directories = sorted(
                path for path in root.iterdir() if path.is_dir() and not path.is_symlink()
            )
            terminal_directories = [
                directory
                for directory in directories
                if (directory / "performance-pre-gate.json").is_file()
                and not (directory / "performance-pre-gate.json").is_symlink()
            ]
            selected_directory = terminal_directories[-1] if terminal_directories else None
            for directory in directories:
                prefix = str(directory.relative_to(store.task_dir(task.id)).as_posix())
                gate = _optional_json(directory / "performance-pre-gate.json")
                attempt_id = directory.name.removesuffix(".json")
                status = (
                    str(gate.get("outcome", "COMPLETED"))
                    if isinstance(gate, dict)
                    else "INCOMPLETE"
                )
                attempts.append(
                    BundleAttemptSummary(
                        kind=kind,  # type: ignore[arg-type]
                        attempt_id=attempt_id,
                        status=status,
                        selected=directory == selected_directory,
                        evidence=_artifact_refs_for_prefix(entries, prefix),
                    )
                )
            if selected_directory is not None:
                selected[kind] = selected_directory.name.removesuffix(".json")
    quality_root = experiment_directory / "quality/attempts"
    if quality_root.is_dir() and not quality_root.is_symlink():
        quality_files = sorted(
            path for path in quality_root.glob("*.json") if path.is_file() and not path.is_symlink()
        )
        terminal_quality = [
            path
            for path in quality_files
            if str((_optional_json(path) or {}).get("state", ""))
            in {"COMPLETED", "FAILED", "TIMED_OUT", "ORPHANED"}
        ]
        selected_quality = terminal_quality[-1] if terminal_quality else None
        for path in quality_files:
            raw = _optional_json(path) or {}
            file_attempt_id = path.stem.removesuffix(".json")
            attempt_id = str(raw.get("attempt_id") or file_attempt_id).removesuffix(".json")
            if attempt_id != file_attempt_id:
                raise BundleError("quality attempt id differs from its artifact path")
            attempts.append(
                BundleAttemptSummary(
                    kind="quality",
                    attempt_id=attempt_id,
                    status=str(raw.get("state", "UNKNOWN")),
                    selected=path == selected_quality,
                    evidence=[
                        entry.artifact
                        for entry in entries
                        if _attempt_coordinate(entry.artifact.path)
                        == ("quality", attempt_id)
                    ],
                )
            )
        if selected_quality is not None:
            selected["quality"] = str(
                (_optional_json(selected_quality) or {}).get("attempt_id")
                or selected_quality.stem
            ).removesuffix(".json")
    profile_state = _optional_json(
        store.task_dir(task.id) / "state/profiles" / f"{experiment_id}.json"
    )
    if profile_state is not None and isinstance(profile_state.get("attempts"), list):
        successful = [
            raw
            for raw in profile_state["attempts"]
            if isinstance(raw, dict) and raw.get("status") == "succeeded"
        ]
        selected_profile = successful[-1] if successful else None
        for raw in profile_state["attempts"]:
            if not isinstance(raw, dict):
                continue
            identifier = str(raw.get("attempt_id") or "unknown").removesuffix(".json")
            succeeded = raw is selected_profile
            attempts.append(
                BundleAttemptSummary(
                    kind="profile",
                    attempt_id=identifier,
                    status=str(raw.get("status", "UNKNOWN")),
                    selected=succeeded,
                    evidence=[
                        entry.artifact
                        for entry in entries
                        if entry.role == BundleArtifactRole.PROFILE
                        and entry.attempt_id == identifier
                    ],
                )
            )
            if succeeded:
                selected["profile"] = identifier
    return ExperimentAttemptIndex(
        task_id=task.id,
        experiment_id=experiment_id,
        attempts=attempts,
        selected=selected,
    )


def _apply_current_attempt_selection(
    entries: list[BundleArtifactEntry],
    attempts: ExperimentAttemptIndex,
    selection: dict[str, str],
) -> list[BundleArtifactEntry]:
    selected_attempts = set(attempts.selected.items())
    selected_paths = set(selection.values())
    result: list[BundleArtifactEntry] = []
    for entry in entries:
        current = entry.current
        if entry.attempt_id is not None:
            coordinate = _attempt_coordinate(entry.artifact.path)
            if coordinate is None and entry.role == BundleArtifactRole.PROFILE:
                coordinate = ("profile", entry.attempt_id)
            current = coordinate in selected_attempts
        if entry.role in {
            BundleArtifactRole.BASELINE,
            BundleArtifactRole.BENCHMARK,
            BundleArtifactRole.DECISION,
        }:
            current = entry.artifact.path in selected_paths
        result.append(entry.model_copy(update={"current": current}))
    return result


def _normalized_model_path(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def resolve_candidate_model_provenance(
    store: ExperimentStore,
    task: OptimizationTask,
    experiment_id: str,
    spec: dict[str, Any],
) -> tuple[ModelInputProvenanceV1 | None, ArtifactRef | None]:
    """Load candidate accounting only from its canonical hash-bound artifact.

    A present artifact must agree with the immutable experiment spec.  Absence is
    represented explicitly so callers cannot accidentally substitute baseline size
    accounting for a different candidate model.
    """

    path = model_input_provenance_path(experiment_id)
    reference = store.artifact_ref(task.id, path)
    if reference is None:
        return None, None
    raw = _verified_json(store, task.id, path)
    try:
        provenance = ModelInputProvenanceV1.model_validate(raw)
    except (TypeError, ValueError) as error:
        raise BundleError("candidate model provenance is invalid") from error
    change = spec.get("change", {}) if isinstance(spec, dict) else {}
    change = change if isinstance(change, dict) else {}
    expected_path = str(change.get("candidate_model_path") or task.model.path)
    expected_sha256 = str(change.get("candidate_model_sha256") or task.model.sha256)
    expected_quantization = change.get("candidate_model_quantization") or task.model.quantization
    if _normalized_model_path(provenance.model_path) != _normalized_model_path(expected_path):
        raise BundleError("candidate model provenance path differs from experiment spec")
    if provenance.model_sha256 != expected_sha256:
        raise BundleError("candidate model provenance SHA-256 differs from experiment spec")
    if (
        provenance.quantization is not None
        and expected_quantization is not None
        and provenance.quantization != str(expected_quantization)
    ):
        raise BundleError("candidate model provenance quantization differs from experiment spec")
    if (
        provenance.architecture is not None
        and task.model.architecture is not None
        and provenance.architecture != task.model.architecture
    ):
        raise BundleError("candidate model provenance architecture differs from task")
    return provenance, reference


def _candidate_model(
    store: ExperimentStore,
    task: OptimizationTask,
    experiment_id: str,
    spec: dict[str, Any],
) -> BundleModelSummary:
    change = spec.get("change", {}) if isinstance(spec, dict) else {}
    change = change if isinstance(change, dict) else {}
    raw_path = change.get("candidate_model_path") or str(task.model.path)
    path = Path(str(raw_path)).expanduser()
    candidate_sha256 = str(change.get("candidate_model_sha256") or task.model.sha256)
    is_task_model = (
        candidate_sha256 == task.model.sha256
        and _normalized_model_path(path) == _normalized_model_path(task.model.path)
    )
    provenance, provenance_ref = resolve_candidate_model_provenance(
        store, task, experiment_id, spec
    )
    packed_bytes = None
    raw_packed_bytes = (
        provenance.packed_bytes
        if provenance is not None
        else task.metadata.get("model_packed_bytes")
        if is_task_model
        else None
    )
    if raw_packed_bytes is not None:
        try:
            packed_bytes = int(raw_packed_bytes)
        except (TypeError, ValueError):
            pass
    effective_bpw = None
    raw_bpw = (
        provenance.effective_bpw
        if provenance is not None
        else change.get("effective_bpw")
        or (task.metadata.get("effective_bpw") if is_task_model else None)
    )
    if raw_bpw is not None:
        try:
            effective_bpw = float(raw_bpw)
        except (TypeError, ValueError):
            pass
    return BundleModelSummary(
        name=path.name,
        path=str(path),
        sha256=candidate_sha256,
        architecture=task.model.architecture,
        quantization=change.get("candidate_model_quantization") or task.model.quantization,
        packed_bytes=packed_bytes,
        effective_bpw=effective_bpw,
        provenance_artifact=provenance_ref,
    )


def _canonical_write(
    store: ExperimentStore,
    task_id: str,
    relative_path: str,
    value: StrictModel,
    *,
    producer: str,
) -> ArtifactRef:
    data = store._json_bytes(value)  # noqa: SLF001 - store owns canonical JSON encoding
    path = store.task_dir(task_id) / relative_path
    existing = store.artifact_ref(task_id, relative_path)
    desired_digest = hashlib.sha256(data).hexdigest()
    if (
        existing is not None
        and existing.sha256 == desired_digest
        and existing.size == len(data)
        and path.is_file()
        and not path.is_symlink()
        and store.verify_artifact(task_id, existing)
        and hashlib.sha256(path.read_bytes()).hexdigest() == desired_digest
    ):
        return existing
    return store.save_bytes(
        task_id,
        relative_path,
        data,
        producer=producer,
        media_type="application/json",
    )


def _build_summary(
    store: ExperimentStore,
    task: OptimizationTask,
    experiment_id: str,
    experiment_directory: Path,
    entries: list[BundleArtifactEntry],
    attempts: ExperimentAttemptIndex,
    selection: dict[str, str],
) -> ExperimentBundleSummary:
    del experiment_directory
    spec = _verified_json(store, task.id, selection["spec"])
    baseline = _verified_json(store, task.id, selection["baseline"])
    candidate = _verified_json(store, task.id, selection["candidate_e2e"])
    quality = (
        _verified_json(store, task.id, selection["quality"])
        if "quality" in selection
        else None
    )
    quality_baseline = (
        _verified_json(store, task.id, selection["quality_baseline"])
        if "quality_baseline" in selection
        else baseline
    )
    selected_gate = _verified_json(store, task.id, selection["decision"])
    final = selection["decision"].endswith("/gate-decision.json") or str(
        selected_gate.get("outcome")
    ) == "REJECT"
    decision = BundleDecisionSummary(
        outcome=str(selected_gate.get("outcome", "UNKNOWN")),
        final=final,
        reasons=[str(value) for value in selected_gate.get("reasons", [])],
        rerun_from_stage=(
            str(selected_gate["rerun_from_stage"])
            if selected_gate.get("rerun_from_stage") is not None
            else None
        ),
    )
    status = decision.outcome
    change = spec.get("change", {}) if isinstance(spec.get("change"), dict) else {}
    updated_at = max(
        (entry.artifact.created_at for entry in entries),
        default=None,
    )
    return ExperimentBundleSummary(
        task_id=task.id,
        experiment_id=experiment_id,
        status=status,
        model=_candidate_model(store, task, experiment_id, spec),
        runtime=task.runtime.name,
        runtime_commit=task.runtime.base_commit,
        gpu=task.gpu.name,
        gfx=task.gpu.gfx_target,
        strategy=str(change.get("kind")) if change.get("kind") is not None else None,
        hypothesis_id=(str(spec.get("hypothesis_id")) if spec.get("hypothesis_id") else None),
        change_summary=(str(change.get("description")) if change.get("description") else None),
        metrics=_metric_summaries(baseline, candidate, selected_gate),
        quality=_quality_summary(task, quality_baseline, quality),
        decision=decision,
        selected_attempts=attempts.selected,
        artifact_count=len(entries),
        source_digest=_source_digest(entries),
        updated_at=updated_at,
    )


def refresh_experiment_bundle(
    store: ExperimentStore,
    task_id: str,
    experiment_id: str,
    *,
    acquire_lock: bool = True,
) -> ExperimentBundleManifest:
    """Collect, summarize, publish, and index one experiment bundle."""

    _safe_experiment_id(experiment_id)
    lock = store.task_lock(task_id) if acquire_lock else nullcontext()
    with lock:
        task = store.load_task(task_id)
        directory = store.task_dir(task_id) / "experiments" / experiment_id
        if directory.is_symlink() or not directory.is_dir():
            raise BundleError(f"experiment not found: {experiment_id}")
        selection = _selected_evidence_paths(store, task, experiment_id)
        _collect_declared_patch(store, task_id, experiment_id)
        _collect_unregistered_outputs(store, task_id, experiment_id)
        entries = _source_entries(store, task, experiment_id, selection)
        attempts = _attempt_index(store, task, experiment_id, entries)
        entries = _apply_current_attempt_selection(entries, attempts, selection)
        attempts_ref = _canonical_write(
            store,
            task_id,
            f"experiments/{experiment_id}/attempts/index.json",
            attempts,
            producer="experiment-bundle",
        )
        summary = _build_summary(
            store,
            task,
            experiment_id,
            directory,
            entries,
            attempts,
            selection,
        )
        summary_ref = _canonical_write(
            store,
            task_id,
            f"experiments/{experiment_id}/summary.json",
            summary,
            producer="experiment-bundle",
        )
        published_at = max(
            [summary_ref.created_at, attempts_ref.created_at]
            + [entry.artifact.created_at for entry in entries]
        )
        manifest = ExperimentBundleManifest(
            task_id=task_id,
            experiment_id=experiment_id,
            source_digest=summary.source_digest,
            summary=summary_ref,
            attempts_index=attempts_ref,
            artifacts=[
                *entries,
                BundleArtifactEntry(
                    artifact=attempts_ref,
                    role=BundleArtifactRole.ATTEMPT,
                ),
                BundleArtifactEntry(
                    artifact=summary_ref,
                    role=BundleArtifactRole.SUMMARY,
                ),
            ],
            published_at=published_at,
        )
        manifest_ref = _canonical_write(
            store,
            task_id,
            f"experiments/{experiment_id}/manifest.json",
            manifest,
            producer="experiment-bundle",
        )
        ExperimentCatalog(store.root).upsert(summary, manifest, manifest_ref)
    return manifest


def refresh_task_bundles(
    store: ExperimentStore,
    task_id: str,
    *,
    terminal_only: bool = True,
) -> list[ExperimentBundleManifest]:
    """Rebuild every experiment projection for a task and refresh the catalog."""

    experiments = store.task_dir(task_id) / "experiments"
    if experiments.is_symlink() or not experiments.is_dir():
        return []
    identifiers: list[str] = []
    active = _optional_json(store.task_dir(task_id) / "state/active-execution.json")
    for path in sorted(experiments.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.is_symlink():
            continue
        gate = _optional_json(path / "gate-decision.json")
        pre_gate = _optional_json(path / "performance-pre-gate.json")
        terminal = gate is not None or (
            isinstance(pre_gate, dict) and str(pre_gate.get("outcome")) == "REJECT"
        )
        if (
            not terminal
            and isinstance(active, dict)
            and active.get("experiment_id") == path.name
            and isinstance(active.get("pre_gate_path"), str)
        ):
            portable = PurePosixPath(str(active["pre_gate_path"]))
            if not portable.is_absolute() and ".." not in portable.parts:
                selected_pre_gate = _optional_json(
                    store.task_dir(task_id) / Path(*portable.parts)
                )
                terminal = isinstance(selected_pre_gate, dict) and str(
                    selected_pre_gate.get("outcome")
                ) == "REJECT"
        if terminal_only and not terminal:
            continue
        identifiers.append(path.name)
    return [refresh_experiment_bundle(store, task_id, identifier) for identifier in identifiers]


def load_experiment_bundle(
    store: ExperimentStore,
    task_id: str,
    experiment_id: str,
) -> tuple[ExperimentBundleSummary, ExperimentBundleManifest]:
    _safe_experiment_id(experiment_id)
    summary = store.load_json(
        task_id,
        f"experiments/{experiment_id}/summary.json",
        ExperimentBundleSummary,
    )
    manifest = store.load_json(
        task_id,
        f"experiments/{experiment_id}/manifest.json",
        ExperimentBundleManifest,
    )
    if summary.task_id != task_id or manifest.task_id != task_id:
        raise BundleError("bundle task id mismatch")
    if summary.experiment_id != experiment_id or manifest.experiment_id != experiment_id:
        raise BundleError("bundle experiment id mismatch")
    if summary.source_digest != manifest.source_digest:
        raise BundleError("bundle summary and manifest source digests differ")
    expected_summary_path = f"experiments/{experiment_id}/summary.json"
    expected_attempts_path = f"experiments/{experiment_id}/attempts/index.json"
    if manifest.summary.path != expected_summary_path:
        raise BundleError("bundle summary path is not canonical")
    if manifest.attempts_index.path != expected_attempts_path:
        raise BundleError("bundle attempt index path is not canonical")
    summary_ref = store.artifact_ref(task_id, manifest.summary.path)
    manifest_ref = store.artifact_ref(task_id, f"experiments/{experiment_id}/manifest.json")
    if summary_ref != manifest.summary or not store.verify_artifact(task_id, manifest.summary):
        raise BundleError("bundle summary is not hash-bound")
    if manifest_ref is None or not store.verify_artifact(task_id, manifest_ref):
        raise BundleError("bundle manifest is not hash-bound")
    if not store.verify_artifact(task_id, manifest.attempts_index):
        raise BundleError("bundle attempt index is not hash-bound")
    attempt_index = store.load_json(
        task_id,
        manifest.attempts_index.path,
        ExperimentAttemptIndex,
    )
    if attempt_index.task_id != task_id or attempt_index.experiment_id != experiment_id:
        raise BundleError("bundle attempt index coordinates differ")
    seen: set[str] = set()
    for entry in manifest.artifacts:
        path = entry.artifact.path
        if path in seen:
            raise BundleError("bundle manifest contains duplicate artifacts")
        seen.add(path)
        registered = store.artifact_ref(task_id, path)
        if registered != entry.artifact or not store.verify_artifact(task_id, entry.artifact):
            raise BundleError(f"bundle artifact is missing or corrupt: {path}")
    source_entries = [
        entry
        for entry in manifest.artifacts
        if entry.artifact.path not in {expected_summary_path, expected_attempts_path}
    ]
    if summary.artifact_count != len(source_entries):
        raise BundleError("bundle source artifact count differs")
    if _source_digest(source_entries) != summary.source_digest:
        raise BundleError("bundle source digest does not verify")
    return summary, manifest


class ExperimentCatalog:
    """Disposable SQLite index over canonical experiment bundles."""

    def __init__(self, store_root: str | Path) -> None:
        self.path = Path(store_root).expanduser().resolve() / CATALOG_FILENAME

    def _connect(self) -> sqlite3.Connection:
        if self.path.is_symlink():
            raise BundleError("experiment catalog cannot be a symbolic link")
        try:
            connection = sqlite3.connect(self.path, timeout=30)
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(
                """
            CREATE TABLE IF NOT EXISTS catalog_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments (
                task_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                status TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_sha256 TEXT,
                quantization TEXT,
                strategy TEXT,
                hypothesis_id TEXT,
                change_summary TEXT,
                updated_at TEXT,
                summary_path TEXT NOT NULL,
                summary_sha256 TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL,
                artifact_count INTEGER NOT NULL,
                PRIMARY KEY (task_id, experiment_id)
            );
            CREATE TABLE IF NOT EXISTS metrics (
                task_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                name TEXT NOT NULL,
                unit TEXT NOT NULL,
                baseline_mean REAL,
                candidate_mean REAL,
                delta_percent REAL,
                candidate_cv_percent REAL,
                PRIMARY KEY (task_id, experiment_id, name),
                FOREIGN KEY (task_id, experiment_id)
                    REFERENCES experiments(task_id, experiment_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS quality_summary (
                task_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                status TEXT NOT NULL,
                baseline_perplexity REAL,
                candidate_perplexity REAL,
                perplexity_regression_percent REAL,
                accuracy_metric TEXT,
                baseline_accuracy REAL,
                candidate_accuracy REAL,
                accuracy_drop_percentage_points REAL,
                PRIMARY KEY (task_id, experiment_id),
                FOREIGN KEY (task_id, experiment_id)
                    REFERENCES experiments(task_id, experiment_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS bundle_artifacts (
                task_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                role TEXT NOT NULL,
                attempt_id TEXT,
                ownership TEXT NOT NULL,
                storage TEXT NOT NULL,
                media_type TEXT NOT NULL,
                producer TEXT NOT NULL,
                size INTEGER NOT NULL,
                current INTEGER NOT NULL,
                PRIMARY KEY (task_id, experiment_id, path),
                FOREIGN KEY (task_id, experiment_id)
                    REFERENCES experiments(task_id, experiment_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS experiments_model_idx
                ON experiments(model_sha256, quantization);
            CREATE INDEX IF NOT EXISTS experiments_status_idx ON experiments(status);
            """
            )
            connection.execute(
                "INSERT OR REPLACE INTO catalog_meta(key, value) VALUES('schema_version', ?)",
                (str(CATALOG_SCHEMA_VERSION),),
            )
        except sqlite3.Error as error:
            raise BundleError(f"cannot open experiment catalog: {error}") from error
        return connection

    def upsert(
        self,
        summary: ExperimentBundleSummary,
        manifest: ExperimentBundleManifest,
        manifest_ref: ArtifactRef,
    ) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO experiments(
                        task_id, experiment_id, status, model_name, model_sha256,
                        quantization, strategy, hypothesis_id, change_summary, updated_at,
                        summary_path, summary_sha256, manifest_path, manifest_sha256,
                        artifact_count
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id, experiment_id) DO UPDATE SET
                        status=excluded.status,
                        model_name=excluded.model_name,
                        model_sha256=excluded.model_sha256,
                        quantization=excluded.quantization,
                        strategy=excluded.strategy,
                        hypothesis_id=excluded.hypothesis_id,
                        change_summary=excluded.change_summary,
                        updated_at=excluded.updated_at,
                        summary_path=excluded.summary_path,
                        summary_sha256=excluded.summary_sha256,
                        manifest_path=excluded.manifest_path,
                        manifest_sha256=excluded.manifest_sha256,
                        artifact_count=excluded.artifact_count
                    """,
                    (
                        summary.task_id,
                        summary.experiment_id,
                        summary.status,
                        summary.model.name,
                        summary.model.sha256,
                        summary.model.quantization,
                        summary.strategy,
                        summary.hypothesis_id,
                        summary.change_summary,
                        summary.updated_at.isoformat() if summary.updated_at else None,
                        manifest.summary.path,
                        manifest.summary.sha256,
                        manifest_ref.path,
                        manifest_ref.sha256,
                        len(manifest.artifacts),
                    ),
                )
                connection.execute(
                    "DELETE FROM metrics WHERE task_id=? AND experiment_id=?",
                    (summary.task_id, summary.experiment_id),
                )
                connection.executemany(
                    """
                    INSERT INTO metrics(
                        task_id, experiment_id, name, unit, baseline_mean, candidate_mean,
                        delta_percent, candidate_cv_percent
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            summary.task_id,
                            summary.experiment_id,
                            metric.name,
                            metric.unit,
                            metric.baseline_mean,
                            metric.candidate_mean,
                            metric.delta_percent,
                            metric.candidate_cv_percent,
                        )
                        for metric in summary.metrics
                    ],
                )
                connection.execute(
                    "DELETE FROM quality_summary WHERE task_id=? AND experiment_id=?",
                    (summary.task_id, summary.experiment_id),
                )
                if summary.quality is not None:
                    quality = summary.quality
                    connection.execute(
                        """
                        INSERT INTO quality_summary(
                            task_id, experiment_id, status, baseline_perplexity,
                            candidate_perplexity, perplexity_regression_percent,
                            accuracy_metric, baseline_accuracy, candidate_accuracy,
                            accuracy_drop_percentage_points
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            summary.task_id,
                            summary.experiment_id,
                            quality.status,
                            quality.baseline_perplexity,
                            quality.candidate_perplexity,
                            quality.perplexity_regression_percent,
                            quality.accuracy_metric,
                            quality.baseline_accuracy,
                            quality.candidate_accuracy,
                            quality.accuracy_drop_percentage_points,
                        ),
                    )
                connection.execute(
                    "DELETE FROM bundle_artifacts WHERE task_id=? AND experiment_id=?",
                    (summary.task_id, summary.experiment_id),
                )
                connection.executemany(
                    """
                    INSERT INTO bundle_artifacts(
                        task_id, experiment_id, path, sha256, role, attempt_id,
                        ownership, storage, media_type, producer, size, current
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            summary.task_id,
                            summary.experiment_id,
                            entry.artifact.path,
                            entry.artifact.sha256,
                            str(entry.role),
                            entry.attempt_id,
                            entry.ownership,
                            entry.storage,
                            entry.artifact.media_type,
                            entry.artifact.producer,
                            entry.artifact.size,
                            int(entry.current),
                        )
                        for entry in manifest.artifacts
                    ],
                )
        except sqlite3.Error as error:
            raise BundleError(f"cannot update experiment catalog: {error}") from error

    def rows(self, *, task_id: str | None = None) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection:
                connection.row_factory = sqlite3.Row
                if task_id is None:
                    rows = connection.execute(
                        "SELECT * FROM experiments ORDER BY updated_at DESC, experiment_id"
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """
                    SELECT * FROM experiments WHERE task_id=?
                    ORDER BY updated_at DESC, experiment_id
                    """,
                        (task_id,),
                    ).fetchall()
        except sqlite3.Error as error:
            raise BundleError(f"cannot read experiment catalog: {error}") from error
        return [dict(row) for row in rows]

    def prune(
        self,
        valid: set[tuple[str, str]],
        *,
        task_id: str | None = None,
    ) -> None:
        """Remove cache rows not backed by a currently discovered bundle."""

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT task_id, experiment_id FROM experiments"
                    + (" WHERE task_id=?" if task_id is not None else ""),
                    ((task_id,) if task_id is not None else ()),
                ).fetchall()
                stale = [tuple(row) for row in rows if tuple(row) not in valid]
                connection.executemany(
                    "DELETE FROM experiments WHERE task_id=? AND experiment_id=?",
                    stale,
                )
        except sqlite3.Error as error:
            raise BundleError(f"cannot prune experiment catalog: {error}") from error


__all__ = [
    "BUNDLE_ATTEMPTS_SCHEMA",
    "BUNDLE_MANIFEST_SCHEMA",
    "BUNDLE_SUMMARY_SCHEMA",
    "CATALOG_FILENAME",
    "MODEL_INPUT_PROVENANCE_SCHEMA",
    "BundleArtifactEntry",
    "BundleArtifactRole",
    "BundleError",
    "BundleMetricSummary",
    "BundleQualitySummary",
    "ExperimentAttemptIndex",
    "ExperimentBundleManifest",
    "ExperimentBundleSummary",
    "ExperimentCatalog",
    "ModelInputProvenanceV1",
    "load_experiment_bundle",
    "model_input_provenance_path",
    "refresh_experiment_bundle",
    "refresh_task_bundles",
    "resolve_candidate_model_provenance",
]
