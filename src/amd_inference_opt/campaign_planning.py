"""Materialize the three bounded campaign adapter plans as immutable evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .campaign_adapters import (
    ArtifactProvenance,
    CampaignAdapterError,
    MixedBitProvenance,
    ModelGeometry,
    plan_mixed_bit_candidates,
    plan_precision_reference_candidates,
    plan_shape_kernel_candidates,
)
from .campaign_models import (
    CampaignCandidate,
    CampaignRecord,
    CampaignStage,
    CampaignStrategy,
)
from .control_policy import content_fingerprint
from .kv_cache import KVCacheError, build_kv_campaign_plan
from .models import ArtifactRef
from .quality_policy import ProvisionalMath100Policy
from .store import ExperimentStore


class CampaignPlanningError(RuntimeError):
    """The campaign planning inputs are incomplete or invalid."""


_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignPlanningError(f"planning_inputs.{label} must be an object")
    return value


def _artifact(value: object, label: str) -> ArtifactProvenance:
    document = _mapping(value, label)
    try:
        return ArtifactProvenance(
            path=str(document["path"]),
            sha256=str(document["sha256"]),
            size_bytes=document.get("size_bytes"),
        )
    except (KeyError, TypeError, ValueError, CampaignAdapterError) as error:
        raise CampaignPlanningError(f"invalid {label}: {error}") from error


def _mixed_provenance(value: object) -> MixedBitProvenance:
    document = _mapping(value, "mixed_bit_provenance")
    try:
        return MixedBitProvenance(
            source_model=_artifact(document.get("source_model"), "source_model"),
            calibration_corpus=_artifact(
                document.get("calibration_corpus"), "calibration_corpus"
            ),
            importance_matrix=_artifact(
                document.get("importance_matrix"), "importance_matrix"
            ),
            quantizer_binary=_artifact(
                document.get("quantizer_binary"), "quantizer_binary"
            ),
            llama_cpp_commit=str(document["llama_cpp_commit"]),
            source_quantization=str(document.get("source_quantization", "BF16")),
            direct_from_source=document.get("direct_from_source", True),
        )
    except (KeyError, TypeError, ValueError, CampaignAdapterError) as error:
        raise CampaignPlanningError(f"invalid mixed_bit_provenance: {error}") from error


def _prepared_mixed_candidate(value: object, index: int) -> tuple[str, str, dict[str, Any]]:
    """Validate one externally prepared, hash-bound mixed-precision arm.

    Preparation remains outside the planner.  This boundary lets a campaign
    describe model-specific tensor assignments without teaching the framework
    a new quantization format or silently accepting an unpinned GGUF.
    """

    document = _mapping(value, f"mixed_bit_prepared_candidates[{index}]")
    candidate_id = document.get("id")
    label = document.get("label")
    base_quantization = document.get("base_quantization")
    effective_bpw = document.get("effective_bpw")
    tensor_assignment = document.get("tensor_assignment")
    if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise CampaignPlanningError(
            f"mixed_bit_prepared_candidates[{index}].id must be a lowercase identifier"
        )
    if not isinstance(label, str) or not label.strip():
        raise CampaignPlanningError(
            f"mixed_bit_prepared_candidates[{index}].label must be non-empty"
        )
    if not isinstance(base_quantization, str) or not base_quantization.strip():
        raise CampaignPlanningError(
            f"mixed_bit_prepared_candidates[{index}].base_quantization must be non-empty"
        )
    if (
        isinstance(effective_bpw, bool)
        or not isinstance(effective_bpw, (int, float))
        or not 0 < float(effective_bpw) <= 32
    ):
        raise CampaignPlanningError(
            f"mixed_bit_prepared_candidates[{index}].effective_bpw must be in (0, 32]"
        )
    if not isinstance(tensor_assignment, Mapping) or not tensor_assignment:
        raise CampaignPlanningError(
            f"mixed_bit_prepared_candidates[{index}].tensor_assignment must be non-empty"
        )
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(precision, str)
        or not precision.strip()
        for name, precision in tensor_assignment.items()
    ):
        raise CampaignPlanningError(
            f"mixed_bit_prepared_candidates[{index}].tensor_assignment must map strings to strings"
        )
    model = _artifact(document.get("model"), f"prepared candidate {candidate_id} model")
    manifest = _artifact(
        document.get("preparation_manifest"),
        f"prepared candidate {candidate_id} preparation_manifest",
    )
    return candidate_id, label, {
        "schema": "gpuopt.prepared-mixed-bit-candidate.v1",
        "strategy": CampaignStrategy.MIXED_BIT.value,
        "candidate_id": candidate_id,
        "label": label,
        "base_quantization": base_quantization,
        "effective_bpw": float(effective_bpw),
        "tensor_assignment": dict(tensor_assignment),
        "model": {
            "path": model.path,
            "sha256": model.sha256,
            "size_bytes": model.size_bytes,
        },
        "preparation_manifest": {
            "path": manifest.path,
            "sha256": manifest.sha256,
            "size_bytes": manifest.size_bytes,
        },
        "prepared_externally": True,
    }


def _save_candidate(
    store: ExperimentStore,
    record: CampaignRecord,
    *,
    candidate_id: str,
    strategy: CampaignStrategy,
    label: str,
    spec: Mapping[str, Any],
) -> CampaignCandidate:
    action_name = {
        CampaignStrategy.MIXED_BIT: "mixed_bit",
        CampaignStrategy.SHAPE_KERNEL: "shape_kernel",
        CampaignStrategy.KV_CACHE: "kv_cache",
    }[strategy]
    semantic_spec = {
        key: value
        for key, value in spec.items()
        if key not in {"candidate_id", "id", "label"}
    }
    fingerprint = content_fingerprint(action_name, semantic_spec)
    reference = store.save_evidence_json(
        record.task_id,
        f"campaign/candidates/{candidate_id}",
        dict(spec),
        producer="campaign-planner",
    )
    return CampaignCandidate(
        id=candidate_id,
        strategy=strategy,
        label=label,
        shared_baseline_ids=[item.id for item in record.shared_baselines],
        spec_artifact=reference,
        content_fingerprint=fingerprint,
    )


def materialize_campaign_plan(
    record: CampaignRecord,
    store: ExperimentStore,
) -> tuple[list[CampaignCandidate], ArtifactRef]:
    """Build all configured candidate specs without executing tools or workloads."""

    if record.current_stage != CampaignStage.PLAN_CANDIDATES:
        raise CampaignPlanningError("campaign planning is only valid at PLAN_CANDIDATES")
    if not record.shared_baselines:
        raise CampaignPlanningError("campaign planning requires shared baseline references")

    inputs = record.config.planning_inputs
    geometry_document = _mapping(inputs.get("model_geometry"), "model_geometry")
    try:
        geometry = ModelGeometry.from_metadata(geometry_document)
    except CampaignAdapterError as error:
        raise CampaignPlanningError(str(error)) from error

    candidates: list[CampaignCandidate] = []
    if CampaignStrategy.MIXED_BIT in record.config.strategy_order:
        provenance = _mixed_provenance(inputs.get("mixed_bit_provenance"))
        raw_reference_quantizations = inputs.get("mixed_bit_reference_quantizations")
        if raw_reference_quantizations is None:
            mixed_specs = plan_mixed_bit_candidates(geometry, provenance)
        else:
            if not isinstance(raw_reference_quantizations, list) or not all(
                isinstance(value, str) for value in raw_reference_quantizations
            ):
                raise CampaignPlanningError(
                    "planning_inputs.mixed_bit_reference_quantizations must be a string array"
                )
            try:
                mixed_specs = plan_precision_reference_candidates(
                    provenance,
                    tuple(raw_reference_quantizations),
                )
            except CampaignAdapterError as error:
                raise CampaignPlanningError(str(error)) from error
        for spec in mixed_specs:
            candidates.append(
                _save_candidate(
                    store,
                    record,
                    candidate_id=f"mixed-{spec.candidate_id}",
                    strategy=CampaignStrategy.MIXED_BIT,
                    label=spec.candidate_id,
                    spec={
                        "schema": "gpuopt.mixed-bit-candidate.v1",
                        "strategy": CampaignStrategy.MIXED_BIT.value,
                        **spec.to_dict(),
                    },
                )
            )
        raw_prepared = inputs.get("mixed_bit_prepared_candidates", [])
        if not isinstance(raw_prepared, list):
            raise CampaignPlanningError(
                "planning_inputs.mixed_bit_prepared_candidates must be an array"
            )
        known_ids = {candidate.id for candidate in candidates}
        for index, value in enumerate(raw_prepared):
            candidate_id, label, spec = _prepared_mixed_candidate(value, index)
            qualified_id = f"mixed-{candidate_id}"
            if qualified_id in known_ids:
                raise CampaignPlanningError(
                    f"prepared mixed-bit candidate repeats id: {candidate_id}"
                )
            candidates.append(
                _save_candidate(
                    store,
                    record,
                    candidate_id=qualified_id,
                    strategy=CampaignStrategy.MIXED_BIT,
                    label=label,
                    spec=spec,
                )
            )
            known_ids.add(qualified_id)

    if CampaignStrategy.SHAPE_KERNEL in record.config.strategy_order:
        for spec in plan_shape_kernel_candidates(
            geometry,
            architecture=record.config.task.gpu.gfx_target,
        ):
            candidates.append(
                _save_candidate(
                    store,
                    record,
                    candidate_id=f"shape-{spec.candidate_id}",
                    strategy=CampaignStrategy.SHAPE_KERNEL,
                    label=spec.operator_group,
                    spec={
                        "schema": "gpuopt.shape-kernel-candidate.v1",
                        "strategy": CampaignStrategy.SHAPE_KERNEL.value,
                        **spec.to_dict(),
                    },
                )
            )

    if CampaignStrategy.KV_CACHE in record.config.strategy_order:
        capabilities = _mapping(inputs.get("kv_capabilities"), "kv_capabilities")
        try:
            kv_plan = build_kv_campaign_plan(
                {
                    "layers": geometry.kv_cache_layer_count,
                    "kv_heads": geometry.kv_head_count,
                    "head_dim": geometry.head_dim,
                    "max_context_tokens": geometry_document.get(
                        "max_context_tokens",
                        geometry_document.get("context_length"),
                    ),
                },
                capabilities,
            )
        except KVCacheError as error:
            raise CampaignPlanningError(str(error)) from error
        for arm in kv_plan.arms:
            candidates.append(
                _save_candidate(
                    store,
                    record,
                    candidate_id=f"kv-{arm.cache_type.value}",
                    strategy=CampaignStrategy.KV_CACHE,
                    label=f"KV cache {arm.cache_type.value}",
                    spec={
                        "schema": "gpuopt.kv-cache-candidate.v1",
                        "strategy": CampaignStrategy.KV_CACHE.value,
                        "campaign_plan_hash": kv_plan.plan_hash,
                        **arm.to_dict(),
                    },
                )
            )

    if not candidates:
        raise CampaignPlanningError("configured strategy order produced no candidates")
    if len(candidates) > record.config.max_candidates:
        raise CampaignPlanningError(
            f"planned {len(candidates)} candidates exceeds max_candidates="
            f"{record.config.max_candidates}"
        )

    policy = ProvisionalMath100Policy()
    aggregate = {
        "schema": "gpuopt.campaign-plan.v1",
        "campaign_id": record.campaign_id,
        "quality_policy": {
            "protocol_id": policy.protocol_id,
            "authoritative": policy.authoritative,
            "campaign_disposition": policy.campaign_disposition,
            "math_questions": 100,
            "max_math_correct_drop": policy.max_math_correct_drop,
            "max_perplexity_regression_percent": (
                policy.max_perplexity_regression_fraction * 100
            ),
            "greedy_accuracy_must_not_decline": True,
        },
        "candidate_count": len(candidates),
        "candidates": [
            {
                "id": candidate.id,
                "strategy": candidate.strategy.value,
                "content_fingerprint": candidate.content_fingerprint,
                "spec_artifact": candidate.spec_artifact.model_dump(mode="json"),
            }
            for candidate in candidates
        ],
    }
    plan_reference = store.save_evidence_json(
        record.task_id,
        "campaign/plan",
        aggregate,
        producer="campaign-planner",
    )
    return candidates, plan_reference


__all__ = ["CampaignPlanningError", "materialize_campaign_plan"]
