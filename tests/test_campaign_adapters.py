from __future__ import annotations

from dataclasses import replace

import pytest

from amd_inference_opt.campaign_adapters import (
    ArtifactProvenance,
    CampaignAdapterError,
    MixedBitCampaignAdapter,
    MixedBitProvenance,
    ModelGeometry,
    ShapeKernelCampaignAdapter,
    ShapePatchKnob,
    TensorRegexValidation,
    plan_precision_reference_candidates,
)


def _artifact(path: str, digit: str) -> ArtifactProvenance:
    return ArtifactProvenance(path=path, sha256=digit * 64, size_bytes=1024)


@pytest.fixture
def geometry() -> ModelGeometry:
    return ModelGeometry(
        block_count=36,
        hidden_size=4096,
        intermediate_size=12288,
        attention_head_count=32,
        kv_head_count=8,
        head_dim=128,
        vocab_size=151936,
    )


@pytest.fixture
def provenance() -> MixedBitProvenance:
    return MixedBitProvenance(
        source_model=_artifact("source-bf16.gguf", "1"),
        calibration_corpus=_artifact("calibration.txt", "2"),
        importance_matrix=_artifact("qwen3.imatrix", "3"),
        quantizer_binary=_artifact("llama-quantize", "4"),
        llama_cpp_commit="a7a6d0d269c896218b6c78e0933bd6a17519d3f6",
    )


def _qwen_tensor_inventory(geometry: ModelGeometry) -> tuple[str, ...]:
    operators = (
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    )
    return tuple(
        [
            f"blk.{block}.{operator}.weight"
            for block in range(geometry.block_count)
            for operator in operators
        ]
        + ["token_embd.weight", "output.weight"]
    )


def test_geometry_derives_head_dim_and_accepts_metadata_aliases() -> None:
    geometry = ModelGeometry.from_metadata(
        {
            "block_count": 36,
            "embedding_length": 4096,
            "feed_forward_length": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "token_count": 151936,
            "model_type": "qwen3",
        }
    )

    assert geometry.head_dim == 128
    assert geometry.kv_width == 1024
    with pytest.raises(CampaignAdapterError, match="head_dim"):
        replace(geometry, head_dim=64)
    with pytest.raises(CampaignAdapterError, match="conflicting hidden_size"):
        ModelGeometry.from_metadata(
            {
                "block_count": 36,
                "hidden_size": 4096,
                "embedding_length": 8192,
                "intermediate_size": 12288,
                "attention_head_count": 32,
                "kv_head_count": 8,
                "vocab_size": 151936,
            }
        )


def test_hybrid_geometry_limits_attention_and_kv_to_full_attention_layers() -> None:
    layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(32)
    ]
    geometry = ModelGeometry.from_metadata(
        {
            "block_count": 32,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "attention_head_count": 16,
            "kv_head_count": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "model_type": "qwen3_5_text",
            "layer_types": layer_types,
        }
    )
    shapes = ShapeKernelCampaignAdapter(geometry).candidates()

    assert geometry.attention_block_indices == (3, 7, 11, 15, 19, 23, 27, 31)
    assert geometry.kv_cache_layer_count == 8
    assert shapes[0].tensor_selector.expected_match_count == 64
    assert shapes[1].tensor_selector.expected_match_count == 16
    assert shapes[2].tensor_selector.expected_match_count == 16
    assert shapes[2].n == 1024


def test_mixed_bit_candidates_have_fixed_order_and_exact_q6_ffn_policies(
    geometry: ModelGeometry,
    provenance: MixedBitProvenance,
) -> None:
    candidates = MixedBitCampaignAdapter(geometry, provenance).candidates()

    assert [candidate.candidate_id for candidate in candidates] == [
        "q5_k_m",
        "q4_k_m",
        "q6_ffn_q5_k",
        "q6_gate_up_q4_k_down_q5_k",
        "q6_ffn_q4_k",
    ]
    assert [candidate.order for candidate in candidates] == [1, 2, 3, 4, 5]
    assert [candidate.base_quantization for candidate in candidates] == [
        "Q5_K_M",
        "Q4_K_M",
        "Q6_K",
        "Q6_K",
        "Q6_K",
    ]
    assert not candidates[0].tensor_overrides
    assert not candidates[1].tensor_overrides
    assert [override.tensor_type for override in candidates[2].tensor_overrides] == ["Q5_K"]
    assert [override.tensor_type for override in candidates[3].tensor_overrides] == [
        "Q4_K",
        "Q5_K",
    ]
    assert [override.operator_group for override in candidates[3].tensor_overrides] == [
        "gate/up",
        "down",
    ]
    assert [override.tensor_type for override in candidates[4].tensor_overrides] == ["Q4_K"]
    assert all(candidate.preserves_non_ffn_q6 for candidate in candidates[2:])
    assert all("--imatrix" in candidate.quantizer_args for candidate in candidates)


def test_configurable_reference_precision_sweep_supports_q5_q6_q8(
    provenance: MixedBitProvenance,
) -> None:
    candidates = plan_precision_reference_candidates(
        provenance,
        ("Q8_0", "Q6_K", "Q5_K_M"),
    )

    assert [candidate.candidate_id for candidate in candidates] == [
        "q8_0",
        "q6_k",
        "q5_k_m",
    ]
    assert [candidate.order for candidate in candidates] == [1, 2, 3]
    assert all(candidate.reference_arm for candidate in candidates)
    assert all(not candidate.tensor_overrides for candidate in candidates)
    assert [candidate.quantizer_args[-1] for candidate in candidates] == [
        "Q8_0",
        "Q6_K",
        "Q5_K_M",
    ]
    with pytest.raises(CampaignAdapterError, match="repeat"):
        plan_precision_reference_candidates(provenance, ("Q6_K", "Q6_K"))
    with pytest.raises(CampaignAdapterError, match="unsupported reference"):
        plan_precision_reference_candidates(provenance, ("Q7_K",))


def test_mixed_regexes_match_only_complete_expected_ffn_inventory(
    geometry: ModelGeometry,
    provenance: MixedBitProvenance,
) -> None:
    inventory = _qwen_tensor_inventory(geometry)
    candidates = MixedBitCampaignAdapter(geometry, provenance).candidates()

    for candidate in candidates:
        candidate.validate_tensor_inventory(inventory)
    all_ffn = candidates[2].tensor_overrides[0].selector
    assert all_ffn.expected_match_count == geometry.block_count * 3

    with pytest.raises(CampaignAdapterError, match="missing blk.0.ffn_gate.weight"):
        candidates[2].validate_tensor_inventory(
            name for name in inventory if name != "blk.0.ffn_gate.weight"
        )
    with pytest.raises(CampaignAdapterError, match="unexpected"):
        all_ffn.matching_names((*inventory, "blk.36.ffn_gate.weight"))
    with pytest.raises(CampaignAdapterError, match="anchored"):
        TensorRegexValidation(r"blk\.[0-9]+\.ffn_down\.weight", ("blk.0.ffn_down.weight",))


def test_mixed_provenance_forbids_requantization_and_unpinned_artifacts(
    provenance: MixedBitProvenance,
) -> None:
    with pytest.raises(CampaignAdapterError, match="requantizing"):
        replace(provenance, direct_from_source=False)
    with pytest.raises(CampaignAdapterError, match="BF16"):
        replace(provenance, source_quantization="Q6_K")
    with pytest.raises(CampaignAdapterError, match="sha256"):
        ArtifactProvenance("source.gguf", "ABC")


def test_shape_adapter_derives_four_real_shapes_in_approved_order(
    geometry: ModelGeometry,
) -> None:
    candidates = ShapeKernelCampaignAdapter(geometry).candidates()

    assert [candidate.candidate_id for candidate in candidates] == [
        "ffn_gate_up",
        "attention_query_output",
        "attention_key_value",
        "vocab_output_head",
    ]
    assert [(candidate.n, candidate.k) for candidate in candidates] == [
        (12288, 4096),
        (4096, 4096),
        (1024, 4096),
        (151936, 4096),
    ]
    assert [candidate.tensor_selector.expected_match_count for candidate in candidates] == [
        72,
        72,
        72,
        1,
    ]
    assert "ffn_down" not in candidates[0].tensor_selector.pattern
    assert candidates[3].tensor_selector.expected_names == ("output.weight",)
    for candidate in candidates:
        candidate.tensor_selector.matching_names(_qwen_tensor_inventory(geometry))


def test_shape_specs_lock_patch_knobs_dispatch_signatures_and_gate(
    geometry: ModelGeometry,
) -> None:
    candidates = ShapeKernelCampaignAdapter(geometry).candidates()
    gate_up = candidates[0]

    assert gate_up.patch_knob.architecture == "gfx1201"
    assert gate_up.patch_knob.quantization == "Q6_K"
    assert gate_up.patch_knob.ncols_dst == 1
    assert gate_up.patch_knob.small_k
    assert gate_up.patch_knob.rows_per_block == 8
    assert gate_up.rows_per_block == 8
    assert gate_up.split_k is None
    assert gate_up.waves_per_block is None
    assert gate_up.rows_per_wave is None
    assert gate_up.vector_load_bytes is None
    assert gate_up.vgpr_limit is None
    assert gate_up.fuse_gate_up is None
    assert gate_up.validation.baseline_grid == (393216, 8, 1)
    assert gate_up.validation.candidate_grid == (49152, 8, 1)
    assert gate_up.validation.workgroup == (32, 8, 1)
    assert gate_up.validation.changed_paths == ("ggml/src/ggml-cuda/mmvq.cu",)
    assert gate_up.validation.independent_clean_base
    assert gate_up.validation.benchmark_token_lengths == (128, 512)
    assert gate_up.validation.max_cv_percent == 2.0
    assert gate_up.validation.min_improvement_percent_each == 1.0

    vocab = candidates[-1]
    assert vocab.validation.baseline_grid == (geometry.vocab_size * 32, 8, 1)
    assert vocab.validation.candidate_grid == (geometry.vocab_size * 4, 8, 1)

    with pytest.raises(CampaignAdapterError, match="gfx1201"):
        ShapeKernelCampaignAdapter(geometry, architecture="gfx1100").candidates()


def test_shape_knob_contract_supports_bounded_nondefault_hypotheses() -> None:
    knob = ShapePatchKnob(
        architecture="gfx1201",
        n=12288,
        k=4096,
        small_k=None,
        split_k=4,
        waves_per_block=8,
        rows_per_wave=2,
        rows_per_block=16,
        vector_load_bytes=16,
        vgpr_limit=192,
        fuse_gate_up=True,
    )

    assert knob.split_k == 4
    assert knob.waves_per_block == 8
    assert knob.rows_per_wave == 2
    assert knob.rows_per_block == 16
    assert knob.vector_load_bytes == 16
    assert knob.vgpr_limit == 192
    assert knob.fuse_gate_up is True

    with pytest.raises(CampaignAdapterError, match="split_k must be a power of two"):
        replace(knob, split_k=3)
    with pytest.raises(CampaignAdapterError, match="waves_per_block must be at most 16"):
        replace(knob, waves_per_block=32)
    with pytest.raises(CampaignAdapterError, match="rows_per_wave must divide"):
        replace(knob, rows_per_wave=32, rows_per_block=16)
    with pytest.raises(CampaignAdapterError, match="vector_load_bytes must be a power"):
        replace(knob, vector_load_bytes=12)
    with pytest.raises(CampaignAdapterError, match="vgpr_limit must be at most 256"):
        replace(knob, vgpr_limit=257)
    with pytest.raises(CampaignAdapterError, match="at least one tuning knob"):
        replace(
            knob,
            split_k=None,
            waves_per_block=None,
            rows_per_wave=None,
            rows_per_block=None,
            vector_load_bytes=None,
            vgpr_limit=None,
            fuse_gate_up=None,
        )
