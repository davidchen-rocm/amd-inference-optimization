from __future__ import annotations

import pytest

from amd_inference_opt.kv_cache import (
    KV_CACHE_DEPTHS,
    QWEN3_8B_KV_BYTES,
    KVCacheError,
    KVCacheType,
    build_kv_campaign_plan,
)

QWEN_GEOMETRY = {
    "block_count": 36,
    "attention_head_count_kv": 8,
    "attention_key_length": 128,
    "context_length": 32768,
}
CAPABILITIES = {
    "cache_types": ["f16", "q8_0", "q4_0"],
    "flash_attn": True,
    "max_context_tokens": 32768,
    "minimum_depth": 1,
    "device_selector": "ROCm0",
}


def test_campaign_plan_derives_all_arms_depths_bytes_and_minimal_canaries() -> None:
    plan = build_kv_campaign_plan(QWEN_GEOMETRY, CAPABILITIES)

    assert [arm.cache_type for arm in plan.arms] == list(KVCacheType)
    assert plan.benchmark_depths == KV_CACHE_DEPTHS
    assert plan.canary_depth == 1
    assert plan.effective_context_tokens == 32768
    for arm in plan.arms:
        assert arm.arm.type_k == arm.arm.type_v == arm.cache_type
        assert arm.arm.flash_attn
        assert arm.canary.depth == 1
        assert arm.canary.generation_tokens == 1
        assert arm.canary.repetitions == 1
        assert arm.scored.depths == KV_CACHE_DEPTHS
        assert arm.scored.generation_tokens == 128
        assert arm.scored.repetitions == 5
        assert arm.scored.noisy_retry_repetitions == 12
        assert arm.kv_bytes_by_depth == QWEN3_8B_KV_BYTES[arm.cache_type]

    document = plan.to_dict()
    assert document["schema"] == "gpuopt.kv-cache-campaign-plan.v1"
    assert document["plan_hash"] == plan.semantic_hash
    assert len(str(document["plan_hash"])) == 64
    assert document["arms"][0]["canary"]["depth"] == 1  # type: ignore[index]


def test_campaign_plan_intersects_model_and_runtime_context() -> None:
    plan = build_kv_campaign_plan(
        QWEN_GEOMETRY,
        {**CAPABILITIES, "max_context_tokens": 30000},
    )
    assert plan.effective_context_tokens == 30000

    with pytest.raises(KVCacheError, match="d28672"):
        build_kv_campaign_plan(
            QWEN_GEOMETRY,
            {**CAPABILITIES, "max_context_tokens": 28799},
        )


def test_campaign_plan_requires_all_cache_types_and_flash_attention() -> None:
    with pytest.raises(KVCacheError, match="q4_0"):
        build_kv_campaign_plan(
            QWEN_GEOMETRY,
            {**CAPABILITIES, "cache_types": ["f16", "q8_0"]},
        )
    with pytest.raises(KVCacheError, match="flash-attention"):
        build_kv_campaign_plan(
            QWEN_GEOMETRY,
            {**CAPABILITIES, "flash_attn": False},
        )
