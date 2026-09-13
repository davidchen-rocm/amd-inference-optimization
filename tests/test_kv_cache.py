from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from amd_inference_opt.kv_cache import (
    KV_CACHE_DEPTHS,
    QWEN3_8B_KV_BYTES,
    KVCacheArm,
    KVCacheBenchmarkProtocol,
    KVCacheError,
    KVCacheType,
    evaluate_kv_cache_gate,
    parse_kv_cache_benchmark_json,
    qwen3_8b_kv_cache_bytes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_kv_cache_campaign import _drop_same_coordinate_warmup  # noqa: E402


def _rows(cache_type: KVCacheType, means: tuple[float, float, float]) -> list[dict[str, object]]:
    return [
        {
            "n_prompt": 0,
            "n_gen": 128,
            "n_depth": depth,
            "type_k": cache_type.value,
            "type_v": cache_type.value,
            "flash_attn": 1,
            "avg_ts": mean,
            "stddev_ts": 0.0,
            "samples_ts": [mean, mean, mean, mean, mean],
        }
        for depth, mean in zip(KV_CACHE_DEPTHS, means, strict=True)
    ]


def _result(cache_type: KVCacheType, means: tuple[float, float, float]):
    return parse_kv_cache_benchmark_json(
        _rows(cache_type, means),
        arm=KVCacheArm(cache_type),
    )


def test_cache_types_and_arm_force_matching_flash_attention() -> None:
    assert [cache_type.value for cache_type in KVCacheType] == ["f16", "q8_0", "q4_0"]
    arm = KVCacheArm(KVCacheType.Q8_0)

    assert arm.type_k == arm.type_v == KVCacheType.Q8_0
    with pytest.raises(KVCacheError, match="flash attention"):
        KVCacheArm(KVCacheType.Q8_0, flash_attn=False)


def test_protocol_freezes_model_identity_and_rocm0_long_context_coordinate(
    tmp_path: Path,
) -> None:
    common = {
        "arm": KVCacheArm(KVCacheType.Q4_0),
        "llama_bench_path": str(tmp_path / "a" / "llama-bench"),
        "model_path": str(tmp_path / "models" / "qwen3-q6.gguf"),
    }
    protocol = KVCacheBenchmarkProtocol(**common)
    moved = KVCacheBenchmarkProtocol(
        **{
            **common,
            "llama_bench_path": str(tmp_path / "b" / "llama-bench"),
            "model_path": str(tmp_path / "elsewhere" / "qwen3-q6.gguf"),
        }
    )
    other_arm = replace(protocol, arm=KVCacheArm(KVCacheType.Q8_0))
    argv = protocol.argv

    assert argv[argv.index("-d") + 1] == "4096,16384,28672"
    assert argv[argv.index("-n") + 1] == "128"
    assert argv[argv.index("-r") + 1] == "5"
    assert argv[argv.index("-ctk") + 1] == "q4_0"
    assert argv[argv.index("-ctv") + 1] == "q4_0"
    assert argv[argv.index("-fa") + 1] == "on"
    assert argv[argv.index("-dev") + 1] == "ROCm0"
    assert protocol.semantic_hash == moved.semantic_hash
    assert protocol.argv_hash != moved.argv_hash
    assert protocol.semantic_hash != other_arm.semantic_hash
    mixed_model = replace(protocol, model_quantization="Q6_K/Q8_0 mixed 7.11 bpw")
    assert mixed_model.semantic_hash != protocol.semantic_hash

    extended = replace(protocol, repetitions=12)
    assert extended.argv[extended.argv.index("-r") + 1] == "12"
    with pytest.raises(KVCacheError, match="five samples or a 12-sample retry"):
        replace(protocol, repetitions=6)
    with pytest.raises(KVCacheError, match="model_quantization"):
        replace(protocol, model_quantization="")


def test_kv_runner_excludes_same_coordinate_warmup_samples() -> None:
    rows = _rows(KVCacheType.F16, (100.0, 90.0, 80.0))
    for index, row in enumerate(rows):
        row["samples_ts"] = [10.0, 20.0, 30.0, 40.0, 50.0] + [
            100.0 + index,
            101.0 + index,
            102.0 + index,
            103.0 + index,
            104.0 + index,
        ]

    scored = _drop_same_coordinate_warmup(rows, scored_repetitions=5)

    assert scored[0]["samples_ts"] == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert scored[0]["avg_ts"] == 102.0
    assert scored[0]["gpuopt_warmup_samples_dropped"] == 5
    assert len(scored[0]["gpuopt_source_samples_ts"]) == 10


def test_strict_parser_validates_all_depths_coordinates_samples_and_cv() -> None:
    arm = KVCacheArm(KVCacheType.Q8_0)
    rows = _rows(KVCacheType.Q8_0, (100.0, 95.0, 90.0))
    parsed = parse_kv_cache_benchmark_json(json.dumps(rows), arm=arm)

    assert tuple(parsed.by_depth()) == KV_CACHE_DEPTHS
    assert parsed.by_depth()[16384].sample_count == 5
    assert parsed.max_cv_percent == 0

    missing = rows[:-1]
    with pytest.raises(KVCacheError, match="exactly 3 rows"):
        parse_kv_cache_benchmark_json(missing, arm=arm)

    for field, value, message in (
        ("n_depth", 123, "n_depth"),
        ("type_k", "f16", "type_k"),
        ("type_v", "q4_0", "type_v"),
        ("flash_attn", -1, "flash attention"),
        ("n_gen", 64, "n_gen"),
    ):
        invalid = [dict(row) for row in rows]
        invalid[0][field] = value
        with pytest.raises(KVCacheError, match=message):
            parse_kv_cache_benchmark_json(invalid, arm=arm)

    wrong_samples = [dict(row) for row in rows]
    wrong_samples[0]["samples_ts"] = [100.0] * 4
    with pytest.raises(KVCacheError, match="exactly 5 samples"):
        parse_kv_cache_benchmark_json(wrong_samples, arm=arm)

    noisy = [dict(row) for row in rows]
    noisy[0]["avg_ts"] = 100.0
    noisy[0]["samples_ts"] = [96.0, 98.0, 100.0, 102.0, 104.0]
    with pytest.raises(KVCacheError, match="CV"):
        parse_kv_cache_benchmark_json(noisy, arm=arm)

    extended = _rows(KVCacheType.Q8_0, (100.0, 95.0, 90.0))
    for row in extended:
        row["samples_ts"] = [row["avg_ts"]] * 12
    parsed_extended = parse_kv_cache_benchmark_json(
        extended,
        arm=arm,
        repetitions=12,
    )
    assert all(record.sample_count == 12 for record in parsed_extended.records)


def test_qwen3_8b_kv_byte_counts_include_packed_block_overhead() -> None:
    expected = {
        KVCacheType.F16: (603_979_776, 2_415_919_104, 4_227_858_432),
        KVCacheType.Q8_0: (320_864_256, 1_283_457_024, 2_246_049_792),
        KVCacheType.Q4_0: (169_869_312, 679_477_248, 1_189_085_184),
    }

    for cache_type, byte_counts in expected.items():
        assert tuple(QWEN3_8B_KV_BYTES[cache_type].values()) == byte_counts
        assert tuple(
            qwen3_8b_kv_cache_bytes(depth, cache_type) for depth in KV_CACHE_DEPTHS
        ) == byte_counts


def test_gate_requires_bounded_short_context_nonregressing_long_and_one_5_percent() -> None:
    baseline = _result(KVCacheType.F16, (100.0, 100.0, 100.0))
    accepted = _result(KVCacheType.Q8_0, (97.0, 105.0, 100.0))
    no_long_win = _result(KVCacheType.Q4_0, (100.0, 104.9, 104.9))
    long_regression = _result(KVCacheType.Q4_0, (100.0, 106.0, 99.9))

    decision = evaluate_kv_cache_gate(baseline, accepted)
    assert decision.passed
    assert decision.improvements_percent[4096] == pytest.approx(-3)
    assert decision.improvements_percent[16384] == pytest.approx(5)
    assert not evaluate_kv_cache_gate(baseline, no_long_win).passed
    assert not evaluate_kv_cache_gate(baseline, long_regression).passed

    noisy_record = replace(accepted.records[0], coefficient_of_variation_percent=2.01)
    noisy = replace(accepted, records=(noisy_record, *accepted.records[1:]))
    noisy_decision = evaluate_kv_cache_gate(baseline, noisy)
    assert not noisy_decision.passed
    assert not noisy_decision.checks["cv_at_most_2_percent"]
