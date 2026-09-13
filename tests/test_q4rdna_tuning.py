from pathlib import Path

import pytest

from amd_inference_opt.q4rdna_tuning import (
    GateQ3RuntimeMeasurement,
    GateUpBenchmarkMeasurement,
    GateUpPairFinalEvidence,
    GateUpPairFinalPolicy,
    GateUpPairRuntimeMeasurement,
    MappingScreenOutcome,
    Q4RDNAFormat,
    Q4RDNATuningError,
    build_gate_up_performance_model,
    drop_initial_warmup_samples,
    evaluate_gate_up_pair_final,
    gate_up_mapping_arm,
    normalize_llama_cli_generation,
    screen_gate_q3_microbenchmark,
    screen_gate_q3_runtime,
    screen_gate_up_mapping,
    screen_gate_up_pair_load,
    screen_gate_up_pair_runtime,
)


def measurement(
    split_waves: int,
    samples: tuple[float, ...],
    *,
    active: bool = True,
) -> GateUpBenchmarkMeasurement:
    return GateUpBenchmarkMeasurement(
        arm_id=f"gate-up-split-{split_waves}",
        split_waves=split_waves,
        samples_tokens_per_second=samples,
        activation_verified=active,
    )


def test_q4rdna_packed_bytes_and_gate_up_limit_model() -> None:
    representation = Q4RDNAFormat()
    assert representation.packed_matrix_bytes(rows=12288, columns=4096) == 26_738_688

    model = build_gate_up_performance_model(
        measured_kernel_us=89.8410385443583,
        hotspot_gpu_time_share_percent=40.597033,
        theoretical_bandwidth_gbps=640.0,
    )
    assert model.packed_weight_bytes == 53_477_376
    assert model.activation_bytes == 16_384
    assert model.output_bytes == 49_152
    assert model.theoretical_kernel_us == pytest.approx(83.6608)
    assert model.bandwidth_efficiency_percent == pytest.approx(93.121, rel=1e-3)
    assert 2.8 < model.maximum_e2e_improvement_percent < 3.0


def test_gate_up_arm_locks_hotspot_and_unsets_old_mapping(tmp_path: Path) -> None:
    sidecar = tmp_path / "model.q4rdna"
    sidecar.write_bytes(b"sidecar")
    arm = gate_up_mapping_arm(4, sidecar_path=sidecar)
    assert arm.id == "gate-up-split-4"
    assert arm.environment["LLAMA_Q4_RDNA_SCOPE"] == "hotspot"
    assert arm.environment["LLAMA_Q4_RDNA_COOP"] == "4"
    assert arm.environment["LLAMA_Q4_RDNA_TRACE"] == "1"
    assert "LLAMA_Q4_RDNA_MAPPING" in arm.unset_environment
    assert "LLAMA_Q4_RDNA_SIDECAR" not in arm.unset_environment


def test_gate_up_arm_rejects_unsupported_width(tmp_path: Path) -> None:
    sidecar = tmp_path / "model.q4rdna"
    sidecar.write_bytes(b"sidecar")
    with pytest.raises(Q4RDNATuningError, match="2, 4, or 8"):
        gate_up_mapping_arm(16, sidecar_path=sidecar)


def test_gate_up_screen_promotes_only_stable_material_candidate() -> None:
    baseline = measurement(8, (100.0, 100.2, 99.8, 100.1, 99.9))
    promoted = screen_gate_up_mapping(
        baseline,
        measurement(4, (101.0, 101.2, 100.8, 101.1, 100.9)),
    )
    assert promoted.outcome == MappingScreenOutcome.PROMOTE
    assert promoted.improvement_percent == pytest.approx(1.0)

    rejected = screen_gate_up_mapping(
        baseline,
        measurement(2, (99.0, 99.2, 98.8, 99.1, 98.9)),
    )
    assert rejected.outcome == MappingScreenOutcome.REJECT


def test_gate_up_screen_marks_noise_and_missing_activation_inconclusive() -> None:
    baseline = measurement(8, (100.0, 100.1, 99.9))
    noisy = screen_gate_up_mapping(
        baseline,
        measurement(4, (90.0, 110.0, 100.0)),
    )
    assert noisy.outcome == MappingScreenOutcome.INCONCLUSIVE

    inactive = screen_gate_up_mapping(
        baseline,
        measurement(4, (101.0, 101.1, 100.9), active=False),
    )
    assert inactive.outcome == MappingScreenOutcome.INCONCLUSIVE


def test_explicit_same_coordinate_warmup_is_not_scored() -> None:
    assert drop_initial_warmup_samples(
        (87.0, 96.0, 96.1, 95.9),
        warmup_samples=1,
    ) == (96.0, 96.1, 95.9)
    with pytest.raises(Q4RDNATuningError, match="consume every"):
        drop_initial_warmup_samples((100.0,), warmup_samples=1)


def test_pair_load_screen_requires_correct_stable_material_result() -> None:
    evidence = {
        "max_absolute_error": 0.0,
        "separate_cv_percent": 0.06,
        "paired_cv_percent": 0.06,
        "paired_improvement_percent": 2.52,
    }
    assert screen_gate_up_pair_load(evidence).outcome == MappingScreenOutcome.PROMOTE

    evidence["max_absolute_error"] = 0.01
    assert screen_gate_up_pair_load(evidence).outcome == MappingScreenOutcome.REJECT
    evidence["max_absolute_error"] = 0.0
    evidence["paired_cv_percent"] = 2.0
    assert screen_gate_up_pair_load(evidence).outcome == MappingScreenOutcome.INCONCLUSIVE


def test_pair_runtime_screen_promotes_stable_real_model_gain() -> None:
    baseline = GateUpPairRuntimeMeasurement(
        arm_id="baseline",
        paired_layout=False,
        samples_tokens_per_second=(100.0, 100.1, 99.9),
        activation_verified=True,
        paired_tensor_count=0,
    )
    candidate = GateUpPairRuntimeMeasurement(
        arm_id="candidate",
        paired_layout=True,
        samples_tokens_per_second=(101.0, 101.1, 100.9),
        activation_verified=True,
        paired_tensor_count=36,
    )

    screen = screen_gate_up_pair_runtime(baseline, candidate)

    assert screen.outcome == MappingScreenOutcome.PROMOTE
    assert screen.improvement_percent == pytest.approx(1.0)


def test_pair_runtime_screen_rejects_unprofitable_candidate() -> None:
    baseline = GateUpPairRuntimeMeasurement(
        arm_id="baseline",
        paired_layout=False,
        samples_tokens_per_second=(100.0, 100.1, 99.9),
        activation_verified=True,
        paired_tensor_count=0,
    )
    candidate = GateUpPairRuntimeMeasurement(
        arm_id="candidate",
        paired_layout=True,
        samples_tokens_per_second=(99.9, 100.0, 100.1),
        activation_verified=True,
        paired_tensor_count=36,
    )

    screen = screen_gate_up_pair_runtime(baseline, candidate)

    assert screen.outcome == MappingScreenOutcome.REJECT


def test_cli_generation_normalizer_ignores_only_timing_footer() -> None:
    first = "\nanswer tokens\n\n[ Prompt: 696.9 t/s | Generation: 86.0 t/s ]\n\n\nExiting...\n"
    second = "\nanswer tokens\n\n[ Prompt: 698.0 t/s | Generation: 85.9 t/s ]\n\nExiting...\n"

    assert normalize_llama_cli_generation(first) == "\nanswer tokens"
    assert normalize_llama_cli_generation(first) == normalize_llama_cli_generation(second)


def test_final_pair_gate_rejects_material_vram_cost() -> None:
    evidence = GateUpPairFinalEvidence(
        tg128_improvement_percent=0.96,
        tg512_improvement_percent=1.10,
        baseline_kernel_average_ns=102_655,
        candidate_kernel_average_ns=101_347,
        extra_vram_bytes=1_925_185_536,
        total_vram_bytes=17_095_983_104,
        correctness_passed=True,
        profiler_complete=True,
    )
    result = evaluate_gate_up_pair_final(
        evidence,
        GateUpPairFinalPolicy(
            minimum_tg128_improvement_percent=0.5,
            minimum_tg512_improvement_percent=0.5,
            maximum_extra_vram_percent=5.0,
        ),
    )

    assert result.outcome == MappingScreenOutcome.REJECT
    assert result.checks["extra_vram_within_budget"] is False
    assert result.metrics["extra_vram_percent"] == pytest.approx(11.26, rel=1e-3)


def test_gate_q3_microbenchmark_requires_speed_and_exact_decode() -> None:
    evidence = {
        "max_absolute_error": 0.0,
        "q4_cv_percent": 0.2,
        "q3_cv_percent": 0.3,
        "q3_improvement_percent": 4.0,
        "fused_weight_byte_reduction_percent": 11.76,
    }
    assert screen_gate_q3_microbenchmark(evidence).outcome == MappingScreenOutcome.PROMOTE
    evidence["q3_improvement_percent"] = -1.0
    assert screen_gate_q3_microbenchmark(evidence).outcome == MappingScreenOutcome.REJECT


def test_gate_q3_runtime_requires_stable_material_gain() -> None:
    baseline = GateQ3RuntimeMeasurement(
        arm_id="q4-gate",
        q3_gate=False,
        samples_tokens_per_second=(100.0, 100.1, 99.9),
        activation_verified=True,
        q3_tensor_count=0,
    )
    candidate = GateQ3RuntimeMeasurement(
        arm_id="q3-gate",
        q3_gate=True,
        samples_tokens_per_second=(103.0, 103.1, 102.9),
        activation_verified=True,
        q3_tensor_count=36,
    )
    assert screen_gate_q3_runtime(baseline, candidate).outcome == MappingScreenOutcome.PROMOTE

    noisy = GateQ3RuntimeMeasurement(
        arm_id="q3-gate",
        q3_gate=True,
        samples_tokens_per_second=(90.0, 110.0, 103.0),
        activation_verified=True,
        q3_tensor_count=36,
    )
    assert screen_gate_q3_runtime(baseline, noisy).outcome == MappingScreenOutcome.INCONCLUSIVE
