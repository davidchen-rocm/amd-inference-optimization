from pathlib import Path

from amd_inference_opt.protocol import (
    BASELINE_UNSET_ENVIRONMENT,
    DecodeBenchmarkProtocol,
    KernelTimingProfileProtocol,
    q4_runtime_coordinate,
)


def test_q4_runtime_coordinates_pin_device_libraries_and_clean_baseline(tmp_path: Path) -> None:
    sidecar = tmp_path / "model.q4rdna"
    sidecar.write_bytes(b"sidecar")

    baseline = q4_runtime_coordinate("baseline")
    old = q4_runtime_coordinate("old", sidecar_path=sidecar)
    split = q4_runtime_coordinate("split", sidecar_path=sidecar)

    for coordinate in (baseline, old, split):
        assert coordinate.env["HSA_VISIBLE_DEVICES"] == "0"
        assert coordinate.env["HIP_VISIBLE_DEVICES"] == "0"
        assert coordinate.env["ROCR_VISIBLE_DEVICES"] == "0"
        assert coordinate.env["LD_LIBRARY_PATH"] == ("/opt/rocm/core-7.14/lib:/opt/rocm/lib")
    assert set(baseline.unset_env) == set(BASELINE_UNSET_ENVIRONMENT)
    assert old.env["LLAMA_Q4_RDNA_MAPPING"] == "old"
    assert old.env["LLAMA_Q4_RDNA_SIDECAR"] == str(sidecar.resolve())
    assert "LLAMA_Q4_RDNA_MAPPING" in split.unset_env
    assert "LLAMA_Q4_RDNA_SIDECAR" not in split.unset_env
    assert "LLAMA_Q4_RDNA_GATE_PAIR" in split.unset_env


def test_decode_protocol_freezes_exact_q4_benchmark_and_separates_runtime_env(
    tmp_path: Path,
) -> None:
    common = {
        "llama_bench_path": str(tmp_path / "llama-bench"),
        "model_path": str(tmp_path / "model.gguf"),
        "cwd": str(tmp_path),
    }
    baseline_coordinate = q4_runtime_coordinate("baseline")
    sidecar = tmp_path / "model.q4rdna"
    split_coordinate = q4_runtime_coordinate("split", sidecar_path=sidecar)
    baseline = DecodeBenchmarkProtocol(
        **common,
        environment=baseline_coordinate.env,
        unset_environment=baseline_coordinate.unset_env,
    )
    split = DecodeBenchmarkProtocol(
        **common,
        environment=split_coordinate.env,
        unset_environment=split_coordinate.unset_env,
    )

    argv = baseline.argv
    assert argv[argv.index("-n") + 1] == "128,512"
    assert argv[argv.index("-b") + 1] == "2048"
    assert argv[argv.index("-ub") + 1] == "512"
    assert argv[argv.index("-t") + 1] == "12"
    assert argv[argv.index("-r") + 1] == "3"
    assert argv[argv.index("-mg") + 1] == "0"
    assert argv[argv.index("-dev") + 1] == "ROCm0"
    assert "--no-warmup" not in argv
    assert baseline.argv_hash == split.argv_hash
    assert baseline.protocol_hash == split.protocol_hash
    assert baseline.command().request_sha256(tmp_path) != split.command().request_sha256(tmp_path)


def test_kernel_timing_protocol_is_short_warmed_up_and_keeps_compute_coordinates(
    tmp_path: Path,
) -> None:
    e2e = DecodeBenchmarkProtocol(
        llama_bench_path=str(tmp_path / "llama-bench"),
        model_path=str(tmp_path / "model.gguf"),
        generation_tokens=(128, 512),
        repetitions=3,
        warmup_runs=0,
        batch_size=2048,
        ubatch_size=512,
        threads=12,
        gpu_layers=999,
        cwd=str(tmp_path),
    )

    profile = KernelTimingProfileProtocol.from_e2e(e2e)
    argv = profile.argv

    assert argv[argv.index("-n") + 1] == "128"
    assert argv[argv.index("-r") + 1] == "1"
    assert argv[argv.index("-b") + 1] == "2048"
    assert argv[argv.index("-ub") + 1] == "512"
    assert argv[argv.index("-t") + 1] == "12"
    assert argv[argv.index("-ngl") + 1] == "999"
    assert argv[argv.index("-dev") + 1] == "ROCm0"
    assert "--no-warmup" not in argv
    assert profile.details["profile_command_hash"] == profile.command_hash
    assert profile.protocol_hash != e2e.protocol_hash


def test_decode_semantic_hash_does_not_depend_on_binary_or_model_path(
    tmp_path: Path,
) -> None:
    baseline = DecodeBenchmarkProtocol(
        llama_bench_path=str(tmp_path / "baseline" / "llama-bench"),
        model_path=str(tmp_path / "models" / "model.gguf"),
    )
    candidate = DecodeBenchmarkProtocol(
        llama_bench_path=str(tmp_path / "candidate-worktree" / "llama-bench"),
        model_path=str(tmp_path / "other-coordinate" / "model.gguf"),
    )

    assert baseline.argv_hash != candidate.argv_hash
    assert baseline.protocol_hash == candidate.protocol_hash
