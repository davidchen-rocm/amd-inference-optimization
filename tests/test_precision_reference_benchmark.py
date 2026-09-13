import sys
from pathlib import Path

import pytest

from amd_inference_opt.llama_cpp import LlamaBenchRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_precision_reference_benchmark import (  # noqa: E402
    PrecisionBenchmarkError,
    _drop_warmup_samples,
    _parse_runtime_libraries,
)


def test_precision_benchmark_excludes_same_coordinate_warmup_samples() -> None:
    record = LlamaBenchRecord(
        phase="prefill",
        prompt_tokens=512,
        generation_tokens=0,
        mean_tokens_per_second=30.0,
        stddev_tokens_per_second=10.0,
        samples_tokens_per_second=(10.0, 20.0, 30.0, 40.0, 42.0),
        coefficient_of_variation=1 / 3,
        raw={"n_prompt": 512, "n_gen": 0},
    )

    scored = _drop_warmup_samples(record, warmup_samples=3)

    assert scored.samples_tokens_per_second == (40.0, 42.0)
    assert scored.mean_tokens_per_second == 41.0
    assert scored.raw["gpuopt_warmup_samples_dropped"] == 3
    assert scored.raw["gpuopt_scored_samples_ts"] == [40.0, 42.0]


def test_runtime_library_closure_hashes_local_dependencies(tmp_path: Path) -> None:
    hip = tmp_path / "libggml-hip.so.0"
    llama = tmp_path / "libllama.so.0"
    hip.write_bytes(b"hip")
    llama.write_bytes(b"llama")

    result = _parse_runtime_libraries(
        f"libggml-hip.so.0 => {hip} (0x1)\n"
        f"libllama.so.0 => {llama} (0x2)\n",
        expected_dir=tmp_path,
    )

    assert result["status"] == "verified"
    assert len(result["closure_sha256"]) == 64
    libraries = result["libraries"]
    assert libraries["libggml-hip.so.0"]["sha256"] != libraries[
        "libllama.so.0"
    ]["sha256"]


def test_runtime_library_closure_rejects_pollution(tmp_path: Path) -> None:
    outside = tmp_path.parent / "libggml-hip.so.0"
    local = tmp_path / "libllama.so.0"
    outside.write_bytes(b"hip")
    local.write_bytes(b"llama")

    with pytest.raises(PrecisionBenchmarkError, match="pollution"):
        _parse_runtime_libraries(
            f"libggml-hip.so.0 => {outside} (0x1)\n"
            f"libllama.so.0 => {local} (0x2)\n",
            expected_dir=tmp_path,
        )
