from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.prepare_mixed_bit_models import (  # noqa: E402
    PreparationError,
    inspect_gguf,
    prepare_mixed_bit_models,
)
from tools.prepare_precision_sweep import prepare_precision_sweep  # noqa: E402


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def write_gguf(
    path: Path,
    *,
    file_type: int | None,
    tensor_types: list[int],
    architecture: str | None = "qwen3",
    general_type: str | None = "model",
) -> None:
    metadata: list[tuple[str, int, object]] = []
    if architecture is not None:
        metadata.append(("general.architecture", 8, architecture))
    if file_type is not None:
        metadata.append(("general.file_type", 4, file_type))
    if general_type is not None:
        metadata.append(("general.type", 8, general_type))
    data = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensor_types), len(metadata)))
    for key, value_type, value in metadata:
        data.extend(_gguf_string(key))
        data.extend(struct.pack("<I", value_type))
        if value_type == 8:
            data.extend(_gguf_string(str(value)))
        else:
            data.extend(struct.pack("<I", int(value)))
    for index, tensor_type in enumerate(tensor_types):
        data.extend(_gguf_string(f"tensor.{index}"))
        data.extend(struct.pack("<IQIQ", 1, 32, tensor_type, index * 32))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def make_bf16(tmp_path: Path) -> Path:
    source = tmp_path / "source-bf16.gguf"
    write_gguf(source, file_type=32, tensor_types=[30, 30, 0])
    return source


def make_tools(tmp_path: Path, *, fail_q4: bool = False) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    writer = r'''
import struct
import sys
from pathlib import Path

def string(value):
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded

def write(path, file_type, tensor_type, general_type="model"):
    metadata = [
        ("general.architecture", 8, "qwen3"),
        ("general.file_type", 4, file_type),
        ("general.type", 8, general_type),
    ]
    types = [tensor_type, tensor_type, 0]
    data = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(types), len(metadata)))
    for key, value_type, value in metadata:
        data.extend(string(key))
        data.extend(struct.pack("<I", value_type))
        data.extend(string(value) if value_type == 8 else struct.pack("<I", value))
    for index, value in enumerate(types):
        data.extend(string(f"tensor.{index}"))
        data.extend(struct.pack("<IQIQ", 1, 32, value, index * 32))
    Path(path).write_bytes(data)
'''
    imatrix = bin_dir / "llama-imatrix"
    imatrix.write_text(
        "#!/usr/bin/env python3\n"
        + writer
        + r'''
assert "--output-format" in sys.argv
assert "--no-ppl" in sys.argv
write(sys.argv[sys.argv.index("-o") + 1], 0, 0, "imatrix")
''',
        encoding="utf-8",
    )
    quantize = bin_dir / "llama-quantize"
    quantize.write_text(
        "#!/usr/bin/env python3\n"
        + writer
        + r'''
assert "--allow-requantize" not in sys.argv
quantization = sys.argv[-2]
if quantization != "Q8_0":
    assert sys.argv[1] == "--imatrix"
if quantization == "Q4_K_M" and FAIL_Q4:
    raise SystemExit(9)
file_types = {"Q4_K_M": 15, "Q5_K_M": 17, "Q6_K": 18, "Q8_0": 7}
tensor_types = {"Q4_K_M": 12, "Q5_K_M": 13, "Q6_K": 14, "Q8_0": 8}
write(sys.argv[-3], file_types[quantization], tensor_types[quantization])
'''.replace("FAIL_Q4", repr(fail_q4)),
        encoding="utf-8",
    )
    imatrix.chmod(0o755)
    quantize.chmod(0o755)
    return bin_dir


def make_calibration(tmp_path: Path) -> Path:
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("fixed calibration corpus\n" * 20, encoding="utf-8")
    return calibration


def test_inspect_gguf_reports_tensor_type_histogram(tmp_path: Path) -> None:
    source = make_bf16(tmp_path)

    inspection = inspect_gguf(source)

    assert inspection["file_type"] == 32
    assert inspection["tensor_count"] == 3
    assert inspection["tensor_type_histogram"] == {"BF16": 2, "F32": 1}


def test_dry_run_is_write_free_and_plans_direct_bf16_quantization(tmp_path: Path) -> None:
    source = make_bf16(tmp_path)
    calibration = make_calibration(tmp_path)
    bin_dir = make_tools(tmp_path)
    output_dir = tmp_path / "output"

    plan = prepare_mixed_bit_models(
        source, calibration, bin_dir, output_dir, dry_run=True
    )

    assert plan["action"] == "dry_run"
    assert plan["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert plan["source"]["gguf"]["file_type"] == 32
    assert plan["toolchain"]["llama_imatrix"]["sha256"]
    for name in ("q5_k_m", "q4_k_m"):
        argv = plan["commands"][name]["argv"]
        assert str(source.resolve()) in argv
        assert "--allow-requantize" not in argv
        assert "--imatrix" in argv
    assert not output_dir.exists()


def test_create_records_provenance_histograms_and_reuses(tmp_path: Path) -> None:
    source = make_bf16(tmp_path)
    calibration = make_calibration(tmp_path)
    bin_dir = make_tools(tmp_path)
    output_dir = tmp_path / "output"

    created = prepare_mixed_bit_models(source, calibration, bin_dir, output_dir)
    manifest_path = output_dir / "preparation.json"
    mtimes = {
        path: path.stat().st_mtime_ns
        for path in (
            output_dir / "imatrix.gguf",
            output_dir / "Q5_K_M.gguf",
            output_dir / "Q4_K_M.gguf",
            manifest_path,
        )
    }
    reused = prepare_mixed_bit_models(source, calibration, bin_dir, output_dir)

    assert created["action"] == "create"
    assert created["provenance"]["no_requantization"] is True
    assert created["provenance"]["original_commands_known"] is True
    assert created["artifacts"]["q5_k_m"]["gguf"]["tensor_type_histogram"] == {
        "F32": 1,
        "Q5_K": 2,
    }
    assert created["artifacts"]["q4_k_m"]["gguf"]["tensor_type_histogram"] == {
        "F32": 1,
        "Q4_K": 2,
    }
    assert created["commands"]["imatrix"]["stdout_sha256"]
    assert created["commands"]["q5_k_m"]["stderr_sha256"]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert reused["action"] == "reuse"
    assert {path: path.stat().st_mtime_ns for path in mtimes} == mtimes


def test_rejects_quantized_source_before_running_tools(tmp_path: Path) -> None:
    source = tmp_path / "q8.gguf"
    write_gguf(source, file_type=7, tensor_types=[8, 8, 0])
    calibration = make_calibration(tmp_path)
    bin_dir = make_tools(tmp_path)

    with pytest.raises(PreparationError, match="refusing requantization"):
        prepare_mixed_bit_models(source, calibration, bin_dir, tmp_path / "output")

    assert not (tmp_path / "output").exists()


def test_adopt_validates_all_artifacts_and_marks_unknown_commands(tmp_path: Path) -> None:
    source = make_bf16(tmp_path)
    calibration = make_calibration(tmp_path)
    bin_dir = make_tools(tmp_path)
    output_dir = tmp_path / "output"
    write_gguf(
        output_dir / "imatrix.gguf",
        file_type=None,
        tensor_types=[0],
        architecture=None,
        general_type="imatrix",
    )
    write_gguf(output_dir / "Q5_K_M.gguf", file_type=17, tensor_types=[13, 13, 0])
    write_gguf(output_dir / "Q4_K_M.gguf", file_type=15, tensor_types=[12, 12, 0])
    before = (output_dir / "Q5_K_M.gguf").stat().st_mtime_ns

    adopted = prepare_mixed_bit_models(
        source,
        calibration,
        bin_dir,
        output_dir,
        adopt_existing=True,
    )

    assert adopted["action"] == "adopt"
    assert adopted["provenance"]["mode"] == "adopt_existing"
    assert adopted["provenance"]["original_commands_known"] is False
    assert adopted["artifacts"]["imatrix"]["gguf"]["general_type"] == "imatrix"
    assert (output_dir / "Q5_K_M.gguf").stat().st_mtime_ns == before
    assert (output_dir / "preparation.json").is_file()


def test_failed_quantization_never_publishes_final_artifacts(tmp_path: Path) -> None:
    source = make_bf16(tmp_path)
    calibration = make_calibration(tmp_path)
    bin_dir = make_tools(tmp_path, fail_q4=True)
    output_dir = tmp_path / "output"

    with pytest.raises(PreparationError, match=r"command failed \(9\)"):
        prepare_mixed_bit_models(source, calibration, bin_dir, output_dir)

    assert not (output_dir / "imatrix.gguf").exists()
    assert not (output_dir / "Q5_K_M.gguf").exists()
    assert not (output_dir / "Q4_K_M.gguf").exists()
    assert not (output_dir / "preparation.json").exists()
    assert (output_dir / "imatrix.gguf.partial").is_file()


def test_precision_sweep_prepares_q8_imatrix_then_q6_and_q5_from_bf16(
    tmp_path: Path,
) -> None:
    source = make_bf16(tmp_path)
    calibration = make_calibration(tmp_path)
    bin_dir = make_tools(tmp_path)
    output_dir = tmp_path / "precision"

    created = prepare_precision_sweep(
        bf16_model=source,
        calibration=calibration,
        llama_bin_dir=bin_dir,
        output_dir=output_dir,
    )
    reused = prepare_precision_sweep(
        bf16_model=source,
        calibration=calibration,
        llama_bin_dir=bin_dir,
        output_dir=output_dir,
    )

    assert created["action"] == "create"
    assert created["quantizations"] == ["Q8_0", "Q6_K", "Q5_K_M"]
    assert created["provenance"]["direct_from_bf16"] is True
    assert created["provenance"]["requantization"] is False
    assert created["artifacts"]["Q8_0"]["gguf"]["file_type"] == 7
    assert created["artifacts"]["Q6_K"]["gguf"]["file_type"] == 18
    assert created["artifacts"]["Q5_K_M"]["gguf"]["file_type"] == 17
    assert created["commands"]["imatrix"]["argv"][2].endswith(
        "Q8_0.gguf.partial"
    )
    assert "--imatrix" in created["commands"]["Q6_K"]["argv"]
    assert "--output-frequency" not in created["commands"]["imatrix"]["argv"]
    assert reused["action"] == "reuse"


def test_precision_sweep_resumes_only_valid_known_partials(tmp_path: Path) -> None:
    source = make_bf16(tmp_path)
    calibration = make_calibration(tmp_path)
    bin_dir = make_tools(tmp_path)
    output_dir = tmp_path / "precision"
    write_gguf(
        output_dir / "Q8_0.gguf.partial",
        file_type=7,
        tensor_types=[8, 8, 0],
    )

    created = prepare_precision_sweep(
        bf16_model=source,
        calibration=calibration,
        llama_bin_dir=bin_dir,
        output_dir=output_dir,
        resume_partials=True,
    )

    assert created["action"] == "create"
    assert created["commands"]["Q8_0"]["execution"] == "reused_valid_partial"
    assert created["commands"]["imatrix"]["stdout_sha256"]
