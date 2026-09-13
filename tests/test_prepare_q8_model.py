from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from amd_inference_opt.models import OptimizationTask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.prepare_q8_model import (  # noqa: E402
    PreparationError,
    build_input_manifest,
    prepare,
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def make_llama_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "llama.cpp"
    repo.mkdir()
    (repo / "CMakeLists.txt").write_text("project(fake_llama)\n", encoding="utf-8")
    (repo / "convert_hf_to_gguf.py").write_text(
        """\
import argparse
import struct
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--outfile", required=True)
parser.add_argument("--outtype", required=True)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
assert args.outtype == "q8_0"
if args.dry_run:
    print("fake total_size = 8.7G")
else:
    def string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    data = b"GGUF" + struct.pack("<IQQ", 3, 1, 2)
    data += string("general.architecture") + struct.pack("<I", 8) + string("qwen3")
    data += string("general.file_type") + struct.pack("<II", 4, 7)
    Path(args.outfile).write_bytes(data)
""",
        encoding="utf-8",
    )
    git(repo, "init", "-q")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "add", ".")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    return repo


def make_model(tmp_path: Path) -> Path:
    model = tmp_path / "hf-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"source-weights")
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 14},
                "weight_map": {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    return model


def test_input_manifest_is_sorted_and_content_addressed(tmp_path: Path) -> None:
    model = make_model(tmp_path)

    first = build_input_manifest(model.resolve())
    second = build_input_manifest(model.resolve())

    assert first == second
    assert [item["path"] for item in first["files"]] == sorted(
        item["path"] for item in first["files"]
    )
    content = {key: value for key, value in first.items() if key != "sha256"}
    expected = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert first["sha256"] == expected


def test_dry_run_validates_converter_without_writing(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    repo = make_llama_repo(tmp_path)
    output = tmp_path / "new-output" / "model.gguf"

    result = prepare(
        model,
        repo,
        output,
        python_executable=Path(sys.executable),
        dry_run=True,
    )

    assert result["action"] == "dry_run"
    assert result["converter"]["commit"] == git(repo, "rev-parse", "HEAD")
    assert result["conversion"]["dry_run_argv"][-1] == "--dry-run"
    assert not output.parent.exists()
    assert not Path(result["output"]["manifest_path"]).exists()


def test_create_records_provenance_and_is_idempotent(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    repo = make_llama_repo(tmp_path)
    output = tmp_path / "output" / "model.gguf"

    created = prepare(model, repo, output, python_executable=Path(sys.executable))
    manifest_path = Path(created["output"]["manifest_path"])
    output_mtime = output.stat().st_mtime_ns
    manifest_mtime = manifest_path.stat().st_mtime_ns
    reused = prepare(model, repo, output, python_executable=Path(sys.executable))

    assert created["action"] == "create"
    output_bytes = output.read_bytes()
    assert output_bytes.startswith(b"GGUF")
    assert created["output"]["sha256"] == hashlib.sha256(output_bytes).hexdigest()
    assert created["conversion"]["argv"][4] == str(output) + ".partial"
    assert created["converter"]["script_sha256"]
    assert created["input_manifest"]["sha256"]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert reused["action"] == "reuse"
    assert output.stat().st_mtime_ns == output_mtime
    assert manifest_path.stat().st_mtime_ns == manifest_mtime


def test_existing_result_is_never_reused_after_input_changes(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    repo = make_llama_repo(tmp_path)
    output = tmp_path / "output" / "model.gguf"
    prepare(model, repo, output, python_executable=Path(sys.executable))
    original = output.read_bytes()
    (model / "model-00001-of-00001.safetensors").write_bytes(b"changed-weights")

    with pytest.raises(PreparationError, match="input_manifest_sha256"):
        prepare(model, repo, output, python_executable=Path(sys.executable))

    assert output.read_bytes() == original


def test_refuses_unpaired_output_or_manifest(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    repo = make_llama_repo(tmp_path)
    output = tmp_path / "output" / "model.gguf"
    output.parent.mkdir()
    output.write_bytes(b"unowned")

    with pytest.raises(PreparationError, match="pass --adopt-existing"):
        prepare(model, repo, output, python_executable=Path(sys.executable))


def test_adopt_existing_validates_and_writes_only_manifest(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    repo = make_llama_repo(tmp_path)
    output = tmp_path / "output" / "model.gguf"
    output.parent.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(repo / "convert_hf_to_gguf.py"),
            str(model),
            "--outfile",
            str(output),
            "--outtype",
            "q8_0",
        ],
        check=True,
    )
    output_mtime = output.stat().st_mtime_ns

    adopted = prepare(
        model,
        repo,
        output,
        python_executable=Path(sys.executable),
        adopt_existing=True,
    )

    assert adopted["action"] == "adopt"
    assert adopted["provenance"]["mode"] == "adopt_existing"
    assert adopted["provenance"]["original_conversion_argv_known"] is False
    assert adopted["provenance"]["validation"]["gguf"]["file_type"] == 7
    assert output.stat().st_mtime_ns == output_mtime
    assert Path(adopted["output"]["manifest_path"]).is_file()


def test_q8_example_pins_hash_and_fixed_dual_decode_gate() -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "examples" / "qwen3-8b-q8-live.yaml").read_text(encoding="utf-8")
    )
    task = OptimizationTask.model_validate(payload)

    assert task.model.quantization == "Q8_0"
    assert task.campaign_kind == "llama_cpp_q8"
    assert task.model.sha256 == "6d205720f8a41e3156d7fce154ad9bad49c474b501253040efb5a11af6600df0"
    assert task.benchmark.required_metrics == [
        "tokens_per_second_tg128",
        "tokens_per_second_tg512",
    ]
    assert [item.minimum_improvement_percent for item in task.objective.metric_requirements] == [
        10.0,
        10.0,
    ]
