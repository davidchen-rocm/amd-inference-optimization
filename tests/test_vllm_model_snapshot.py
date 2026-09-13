from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import amd_inference_opt.vllm_model_snapshot as snapshot_module
from amd_inference_opt.vllm_model_snapshot import (
    VLLMModelSnapshotError,
    VLLMModelSnapshotManifest,
    capture_vllm_model_snapshot,
    verify_vllm_model_snapshot,
)


def _capture(root: Path) -> VLLMModelSnapshotManifest:
    return capture_vllm_model_snapshot(
        root,
        model_id="Qwen/Qwen3-8B",
        revision="a" * 40,
        tokenizer_revision="a" * 40,
    )


def test_snapshot_is_deterministic_and_relocatable(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        (root / "config.json").write_text('{"model_type":"qwen3"}\n')
        (root / "model.safetensors").write_bytes(b"weights")

    left = _capture(first)
    right = _capture(second)

    assert left.snapshot_digest == right.snapshot_digest
    assert left.total_bytes == right.total_bytes
    assert [item.relative_path for item in left.files] == [
        "config.json",
        "model.safetensors",
    ]

    blob = tmp_path / "blob"
    blob.write_bytes(b"weights")
    linked = tmp_path / "linked"
    copied = tmp_path / "copied"
    linked.mkdir()
    copied.mkdir()
    (linked / "model.safetensors").symlink_to(blob)
    (copied / "model.safetensors").write_bytes(blob.read_bytes())
    assert _capture(linked).snapshot_digest == _capture(copied).snapshot_digest


def test_snapshot_verification_detects_model_mutation(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    weight = root / "model.safetensors"
    weight.write_bytes(b"original")
    manifest = _capture(root)
    verify_vllm_model_snapshot(manifest)

    weight.write_bytes(b"changed")
    with pytest.raises(VLLMModelSnapshotError, match="no longer matches"):
        verify_vllm_model_snapshot(manifest)


def test_file_symlink_is_hashed_but_directory_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "blob"
    target.write_bytes(b"shared-weights")
    root = tmp_path / "model"
    root.mkdir()
    (root / "model.safetensors").symlink_to(target)

    manifest = _capture(root)
    assert manifest.files[0].storage == "symlink"

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "config.json").write_text("{}")
    (root / "linked-dir").symlink_to(nested, target_is_directory=True)
    with pytest.raises(VLLMModelSnapshotError, match="directory symlinks"):
        _capture(root)

    root_link = tmp_path / "model-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(VLLMModelSnapshotError, match="root must not be a symlink"):
        _capture(root_link)


def test_manifest_rejects_inconsistent_digest(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("{}")
    manifest = _capture(root)

    with pytest.raises(ValidationError, match="inconsistent"):
        VLLMModelSnapshotManifest.model_validate(
            {**manifest.model_dump(by_alias=True), "snapshot_digest": "0" * 64}
        )


def test_capture_tool_refuses_output_symlink(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("{}")
    victim = tmp_path / "victim.json"
    victim.write_text("keep-me")
    output = tmp_path / "manifest.json"
    output.symlink_to(victim)
    tool = Path(__file__).parents[1] / "tools/capture_vllm_model_snapshot.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--model-dir",
            str(root),
            "--model-id",
            "Qwen/Test",
            "--revision",
            "a" * 40,
            "--tokenizer-revision",
            "a" * 40,
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "refusing to replace symlink" in completed.stderr
    assert victim.read_text() == "keep-me"


def test_snapshot_detects_earlier_file_changed_while_later_file_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "model"
    root.mkdir()
    first = root / "a.safetensors"
    first.write_bytes(b"first")
    (root / "b.safetensors").write_bytes(b"second")
    original = snapshot_module._stable_file
    calls = 0

    def mutate_after_first(*, lexical: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            first.write_bytes(b"changed-after-hash")
        return original(lexical=lexical)

    monkeypatch.setattr(snapshot_module, "_stable_file", mutate_after_first)
    with pytest.raises(VLLMModelSnapshotError, match="changed after hashing"):
        _capture(root)
