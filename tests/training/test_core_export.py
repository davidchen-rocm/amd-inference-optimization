import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from amd_inference_opt.vllm_model_snapshot import (
    VLLMModelSnapshotError,
    capture_vllm_model_snapshot,
)
from macfit_training.artifacts import finish_artifacts, write_json
from macfit_training.config import validate_job_input
from macfit_training.export import export_merged, extract_adapter, merge_cpu


@pytest.fixture
def completed_job(tmp_path):
    job_dir, model_dir = tmp_path / "job", tmp_path / "base"
    job_dir.mkdir()
    model_dir.mkdir()
    # CPU test data exercises hash binding; these bytes are never treated as model weights.
    (model_dir / "model.safetensors").write_bytes(b"base snapshot fixture")
    request = json.loads(
        (Path(__file__).parents[2] / "examples/training/training-input.json").read_text()
    )
    job = validate_job_input("training", request)
    base = job["base_model"]
    snapshot = capture_vllm_model_snapshot(
        model_dir,
        model_id=base["repo_id"],
        revision=base["revision"],
        tokenizer_revision=base["revision"],
    )
    write_json(job_dir / "input.json", job)
    write_json(job_dir / "model-snapshot.json", snapshot.model_dump(mode="json"))
    adapter = job_dir / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter fixture")
    write_json(
        adapter / "adapter_config.json",
        {
            "base_model_name_or_path": base["repo_id"],
            "revision": base["revision"],
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
        },
    )
    result = {
        "method": "lora_sft",
        "base_model": base,
        "evaluation": {},
        "provenance": {"model_snapshot_sha256": snapshot.snapshot_digest},
    }
    result["artifacts"] = finish_artifacts(job_dir, job, result, adapter_dir=adapter)
    write_json(job_dir / "result.json", result)
    return job_dir, model_dir


def test_export_uses_verified_bundle_and_publishes_manifest_last(
    completed_job, tmp_path, monkeypatch
):
    job_dir, model_dir = completed_job
    output = tmp_path / "merged"
    observed = []

    def merge(base, adapter, destination):
        assert base == model_dir
        assert (adapter / "adapter_model.safetensors").read_bytes() == b"adapter fixture"
        assert (destination / "EXPORT_INCOMPLETE.json").exists()
        assert not (destination / "export-manifest.json").exists()
        observed.append(True)
        (destination / "model.safetensors").write_bytes(b"merged fixture")
        write_json(destination / "config.json", {"model_type": "qwen3"})

    monkeypatch.setattr("macfit_training.export.merge_cpu", merge)
    # The mutable adapter/ directory is not trusted; export reads the hash-bound bundle.
    (job_dir / "adapter/adapter_model.safetensors").write_bytes(b"changed after packaging")
    result = export_merged(job_dir, output)
    assert observed == [True]
    assert result["device"] == "cpu"
    assert len(result["files"]) == 2
    assert not (output / "EXPORT_INCOMPLETE.json").exists()
    assert json.loads((output / "export-manifest.json").read_text()) == result
    with pytest.raises(ValueError, match="new output"):
        export_merged(job_dir, output)
    assert observed == [True]


@pytest.mark.parametrize("target", ["artifact", "manifest", "snapshot", "input"])
def test_export_rejects_tampering_or_symlink_before_loading_weights(
    completed_job, tmp_path, monkeypatch, target
):
    job_dir, model_dir = completed_job
    if target == "artifact":
        (job_dir / "artifacts/adapter.tar.gz").write_bytes(b"modified")
    elif target == "snapshot":
        (model_dir / "model.safetensors").write_bytes(b"modified base")
    else:
        path = job_dir / ("artifacts/manifest.json" if target == "manifest" else "input.json")
        destination = tmp_path / "linked-source"
        path.rename(destination)
        path.symlink_to(destination)
    monkeypatch.setattr(
        "macfit_training.export.merge_cpu", lambda *_args: pytest.fail("Must fail before loading")
    )
    with pytest.raises((ValueError, VLLMModelSnapshotError)):
        export_merged(job_dir, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_export_rejects_unsafe_output_and_leaves_failed_export_incomplete(
    completed_job, tmp_path, monkeypatch
):
    job_dir, model_dir = completed_job
    link = tmp_path / "linked-parent"
    link.symlink_to(tmp_path, target_is_directory=True)
    for output in (job_dir / "output", model_dir / "output", link / "output"):
        with pytest.raises(ValueError):
            export_merged(job_dir, output)

    def failure(*_args):
        raise RuntimeError("Simulated interrupted merge")

    monkeypatch.setattr("macfit_training.export.merge_cpu", failure)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="interrupted"):
        export_merged(job_dir, output)
    assert (output / "EXPORT_INCOMPLETE.json").exists()
    assert not (output / "export-manifest.json").exists()
    with pytest.raises(ValueError, match="new output"):
        export_merged(job_dir, output)


@pytest.mark.parametrize("name,is_link", [("../escape", False), ("adapter/link", True)])
def test_archive_traversal_and_links_never_extract(tmp_path, name, is_link):
    archive_path = tmp_path / "adapter.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        if is_link:
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
        else:
            member.size = 4
        archive.addfile(member, None if is_link else io.BytesIO(b"data"))
    destination = tmp_path / "extract"
    destination.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        extract_adapter(archive_path, destination, {})
    assert list(destination.iterdir()) == []


def test_cpu_merge_disables_network_and_adapter_auto_device_selection(tmp_path, monkeypatch):
    calls = {}

    class Model:
        @classmethod
        def from_pretrained(cls, path, **options):
            calls["base"] = options
            return cls()

        def save_pretrained(self, output, **options):
            assert options["safe_serialization"] is True
            calls["saved"] = output

    class Adapter:
        @classmethod
        def from_pretrained(cls, base, path, **options):
            calls["adapter"] = options
            return cls()

        def merge_and_unload(self, **options):
            assert options["safe_merge"] is True
            return Model()

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, path, **options):
            calls["tokenizer"] = options
            return cls()

        def save_pretrained(self, output):
            assert calls["saved"] == output

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=Adapter))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=Model, AutoTokenizer=Tokenizer),
    )
    merge_cpu(tmp_path / "base", tmp_path / "adapter", tmp_path / "output")
    assert calls["base"]["device_map"] == "cpu"
    assert calls["adapter"]["torch_device"] == "cpu"
    assert calls["adapter"]["device_map"] == {"": "cpu"}
    assert all(calls[k]["local_files_only"] is True for k in ("base", "adapter", "tokenizer"))
    assert calls["base"]["trust_remote_code"] is False
    assert calls["tokenizer"]["trust_remote_code"] is False
