"""Offline CPU merge of a completed, hash-verified LoRA job into a new directory."""

from __future__ import annotations

import json
import os
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import canonical_sha256, describe_artifact, write_json
from .config import validate_job_input

MAX_JSON = 8 * 1024 * 1024
MAX_ADAPTER = 1024**3
ARTIFACT_TYPES = {
    "resolved-config.json": "configuration",
    "adapter.tar.gz": "lora_adapter",
    "evaluation.json": "evaluation",
    "manifest.json": "manifest",
}
ARTIFACT_NAMES = set(ARTIFACT_TYPES)
ADAPTER_NAMES = {"adapter_config.json", "adapter_model.safetensors", "README.md", "USAGE.md"}


def real_path(path: Path) -> Path:
    """Reject links in existing path components rather than resolving through them."""
    path = path.expanduser().absolute()
    if ".." in path.parts:
        raise ValueError("Export paths cannot contain parent traversal.")
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("Export paths must not contain symbolic links.")
    return path


def read_json(path: Path) -> dict[str, Any]:
    path = real_path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_JSON:
            raise ValueError("Export metadata must be a bounded, private regular JSON file.")
        data = stream.read(MAX_JSON + 1)
    if len(data) > MAX_JSON:
        raise ValueError("Export metadata exceeded its size limit.")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Export metadata must be a JSON object.")
    return value


def verified_job(job_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    job_dir = real_path(job_dir)
    job = validate_job_input("training", read_json(job_dir / "input.json"))
    result = read_json(job_dir / "result.json")
    if result.get("method") != "lora_sft" or result.get("base_model") != job["base_model"]:
        raise ValueError("A completed LoRA training result with the pinned base model is required.")
    declared = result.get("artifacts")
    if not isinstance(declared, list) or len(declared) != len(ARTIFACT_NAMES):
        raise ValueError("The completed job's artifact inventory is invalid.")
    names = [row.get("name") for row in declared if isinstance(row, dict)]
    if len(names) != len(declared) or any(not isinstance(name, str) for name in names):
        raise ValueError("The completed job's artifact inventory is invalid.")
    if set(names) != ARTIFACT_NAMES:
        raise ValueError("The completed job contains missing or unexpected artifacts.")
    artifact_dir = real_path(job_dir / "artifacts")
    if {path.name for path in artifact_dir.iterdir()} != ARTIFACT_NAMES:
        raise ValueError("The completed job contains unexpected artifact files.")
    for row in declared:
        path = real_path(artifact_dir / row["name"])
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_ADAPTER:
            raise ValueError("An artifact is not a bounded private regular file.")
        observed = describe_artifact(path, ARTIFACT_TYPES[row["name"]])
        if observed != row:
            raise ValueError("The completed job failed artifact hash verification.")
    manifest = read_json(artifact_dir / "manifest.json")
    expected_records = [row for row in declared if row["name"] != "manifest.json"]
    if (
        manifest.get("schema") != "macfit-training-artifacts.v1"
        or manifest.get("input_sha256") != canonical_sha256(job)
        or manifest.get("base_model") != job["base_model"]
        or manifest.get("artifacts") != expected_records
        or read_json(artifact_dir / "resolved-config.json") != job
    ):
        raise ValueError("The artifact manifest does not match this completed job.")
    return job, result


def extract_adapter(archive_path: Path, output: Path, base_model: dict[str, Any]) -> None:
    """Extract only bounded flat regular safetensors/config files; never tar links."""
    seen, total = set(), 0
    with tarfile.open(real_path(archive_path), "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                len(path.parts) != 2
                or path.parts[0] != "adapter"
                or path.parts[1] not in ADAPTER_NAMES
                or not member.isfile()
                or member.name in seen
                or not 0 <= member.size <= MAX_ADAPTER
            ):
                raise ValueError("The adapter archive contains an unsafe or unsupported entry.")
            seen.add(member.name)
            total += member.size
            if total > MAX_ADAPTER:
                raise ValueError("The expanded adapter archive is too large.")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("The adapter archive is incomplete.")
            with source, (output / path.parts[1]).open("xb") as destination:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("The adapter archive ended unexpectedly.")
                    destination.write(chunk)
                    remaining -= len(chunk)
    if not {"adapter/adapter_config.json", "adapter/adapter_model.safetensors"} <= seen:
        raise ValueError("The adapter archive is missing its configuration or safetensors weights.")
    config = read_json(output / "adapter_config.json")
    if (
        config.get("base_model_name_or_path") != base_model["repo_id"]
        or config.get("revision") != base_model["revision"]
        or config.get("peft_type") != "LORA"
        or config.get("task_type") != "CAUSAL_LM"
    ):
        raise ValueError("The adapter configuration does not match the pinned base model.")


def merge_cpu(base_dir: Path, adapter_dir: Path, output: Path) -> None:
    """GPU-free actual PEFT merge; all model/tokenizer files must already be local."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    adapted = PeftModel.from_pretrained(
        base,
        adapter_dir,
        local_files_only=True,
        is_trainable=False,
        torch_device="cpu",
        device_map={"": "cpu"},
    )
    merged = adapted.merge_and_unload(safe_merge=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(
        base_dir, local_files_only=True, trust_remote_code=False
    )
    tokenizer.save_pretrained(output)


def export_merged(job_dir: Path, output: Path, *, base_model_dir: Path | None = None) -> dict:
    from amd_inference_opt.vllm_model_snapshot import (
        VLLMModelSnapshotManifest,
        verify_vllm_model_snapshot,
    )

    job_dir, output = real_path(job_dir), real_path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("The export needs a new output directory inside an existing parent.")
    if job_dir == output or job_dir in output.parents:
        raise ValueError("Merged exports must be outside the original job directory.")
    job, result = verified_job(job_dir)
    snapshot_data = read_json(job_dir / "model-snapshot.json")
    # The worker serializes model_dump() names; the shared manifest parser uses its alias.
    if "schema_name" in snapshot_data:
        snapshot_data["schema"] = snapshot_data.pop("schema_name")
    snapshot = VLLMModelSnapshotManifest.model_validate(snapshot_data)
    base = job["base_model"]
    if (
        snapshot.model_id != base["repo_id"]
        or snapshot.revision != base["revision"]
        or snapshot.tokenizer_revision != base["revision"]
        or snapshot.snapshot_digest != result.get("provenance", {}).get("model_snapshot_sha256")
    ):
        raise ValueError("The local base-model snapshot does not match the completed training job.")
    base_dir = real_path(base_model_dir if base_model_dir is not None else snapshot.root)
    if base_dir == output or output in base_dir.parents or base_dir in output.parents:
        raise ValueError("The export and base-model snapshot must be separate directories.")
    verify_vllm_model_snapshot(snapshot.model_copy(update={"root": base_dir}))
    output.mkdir(mode=0o700, exist_ok=False)
    marker = output / "EXPORT_INCOMPLETE.json"
    write_json(
        marker, {"complete": False, "message": "Only export-manifest.json confirms success."}
    )
    with tempfile.TemporaryDirectory(prefix=".adapter-", dir=output) as temporary:
        adapter_dir = Path(temporary)
        extract_adapter(job_dir / "artifacts" / "adapter.tar.gz", adapter_dir, base)
        merge_cpu(base_dir, adapter_dir, output)
    files = sorted(path for path in output.iterdir() if path != marker)
    if not any(path.suffix == ".safetensors" for path in files):
        raise RuntimeError("Merge did not produce safe serialized model weights.")
    records = [describe_artifact(path, "merged_model_file") for path in files]
    manifest = {
        "schema": "macfit-merged-export.v1",
        "base_model": base,
        "source_input_sha256": canonical_sha256(job),
        "source_model_snapshot_sha256": snapshot.snapshot_digest,
        "source_adapter_sha256": next(
            row["sha256"] for row in result["artifacts"] if row["type"] == "lora_adapter"
        ),
        "dtype": "bfloat16",
        "device": "cpu",
        "files": records,
        "format": "transformers_safetensors",
        "notes": "Merged weights require a compatible Transformers runtime. Not a GGUF or app. "
        "Re-evaluate outputs after merging; floating-point rounding can change generations.",
    }
    write_json(output / "export-manifest.json", manifest)
    marker.unlink()
    return manifest
