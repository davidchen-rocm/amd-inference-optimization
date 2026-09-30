"""Atomic, hash-bound worker artifacts with portable adapter usage instructions."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def describe_artifact(path: Path, artifact_type: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Artifacts must be regular files.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "id": path.name.replace(".", "-"),
        "type": artifact_type,
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def training_source_identity() -> dict[str, Any]:
    """Bind the training implementation as well as the reused inference infrastructure."""
    root = Path(__file__).resolve().parent
    names = (
        "config.py",
        "data.py",
        "trainer.py",
        "generation.py",
        "evaluation.py",
        "artifacts.py",
        "worker.py",
        "cli.py",
    )
    files = {name: describe_artifact(root / name, "source")["sha256"] for name in names}
    return {"files": files, "sha256": canonical_sha256(files)}


def bundle_adapter(directory: Path, output: Path, base_model: dict[str, Any]) -> None:
    """Export adapter weights only; never silently present them as a standalone model."""
    readme = f"""# MacFit LoRA adapter

This bundle contains a PEFT LoRA adapter, not a standalone model or Mac application.
Load the original {base_model["repo_id"]} checkpoint at revision
{base_model["revision"]}, then attach this adapter with PEFT.
The model license and base weights remain separate. This is not evidence of improved quality.

Example (a compatible PyTorch / Transformers / PEFT environment is required):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(
    {base_model["repo_id"]!r}, revision={base_model["revision"]!r},
    torch_dtype='auto', trust_remote_code=False)
model = PeftModel.from_pretrained(base, './adapter')
tokenizer = AutoTokenizer.from_pretrained(
    {base_model["repo_id"]!r}, revision={base_model["revision"]!r}, trust_remote_code=False)
```

Use the model's chat template with enable_thinking=False to match training.
Review evaluation.json before choosing this adapter for your task.
"""
    (directory / "USAGE.md").write_text(readme, encoding="utf-8")
    files = sorted(directory.rglob("*"))
    if not (directory / "adapter_model.safetensors").is_file():
        raise RuntimeError("Training did not produce adapter weights.")
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            if path.is_symlink() or not path.is_file():
                raise ValueError("Adapter output contains an unsupported file.")
            archive.add(
                path, arcname="adapter/" + path.relative_to(directory).as_posix(), recursive=False
            )


def finish_artifacts(
    job_dir: Path,
    job: dict[str, Any],
    result: dict[str, Any],
    *,
    adapter_dir: Path | None = None,
) -> list[dict[str, Any]]:
    directory = job_dir / "artifacts"
    directory.mkdir(exist_ok=True)
    write_json(directory / "resolved-config.json", job)
    records = [describe_artifact(directory / "resolved-config.json", "configuration")]
    if adapter_dir is not None:
        bundle_adapter(adapter_dir, directory / "adapter.tar.gz", job["base_model"])
        records.append(describe_artifact(directory / "adapter.tar.gz", "lora_adapter"))
    filename = "evaluation.json" if "evaluation" in result else "generation.json"
    write_json(directory / filename, result.get("evaluation", result))
    records.append(
        describe_artifact(
            directory / filename, "evaluation" if "evaluation" in result else "generated_examples"
        )
    )
    manifest = {
        "schema": "macfit-training-artifacts.v1",
        "input_sha256": canonical_sha256(job),
        "base_model": job["base_model"],
        "provenance": result["provenance"],
        "artifacts": records,
        "adapter_is_standalone_model": False,
        "usage": "The adapter requires its exact base model and a compatible PEFT runtime.",
    }
    write_json(directory / "manifest.json", manifest)
    return [*records, describe_artifact(directory / "manifest.json", "manifest")]
