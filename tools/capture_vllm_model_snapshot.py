#!/usr/bin/env python3
"""Create a canonical local Hugging Face snapshot manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from amd_inference_opt.vllm_model_snapshot import capture_vllm_model_snapshot


def _write_atomic(path: Path, payload: bytes) -> None:
    lexical = path.expanduser()
    lexical.parent.mkdir(parents=True, exist_ok=True)
    if lexical.is_symlink():
        raise RuntimeError(f"refusing to replace symlink: {lexical}")
    destination = lexical.parent.resolve() / lexical.name
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    model_root = arguments.model_dir.expanduser().resolve()
    lexical_output = arguments.output.expanduser()
    if lexical_output.is_symlink():
        raise RuntimeError(f"refusing to replace symlink: {lexical_output}")
    output = lexical_output.parent.resolve() / lexical_output.name
    if output == model_root or output.is_relative_to(model_root):
        raise RuntimeError("snapshot manifest must be stored outside the model directory")
    manifest = capture_vllm_model_snapshot(
        model_root,
        model_id=arguments.model_id,
        revision=arguments.revision,
        tokenizer_revision=arguments.tokenizer_revision,
    )
    payload = (
        json.dumps(
            manifest.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _write_atomic(output, payload)
    print(manifest.snapshot_digest)


if __name__ == "__main__":
    main()
