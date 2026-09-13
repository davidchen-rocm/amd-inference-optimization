#!/usr/bin/env python3
"""Capture a hash-bound vLLM/PyTorch environment manifest without loading a GPU."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from amd_inference_opt.vllm_environment import capture_vllm_environment


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require",
        action="append",
        dest="required",
        help="Required distribution; repeat as needed (default: vllm, torch, amdsmi).",
    )
    parser.add_argument(
        "--optional",
        action="append",
        dest="optional",
        help="Optional accelerator distribution; repeat as needed.",
    )
    arguments = parser.parse_args()
    keyword = {}
    if arguments.required is not None:
        keyword["required_distributions"] = tuple(arguments.required)
    if arguments.optional is not None:
        keyword["optional_distributions"] = tuple(arguments.optional)
    manifest = capture_vllm_environment(**keyword)
    payload = (
        json.dumps(
            manifest.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _write_atomic(arguments.output, payload)
    print(manifest.identity_sha256)


if __name__ == "__main__":
    main()
