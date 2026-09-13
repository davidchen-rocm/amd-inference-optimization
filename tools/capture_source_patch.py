#!/usr/bin/env python3
"""Capture an exact Git worktree diff and hash-bound provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base", default="HEAD")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    base_commit = _git(repo, "rev-parse", args.base).decode().strip()
    patch = _git(repo, "diff", "--binary", "--full-index", args.base, "--", *args.paths)
    if not patch:
        raise SystemExit("refusing to capture an empty source patch")
    patch_sha256 = hashlib.sha256(patch).hexdigest()
    _atomic_bytes(output, patch)
    document = {
        "schema": "gpuopt.source-patch-provenance.v1",
        "repo": str(repo),
        "base_commit": base_commit,
        "paths": args.paths,
        "patch": str(output),
        "patch_sha256": patch_sha256,
        "patch_size_bytes": len(patch),
    }
    _atomic_bytes(
        manifest,
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
