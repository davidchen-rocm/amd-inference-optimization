#!/usr/bin/env python3
"""Prepare a clean Q4_RDNA-capable llama.cpp worktree and freeze its full patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

BASE_COMMIT = "a7a6d0d269c896218b6c78e0933bd6a17519d3f6"
SOURCE_HASHES = {
    "llama-cpp-q4rdna-integration.patch": (
        "d214de5f4c69ba26b61a53133db0c4c91359eed113239fef596bb20e72e024c8"
    ),
    "q4rdna.cu": "46b24d2a747ee4497aa5f36d6f380718bc27ad867a9e94bc684018783bad7840",
    "q4rdna.cuh": "178b2471e209ae71f80fe9f8af36b7ce3b22c2e2d503efc1aa57c58545ee1676",
}
DESTINATIONS = {
    "q4rdna.cu": "ggml/src/ggml-cuda/q4rdna.cu",
    "q4rdna.cuh": "ggml/src/ggml-cuda/q4rdna.cuh",
}
EXPECTED_CHANGED_PATHS = {
    "ggml/src/ggml-cuda/ggml-cuda.cu",
    *DESTINATIONS.values(),
}


class PreparationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, shell=False)
    if result.returncode != 0:
        raise PreparationError(
            f"command failed ({result.returncode}): {argv!r}\n{result.stderr.strip()}"
        )
    return result


def prepare(clean_repo: Path, evidence_source: Path, output_root: Path) -> dict[str, object]:
    clean_repo = clean_repo.resolve()
    evidence_source = evidence_source.resolve()
    output_root = output_root.resolve()
    worktree = output_root / "source"
    frozen_patch = output_root / "q4rdna-complete.patch"
    manifest_path = output_root / "preparation.json"

    if output_root.exists():
        raise PreparationError(f"refusing to overwrite existing output: {output_root}")
    if not (clean_repo / "CMakeLists.txt").is_file():
        raise PreparationError(f"not a llama.cpp checkout: {clean_repo}")
    source_checkout_dirty = bool(
        run(["git", "status", "--porcelain"], cwd=clean_repo).stdout.strip()
    )
    resolved = run(
        ["git", "rev-parse", "--verify", f"{BASE_COMMIT}^{{commit}}"], cwd=clean_repo
    ).stdout.strip()
    if resolved != BASE_COMMIT:
        raise PreparationError(f"base commit mismatch: {resolved}")

    for name, expected in SOURCE_HASHES.items():
        path = evidence_source / name
        if not path.is_file():
            raise PreparationError(f"missing public evidence source: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise PreparationError(f"source hash mismatch for {name}: {actual}")

    output_root.mkdir(parents=True)
    run(
        ["git", "worktree", "add", "--detach", str(worktree), BASE_COMMIT],
        cwd=clean_repo,
    )
    integration = evidence_source / "llama-cpp-q4rdna-integration.patch"
    run(["git", "apply", "--check", "--", str(integration)], cwd=worktree)
    run(["git", "apply", "--", str(integration)], cwd=worktree)
    for name, destination in DESTINATIONS.items():
        target = worktree / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(evidence_source / name, target)

    changed = {
        line[3:]
        for line in run(["git", "status", "--porcelain"], cwd=worktree).stdout.splitlines()
        if len(line) >= 4
    }
    if changed != EXPECTED_CHANGED_PATHS:
        raise PreparationError(
            f"prepared source changed unexpected paths: {sorted(changed)}"
        )
    run(["git", "add", "-N", "--", *sorted(DESTINATIONS.values())], cwd=worktree)
    diff = run(
        [
            "git",
            "diff",
            "--binary",
            "--full-index",
            "HEAD",
            "--",
            *sorted(EXPECTED_CHANGED_PATHS),
        ],
        cwd=worktree,
    ).stdout
    if not diff.strip():
        raise PreparationError("prepared runtime produced an empty patch")
    frozen_patch.write_text(diff, encoding="utf-8")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "base_commit": BASE_COMMIT,
        "clean_repo": str(clean_repo),
        "source_checkout_dirty": source_checkout_dirty,
        "source_checkout_changes_excluded": True,
        "worktree": str(worktree),
        "public_evidence_source": str(evidence_source),
        "source_hashes": SOURCE_HASHES,
        "changed_paths": sorted(changed),
        "frozen_patch": str(frozen_patch),
        "frozen_patch_sha256": sha256_file(frozen_patch),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-repo", type=Path, required=True)
    parser.add_argument("--evidence-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.clean_repo, args.evidence_source, args.output_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
