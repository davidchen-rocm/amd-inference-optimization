#!/usr/bin/env python3
"""Generate two independent, exact-shape llama.cpp Q6_K patch candidates."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amd_inference_opt.shape_kernels import LOCKED_QWEN3_8B_Q6_K_SHAPES, LockedQwenShape

BASE_COMMIT = "a7a6d0d269c896218b6c78e0933bd6a17519d3f6"
TARGET_PATH = Path("ggml/src/ggml-cuda/mmvq.cu")

_ROWS_SIGNATURE = (
    "static constexpr __host__ __device__ int calc_rows_per_block("
    "int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {"
)
_GENERIC_ROWS_IF = (
    "    if (table_id == MMVQ_PARAMETERS_GENERIC || "
    "table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING) {"
)
ROWS_PER_BLOCK_ANCHOR = _ROWS_SIGNATURE + "\n" + _GENERIC_ROWS_IF + "\n"
ROWS_PER_BLOCK_REPLACEMENT = (
    _ROWS_SIGNATURE
    + "\n"
    + "    if (table_id == MMVQ_PARAMETERS_RDNA4 && ncols_dst == 1 && small_k) {\n"
    + "        return nwarps;\n"
    + "    }\n"
    + _GENERIC_ROWS_IF
    + "\n"
)
_USE_LINE = (
    "        bool          use                   = nwarps > 1 && "
    "blocks_per_row_x < nwarps * blocks_per_iter_1warp;"
)
SHAPE_ANCHOR = (
    _USE_LINE + "\n\n" + "        constexpr std::array<ggml_type, 2> iq_slow_turing = {\n"
)


class ShapePatchError(RuntimeError):
    """The clean source coordinate or requested output is unsafe."""


@dataclass(frozen=True)
class _Variant:
    filename_stem: str
    shape: LockedQwenShape


VARIANTS = (
    _Variant(
        "q6-k-gfx1201-n12288-k4096-small-k-rpb8",
        LOCKED_QWEN3_8B_Q6_K_SHAPES[0],
    ),
    _Variant(
        "q6-k-gfx1201-n4096-k4096-small-k-rpb8",
        LOCKED_QWEN3_8B_Q6_K_SHAPES[1],
    ),
)


def _run(
    argv: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        shell=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ShapePatchError(f"command failed ({result.returncode}): {argv!r}: {detail}")
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_id(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git object identity


def _replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ShapePatchError(
            f"expected exactly one {description} anchor in {TARGET_PATH}; found {count}"
        )
    return source.replace(old, new, 1)


def _shape_replacement(shape: LockedQwenShape) -> str:
    return (
        _USE_LINE
        + f"""

        // Exact Qwen3-8B decode candidate: gfx1201 Q6_K, ncols_dst=1,
        // N={shape.n}, K={shape.k}. Keep every other RDNA shape on the stock path.
        if (!has_ids && cc == GGML_CUDA_CC_OFFSET_AMD + 0x1201 &&
                type == GGML_TYPE_Q6_K && c_ncols_dst == 1 &&
                nrows_x == {shape.n} && ncols_x == {shape.k}) {{
            return true;
        }}

        constexpr std::array<ggml_type, 2> iq_slow_turing = {{
"""
    )


def _candidate_source(source: str, shape: LockedQwenShape) -> str:
    candidate = _replace_once(
        source,
        ROWS_PER_BLOCK_ANCHOR,
        ROWS_PER_BLOCK_REPLACEMENT,
        "rows-per-block",
    )
    return _replace_once(
        candidate,
        SHAPE_ANCHOR,
        _shape_replacement(shape),
        "shape selector",
    )


def _unified_patch(original: str, candidate: str) -> str:
    relative = TARGET_PATH.as_posix()
    body = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
            n=3,
        )
    )
    if not body:
        raise ShapePatchError("shape candidate unexpectedly produced an empty patch")
    original_id = _git_blob_id(original.encode())
    candidate_id = _git_blob_id(candidate.encode())
    return (
        f"diff --git a/{relative} b/{relative}\n"
        f"index {original_id}..{candidate_id} 100644\n" + "".join(body)
    )


def _manifest(
    *,
    clean_repo: Path,
    commit: str,
    source: str,
    candidate: str,
    variant: _Variant,
    patch_name: str,
    patch: str,
) -> dict[str, Any]:
    shape = variant.shape
    return {
        "schema": "gpuopt.llama-q6-shape-patch.v1",
        "variant_id": variant.filename_stem,
        "base": {
            "commit": commit,
            "clean_checkout": str(clean_repo),
            "target_path": TARGET_PATH.as_posix(),
            "target_sha256": _sha256(source.encode()),
        },
        "target": {
            "architecture": shape.architecture,
            "kernel": "mul_mat_vec_q<GGML_TYPE_Q6_K,1,has_fusion,small_k>",
            "quantization": shape.kernel_type,
            "ncols_dst": shape.ncols_dst,
            "N": shape.n,
            "K": shape.k,
            "operator": shape.operator,
            "small_k": True,
            "rows_per_block": shape.rows_per_block,
        },
        "expected_raw_dispatch_signature": {
            "baseline": {
                "grid": list(shape.baseline_grid),
                "workgroup": list(shape.workgroup),
            },
            "candidate": {
                "grid": list(shape.candidate_grid),
                "workgroup": list(shape.workgroup),
            },
        },
        "candidate_target_sha256": _sha256(candidate.encode()),
        "patch": {
            "path": patch_name,
            "sha256": _sha256(patch.encode()),
            "applies_independently_to_base": True,
        },
        "scope_guards": {
            "source_files_changed": [TARGET_PATH.as_posix()],
            "source_checkout_modified": False,
            "build_performed": False,
            "gpu_workload_run": False,
        },
    }


def generate_shape_patches(
    clean_repo: str | Path,
    output_root: str | Path,
    *,
    expected_commit: str = BASE_COMMIT,
) -> tuple[dict[str, Any], ...]:
    """Generate both independent patches without modifying ``clean_repo``."""

    repo = Path(clean_repo).resolve()
    output = Path(output_root).resolve()
    if not repo.is_dir() or not (repo / "CMakeLists.txt").is_file():
        raise ShapePatchError(f"not a llama.cpp source checkout: {repo}")
    if output.exists():
        raise ShapePatchError(f"refusing to overwrite existing output: {output}")
    if not output.parent.is_dir():
        raise ShapePatchError(f"output parent does not exist: {output.parent}")
    target = repo / TARGET_PATH
    if not target.is_file() or target.is_symlink():
        raise ShapePatchError(f"target is not a regular source file: {target}")

    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    if commit != expected_commit:
        raise ShapePatchError(f"base commit mismatch: {commit}; expected {expected_commit}")
    status = _run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo).stdout
    if status.strip():
        raise ShapePatchError(f"source checkout is not clean: {repo}")

    source = target.read_text(encoding="utf-8")
    committed = _run(["git", "show", f"{commit}:{TARGET_PATH.as_posix()}"], cwd=repo).stdout
    if source != committed:
        raise ShapePatchError(f"working target does not exactly match {commit}")

    generated: list[tuple[_Variant, str, str, str, dict[str, Any]]] = []
    for variant in VARIANTS:
        candidate = _candidate_source(source, variant.shape)
        patch = _unified_patch(source, candidate)
        _run(["git", "apply", "--check", "-"], cwd=repo, input_text=patch)
        patch_name = variant.filename_stem + ".patch"
        manifest_name = variant.filename_stem + ".manifest.json"
        manifest = _manifest(
            clean_repo=repo,
            commit=commit,
            source=source,
            candidate=candidate,
            variant=variant,
            patch_name=patch_name,
            patch=patch,
        )
        generated.append((variant, patch_name, manifest_name, patch, manifest))

    # Recheck immediately before the only write. No worktree, apply, build, or
    # GPU command is used; the output directory is atomically renamed in place.
    if _run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo).stdout.strip():
        raise ShapePatchError("source checkout changed during generation")
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp:
        staging = Path(temp)
        for _, patch_name, manifest_name, patch, manifest in generated:
            (staging / patch_name).write_text(patch, encoding="utf-8")
            (staging / manifest_name).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        os.rename(staging, output)

    return tuple(item[4] for item in generated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifests = generate_shape_patches(args.clean_repo, args.output_root)
    print(json.dumps(manifests, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
