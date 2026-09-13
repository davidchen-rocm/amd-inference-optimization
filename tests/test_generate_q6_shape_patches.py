from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.generate_q6_shape_patches import (  # noqa: E402
    ROWS_PER_BLOCK_ANCHOR,
    SHAPE_ANCHOR,
    TARGET_PATH,
    ShapePatchError,
    generate_shape_patches,
)


def _git(repo: Path, *args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _fixture_source() -> str:
    return (
        "#include <array>\n\n" + ROWS_PER_BLOCK_ANCHOR + "        return 1;\n"
        "    }\n"
        "    return 1;\n"
        "}\n\n"
        "template <ggml_type type>\n"
        "static void launch() {\n" + SHAPE_ANCHOR + "            GGML_TYPE_IQ3_XXS,\n"
        "            GGML_TYPE_IQ3_S,\n"
        "        };\n"
        "}\n"
    )


def _make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "llama.cpp"
    target = repo / TARGET_PATH
    target.parent.mkdir(parents=True)
    (repo / "CMakeLists.txt").write_text("project(fixture)\n", encoding="utf-8")
    target.write_text(_fixture_source(), encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "add", ".")
    _git(
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
    return repo, _git(repo, "rev-parse", "HEAD").strip()


def test_generates_two_independent_exact_shape_patches_and_manifests(
    tmp_path: Path,
) -> None:
    repo, commit = _make_repo(tmp_path)
    original = (repo / TARGET_PATH).read_bytes()
    output = tmp_path / "generated"

    manifests = generate_shape_patches(repo, output, expected_commit=commit)

    assert len(manifests) == 2
    assert (repo / TARGET_PATH).read_bytes() == original
    assert _git(repo, "status", "--porcelain") == ""
    patches = sorted(output.glob("*.patch"))
    manifest_paths = sorted(output.glob("*.manifest.json"))
    assert len(patches) == len(manifest_paths) == 2

    for patch_path in patches:
        patch = patch_path.read_text(encoding="utf-8")
        _git(repo, "apply", "--check", "-", input_text=patch)
        assert "MMVQ_PARAMETERS_RDNA4 && ncols_dst == 1 && small_k" in patch
        assert "cc == GGML_CUDA_CC_OFFSET_AMD + 0x1201" in patch
        assert "return true;" in patch

    patch_a = next(path for path in patches if "n12288" in path.name)
    patch_b = next(path for path in patches if "n4096-" in path.name)
    assert "nrows_x == 12288 && ncols_x == 4096" in patch_a.read_text()
    assert "nrows_x == 4096 && ncols_x == 4096" not in patch_a.read_text()
    assert "nrows_x == 4096 && ncols_x == 4096" in patch_b.read_text()
    assert "nrows_x == 12288 && ncols_x == 4096" not in patch_b.read_text()

    by_n = {manifest["target"]["N"]: manifest for manifest in manifests}
    assert by_n[12288]["expected_raw_dispatch_signature"] == {
        "baseline": {"grid": [393216, 8, 1], "workgroup": [32, 8, 1]},
        "candidate": {"grid": [49152, 8, 1], "workgroup": [32, 8, 1]},
    }
    assert by_n[4096]["expected_raw_dispatch_signature"]["candidate"]["grid"] == [
        16384,
        8,
        1,
    ]
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        patch_path = output / manifest["patch"]["path"]
        assert manifest["base"]["commit"] == commit
        assert manifest["scope_guards"]["source_checkout_modified"] is False
        assert manifest["patch"]["sha256"] == hashlib.sha256(patch_path.read_bytes()).hexdigest()


def test_refuses_dirty_source_without_creating_output(tmp_path: Path) -> None:
    repo, commit = _make_repo(tmp_path)
    output = tmp_path / "generated"
    (repo / TARGET_PATH).write_text(_fixture_source() + "// dirty\n", encoding="utf-8")

    with pytest.raises(ShapePatchError, match="not clean"):
        generate_shape_patches(repo, output, expected_commit=commit)

    assert not output.exists()


def test_refuses_wrong_commit_and_existing_output(tmp_path: Path) -> None:
    repo, commit = _make_repo(tmp_path)
    output = tmp_path / "generated"

    with pytest.raises(ShapePatchError, match="base commit mismatch"):
        generate_shape_patches(repo, output, expected_commit="0" * 40)
    output.mkdir()
    with pytest.raises(ShapePatchError, match="refusing to overwrite"):
        generate_shape_patches(repo, output, expected_commit=commit)
