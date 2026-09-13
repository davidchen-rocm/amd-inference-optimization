from __future__ import annotations

from pathlib import Path

from amd_inference_opt.resource_lock import exclusive_gpu_lock


def test_gpu_lock_uses_a_stable_inode_and_releases(tmp_path: Path) -> None:
    path = tmp_path / "gpu-0.lock"
    with exclusive_gpu_lock(path):
        first_inode = path.stat().st_ino

    with exclusive_gpu_lock(path):
        assert path.stat().st_ino == first_inode
