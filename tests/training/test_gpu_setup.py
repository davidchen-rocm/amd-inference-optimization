from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "deploy/macfit-training/gpu_setup.py"
SPEC = importlib.util.spec_from_file_location("macfit_gpu_setup", PATH)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def private_env(path, extra=""):
    path.write_text(
        "MACFIT_TRAINING_GATEWAY_SECRET=synthetic-test-gateway-secret-long-value\n"
        "MACFIT_FIREBASE_PROJECT_ID=macfit-example\n"
        "MACFIT_GPU_DEADLINE=2026-10-01T04:00:00Z\n"
        "MACFIT_STOP_ACCEPTING_AT=2026-10-01T03:00:00Z\n" + extra
    )
    path.chmod(0o600)
    return path


def test_dry_run_never_reads_credentials_or_mutates_host(tmp_path, capsys):
    missing = tmp_path / "missing-secret.env"
    assert setup.main(["--base", str(tmp_path / "absent"), "--env-file", str(missing)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["apply"] is False and plan["listener"] == "127.0.0.1:8791"
    assert not (tmp_path / "absent").exists()


def test_unit_preserves_worker_cleanup_and_private_gpu_paths():
    unit = setup.render_unit(Path("/srv/macfit-training"))
    assert "User=macfit-train" in unit
    assert "KillMode=control-group" in unit and "TimeoutStopSec=30" in unit
    assert "-m macfit_training.service" in unit
    assert "EnvironmentFile=/etc/macfit-training/gpu.env" in unit
    assert "WorkingDirectory=/srv/macfit-training/repo" in unit
    assert (
        "ReadWritePaths=/srv/macfit-training/data /srv/macfit-training/cache "
        "/srv/macfit-training/tmp"
        in unit
    )
    assert "gateway-secret" not in unit
    assert "PrivateDevices=" not in unit  # GPU render devices must remain accessible.
    assert "MACFIT_MODEL_CACHE=/srv/macfit-training/cache" in unit
    with pytest.raises(ValueError):
        setup.render_unit(Path("/srv/unsafe%specifier"))
    with pytest.raises(ValueError):
        setup.render_unit(Path("/srv/path with spaces"))


def test_env_requires_private_file_expected_paths_and_explicit_deadline(tmp_path):
    path = private_env(tmp_path / "gpu.env")
    assert b"MACFIT_GPU_DEADLINE" in setup.load_env(path, Path("/srv/macfit-training"))
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        setup.load_env(path, Path("/srv/macfit-training"))
    private_env(path, "MACFIT_MODEL_CACHE=/root/private-cache\n")
    with pytest.raises(ValueError, match="MACFIT_MODEL_CACHE"):
        setup.load_env(path, Path("/srv/macfit-training"))
    private_env(path)
    link = tmp_path / "link.env"
    link.symlink_to(path)
    with pytest.raises(OSError):
        setup.load_env(link, Path("/srv/macfit-training"))


def test_atomic_install_is_idempotent_and_preserves_private_mode(tmp_path):
    destination = tmp_path / "gpu.env"
    content = b"synthetic test contents\n"
    assert setup.atomic_write(destination, content, 0o600) is True
    assert setup.atomic_write(destination, content, 0o600) is False
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    link = tmp_path / "link"
    link.symlink_to(destination)
    with pytest.raises(ValueError):
        setup.atomic_write(link, b"not written", 0o600)
    assert destination.read_bytes() == content


def test_ownership_update_preserves_immutable_inputs_and_cache_symlinks(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    target = directory / "input.json"
    target.write_text("{}")
    target.chmod(0o400)
    blob = directory / "blob"
    blob.write_text("model bytes")
    blob.chmod(0o644)
    link = directory / "snapshot"
    link.symlink_to(blob)
    setup.private_tree(directory, os.getuid(), os.getgid())
    assert stat.S_IMODE(target.stat().st_mode) == 0o400
    assert stat.S_IMODE(blob.stat().st_mode) == 0o600
    assert link.is_symlink()
