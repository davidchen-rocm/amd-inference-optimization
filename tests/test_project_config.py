from pathlib import Path

import pytest
import yaml

from amd_inference_opt.project_config import (
    ProjectConfigError,
    check_project_config,
    discover_project_root,
    initialize_project_config,
    load_project_config,
    set_project_config_value,
)


def test_config_init_discovery_read_and_atomic_set(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    llama = tmp_path / "llama.cpp"
    llama.mkdir()

    initialized = initialize_project_config(
        project,
        model_root=models,
        llama_cpp_repo=llama,
        rocm_mcp_command=["python", "-m", "rocm_issue_agent.mcp"],
    )
    config_path = project / ".gpuopt/config.yaml"
    nested = project / "examples" / "nested"
    nested.mkdir(parents=True)

    assert initialized.model_root == models.resolve()
    assert initialized.store_root == (project / ".gpuopt/store").resolve()
    assert discover_project_root(nested) == project.resolve()
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["schema"] == (
        "gpuopt.project-config.v1"
    )

    updated = set_project_config_value(project, "gpu-device", "2")
    updated = set_project_config_value(project, "gpu-gfx-target", "GFX942")
    assert updated.gpu_device == 2
    assert updated.gpu_gfx_target == "gfx942"
    assert load_project_config(project).gpu_device == 2
    assert load_project_config(project).gpu_gfx_target == "gfx942"
    assert list(project.glob(".gpuopt/.config.yaml.*")) == []


def test_config_rejects_overwrite_unknown_keys_and_invalid_update(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    initialize_project_config(project, model_root=models)
    path = project / ".gpuopt/config.yaml"
    original = path.read_bytes()

    with pytest.raises(ProjectConfigError, match="already exists"):
        initialize_project_config(project, model_root=models)
    with pytest.raises(ProjectConfigError, match="unsupported project setting"):
        set_project_config_value(project, "arbitrary-command", "oops")
    with pytest.raises(ProjectConfigError, match="cannot update"):
        set_project_config_value(project, "gpu-device", "not-an-int")
    with pytest.raises(ProjectConfigError, match="cannot update"):
        set_project_config_value(project, "gpu-gfx-target", "mi300x")

    assert path.read_bytes() == original


def test_config_is_strict_and_rejects_symlinked_control_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    initialize_project_config(project, model_root=models)
    path = project / ".gpuopt/config.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["unknown"] = True
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ProjectConfigError, match="invalid project config"):
        load_project_config(project)

    path.unlink()
    external = tmp_path / "external.yaml"
    external.write_text("schema: gpuopt.project-config.v1\n", encoding="utf-8")
    path.symlink_to(external)
    with pytest.raises(ProjectConfigError, match="unsafe"):
        load_project_config(project)
    with pytest.raises(ProjectConfigError, match="symlink"):
        discover_project_root(project)


def test_config_doctor_reports_missing_and_unset_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    config = initialize_project_config(project, model_root=models)

    checks = {item.name: item for item in check_project_config(config)}

    assert checks["model_root"].status == "OK"
    assert checks["store_root"].status == "OK"
    assert checks["llama_cpp_repo"].status == "UNSET"
    assert checks["llama_cpp_build_dir"].status == "UNSET"


def test_discovery_can_return_none(tmp_path: Path) -> None:
    assert discover_project_root(tmp_path, required=False) is None
    with pytest.raises(ProjectConfigError, match="no .gpuopt/config.yaml"):
        discover_project_root(tmp_path)
