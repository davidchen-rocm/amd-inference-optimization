import hashlib
import json
from pathlib import Path

import pytest

from amd_inference_opt.model_library import (
    ModelFormat,
    ModelLibraryError,
    ModelRole,
    ProvenanceStatus,
    link_model,
    list_models,
    load_model_catalog,
    scan_model_library,
    show_model,
)
from amd_inference_opt.project_config import initialize_project_config


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _origin(root: Path, name: str = "Tiny") -> tuple[Path, bytes]:
    directory = root / "origin" / name
    directory.mkdir(parents=True)
    weights = b"safe-tensor-weights"
    (directory / "model.safetensors").write_bytes(weights)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "model.safetensors"}}), encoding="utf-8"
    )
    (directory / "config.json").write_text(
        json.dumps({"model_type": "qwen", "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )
    return directory, weights


def _derived(root: Path, name: str = "Tiny", quant: str = "Q6_K") -> tuple[Path, bytes]:
    directory = root / "derived" / name / quant
    directory.mkdir(parents=True)
    payload = f"{name}-{quant}-gguf".encode()
    path = directory / f"{name}-{quant}.gguf"
    path.write_bytes(payload)
    return path, payload


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    initialize_project_config(project, model_root=models)
    return project, models


def test_scan_classifies_origin_and_derived_with_stable_identity(tmp_path: Path) -> None:
    project, models = _project(tmp_path)
    _, origin_bytes = _origin(models)
    derived_path, derived_bytes = _derived(models)

    first = scan_model_library(project)
    second = scan_model_library(project)
    origin = show_model(second, "origin/Tiny")
    derived = show_model(second, "derived/Tiny/Q6_K")

    assert [entry.id for entry in first.entries] == [entry.id for entry in second.entries]
    assert origin.role == ModelRole.ORIGIN
    assert origin.format == ModelFormat.HF_SAFETENSORS
    assert origin.architecture == "qwen"
    assert origin.quantization == "BF16"
    assert origin.size == len(origin_bytes)
    assert derived.role == ModelRole.DERIVED
    assert derived.model_path == derived_path.resolve()
    assert derived.sha256 == _sha(derived_bytes)
    assert derived.size == len(derived_bytes)
    assert derived.quantization == "Q6_K"
    assert derived.origin_model_id == origin.id
    assert derived.provenance_status == ProvenanceStatus.INFERRED
    assert load_model_catalog(project) == second
    assert list_models(second, role=ModelRole.DERIVED, quantization="q6_k") == [derived]


def test_explicit_link_is_declared_not_falsely_verified(tmp_path: Path) -> None:
    project, models = _project(tmp_path)
    _origin(models, "Source")
    _derived(models, "Renamed", "Q5_K_M")
    before = scan_model_library(project)
    assert show_model(before, "derived/Renamed/Q5_K_M").provenance_status == "MISSING"

    link = link_model(project, "derived/Renamed/Q5_K_M", "origin/Source")
    after = load_model_catalog(project)
    derived = show_model(after, "derived/Renamed/Q5_K_M")
    origin = show_model(after, "origin/Source")

    assert link.derived_id == derived.id
    assert derived.origin_model_id == origin.id
    assert derived.source_model_id == origin.id
    assert derived.provenance_status == ProvenanceStatus.DECLARED
    assert (project / ".gpuopt/model-links.json").is_file()


def test_matching_preparation_manifest_is_verified(tmp_path: Path) -> None:
    project, models = _project(tmp_path)
    origin_dir, origin_bytes = _origin(models)
    derived_path, derived_bytes = _derived(models, quant="Q8_0")
    manifest = {
        "source_model_dir": str(origin_dir),
        "input_manifest": {
            "files": [
                {
                    "path": "model.safetensors",
                    "size": len(origin_bytes),
                    "sha256": _sha(origin_bytes),
                }
            ]
        },
        "output": {
            "path": str(derived_path),
            "size": len(derived_bytes),
            "sha256": _sha(derived_bytes),
        },
    }
    preparation = derived_path.with_name(derived_path.name + ".preparation.json")
    preparation.write_text(json.dumps(manifest), encoding="utf-8")

    catalog = scan_model_library(project)
    origin = show_model(catalog, "origin/Tiny")
    derived = show_model(catalog, "derived/Tiny/Q8_0")

    assert derived.provenance_status == ProvenanceStatus.VERIFIED
    assert derived.source_model_id == origin.id
    assert derived.origin_model_id == origin.id
    assert derived.preparation_manifest == preparation.resolve()


def test_invalid_manifest_does_not_upgrade_inferred_provenance(tmp_path: Path) -> None:
    project, models = _project(tmp_path)
    origin_dir, origin_bytes = _origin(models)
    derived_path, derived_bytes = _derived(models, quant="Q5_K_M")
    preparation = derived_path.with_name(derived_path.name + ".preparation.json")
    preparation.write_text(
        json.dumps(
            {
                "source_model_dir": str(origin_dir),
                "input_manifest": {
                    "files": [
                        {
                            "path": "model.safetensors",
                            "size": len(origin_bytes),
                            "sha256": _sha(origin_bytes),
                        }
                    ]
                },
                "output": {
                    "path": str(derived_path),
                    "size": len(derived_bytes),
                    "sha256": "0" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    derived = show_model(scan_model_library(project), "derived/Tiny/Q5_K_M")

    assert derived.provenance_status == ProvenanceStatus.INFERRED
    assert derived.preparation_manifest is None


def test_symlinked_model_artifacts_are_never_followed(tmp_path: Path) -> None:
    project, models = _project(tmp_path)
    _origin(models)
    variant = models / "derived" / "Tiny" / "Q4_K_M"
    variant.mkdir(parents=True)
    external = tmp_path / "external.gguf"
    external.write_bytes(b"do-not-hash")
    (variant / "escaped.gguf").symlink_to(external)

    catalog = scan_model_library(project)

    assert list_models(catalog, role=ModelRole.DERIVED) == []
    assert any("symlinked derived model skipped" in warning for warning in catalog.warnings)


def test_multiple_gguf_files_receive_unambiguous_aliases(tmp_path: Path) -> None:
    project, models = _project(tmp_path)
    variant = models / "derived" / "Tiny" / "Q4_K_M"
    variant.mkdir(parents=True)
    (variant / "a.gguf").write_bytes(b"a")
    (variant / "b.gguf").write_bytes(b"b")

    catalog = scan_model_library(project)

    aliases = [entry.aliases[0] for entry in catalog.entries]
    assert aliases == ["derived/Tiny/Q4_K_M/a.gguf", "derived/Tiny/Q4_K_M/b.gguf"]
    with pytest.raises(ModelLibraryError, match="not found"):
        show_model(catalog, "derived/Tiny/Q4_K_M")


def test_changed_model_content_gets_new_identity(tmp_path: Path) -> None:
    project, models = _project(tmp_path)
    path, _ = _derived(models)
    first = show_model(scan_model_library(project), "derived/Tiny/Q6_K")
    path.write_bytes(b"replacement-model")

    second = show_model(scan_model_library(project), "derived/Tiny/Q6_K")

    assert first.sha256 != second.sha256
    assert first.id != second.id


def test_catalog_loader_rejects_symlink(tmp_path: Path) -> None:
    project, models = _project(tmp_path)
    _origin(models)
    scan_model_library(project)
    path = project / ".gpuopt/model-catalog.json"
    external = tmp_path / "catalog.json"
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)

    with pytest.raises(ModelLibraryError, match="unsafe"):
        load_model_catalog(project)
