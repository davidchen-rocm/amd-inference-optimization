from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from amd_inference_opt.vllm_environment import (
    PythonDistributionDigest,
    RuntimeFileDigest,
    VLLMEnvironmentError,
    VLLMEnvironmentManifest,
    _files_identity,
    capture_distribution,
    capture_vllm_environment,
    environment_identity_sha256,
)


class _FakeDistribution:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.version = "1.2.3"
        self.metadata = {"Name": "vllm"}
        self.files = [
            Path("vllm/__init__.py"),
            Path("vllm/kernel.so"),
            Path("vllm/__pycache__/ignored.pyc"),
        ]

    def locate_file(self, item: Path) -> Path:
        return self.root / item


def _distribution() -> PythonDistributionDigest:
    files = [
        RuntimeFileDigest(relative_path="vllm/a.py", sha256="a" * 64, size_bytes=7)
    ]
    return PythonDistributionDigest(
        name="vllm",
        version="1.0",
        status="present",
        file_count=1,
        total_bytes=7,
        files_sha256=_files_identity(files),
        files=files,
    )


def test_capture_distribution_hashes_installed_bytes_without_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm/__pycache__").mkdir()
    (tmp_path / "vllm/__init__.py").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "vllm/kernel.so").write_bytes(b"ELF-test")
    (tmp_path / "vllm/__pycache__/ignored.pyc").write_bytes(b"machine-local")
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: _FakeDistribution(tmp_path),
    )

    result = capture_distribution("vllm", required=True)

    assert result.status == "present"
    assert result.version == "1.2.3"
    assert result.file_count == 2
    assert [item.relative_path for item in result.files] == [
        "vllm/__init__.py",
        "vllm/kernel.so",
    ]
    assert result.files_sha256 is not None


def test_optional_missing_distribution_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> None:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)

    result = capture_distribution("aiter", required=False)

    assert result.status == "missing"
    assert result.file_count == 0


def test_required_missing_distribution_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> None:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)

    with pytest.raises(VLLMEnvironmentError, match="required.*vllm"):
        capture_distribution("vllm", required=True)


def test_manifest_identity_detects_tampering() -> None:
    distribution = _distribution()
    identity = environment_identity_sha256(
        python_executable="/opt/venv/bin/python",
        python_executable_resolved="/usr/bin/python3.13",
        python_executable_sha256="c" * 64,
        python_version="3.13.7",
        python_prefix="/opt/venv",
        python_base_prefix="/usr",
        pyvenv_cfg_sha256="d" * 64,
        probe_sha256="e" * 64,
        framework_source_sha256="f" * 64,
        distributions=[distribution],
        required_distributions=["vllm"],
    )
    manifest = VLLMEnvironmentManifest(
        python_executable=Path("/opt/venv/bin/python"),
        python_executable_resolved=Path("/usr/bin/python3.13"),
        python_executable_sha256="c" * 64,
        python_version="3.13.7",
        python_prefix=Path("/opt/venv"),
        python_base_prefix=Path("/usr"),
        pyvenv_cfg_sha256="d" * 64,
        probe_sha256="e" * 64,
        framework_source_sha256="f" * 64,
        distributions=[distribution],
        required_distributions=["vllm"],
        identity_sha256=identity,
    )

    assert manifest.identity_sha256 == identity
    with pytest.raises(ValidationError, match="identity_sha256"):
        VLLMEnvironmentManifest.model_validate(
            {
                **manifest.model_dump(mode="json", by_alias=True),
                "python_version": "3.13.8",
            }
        )


def test_capture_requires_target_interpreter_process(tmp_path: Path) -> None:
    other = tmp_path / "python"
    other.write_bytes(b"not-current-python")

    with pytest.raises(VLLMEnvironmentError, match="must run under the target"):
        capture_vllm_environment(python_executable=other)


def test_capture_rejects_another_venv_using_the_same_base_python(tmp_path: Path) -> None:
    other = tmp_path / "other-venv" / "bin" / "python"
    other.parent.mkdir(parents=True)
    other.symlink_to(Path(sys.executable).resolve())
    assert os.path.samefile(other, sys.executable)

    with pytest.raises(VLLMEnvironmentError, match="must run under the target"):
        capture_vllm_environment(python_executable=other)


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("relative_path", "vllm/b.py")])
def test_distribution_rejects_file_inventory_tampering(field: str, value: str) -> None:
    payload = _distribution().model_dump(mode="json")
    payload["files"][0][field] = value

    with pytest.raises(ValidationError, match="files_sha256"):
        PythonDistributionDigest.model_validate(payload)


def test_distribution_rejects_inconsistent_byte_count() -> None:
    payload = _distribution().model_dump(mode="json")
    payload["total_bytes"] += 1

    with pytest.raises(ValidationError, match="total_bytes"):
        PythonDistributionDigest.model_validate(payload)


@pytest.mark.parametrize("paths", [("vllm/a.py", "vllm/a.py"), ("vllm/b.py", "vllm/a.py")])
def test_distribution_rejects_duplicate_or_unsorted_files(paths: tuple[str, str]) -> None:
    files = [RuntimeFileDigest(relative_path=path, sha256="a" * 64, size_bytes=7) for path in paths]

    with pytest.raises(ValidationError, match="sorted and unique"):
        PythonDistributionDigest(
            name="vllm",
            version="1.0",
            status="present",
            file_count=2,
            total_bytes=14,
            files_sha256=_files_identity(files),
            files=files,
        )


def test_current_python_capture_can_use_small_fake_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm/__pycache__").mkdir()
    (tmp_path / "vllm/__init__.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "vllm/kernel.so").write_bytes(b"kernel")
    (tmp_path / "vllm/__pycache__/ignored.pyc").write_bytes(b"ignored")
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: _FakeDistribution(tmp_path),
    )

    result = capture_vllm_environment(
        python_executable=sys.executable,
        required_distributions=("vllm",),
        optional_distributions=(),
    )

    assert result.python_executable == Path(os.path.abspath(sys.executable))
    assert result.python_executable_resolved == Path(sys.executable).resolve()
    assert result.python_prefix == Path(sys.prefix).resolve()
    assert result.probe_sha256
    assert result.identity_sha256
    assert result.required_distributions == ["vllm"]
