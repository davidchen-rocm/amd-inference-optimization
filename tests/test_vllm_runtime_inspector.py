from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from amd_inference_opt.vllm_adapter import (
    RuntimeVerificationStatus,
    SpawnedProcess,
    VLLMRuntimeEvidence,
    VLLMServerSpec,
    canonical_sha256,
)
from amd_inference_opt.vllm_environment import (
    PythonDistributionDigest,
    RuntimeFileDigest,
    VLLMEnvironmentManifest,
    _files_identity,
    environment_identity_sha256,
)
from amd_inference_opt.vllm_runtime_inspector import (
    SubprocessTargetEnvironmentCapture,
    VerifiedVLLMRuntimeInspector,
    VLLMRuntimeInspectionError,
)

PYTHON_SHA = "a" * 64
MODEL_SHA = "b" * 64
IMAGE_SHA = "c" * 64


def _environment_manifest(path: Path) -> VLLMEnvironmentManifest:
    distributions = []
    for index, name in enumerate(("vllm", "torch", "amdsmi"), start=1):
        file = RuntimeFileDigest(
            relative_path=f"{name}/__init__.py",
            sha256=str(index) * 64,
            size_bytes=index,
        )
        distributions.append(
            PythonDistributionDigest(
                name=name,
                version="1.0",
                status="present",
                file_count=1,
                total_bytes=index,
                files_sha256=_files_identity([file]),
                files=[file],
            )
        )
    identity = environment_identity_sha256(
        python_executable="/opt/venv/bin/python3",
        python_executable_resolved="/usr/bin/python3.12",
        python_executable_sha256=PYTHON_SHA,
        python_version="3.12 fixture",
        python_prefix="/opt/venv",
        python_base_prefix="/usr",
        pyvenv_cfg_sha256="e" * 64,
        probe_sha256="f" * 64,
        framework_source_sha256="9" * 64,
        distributions=distributions,
        required_distributions=["vllm", "torch", "amdsmi"],
    )
    manifest = VLLMEnvironmentManifest(
        python_executable=Path("/opt/venv/bin/python3"),
        python_executable_resolved=Path("/usr/bin/python3.12"),
        python_executable_sha256=PYTHON_SHA,
        python_version="3.12 fixture",
        python_prefix=Path("/opt/venv"),
        python_base_prefix=Path("/usr"),
        pyvenv_cfg_sha256="e" * 64,
        probe_sha256="f" * 64,
        framework_source_sha256="9" * 64,
        distributions=distributions,
        required_distributions=["vllm", "torch", "amdsmi"],
        identity_sha256=identity,
    )
    path.write_text(
        manifest.model_dump_json(by_alias=True),
        encoding="utf-8",
    )
    return manifest


class _Capture:
    def __init__(self, manifest: VLLMEnvironmentManifest) -> None:
        self.manifest = manifest
        self.calls = 0

    def capture(
        self,
        spec: VLLMServerSpec,
        declared: VLLMEnvironmentManifest,
    ) -> VLLMEnvironmentManifest:
        assert spec.argv[0] == str(declared.python_executable)
        self.calls += 1
        return self.manifest


def _spec(tmp_path: Path, environment_sha: str, **overrides: Any) -> VLLMServerSpec:
    values: dict[str, Any] = {
        "argv": (
            "/opt/venv/bin/python3",
            "-I",
            "-m",
            "vllm.entrypoints.openai.api_server",
        ),
        "env": {
            "PYTHONNOUSERSITE": "1",
            "ROCR_VISIBLE_DEVICES": "GPU-uuid",
        },
        "unset_env": (
            "HIP_VISIBLE_DEVICES",
            "HSA_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "HSA_OVERRIDE_GFX_VERSION",
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONSTARTUP",
        ),
        "cwd": str(tmp_path),
        "native_executable_sha256": PYTHON_SHA,
        "environment_manifest_sha256": environment_sha,
        "vllm_version": "1.0",
        "model": "org/model",
        "model_revision": "revision",
        "model_snapshot_sha256": MODEL_SHA,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "config": {"fixture": True},
        "expected_served_model": "org/model",
    }
    values.update(overrides)
    return VLLMServerSpec(**values)


def _base(spec: VLLMServerSpec, process: SpawnedProcess) -> VLLMRuntimeEvidence:
    return VLLMRuntimeEvidence(
        verification=RuntimeVerificationStatus.VERIFIED,
        captured_at="2026-08-23T00:00:00Z",
        observed_pid=process.pid,
        observed_boot_id=process.boot_id,
        observed_start_ticks=process.start_ticks,
        pid_executable_path=spec.argv[0],
        pid_executable_sha256=PYTHON_SHA,
        native_executable_matches=True,
        process_environment_sha256="d" * 64,
        declared_environment_sha256=canonical_sha256(
            {"set": dict(spec.env), "unset": list(spec.unset_env)}
        ),
        declared_environment_matches=True,
        unset_environment_absent=True,
        rocr_visible_devices="GPU-uuid",
    )


def test_native_runtime_inspector_recaptures_package_identity(tmp_path: Path) -> None:
    manifest = _environment_manifest(tmp_path / "environment.json")
    spec = _spec(tmp_path, manifest.identity_sha256)
    process = SpawnedProcess(pid=4100, boot_id="boot", start_ticks=99)
    capture = _Capture(manifest)
    inspector = VerifiedVLLMRuntimeInspector(
        tmp_path / "environment.json", environment_capture=capture
    )
    inspector.local = type("Local", (), {"inspect": lambda self, s, p: _base(s, p)})()
    inspector.preflight(spec)

    evidence = inspector.inspect(spec, process)

    assert evidence.verification is RuntimeVerificationStatus.VERIFIED
    assert evidence.observed_environment_manifest_sha256 == manifest.identity_sha256
    assert evidence.container_id is None
    assert capture.calls == 1


@pytest.mark.parametrize(
    "missing_field",
    ["native_executable_matches", "declared_environment_matches", "unset_environment_absent"],
)
def test_package_provenance_cannot_upgrade_missing_native_evidence(
    tmp_path: Path, missing_field: str
) -> None:
    manifest = _environment_manifest(tmp_path / "environment.json")
    spec = _spec(tmp_path, manifest.identity_sha256)
    process = SpawnedProcess(pid=4100, boot_id="boot", start_ticks=99)
    base = replace(
        _base(spec, process),
        verification=RuntimeVerificationStatus.UNVERIFIED,
        reason="native process evidence is unavailable",
        **{missing_field: None},
    )
    inspector = VerifiedVLLMRuntimeInspector(
        tmp_path / "environment.json", environment_capture=_Capture(manifest)
    )
    inspector.local = type("Local", (), {"inspect": lambda self, s, p: base})()
    inspector.preflight(spec)

    evidence = inspector.inspect(spec, process)

    assert evidence.verification is RuntimeVerificationStatus.UNVERIFIED
    assert evidence.gate_eligible is False
    assert evidence.reason == base.reason


def test_outer_container_requires_matching_repo_digest_and_cgroup(tmp_path: Path) -> None:
    manifest = _environment_manifest(tmp_path / "environment.json")
    proc = tmp_path / "proc"
    (proc / "4100").mkdir(parents=True)
    (proc / "4100/cgroup").write_text("0::/machine.slice/container-1\n", encoding="utf-8")
    container_path = tmp_path / "container.json"
    container_path.write_text(
        json.dumps(
            {
                "schema": "gpuopt.vllm-container-runtime.v1",
                "container_id": "container-1",
                "container_init_pid": 1,
                "container_image_id": f"sha256:{IMAGE_SHA}",
                "repo_digests": [f"registry.invalid/vllm@sha256:{IMAGE_SHA}"],
                "cgroup_v2_binding": "/machine.slice/container-1",
                "devices": ["/dev/kfd", "/dev/dri/renderD128"],
            }
        ),
        encoding="utf-8",
    )
    spec = _spec(
        tmp_path,
        manifest.identity_sha256,
        image_digest=f"sha256:{IMAGE_SHA}",
    )
    process = SpawnedProcess(pid=4100, boot_id="boot", start_ticks=99)
    capture = _Capture(manifest)
    inspector = VerifiedVLLMRuntimeInspector(
        tmp_path / "environment.json",
        container_manifest=container_path,
        proc_root=proc,
        environment_capture=capture,
    )
    inspector.local = type(
        "Local",
        (),
        {
            "inspect": lambda self, s, p: replace(
                _base(s, p),
                verification=RuntimeVerificationStatus.UNVERIFIED,
                reason="native identity verified; outer image requires container proof",
            )
        },
    )()
    inspector.preflight(spec)

    evidence = inspector.inspect(spec, process)

    assert evidence.verification is RuntimeVerificationStatus.VERIFIED
    assert evidence.container_binding_kind == "cgroup_v2"
    assert evidence.process_binding_id == "/machine.slice/container-1"
    assert evidence.image_digest_matches is True


def test_package_drift_is_stopped_before_server_spawn(tmp_path: Path) -> None:
    manifest = _environment_manifest(tmp_path / "environment.json")
    spec = _spec(tmp_path, manifest.identity_sha256)
    drifted = manifest.model_copy(update={"identity_sha256": "0" * 64})
    inspector = VerifiedVLLMRuntimeInspector(
        tmp_path / "environment.json", environment_capture=_Capture(drifted)
    )
    with pytest.raises(VLLMRuntimeInspectionError, match="changed"):
        inspector.preflight(spec)


def test_failed_preflight_invalidates_previous_package_capture(tmp_path: Path) -> None:
    manifest = _environment_manifest(tmp_path / "environment.json")
    spec = _spec(tmp_path, manifest.identity_sha256)
    process = SpawnedProcess(pid=4100, boot_id="boot", start_ticks=99)
    capture = _Capture(manifest)
    inspector = VerifiedVLLMRuntimeInspector(
        tmp_path / "environment.json", environment_capture=capture
    )
    inspector.local = type("Local", (), {"inspect": lambda self, s, p: _base(s, p)})()
    inspector.preflight(spec)
    capture.manifest = manifest.model_copy(update={"identity_sha256": "0" * 64})

    with pytest.raises(VLLMRuntimeInspectionError, match="changed"):
        inspector.preflight(spec)

    evidence = inspector.inspect(spec, process)
    assert evidence.verification is RuntimeVerificationStatus.UNVERIFIED
    assert evidence.reason is not None
    assert "not preflighted" in evidence.reason


def test_target_capture_uses_isolated_lexical_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _environment_manifest(tmp_path / "environment.json")
    spec = _spec(tmp_path, manifest.identity_sha256)
    captured: dict[str, Any] = {}

    def fake_run(argv: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=manifest.model_dump_json(by_alias=True),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/tmp/shadow")
    monkeypatch.setenv("PYTHONHOME", "/tmp/shadow-home")

    observed = SubprocessTargetEnvironmentCapture().capture(spec, manifest)

    assert observed.identity_sha256 == manifest.identity_sha256
    assert captured["argv"] == (
        "/opt/venv/bin/python3",
        "-I",
        "-m",
        "amd_inference_opt.vllm_environment",
    )
    environment = captured["env"]
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert captured["cwd"] == spec.cwd
    assert captured["stdin"] is subprocess.DEVNULL
