"""Production runtime inspector for native vLLM processes.

``LocalRuntimeInspector`` deliberately proves only PID/executable/environment.
This adapter adds a fresh installed-package manifest check and, when requested,
an outer-container RepoDigest + cgroup-v2 binding supplied by a trusted launcher.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, ValidationError, field_validator, model_validator

from .models import StrictModel, utc_now
from .vllm_adapter import (
    LocalRuntimeInspector,
    RuntimeVerificationStatus,
    SpawnedProcess,
    VLLMRuntimeEvidence,
    VLLMServerSpec,
)
from .vllm_environment import (
    VLLMEnvironmentManifest,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_OCI_DIGEST = re.compile(r"[^@\s]+@sha256:([0-9a-f]{64})")


class VLLMRuntimeInspectionError(RuntimeError):
    """Runtime provenance cannot be established without guessing."""


class TargetEnvironmentCapture(Protocol):
    """Recapture package bytes from the exact lexical Python in the server spec."""

    def capture(
        self,
        spec: VLLMServerSpec,
        declared: VLLMEnvironmentManifest,
    ) -> VLLMEnvironmentManifest: ...


class SubprocessTargetEnvironmentCapture:
    """Run the project-owned capture module inside the target virtual environment."""

    def __init__(self, *, timeout_seconds: float = 900.0) -> None:
        self.timeout_seconds = timeout_seconds

    def capture(
        self,
        spec: VLLMServerSpec,
        declared: VLLMEnvironmentManifest,
    ) -> VLLMEnvironmentManifest:
        environment = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        argv = (
            spec.argv[0],
            "-I",
            "-m",
            "amd_inference_opt.vllm_environment",
        )
        try:
            result = subprocess.run(
                argv,
                cwd=spec.cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise VLLMRuntimeInspectionError(
                "target vLLM Python could not execute the isolated environment probe"
            ) from error
        if result.returncode != 0:
            detail = result.stderr[-2000:].strip()
            raise VLLMRuntimeInspectionError(
                "target vLLM Python environment probe failed"
                + (f": {detail}" if detail else "")
            )
        if len(result.stdout.encode("utf-8")) > 128 * 1024 * 1024:
            raise VLLMRuntimeInspectionError(
                "target vLLM Python environment probe output exceeded 128 MiB"
            )
        try:
            payload = json.loads(result.stdout)
            observed = VLLMEnvironmentManifest.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as error:
            raise VLLMRuntimeInspectionError(
                "target vLLM Python returned an invalid environment manifest"
            ) from error
        expected_lexical = Path(os.path.abspath(spec.argv[0]))
        if observed.python_executable != expected_lexical:
            raise VLLMRuntimeInspectionError(
                "target probe did not run through the exact lexical Python executable"
            )
        if observed.probe_sha256 != declared.probe_sha256:
            raise VLLMRuntimeInspectionError(
                "target environment probe implementation differs from the declared probe"
            )
        return observed


class VLLMContainerRuntimeManifest(StrictModel):
    """Observation exported by the trusted outer-container launcher."""

    schema_id: Literal["gpuopt.vllm-container-runtime.v1"] = Field(alias="schema")
    container_id: str
    container_init_pid: int = Field(gt=0)
    container_image_id: str
    repo_digests: list[str] = Field(min_length=1)
    cgroup_v2_binding: str
    devices: list[str] = Field(default_factory=list)
    captured_at: str = Field(default_factory=lambda: utc_now().isoformat())

    @field_validator("container_id", "container_image_id", "cgroup_v2_binding")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(character in normalized for character in "\x00\r\n"):
            raise ValueError("container identity fields must be non-empty single-line text")
        return normalized

    @field_validator("repo_digests")
    @classmethod
    def valid_repo_digests(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("container RepoDigests must be unique")
        if any(_OCI_DIGEST.fullmatch(value) is None for value in values):
            raise ValueError("container RepoDigests must use name@sha256:<digest>")
        return values

    @field_validator("devices")
    @classmethod
    def valid_devices(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(not value.strip() for value in values):
            raise ValueError("container devices must be unique non-empty values")
        return values

    @model_validator(mode="after")
    def valid_unified_cgroup(self) -> VLLMContainerRuntimeManifest:
        if not self.cgroup_v2_binding.startswith("/"):
            raise ValueError("cgroup_v2_binding must be an absolute unified cgroup path")
        return self


def _load_json_model(path: Path, model: type[StrictModel]) -> StrictModel:
    lexical = path.expanduser()
    if lexical.is_symlink():
        raise VLLMRuntimeInspectionError(f"runtime manifest cannot be a symlink: {lexical}")
    try:
        stat = lexical.stat(follow_symlinks=False)
        if not lexical.is_file() or stat.st_size > 128 * 1024 * 1024:
            raise VLLMRuntimeInspectionError(
                f"runtime manifest is not a bounded regular file: {lexical}"
            )
        payload = lexical.read_text(encoding="utf-8")
        parsed = model.model_validate_json(payload)
        after = lexical.stat(follow_symlinks=False)
    except (OSError, UnicodeError, ValidationError) as error:
        raise VLLMRuntimeInspectionError(f"invalid runtime manifest: {lexical}") from error
    before_identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise VLLMRuntimeInspectionError("runtime manifest changed while it was read")
    return parsed


def _unified_cgroup(proc_root: Path, pid: int) -> str:
    try:
        lines = (proc_root / str(pid) / "cgroup").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise VLLMRuntimeInspectionError("cannot read process cgroup membership") from error
    values = [line[3:] for line in lines if line.startswith("0::/")]
    if len(values) != 1:
        raise VLLMRuntimeInspectionError("process has no unique cgroup-v2 membership")
    return "/" + values[0].lstrip("/")


class VerifiedVLLMRuntimeInspector:
    """Bind PID evidence to fresh package bytes and optional outer image proof."""

    def __init__(
        self,
        environment_manifest: str | Path,
        *,
        container_manifest: str | Path | None = None,
        proc_root: str | Path = "/proc",
        environment_capture: TargetEnvironmentCapture | None = None,
    ) -> None:
        self.environment_manifest_path = Path(environment_manifest)
        self.container_manifest_path = (
            Path(container_manifest) if container_manifest is not None else None
        )
        self.proc_root = Path(proc_root)
        self.local = LocalRuntimeInspector(proc_root=self.proc_root)
        self.environment_capture = environment_capture or SubprocessTargetEnvironmentCapture()
        self._preflight_environment: dict[
            str, tuple[VLLMEnvironmentManifest, VLLMEnvironmentManifest]
        ] = {}

    def _capture_environment(
        self, spec: VLLMServerSpec
    ) -> tuple[VLLMEnvironmentManifest, VLLMEnvironmentManifest]:
        declared = _load_json_model(
            self.environment_manifest_path, VLLMEnvironmentManifest
        )
        assert isinstance(declared, VLLMEnvironmentManifest)
        if declared.identity_sha256 != spec.environment_manifest_sha256:
            raise VLLMRuntimeInspectionError(
                "environment manifest identity differs from the server spec"
            )
        if declared.python_executable_sha256 != spec.native_executable_sha256:
            raise VLLMRuntimeInspectionError(
                "environment manifest Python executable differs from the server spec"
            )
        required_names = {name.lower() for name in declared.required_distributions}
        missing_runtime_evidence = {"vllm", "torch", "amdsmi"} - required_names
        if missing_runtime_evidence:
            raise VLLMRuntimeInspectionError(
                "MI300X environment manifest must bind required distributions: "
                + ", ".join(sorted(missing_runtime_evidence))
            )
        expected_lexical = Path(os.path.abspath(spec.argv[0]))
        if declared.python_executable != expected_lexical:
            raise VLLMRuntimeInspectionError(
                "environment manifest was captured through a different lexical Python"
            )
        observed = self.environment_capture.capture(spec, declared)
        if observed.identity_sha256 != declared.identity_sha256:
            raise VLLMRuntimeInspectionError(
                "installed vLLM/PyTorch package bytes changed after environment capture"
            )
        return declared, observed

    def preflight(self, spec: VLLMServerSpec) -> None:
        """Finish the expensive package recapture before a paid GPU process is spawned."""

        self._preflight_environment.pop(spec.request_hash, None)
        self._preflight_environment[spec.request_hash] = self._capture_environment(spec)

    def _environment(
        self, spec: VLLMServerSpec
    ) -> tuple[VLLMEnvironmentManifest, VLLMEnvironmentManifest]:
        evidence = self._preflight_environment.get(spec.request_hash)
        if evidence is None:
            raise VLLMRuntimeInspectionError(
                "target environment was not preflighted before the server process started"
            )
        return evidence

    def _container(
        self, spec: VLLMServerSpec, process: SpawnedProcess
    ) -> VLLMContainerRuntimeManifest | None:
        if spec.image_digest is None:
            if self.container_manifest_path is not None:
                raise VLLMRuntimeInspectionError(
                    "native server cannot claim an outer-container manifest"
                )
            return None
        if self.container_manifest_path is None:
            raise VLLMRuntimeInspectionError(
                "outer image digest requires a trusted container runtime manifest"
            )
        manifest = _load_json_model(
            self.container_manifest_path, VLLMContainerRuntimeManifest
        )
        assert isinstance(manifest, VLLMContainerRuntimeManifest)
        expected = spec.identity.image_sha256
        observed = {
            match.group(1)
            for value in manifest.repo_digests
            if (match := _OCI_DIGEST.fullmatch(value)) is not None
        }
        if expected not in observed:
            raise VLLMRuntimeInspectionError(
                "trusted container manifest does not contain the requested RepoDigest"
            )
        process_binding = _unified_cgroup(self.proc_root, process.pid)
        if process_binding != manifest.cgroup_v2_binding:
            raise VLLMRuntimeInspectionError(
                "spawned vLLM process is outside the observed container cgroup"
            )
        return manifest

    def inspect(
        self,
        spec: VLLMServerSpec,
        process: SpawnedProcess,
    ) -> VLLMRuntimeEvidence:
        base = self.local.inspect(spec, process)
        if base.verification is RuntimeVerificationStatus.MISMATCH:
            return base
        # Package and container evidence cannot fill a gap in the actual PID's
        # executable or environment. Only complete native proof may be upgraded
        # when LocalRuntimeInspector could not verify an outer image digest.
        if (
            base.native_executable_matches is not True
            or base.declared_environment_matches is not True
            or base.unset_environment_absent is not True
        ):
            return replace(
                base,
                verification=RuntimeVerificationStatus.UNVERIFIED,
                reason=base.reason or "native process identity evidence is incomplete",
            )
        try:
            declared, _ = self._environment(spec)
            container = self._container(spec, process)
        except VLLMRuntimeInspectionError as error:
            return replace(
                base,
                verification=RuntimeVerificationStatus.UNVERIFIED,
                reason=str(error),
            )
        updates: dict[str, object] = {
            "verification": RuntimeVerificationStatus.VERIFIED,
            "observed_environment_manifest_sha256": declared.identity_sha256,
            "reason": None,
        }
        if container is not None:
            updates.update(
                {
                    "container_id": container.container_id,
                    "container_init_pid": container.container_init_pid,
                    "container_image_id": container.container_image_id,
                    "container_binding_kind": "cgroup_v2",
                    "process_binding_id": container.cgroup_v2_binding,
                    "container_binding_id": container.cgroup_v2_binding,
                    "container_repo_digests": tuple(container.repo_digests),
                    "container_devices": tuple(container.devices),
                    "image_digest_matches": True,
                }
            )
        return replace(base, **updates)


__all__ = [
    "SubprocessTargetEnvironmentCapture",
    "TargetEnvironmentCapture",
    "VLLMContainerRuntimeManifest",
    "VLLMRuntimeInspectionError",
    "VerifiedVLLMRuntimeInspector",
]
