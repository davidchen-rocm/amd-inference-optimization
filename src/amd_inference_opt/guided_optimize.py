"""Resolve a small human request into one complete OptimizationTask."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .architecture import cmake_gfx_target
from .eval_suites import load_frozen_suite
from .llama_cpp import sha256_file
from .model_library import (
    ModelCatalogEntry,
    ModelLibraryCatalog,
    ModelLibraryError,
    ModelRole,
)
from .models import (
    AccuracyRequirement,
    BenchmarkProtocol,
    CampaignKind,
    ChangePolicy,
    EnvironmentRequirements,
    GPUTarget,
    MCPConfig,
    ModelTarget,
    OptimizationObjective,
    OptimizationTask,
    PerformanceMetricRequirement,
    QualityConstraints,
    ROCmTarget,
    RuntimeTarget,
    TaskBudgets,
    WorkloadConfig,
)
from .project_config import ProjectConfig


class GuidedOptimizationError(RuntimeError):
    """A guided request cannot be resolved without user-visible action."""


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_").lower()
    return normalized or "model"


def default_task_id(entry: ModelCatalogEntry) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    variant = entry.quantization or entry.variant or "model"
    return f"{_slug(entry.model_name)}-{_slug(variant)}-{stamp}"


def resolve_runnable_model(
    catalog: ModelLibraryCatalog,
    reference: str,
    baseline_quantization: str | None,
) -> ModelCatalogEntry:
    selected = catalog.resolve(reference)
    if selected.role == ModelRole.DERIVED:
        if baseline_quantization is not None and (
            selected.quantization or ""
        ).upper() != baseline_quantization.upper():
            raise GuidedOptimizationError(
                "selected derived model does not match --baseline"
            )
        return selected
    if baseline_quantization is None:
        raise GuidedOptimizationError(
            "an origin model requires --baseline (for example Q6_K)"
        )
    matches = [
        entry
        for entry in catalog.entries
        if entry.role == ModelRole.DERIVED
        and (entry.quantization or "").upper() == baseline_quantization.upper()
        and (
            entry.origin_model_id == selected.id
            or entry.model_name.casefold() == selected.model_name.casefold()
        )
    ]
    if not matches:
        expected = (
            catalog.model_root
            / "derived"
            / selected.model_name
            / baseline_quantization
        )
        raise GuidedOptimizationError(
            "MODEL_PREPARATION_REQUIRED: no matching derived GGUF; expected under "
            f"{expected}"
        )
    if len(matches) != 1:
        rendered = ", ".join(entry.id for entry in matches)
        raise GuidedOptimizationError(
            f"multiple derived models match; choose one exact model id: {rendered}"
        )
    return matches[0]


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        shell=False,
        timeout=30,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise GuidedOptimizationError(
            f"cannot resolve llama.cpp commit: {result.stderr.strip()[-500:]}"
        )
    return result.stdout.strip()


def _cmake_coordinates(build_dir: Path) -> tuple[list[str], str]:
    cache = build_dir / "CMakeCache.txt"
    values: dict[str, str] = {}
    target_entries = 0
    if cache.is_file() and not cache.is_symlink():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line or line.startswith(("//", "#")):
                continue
            key_type, value = line.split("=", 1)
            key = key_type.split(":", 1)[0]
            if key in {
                "CMAKE_BUILD_TYPE",
                "GGML_HIP",
                "GGML_HIP_GRAPHS",
                "GGML_HIP_MMQ_MFMA",
                "GGML_HIP_NO_VMM",
                "AMDGPU_TARGETS",
            }:
                if key == "AMDGPU_TARGETS":
                    target_entries += 1
                values[key] = value
    if target_entries > 1:
        raise GuidedOptimizationError(
            "llama.cpp CMake cache contains multiple AMDGPU_TARGETS entries"
        )
    flags = [f"-D{key}={value}" for key, value in sorted(values.items())]
    try:
        gfx = cmake_gfx_target(flags)
    except ValueError as error:
        raise GuidedOptimizationError(
            f"llama.cpp build has no valid single architecture coordinate: {error}"
        ) from error
    return flags, gfx


def _origin_entry(
    catalog: ModelLibraryCatalog, selected: ModelCatalogEntry
) -> ModelCatalogEntry | None:
    if selected.origin_model_id:
        try:
            origin = catalog.resolve(selected.origin_model_id)
        except ModelLibraryError:
            return None
        return origin if origin.role == ModelRole.ORIGIN else None
    matches = [
        entry
        for entry in catalog.entries
        if entry.role == ModelRole.ORIGIN
        and entry.model_name.casefold() == selected.model_name.casefold()
    ]
    return matches[0] if len(matches) == 1 else None


def _quality_metadata(
    project_root: Path,
    model_sha256: str,
    project: ProjectConfig,
    quality_suites: tuple[str, ...],
    quality_environment: dict[str, str],
) -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[2]
    math_manifest = repository_root / "fixtures/q8-runtime-quality/manifest.json"
    supported = {"math-100.v1", "general-100.v1"}
    unknown = sorted(set(quality_suites) - supported)
    if unknown:
        raise GuidedOptimizationError(
            f"unsupported quality suite(s): {', '.join(unknown)}"
        )
    if not quality_suites:
        raise GuidedOptimizationError("at least one quality suite is required")
    general_dir = project_root / ".gpuopt/eval-suites/general-100.v1"
    if not math_manifest.is_file():
        raise GuidedOptimizationError(f"math quality fixture is missing: {math_manifest}")
    if "general-100.v1" in quality_suites:
        try:
            load_frozen_suite(general_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            raise GuidedOptimizationError(
                "general-100.v1 is not prepared; run "
                f"`gpuopt eval prepare general-100.v1 --project {project_root}`"
            ) from exc
    evaluator = repository_root / "tools/q8_runtime_quality_eval.py"
    command = [
        sys.executable,
        str(evaluator),
        "--baseline",
        "{baseline_binary}",
        "--candidate",
        "{candidate_binary}",
        "--model",
        "{model}",
        "--fixture-manifest",
        str(math_manifest),
        "--expected-model-sha256",
        model_sha256,
        "--math-limit",
        "100",
        "--output",
        "{output}",
    ]
    if "general-100.v1" in quality_suites:
        output_index = command.index("--expected-model-sha256")
        command[output_index:output_index] = [
            "--general-suite-dir",
            str(general_dir),
        ]
    return {
        "quality_command_json": json.dumps(command, separators=(",", ":")),
        "quality_env_json": json.dumps(
            quality_environment,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "quality_cwd": str(repository_root),
        "quality_timeout_seconds": "14400",
        "quality_suites": json.dumps(quality_suites),
        "quality_evaluator_sha256": sha256_file(evaluator),
    }


def build_optimization_task(
    *,
    project_root: Path,
    project: ProjectConfig,
    catalog: ModelLibraryCatalog,
    model: ModelCatalogEntry,
    task_id: str | None = None,
    minimum_improvement_percent: float = 1.0,
    max_experiments: int = 6,
    quality_suites: tuple[str, ...] | None = None,
) -> OptimizationTask:
    """Build the complete, validated local llama.cpp optimization task."""

    if model.role != ModelRole.DERIVED or model.format.value != "GGUF":
        raise GuidedOptimizationError("optimization requires a derived GGUF model")
    repository = project.llama_cpp_repo
    build_dir = project.llama_cpp_build_dir
    if repository is None or not repository.is_dir() or repository.is_symlink():
        raise GuidedOptimizationError("configured llama.cpp repository is missing or unsafe")
    if build_dir is None or not build_dir.is_dir() or build_dir.is_symlink():
        raise GuidedOptimizationError("configured llama.cpp build directory is missing or unsafe")
    binary = build_dir / "bin/llama-bench"
    for name in ("llama-bench", "llama-cli", "llama-perplexity"):
        tool = build_dir / f"bin/{name}"
        if not tool.is_file() or tool.is_symlink():
            raise GuidedOptimizationError(f"required llama.cpp tool is missing: {tool}")
    flags, gfx = _cmake_coordinates(build_dir)
    if project.gpu_gfx_target is not None and project.gpu_gfx_target != gfx:
        raise GuidedOptimizationError(
            "configured gpu_gfx_target does not match the llama.cpp AMDGPU_TARGETS "
            f"coordinate: {project.gpu_gfx_target} != {gfx}"
        )
    gfx = project.gpu_gfx_target or gfx
    origin = _origin_entry(catalog, model)
    selected_quality_suites = tuple(quality_suites or project.default_quality_suites)
    accuracy_requirements = []
    if "math-100.v1" in selected_quality_suites:
        accuracy_requirements.append(
            AccuracyRequirement(
                metric="math_accuracy", max_drop_percentage_points=2.0
            )
        )
    if "general-100.v1" in selected_quality_suites:
        accuracy_requirements.append(
            AccuracyRequirement(
                metric="general_accuracy", max_drop_percentage_points=2.0
            )
        )
    mcp_command = list(project.rocm_mcp_command)
    executable_token = mcp_command[0]
    if "/" in executable_token:
        resolved_mcp = Path(executable_token).expanduser().resolve()
        available = resolved_mcp.is_file() and os.access(resolved_mcp, os.X_OK)
    else:
        located = shutil.which(executable_token)
        resolved_mcp = Path(located).resolve() if located else Path(executable_token)
        available = located is not None
    if not available:
        raise GuidedOptimizationError(
            "configured ROCm MCP command is not executable; fix it with "
            "`gpuopt config set rocm-mcp-command /absolute/path/to/rocm-agent-mcp`"
        )
    mcp_command[0] = str(resolved_mcp)
    rocm_library_path = f"{binary.parent.resolve()}:/opt/rocm/core-7.14/lib:/opt/rocm/lib"
    quality_environment = {
        "HSA_VISIBLE_DEVICES": str(project.gpu_device),
        "HIP_VISIBLE_DEVICES": str(project.gpu_device),
        "ROCR_VISIBLE_DEVICES": str(project.gpu_device),
    }
    metadata = {
        "model_catalog_id": model.id,
        "model_packed_bytes": str(model.size),
        "model_provenance_status": model.provenance_status.value,
        "benchmark_generation_tokens": "[128,512]",
        "benchmark_extra_args": "[]",
        "prepared_source_path": str(repository.resolve()),
        "rocm_library_path": rocm_library_path,
        "profile_capture_limits": json.dumps(
            {
                "max_trace_bytes": 200_000_000,
                "max_trace_files": 64,
                "max_events_per_type": 10_000,
                "max_percentile_samples_per_kernel": 20,
            },
            separators=(",", ":"),
        ),
        **_quality_metadata(
            project_root,
            model.sha256,
            project,
            selected_quality_suites,
            quality_environment,
        ),
    }
    if origin is not None:
        metadata.update(
            {
                "origin_model_path": str(origin.model_path),
                "origin_model_sha256": origin.sha256,
            }
        )
    build_flags_hash = hashlib.sha256("\0".join(flags).encode()).hexdigest()
    metadata["build_flags_hash"] = build_flags_hash
    return OptimizationTask(
        id=task_id or default_task_id(model),
        campaign_kind=CampaignKind.LLAMA_CPP_CONSUMER_AMD_FINAL,
        model=ModelTarget(
            path=model.model_path,
            sha256=model.sha256,
            architecture=model.architecture or (origin.architecture if origin else None),
            quantization=model.quantization,
        ),
        runtime=RuntimeTarget(
            repo_path=repository,
            base_commit=_git_commit(repository),
            build_flags=flags,
            build_dir=build_dir,
            prepared_binary_path=binary,
            prepared_binary_sha256=sha256_file(binary),
        ),
        gpu=GPUTarget(gfx_target=gfx, device_id=project.gpu_device),
        rocm=ROCmTarget(),
        workload=WorkloadConfig(seed=20260823),
        benchmark=BenchmarkProtocol(
            warmup_runs=1,
            sample_count=5,
            required_metrics=[
                "tokens_per_second_tg128",
                "tokens_per_second_tg512",
            ],
        ),
        objective=OptimizationObjective(
            minimum_improvement_percent=minimum_improvement_percent,
            metric_requirements=[
                PerformanceMetricRequirement(
                    metric="tokens_per_second_tg128",
                    minimum_improvement_percent=minimum_improvement_percent,
                ),
                PerformanceMetricRequirement(
                    metric="tokens_per_second_tg512",
                    minimum_improvement_percent=minimum_improvement_percent,
                ),
            ],
        ),
        quality=QualityConstraints(
            max_ppl_regression_percent=0.5,
            accuracy_requirements=accuracy_requirements,
        ),
        change_policy=ChangePolicy(
            locked_coordinates=[
                "model_sha256",
                "benchmark_protocol",
                "gpu_device",
            ]
        ),
        environment=EnvironmentRequirements(
            require_fresh_capture=True,
            require_run_identity=True,
            required_run_match_fields=[
                "protocol_hash",
                "model_sha256",
                "environment_hash",
            ],
        ),
        budgets=TaskBudgets(max_experiments=max_experiments),
        mcp=MCPConfig(command=mcp_command),
        metadata=metadata,
    )


__all__ = [
    "GuidedOptimizationError",
    "build_optimization_task",
    "default_task_id",
    "resolve_runnable_model",
]
