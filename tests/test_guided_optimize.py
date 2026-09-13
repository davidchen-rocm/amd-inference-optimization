from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from amd_inference_opt.cli import app
from amd_inference_opt.eval_suites import DatasetSourceV1, prepare_general_100
from amd_inference_opt.guided_optimize import (
    GuidedOptimizationError,
    build_optimization_task,
    resolve_runnable_model,
)
from amd_inference_opt.model_library import scan_model_library, show_model
from amd_inference_opt.project_config import (
    initialize_project_config,
    load_project_config,
)

runner = CliRunner()


def _rows(source: DatasetSourceV1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(25):
        if source.source_id == "mmlu_general":
            rows.append(
                {
                    "id": f"mmlu-{index}",
                    "question": f"MMLU {index}?",
                    "choices": ["a", "b", "c", "d"],
                    "answer": index % 4,
                }
            )
        elif source.source_id == "arc_challenge":
            rows.append(
                {
                    "id": f"arc-{index}",
                    "question": f"ARC {index}?",
                    "choices": {
                        "label": ["A", "B", "C", "D"],
                        "text": ["a", "b", "c", "d"],
                    },
                    "answerKey": "ABCD"[index % 4],
                }
            )
        elif source.source_id == "hellaswag":
            rows.append(
                {
                    "ind": index,
                    "ctx": f"HellaSwag {index}",
                    "endings": ["a", "b", "c", "d"],
                    "label": str(index % 4),
                }
            )
        else:
            rows.append(
                {
                    "id": f"wino-{index}",
                    "sentence": f"Wino {index} _.",
                    "option1": "Alice",
                    "option2": "Bob",
                    "answer": str(index % 2 + 1),
                }
            )
    return rows


def _git_repository(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Tests"], check=True)
    (path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n")
    subprocess.run(["git", "-C", str(path), "add", "CMakeLists.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _guided_fixture(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    model_root = tmp_path / "models"
    origin = model_root / "origin" / "Tiny"
    origin.mkdir(parents=True)
    (origin / "model.safetensors").write_bytes(b"origin")
    (origin / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"tensor": "model.safetensors"}}),
        encoding="utf-8",
    )
    (origin / "config.json").write_text(
        json.dumps({"model_type": "qwen", "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )
    derived_dir = model_root / "derived" / "Tiny" / "Q6_K"
    derived_dir.mkdir(parents=True)
    model_path = derived_dir / "Tiny-Q6_K.gguf"
    model_path.write_bytes(b"GGUF-fixture")

    repository = tmp_path / "llama.cpp"
    commit = _git_repository(repository)
    build = tmp_path / "llama-build"
    binary_dir = build / "bin"
    binary_dir.mkdir(parents=True)
    for name in ("llama-bench", "llama-cli", "llama-perplexity"):
        tool = binary_dir / name
        tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        tool.chmod(0o755)
    (build / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\nGGML_HIP:BOOL=ON\nAMDGPU_TARGETS:STRING=gfx1201\n",
        encoding="utf-8",
    )
    mcp = tmp_path / "fake-rocm-agent-mcp"
    mcp.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mcp.chmod(0o755)
    initialize_project_config(
        project_root,
        model_root=model_root,
        llama_cpp_repo=repository,
        llama_cpp_build_dir=build,
        gpu_device=1,
        rocm_mcp_command=[str(mcp)],
    )
    prepare_general_100(
        project_root / ".gpuopt/eval-suites/general-100.v1",
        loader=lambda source: _rows(source),
    )
    project = load_project_config(project_root)
    catalog = scan_model_library(project_root)
    return project_root, project, catalog, model_path, commit


def test_resolve_origin_to_requested_derived_baseline(tmp_path: Path) -> None:
    project_root, _, catalog, model_path, _ = _guided_fixture(tmp_path)
    origin = show_model(catalog, "origin/Tiny")

    selected = resolve_runnable_model(catalog, origin.id, "q6_k")

    assert selected.model_path == model_path.resolve()
    assert selected.quantization == "Q6_K"
    with pytest.raises(GuidedOptimizationError, match="requires --baseline"):
        resolve_runnable_model(catalog, origin.id, None)
    with pytest.raises(GuidedOptimizationError, match="MODEL_PREPARATION_REQUIRED"):
        resolve_runnable_model(catalog, origin.id, "Q4_K_M")
    assert project_root.is_dir()


def test_resolve_derived_rejects_conflicting_baseline(tmp_path: Path) -> None:
    _, _, catalog, _, _ = _guided_fixture(tmp_path)

    with pytest.raises(GuidedOptimizationError, match="does not match --baseline"):
        resolve_runnable_model(catalog, "derived/Tiny/Q6_K", "Q5_K_M")


def test_build_task_is_complete_without_executing_gpu_work(tmp_path: Path) -> None:
    project_root, project, catalog, model_path, commit = _guided_fixture(tmp_path)
    selected = resolve_runnable_model(catalog, "origin/Tiny", "Q6_K")

    task = build_optimization_task(
        project_root=project_root,
        project=project,
        catalog=catalog,
        model=selected,
        task_id="guided-unit",
        minimum_improvement_percent=2.5,
        max_experiments=3,
    )

    assert task.id == "guided-unit"
    assert task.model.path == model_path.resolve()
    assert task.model.quantization == "Q6_K"
    assert task.model.architecture == "qwen"
    assert task.runtime.base_commit == commit
    assert task.runtime.build_flags == [
        "-DAMDGPU_TARGETS=gfx1201",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_HIP=ON",
    ]
    assert task.gpu.device_id == 1
    assert task.gpu.gfx_target == "gfx1201"
    assert task.mcp.command == [str((tmp_path / "fake-rocm-agent-mcp").resolve())]
    assert task.budgets.max_experiments == 3
    assert task.objective.minimum_improvement_percent == 2.5
    assert task.benchmark.required_metrics == [
        "tokens_per_second_tg128",
        "tokens_per_second_tg512",
    ]
    assert [item.metric for item in task.quality.accuracy_requirements] == [
        "math_accuracy",
        "general_accuracy",
    ]
    quality_command = json.loads(task.metadata["quality_command_json"])
    assert "--general-suite-dir" in quality_command
    assert task.metadata["quality_suites"] == '["math-100.v1", "general-100.v1"]'
    assert json.loads(task.metadata["quality_env_json"]) == {
        "HSA_VISIBLE_DEVICES": "1",
        "HIP_VISIBLE_DEVICES": "1",
        "ROCR_VISIBLE_DEVICES": "1",
    }


def test_build_task_accepts_explicit_matching_gfx942_target(tmp_path: Path) -> None:
    project_root, project, catalog, _, _ = _guided_fixture(tmp_path)
    (project.llama_cpp_build_dir / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\nGGML_HIP:BOOL=ON\nAMDGPU_TARGETS:STRING=gfx942\n",
        encoding="utf-8",
    )
    project.gpu_gfx_target = "gfx942"
    selected = resolve_runnable_model(catalog, "derived/Tiny/Q6_K", None)

    task = build_optimization_task(
        project_root=project_root,
        project=project,
        catalog=catalog,
        model=selected,
    )

    assert task.gpu.gfx_target == "gfx942"
    assert "-DAMDGPU_TARGETS=gfx942" in task.runtime.build_flags


@pytest.mark.parametrize(
    "target_lines",
    [
        "",
        "AMDGPU_TARGETS:STRING=gfx942;gfx1201\n",
        "AMDGPU_TARGETS:STRING=mi300x\n",
        "AMDGPU_TARGETS:STRING=gfx942\nAMDGPU_TARGETS:STRING=gfx942\n",
    ],
)
def test_build_task_fails_closed_for_ambiguous_cmake_target(
    tmp_path: Path,
    target_lines: str,
) -> None:
    project_root, project, catalog, _, _ = _guided_fixture(tmp_path)
    (project.llama_cpp_build_dir / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\nGGML_HIP:BOOL=ON\n" + target_lines,
        encoding="utf-8",
    )
    selected = resolve_runnable_model(catalog, "derived/Tiny/Q6_K", None)

    with pytest.raises(GuidedOptimizationError, match="AMDGPU_TARGETS|architecture"):
        build_optimization_task(
            project_root=project_root,
            project=project,
            catalog=catalog,
            model=selected,
        )


def test_build_task_rejects_explicit_gfx_target_build_mismatch(tmp_path: Path) -> None:
    project_root, project, catalog, _, _ = _guided_fixture(tmp_path)
    project.gpu_gfx_target = "gfx942"
    selected = resolve_runnable_model(catalog, "derived/Tiny/Q6_K", None)

    with pytest.raises(GuidedOptimizationError, match="does not match"):
        build_optimization_task(
            project_root=project_root,
            project=project,
            catalog=catalog,
            model=selected,
        )


def test_build_task_reports_missing_runtime_and_suite_prerequisites(
    tmp_path: Path,
) -> None:
    project_root, project, catalog, _, _ = _guided_fixture(tmp_path)
    selected = resolve_runnable_model(catalog, "derived/Tiny/Q6_K", None)
    missing_runtime = project.model_copy(update={"llama_cpp_build_dir": tmp_path / "missing"})

    with pytest.raises(GuidedOptimizationError, match="build directory is missing"):
        build_optimization_task(
            project_root=project_root,
            project=missing_runtime,
            catalog=catalog,
            model=selected,
        )

    suite = project_root / ".gpuopt/eval-suites/general-100.v1"
    (suite / "cases.jsonl").write_bytes(b"tampered")
    with pytest.raises(GuidedOptimizationError, match="gpuopt eval prepare general-100.v1"):
        build_optimization_task(
            project_root=project_root,
            project=project,
            catalog=catalog,
            model=selected,
        )


def test_optimize_dry_run_resolves_everything_without_creating_task(
    tmp_path: Path,
) -> None:
    project_root, project, _, _, _ = _guided_fixture(tmp_path)

    result = runner.invoke(
        app,
        [
            "optimize",
            "--model",
            "origin/Tiny",
            "--baseline",
            "Q6_K",
            "--task-id",
            "guided-dry-run",
            "--project",
            str(project_root),
            "--minimum-improvement",
            "2.5",
            "--max-experiments",
            "3",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["schema"] == "gpuopt.guided-optimization-request.v1"
    assert payload["dry_run"] is True
    assert payload["baseline_quantization"] == "Q6_K"
    assert payload["task"]["id"] == "guided-dry-run"
    assert payload["task"]["budgets"]["max_experiments"] == 3
    assert payload["quality_suites"] == ["math-100.v1", "general-100.v1"]
    assert payload["next_action"]["kind"] == "CREATE_TASK"
    command = payload["next_action"]["command"]
    assert "--no-execute" in command
    assert "--dry-run" not in command
    assert not (project.store_root / "guided-dry-run").exists()

    command_argv = shlex.split(command)
    assert command_argv[0] == "gpuopt"
    created = runner.invoke(app, command_argv[1:])
    assert created.exit_code == 0, created.output
    created_payload = json.loads(created.stdout)
    assert created_payload["task_id"] == "guided-dry-run"
    assert created_payload["execution_started"] is False
    assert created_payload["next_action"]["kind"] == "RESUME"


def test_optimize_no_execute_persists_provenance_and_one_resume_action(
    tmp_path: Path,
) -> None:
    project_root, project, _, model_path, _ = _guided_fixture(tmp_path)

    result = runner.invoke(
        app,
        [
            "optimize",
            "--model",
            "derived/Tiny/Q6_K",
            "--task-id",
            "guided-created",
            "--project",
            str(project_root),
            "--quality",
            "math-100.v1",
            "--no-execute",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    task_dir = project.store_root / "guided-created"
    assert payload["task_id"] == "guided-created"
    assert payload["stage"] == "CREATE_TASK"
    assert payload["execution_started"] is False
    assert payload["quality_suites"] == ["math-100.v1"]
    assert payload["next_action"] == {
        "kind": "RESUME",
        "command": f"gpuopt resume guided-created --store {project.store_root}",
    }
    assert (task_dir / "task.json").is_file()
    assert (task_dir / "state/workflow.json").is_file()
    assert (task_dir / "artifacts/guided-request.json").is_file()
    provenance = json.loads(
        (task_dir / "artifacts/model-input-provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["model_path"] == str(model_path.resolve())
    assert provenance["packed_bytes"] == model_path.stat().st_size
    assert provenance["quantization"] == "Q6_K"


def test_optimize_noninteractive_errors_include_exact_remediation(
    tmp_path: Path,
) -> None:
    project_root, project, _, _, _ = _guided_fixture(tmp_path)

    missing_model = runner.invoke(app, ["optimize", "--project", str(project_root), "--dry-run"])
    missing_baseline = runner.invoke(
        app,
        [
            "optimize",
            "--model",
            "origin/Tiny",
            "--project",
            str(project_root),
            "--dry-run",
        ],
    )
    project.rocm_mcp_command = ["definitely-missing-rocm-mcp"]
    from amd_inference_opt.project_config import save_project_config

    save_project_config(project_root, project)
    missing_mcp = runner.invoke(
        app,
        [
            "optimize",
            "--model",
            "derived/Tiny/Q6_K",
            "--project",
            str(project_root),
            "--dry-run",
        ],
    )

    assert missing_model.exit_code == 2
    assert "--model is required when stdin is not interactive" in missing_model.stderr
    assert missing_baseline.exit_code == 2
    assert "origin model requires --baseline" in missing_baseline.stderr
    assert missing_mcp.exit_code == 2
    assert "gpuopt config set rocm-mcp-command" in missing_mcp.stderr
