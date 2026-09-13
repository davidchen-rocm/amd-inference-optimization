from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from amd_inference_opt.cli import app
from amd_inference_opt.eval_suites import DatasetSourceV1, prepare_general_100

runner = CliRunner()


def _json(result) -> dict[str, Any]:
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _model_library(root: Path) -> tuple[Path, Path]:
    model_root = root / "models"
    origin = model_root / "origin" / "Tiny"
    origin.mkdir(parents=True)
    (origin / "model.safetensors").write_bytes(b"origin-weights")
    (origin / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"tensor": "model.safetensors"}}),
        encoding="utf-8",
    )
    (origin / "config.json").write_text(
        json.dumps({"model_type": "qwen", "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )
    derived = model_root / "derived" / "Tiny" / "Q6_K"
    derived.mkdir(parents=True)
    model = derived / "Tiny-Q6_K.gguf"
    model.write_bytes(b"derived-gguf")
    return model_root, model


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


def test_config_commands_are_project_local_and_machine_readable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    model_root, _ = _model_library(tmp_path)

    initialized = _json(
        runner.invoke(
            app,
            [
                "config",
                "init",
                "--project",
                str(project),
                "--model-root",
                str(model_root),
            ],
        )
    )
    assert initialized["schema_name"] == "gpuopt.project-config.v1"
    assert initialized["model_root"] == str(model_root.resolve())

    updated = _json(
        runner.invoke(
            app,
            ["config", "set", "gpu-device", "1", "--project", str(project)],
        )
    )
    shown = _json(runner.invoke(app, ["config", "show", "--project", str(project)]))
    doctor = _json(runner.invoke(app, ["config", "doctor", "--project", str(project)]))

    assert updated["gpu_device"] == shown["gpu_device"] == 1
    assert doctor["status"] == "CHECK"
    checks = {item["name"]: item["status"] for item in doctor["checks"]}
    assert checks == {
        "model_root": "OK",
        "store_root": "OK",
        "llama_cpp_repo": "UNSET",
        "llama_cpp_build_dir": "UNSET",
    }


def test_model_commands_scan_filter_show_and_link_without_gpu(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    model_root, model_path = _model_library(tmp_path)
    _json(
        runner.invoke(
            app,
            [
                "config",
                "init",
                "--project",
                str(project),
                "--model-root",
                str(model_root),
            ],
        )
    )

    scanned = _json(runner.invoke(app, ["model", "scan", "--project", str(project)]))
    listed = _json(
        runner.invoke(
            app,
            [
                "model",
                "list",
                "--role",
                "derived",
                "--quantization",
                "q6_k",
                "--project",
                str(project),
            ],
        )
    )
    shown = _json(
        runner.invoke(
            app,
            ["model", "show", "derived/Tiny/Q6_K", "--project", str(project)],
        )
    )

    assert len(scanned["entries"]) == 2
    assert listed["count"] == 1
    assert shown["role"] == "DERIVED"
    assert shown["model_path"] == str(model_path.resolve())
    assert shown["provenance_status"] == "INFERRED"

    linked = _json(
        runner.invoke(
            app,
            [
                "model",
                "link",
                "derived/Tiny/Q6_K",
                "--origin",
                "origin/Tiny",
                "--project",
                str(project),
            ],
        )
    )
    after = _json(
        runner.invoke(
            app,
            ["model", "show", "derived/Tiny/Q6_K", "--project", str(project)],
        )
    )
    assert linked["derived_alias"] == "derived/Tiny/Q6_K"
    assert after["provenance_status"] == "DECLARED"


def test_eval_commands_report_missing_then_reuse_frozen_suite_offline(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    before = _json(runner.invoke(app, ["eval", "list", "--project", str(project)]))
    statuses = {item["id"]: item["status"] for item in before["items"]}
    assert statuses["math-100.v1"] == "AVAILABLE"
    assert statuses["general-100.v1"] == "MISSING"

    suite_dir = project / ".gpuopt/eval-suites/general-100.v1"
    prepare_general_100(suite_dir, loader=lambda source: _rows(source))
    prepared = _json(
        runner.invoke(
            app,
            ["eval", "prepare", "general-100.v1", "--project", str(project)],
        )
    )
    listed = _json(runner.invoke(app, ["eval", "list", "--project", str(project)]))
    shown = _json(
        runner.invoke(
            app,
            ["eval", "show", "general-100.v1", "--project", str(project)],
        )
    )

    assert prepared["suite"]["total_cases"] == 100
    assert {item["id"]: item["status"] for item in listed["items"]}["general-100.v1"] == "AVAILABLE"
    assert shown["manifest"]["suite_id"] == "general-100.v1"
    assert shown["case_count"] == 100


def test_project_commands_return_clear_errors(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    missing_config = runner.invoke(app, ["model", "scan", "--project", str(project)])
    unknown_suite = runner.invoke(app, ["eval", "show", "unknown.v1", "--project", str(project)])

    assert missing_config.exit_code == 2
    assert "project configuration" in missing_config.stderr
    assert unknown_suite.exit_code == 2
    assert "unknown suite: unknown.v1" in unknown_suite.stderr
