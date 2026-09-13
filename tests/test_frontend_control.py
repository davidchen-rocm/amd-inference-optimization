from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from amd_inference_opt.frontend_api import ControlPlaneReader, ReadOnlyStoreSource
from amd_inference_opt.frontend_control import (
    DraftOptionsV1,
    DraftStore,
    KernelExperimentMode,
    MixedBitDraftV1,
    MixedBitMode,
    OptimizationDraftRequestV1,
    QualityDraftV1,
    build_model_catalog,
)
from amd_inference_opt.models import OptimizationTask, WorkflowRecord
from amd_inference_opt.store import ExperimentStore
from amd_inference_opt.ui_server import create_control_plane_server

MODEL_SHA = "a" * 64


def _reader(root: Path) -> ControlPlaneReader:
    store = ExperimentStore(root)
    for identifier, quantization in (("q5-run", "Q5_K_M"), ("q6-run", "Q6_K")):
        store.create_task(
            OptimizationTask.model_validate(
                {
                    "id": identifier,
                    "model": {
                        "path": "/models/Qwen.gguf",
                        "sha256": MODEL_SHA,
                        "architecture": "qwen",
                        "quantization": quantization,
                    },
                    "runtime": {"repo_path": "/src/llama.cpp", "base_commit": "abc"},
                    "gpu": {"gfx_target": "gfx1201", "name": "RX 9070 XT"},
                    "mcp": {"command": ["/opt/rocm-agent-mcp"]},
                }
            )
        )
        store.save_workflow(WorkflowRecord(task_id=identifier))
    return ControlPlaneReader([ReadOnlyStoreSource("local", root)])


def _request(
    address: tuple[str, int],
    method: str,
    path: str,
    *,
    payload: object | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    body = json.dumps(payload).encode() if payload is not None else None
    request_headers = dict(headers or {})
    if body is not None:
        request_headers["Content-Length"] = str(len(body))
    connection = http.client.HTTPConnection(*address, timeout=3)
    connection.request(method, path, body=body, headers=request_headers)
    response = connection.getresponse()
    decoded = json.loads(response.read())
    connection.close()
    return response.status, decoded


def _draft() -> OptimizationDraftRequestV1:
    return OptimizationDraftRequestV1.model_validate(
        {
            "name": "Qwen mixed-bit",
            "model": {
                "source_id": "local",
                "run_id": "q6-run",
                "model_sha256": MODEL_SHA,
            },
        }
    )


def test_model_catalog_groups_quantizations_and_keeps_run_coordinates(tmp_path: Path) -> None:
    catalog = build_model_catalog(_reader(tmp_path / "store"))

    assert len(catalog.items) == 1
    model = catalog.items[0]
    assert model.quantizations == ["Q5_K_M", "Q6_K"]
    assert {(run.source_id, run.run_id) for run in model.runs} == {
        ("local", "q5-run"),
        ("local", "q6-run"),
    }
    assert model.origin.role == "ORIGIN"
    assert {variant.role for variant in model.variants} == {"BASELINE"}
    assert model.summary == "ORIGIN · 2 variants · 2 runs · 0 methods"


def test_draft_defaults_encode_weight_and_kernel_evidence_policy() -> None:
    draft = _draft()
    assignments = {item.group.value: item.precision for item in draft.mixed_bit.assignments}

    assert draft.mixed_bit.mode == MixedBitMode.SENSITIVITY_GUIDED
    assert assignments["attention_q"] == "Q6_K"
    assert assignments["ffn_gate"] == "Q5_K"
    assert draft.kernel_mapping.mode == KernelExperimentMode.EVIDENCE_THEN_TUNE
    assert "matrix_shape_wave_mapping" in {
        item.value for item in draft.llama_cpp.techniques
    }
    assert draft.quality.math_problem_count == 100
    assert draft.benchmark.generation_lengths == [128, 512]


def test_balanced_quality_draft_is_additive_and_legacy_default_remains() -> None:
    legacy = QualityDraftV1()
    balanced = QualityDraftV1(policy="balanced-200.v1")
    options = DraftOptionsV1()

    assert legacy.policy == "provisional-math-100.v1"
    assert legacy.general_problem_count is None
    assert legacy.suite_ids == ("math-100.v1",)
    assert balanced.general_problem_count == 100
    assert balanced.suite_ids == ("math-100.v1", "general-100.v1")
    assert options.quality_policies == [
        "provisional-math-100.v1",
        "balanced-200.v1",
    ]


def test_catalog_projects_origin_lineage_performance_and_accuracy(tmp_path: Path) -> None:
    root = tmp_path / "lineage"
    store = ExperimentStore(root)
    task = OptimizationTask.model_validate(
        {
            "id": "qwen35-run",
            "model": {
                "path": "/models/Q8_0.gguf",
                "sha256": MODEL_SHA,
                "architecture": "qwen",
                "quantization": "Q8_0",
            },
            "runtime": {"repo_path": "/src/llama.cpp", "base_commit": "abc"},
            "gpu": {"gfx_target": "gfx1201"},
            "mcp": {"command": ["/opt/rocm-agent-mcp"]},
            "metadata": {
                "hf_model": "Qwen/Qwen3.5-9B",
                "hf_revision": "revision-1",
            },
        }
    )
    store.create_task(task)
    spec = store.save_evidence_json(
        task.id,
        "campaign/candidates/q7mix",
        {
            "base_quantization": "Q6_K",
            "effective_bpw": 7.11,
            "provenance": {
                "source_quantization": "BF16",
                "source_model": {"sha256": "b" * 64},
            },
        },
        producer="test",
    )
    performance = store.save_evidence_json(
        task.id,
        "campaign/q7mix/performance",
        {
            "metric_improvements_percent": {"tg128": 16.4, "tg512": 15.8},
            "checks": [
                {"name": "performance.tg128", "detail": "69.25 vs 59.48 tok/s"},
                {"name": "performance.tg512", "detail": "68.99 vs 59.59 tok/s"},
            ],
        },
        producer="test",
    )
    quality = store.save_evidence_json(
        task.id,
        "campaign/q7mix/quality-input",
        {
            "quality_policy": "balanced-200.v1",
            "baseline": {
                "math_correct": 70,
                "math_total": 100,
                "general_correct": 72,
                "general_total": 100,
                "perplexity": 2.0,
            },
            "candidate": {
                "math_correct": 68,
                "math_total": 100,
                "general_correct": 71,
                "general_total": 100,
                "perplexity": 2.01,
            },
        },
        producer="test",
    )
    store.save_json(
        task.id,
        "state/campaign.json",
        {
            "current_stage": "COMPLETE",
            "status": "EXPERIMENTAL_ACCEPTED",
            "candidates": [
                {
                    "id": "mixed-q7mix",
                    "label": "Q6/Q8 mixed 7.11 bpw",
                    "strategy": "mixed_bit",
                    "disposition": "EXPERIMENTAL_ACCEPTED",
                    "selected_as_winner": True,
                    "spec_artifact": spec.model_dump(mode="json"),
                    "result_artifacts": [
                        performance.model_dump(mode="json"),
                        quality.model_dump(mode="json"),
                    ],
                    "reasons": ["quality and performance pass"],
                }
            ],
        },
        producer="test",
    )

    reader = ControlPlaneReader([ReadOnlyStoreSource("local", root)])
    model = build_model_catalog(reader).items[0]
    winner = next(item for item in model.variants if item.role == "ACCEPTED")

    assert model.name == "Qwen/Qwen3.5-9B"
    assert model.origin.role == "ORIGIN"
    assert model.origin.sha256 == "b" * 64
    assert winner.performance.tg128 == 69.25
    assert winner.performance.tg512_delta_percent == 15.8
    assert winner.quality.accuracy_percent == 68.0
    assert winner.quality.accuracy_delta_points == -2.0
    assert winner.quality.policy == "balanced-200.v1"
    assert winner.quality.math_accuracy_percent == 68.0
    assert winner.quality.general_accuracy_percent == 71.0
    assert winner.quality.general_accuracy_delta_points == -1.0


def test_draft_rejects_arbitrary_fields_and_inconsistent_modes() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        OptimizationDraftRequestV1.model_validate(
            {
                "name": "unsafe",
                "model": {"source_id": "local", "run_id": "q6-run"},
                "shell_command": "rm -rf anything",
            }
        )
    with pytest.raises(ValidationError, match="disabled mixed-bit"):
        MixedBitDraftV1(mode=MixedBitMode.DISABLED)


def test_sqlite_draft_store_round_trip_is_immutable(tmp_path: Path) -> None:
    store = DraftStore(tmp_path / "control" / "drafts.sqlite3")
    record = store.create(_draft())

    assert store.get(record.id) == record
    assert store.list().items == [record]
    assert len(record.content_sha256) == 64


def test_http_builder_requires_csrf_and_creates_only_a_draft(tmp_path: Path) -> None:
    reader = _reader(tmp_path / "runs")
    draft_store = DraftStore(tmp_path / "control" / "drafts.sqlite3")
    server = create_control_plane_server(reader, port=0, draft_store=draft_store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = (str(server.server_address[0]), int(server.server_address[1]))
    experiment_root = tmp_path / "runs" / "q6-run" / "experiments"
    before_experiments = sorted(path.name for path in experiment_root.iterdir())
    try:
        status, meta = _request(address, "GET", "/api/v1/builder/meta")
        assert status == 200
        assert meta["draft_writes_enabled"] is True
        assert meta["experiment_execution_enabled"] is False

        payload = _draft().model_dump(mode="json")
        status, failure = _request(
            address,
            "POST",
            "/api/v1/drafts",
            payload=payload,
            headers={"Content-Type": "application/json"},
        )
        assert status == 403
        assert failure["code"] == "invalid_csrf"

        status, created = _request(
            address,
            "POST",
            "/api/v1/drafts",
            payload=payload,
            headers={
                "Content-Type": "application/json",
                "X-GPUOPT-CSRF": str(meta["csrf_token"]),
            },
        )
        assert status == 201
        assert created["status"] == "DRAFT"
        assert draft_store.get(created["id"]).request.name == "Qwen mixed-bit"
        assert sorted(path.name for path in experiment_root.iterdir()) == before_experiments
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
