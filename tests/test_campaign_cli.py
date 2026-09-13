from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from amd_inference_opt.campaign import CampaignEngine
from amd_inference_opt.campaign_models import (
    CampaignConfig,
    CampaignStrategy,
    SharedBaselineReference,
)
from amd_inference_opt.campaign_planning import materialize_campaign_plan
from amd_inference_opt.campaign_store import CampaignStore
from amd_inference_opt.cli import app
from amd_inference_opt.quality_policy import ProvisionalMath100Policy
from amd_inference_opt.store import ExperimentStore

runner = CliRunner()


def _hash(character: str) -> str:
    return character * 64


def _config() -> dict[str, object]:
    artifacts = {
        "source_model": {"path": "/models/qwen-bf16", "sha256": _hash("a")},
        "calibration_corpus": {"path": "/data/train.jsonl", "sha256": _hash("b")},
        "importance_matrix": {"path": "/models/imatrix.gguf", "sha256": _hash("c")},
        "quantizer_binary": {"path": "/bin/llama-quantize", "sha256": _hash("d")},
        "llama_cpp_commit": "a7a6d0d",
    }
    return {
        "schema_version": 1,
        "id": "campaign-unit",
        "task": {
            "schema_version": 1,
            "id": "campaign-task",
            "model": {
                "path": "/models/qwen-q6.gguf",
                "sha256": _hash("e"),
                "architecture": "qwen",
                "quantization": "Q6_K",
            },
            "runtime": {"repo_path": "/src/llama.cpp", "base_commit": "a7a6d0d"},
            "mcp": {"command": ["/opt/rocm-agent-mcp"]},
        },
        "planning_inputs": {
            "model_geometry": {
                "block_count": 36,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "attention_head_count": 32,
                "kv_head_count": 8,
                "vocab_size": 151936,
                "max_context_tokens": 32768,
            },
            "mixed_bit_provenance": artifacts,
            "kv_capabilities": {
                "supported_cache_types": ["f16", "q8_0", "q4_0"],
                "flash_attention": True,
                "max_context_tokens": 32768,
            },
        },
    }


def _invoke(arguments: list[str]) -> dict[str, object]:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _write_input(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_campaign_plan_accepts_manifest_bound_prepared_mixed_candidate(
    tmp_path: Path,
) -> None:
    document = _config()
    planning = document["planning_inputs"]
    assert isinstance(planning, dict)
    planning["mixed_bit_reference_quantizations"] = ["Q8_0"]
    planning["mixed_bit_prepared_candidates"] = [
        {
            "id": "q7mix",
            "label": "Q6/Q8 mixed 7.11 bpw",
            "base_quantization": "Q6_K",
            "effective_bpw": 7.11,
            "model": {
                "path": "/models/q7mix.gguf",
                "sha256": _hash("7"),
                "size_bytes": 7000,
            },
            "preparation_manifest": {
                "path": "/models/q7mix.preparation.json",
                "sha256": _hash("8"),
                "size_bytes": 700,
            },
            "tensor_assignment": {
                "default_weight": "Q6_K",
                "full_attention_q_k_v_output": "Q8_0",
            },
        }
    ]
    campaign_store = CampaignStore(tmp_path / "prepared-store")
    record = CampaignEngine().new(CampaignConfig.model_validate(document))
    record = CampaignEngine().advance(record)
    inspection = campaign_store.store.create_task(record.config.task)
    assert inspection.is_dir()
    inspection_ref = campaign_store.store.save_evidence_json(
        record.task_id,
        "campaign/inspection",
        {"gpu": "gfx1201"},
        producer="test",
    )
    record = CampaignEngine().advance(record, evidence=[inspection_ref])
    baseline_ref = campaign_store.store.save_evidence_json(
        record.task_id,
        "campaign/baseline",
        {"tg128": 60.0},
        producer="test",
    )
    record = CampaignEngine().advance(
        record,
        shared_baselines=[
            SharedBaselineReference(
                id="q8-baseline",
                task_id=record.task_id,
                kind="decode-and-quality",
                artifact=baseline_ref,
            )
        ],
    )

    candidates, _ = materialize_campaign_plan(record, campaign_store.store)

    prepared = next(candidate for candidate in candidates if candidate.id == "mixed-q7mix")
    assert prepared.strategy == CampaignStrategy.MIXED_BIT
    assert prepared.label == "Q6/Q8 mixed 7.11 bpw"
    assert prepared.spec_artifact is not None
    spec = campaign_store.store.load_json(record.task_id, prepared.spec_artifact.path)
    assert spec["effective_bpw"] == 7.11
    assert spec["model"]["sha256"] == _hash("7")


def test_campaign_cli_runs_recorded_control_plane_to_experimental_accept(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    config_path = _write_input(tmp_path / "campaign.json", _config())
    created = _invoke(
        [
            "campaign",
            "create",
            "--config",
            str(config_path),
            "--store",
            str(store_root),
        ]
    )
    assert created["quality_policy"] == "provisional-math-100.v1"

    advanced = _invoke(
        ["campaign", "run", "campaign-unit", "--store", str(store_root)]
    )
    assert advanced["stage"] == "INSPECT_TARGET"
    waiting = _invoke(
        ["campaign", "run", "campaign-unit", "--store", str(store_root)]
    )
    assert waiting["action"] == "waiting_for_inspection"

    store = ExperimentStore(store_root)
    inspection = store.save_evidence_json(
        "campaign-task",
        "campaign/inspection",
        {"gpu": "gfx1201", "runtime": "llama.cpp"},
        producer="test",
    )
    inspection_input = _write_input(
        tmp_path / "inspection.json", {"evidence_paths": [inspection.path]}
    )
    baseline_stage = _invoke(
        [
            "campaign",
            "run",
            "campaign-unit",
            "--input",
            str(inspection_input),
            "--store",
            str(store_root),
        ]
    )
    assert baseline_stage["stage"] == "CAPTURE_SHARED_BASELINES"

    baseline = store.save_evidence_json(
        "campaign-task",
        "campaign/shared-q6-baseline",
        {"tg128": 79.63, "tg512": 79.25},
        producer="test",
    )
    baseline_input = _write_input(
        tmp_path / "baseline.json",
        {
            "shared_baselines": [
                {
                    "id": "q6-decode",
                    "task_id": "campaign-task",
                    "kind": "decode-performance-and-quality",
                    "artifact": baseline.model_dump(mode="json"),
                    "coordinate_hash": _hash("f"),
                }
            ]
        },
    )
    planned_stage = _invoke(
        [
            "campaign",
            "run",
            "campaign-unit",
            "--input",
            str(baseline_input),
            "--store",
            str(store_root),
        ]
    )
    assert planned_stage["stage"] == "PLAN_CANDIDATES"

    plan = _invoke(
        ["campaign", "plan", "campaign-unit", "--store", str(store_root)]
    )
    assert plan["candidate_count"] == 12
    assert plan["stage"] == "RUN_MIXED_BIT"

    accepted_performance = store.save_evidence_json(
        "campaign-task",
        "campaign/performance-accepted",
        {
            "outcome": "ACCEPT",
            "checks": [{"name": "performance", "passed": True, "detail": "+10%"}],
            "reasons": ["performance gate passed"],
            "improvement_percent": 10.4,
        },
        producer="test",
    )
    rejected_performance = store.save_evidence_json(
        "campaign-task",
        "campaign/performance-rejected",
        {
            "outcome": "REJECT",
            "checks": [{"name": "performance", "passed": False, "detail": "slower"}],
            "reasons": ["performance gate failed"],
            "improvement_percent": -1,
        },
        producer="test",
    )
    protocol_hash = ProvisionalMath100Policy().protocol.protocol_hash
    provisional_quality = store.save_evidence_json(
        "campaign-task",
        "campaign/provisional-quality",
        {
            "baseline": {
                "math_correct": 62,
                "math_total": 100,
                "perplexity": 2.34,
                "greedy_correct": 3,
                "greedy_total": 8,
                "protocol_hash": protocol_hash,
            },
            "candidate": {
                "math_correct": 64,
                "math_total": 100,
                "perplexity": 2.341,
                "greedy_correct": 5,
                "greedy_total": 8,
                "protocol_hash": protocol_hash,
            },
        },
        producer="test",
    )
    while True:
        status = _invoke(
            ["campaign", "status", "campaign-unit", "--store", str(store_root)]
        )
        stage = status["current_stage"]
        if stage not in {"RUN_MIXED_BIT", "RUN_SHAPE_KERNEL", "RUN_KV_CACHE"}:
            break
        running = next(
            candidate
            for candidate in status["candidates"]
            if candidate["disposition"] == "RUNNING"
        ) if any(
            candidate["disposition"] == "RUNNING"
            for candidate in status["candidates"]
        ) else None
        if running is None:
            _invoke(["campaign", "run", "campaign-unit", "--store", str(store_root)])
            continue
        accepted = running["id"] == "mixed-q5_k_m"
        candidate_input = _write_input(
            tmp_path / "candidate.json",
            {
                "candidate_id": running["id"],
                "performance_gate_path": (
                    accepted_performance.path if accepted else rejected_performance.path
                ),
                **(
                    {"provisional_quality_path": provisional_quality.path}
                    if accepted
                    else {}
                ),
                "reasons": ["recorded integration evidence"],
            },
        )
        _invoke(
            [
                "campaign",
                "run",
                "campaign-unit",
                "--input",
                str(candidate_input),
                "--store",
                str(store_root),
            ]
        )

    assert status["current_stage"] == "SELECT_WINNERS"
    selected = _invoke(
        ["campaign", "run", "campaign-unit", "--store", str(store_root)]
    )
    assert selected["selected_candidate_ids"] == ["mixed-q5_k_m"]

    combination_evidence = store.save_evidence_json(
        "campaign-task",
        "campaign/combination",
        {"model": "Q5_K_M", "kernel": "stock", "kv": "f16"},
        producer="test",
    )
    build_input = _write_input(
        tmp_path / "combination.json",
        {
            "evidence_paths": [combination_evidence.path],
            "compatibility": {
                "disposition": "COMPATIBLE",
                "candidate_ids": ["mixed-q5_k_m"],
                "checks": {"model_runtime": True},
            },
            "final_combination": {
                "disposition": "BUILT",
                "candidate_ids": ["mixed-q5_k_m"],
                "components": {"model": "mixed-q5_k_m", "kv_cache": "f16"},
                "checks": {"build": True},
            },
        },
    )
    final_stage = _invoke(
        [
            "campaign",
            "run",
            "campaign-unit",
            "--input",
            str(build_input),
            "--store",
            str(store_root),
        ]
    )
    assert final_stage["stage"] == "FINAL_VALIDATION"

    final_evidence = store.save_evidence_json(
        "campaign-task",
        "campaign/final-validation",
        {"quality_questions": 100, "quality_passed": True},
        producer="test",
    )
    final_input = _write_input(
        tmp_path / "final.json",
        {
            "evidence_paths": [final_evidence.path],
            "final_status": "EXPERIMENTAL_ACCEPTED",
            "final_combination": {
                "disposition": "VALIDATED",
                "candidate_ids": ["mixed-q5_k_m"],
                "components": {"model": "mixed-q5_k_m", "kv_cache": "f16"},
                "checks": {"performance": True, "provisional_quality": True},
            },
        },
    )
    completed = _invoke(
        [
            "campaign",
            "run",
            "campaign-unit",
            "--input",
            str(final_input),
            "--store",
            str(store_root),
        ]
    )
    assert completed["status"] == "EXPERIMENTAL_ACCEPTED"
    report = json.loads(
        (store.task_dir("campaign-task") / "reports/campaign-final.json").read_text()
    )
    assert report["quality_protocol_pending_replacement"] is True
    assert report["production_ready"] is False
