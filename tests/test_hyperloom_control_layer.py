from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from amd_inference_opt.architecture import ArchitectureFamily, architecture_profile
from amd_inference_opt.campaign import CampaignEngine
from amd_inference_opt.campaign_models import (
    CampaignCandidate,
    CampaignConfig,
    CampaignStage,
    CampaignStrategy,
    CandidateDisposition,
    SharedBaselineReference,
)
from amd_inference_opt.campaign_store import CampaignStore
from amd_inference_opt.control_policy import (
    ACTION_CATALOGUE,
    AttemptDisposition,
    OptimizationLedger,
    ProfileRefreshPolicy,
    content_fingerprint,
    record_optimization_attempt,
)
from amd_inference_opt.frontend_api import ReadOnlyStoreSource
from amd_inference_opt.models import ArtifactRef, GPUTarget
from amd_inference_opt.platform_probe import probe_local_platform
from amd_inference_opt.session_breakdown import SessionBreakdownV1


def _hash(character: str) -> str:
    return character * 64


def _artifact(path: str, character: str = "a") -> ArtifactRef:
    return ArtifactRef(
        path=path,
        sha256=_hash(character),
        size=1,
        producer="test",
        media_type="application/json",
    )


def _config(task_id: str = "hyperloom-local") -> CampaignConfig:
    return CampaignConfig.model_validate(
        {
            "id": "hyperloom-local-campaign",
            "task": {
                "id": task_id,
                "model": {
                    "path": "/models/model.gguf",
                    "sha256": _hash("1"),
                    "architecture": "qwen",
                    "quantization": "Q6_K",
                },
                "runtime": {
                    "repo_path": "/src/llama.cpp",
                    "base_commit": "abc123",
                },
                "gpu": {"gfx_target": "gfx1201", "device_id": 0},
                "mcp": {"command": ["rocm-agent-mcp"]},
            },
            "strategy_order": ["mixed_bit"],
            "planning_inputs": {},
        }
    )


def test_architecture_registry_supports_rdna4_cdna3_and_unknown_targets() -> None:
    local = architecture_profile("gfx1201", board_type="radeon-rx-9070-xt")
    assert local.family == ArchitectureFamily.RDNA4
    assert local.wavefront_size == 32
    assert local.compute_units == 64

    mi300x = architecture_profile("gfx942", board_type="mi300x")
    assert mi300x.family == ArchitectureFamily.CDNA3
    assert mi300x.wavefront_size == 64
    assert mi300x.board_type == "mi300x"

    future = architecture_profile("gfx9999")
    assert future.family == ArchitectureFamily.UNKNOWN
    assert "unrecognized" in future.source

    assert GPUTarget(gfx_target="GFX942").gfx_target == "gfx942"
    with pytest.raises(ValidationError):
        GPUTarget(gfx_target="mi300x")


def test_platform_probe_reads_multi_domain_topology_without_rocm(tmp_path: Path) -> None:
    def write(relative: str, value: str = "1") -> None:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    write("sys/devices/system/cpu/smt/active")
    write("sys/devices/system/cpu/cpu0/topology/physical_package_id", "0")
    write("sys/devices/system/cpu/cpu1/topology/physical_package_id", "1")
    write("sys/devices/system/node/node0/cpulist", "0")
    write("sys/devices/system/node/node1/cpulist", "1")
    write("sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "performance")
    write("sys/devices/system/cpu/cpufreq/boost")
    write("proc/cpuinfo", "model name\t: Fixture CPU\n")
    write("proc/sys/kernel/osrelease", "fixture-kernel")
    write("sys/bus/pci/drivers/amdgpu/0000:01:00.0/vendor")
    write("sys/bus/pci/drivers/amdgpu/0002:02:00.0/vendor")

    result = probe_local_platform(GPUTarget(), root=tmp_path)
    assert result.status == "ok"
    assert result.cpu is not None
    assert result.cpu.sockets == 2
    assert result.cpu.numa_nodes == 2
    assert result.cpu.nps == "NPS1"
    assert result.amdgpu_host_device_count == 2
    assert result.architecture.gfx_target == "gfx1201"


def test_action_catalogue_fingerprint_ledger_and_profile_watermark() -> None:
    assert ACTION_CATALOGUE["shape_kernel"].required_lanes == [
        "workspace_mutation",
        "build",
        "gpu",
        "benchmark",
    ]
    first = content_fingerprint("mixed_bit", {"precision": "Q6", "layers": [1, 2]})
    reordered = content_fingerprint("mixed_bit", {"layers": [1, 2], "precision": "Q6"})
    assert first == reordered

    ledger = record_optimization_attempt(
        OptimizationLedger(),
        candidate_id="mixed-q6",
        action_name="mixed_bit",
        fingerprint=first,
        disposition=AttemptDisposition.REJECTED,
    )
    with pytest.raises(ValueError, match="already been evaluated"):
        record_optimization_attempt(
            ledger,
            candidate_id="renamed-q6",
            action_name="mixed_bit",
            fingerprint=first,
            disposition=AttemptDisposition.REJECTED,
        )

    policy = ProfileRefreshPolicy()
    assert policy.should_refresh(
        baseline_profiled=False,
        cumulative_gain_percent=0,
        last_profile_gain_percent=None,
    )[0]
    assert not policy.should_refresh(
        baseline_profiled=True,
        cumulative_gain_percent=9.99,
        last_profile_gain_percent=0,
    )[0]
    assert policy.should_refresh(
        baseline_profiled=True,
        cumulative_gain_percent=10,
        last_profile_gain_percent=0,
    )[0]


def test_campaign_builds_stack_only_from_selected_accepted_candidate() -> None:
    engine = CampaignEngine()
    record = engine.new(_config())
    record = engine.advance(record)
    record = engine.advance(record, evidence=[_artifact("artifacts/inspect.json")])
    baseline = SharedBaselineReference(
        id="baseline",
        task_id=record.task_id,
        kind="performance",
        artifact=_artifact("artifacts/baseline.json", "b"),
    )
    record = engine.advance(record, shared_baselines=[baseline])
    fingerprint = content_fingerprint("mixed_bit", {"precision": "Q7"})
    candidate = CampaignCandidate(
        id="mixed-q7",
        strategy=CampaignStrategy.MIXED_BIT,
        label="Q7",
        shared_baseline_ids=[baseline.id],
        spec_artifact=_artifact("artifacts/q7-spec.json", "c"),
        content_fingerprint=fingerprint,
    )
    record = engine.advance(
        record,
        candidates=[candidate],
        evidence=[_artifact("artifacts/plan.json", "d")],
    )
    assert record.current_stage == CampaignStage.RUN_MIXED_BIT
    record = engine.record_candidate_result(
        record,
        "mixed-q7",
        CandidateDisposition.EXPERIMENTAL_ACCEPTED,
        artifacts=[_artifact("artifacts/q7-gate.json", "e")],
    )
    assert len(record.optimization_ledger.attempts) == 1
    assert record.optimization_ledger.accepted_stack == []
    record = engine.advance(record)
    assert record.current_stage == CampaignStage.SELECT_WINNERS
    record = engine.advance(record, selected_candidate_ids=["mixed-q7"])
    assert [entry.candidate_id for entry in record.optimization_ledger.accepted_stack] == [
        "mixed-q7"
    ]


def test_campaign_store_auto_persists_platform_and_session_breakdown(tmp_path: Path) -> None:
    store = CampaignStore(tmp_path / "store")
    record = store.create(_config("breakdown-task"))
    assert record.platform_evidence is not None
    assert store.store.verify_artifact(record.task_id, record.platform_evidence)

    report = store.store.load_json(
        record.task_id,
        "reports/session-breakdown.json",
        SessionBreakdownV1,
    )
    assert report.session.campaign_id == record.campaign_id
    assert report.architecture.gfx_target == "gfx1201"
    assert report.control_plane.profile_gain_watermark_percent == 10
    assert "mixed_bit" in {item.name for item in report.control_plane.action_catalogue}
    assert report.information_collection.capability_snapshot is not None
    assert report.information_collection.runtime_health is not None
    assert "intellikit" in {
        item.id for item in report.information_collection.provider_catalogue
    }

    source = ReadOnlyStoreSource("local", tmp_path / "store")
    detail = source.detail(record.task_id)
    assert detail.final_report is not None
    assert detail.final_report.path == "reports/session-breakdown.json"
    assert "rocm-issue-agent" in {item.id for item in detail.evidence_providers}
    assert "intellikit" in {item.id for item in detail.evidence_providers}

    manifest = json.loads(
        (store.store.task_dir(record.task_id) / "artifacts/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "reports/session-breakdown.md" in manifest["artifacts"]
