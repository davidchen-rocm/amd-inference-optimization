from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from amd_inference_opt.frontend_api import ReadOnlyStoreSource
from amd_inference_opt.models import OptimizationTask, WorkflowRecord
from amd_inference_opt.optimization_map import (
    CapabilityAvailability,
    CapabilityRunStatus,
    OptimizationTopologyConfig,
    load_default_topology,
)
from amd_inference_opt.store import ExperimentStore


def _store(root: Path) -> ExperimentStore:
    store = ExperimentStore(root)
    store.create_task(
        OptimizationTask.model_validate(
            {
                "id": "map-run",
                "model": {
                    "path": "/models/qwen.gguf",
                    "architecture": "qwen",
                    "quantization": "Q4_K_M",
                },
                "runtime": {"repo_path": "/src/llama.cpp", "base_commit": "abc123"},
                "gpu": {"gfx_target": "gfx1201", "name": "RX 9070 XT"},
                "mcp": {"command": ["/opt/rocm-agent-mcp"]},
            }
        )
    )
    store.save_workflow(WorkflowRecord(task_id="map-run"))
    return store


def test_default_topology_is_config_driven_and_internally_consistent() -> None:
    topology = load_default_topology()
    node_ids = {item.id for item in topology.nodes}

    assert len(topology.nodes) == 34
    assert len(topology.edges) == 46
    assert {item.layer.value for item in topology.nodes} == {
        "MODEL",
        "KERNEL",
        "RUNTIME",
        "EVIDENCE",
        "VALIDATION",
    }
    assert {
        "mixed-precision",
        "weight-layout-fused-dequant",
        "tile-wave-autotune",
        "kv-cache-quantization",
        "hip-graph-ab",
        "buffer-memory-audit",
        "mfma-mmq-evidence",
        "evidence-provider-registry",
        "kernel-shape-manifest",
        "magpie-evidence-import",
        "tracelens-evidence-import",
        "intellikit-evidence-import",
        "runtime-health-evidence",
    } <= node_ids
    availability = {item.id: item.availability for item in topology.nodes}
    assert availability["mixed-precision"] == CapabilityAvailability.IMPLEMENTED
    assert availability["tile-wave-autotune"] == CapabilityAvailability.PLANNED
    assert availability["scheduler-audit"] == CapabilityAvailability.NOT_IMPLEMENTED


def test_topology_rejects_edges_to_unknown_components() -> None:
    payload = load_default_topology().model_dump(mode="json", by_alias=True)
    payload["edges"][0]["target"] = "missing-component"
    with pytest.raises(ValidationError, match="unknown node"):
        OptimizationTopologyConfig.model_validate(payload)


def test_run_projection_separates_framework_availability_from_run_status(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "store")
    store.save_json(
        "map-run",
        "artifacts/baseline.json",
        {"benchmark": {"metrics": {}}},
        producer="test",
    )
    store.save_json(
        "map-run",
        "experiments/q4rdna-split-k/spec.json",
        {
            "hypothesis_id": "split-k-wave32",
            "change": {
                "kind": "runtime_config",
                "description": "Q4_RDNA split-K mapping",
            },
        },
        producer="test",
    )
    store.save_json(
        "map-run",
        "experiments/q4rdna-split-k/e2e-result.json",
        {"status": "SUCCEEDED", "metrics": {}},
        producer="test",
    )
    store.save_json(
        "map-run",
        "experiments/q4rdna-split-k/gate-decision.json",
        {"outcome": "ACCEPT", "reasons": ["performance and quality passed"]},
        producer="test",
    )

    graph = ReadOnlyStoreSource("main", tmp_path / "store").optimization_map("map-run")
    nodes = {item.id: item for item in graph.nodes}

    assert nodes["shared-baseline"].run_status == CapabilityRunStatus.COMPLETE
    assert nodes["split-k"].availability == CapabilityAvailability.PARTIAL
    assert nodes["split-k"].run_status == CapabilityRunStatus.ACCEPT
    assert nodes["q4rdna-hybrid"].run_status == CapabilityRunStatus.ACCEPT
    assert nodes["tile-wave-autotune"].availability == CapabilityAvailability.PLANNED
    assert nodes["tile-wave-autotune"].run_status == CapabilityRunStatus.NOT_STARTED
    assert nodes["scheduler-audit"].run_status == CapabilityRunStatus.UNAVAILABLE
    assert nodes["split-k"].experiment_ids == ["q4rdna-split-k"]
    assert all(edge.source in nodes and edge.target in nodes for edge in graph.edges)
