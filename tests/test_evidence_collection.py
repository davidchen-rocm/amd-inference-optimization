from __future__ import annotations

import json
from pathlib import Path

from amd_inference_opt.evidence_catalog import (
    EVIDENCE_PROVIDERS,
    ProviderAvailability,
    ProviderIntegration,
    probe_evidence_capabilities,
)
from amd_inference_opt.external_evidence import (
    IntelliKitTool,
    RooflineArchitectureProvenance,
    RooflineProvenanceKind,
    import_intellikit_json,
    import_magpie_benchmark_report,
    import_tracelens_roofline_csv,
)
from amd_inference_opt.kernel_manifest import (
    AttributionStatus,
    ShapeAttributionRule,
    TensorCoordinate,
    build_kernel_shape_manifest,
)
from amd_inference_opt.runtime_health import capture_runtime_health


def test_provider_catalogue_distinguishes_installed_from_integrated(monkeypatch) -> None:
    by_id = {provider.id: provider for provider in EVIDENCE_PROVIDERS}
    assert by_id["rocm-issue-agent"].integration == ProviderIntegration.DIRECT
    assert by_id["magpie"].integration == ProviderIntegration.ADAPTER
    assert (
        by_id["intellikit"].integration
        == ProviderIntegration.AVAILABLE_NOT_INTEGRATED
    )
    assert {item.id for item in by_id["intellikit"].capabilities} >= {
        "kernel-isolation",
        "hardware-counters",
        "source-line-stalls",
        "hsa-isa-inspection",
        "kernel-correctness",
    }

    monkeypatch.setattr("amd_inference_opt.evidence_catalog.shutil.which", lambda _: None)
    monkeypatch.setattr("amd_inference_opt.evidence_catalog._module_available", lambda _: False)
    snapshot = probe_evidence_capabilities()
    probes = {item.provider_id: item for item in snapshot.providers}
    assert probes["intellikit"].availability == ProviderAvailability.UNAVAILABLE
    assert probes["gpuopt-local-health"].availability == ProviderAvailability.AVAILABLE
    assert (
        probes["hyperloom-host-probe"].availability
        == ProviderAvailability.UNKNOWN
    )


def test_magpie_import_requires_real_requests_and_does_not_invent_zero_telemetry(
    tmp_path: Path,
) -> None:
    report = tmp_path / "benchmark_report.json"
    report.write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "model": "fixture/model",
                "throughput": {
                    "request_throughput": 2.5,
                    "output_throughput": 100.0,
                    "total_token_throughput": 1100.0,
                    "completed_requests": 40,
                },
                "latency": {"ttft": {"mean_ms": 10, "p99_ms": 20}},
                "gpu_monitor": [{"temperature_c": 60}, {"temperature_c": 64}],
                "gap_analysis": {
                    "top_kernels": [
                        {
                            "name": "hot",
                            "calls": 4,
                            "self_cuda_total_us": 80,
                            "pct_total": 50,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    result = import_magpie_benchmark_report(report)

    assert result.gate_eligible is True
    assert result.completed_requests == 40
    assert result.latency_ms["ttft.mean_ms"] == 10
    assert result.gap_kernels[0].average_duration_us == 20
    assert result.telemetry is not None
    assert result.telemetry.average_temperature_c == 62
    assert result.telemetry.average_power_w is None
    assert "power" in result.telemetry.missing_fields


def test_tracelens_and_intellikit_imports_remain_analysis_only(tmp_path: Path) -> None:
    roofline = tmp_path / "roofline.csv"
    roofline.write_text(
        "Name,Calls,Self CUDA total (us),% Total,arithmetic_intensity,bound_type,M,N,K\n"
        "gemm_kernel,10,500,75,12.5,memory,128,4096,4096\n",
        encoding="utf-8",
    )
    trace = import_tracelens_roofline_csv(
        roofline,
        architecture=RooflineArchitectureProvenance(
            kind=RooflineProvenanceKind.STATIC_FALLBACK,
            architecture="gfx942",
            source="bundled architecture table",
        ),
        phase="decode",
    )
    assert trace.gate_eligible is False
    assert trace.kernels[0].m == 128
    assert trace.kernels[0].bound == "memory"
    assert trace.architecture.kind == RooflineProvenanceKind.STATIC_FALLBACK

    metrix = tmp_path / "metrix.json"
    metrix.write_text(
        json.dumps({"schema": "metrix.fixture.v1", "status": "completed", "kernel_name": "hot"}),
        encoding="utf-8",
    )
    envelope = import_intellikit_json(
        metrix,
        tool=IntelliKitTool.METRIX,
        capability="hardware-counters",
    )
    assert envelope.gate_eligible is False
    assert envelope.source_schema == "metrix.fixture.v1"
    assert "tool-specific" in envelope.warnings[0]


def _kernel_payload() -> dict:
    return {
        "schema": "gpuopt.raw-kernel-evidence.v1",
        "kernels": [
            {
                "kernel_id": "shape-shared",
                "name": "mul_mat_vec_q6",
                "grid": [393216, 8, 1],
                "workgroup": [32, 8, 1],
                "dispatch_count": 36,
                "total_duration_ns": 3600,
                "gpu_kernel_time_share_percent": 70,
                "resource_usage": {"vgpr_count": 24, "lds_bytes": 2048},
            },
            {
                "kernel_id": "shape-shared",
                "name": "mul_mat_vec_q6",
                "grid": [131072, 8, 1],
                "workgroup": [32, 8, 1],
                "dispatch_count": 36,
                "total_duration_ns": 1200,
                "gpu_kernel_time_share_percent": 20,
                "resource_usage": {"vgpr_count": 16, "lds_bytes": 1024},
            },
        ],
    }


def test_shape_manifest_is_config_driven_and_preserves_unresolved_variants() -> None:
    rule = ShapeAttributionRule(
        id="ffn-gate-up",
        kernel_name_regex="mul_mat_vec_q6",
        grid=(393216, 8, 1),
        workgroup=(32, 8, 1),
        graph_variant="decode-capture-0",
        node_ordinal=17,
        coordinate=TensorCoordinate(
            operator="ffn_gate_up",
            phase="decode",
            dtype="q6_k",
            quantization="Q6_K",
            m=1,
            n=12288,
            k=4096,
        ),
        evidence_basis="locked model geometry and exact launch signature",
    )

    first = build_kernel_shape_manifest(_kernel_payload(), [rule])
    second = build_kernel_shape_manifest(_kernel_payload(), [rule])

    assert first.resolved_variants == 1
    assert first.unresolved_variants == 1
    assert first.variants[0].coordinate.n == 12288
    assert first.variants[0].resource_usage.vgpr_count == 24
    assert first.variants[1].attribution_status == AttributionStatus.UNRESOLVED
    assert first.variants[0].signature == second.variants[0].signature


def test_runtime_health_is_bounded_and_redacts_process_arguments(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    process = proc / "123"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(
        b"/opt/llama-bench\0-m\0/private/model.gguf\0--secret-token\0value\0"
    )
    snapshot = capture_runtime_health(
        workspace=tmp_path,
        proc_root=proc,
        extra_paths=[tmp_path / "exists", tmp_path / "missing"],
    )

    assert snapshot.processes_known is True
    assert snapshot.relevant_processes[0].command_name == "llama-bench"
    serialized = snapshot.model_dump_json()
    assert "private/model" not in serialized
    assert "secret-token" not in serialized
    assert snapshot.paths[1].status == "MISSING"
