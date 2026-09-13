from __future__ import annotations

import json
from pathlib import Path

import pytest

from amd_inference_opt.command import CommandResult
from amd_inference_opt.final_report import (
    CapabilityReportEntry,
    CapabilityReportStatus,
    Gfx1201FinalReport,
    persist_gfx1201_final_report,
    render_gfx1201_markdown,
)
from amd_inference_opt.gfx1201_campaign import (
    CapabilityOutcome,
    CapabilityResult,
    Gfx1201CampaignConfig,
    Gfx1201CampaignEngine,
    Gfx1201CampaignError,
    Gfx1201Stage,
)
from amd_inference_opt.hip_graph_ab import (
    HipGraphABOutcome,
    HipGraphABSpec,
    HipGraphRunCoordinates,
    PairedMetricSamples,
    evaluate_hip_graph_ab,
    run_hip_graph_ab,
)
from amd_inference_opt.llama_cpp import sha256_file
from amd_inference_opt.memory_audit import (
    BufferLifetimeEvidence,
    MemoryAuditOutcome,
    MemoryOperationAggregate,
    MemoryOpportunityKind,
    MemoryPatchEvaluation,
    MemoryTraceEvidence,
    audit_memory_reuse,
    evaluate_memory_patch,
    memory_trace_from_rocm,
)
from amd_inference_opt.mfma_mmq import (
    MatrixInstructionFamily,
    MatrixPathEvidence,
    MeaningfulOptimizationGap,
    ShapeCoordinate,
    StallEvidence,
    close_mfma_mmq_evidence,
    extract_kernel_instructions,
    infer_matrix_instruction_family,
    parse_amdgpu_kernel_metadata,
)
from amd_inference_opt.mixed_precision import (
    MixedPrecisionCandidateResult,
    MixedPrecisionMetrics,
    PackedModelAccounting,
    Precision,
    PrecisionRouteRule,
    PrecisionSearchPoint,
    PrecisionSearchSpace,
    SensitivityEvidence,
    TensorInventory,
    TensorInventoryEntry,
    TensorSensitivityScore,
    build_external_policy,
    plan_precision_policies,
    rank_pareto,
)
from amd_inference_opt.mixed_precision_llama import build_llama_quantize_pack_spec
from amd_inference_opt.models import ArtifactRef, MCPConfig, OptimizationTask
from amd_inference_opt.store import ExperimentStore


def _hash(character: str) -> str:
    return character * 64


def _artifact(name: str, character: str = "a", size: int = 10) -> ArtifactRef:
    return ArtifactRef(
        path=f"artifacts/{name}",
        sha256=_hash(character),
        size=size,
        producer="test",
        media_type="application/json",
    )


def _task() -> OptimizationTask:
    return OptimizationTask.model_validate(
        {
            "id": "gfx1201-test",
            "model": {
                "path": "/models/model.gguf",
                "sha256": _hash("1"),
                "architecture": "qwen",
                "quantization": "Q6_K",
            },
            "runtime": {
                "repo_path": "/src/llama.cpp",
                "base_commit": "abc123",
                "build_flags": ["-DGGML_HIP_GRAPHS=ON"],
            },
            "gpu": {"gfx_target": "gfx1201", "name": "RX 9070 XT"},
            "mcp": MCPConfig(command=["rocm-agent-mcp"]).model_dump(),
        }
    )


def _capability(
    name: str, outcome: CapabilityOutcome = CapabilityOutcome.COMPLETE
) -> CapabilityResult:
    return CapabilityResult(
        capability=name,
        outcome=outcome,
        summary="complete",
        evidence=[_artifact(f"{name}.json")],
    )


def test_gfx1201_campaign_enforces_strict_sequence_and_mixed_winner() -> None:
    engine = Gfx1201CampaignEngine()
    record = engine.new(Gfx1201CampaignConfig(id="campaign", task=_task()))
    record = engine.advance(record)
    assert record.current_stage == Gfx1201Stage.INSPECT_TARGET
    with pytest.raises(Gfx1201CampaignError, match="target_inspection"):
        engine.advance(record, result=_capability("hip_graph_ab"))
    record = engine.advance(record, result=_capability("target_inspection"))
    record = engine.advance(record, result=_capability("shared_baselines"))
    with pytest.raises(Gfx1201CampaignError, match="must ACCEPT"):
        engine.advance(
            record,
            result=_capability("mixed_precision", CapabilityOutcome.INCONCLUSIVE),
        )
    record = engine.advance(
        record,
        result=_capability("mixed_precision", CapabilityOutcome.ACCEPT),
        selected_mixed_precision_candidate_id="q5",
    )
    assert record.current_stage == Gfx1201Stage.RUN_HIP_GRAPH_AB
    assert record.selected_mixed_precision_candidate_id == "q5"


def test_gfx1201_campaign_accepts_balanced_quality_policy() -> None:
    config = Gfx1201CampaignConfig(
        id="balanced-campaign",
        task=_task(),
        quality_policy="balanced-200.v1",
    )

    assert config.quality_policy == "balanced-200.v1"


def _inventory() -> TensorInventory:
    return TensorInventory(
        model_sha256=_hash("1"),
        source_artifact=_artifact("source.gguf", "1"),
        entries=[
            TensorInventoryEntry(
                name="blk.0.ffn_gate.weight",
                shape=[4, 8],
                elements=32,
                storage_bytes=64,
                precision=Precision.BF16,
                operator_group="ffn_gate",
            ),
            TensorInventoryEntry(
                name="blk.0.ffn_up.weight",
                shape=[4, 8],
                elements=32,
                storage_bytes=64,
                precision=Precision.BF16,
                operator_group="ffn_up",
            ),
            TensorInventoryEntry(
                name="output.weight",
                shape=[4, 8],
                elements=32,
                storage_bytes=64,
                precision=Precision.BF16,
                operator_group="output",
                protected=True,
            ),
            TensorInventoryEntry(
                name="blk.0.attn_norm.weight",
                shape=[4],
                elements=4,
                storage_bytes=16,
                precision=Precision.F32,
                operator_group="other",
                quantizable=False,
            ),
        ],
    )


def _sensitivity() -> SensitivityEvidence:
    return SensitivityEvidence(
        model_sha256=_hash("1"),
        scores=[
            TensorSensitivityScore(
                tensor_name="blk.0.ffn_gate.weight",
                score=1,
                source_metric="imatrix",
            ),
            TensorSensitivityScore(
                tensor_name="blk.0.ffn_up.weight",
                score=9,
                source_metric="imatrix",
            ),
            TensorSensitivityScore(
                tensor_name="output.weight",
                score=10,
                source_metric="imatrix",
            ),
        ],
        calibration_artifacts=[_artifact("imatrix.gguf", "2")],
    )


def test_mixed_precision_planner_is_bounded_sensitive_and_supports_q4rdna() -> None:
    inventory = _inventory()
    sensitivity = _sensitivity()
    external = build_external_policy(
        "q4rdna-qkv-fallback",
        inventory,
        provider="q4_rdna",
        base_precision=Precision.Q4_RDNA,
        routes=[
            PrecisionRouteRule(
                pattern=r"^output\.weight$",
                precision=Precision.Q6_K,
                reason="protected output fallback",
            )
        ],
        evidence=[_artifact("q4rdna-sidecar", "3")],
    )
    policies = plan_precision_policies(
        inventory,
        sensitivity,
        PrecisionSearchSpace(
            whole_model_precisions=[Precision.Q6_K],
            search_points=[
                PrecisionSearchPoint(
                    id="ffn-half-q4",
                    eligible_groups=["ffn_gate", "ffn_up"],
                    q4_fraction=0.5,
                    q5_fraction=0,
                )
            ],
            max_candidates=3,
        ),
        external_policies=[external],
    )
    assert [policy.id for policy in policies] == [
        "q6_k",
        "ffn-half-q4",
        "q4rdna-qkv-fallback",
    ]
    generated = policies[1]
    assignments = {item.tensor_name: item.precision for item in generated.assignments}
    assert assignments["blk.0.ffn_gate.weight"] == Precision.Q4_K
    assert assignments["blk.0.ffn_up.weight"] == Precision.Q6_K
    assert assignments["output.weight"] == Precision.Q6_K
    assert assignments["blk.0.attn_norm.weight"] == Precision.F32


def test_llama_pack_spec_is_exact_and_hash_bound() -> None:
    policy = plan_precision_policies(
        _inventory(),
        _sensitivity(),
        PrecisionSearchSpace(whole_model_precisions=[Precision.Q5_K], max_candidates=1),
    )[0]
    spec = build_llama_quantize_pack_spec(
        policy,
        quantizer_path="/bin/llama-quantize",
        quantizer_binary=_artifact("quantizer", "4"),
        source_model_path="/models/source.gguf",
        source_model=_artifact("source.gguf", "1"),
        imatrix_path="/models/imatrix.gguf",
        importance_matrix=_artifact("imatrix.gguf", "2"),
        output_path="/models/output.gguf",
        threads=12,
    )
    assert spec.argv[-2:] == ["Q5_K_M", "12"]
    assert "--tensor-type" in spec.argv


def _mixed_result(
    identifier: str, tg: float, bpw: float, *, quality: bool = True
) -> MixedPrecisionCandidateResult:
    policy = plan_precision_policies(
        _inventory(),
        _sensitivity(),
        PrecisionSearchSpace(whole_model_precisions=[Precision.Q6_K], max_candidates=1),
    )[0].model_copy(update={"id": identifier})
    return MixedPrecisionCandidateResult(
        candidate_id=identifier,
        policy=policy,
        accounting=PackedModelAccounting(
            total_elements=100,
            eligible_linear_elements=96,
            logical_packed_weight_bytes=round(bpw * 100 / 8),
            linear_packed_weight_bytes=round(bpw * 96 / 8),
            container_bytes=100,
            effective_bpw=bpw,
            linear_effective_bpw=bpw,
            precision_elements={Precision.Q6_K: 100},
        ),
        metrics=MixedPrecisionMetrics(
            tg128=tg,
            tg512=tg,
            ppl=3.4,
            accuracy=0.6,
            tg128_cv_percent=1,
            tg512_cv_percent=1,
        ),
        performance_stable=True,
        quality_passed=quality,
        evidence=[_artifact(f"{identifier}.json")],
    )


def test_pareto_excludes_quality_failure_and_uses_deterministic_winner() -> None:
    frontier = rank_pareto(
        [
            _mixed_result("fast-large", 110, 6),
            _mixed_result("small", 100, 5),
            _mixed_result("bad-quality", 120, 4, quality=False),
        ]
    )
    assert frontier.frontier_candidate_ids == ["fast-large", "small"]
    assert frontier.selected_candidate_id == "fast-large"
    assert frontier.excluded_candidate_ids == ["bad-quality"]


def _graph_spec() -> HipGraphABSpec:
    return HipGraphABSpec(
        binary_sha256=_hash("1"),
        model_sha256=_hash("2"),
        benchmark_protocol_hash=_hash("3"),
        build_flags=["-DGGML_HIP_GRAPHS=ON"],
    )


def _paired(on: float, off: float) -> PairedMetricSamples:
    return PairedMetricSamples(unit="tokens/s", graph_on=[on] * 7, graph_off=[off] * 7)


@pytest.mark.parametrize(
    ("on", "expected"),
    [
        (103.0, HipGraphABOutcome.MATERIAL_BENEFIT),
        (100.0, HipGraphABOutcome.NO_MATERIAL_EFFECT),
        (98.0, HipGraphABOutcome.HARMFUL),
    ],
)
def test_hip_graph_ab_deterministic_outcomes(on: float, expected: HipGraphABOutcome) -> None:
    metrics = {name: _paired(on, 100) for name in ("pp512", "tg128", "tg512")}
    decision = evaluate_hip_graph_ab(_graph_spec(), metrics, evidence=[_artifact("ab.json")])
    assert decision.outcome == expected


def test_hip_graph_executor_runs_alternating_same_binary_pairs(tmp_path: Path) -> None:
    binary = tmp_path / "llama-bench"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-test")
    store = ExperimentStore(tmp_path / "store")
    store.create_task(_task())
    coordinates = HipGraphRunCoordinates(
        binary_path=binary,
        model_path=model,
        cwd=tmp_path,
    )
    spec = HipGraphABSpec(
        binary_sha256=sha256_file(binary),
        model_sha256=sha256_file(model),
        benchmark_protocol_hash=coordinates.protocol_hash,
        build_flags=["-DGGML_HIP_GRAPHS=ON"],
    )

    class FakeRunner:
        def __init__(self) -> None:
            self.arms: list[str] = []

        def run(self, argv: list[str], **kwargs: object) -> CommandResult:
            environment = kwargs["env"]
            assert isinstance(environment, dict)
            arm = "off" if environment.get("GGML_CUDA_DISABLE_GRAPHS") == "1" else "on"
            self.arms.append(arm)
            rate = 103.0 if arm == "on" else 100.0
            payload = [
                {"n_prompt": 512, "n_gen": 0, "avg_ts": rate, "samples_ts": [rate] * 6},
                {"n_prompt": 0, "n_gen": 128, "avg_ts": rate, "samples_ts": [rate] * 6},
                {"n_prompt": 0, "n_gen": 512, "avg_ts": rate, "samples_ts": [rate] * 6},
            ]
            return CommandResult(
                argv=tuple(argv),
                cwd=str(tmp_path),
                started_at="2026-08-18T00:00:00+00:00",
                duration_seconds=1,
                exit_code=0,
                stdout=json.dumps(payload),
                stderr="",
            )

    runner = FakeRunner()
    decision, reference = run_hip_graph_ab(
        spec,
        coordinates,
        task_id="gfx1201-test",
        store=store,
        runner=runner,  # type: ignore[arg-type]
    )
    assert decision.outcome == HipGraphABOutcome.MATERIAL_BENEFIT
    assert runner.arms[:4] == ["off", "on", "on", "off"]
    assert coordinates.argv[coordinates.argv.index("-r") + 1] == "6"
    assert len(decision.evidence) == 14
    assert store.verify_artifact("gfx1201-test", reference)
    assert decision.metrics[0].graph_on_latency_ms < decision.metrics[0].graph_off_latency_ms


def test_memory_audit_distinguishes_found_no_action_and_inconclusive() -> None:
    evidence = MemoryTraceEvidence(
        workload_succeeded=True,
        trace_complete=True,
        steady_state_isolated=True,
        e2e_duration_ns=1_000,
        coverage_percent=100,
        allocations=[
            MemoryOperationAggregate(
                operation="hipMalloc workspace",
                count=10,
                total_duration_ns=20,
                steady_state=True,
            )
        ],
        artifacts=[_artifact("trace.json")],
    )
    assert audit_memory_reuse(evidence).outcome == MemoryAuditOutcome.OPPORTUNITY_FOUND
    assert (
        audit_memory_reuse(evidence.model_copy(update={"allocations": []})).outcome
        == MemoryAuditOutcome.NO_ACTION
    )
    incomplete = memory_trace_from_rocm(
        {
            "trace_status": "completed",
            "workload_exit_code": 0,
            "memory_allocation_count": 3,
            "warning_details": [],
        },
        e2e_duration_ns=1_000,
        steady_state_isolated=True,
        artifacts=[_artifact("mcp.json")],
    )
    assert audit_memory_reuse(incomplete).outcome == MemoryAuditOutcome.INCONCLUSIVE


def test_memory_audit_detects_evidenced_nonoverlapping_buffer_reuse() -> None:
    evidence = MemoryTraceEvidence(
        workload_succeeded=True,
        trace_complete=True,
        steady_state_isolated=True,
        e2e_duration_ns=10_000,
        coverage_percent=100,
        lifetimes=[
            BufferLifetimeEvidence(
                buffer_id="temporary-a",
                size_bytes=4096,
                first_use_ns=10,
                last_use_ns=100,
                reusable_with_buffer_id="temporary-b",
                estimated_allocation_duration_ns=200,
            ),
            BufferLifetimeEvidence(
                buffer_id="temporary-b",
                size_bytes=4096,
                first_use_ns=200,
                last_use_ns=300,
            ),
        ],
        artifacts=[_artifact("memory-lifetimes.json")],
    )
    result = audit_memory_reuse(evidence)
    assert result.outcome == MemoryAuditOutcome.OPPORTUNITY_FOUND
    assert result.opportunities[0].kind == MemoryOpportunityKind.BUFFER_LIFETIME_REUSE


def test_memory_patch_gate_requires_correctness_latency_and_memory_reduction() -> None:
    evaluation = MemoryPatchEvaluation(
        opportunity_id="workspace",
        correctness_passed=True,
        baseline_tg128=100,
        candidate_tg128=101,
        baseline_tg512=100,
        candidate_tg512=101,
        baseline_kernel_or_runtime_latency_ns=1000,
        candidate_kernel_or_runtime_latency_ns=900,
        baseline_memory_operation_count=10,
        candidate_memory_operation_count=5,
        baseline_memory_duration_ns=200,
        candidate_memory_duration_ns=100,
        evidence=[_artifact("memory-patch.json")],
    )
    assert evaluate_memory_patch(evaluation).outcome.value == "ACCEPT"
    assert (
        evaluate_memory_patch(evaluation.model_copy(update={"correctness_passed": False}))
        .outcome.value
        == "REJECT"
    )


def test_mfma_mmq_uses_actual_isa_and_explicit_unavailable_resources() -> None:
    disassembly = """
000000 <q6_mmq_kernel>:
    v_wmma_f32_16x16x16_f16 v[0:7], v[8:9], v[10:11]
    s_endpgm
000100 <other>:
    s_endpgm
"""
    instructions = extract_kernel_instructions(disassembly, "q6_mmq_kernel")
    assert (
        infer_matrix_instruction_family(instructions, isa_complete=True)
        == MatrixInstructionFamily.WMMA
    )
    resources = parse_amdgpu_kernel_metadata(
        """.name: q6_mmq_kernel
.vgpr_count: 32
.sgpr_count: 24
.group_segment_fixed_size: 4096
.private_segment_fixed_size: 0
.wavefront_size: 32
""",
        "q6_mmq_kernel",
    )
    assert resources.vgpr == 32
    path = MatrixPathEvidence(
        gfx_target="gfx1201",
        shape=ShapeCoordinate(
            id="ffn-gate-up", operator="ffn_gate_up", phase="prefill", n=12288, k=4096
        ),
        selected_kernel="q6_mmq_kernel",
        mmq_enabled=True,
        matrix_instruction_family=MatrixInstructionFamily.WMMA,
        matrix_instructions=instructions,
        isa_complete=True,
        workgroup=[32, 8, 1],
        resources=resources,
        average_duration_ns=100,
        total_duration_ns=1_000,
        gpu_time_share_percent=20,
        stalls=StallEvidence(unavailable=["compute", "memory", "dependency"]),
        bottleneck="unknown",
        meaningful_gap=MeaningfulOptimizationGap.INCONCLUSIVE,
        missing_evidence=["stall counters unavailable"],
        evidence=[_artifact("isa.txt")],
    )
    closure = close_mfma_mmq_evidence(
        "gfx1201", [path], required_shape_ids=["ffn-gate-up", "attention-output"]
    )
    assert not closure.evidence_closed
    assert closure.missing_shape_ids == ["attention-output"]


def test_final_report_requires_all_capabilities_and_renders_links() -> None:
    def entry(identifier: str, section: str) -> CapabilityReportEntry:
        return CapabilityReportEntry(
            id=identifier,
            title=identifier,
            section=section,
            status=CapabilityReportStatus.COMPLETE,
            summary="done",
            experiment_ids=[f"exp-{identifier}"],
            evidence=[_artifact(f"{identifier}-evidence.json")],
            benchmarks=[_artifact(f"{identifier}-bench.json")],
        )

    report = Gfx1201FinalReport(
        task_id="task",
        campaign_id="campaign",
        model_name="Qwen3-8B",
        model_sha256=_hash("1"),
        quantization="mixed",
        runtime_commit="abc",
        gpu="RX 9070 XT",
        model=[entry("mixed_precision", "model")],
        kernel=[entry("mfma_mmq_evidence", "kernel")],
        runtime=[
            entry("hip_graph_ab", "runtime"),
            entry("memory_reuse_audit", "runtime"),
        ],
    )
    markdown = render_gfx1201_markdown(report)
    assert "gfx1201 Final Capability Report" in markdown
    assert "mixed_precision-evidence.json" in markdown


def test_final_report_persistence_rechecks_every_linked_artifact(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "store")
    store.create_task(_task())

    def entry(identifier: str, section: str) -> CapabilityReportEntry:
        evidence = store.save_evidence_json(
            "gfx1201-test",
            f"report/{identifier}/evidence",
            {"id": identifier},
            producer="test",
        )
        benchmark = store.save_evidence_json(
            "gfx1201-test",
            f"report/{identifier}/benchmark",
            {"tg128": 100},
            producer="test",
        )
        return CapabilityReportEntry(
            id=identifier,
            title=identifier,
            section=section,
            status=CapabilityReportStatus.COMPLETE,
            summary="done",
            experiment_ids=[f"exp-{identifier}"],
            evidence=[evidence],
            benchmarks=[benchmark],
        )

    report = Gfx1201FinalReport(
        task_id="gfx1201-test",
        campaign_id="campaign",
        model_name="Qwen3-8B",
        model_sha256=_hash("1"),
        quantization="mixed",
        runtime_commit="abc",
        gpu="RX 9070 XT",
        model=[entry("mixed_precision", "model")],
        kernel=[entry("mfma_mmq_evidence", "kernel")],
        runtime=[
            entry("hip_graph_ab", "runtime"),
            entry("memory_reuse_audit", "runtime"),
        ],
    )
    json_ref, markdown_ref = persist_gfx1201_final_report(report, store)
    assert store.verify_artifact("gfx1201-test", json_ref)
    assert store.verify_artifact("gfx1201-test", markdown_ref)
