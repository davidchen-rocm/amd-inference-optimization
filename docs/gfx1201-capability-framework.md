# gfx1201 capability-closure framework

This phase is one strictly ordered campaign. A later capability cannot start until the
previous capability has a persisted, SHA-256-bound result:

```text
inspect target
→ shared baselines
→ mixed precision
→ HIP Graph A/B
→ memory reuse audit
→ MFMA/MMQ evidence
→ gfx1201 final report
```

The top-level state is `state/gfx1201-campaign.json` (schema version 2). Create and
inspect it with:

```bash
gpuopt gfx1201 create --config examples/gfx1201-capability-campaign.yaml --store .gpuopt
gpuopt gfx1201 status TASK_ID --store .gpuopt
```

## 1. Mixed precision

The formal sub-workflow is:

```text
INSPECT_MODEL → CAPTURE_SENSITIVITY → PLAN_ASSIGNMENTS → PACK → BENCHMARK
→ QUALITY → ACCOUNT_BYTES → PARETO_RANK → DECIDE → COMPLETE
```

`TensorNamingConfig` classifies a model inventory by configured, anchored regular
expressions. `SensitivityEvidence` supplies per-tensor scores. The bounded planner emits
at most 16 exact `tensor → precision` assignments across Q8/Q6/Q5/Q4, configured
sensitivity search points, and optional external providers such as Q4_RDNA hybrid.

```bash
gpuopt gfx1201 mixed-plan TASK_ID --input mixed-plan.yaml --store .gpuopt
gpuopt gfx1201 mixed-rank TASK_ID --input mixed-results.yaml --store .gpuopt
```

Every packed candidate records the source model, imatrix, quantizer, exact argv,
assignment hash, output inventory, packed bytes, effective bpw, tg128, tg512, PPL,
provisional 100-question accuracy, and evidence references. The Pareto rank excludes
unstable or quality-failing candidates. The current quality result is explicitly
provisional; it cannot be presented as production acceptance.

The planner is model-config driven. A compatible dense model needs model geometry,
tensor naming rules, benchmark coordinates, and a precision search policy. It does not
require Qwen3-8B names in the workflow code.

## 2. HIP Graph A/B

The A/B runner uses one binary compiled with `GGML_HIP_GRAPHS=ON`. The OFF arm differs
only by `GGML_CUDA_DISABLE_GRAPHS=1`; the ON arm explicitly unsets that variable. It
alternates ON/OFF order across 7 paired samples, validates the exact binary, model, and
protocol hashes, and saves every command envelope before evaluating:

- pp512 and derived latency;
- tg128 and derived latency;
- tg512 and derived latency;
- sample CV and paired 95% confidence interval;
- optional kernel, launch, CPU launch-gap, and GPU idle-gap trace evidence.

```bash
gpuopt gfx1201 hip-graph-ab TASK_ID --input hip-graph.yaml --store .gpuopt
gpuopt gfx1201 hip-graph-ab TASK_ID --input hip-graph.yaml --store .gpuopt --execute
```

Valid outcomes are `MATERIAL_BENEFIT`, `NO_MATERIAL_EFFECT`, `HARMFUL`, and
`INCONCLUSIVE`. A small effect is a valid conclusion and stops further graph work.

## 3. Buffer and memory reuse audit

`MemoryTraceEvidence` consumes normalized runtime evidence. It never reparses private
ROCm profiler CSV. The audit checks steady-state allocation/workspace repetition,
copy/memset cost, buffer lifetimes, and immediate global-memory producer/consumer
traffic. It selects at most two issues and only when each has at least a configured 1%
E2E upper bound.

The only outcomes are `OPPORTUNITY_FOUND`, `NO_ACTION`, and `INCONCLUSIVE`. Count-only
or incomplete traces produce `INCONCLUSIVE`; they never become a fabricated zero-cost
result. A discovered patch is still executed by the normal isolated Experiment Runner
and must pass before/after latency, tg128, tg512, memory-operation, and correctness
gates.

## 4. MFMA/MMQ evidence closure

Each configured shape records selected kernel, MMQ enablement, actual ISA instruction
family, workgroup/grid, wave mapping, VGPR/SGPR/LDS/scratch, occupancy when available,
latency, stalls, bottleneck, and `YES/NO/INCONCLUSIVE` optimization gap. Claims of MFMA
or WMMA require a matching instruction from actual disassembly; build flags or kernel
names are not enough. Missing counters stay explicitly unavailable.

The shape set is configuration data. For Qwen3-8B it includes 12288×4096, 4096×4096,
1024×4096, the vocabulary head, and any additional measured MMQ/GEMM hotspot.

## Final report

`gfx1201-final-report.json` and `.md` require entries for mixed precision, HIP Graph,
memory reuse, and MFMA/MMQ. Every capability must link experiment IDs plus registered
evidence and benchmark artifacts whose hashes still verify. Once written, the campaign
advances to `COMPLETE`, ending gfx1201 feature expansion.

Current platform limits are preserved in the report: unsupported metadata, counters,
or telemetry remain `INCONCLUSIVE` or unavailable. They are not inferred from timing.

