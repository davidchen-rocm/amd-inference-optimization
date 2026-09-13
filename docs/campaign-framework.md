# Multi-strategy consumer AMD campaign

The campaign coordinator sits above the existing single-experiment workflow. It
does not replace the experiment runner, quality worker, ROCm MCP adapter, or
deterministic gates. It keeps independent failures from ending exploration and
only combines candidates that passed their own evidence gates.

## Candidate families

- Mixed-bit emits five fixed plans: stock Q5_K_M, stock Q4_K_M, and three Q6-base
  FFN policies. Every arm is independently quantized from the frozen BF16 input.
- Shape-kernel emits four geometry-bound plans for gate/up, attention query/output,
  key/value, and the vocabulary head. The default hypothesis is the existing
  small-k path with eight rows per block; structured specs can instead bind
  split-K, waves, rows, vector loads, VGPR limits, and gate/up fusion. The Agent
  still authors one bounded patch at a time.
- KV-cache emits matching f16, q8_0, and q4_0 K/V arms with Flash Attention at
  depths 4096, 16384, and 28672, plus a minimal canary before each scored arm.

## Temporary quality boundary

`provisional-math-100.v1` is deliberately evidence-only. It selects a frozen,
stratified 100-question set with seed 20260815 and quotas 12/12/32/44. A candidate
passes when it loses at most two answers, PPL regresses by at most 0.5%, and greedy
canary accuracy does not decline. A passing candidate is
`EXPERIMENTAL_ACCEPTED`; the campaign report always records
`quality_protocol_pending_replacement=true` and `production_ready=false`.

Replacing this evaluation requires a new `QualityPolicy` implementation and a
new protocol hash. It does not require changes to Campaign, Runner, or Store.

## Control-plane CLI

```bash
gpuopt campaign create --config examples/consumer-amd-campaign.yaml --store .gpuopt
gpuopt campaign run qwen3-8b-consumer-amd --store .gpuopt
gpuopt campaign status qwen3-8b-consumer-amd --store .gpuopt
```

`campaign run` advances one persisted revision. At evidence boundaries it returns
the exact required input instead of guessing. Stage input is JSON or YAML passed
with `--input`; every `evidence_paths` entry must already be registered in the
owning task's artifact manifest. At `PLAN_CANDIDATES`, run:

```bash
gpuopt campaign plan qwen3-8b-consumer-amd --store .gpuopt
```

The planner writes each candidate spec and the aggregate plan as immutable,
versioned evidence. Candidate results are supplied to `campaign run` with the
running candidate ID and a registered `performance_gate_path`. A performance
pass additionally requires `provisional_quality_path` containing hash-bound
baseline and candidate measurements. The framework derives
`EXPERIMENTAL_ACCEPTED`, `REJECTED`, or `INCONCLUSIVE`; callers cannot declare an
acceptance. The coordinator then activates the next candidate, selects
experimental winners, checks compatibility, and requires a fresh final-combination
validation.
