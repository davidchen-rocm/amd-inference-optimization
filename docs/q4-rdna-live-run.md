# Q4_RDNA live vertical-slice report

Date: 2026-08-16/17  
Task: `q4-rdna-live`  
GPU: Radeon RX 9070 XT (`gfx1201`)  
Runtime: llama.cpp at `a7a6d0d269c896218b6c78e0933bd6a17519d3f6`

## Outcome

The framework completed the intended live vertical slice and issued `ACCEPT` for the
split-K runtime configuration. The candidate reused the same prepared binary, model,
sidecar, ROCm runtime, GPU device, and benchmark protocol as the production Q4_K_M
baseline.

| Coordinate | tg128 tokens/s | tg512 tokens/s | Gate |
|---|---:|---:|---|
| Q4_K_M baseline | 96.8011 | 96.3548 | baseline |
| Q4_RDNA old mapping | 61.0908 | 61.2526 | `REJECT` |
| Q4_RDNA split-K | 111.5753 | 111.0013 | `ACCEPT` |

Split-K improved throughput by 15.2624% at tg128 and 15.2006% at tg512. Both
requirements were at least 10%.

Fresh quality evaluation used 16,384 deterministic perplexity tokens and all 848
deterministic zero-shot MMLU math questions:

| Representation | Perplexity | Math correct | Math accuracy |
|---|---:|---:|---:|
| BF16 | 3.369526 | 521/848 | 61.439% |
| Q4_K_M | 3.434980 | 512/848 | 60.377% |
| Q4_RDNA | 3.484499 | 499/848 | 58.844% |

Relative to Q4_K_M, Q4_RDNA perplexity regressed 1.4416% (budget: at most 1.5%)
and math accuracy dropped 1.5330 percentage points (budget: at most 2.0). Correctness,
performance, stability, identity, perplexity, and accuracy checks all passed.

## Kernel evidence

The live short-profile protocol used one warmed-up tg128 repetition. It is independent
from the authoritative unprofiled tg128/tg512, three-repetition E2E protocol.

The old mapping's three Q4_RDNA GEMV families consumed 1,814.20 ms, or 90.34% of
parsed GPU kernel time. The split-K Q4_RDNA GEMV families consumed 832.81 ms, or
81.04% of parsed GPU kernel time. Within these comparable short traces, split-K cut
Q4_RDNA GEMV aggregate duration by 54.09%.

Observed split-K average durations included 89.84 us for the fused 12288-row path,
31.39 us and 16.62 us for split-8 paths, and 5.25 us for the small split-32 path.
Hardware PMC, whole-GPU occupancy, workload telemetry, and integrated kernel metadata
were unavailable and were not treated as zero.

## Accepted configuration and provenance

The accepted delta is a runtime configuration, not a historical old-to-split source
patch:

- set `LLAMA_Q4_RDNA_SIDECAR` to the frozen sidecar;
- unset `LLAMA_Q4_RDNA_MAPPING` to select the prepared binary's default split-K path;
- unset other Q4_RDNA/Smithy experimental controls;
- reuse binary SHA-256
  `b3fbcd9151ea98b1420aaaae5d3e77a413f80acf449257d7d4fb2f5c9d2f35c2`.

The prepared-runtime prerequisite patch is separately preserved as
`reports/final-source.patch`, SHA-256
`02ac94bd3a00661a2f4ddf6c7c71c96573196942fdc832c4c75d087eba781457`.
It contains the complete Q4_RDNA integration on the clean base commit. It must not be
misrepresented as the lost historical old-to-split patch.

## Framework findings from the live run

Changes completed during the run:

- split the short Level-2 profile protocol from the full E2E protocol after a 205 MB
  trace exceeded the ROCm Issue Agent's 200 MB evidence budget;
- preserved consumed approvals and generated a new exact attempt instead of replaying
  a failed approval;
- accepted percentile-sample-limit `partial` evidence only when aggregate timing,
  workload status, profiler status, and hotspot ranking fields are complete;
- bound Agent evidence IDs to `path@sha256` and excluded mutable workflow/control
  state from the evidence catalog;
- persisted old mapping as a real negative experiment and skipped its expensive
  quality evaluation after the performance pre-gate failed.

P0 hardening implemented after the quality evaluation exposed a duplicate-launch risk:

- added a stable-inode per-task lock and persistent quality execution attempts;
- `run-live` now resumes/monitors the existing exact request instead of launching a
  second evaluator;
- each attempt binds command/spec hashes, Linux boot ID and process start ticks,
  heartbeat, timeout, stdout, stderr, output hashes, and terminal result;
- timeout cleanup targets the validated evaluator process group, a vanished worker is
  marked `ORPHANED` without automatic retry, and terminal finalization is idempotent.

The remaining P1 improvement is evaluator-level atomic per-variant checkpoints. The
framework now prevents duplicate execution, but checkpoints would make partial progress
portable and reduce the cost of an explicitly approved retry after a host failure.

## ROCm Issue Agent follow-up

Recommended contract improvements:

- replace warning-string interpretation with stable warning codes and structured
  completeness fields (`aggregate_timing_complete`, `percentiles_complete`,
  `hotspot_ranking_reliable`, and `trace_complete`);
- expose trace/event/byte budgets, observed usage, parsed coverage, and the GPU-share
  denominator;
- use bounded streaming quantiles or explicitly typed approximate percentiles;
- fix privacy-redaction false positives that changed C++ anonymous-namespace kernel
  names into `[IP_ADDRESS]` strings, and provide a stable hashed kernel ID;
- return structured unavailable evidence consistently instead of mixing evidence
  status with MCP `isError` for trace preparation failures;
- add Level-0 clock, temperature, power, and throttle telemetry;
- include artifact hashes, ownership, and retention in case manifests.

ROCm Issue Agent should remain the evidence provider. Optimization hypotheses, source
changes, experiment management, and final performance/quality verdicts remain outside
its boundary.
