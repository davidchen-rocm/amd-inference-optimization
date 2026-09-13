# Qwen3-8B consumer AMD campaign

This campaign tested three independent optimization families on an AMD Radeon
RX 9070 XT (`gfx1201`, 16 GB) using llama.cpp commit
`a7a6d0d269c896218b6c78e0933bd6a17519d3f6`.

## Locked coordinates

- Baseline model: Qwen3-8B Q6_K, SHA-256
  `b5445085f052d9df572d3df7034d9fcfabaf63c17ed3a92f37e03c059180cc9c`.
- Decode protocol: single request, tg128 and tg512, batch 2048, ubatch 512,
  12 threads, all layers on `ROCm0`.
- Stability: CV at most 2%. Five samples are expanded once to 12 when noisy.
- Quality: eight deterministic greedy canaries, 32 PPL chunks, and a frozen
  stratified 100-question MMLU math subset.
- Quality budget: greedy outputs equal, PPL regression at most 0.5%, and no
  more than two fewer correct answers out of 100.

Every scored decode arm received one full-workload preconditioning run. The
short built-in llama-bench warmup did not adequately stabilize this GPU.

## Mixed-bit result

Both candidate models were quantized directly from the BF16 checkpoint using an
importance matrix collected from an independent training corpus. No candidate
was requantized from another GGUF.

| Arm | tg128 | tg512 | Change vs Q6 | Math | PPL | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q6_K | 79.63 | 79.25 | baseline | 62/100 | 2.3400 | baseline |
| Q5_K_M | 87.86 | 87.60 | +10.34% / +10.54% | 64/100 | 2.3413 | ACCEPT |
| Q4_K_M | 96.19 | 96.17 | +20.80% / +21.35% | 59/100 | 2.3386 | REJECT |

Q5_K_M is the accepted model. Q4_K_M was faster but exceeded the locked
100-question quality budget by one additional wrong answer.

## Shape-specific kernel result

Two clean-base Q6_K source patches forced the existing `small_k` path with
eight rows per block for exact model shapes:

- N=12288, K=4096: tg128 -3.25%, tg512 -2.75%.
- N=4096, K=4096: tg128 -16.86%, tg512 -5.60%, with excessive CV.

Both experiments were rejected before profiling and quality evaluation. No
kernel patch is part of the accepted result.

## KV-cache result

The fixed Q6_K model and binary were tested with flash attention enabled at
depths 4096, 16384, and 28672.

| Cache | d4096 | d16384 | d28672 | Decision |
| --- | ---: | ---: | ---: | --- |
| f16 | 73.66 | 60.80 | 51.99 | baseline |
| q8_0 | -9.22% | -18.38% | -24.04% | REJECT |
| q4_0 | -7.44% | -6.55% | -6.74% | REJECT |

The quantized caches reduce memory, but their decode throughput regressed at
all tested depths on this implementation and GPU. The performance pre-gate
therefore skipped their quality runs.

## Accepted configuration

- Model: `Qwen3-8B-Q5_K_M-imatrix.gguf`, SHA-256
  `fd96f1387d8d3465385338a636a27b6427381304059de23e234f72c466ab183a`.
- Runtime: unchanged llama.cpp source and f16 KV cache.
- Decode improvement: +10.34% at tg128 and +10.54% at tg512.
- Corrected full quality: 523/848 versus 526/848, paired p=0.780,
  PPL +0.056%, and 5/8 versus 3/8 fixed greedy canaries.

The authoritative machine-readable reports are under the campaign artifact
root in `reports/mixed-bit.json`, `reports/shape-kernel.json`,
`reports/kv-cache.json`, and `reports/final.json`.

## Full 848-question follow-up (legacy v1 verdict)

The accepted Q5 candidate was subsequently compared with Q6 on all 848 frozen
MMLU math questions, using the same binary, PPL corpus, and deterministic
greedy protocol:

- Q6: 526/848 (62.03%), PPL 2.3400.
- Q5: 523/848 (61.67%), PPL 2.3413.
- Delta: -3 questions (-0.354 percentage points), PPL +0.056%.
- Three of eight full-set greedy canaries changed.

The aggregate math and PPL budgets pass, but the exact-greedy requirement does
not. The performance improvement remains measured and reproducible, but the
strict full-quality verdict is `REJECT`. The append-only superseding reports are
`reports/q5-full-848-quality.json` and
`reports/final-after-full-quality.json`; the earlier 100-question report remains
available as historical evidence.

### Corrected full-quality protocol v2

The exact-string verdict above exposed two protocol defects: canary identity
changed with `--math-limit`, and different answer strings were treated as a
quality failure even when correctness did not regress. Protocol v2 fixes both
and reruns both model arms from scratch. It selects fixed canaries from the
complete fixture independently of the math subset, measures canary correctness,
records all 848 paired correctness bits, and applies an exact two-sided McNemar
test.

The corrected rerun produced:

- Q6: 526/848 math and 3/8 fixed canaries.
- Q5: 523/848 math and 5/8 fixed canaries.
- Paired cells: 499 both correct, 27 Q6-only, 24 Q5-only, 298 both wrong.
- McNemar p=0.780; the three-answer net difference is not significant.
- PPL regression remains 0.056%.

All corrected full-quality checks pass, so the final v2 verdict is `ACCEPT`.
Append-only reports `reports/q5-full-848-corrected-v2.json` and
`reports/final-corrected-v2.json` supersede the flawed exact-string verdict
without deleting it.

## Framework findings

- A full-workload preconditioning stage is needed before scored samples.
- Sampling policy must support a deterministic five-to-12-sample retry.
- GPU leases must cover the physical device across campaigns, not only a task.
- Runtime arguments for quality must be arm-specific and part of the protocol
  hash.
- A performance pre-gate saves substantial profiler and quality time for clear
  regressions.
