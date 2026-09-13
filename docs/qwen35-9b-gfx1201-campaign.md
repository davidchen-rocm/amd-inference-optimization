# Qwen3.5-9B on gfx1201 optimization campaign

## Result

This campaign used Qwen3.5-9B text-only on an AMD Radeon RX 9070 XT (gfx1201, 16 GB), llama.cpp commit `a7a6d0d269c896218b6c78e0933bd6a17519d3f6`, and ROCm 7.14.

Final status: `EXPERIMENTAL_ACCEPTED`.

Selected model: `Q7_MIX_Q6_Q8`, effective 7.1172 bpw, 7,965,696,768 bytes, SHA-256 `39e4be97738d1e943701314a30c762f4db2920be70836bf85d754bc04e3dd62e`.

The Q7 assignment uses Q6_K by default and preserves Q8_0 for token embeddings, the output head, and Q/K/V/O weights in the eight full-attention layers. It was built from the frozen BF16 source and was not produced by requantizing another GGUF.

## Precision sweep

All arms used the same binary, device, full-GPU offload, benchmark coordinates, and same-coordinate warmup policy.

| Model | Container bpw | Size | pp512 | tg128 | tg512 | Math/100 | PPL | Outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Q8_0 baseline | 8.5126 | 9.53 GB | 4123.30 | 59.48 | 59.59 | 70 | 2.0066 | reference |
| Q6_K | 6.5753 | 7.36 GB | 2309.99 | 71.56 | 71.88 | 67 | 2.0034 | rejected: quality -3 |
| Q5_K_M | 5.7790 | 6.47 GB | 3520.44 | 78.17 | 78.51 | 65 | 1.9956 | rejected: quality -5 |
| Q7 mixed | 7.1172 | 7.97 GB | 2416.30 | 69.25 | 68.99 | 68 | 2.0038 | experimental accept |

Q7 mixed improves tg128 by 16.43% and tg512 by 15.77% versus Q8. The 100-question provisional gate allows a maximum two-question drop; Q7 lands exactly on that boundary. Q5 and Q6 are faster, but the framework rejected them instead of treating speed alone as success.

The current quality protocol is `provisional-math-100.v1`, protocol hash `71e52cafc8928e2f66bc25ecc9b06cc88291973eadda98a3a68dd66be89c99bf`. This is experimental evidence, not a production quality claim.

## Kernel evidence

Q8 and Q7 decode profiles each captured 119,318 dispatches at 100% event coverage. The Q7 policy preserved dispatch structure while reducing weight traffic:

- fused FFN 12288x4096 total kernel time: -18.70%
- fused 4096-output group: -16.29%
- linear/QKV 8192x4096 group: -12.40%
- plain 4096 group: -17.16%
- Q8 vocab output head: +0.18%
- Q8 K/V 1024x4096 group: +5.30%

This supports the mixed-bit interpretation: most of the decode gain comes from lower Q6 weight traffic, while the Q8-sensitive tensors do not become faster.

The prefill MMQ profile captured 2,397 dispatches at 100% coverage. The three largest Q6 `mul_mat_q` launch groups account for 76.98% of profiled GPU time. They use 248 VGPRs per thread and a 32x8 workgroup.

gfx1201 code objects were extracted from a clean, same-commit rebuild of `libggml-hip.so`. The Q6 prefill kernel `mul_mat_q<(ggml_type)14, 128, false>` contains 16 `v_wmma_i32_16x16x16_iu8` instructions and no `v_mfma`. Decode MMVQ uses `v_dot4_i32_iu8` and no WMMA/MFMA. Therefore the same-source RDNA4 path is:

```text
prefill: mul_mat_q / MMQ -> RDNA4 WMMA
decode:  mul_mat_vec_q / MMVQ -> dot4
```

The `GGML_HIP_MMQ_MFMA=ON` build option does not mean that gfx1201 executes MFMA; MFMA is the CDNA path.

The exact dynamic library loaded during profiling was overwritten six minutes later by a same-source rebuild. Quality evidence had recorded the earlier library SHA-256 `40082b...ab317`; the extracted library is `ab0238...b21`. Consequently the instruction-family conclusion is strong same-source evidence, but exact profile-to-code-object identity is marked `INCONCLUSIVE`. This does not affect the Q7 performance/quality result, whose compared arms used the same recorded runtime library.

## Other optimization strategies

### Shape-specific kernel

The exact Q6 12288x4096 small-K/rows-per-block mapping was tested on the Q7 model. It regressed tg128 by 2.67% and tg512 by 2.15%, so it was rejected before quality or deeper profiling. Other shape arms were skipped where prior same-GPU evidence was already negative or the Q7 precision assignment made the patch inapplicable.

### KV-cache quantization

F16, Q8_0, and Q4_0 cache arms were tested at 4K, 16K, and 28,672 token depths with Flash Attention. Q8 regressed by 1.89%, 2.05%, and 2.59%; Q4 regressed by 1.88%, 1.99%, and 2.30%. Both candidates were rejected before quality.

Qwen3.5-9B has only eight full-attention layers, so its theoretical KV footprint is smaller than a 32-layer full-attention model. At 32K, K+V storage is approximately 1.00 GiB for F16, 0.53 GiB for Q8_0, and 0.28 GiB for Q4_0. The memory saving is real, but it did not improve single-request decode throughput in this experiment.

### HIP Graph A/B

Seven paired ON/OFF samples produced:

- pp512: -0.05%, 95% CI [-0.52%, +0.42%]
- tg128: +1.89%, 95% CI [+1.86%, +1.92%]
- tg512: +1.96%, 95% CI [+1.85%, +2.07%]

Graph ON reduced explicit `hipLaunchKernel` calls and issued 127 `hipGraphLaunch` calls, but did not cross the fixed 2% materiality threshold. The strict verdict is `INCONCLUSIVE`, not a claimed win. The OFF HIP API trace hit its 250,000-event budget, so API totals and CPU-gap comparisons are lower-bound/partial evidence.

### Buffer and memory reuse audit

The audit outcome is `INCONCLUSIVE`. The installed profiler did not expose memory-copy byte counts, and the captured events did not isolate model load from steady state. No memory patch was created from incomplete evidence.

## Framework coverage added or exercised

- config-driven prepared mixed-bit candidates with tensor assignments and effective bpw
- resumable Q8/Q6/Q5 preparation and custom Q7 mixed packing
- same-coordinate benchmark warmup and stable multi-metric gates
- fixed stratified 100-question provisional quality policy
- model-generic KV-cache protocols and per-depth gates
- strict HIP Graph paired A/B evaluation
- raw rocprof fallback with geometry/resource-aware kernel grouping
- raw runtime trace normalization for HIP APIs, kernels, copies, and gaps
- evidence-bound `SKIPPED`/`NO_ACTION` campaign outcomes
- execution-map persistence by phase, operator, shape, kernel, and resource usage
- actual AMDGPU code-object ISA evidence for MMQ/WMMA and MMVQ/dot4

## Provenance

- Campaign store: `.gpuopt-qwen35-live/qwen35-9b-gfx1201-task-v2`
- Campaign state: `state/campaign.json`, revision 21
- Final validation correction: `artifacts/evidence/qwen35/final-validation/v000002.json`
- Q7 preparation: `artifacts/evidence/qwen35/q7/preparation/v000001.json`
- Performance: `artifacts/evidence/qwen35/q7/benchmark/v000001.json`
- Quality: `artifacts/evidence/qwen35/quality/100/v000001.json`
- Kernel comparison: `artifacts/evidence/qwen35/profiles/q8-vs-q7mix-shape-comparison/v000001.json`
- Execution map: `artifacts/evidence/execution-map/v000001.json`
- HIP Graph decision: `artifacts/evidence/gfx1201/hip-graph-ab/decision-with-trace/v000002.json`
- Memory audit: `artifacts/evidence/qwen35/memory-audit/result/v000001.json`
- MMQ/ISA closure: `artifacts/evidence/qwen35/mfma-mmq/closure/v000002.json`

## Current limitation

The final result is intentionally labeled `EXPERIMENTAL_ACCEPTED`. The selected model passed the user-requested 100-question temporary suite, but it sits exactly at the allowed quality boundary. A redesigned, larger quality suite should be run before calling it production-ready. Standalone runners must also freeze the full dynamic-library closure at process start; launcher-executable SHA alone is insufficient runtime identity.
