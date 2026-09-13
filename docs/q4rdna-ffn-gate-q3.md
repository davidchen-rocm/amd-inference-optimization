# Q4_RDNA FFN gate Q3 experiment

This experiment tested the representation change that the paired-layout experiment did not:
only the 36 Qwen3-8B `ffn_gate.weight` tensors changed from Q4_RDNA to Q3_RDNA. FFN up,
attention, output, the GGUF model, binary, split-K width, workload, and benchmark protocol were
held fixed.

## Result

**REJECT.** The candidate saved 226,492,416 bytes (216 MiB), reduced routed-sidecar effective
precision from 4.25 to 3.989 bpw, and improved tg128 by 2.76%. The stable tg512 rerun improved by
1.77%, below the pre-registered 2% gate. Profiling and the provisional 100-question quality test
were therefore skipped.

| Stage | Baseline | Candidate | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| Exact 12288x4096 fused microbenchmark | 95.895 us | 83.881 us | +14.32% | PROMOTE |
| Sidecar data | 3,689,938,944 B | 3,463,446,528 B | -6.14% | PASS |
| tg128 | 109.970 tok/s | 113.002 tok/s | +2.76% | PROMOTE |
| tg512 | 110.271 tok/s | 112.226 tok/s | +1.77% | REJECT |

The first tg512 bracket was preserved as `INCONCLUSIVE` because baseline drift was 1.09%. A new
attempt reduced drift to 0.043% and produced the final performance result.

## What changed

- The sidecar v2 index records Q3 or Q4 per tensor.
- Q3 packs 64 values plus FP16 scales into 26 bytes per group (3.25 bpw); Q4 uses 34 bytes
  (4.25 bpw).
- A fused wave32 split-K kernel reads Q4 up weights and Q3 gate weights and performs SwiGLU.
- Standalone matrix paths reject Q3 entries, so Q3 gate tensors cannot silently run through a Q4
  decoder.
- Runtime activation evidence confirmed 0 Q3 tensors for baseline and exactly 36 for candidate.

## Interpretation

The performance model was directionally correct: fewer gate bytes made both the controlled kernel
and end-to-end decode faster. The model-level gain was smaller than the microbenchmark gain because
only one of the two fused matrices changed and the rest of token generation was unchanged. For the
longer coordinate, the 3-bit unpack work consumed enough of the traffic reduction that it missed the
materiality gate.

The next bounded experiments are either sensitivity-selected Q3 gate layers (quality-first memory
savings) or a Q3 vector-load/unpack improvement. This all-36-layer candidate is not accepted as the
new default.

## Evidence

- `patches/q4rdna-ffn-gate-q3.patch` (applies after the frozen Q4_RDNA base patch)
- `.gpuopt-q4rdna-q3/final-decision.json`
- `.gpuopt-q4rdna-q3/gate-q3-microbench-attempt-0001/summary.json`
- `.gpuopt-q4rdna-q3/e2e/tg128/summary.json`
- `.gpuopt-q4rdna-q3/e2e/tg512/summary.json` (preserved inconclusive attempt)
- `.gpuopt-q4rdna-q3/e2e/tg512-attempt-0002/summary.json`
