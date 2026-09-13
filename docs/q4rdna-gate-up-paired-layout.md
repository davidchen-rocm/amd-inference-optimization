# Q4_RDNA gate/up paired-layout experiment

This experiment follows the framework's evidence-first loop and changes one
variable at a time. It targets the accepted Q4_RDNA split-K runtime on the
RX 9070 XT (`gfx1201`), not stock Q4_K_M.

## 1. Find the hotspot

The accepted split-K trace identified fused FFN gate/up at shape
`12288 x 4096` as the largest Q4_RDNA kernel family. It used split8, a
`98304 x 1 x 1` grid, `256 x 1 x 1` workgroups, and about 40.6% of GPU kernel
time in the original long trace.

## 2. Build the performance model

Two Q4_RDNA matrices require 53,477,376 packed weight bytes per fused call.
Including activation and output gives a 53,542,912-byte lower bound. At
640 GB/s the theoretical duration is 83.661 us; the accepted measured duration
was 89.841 us, or about 93.1% of theoretical bandwidth. The maximum modeled E2E
opportunity in this kernel was therefore only about 2.87%.

## 3. Test mapping before changing representation

The first single-variable screen changed only `LLAMA_Q4_RDNA_COOP`:

- split8: retained as baseline
- split4: -0.55% tg128, REJECT
- split2: -1.30% tg128, REJECT

This closed the "too many split waves" hypothesis. Mapping was no longer the
best layer to optimize.

## 4. Minimal paired-load experiment

The next hypothesis was that gate and up quant bytes could be read as one
16-bit pair instead of two byte loads while keeping quantization, split-K,
arithmetic order, grid, and workgroup unchanged.

The exact-shape HIP microbenchmark produced:

- max absolute error: 0
- kernel improvement: +2.60%
- VGPR: 43 -> 41
- LDS: unchanged at 2,048 bytes
- scratch: 0 for both arms

This passed the microbenchmark Gate and justified source integration.

## 5. Real-model A/B

Both arms used the same binary, model, sidecar, runtime arguments, and
shape-specific split mapping. The only runtime change was
`LLAMA_Q4_RDNA_GATE_PAIR=1`.

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| tg128 | 109.784 tok/s | 110.842 tok/s | +0.964% |
| tg512 | 109.101 tok/s | 110.296 tok/s | +1.096% |

The deterministic generation canary passed with identical generation SHA256.

## 6. Kernel evidence

ROCm Issue Agent cases `CASE-20260822-001` and `CASE-20260822-002` were both
complete with no warnings or excluded rows.

| Kernel metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| dispatches | 324 | 324 | unchanged |
| grid | 98304 x 1 x 1 | 98304 x 1 x 1 | unchanged |
| workgroup | 256 x 1 x 1 | 256 x 1 x 1 | unchanged |
| average | 102.655 us | 101.347 us | +1.274% |
| p50 | 94.721 us | 95.801 us | -1.140% |
| p95 | 147.561 us | 128.961 us | +12.605% |

The hypothesis was only partly correct: the paired path reduced long-tail
latency and slightly improved the mean, but did not improve the typical p50
dispatch.

## 7. Final Gate

The candidate duplicates 1,925,185,536 bytes (1.79 GiB), or 11.26% of the
RX 9070 XT's 16 GiB VRAM. This exceeds the experiment's explicit 5% additional
VRAM budget. Final outcome for the default consumer-GPU configuration:

**REJECT**

The optional patch and all evidence are retained. The useful next hypothesis is
to create the paired layout directly in the packed representation so it does not
duplicate weights, or to reduce bytes/token through a mixed-bit representation.

## Framework lessons captured

- Drop explicit same-coordinate warmup samples; do not trust the first sample.
- Bracket candidates with baseline-before and baseline-after and gate drift.
- Never apply a global split override during a full shape-specific run.
- Bind CLI context size and single-turn behavior in correctness protocols.
- Compare generated content separately from volatile throughput telemetry.
- Keep profiler approvals one-use and bind environment as well as argv.
- Make performance, correctness, and VRAM budgets explicit configuration.

The profile helper requires explicit `--mcp-command` and `--mcp-cwd` paths for
the local ROCm Issue Agent installation. `--agent-home` defaults to the current
user's `.rocm-agent` directory; deployment paths are not embedded in the tool.
