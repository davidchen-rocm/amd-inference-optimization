# MI300X research window: 30 September 2026

These measurements come from one temporary MI300X VF with about 191.7 GiB of
driver-reported memory. The installed runtime was Python 3.12.3, PyTorch
2.10.0+rocm7.1, Transformers 4.57.6, PEFT 0.18.1 and Accelerate 1.12.0.
Model revisions are fixed in the research catalog. The independent, hash-verified
raw evidence and completed training artifacts are retained in the private
OpenShift archive; this page is a compact account of the measured results.

## Inference

The BF16 Transformers SDPA baseline covered five model sizes, prompt lengths
128/512/2,048 and batch sizes 1/4/16: 45 coordinates. Each coordinate used two
excluded warmups, five scored repetitions and 64 greedy decode steps. The
following medians are for **2,048 input tokens and batch 16**. Decode throughput
is the total across the batch, and memory is PyTorch peak allocated memory.

| Qwen3 model | Aggregate decode tokens/s | Peak allocated GiB |
| --- | ---: | ---: |
| 0.6B | 1,052.29 | 5.50 |
| 4B | 793.96 | 14.53 |
| 8B | 733.05 | 23.08 |
| 14B | 515.02 | 37.02 |
| 32B | 275.35 | 75.04 |

A separate 8,192-token campaign used one excluded warmup and three scored
repetitions with 64 decode steps:

| Model | Batch | Aggregate decode tokens/s | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| 14B | 1 | 31.20 | 29.95 |
| 14B | 4 | 110.75 | 37.02 |
| 32B | 1 | 18.05 | 64.59 |
| 32B | 4 | 63.02 | 75.04 |

At 16,384 input tokens, further batch-one, batch-four and 32B batch-eight
campaigns used the same
one-warmup/three-scored/64-decode protocol:

| Model | Batch | Aggregate prefill tokens/s | Aggregate decode tokens/s | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| 14B | 1 | 12,578.03 | 20.87 | 32.31 |
| 14B | 4 | 12,613.45 | 69.25 | 46.47 |
| 32B | 1 | 5,241.71 | 12.38 | 68.08 |
| 32B | 4 | 5,301.84 | 39.91 | 88.98 |
| 32B | 8 | 5,477.21 | 56.23 | 116.86 |

The batch-eight admission used observed batch-one/four allocated and reserved
memory with additional headroom. Its actual peak reserved memory was 125.39 GiB.
For 32B, batch eight raised aggregate decode throughput relative to batch four,
but did not raise per-sequence throughput; this is not an eight-user serving
latency result. All three scored samples had finite checked logits and identical
output token hashes. No batch-sixteen long-context run was attempted.

These are compute measurements with synthetic repeated prompts, not a serving
latency or answer-quality guarantee. The baseline does not identify the actual
native SDPA implementation. The separate forced Flash Attention preference
probe produced timings for its AOTriton arm, but profiling failed. The CK arm
failed because the installed PyTorch wheel was built without CK SDPA support.
There is therefore no completed paired backend speedup comparison or profiler
proof of kernel identity.

## Task-specific LoRA

The baseline campaign trained six rank-16 adapters on 100 synthetic, balanced
English/Chinese policy examples. The 20 evaluation questions were excluded
from training, but shared the same fictional shops and policy facts. Strict
scoring required the exact six-key JSON schema and correct values.

| Model | Base /20 | Quick: 25 updates /20 | Standard: 75 updates /20 |
| --- | ---: | ---: | ---: |
| 0.6B | 0 | 13 | 15 |
| 4B | 14 | 12 | 15 |
| 8B | 12 | 11 | 14 |

The adaptive diagnostic was designed after observing baseline failures. It used
360 new training rows, more explicit system instructions and **the same 20 known
evaluation questions**. Its three standard adapters each received 270 updates.
Its base and adapter were evaluated under that campaign's own identical prompt:

| Model | Base /20 | Adapter /20 |
| --- | ---: | ---: |
| 0.6B | 0 | 16 |
| 4B | 8 | 18 |
| 8B | 12 | 16 |

All three adaptive adapters returned valid JSON on all 20 questions. This is an
adaptive diagnostic, not new independent accuracy. Changed prompts and training
data prevent attributing cross-campaign score differences solely to LoRA. The
fixture is synthetic, uses one seed and has no independent human-review claim.
Lower training loss and larger models did not consistently improve exact answers.

The ten-question English/Chinese regression uses unrelated arithmetic, string and
JSON instructions with strict requested-output scoring:

| Adapter | Base /10 | Adapter /10 |
| --- | ---: | ---: |
| Baseline 4B standard | 7 | 8 |
| Baseline 8B standard | 9 | 9 |
| Adaptive 0.6B standard | 1 | 1 |
| Adaptive 4B standard | 7 | 6 |
| Adaptive 8B standard | 9 | 9 |

Some 0.6B failures contain correct content inside unwanted JSON wrappers, which
still violates the task. Actual factual errors also occur: one output marks the
odd total 7 as even, and adaptive 4B answers `100 - 37` with `73`. These ten
questions are a small strict regression diagnostic, not broad arithmetic or
general-knowledge accuracy. Task-specific gains can accompany other regressions.

A later prospective transfer diagnostic fixed 30 new English/Chinese questions
about three new fictional shops before generating either branch's outputs. The
facts and questions were disjoint from the earlier training and evaluation
fixtures. Both branches used the same new reference and system prompt, with
greedy generation and strict typed scoring. The pinned 4B base matched **25/30**
labels; the adaptive 4B adapter matched **24/30**, with four regressed and three
improved answers.

The adapter returned comma-separated strings for all three requested weekday
arrays; the base passed two of those three array questions. Exact closing-time
boundaries also remained problematic. This fixture was designed after observing
earlier baseline errors, so it is a prospective transfer diagnostic, not an
independent estimate of general accuracy. No further training or fixture tuning
followed these outputs. Its pre-output declaration, complete labels, original
answers and separately verified archive hashes are retained. The known-question
adaptive gain did not establish a stable gain on these new scenarios.

## Larger-model update capacity

Both pinned 14B and 32B checkpoints completed five real rank-8 LoRA/AdamW updates
with a frozen BF16 base, FP32 adapters, batch one, sequence length 128 and exactly
32 supervised continuation tokens per update. Two warmup updates were excluded
from the median of three timed updates. All checked gradients/losses were finite
and a selected LoRA B matrix actually changed.

| Model | Median optimizer update seconds | Peak allocated GiB |
| --- | ---: | ---: |
| 14B | 0.269 | 28.28 |
| 32B | 0.426 | 62.43 |

This short synthetic capacity check does not establish task quality, long-sequence
SFT performance or that the website's rank-16 recipes fit. No adapter was published
from it. Two earlier attempts failed during tokenizer/config validation and are
preserved separately; only the corrected third run supplies these measurements.

Repeating the bounded five-update probe at sequence length 256 measured 0.265
seconds / 28.56 GiB for 14B and 0.426 seconds / 62.52 GiB for 32B. Three timed
samples per model are too few to interpret the small timing differences as an
improvement. The number of supervised continuation tokens remained 32.

## CPU portability

The completed adaptive 0.6B job was restored from a resolved OpenShift archive
version. All seven necessary job files were verified against that archive's
manifest before and after transfer. Its pinned base and adapter were merged on
CPU and the resulting Transformers safetensors export was reloaded on CPU.

Across six original evaluation questions, the PEFT CPU and merged CPU branches
produced identical token IDs and exact text on **6/6** questions. Both branches
matched the strict original labels on **5/6**. A separate child independently
verified the saved export inventory and source artifacts after the comparison
child exited. The entire check took 92.76 seconds with four CPU threads.

This establishes the tested adapter's merge/reload portability for those outputs;
it does not establish that every answer is correct or certify GGUF, MLX, macOS,
quantization or a website inference runtime. The private merged weights are
excluded from the restricted job backup and can be regenerated from the retained
adapter plus the exact, re-downloadable base revision.

See [research protocols](research.md) for the bounded inference, training,
regression and CPU export commands. All accelerator work shares the service's
GPU lock, uses isolated children and has an absolute cutoff. Reports preserve
failed and unsupported attempts separately from successful measurements.
