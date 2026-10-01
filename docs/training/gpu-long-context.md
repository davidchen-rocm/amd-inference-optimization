# Follow-up: 8,192-token inference capacity

The existing benchmark CLI supports this experiment without code changes or an
expanded website training registry. Use the same pinned checkpoints, ROCm Python
environment, BF16 SDPA baseline, cache and shared GPU lease as the shorter-context
campaign. Run only after its current lease holder finishes.

## Bounded protocol

Run two separate campaigns so the decode length remains a controlled coordinate.
Each measures 14B and 32B at batch sizes 1 and 4, one warmup and three scored
repetitions. Each model gets at most 20 minutes, including lock waiting and model
verification; the complete campaign gets at most 40 minutes. All work stops by
23:30 Detroit time in this example, leaving time before the GPU's 23:45 cutoff.

```sh
MACFIT_GPU_LOCK=/srv/macfit-training/data/gpu.lock \
MACFIT_MODEL_CACHE=/srv/macfit-training/cache \
HF_HOME=/srv/macfit-training/cache/huggingface \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/srv/macfit-training/venv/bin/python -m macfit_training.benchmark \
  --models qwen3-14b,qwen3-32b --batch-sizes 1,4 --prompt-tokens 8192 \
  --decode-steps 32 --warmup-repetitions 1 --repetitions 3 \
  --walltime-seconds 2400 --model-timeout-seconds 1200 \
  --stop-at 2026-09-30T23:30:00-04:00 \
  --output /srv/macfit-training/data/evidence/gpu-capability-long-d32.json

MACFIT_GPU_LOCK=/srv/macfit-training/data/gpu.lock \
MACFIT_MODEL_CACHE=/srv/macfit-training/cache \
HF_HOME=/srv/macfit-training/cache/huggingface \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/srv/macfit-training/venv/bin/python -m macfit_training.benchmark \
  --models qwen3-14b,qwen3-32b --batch-sizes 1,4 --prompt-tokens 8192 \
  --decode-steps 64 --warmup-repetitions 1 --repetitions 3 \
  --walltime-seconds 2400 --model-timeout-seconds 1200 \
  --stop-at 2026-09-30T23:30:00-04:00 \
  --output /srv/macfit-training/data/evidence/gpu-capability-long-d64.json
```

Set the cache variables to the actual cache used when prefetching these pinned
checkpoints; the paths above are deployment examples. Offline mode deliberately
fails if the complete model or tokenizer snapshot is missing. Run as the training
account with its normal access to that directory. Do not invoke the unsupervised
internal child entrypoint.

For a fresh date, change the absolute cutoff and choose new output filenames.
Each campaign creates one main JSON and two per-model JSON files: six additional
backup evidence files for both decode lengths. Count all selected evidence before
running. The current backup exporter accepts at most 64 files, including explicitly
supplied evidence, and bounds each JSON to 1 MiB. Do not let redundant snapshots
exhaust that inventory. The default long-context protocol's results are much
smaller than 1 MiB, but check actual file sizes before claiming backup success.

## Capacity estimate and batch 16 admission

The pinned [14B configuration](https://huggingface.co/Qwen/Qwen3-14B/blob/40c069824f4251a91eefaf281ebe4c544efd3e18/config.json)
has 40 layers and 40 query heads; the pinned
[32B configuration](https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json)
has 64 layers and 64 query heads. Both use 8 KV heads, head dimension 128, BF16,
untied embeddings and a declared 40,960-token context. Their different MLP sizes
are included in the architectural weight estimates below. These are calculations
from the configs, not measurements of runtime memory.

For prompt length 8,192 and 64 decode steps, BF16 KV bytes are
`2 * layers * KV_heads * head_dim * 8256 * batch * 2`. The final factor is bytes
per BF16 element; the leading factor accounts for keys and values.

| Model | BF16 weight estimate | KV, batch 1 | KV, batch 4 | KV, batch 16 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 14B | 27.51 GiB | 1.26 GiB | 5.04 GiB | 20.16 GiB |
| Qwen3 32B | 61.02 GiB | 2.02 GiB | 8.06 GiB | 32.25 GiB |

Weights plus KV are lower bounds. Activations, SDPA backend choice, temporary
buffers and allocator reservation still matter. In particular, one dense FP32
attention-score tensor for 32B prefill at this context is about 16 GiB per batch
sequence; at batch 16 that tensor alone is 256 GiB. PyTorch documents that its
math SDPA backend keeps BF16-input intermediates in FP32 and that fused backends
have input restrictions and may not be selected. See the official
[SDPA documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html).

Therefore do not assume batch 16 fits simply because weights plus KV total about
93 GiB. First obtain successful batch 4 measurements and establish that a fused
attention backend actually runs. As a conservative planning check, take measured
32B weight allocation `W`, batch-4 peak allocation `P4`, and model-runtime total
VRAM `T`. Estimate `P16 = W + 4 * (P4 - W)` and require
`1.25 * P16 + 16 GiB < T`, with enough physically free VRAM for that margin.
This heuristic is an admission screen, not a guarantee. If it passes, batch 16
may be tested in a separate, fresh one-model child campaign with the same cutoff
and at most 20 minutes. Keep OOM or timeout evidence as incomplete.

## One controlled ROCm comparison

Compare the installed PyTorch ROCm Flash Attention preference `aotriton` against
`ck`, holding the model revision, prompt, batch, decode length, dtype, seed and
repetition protocol fixed. A practical first shape is Qwen3 14B, batch 1 and 4,
context 8,192, decode 64. This changes one backend preference and needs no runtime
installation. PyTorch exposes `preferred_rocm_fa_library` and the CK preference
environment variable in its official
[backend documentation](https://docs.pytorch.org/docs/2.10/backends.html#torch.backends.cuda.preferred_rocm_fa_library).

The preference permits fallback, so two preference settings alone do not prove
two different kernels ran. Before labelling results as a kernel comparison, record
the requested preference and verify actual attention dispatch with a bounded
profiler trace or a forced Flash Attention context that fails instead of silently
using math SDPA. Keep profiling outside timed samples. Reject unsupported arms
without installing packages. Compare raw samples and output hashes; inspect any
numerical/token differences rather than assuming equivalent answers. The existing
CLI's unforced SDPA baseline remains useful, but does not itself establish which
Flash Attention implementation was chosen.
