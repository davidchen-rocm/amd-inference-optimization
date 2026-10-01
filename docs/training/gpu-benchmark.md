# Reproducible ROCm capacity baseline

This operator CLI measures the existing Transformers runtime without opening a
network service or changing the website's model whitelist. It uses pinned Qwen3
0.6B, 4B, 8B, 14B and 32B checkpoints. The two larger models are available only to
this local benchmark; website training retains its reviewed model registry.

Validate the full protocol on any machine, without importing Torch or using a GPU:

```sh
python -m macfit_training.benchmark --validate-only
```

Use the existing ROCm Python environment and training account. All workloads on
the host must use the same advisory GPU lock. Supply the real lock and cache paths
from your service configuration, a fresh output filename, and an explicit cutoff:

```sh
MACFIT_GPU_LOCK=/absolute/path/shared-gpu.lock \
MACFIT_MODEL_CACHE=/absolute/path/model-cache \
python -m macfit_training.benchmark \
  --models qwen3-0-6b,qwen3-4b,qwen3-8b,qwen3-14b,qwen3-32b \
  --batch-sizes 1,4,16 --prompt-tokens 128,512,2048 \
  --decode-steps 64 --warmup-repetitions 1 --repetitions 3 \
  --walltime-seconds 3600 --model-timeout-seconds 1200 \
  --stop-at 2026-09-30T23:40:00-04:00 \
  --output /absolute/path/data/evidence/gpu-capability-transformers.json
```

The first load may download large public checkpoints. Downloading and hashing,
waiting for the GPU lock, warmup, and native GPU operations all count toward the
supervised process deadline. Every model runs in a separate process group. An
out-of-memory failure or timeout records incomplete evidence, terminates that
model's process group, and allows another model to run within the remaining
campaign budget. The supervisor does not silently lower batch size or context
length. It never launches the next model after the absolute deadline. Leave time
after this cutoff for storage verification and a final backup.

The harness requires a ROCm GPU with BF16 support and loads all weights onto one
GPU, using PyTorch BF16 and Transformers SDPA attention. `trust_remote_code=False`
and pinned checkpoint revisions apply to model and tokenizer loading. A snapshot
digest records the actual downloaded files. It rejects CPU or split-device
offloading, making results comparable under this specific baseline.

Each shape coordinate excludes at least one complete warmup. Measured prefill
consumes exactly the requested token count for each sequence and computes one
next-token logit slice with a KV cache. Measured decode performs exactly the
requested number of subsequent one-token cached forwards. Timings synchronize
the GPU at each phase boundary and include Python dispatch. Decode also includes
mask updates and greedy argmax; prefill ends after the forward and synchronization.
Prefill throughput counts batch × prompt tokens; decode throughput
counts batch × decode steps. The prefill-generated first token is excluded from
decode throughput. Reported medians, ranges and coefficient of variation retain
the individual scored samples; unusually variable measurements remain visible.

Inputs are fixed repetitions of a public fictional support reference, repeated
identically across batch sequences. Greedy decode intentionally continues beyond
EOS to make compute length identical between models. These are synthetic compute
measurements, not conversational latency or accuracy scores. Finite-logit checks
cover prefill and the final decode step. Output token hashes reveal whether
repeated greedy runs were identical; a mismatch is reported rather than hidden.
Neither check establishes semantic answer quality.

The allocator is cleared between shape coordinates, then warmed for that exact
shape. Peak allocated and reserved bytes are PyTorch statistics and include model
weights and the KV cache. They exclude other processes and non-PyTorch allocations;
they must not be presented as total physical GPU use. This baseline does not use
vLLM, quantization, continuous batching, speculative decoding or custom kernels.

The main JSON and one JSON per model are written atomically after each real event.
All JSON evidence is strictly smaller than 1 MiB. Existing output names are refused,
so a second run cannot overwrite earlier evidence. Schema version 2 uses a main
index containing every completed case's summary, a compact tail of 256 events,
and each per-model filename plus its SHA-256 of the saved file bytes. The per-model
JSON retains every raw scored sample; the main index omits those duplicate samples.
A flushed JSONL journal streams complete progress. Model stderr stays in a
separate private diagnostic log. Only JSON evidence
under `data/evidence` enters the existing verified OpenShift backup; no credentials
or environment-variable values are emitted. The CLI itself does not trigger a
backup. Check the backup manifest and receipt after running a campaign.

For a bounded 14B/32B follow-up at 8,192 tokens, capacity reasoning and a controlled
ROCm attention comparison, see [the long-context protocol](gpu-long-context.md).
