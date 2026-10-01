# Real ROCm teaching jobs

For editable configuration fields, installation, result interpretation, adapter
usage and offline CPU merge export, read the
[reusable training guide](../../docs/training/README.md).

These small examples use policies for a **fictional** shop. They exercise the real
worker; they are not evidence that a model learned a production task.

Install the repository's training dependencies into the ROCm PyTorch environment.
Run validation on any CPU host before copying an input to the GPU worker:

```sh
python -m macfit_training.cli validate --kind training --input examples/training/training-input.json
```

On a ROCm GPU host, prepare a fresh private job directory containing `input.json`:

```sh
mkdir -m 700 /path/to/new-job
cp examples/training/training-input.json /path/to/new-job/input.json
MACFIT_GPU_LOCK=/path/to/shared-gpu.lock MACFIT_MODEL_CACHE=/path/to/model-cache \
  python -m macfit_training.worker --kind training --job-dir /path/to/new-job
```

Use `generation-input.json` and `--kind generation` to propose three real model
outputs for human review. Invalid JSON or repeated questions trigger bounded
retries and a recoverable failure; illustrative templates are never substituted.
For full datasets, send exactly three approved, corrected seeds with purpose
`dataset` and target count `12`, `24`, or `48`. The corrected seeds retain their
text and approval. New examples remain unapproved.

The public registry pins Qwen3 0.6B, 1.7B, 4B and 8B to immutable HF revisions.
The worker does not execute model-supplied code. System/user prompt tokens are
masked from training loss. Responses and EOS are supervised; overlong examples
fail explicitly rather than silently truncating the answer. The quick preset
uses one epoch, the standard preset three, with capped optimizer steps and a
one-hour total process deadline.

`events.jsonl` contains actual completed units. `result.json` appears atomically
only after success; `error.json` describes failure. Each run needs a fresh job
directory. Stop the worker with SIGTERM; a production supervisor must bound the
grace period and verify all child processes are gone before declaring cancellation.

Training compares the base and adapted models on identical held-out questions,
using greedy output and measured expected-answer loss. Loss and exact matching
are not a semantic quality judgment. Review the actual outputs. Downloads include
`adapter.tar.gz`, `evaluation.json`, the resolved configuration and a hash manifest.
The adapter requires the exact original model and a PEFT runtime; it is not a
standalone executable model for Mac. Model/environment provenance uses the
repository's existing immutable snapshot and runtime-manifest utilities.

Official revision source: `https://huggingface.co/api/models/Qwen/Qwen3-{size}`;
registry revisions were resolved on 2026-09-30.
