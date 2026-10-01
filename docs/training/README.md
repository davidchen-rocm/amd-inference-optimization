# Reusable MacFit LoRA training

The framework runs real supervised LoRA training and model-generated teaching
examples on ROCm. Each run takes an editable JSON configuration, preserves an
immutable input snapshot, and produces measured results plus downloadable files.
It also provides an offline CPU command to merge a completed adapter into its
exact base model. The web service and this CLI use the same validation rules.

For the website connection, authentication and persistent backups, see the
[OpenShift deployment guide](../../deploy/macfit-training/README.md) and
[service configuration](../../src/macfit_training/service/README.md). The CLI can
be reused independently on a compatible GPU host after this temporary website
deployment ends; it does not require the original server or cluster.

## Start with a configuration file

Copy [training-input.json](../../examples/training/training-input.json) to a new
file such as `my-training.json`, then edit it. This complete example contains 12
training answers and three held-out questions for a **fictional** shop. Replace
the example content with your reviewed data before a real task.

```sh
cp examples/training/training-input.json my-training.json
python -m macfit_training.cli validate --kind training --input my-training.json
```

The command prints the resolved configuration, including the fixed model
revision and training settings; it does not load a model or use a GPU. If you
installed the package, `gpuopt-train` is equivalent to
`python -m macfit_training.cli`.

| Editable field | Meaning |
| --- | --- |
| `model_id` | `qwen3-0-6b`, `qwen3-1-7b`, `qwen3-4b`, or `qwen3-8b` |
| `task` | `answers`, `writing`, `format`, `classify`, or `custom` |
| `goal` | Your instructions, at least 12 and at most 6,000 characters |
| `language` | `en`, `zh`, or `multi` |
| `source` | Optional reference material, at most 12,000 characters |
| `preset` | `quick` or `standard` |
| `training` | 3–500 reviewed examples with unique IDs and questions |
| `evaluation` | 3–50 reviewed, separate questions and expected answers |

One training row has this shape:

```json
{
  "id": "train-001",
  "approved": true,
  "messages": [
    {"role": "user", "content": "When will my order ship?"},
    {"role": "assistant", "content": "Orders ship within two business days."}
  ]
}
```

You may prepend one `system` message. Without one, the framework inserts the
job's `goal` and `source` as the system prompt. An evaluation row instead uses
`id`, `approved: true`, `question`, `expected`, and optional `system`. Content
whitespace is preserved. Mark rows approved only after reviewing their contents.

The total input is limited to 3 MiB. Assistant answers can contain up to 24,000
characters and questions up to 12,000, but the **complete tokenized training
example**, including system text and answer, must fit 2,048 tokens. Shorten an
overlong example; the worker fails explicitly without silently truncating it.
Question deduplication uses Unicode normalization, case folding and whitespace
normalization. It catches exact normalized overlap, not semantic paraphrases.

The JSON file is independent of the website. It intentionally does not accept
arbitrary Python, shell commands, paths, model repositories or hyperparameters.
Do not add or edit server-derived `base_model`, `training_config`, or
`generation_config` fields. Changing the model whitelist or presets requires a
reviewed change to `src/macfit_training/config.py` and new validation, not a
browser-supplied override. A training run starts from the original base model;
checkpoint resume and continued training from an earlier adapter are not yet
implemented.

## Install and run on ROCm

Use Python 3.11 or newer. Install the ROCm build of PyTorch appropriate to the
host first; the package's version pin does not choose the accelerator wheel for
you. The tested GPU environment uses PyTorch `2.10.0+rocm7.1`, Transformers
`4.57.6`, PEFT `0.18.1`, Accelerate `1.12.0` and Safetensors `0.8.0`.

```sh
python -m pip install -e '.[training]'
python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'
```

The version check must show ROCm/HIP and an available GPU. On an installed
service host, use its existing Python environment and account. Do not replace a
working production environment just to run this example.

Prepare a new private directory for every run:

```sh
mkdir -m 700 /absolute/path/job-001
cp my-training.json /absolute/path/job-001/input.json
MACFIT_GPU_LOCK=/absolute/path/shared-gpu.lock \
MACFIT_MODEL_CACHE=/absolute/path/model-cache \
  python -m macfit_training.cli run --kind training --job-dir /absolute/path/job-001
```

Choose writable paths owned by the training account. Every CLI/service process
sharing the GPU must use the **same** `MACFIT_GPU_LOCK`; an advisory lock cannot
coordinate unrelated programs that ignore it. The default is
`/tmp/macfit-training-gpu.lock`. The first run needs network access to download
the whitelisted public checkpoint at its immutable revision. No model-supplied
Python is executed (`trust_remote_code=False`).

`quick` uses one epoch and at most 50 optimizer steps; `standard` uses three
epochs and at most 300. Both use batch size 1, gradient accumulation of 4, AdamW
at `2e-4`, LoRA rank 16 / alpha 32 / dropout 0.05, BF16 base weights, gradient
checkpointing and attention plus MLP projections. The base weights are frozen.
For a short final accumulation group, the gradient divisor is the actual group
size. With 12 examples, `quick` performs three optimizer steps. Full fine-tuning,
quantized training and multi-GPU training are not implemented by this entrypoint.

System and user tokens have loss labels `-100`; only the assistant answer and
its EOS token are supervised. Qwen3's chat template uses
`enable_thinking=False` for training and evaluation. An empty closed thinking
prefix is part of the masked prompt, not a supervised chain of thought.

Jobs have a one-hour process deadline, including waiting for the lock, model
download, hashing, training and evaluation. `events.jsonl` reports actual stages
and completed examples/questions/optimizer steps. SIGTERM requests cancellation.
The production supervisor also enforces a grace period and process cleanup; the
standalone CLI alone cannot guarantee immediate interruption inside native code.
Use a fresh job directory after failure or cancellation. An existing run is
never silently resumed or overwritten.

## Generate candidate teaching examples

Copy and edit [generation-input.json](../../examples/training/generation-input.json).
Use `purpose: "preview"`, `target_count: 3` and empty `seeds` for three proposed
examples. Validate and run it as above using `--kind generation`.

The website chooses Qwen3 4B (`qwen3-4b`) for generation by default when available.
The generation model only drafts examples; the training base model is selected
separately in **Train & test**. For the CLI, set `model_id` independently in each
generation or training JSON file. The checked-in generation example uses 0.6B
for a small smoke run; change it to `qwen3-4b` to match the website's default.
A smaller model can exhaust its retries on malformed or repeated examples.
Choosing 4B does not waive human review or guarantee correct, diverse data.

For a dataset, set `purpose: "dataset"`, choose `target_count` 12, 24 or 48, and
provide exactly three corrected, approved seeds:

```json
{
  "id": "seed-001",
  "question": "When does an order leave your shop?",
  "answer": "Within two business days.",
  "approved": true
}
```

The seeds are included in the requested total and retain their content and
approval. All newly generated examples have `approved: false`. Review their
answers before constructing training rows. Generation does not create or approve
an independent evaluation set for you. Reserve separate human-reviewed tests.

The worker samples one actual LM output at a time, validates its JSON and rejects
duplicate questions. Each requested slot has a different task-specific intent;
the prompt includes recent accepted questions. A rejected output is fed back
with its failure reason and a request to correct it. These prompt instructions
encourage diversity; only exact normalized question uniqueness is enforced.
It retries at most three times per new example. Persistent
invalid output fails with a retryable error and keeps your input unchanged; it
does not substitute templates or publish a partial successful dataset. Prompts
are bounded to 4,096 input tokens and generation to 768 output tokens per attempt.

## Read results and judge quality

`result.json` is written atomically after all output files have been produced.
`error.json` describes a failed job. Do not infer success from adapter files or a
progress message alone. Service downloads become available only after the worker
exits successfully and their sizes and SHA-256 hashes have been verified.

| File | Contents |
| --- | --- |
| `artifacts/adapter.tar.gz` | PEFT adapter weights, configuration and usage notes |
| `artifacts/evaluation.json` | Actual base/adapter answers, expected-answer losses and exact-match rates |
| `artifacts/resolved-config.json` | Immutable, fully resolved input and preset |
| `artifacts/manifest.json` | Model identity, provenance, input hash and output hashes |
| `result.json` | Complete result and the artifact inventory, including the manifest's hash |
| `model-snapshot.json` | Private inventory/hash of the model and tokenizer files |
| `runtime-environment.json` | Private package, interpreter and reused framework identities |
| `training-source.json` | Private hashes of the training implementation files |

Generation jobs have `generation.json` instead of evaluation and adapter files.
Keep private metadata with the job if you will later perform a verified merge.
The download manifest contains hashes for the other artifacts; its own hash is
in the outer `result.json` inventory, avoiding a circular hash.

Evaluation uses identical held-out prompts before and after training, greedy
decoding, no thinking, and at most 256 new tokens per answer. Expected-answer loss
is teacher-forced assistant-token cross entropy, weighted by supervised token
count. The training loss is measured over the actual optimization passes, not
recomputed as a final score. Exact matching ignores normalized case/whitespace;
it is not a semantic correctness measure. Check `stopped_by_limit` in the answer
measurements when an output appears truncated.

Lower loss does **not** establish better answers. In the MI300X smoke run on
2026-09-30, Qwen3 0.6B completed 12 rows / 3 steps in about 32.7 seconds, with
10,092,544 trainable LoRA parameters out of 606,142,464 total parameters. Held-out
expected-answer loss decreased from about 1.312 to 0.899, yet the adapted model
answered a shipping question incorrectly. This verifies training and artifact
production, not production answer quality. The generated adapter bundle was
37,316,794 bytes; size and timing vary between runs.

The minimum three tests are a workflow requirement, not statistical evidence.
Inspect the actual before/after responses, keep representative and difficult
questions separate from training, check unrelated capabilities for regressions,
and expand testing before using the model. When `source` is supplied in the
default system prompt, both training and evaluation receive that same reference;
this is a test of answering with the reference, not proof the weights memorized
the facts. Do not repeatedly tune against your final acceptance set.

## Use an adapter

An adapter is not a standalone model, executable app, GGUF or MLX package. It
requires its exact base model revision and a compatible Transformers/PEFT
runtime. `adapter/USAGE.md` inside the bundle includes a minimal loader using
the pinned repository and revision. After verifying the bundle hash against
`result.json`, extract it to a new directory and attach it with
`PeftModel.from_pretrained(base, adapter_directory)`. Apply the base tokenizer's
chat template with `enable_thinking=False` and the same system instructions used
for your task. Keep base-model licensing and distribution requirements with any
deployment. See the official [PEFT checkpoint format documentation](https://huggingface.co/docs/peft/developer_guides/checkpoint).

## Export merged weights offline on CPU

This optional local CLI command verifies a completed job and creates a new
Transformers Safetensors checkpoint. It does not expand the website API's model
whitelist or start a GPU job.

```sh
python -m macfit_training.cli export-merged \
  --job-dir /absolute/path/job-001 \
  --output /absolute/path/merged-001
```

The complete original job, including `model-snapshot.json`, must be available.
By default the command reads the original local snapshot path. To use an exact
copy on another machine, supply `--base-model-dir /absolute/path/local-snapshot`.
The copy must contain the same model/tokenizer files and bytes as the captured
snapshot; an arbitrary checkpoint with the same display name is rejected.

This is an **offline** operation: every model/tokenizer load uses
`local_files_only=True`, `trust_remote_code=False`, CPU placement and BF16
weights. Install PyTorch, Transformers, PEFT, Accelerate and Safetensors in the
export environment first. No GPU is required. A 0.6B BF16 base has roughly 1.2 GB
of raw weights, but peak RAM also includes the runtime, adapter, loaded tensors
and merge temporaries. **1.5 GB of total RAM is not a sufficient guarantee.**
Allow several GB of free RAM for 0.6B and substantially more for larger models,
plus disk space for another complete checkpoint.

Before loading weights, export verifies input/config/manifest identities, all
artifact hashes and the exact local base snapshot. Job metadata, artifact paths,
output paths and their parent directories must not be symlinks. HF cache files
may remain ordinary blob symlinks inside a real snapshot directory; the reused
snapshot verifier hashes their target bytes. The adapter archive accepts only
bounded, flat regular adapter files, never links or traversal paths. Export
reads the verified archive rather than the mutable `job/adapter/` directory.

The output parent must already exist; the output directory must not exist and
must be outside the job and base-model directories. Existing files are never
overwritten. A failed/interrupted export retains `EXPORT_INCOMPLETE.json`; retry
with a fresh output path. Only a completed `export-manifest.json` plus the absence
of that marker indicates a complete export. The manifest hashes the exported
files and identifies the source adapter, pinned base and input.

PEFT `merge_and_unload(safe_merge=True)` merges the adapter and checks for invalid
weights; the resulting checkpoint no longer requires a separate PEFT adapter.
It still needs a model runtime. Floating-point rounding may change outputs, so
rerun your acceptance tests against the merged checkpoint. `safe_merge` is not
a quality or factuality check. GGUF conversion, quantization, MLX packaging and
online model hosting are separate steps and are not provided by this command.
See the official [PEFT LoRA merge reference](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/lora#peft.LoraModel.merge_and_unload)
and [Transformers local loading options](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained).

## Validate changes without a GPU

```sh
python -m pip install -e '.[dev]'
pytest -q tests/training/test_core_training.py tests/training/test_worker_training.py tests/training/test_core_export.py
```

These tests verify validation, approval, held-out separation, assistant-only
labels, generation failure behavior, artifact integrity and export boundaries.
They do not stand in for real GPU training, real model generation or real CPU
merge execution. GPU evidence is recorded separately; do not present mocked
model calls as model-quality measurements.

## Deployment status and verification scope

The temporary deployment's configured GPU work cutoff is **2026-09-30 at 23:45
America/Detroit** (`2026-10-01T03:45:00Z`). Admission can close earlier when the
remaining window cannot fit a job and its cleanup reserve. The OpenShift final
backup is scheduled for **23:55** (`03:55:00Z`), ten minutes after that cutoff.
Scheduling this job does not prove that its transfer or verification succeeded.

At the 2026-09-30 deployment checkpoint, a verified backup contained three job
records. That establishes recovery of those saved records, not three successful
training runs or the completion of the still-scheduled final backup. Check the
current receipt's `source_created_at`, the finalizer Job result, and
`/archive/finalization.json` with `verified: true` after the scheduled run. A
paused CronJob alone is insufficient. The CPU archive provides authenticated
history and artifact reads; it does not generate examples, train, or host model
inference. Failed final backups retain the last verified snapshot.

Website job access requires both the private bridge gateway and a Firebase ID
token for the configured project. Ownership comes from the verified token, not
a submitted user ID. The verifier checks signature, issuer, audience, required
claims and expiration with a 30-second clock tolerance. **It does not query
Firebase revocation or disabled-account state.** A previously issued token can
remain usable until its expiration plus that tolerance after an administrative
revocation. Immediate per-user revocation requires additional server-side
verification; it is not implemented here. Certificate refresh also requires
access to Google's signing-certificate endpoint, including for CPU archive reads.
Expired key caches fail closed if refresh fails.

[validation.json](validation.json) preserves the earlier Linux CPU run:
**644 tests passed at 2026-09-30T20:59:14Z** against the bundle identified there.
It is historical evidence, not a full-suite result for every later change.
[deployment-validation.json](deployment-validation.json) separately records a
later **54-test focused CPU run** for archive, backup, finalizer and manifest
behavior, with its exact command, source fingerprint and private-log hash.

[linux-final-validation.json](linux-final-validation.json) records the subsequent
complete Python gate: **650 tests passed at 2026-09-30T21:20:39Z** on Linux x86_64
with Python 3.12.3, using `python -m pytest -m 'not gpu' -q`. It ran from an isolated
310-file public-source export with GPU visibility disabled. Bundle, per-file
inventory and log hashes identify exactly what was tested; exported source bytes
were unchanged after the run. The active service was not replaced or restarted.
The single Starlette TestClient/httpx deprecation warning did not fail a test.

These CPU results do not replace live OpenShift probes, real GPU execution, an
interactive Firebase/Google sign-in test, or human assessment of model quality.
Deployment checks must also confirm image pulls, PVC permissions, SSH access,
gateway authentication, archive fallback and the scheduled final backup result.
