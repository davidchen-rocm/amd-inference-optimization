# Large-model LoRA training capacity probe

This operator-only CLI measures five real optimizer updates on a tiny synthetic
causal-LM continuation batch. It answers whether this bounded LoRA configuration
can perform forward, backward and AdamW steps on the available ROCm GPU. It does
**not** produce a useful fine-tuned model, assess task quality, or publish adapter
weights. The website's training model whitelist remains unchanged.

The two allowed identities come from the pinned benchmark catalog:

| Model | Immutable revision |
| --- | --- |
| Qwen3 14B | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| Qwen3 32B | `9216db5781bf21249d130ec9da846c4624c16137` |

Both models must already exist in the host's Hugging Face cache. Snapshot lookup,
model loading and tokenization use local files only, without model-provided
Python or new downloads. The actual model/tokenizer files are hashed with the
existing snapshot verifier; the evidence retains the complete per-file manifest
as well as its digest, file count and bytes. The private manifest includes the
original cache root; change only that root when verifying an exact redownload or
copy after the temporary host expires. The file inventory and pinned coordinates
remain hash-bound. A missing cached checkpoint fails explicitly.

Run as the training account with its existing runtime and shared GPU lease:

```sh
MACFIT_MODEL_CACHE=/absolute/path/model-cache \
MACFIT_GPU_LOCK=/absolute/path/shared-gpu.lock \
  python -m macfit_training.training_capacity \
  --models qwen3-14b qwen3-32b --sequence-length 128 \
  --walltime-seconds 1200 --model-timeout-seconds 600 \
  --stop-at 2026-10-01T03:45:00Z \
  --output /absolute/path/fresh-training-capacity.json
```

The example absolute cutoff is timezone-qualified; change it to your host's real
GPU deadline. A fresh output name is required. The parent enforces the earliest
of total walltime, per-model timeout and absolute cutoff, including GPU-lock
waiting, cache hashing, model loading and training. Eight seconds are reserved
for termination/reaping. Every isolated process group is cleaned on success,
timeout or interruption. Failed or skipped models remain explicit in the atomic
JSON. This bounds user processes, not an unresponsive driver or host. The hidden
child mode relies on its supervising parent and must not be invoked directly.

## What is trained and measured

The frozen base uses BF16. Rank-8 LoRA targets the attention and MLP projections,
with alpha 16, dropout 0.05, non-reentrant gradient checkpointing and no KV cache.
PEFT promotes trainable adapters to FP32, matching the website trainer's normal
precision approach. The evidence checks and records actual frozen/trainable
parameter counts and dtypes, as well as AdamW state dtypes. All base and adapter
parameters must reside on the GPU; CPU/offloaded or full-parameter training is
refused. The website uses rank 16 and longer examples, so this smaller probe
cannot establish that all website configurations fit.

Batch size is fixed at one and sequence length is exactly 128 or 256. Fixed
public fictional reference and target texts are tokenized into valid vocabulary
IDs. The reference is repeated to the prefix length, the target to 31 tokens,
and one EOS token is appended. Prefix labels are `-100`; causal shifting leaves
exactly **32 supervised continuation tokens**. The report preserves the texts,
their SHA-256 values, and canonical-JSON hashes of the actual input IDs, labels
and attention mask. This synthetic workload measures capacity; it is not an
independent accuracy test or a normal chat-template training dataset.

Token validation uses `len(tokenizer)`, including added special tokens, rather
than the base-only `tokenizer.vocab_size`. The total must fit the pinned model
configuration, and both loaded input/output embedding row counts must agree
with that configuration. All dimensions and the actual EOS ID are retained in
the evidence. This permits added EOS tokens and the model's padded vocabulary
without silently resizing or modifying the pinned base.

Two real warmup updates create optimizer state and warm the workload, followed
by three timed updates of the same batch. Warmups change adapter weights but are
excluded from the timing summary. Every update checks finite nonnegative loss
and finite gradient norm, clips gradients to 1.0, and applies AdamW at `2e-4`
with weight decay 0.01. A selected LoRA B matrix must actually change. No
inference-mode or no-gradient substitute is used.

Synchronized host timing surrounds zero-grad, forward, loss validation, backward,
gradient checks/clipping and AdamW. Individual samples and median/minimum/maximum
step times are retained. Rates count real optimizer steps and exactly 32
supervised tokens per step; they are **not generated tokens per second**. Memory
peaks include this model, adapter, batch and optimizer and use PyTorch allocated
and reserved statistics; they exclude other processes/non-Torch allocations.

The JSON remains strictly smaller than 1 MiB and can enter the existing verified
OpenShift evidence backup. Only evidence is preserved: trained tensors and
optimizer state are discarded when the child exits. Confirm the independent
backup manifest and receipt after running the probe.

See the official [PEFT model precision documentation](https://huggingface.co/docs/peft/main/en/package_reference/peft_model)
and [PyTorch AdamW API](https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html).
The actual loaded package versions are recorded, so current documentation must
not substitute for the installed runtime's behavior.
