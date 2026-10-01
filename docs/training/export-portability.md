# Offline CPU export portability check

`macfit_training.export_verification` is an operator-only check of a completed LoRA
job. It creates a fresh Transformers safetensors export with the existing CPU merge
implementation, reloads the original pinned base plus its verified PEFT adapter on
CPU, reloads the merged export on CPU, and compares six original held-out questions.
The inputs come from the completed job, including its original goal/reference or an
explicit per-question system prompt. No policy benchmark labels or fixture IDs are
hardcoded into this tool.

Restore a completed job's `input.json`, `result.json`, `model-snapshot.json` and its
four registered artifact files before using this command. The model snapshot must
already be local. An optional `--base-model-dir` can point at a restored copy of the
same exact snapshot: every model/tokenizer file must still pass its recorded hash
inventory. Both output parent directories must exist. The merged directory and
compact evidence file must be fresh, separate from each other and outside the job.

For example, with private local paths chosen by the operator:

```sh
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
PYTHONPATH=src python -m macfit_training.export_verification \
  --job-dir /private/jobs/completed-lora \
  --base-model-dir /private/models/pinned-qwen3-snapshot \
  --merged-output /private/exports/completed-lora-merged \
  --output /private/evidence/cpu-export-portability.json \
  --question-ids eval-juniper-1 eval-juniper-2 eval-juniper-3 eval-juniper-4 \
                 eval-cedar-1 eval-cedar-3 \
  --walltime-seconds 600 \
  --stop-at 2026-10-01T03:45:00Z
```

These six example IDs are operator-selected questions from the fictional policy
experiment. Choose exactly six distinct IDs from any other job's evaluation list.
When no IDs are supplied, the deterministic selector covers distinct JSON `topic`
fields first, then balances the remaining questions using their declared language
or Chinese characters in their question text. It never reads expected answers into
the model prompt. At least six evaluation questions are required.

`--validate-only` checks bounded metadata, identities, selected questions and output
paths without importing Torch. It does not prove that the large adapter or snapshot
bytes are intact. Actual artifact and model byte verification runs inside supervised
children, within the same total walltime budget as model loading, merging, generation
and final export validation. The maximum is 600 seconds, with eight seconds reserved
for terminating/reaping process groups. An earlier absolute `--stop-at` wins.

The comparison child disables all GPU visibility, uses four Torch CPU threads and
one interop thread before loading or merging models, and disables network downloads
and remote model code. Each branch uses greedy decoding, the original chat template
with thinking disabled, at most 2048 prompt tokens and at most 96 generated tokens.
The pinned base is loaded in BF16. The evidence separately records actual parameter
dtype inventories because PEFT adapter parameters may have another dtype.

The evidence contains original source input, adapter archive and snapshot identities,
metadata hashes, selected question hashes, merge file inventory, actual output text
and token IDs, strict expected-answer scores, limit termination and per-case exact
text/token differences. Strict JSON answers use typed structural equality and reject
duplicate keys, prose, code fences and nonstandard numeric constants. Other expected
answers use Unicode NFKC and collapsed whitespace, preserving letter case. Questions
and labels are the original job's evaluation rows. Finite raw logits are checked at
only the prompt and generated-sequence last-token boundaries; this is not a claim
that every intermediate generation logit was inspected.

After the comparison child and its descendants have been reaped, a second bounded
child verifies the original source artifact bytes again and independently hashes
every saved merged model/tokenizer file against the same export manifest. The parent
accepts success only when source identities, selected questions, actual scores,
CPU/thread placement and this post-exit inventory attestation all agree. Cleanup is
attempted exactly once per newly spawned process group. Failure evidence records
exception types, without persisting exception messages or process arguments.

Keep the sub-1-MiB JSON evidence in the backup. Keep full merged weights in the private
export directory outside the backup; they can be regenerated from the retained
adapter archive and separately obtained exact base snapshot. A timed-out export is
not a usable completed model. The tool never publishes weights or changes a service.

Six questions diagnose portability; they do not establish general model quality.
BF16 merge rounding and CPU arithmetic can change output tokens without changing a
structured answer. The evidence preserves both outcomes. This check exports and
reloads Transformers safetensors. It does not certify GGUF, MLX, a Mac app, website
serving, quantization or another inference runtime.
