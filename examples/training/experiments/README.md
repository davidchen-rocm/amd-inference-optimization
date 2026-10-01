# Bilingual policy-to-JSON LoRA experiment

This reproducible experiment uses five fictional shops, an explicit reference
table, 100 author-specified training examples and 20 held-out prompts. It asks
the model to read English or Chinese requests, look up policy facts, make return
and support-hour decisions, abstain on missing policies, and produce a strict JSON
object. No model-generated data, customer data or contact details are included.

The input questions are balanced: 50 English and 50 Chinese training rows;
10 English and 10 Chinese held-out rows. Every shop appears in both languages.
Questions in the evaluation split never appear in training. The same reference
facts are deliberately supplied at training and evaluation: this tests applying
instructions and supplied facts, rather than remembering unavailable knowledge.

The 20 held-out questions have four groups, five questions each:

| Group | Check |
| --- | --- |
| Shipping | Reject a false 99-day claim and a request to wrap the answer in Markdown. |
| Return boundary | Distinguish the included final return day from the first excluded day. |
| Support boundary | Include opening time and exclude closing time on a working weekday. |
| Unknown policy | Avoid confirming invented student discounts and warranties. |

The JSON schema has exactly six fields: `shop`, `language`, `topic`, `status`,
`value` and `unit`. Expected answers and policies are plain text and JSON, so a
maintainer can audit the entire dataset without a model or GPU.

Six inputs share exactly the same train/test split and pinned model registry:

| Input | Epochs | Optimizer steps |
| --- | ---: | ---: |
| `qwen3-0-6b-quick.json` | 1 | 25 |
| `qwen3-0-6b-standard.json` | 3 | 75 |
| `qwen3-4b-quick.json` | 1 | 25 |
| `qwen3-4b-standard.json` | 3 | 75 |
| `qwen3-8b-quick.json` | 1 | 25 |
| `qwen3-8b-standard.json` | 3 | 75 |

All runs use seed 42, rank-16 LoRA, BF16, a learning rate of 0.0002, assistant-only
loss, gradient accumulation of four rows and the framework's one-hour process
deadline. They use the existing worker without relaxing the production API.
Each run measures the base model before training and the adapter after training
on identical held-out questions, with greedy decoding and thinking disabled.

Recreate the checked-in files and verify the fixtures on a CPU host:

```sh
PYTHONPATH=src python -m macfit_training.experiments materialize \
  --output examples/training/experiments
PYTHONPATH=src python -m macfit_training.cli validate --kind training \
  --input examples/training/experiments/qwen3-0-6b-quick.json
PYTHONPATH=src python -m pytest tests/training/test_lora_experiments.py -q
```

Use `--models qwen3-8b` to materialize only the 8B pair, or supply multiple
distinct registry IDs after `--models`. The default is 0.6B, 4B and 8B; the pinned
1.7B model is also selectable. Arbitrary model paths and repeated IDs are rejected.

Run one input in a fresh private directory using the existing supervised worker
as described in the [training guide](../../../docs/training/README.md). Run GPU
jobs sequentially; do not bypass the exclusive GPU lock. The experiment does not
download, train or install anything merely by materializing its input files.
For a limited GPU window, start with the 0.6B quick run to check the full pipeline,
then compare its standard run and the 4B/8B runs. The one-hour cap is a bound, not a
runtime prediction; model load, shared GPU contention and inference length all
affect elapsed time. Do not start a run that cannot finish before the reserved
backup period. A larger model or longer training is not assumed to be better.

Score a successful worker's real outputs without running the model again:

```sh
PYTHONPATH=src python -m macfit_training.experiments assess \
  --input examples/training/experiments/qwen3-0-6b-quick.json \
  --evaluation /path/to/job/artifacts/evaluation.json \
  --output /path/to/training-data/evidence/policy-json-0-6b-quick.json
```

The scorer verifies every evaluation ID, question and expected answer against
the fixed fixture. It binds the resolved input and actual evaluation artifact by
SHA-256. It rejects missing rows, duplicates, altered ground truth, duplicate
JSON keys, nonstandard constants, Markdown fences and extra schema fields.
JSON key order and whitespace do not affect structured exactness. Wrong types,
including a boolean used as an integer, fail schema validation.

The assessment reports strict JSON, schema, grounded fact, language route,
topic route and structured exact-match rates before and after training. It also
reports the narrow count of parsed `status=known` claims on rows whose policy is
unknown. That count does not detect every possible hallucination. Review the
actual before/after text and expected-answer loss alongside these measurements.

Keep the assessment under the training service's `data/evidence` directory so
the configured OpenShift backup preserves it with jobs and source code. Do not
add an unregistered file to a completed job's hash-bound artifact directory.

This is a small synthetic fixture with one seed and five examples per group.
High scores do not establish performance on real users, arbitrary policies,
general bilingual tasks or production workloads. A drop in expected-answer loss
does not compensate for a new wrong fact or invalid JSON. Adapters remain PEFT
adapters requiring the exact original model; they are not standalone Mac models.

## Independent capability checks

After training, run a separate ten-question diagnostic for unrelated arithmetic,
string manipulation, JSON reasoning and ordering tasks. Five prompts are English
and five are Chinese. The diagnostic uses an independent generic system prompt,
greedy decoding, thinking disabled, and at most 64 generated tokens per question.
It checks the original base model and then its LoRA adapter on the same prompts.

```sh
PYTHONPATH=src python -m macfit_training.regression \
  --jobs /path/to/completed-job-a /path/to/completed-job-b \
  --output /path/to/training-data/evidence/lora-regression.json \
  --walltime-seconds 1800 --job-timeout-seconds 600 \
  --stop-at 2026-10-01T03:45:00Z
```

Use a fresh output filename. Set the absolute stop time for the current GPU
window; the timestamp above is only an example. `--validate-only` checks the
operator's paths, limits and timestamp without loading model weights. A live run
verifies each completed job's artifact hashes, provenance, recorded pinned
snapshot and local model file hashes. It safely extracts the hash-bound adapter
bundle instead of trusting the mutable `adapter` directory. All model loading
is local-only, requires ROCm BF16, disables remote code and uses safetensors.

A parent process supervises one child per job, bounds lock wait, hashing, model
load and native GPU calls, and terminates the child's process group on timeout.
It reserves eight seconds for cleanup before the absolute deadline. Children
share the training GPU lock. Atomic evidence contains actual base/adapter outputs,
input/snapshot/adapter hashes, runtime versions, pass fractions and individual
changed outcomes; it remains below one MiB and preserves partial failure status.
The protocol binds the regression source and imported strict JSON helper source
separately, so helper changes remain visible without changing this wrapper.

Text checks use NFKC normalization and trim surrounding whitespace; explanations
are rejected. JSON checks ignore key order but require exact keys and types and
reject duplicate keys. These ten tiny checks can expose an obvious regression;
they cannot establish broad preservation of capability or overall model quality.
