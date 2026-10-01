# Adaptive policy-to-JSON robustness diagnostic

This separate experiment was designed **after inspecting failures from the
original policy-to-JSON runs**. It trains on 360 deterministic synthetic examples
to address conflicting output formats, false policy claims, return limits,
support-time boundaries and missing policies. It deliberately retains the
**exact original 20 evaluation questions and expected answers** for comparison.
Consequently, its scores are an adaptive diagnostic and must not be described as
fresh unseen test accuracy or general model performance.

The original [baseline experiment](../experiments/README.md) and assessor remain
unchanged. This version uses its own dataset manifest, goal and strict assessor.

| Training group | English rows | Chinese rows | Total |
| --- | ---: | ---: | ---: |
| Shipping false claims and conflicting formats | 40 | 40 | 80 |
| Return eligibility and calendar-day boundaries | 50 | 50 | 100 |
| Support opening, closing and weekday boundaries | 60 | 60 | 120 |
| Unknown student discounts and warranties | 30 | 30 | 60 |
| Total | 180 | 180 | 360 |

Five fictional shops share the original explicit reference table. Training
questions are unique after the worker's normalization and do not contain the
held-out question text. The factual source is deliberately shared with the
evaluation. Templates are authored in code; labels are computed independently
from policy values, unused/used state, calendar age, weekday and UTC minute.
Return eligibility includes the last permitted day. Support hours include the
opening instant and exclude the closing instant. Unknown policy values remain
null even when a prompt requests an invented offer.

No model-generated labels or real customer data are used. Independent human
review has **not** occurred. The input's `approved=true` fields satisfy the
existing bounded worker contract for this operator-created synthetic fixture;
the versioned manifest explicitly records that they do not certify human review.
Audit the reference table and deterministic label rules before reuse for a real
task.

The default materialization creates Qwen3 0.6B, 4B and 8B standard inputs. All
three use the pinned registry, rank-16 BF16 LoRA, three epochs and 270 optimizer
steps, with the existing one-hour total job cap. They use the same training and
evaluation data. `--models` selects distinct registry IDs; arbitrary checkpoints
are rejected. The optional 1.7B registry model is supported.

```sh
PYTHONPATH=src python -m macfit_training.robustness materialize \
  --output examples/training/robustness
PYTHONPATH=src python -m macfit_training.cli validate --kind training \
  --input examples/training/robustness/qwen3-4b-standard.json
PYTHONPATH=src python -m pytest tests/training/test_lora_robustness.py -q
```

Run an input in a fresh private job directory with the existing supervised worker
and shared GPU lock. Do not start a job that cannot finish before the GPU backup
reserve. Materialization and validation do not load models or use a GPU.

The existing trusted operator campaign can submit the three standard recipes
sequentially through the normal supervised queue and write bound assessments:

```sh
PYTHONPATH=src python -m macfit_training.campaign \
  --data-dir /path/to/training-data \
  --campaign-id adaptive-robustness-v1 \
  --variant robustness --stop-at 2026-10-01T03:45:00Z
```

Choose a **new campaign ID** distinct from the baseline run, and set the stop time
for the current GPU window. The default variant remains `baseline`, with its
original six quick/standard recipes. Campaign evidence records the variant;
legacy evidence without that field is treated as baseline. Resuming an ID under
a different variant is rejected. Artifact hash checks and failure cancellation
apply to both variants, and no HTTP or account-authentication bypass is added.

Score the actual successful worker artifact with this version's assessor:

```sh
PYTHONPATH=src python -m macfit_training.robustness assess \
  --input examples/training/robustness/qwen3-4b-standard.json \
  --evaluation /path/to/job/artifacts/evaluation.json \
  --output /path/to/training-data/evidence/adaptive-robustness-4b.json
```

The assessor binds its own training rows, reference, original evaluation and
goal by SHA-256. It verifies all artifact identities, questions and expected
answers before reusing the strict per-output scorer. It reports actual input,
training, evaluation artifact and scorer source hashes; before/after strict JSON,
schema, grounded fact, language/intent and structured exactness metrics; and the
same four evaluation groups. Missing or altered results are rejected. Markdown
fences, duplicate keys, extra schema fields, wrong value types and incorrect
boundary decisions receive no structured-correct credit.

The scorer records its own wrapper source and the actual imported strict helper
module separately (`strict_scoring_helpers_source_sha256`). Changing JSON parsing
or factual scoring helpers therefore changes the recorded helper identity even
when this adaptive wrapper has not changed.

Store the assessment under `data/evidence` for OpenShift backup. Preserve the
actual outputs and the reusable adapter alongside it. Also run the independent
[capability regression checks](../experiments/README.md#independent-capability-checks)
to look for effects on unrelated simple tasks.

The system prompt is also more explicit in this version. Therefore before/after
figures isolate adapter effects within this version, while differences from
the original experiment can reflect **both prompt and training changes**.
The evaluation is small, adaptively reused and covers only five examples per
group with one fixed seed. Higher scores do not establish production quality or
general robustness, and lower loss alone is insufficient evidence of correctness.
