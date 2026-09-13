# AgentDecision contract

Return one JSON object. Use the schema supplied in `agent-context.json` as the final authority.

Required fields:

```json
{
  "current_stage": "GENERATE_HYPOTHESIS",
  "evidence_used": ["artifact-ref"],
  "conclusion": "Concise evidence-bounded conclusion",
  "confidence": 0.8,
  "missing_evidence": [],
  "requested_profile_level": null,
  "execution_map_updates": [],
  "bottleneck_assessment": null,
  "limit_estimate": null,
  "hypothesis": null,
  "proposed_experiment": null,
  "proposed_next_stage": "CREATE_EXPERIMENT"
}
```

Apply these invariants:

- Match `current_stage` to the context exactly.
- Keep `confidence` between 0 and 1.
- Cite only evidence refs present in the current task.
- Cite the immutable artifact identity supplied by the context; a path without its captured digest is
  not sufficient evidence for a live decision.
- Request at most one next profiling level and explain which uncertainty it resolves.
- Propose one primary hypothesis per experiment.
- Treat `proposed_next_stage` as a request; never mutate state directly.
- Omit a conclusion that the Gate Engine must make. Never set a gate verdict.

For a hypothesis, include observed evidence, interpretation, proposed change, expected performance signature, expected end-to-end effect, risks, required validation, and stop condition.

For an experiment, identify either a source patch artifact or a runtime configuration delta, the build or binary-reuse rule, smoke test, microbenchmark when available, end-to-end benchmark, quality evaluation, and expected artifact outputs.

Use `change.env` to set run-only variables and `change.unset_env` to restore default behavior by
removing inherited overrides. Do not encode unsetting as an empty string. Source mutation and runtime
environment are orthogonal: never leak a run-only setting into configure or build. Commands use the
stage names `build`, `smoke`, `microbench`, `e2e`, and `quality`, with typed argv, cwd, environment,
unset list, timeout, and expected binary identity. Benchmark commands emit llama-bench JSON. The
quality adapter may normalize a supported evaluator schema into `QualityResult`; it must preserve the
raw evaluator output and bind both results to one protocol coordinate.

For runtime-configuration experiments, set `build.reuse_binary` and cite the exact prepared binary
hash. The baseline and every candidate must use that same hash. When the performance pre-gate fails,
request preservation/profiling of the negative experiment and skip expensive quality work. Only a
performance-eligible candidate proceeds to the full deterministic correctness and quality gate.
