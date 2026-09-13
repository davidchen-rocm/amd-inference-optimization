# Skill forward-test: old mapping to split-K

This fixture records a fresh, read-only Codex run against the local
`amd-inference-optimization-workflow` skill on 2026-08-16. The context intentionally exposes only
same-binary E2E results, Level-2 timing, capability availability, and runtime-control inspection.

The produced decision passed the behavior checks:

- all structured measured claims cite context evidence IDs that include the captured SHA-256;
- absent metadata, counters, occupancy, telemetry, and hardware behavior remain unknown;
- no Level-3 request is made because the capability evidence says it cannot produce the data;
- the failed old-mapping result remains cited, while the new hypothesis changes only the mapping;
- the split-K hypothesis explicitly reuses the exact binary and protocol hashes;
- no new gate verdict is announced; and
- full correctness and quality evaluation is conditional on passing the performance pre-gate.

An intentionally incomplete-schema dry run exposed a process-level failure: the fresh Agent
inspected the repository's recorded replay while looking for schema examples, even though the prompt
and skill limited evidence to the context. It did not copy replay-only measurements into the final
decision. The recorded decision above comes from the follow-up run with the complete framework-
generated schema; that run read only the skill, its direct reference, and the supplied context.

The live framework should always emit the complete `decision_schema`. The skill can be hardened
further by stating that missing schema details are a handoff error rather than permission to search
fixtures or reports.
