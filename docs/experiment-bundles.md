# Experiment Bundles

An Experiment Bundle gives every experiment one stable entry point without moving or
duplicating its original evidence. The original `spec`, runner output, benchmarks,
quality results, profiler envelopes, approvals, and logs remain authoritative.

```text
<store>/<task>/
├── artifacts/baseline.json
├── experiments/<experiment>/
│   ├── summary.json          # performance, quality, change, and decision
│   ├── manifest.json         # hash-bound bundle commit marker
│   ├── attempts/index.json   # selected runner/profile/E2E/quality attempts
│   └── ... existing evidence
└── workspaces/<experiment>/  # new source-patch worktrees, outside the bundle

<store>/catalog.sqlite3       # rebuildable UI/CLI query cache
```

`summary.json` is the normal human-facing file. It contains model/runtime identity,
the hypothesis and change, baseline/candidate metric means and CV, quality deltas,
the Gate decision, and selected attempts. `manifest.json` is written last and binds
the summary plus all registered evidence by SHA-256 and size. It deliberately excludes
itself to avoid a circular hash.

The task-level `artifacts/manifest.json` remains the preview and integrity authority.
The bundle manifest cannot make an unregistered file trusted. ROCm Issue Agent raw
case artifacts remain owned by its case store; the Bundle references the registered
MCP envelope and ownership metadata instead of copying traces.

## Commands

Backfill or refresh all experiments in one task:

```bash
gpuopt bundle rebuild TASK_ID --store STORE
```

Rebuild a single experiment:

```bash
gpuopt bundle rebuild TASK_ID --experiment EXPERIMENT_ID --store STORE
```

Read the canonical result or query the derived catalog:

```bash
gpuopt bundle show TASK_ID EXPERIMENT_ID --store STORE
gpuopt bundle list --task TASK_ID --store STORE
```

`run-live --execute` refreshes Bundles after each pause or completed experiment. A
Bundle failure is reported as a maintenance warning and never changes an already
persisted performance or quality verdict.

## Collection and retention

The collector only registers regular, non-symlink files from bounded runner, quality,
E2E-rerun, and extended-verification directories. It does not copy data and explicitly
excludes source worktrees and build directories. New source-patch worktrees live in
`workspaces/<experiment>` so experiment result directories remain small and readable.

The SQLite catalog is only a cache. Delete or rebuild it from verified JSON Bundles at
any time; Gate decisions and evidence never depend on it.
