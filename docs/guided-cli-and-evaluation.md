# Guided CLI and evaluation suites

The guided CLI keeps machine-specific paths in one project file and resolves a
small request into the existing, fully validated `OptimizationTask`. It does not
bypass workflow states, evidence requirements, or ROCm approval.

## One-time setup

```bash
gpuopt config init \
  --project /path/to/amd-inference-optimization \
  --model-root /path/to/model-library \
  --runtime-repo /path/to/llama.cpp \
  --runtime-build /path/to/llama-build \
  --mcp-command /absolute/path/to/rocm-agent-mcp

gpuopt config doctor --project /path/to/amd-inference-optimization
```

Configuration is stored atomically in `.gpuopt/config.yaml`. Supported updates
are deliberately limited; for example:

```bash
gpuopt config set gpu-device 0 --project /path/to/project
gpuopt config set rocm-mcp-command /absolute/path/to/rocm-agent-mcp --project /path/to/project
```

The model library uses these roles:

```text
model-library/origin/<model>/...
model-library/derived/<model>/<quantization>/*.gguf
```

Scan and inspect it with:

```bash
gpuopt model scan --project /path/to/project
gpuopt model list --project /path/to/project
gpuopt model show origin/Qwen3.5-0.8B --project /path/to/project
```

Directory-name matching is reported as `INFERRED`, not verified provenance. An
explicit `model link` is `DECLARED`; only matching source and output hashes in a
preparation manifest produce `VERIFIED`.

## Frozen quality data

Install preparation-only dependencies and download the general suite once:

```bash
pip install -e '.[eval-prep]'
gpuopt eval prepare general-100.v1 --project /path/to/project
gpuopt eval list --project /path/to/project
gpuopt eval show general-100.v1 --project /path/to/project
```

The optional full English final-validation library is prepared separately:

```bash
gpuopt eval prepare english-full.v1 --project /path/to/project
gpuopt eval show english-full.v1 --project /path/to/project
```

It contains complete pinned splits for knowledge, commonsense, mathematics,
instruction following, code, and long-context tasks. Chinese LongBench tasks
are excluded. Each source keeps its own scorer coordinate; preparing the data
does not claim that every scorer was executed.

Preparation deterministically freezes 25 cases each from MMLU general,
ARC-Challenge, HellaSwag, and WinoGrande. Experiment execution reads the frozen
manifest and JSONL fixture without network access and rejects changed bytes.

## Resolve, create, and resume

Start with a dry run. This validates model, runtime, MCP, quality-suite and
benchmark coordinates but creates no task:

```bash
gpuopt optimize \
  --model origin/Qwen3.5-0.8B \
  --baseline Q6_K \
  --quality math-100.v1 \
  --quality general-100.v1 \
  --project /path/to/project \
  --dry-run
```

The JSON response contains exactly one `next_action`. For a dry run it is a
shell-safe `CREATE_TASK` command using `--no-execute`. Running that command
creates the task, freezes model-input provenance and returns one `RESUME`
action. `gpuopt resume TASK_ID --store STORE` then advances to the next evidence,
Agent decision, exact MCP approval, or terminal result.

For scripts, always pass the required parameters explicitly. Missing `--model`
or an origin model without `--baseline` fails instead of prompting when stdin is
not interactive.
