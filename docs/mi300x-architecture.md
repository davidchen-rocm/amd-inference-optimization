# MI300X + vLLM architecture

Start with the [setup guide](vllm-mi300x.md) to configure a model, runtime, and GPU.
The CLI supports task creation, inspection, and next-action planning. It does not
currently compose the full GPU execution workflow automatically.

## Components

| Component | Responsibility |
|---|---|
| `store.py`, `models.py`, `gates.py` | Persist evidence, identities, and decisions shared by the backends. |
| `vllm_models.py`, `vllm_workflow.py` | Validate serving coordinates, workflow state, revisions, and approvals. |
| `vllm_adapter.py`, `vllm_ports.py` | Define execution interfaces for serving, benchmarks, quality checks, and profiling. |
| `mi300x_observation.py`, `vllm_runtime_inspector.py` | Inspect the device and runtime; incomplete observations remain unverified. |
| `process_group.py`, `profile_contract.py` | Manage native process sessions and negotiate profiler capabilities. |
| `cli_vllm.py`, `cli_common.py` | Provide vLLM commands and shared document, output, and error handling. |

The vLLM and llama.cpp backends share evidence contracts, but use different
workloads and metrics. Online serving throughput must not be treated as
single-request llama-bench decode performance.

## Execution limits

- `gpuopt vllm create` creates local task state. `status`, `next`, and `probe-plan`
  inspect state or generate plans. `gpuopt resume` dispatches vLLM tasks to
  next-action planning; it does not start a GPU workload.
- Runtime verification requires consistent process, executable, package, and
  container identities. An additional manifest cannot replace missing process
  observations. Failed preflight checks invalidate cached evidence.
- Native session cleanup does not manage a Docker daemon or guarantee cleanup of
  detached container workloads. A complete deployment needs an external
  supervisor, bounded timeouts, and endpoint, container, and GPU-process checks.
- Full execution composition, workload-window VF telemetry, quality-result
  integration, and automatic evidence collection still require implementation
  and validation. Backend interfaces alone do not establish those capabilities.
- The current workflow fixes dtype and quantization. A representation change
  needs its own policy, model identity, quality checks, and kernel evidence.

## Local evidence

Keep runtime manifests, raw logs, profiler traces, and experiment stores in local
ignored directories. Use your own model and environment coordinates; examples
are templates rather than measurements from your machine. See the
[publication guidelines](publication-privacy.md).
