# Local control-plane UI V1

Start the local UI on port 4561:

```bash
gpuopt ui \
  --store q8=/workspace/amd-inference-optimization/.gpuopt-q8-live \
  --store q4=/workspace/amd-inference-optimization/.gpuopt-live \
  --draft-db /workspace/amd-inference-optimization/.gpuopt-ui/control-plane.sqlite3 \
  --port 4561
```

Open `http://127.0.0.1:4561`. The server accepts only loopback hosts and performs no
CORS opt-in. Experiment Stores, evidence, approvals, and execution remain read-only.
Text previews are limited to 512 KiB and require a matching size and SHA-256;
symlinks and path traversal are rejected.

The only write endpoint is `POST /api/v1/drafts`. It requires the session CSRF token,
`application/json`, a body no larger than 128 KiB, and the strict
`OptimizationDraftRequestV1` schema. It writes to the separate SQLite draft database,
not an Experiment Store. All other POST/PUT/PATCH/DELETE/OPTIONS routes are rejected.
A draft is immutable and cannot execute a command.

The stable browser contract is `gpuopt.control-plane.v1`, owned by
`frontend_api.py`. The UI does not import internal workflow models directly.

The main screens are:

1. dashboard and run list;
2. model optimization catalogue;
3. optimization experiment builder;
4. immutable draft list/detail;
5. run/campaign detail;
6. candidate or experiment record;
7. verified artifact preview;
8. optimization workflow map.

The model catalogue is a derived database view over existing stores. It groups a
family by its recorded HF/source identity and renders a strict lineage:

```text
ORIGIN model (usually BF16/HF source)
  ├─ BASELINE variant
  ├─ CANDIDATE quantization or mixed-bit variant
  └─ ACCEPTED variant
```

For example, `Qwen/Qwen3.5-9B` is the Origin; Q8 is its benchmark baseline and
Q6/Q5/Q6-Q8 mixed are derived variants. Each variant retains its source run and
evidence links. The catalogue does not replace persisted evidence.

The model detail page gives priority to two results: tg128/tg512 tokens/s and the
recorded math accuracy. Perplexity, gate reasons, run internals, kernel evidence, and
artifacts are placed under **Validation & evidence** or **Runs and optimization
methods** disclosures.

The builder currently controls:

- stock Q8/Q6/Q5/Q4 reference arms;
- config-driven tensor-group precision assignments;
- sensitivity-guided mixed-bit planning;
- evidence-first or unconditional shape-kernel experiments;
- bounded split-K, wave, vector-load, VGPR, and fusion knobs;
- matrix-shape execution mapping that may patch llama.cpp `mmvq.cu`;
- optional weight-layout/fused-dequant, gate/up fusion, HIP Graph A/B, KV-cache,
  and buffer-reuse evidence workflows;
- tg128/tg512 repetitions and CV budget;
- the temporary 100-question/PPL/greedy quality policy.

The recommended default keeps embedding, output, and attention tensors at Q6, uses
Q5 for FFN gate/up/down, and creates a kernel experiment only after profiling proves
that the new weight format has a mapping gap.

There are thirteen small view components: connection status, summary cards, run table,
capability matrix, stage pipeline, candidate/experiment table, metrics/quality panel,
event timeline, artifact browser, record inspector, map legend, connected capability
board, and component inspector.

The workflow map is opened from **Open optimization map** on any run. Its topology is
configuration-driven and currently contains 34 capability components across five
layers: Model, Kernel, Runtime, Evidence, and Validation. Every node deliberately
shows two independent states:

- **Framework** availability: implemented, partial, planned, or not implemented.
- **This run** status: not started, running, complete, accepted, rejected,
  inconclusive, or unavailable.

Clicking a node shows its inputs, outputs, linked experiments, and registered
evidence. Lines describe dependencies and evidence flow; their color reflects the
current run without changing any workflow state.

For orientation, the backend has ten major components: task/campaign coordinator,
mixed-precision workflow, HIP Graph A/B, memory audit, MFMA/MMQ closure, ROCm MCP
adapter, experiment runner, evidence store/gates/reporting, and the read-only control
plane, plus the isolated draft database. The browser is intentionally not an
execution or approval surface.

Stable endpoints include:

```text
GET  /api/v1/catalog/models
GET  /api/v1/builder/meta
GET  /api/v1/builder/options
GET  /api/v1/builder/schema
GET  /api/v1/drafts
GET  /api/v1/drafts/{id}
POST /api/v1/drafts
```
