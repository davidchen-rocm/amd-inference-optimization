# HyperLoom-inspired local control layer

This project remains a local `llama.cpp` optimization framework for the
configured AMD GPU. It does not depend on HyperLoom at runtime and does not yet
claim MI300X support.

The integration adds four local control-plane capabilities:

1. A stable action catalogue declaring expected gain, risk, GPU/workspace lanes,
   side effects, and required evidence.
2. Semantic candidate fingerprints so renaming an already-tested configuration
   cannot bypass experiment deduplication.
3. An accepted optimization stack assembled only from experimentally accepted
   and explicitly selected campaign candidates.
4. A continuously refreshed `reports/session-breakdown.{json,md}` projection for
   the CLI and read-only frontend.

Campaign creation also captures a read-only platform record containing CPU,
NUMA, governor, boost, host AMDGPU count, and the configured architecture
profile. ROCm runtime, kernel, telemetry, and validation evidence continues to
come from ROCm Issue Agent or the existing raw-profiler fallback.

## Evidence-provider layer

The framework records the information-collection boundary explicitly:

- `ROCm Issue Agent` is the direct, primary local GPU observer.
- raw `rocprofv3` is an explicitly labelled fallback, never disguised as MCP.
- Magpie benchmark reports and TraceLens compact roofline CSVs have hash-bound
  import adapters.
- IntelliKit tools (Kerncap, Metrix, Linex, Nexus, Accordo and uProf MCP) are
  catalogued separately from integration state. A detected installation does
  not mean the workflow executed it.
- unknown IntelliKit JSON is analysis-only until a tool-specific completeness
  adapter exists; it cannot enter an automatic Gate.
- a local runtime-health snapshot records disk, shared-memory, relevant process,
  dependency and path health without duplicating ROCm GPU telemetry.
- HyperLoom's Python Host Probe, privileged BIOS/BMC audit, multi-node probes and
  full robustness reactor remain visible as `AVAILABLE_NOT_INTEGRATED`; the UI
  must not present them as local capabilities.

Kernel evidence can be projected into a config-driven shape manifest. Its stable
variant signature includes kernel identity, graph variant, node ordinal, launch
geometry, dtype/quantization, M/N/K and resource usage. M/N/K and operator labels
must come from explicit attribution rules; unmatched or multiply-matched launches
remain `UNRESOLVED` or `AMBIGUOUS` rather than being inferred from grid size.

Useful commands:

```bash
gpuopt campaign actions
gpuopt campaign breakdown CAMPAIGN_ID --store .gpuopt
gpuopt evidence capabilities --task-id TASK_ID --store .gpuopt
gpuopt evidence runtime-health TASK_ID --store .gpuopt
gpuopt evidence import-magpie TASK_ID benchmark_report.json --store .gpuopt
gpuopt evidence import-tracelens TASK_ID roofline.csv \
  --architecture gfx942 --provenance-kind LIVE_MEASURED \
  --provenance-source measured-architecture.json --store .gpuopt
gpuopt evidence shape-manifest TASK_ID \
  --kernel-evidence kernel-evidence.json --rules operator-map.yaml --store .gpuopt
gpuopt ui --store local=.gpuopt --port 4561
```

`GPUTarget.gfx_target` now accepts canonical future `gfxNNNN` values, but the
active local optimization registry contains only `gfx1201`. Unknown targets are
reported as `family=unknown` and are not silently treated as supported.

MI300X/gfx942 runtime adapters, remote SSH, containers, multi-GPU execution, and
live validation are a future phase. The provider catalogue describes possible
evidence sources but does not claim they are supported locally.
