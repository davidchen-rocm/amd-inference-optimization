# Third-party notices

## AMD HyperLoom

Selected read-only host-platform probe logic in
`src/amd_inference_opt/platform_probe.py` is adapted from AMD HyperLoom:

- Project: <https://github.com/AMD-AGI/Hyperloom>
- Release: `v1.0.0b2`
- Original file: `src/hyperloom/common/platform_probe.py`
- Original file SHA-256: `7f2fa509b8ac7752e069d085f3997bf790c8e918af08d77632a88a7792c5b3ee`
- Copyright: 2026 Advanced Micro Devices, Inc.
- License: MIT

The local adaptation removes HyperLoom-specific provenance dependencies, emits
gpuopt Pydantic evidence, and does not include HyperLoom's orchestration agents,
TraceLens, Magpie, GEAK, BMC, multi-node, or remote-execution code.

HyperLoom's action catalogue, semantic candidate deduplication, optimization
stack, profile watermark, and session-breakdown concepts informed original
gpuopt implementations in `control_policy.py` and `session_breakdown.py`; those
files do not copy HyperLoom source text.

HyperLoom's provider layering, trace-shape manifest, host/runtime health and
evidence-lifecycle concepts also informed original gpuopt implementations in
`evidence_catalog.py`, `kernel_manifest.py`, `external_evidence.py`, and
`runtime_health.py`. These files define local schemas and adapters and do not
copy HyperLoom, Magpie, TraceLens, or IntelliKit source text.

The MIT license for HyperLoom is available at:
<https://github.com/AMD-AGI/Hyperloom/blob/v1.0.0b2/LICENSE>.
