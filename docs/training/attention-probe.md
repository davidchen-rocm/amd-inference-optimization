# Controlled ROCm attention preferences

This separate operator probe compares `aotriton` and `ck` Flash Attention library
preferences with one fixed BF16 model and identical synthetic inference shapes.
It uses the installed runtime and opens no HTTP service. Both arms run in isolated
children under the existing shared GPU lock, process deadline and absolute cutoff.

```sh
MACFIT_GPU_LOCK=/srv/macfit-training/data/gpu.lock \
MACFIT_MODEL_CACHE=/srv/macfit-training/cache \
HF_HOME=/srv/macfit-training/cache/huggingface \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/srv/macfit-training/venv/bin/python -m macfit_training.attention_probe \
  --model qwen3-8b --batch-sizes 1,4 --prompt-tokens 2048,8192 \
  --walltime-seconds 1800 --model-timeout-seconds 900 \
  --stop-at 2026-09-30T23:30:00-04:00 \
  --output /srv/macfit-training/data/evidence/gpu-capability-attention.json
```

Only Qwen3 8B or 14B, batches 1/4 and contexts 2,048/8,192 are permitted. Decode
is fixed at 64, seed at 42, warmups at two and scored repetitions at five. Supply
your actual cache/lease paths and a fresh output filename. The first arm's lock
wait and execution consume the shared campaign budget; the second receives only
the remaining time. Add `--validate-only` to inspect the protocol without Torch.

The child records requested and returned library preference, forces
`sdpa_kernel(FLASH_ATTENTION)`, and lets an unsupported native operation fail
rather than substitute math attention. The APIs are documented by PyTorch's
[ROCm library preference](https://docs.pytorch.org/docs/2.10/backends.html#torch.backends.cuda.preferred_rocm_fa_library)
and [SDPA context](https://docs.pytorch.org/docs/2.10/generated/torch.nn.attention.sdpa_kernel.html).
A compiled flash implementation does not require the Python `triton` package;
this probe does not assume that either preference is available in a given wheel.

Before both warmups, one additional profiling pass measures only prefill plus
one cached decode step at the first coordinate. It does not enter scored timing
samples. Profiling disables shapes, memory and stacks, bounds aggregate events
and names, and retains at most 128 displayed device-kernel/operator names per
group. SHA-256 fingerprints cover all full recorded names, including names omitted
or shortened for display. Raw traces and local paths are not exported. The
[PyTorch profiler API](https://docs.pytorch.org/docs/2.10/profiler.html) supplies
the CPU operator and GPU device events.

If profiling is unavailable, its status says unsupported or unverified; successful
preference timing can still be retained. An actual model-forward failure is never
swallowed as a profiler limitation. Native-library failures are marked unsupported,
OOM and timeout remain incomplete, and no packages are installed to repair an arm.

Reports retain the label **preference arms**. A different kernel-name fingerprint
is an observation, not proof that two particular attention libraries executed:
library preferences can permit fallback. The comparison never automatically claims
a distinct backend or faster production serving. It reports measured throughput,
raw-sample variation and whether greedy output token-hash sets matched; this does
not establish semantic answer quality.

One comparison produces five JSON evidence files: its summary, two arm indexes
and two complete model reports. All are strictly smaller than 1 MiB. The operator
checks the current 64-file backup inventory bound and reserves one slot for
external runtime evidence. The existing verified backup exporter can preserve
these JSON files; check its actual manifest and receipt after execution.
