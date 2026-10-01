# Research during a temporary GPU window

Measure model speed, memory, task correctness and capability changes separately.
The tools below use the installed ROCm environment without opening another
network service or changing the website's reviewed model registry.

The [30 September MI300X results](mi300x-results.md) summarize the measured
inference, synthetic policy diagnostics and actual CPU restore/merge check.

| Question | Tool and protocol |
| --- | --- |
| Does native BF16 computation work? | [Hardware probe](gpu-capabilities.md): numerical reference check and timed GEMM. |
| How do size, batch and context affect inference? | [Inference baseline](gpu-benchmark.md): pinned 0.6B–32B, repeated prefill/decode timings and memory. |
| Can a larger model handle a longer prompt? | [Long-context plan](gpu-long-context.md): separate 14B/32B measurements with bounded contexts. |
| Can the GPU perform larger LoRA updates? | [Training capacity probe](gpu-training-capacity.md): five actual rank-8 LoRA updates on 14B/32B, without a quality or adapter-publication claim. |
| Does an installed attention preference change performance? | [Attention probe](attention-probe.md): isolated preference arms, recorded native failures and profiling status. |
| Does LoRA improve the requested task? | [Policy experiment](../../examples/training/experiments/README.md): actual base/adapter outputs on identical held-out questions. |
| Can targeted examples repair observed failures? | [Adaptive robustness diagnostic](../../examples/training/robustness/README.md): a separately bound 360-row dataset and explicit evaluation-reuse disclosure. |
| Does task training change unrelated behavior? | The independent ten-question regression CLI in the policy experiment guide. |
| Can a saved adapter still work after the GPU disappears? | [CPU export portability check](export-portability.md): verified restore inputs, CPU merge/reload and six actual output comparisons. |

Use the same training account, Python environment, `MACFIT_GPU_LOCK`, model
cache and `HF_HOME` as the service. The advisory lock serializes coordinated
workloads; it does not stop unrelated programs. Cached checkpoints avoid repeat
downloads, but pinned snapshot hashes must still be verified. Model caches are
re-downloadable and are deliberately excluded from private job backups.

The local campaign coordinator submits the six policy recipes through the
existing supervised queue. It requires operator filesystem access; there is no
public route for this administrative action. Website submissions continue to
require authenticated accounts. Choose a stable campaign ID to recover the same
job identities after a coordinator restart:

```sh
python -m macfit_training.campaign \
  --data-dir /absolute/path/training-data \
  --campaign-id my-policy-study \
  --stop-at 2026-09-30T23:45:00-04:00
```

Set the timestamp to your actual GPU cutoff. Queue admission retains normal
active-job and deadline limits; it may stop accepting work well before the
cutoff to leave the full permitted job runtime. The coordinator keeps a separate
host lease so two instances cannot create competing research sequences. It
records verified artifact hashes and actual scores under `data/evidence`, and
requests cancellation of its own active job if coordination fails or its time
budget closes. The service supervisor enforces the GPU job process deadline.

To run the adaptive dataset instead, add `--variant robustness` and choose a new
campaign ID. This selects three standard runs with its own dataset builder and
scorer. A campaign cannot resume across dataset variants. The adaptive dataset
also changes the system instructions, so compare each run's base and adapter
under its own identical prompt; differences between variants do not isolate
training-data effects.

Read `status`, terminal job results and the actual transcripts before calling a
run successful. A lower loss can accompany worse answers. The policy fixture is
synthetic, uses one seed and shares reference facts between train and evaluation;
its scores are limited diagnostics, not a general accuracy or production-quality
claim. Inference throughput is aggregate compute throughput for the stated
batch/context, not an interactive serving guarantee. GEMM TFLOPs/s are a separate
microbenchmark and must not be substituted for model throughput.

Keep JSON evidence below the backup's per-file bound and check the combined
evidence-file count before adding campaigns. Back up registered datasets,
adapters, results, evidence and source while the GPU host is available. A backup
is established only after a successful verified export/restore receipt; a timer
or uploaded archive alone is insufficient. Reserve time after all GPU cutoffs
for the final backup and an independent archive read.
