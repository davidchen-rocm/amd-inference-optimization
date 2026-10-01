# Measured ROCm hardware capability

The standalone probe records real BF16 matrix multiplication results without
loading a model or changing the training server. It is a GPU microbenchmark:
**its TFLOPs/s are not LLM token throughput or evidence of answer quality**.

Run it with the same Python environment and training account as the GPU worker,
using the **same** `MACFIT_GPU_LOCK` path as every other coordinated GPU process:

```sh
MACFIT_GPU_LOCK=/absolute/path/shared-gpu.lock \
  python -m macfit_training.hardware_probe \
  --output /absolute/path/fresh-hardware-probe.json --walltime-seconds 120
```

The output parent must already exist. A fresh output name is required; previous
evidence is never overwritten. The normal entrypoint starts an isolated child
and enforces a total wall-time budget of 1–180 seconds, including GPU-lock waiting,
runtime startup, allocations and measurements. On timeout or interruption it
kills the child process group and returns a nonzero exit without publishing a
success report. This bounds the user process; it cannot repair a GPU driver or
host that is itself unresponsive. Do not invoke the internal `--worker` mode
directly, because it relies on its supervising parent for this bound.

All accelerator imports occur inside the GPU child after the lease is acquired.
It requires an available ROCm GPU and uses device 0. The parent and `--help`
remain usable without PyTorch. The report contains GPU name/architecture, total
and initially free VRAM, PyTorch/ROCm/Python versions, source hash and seed;
it does not collect credentials, environment dumps or local file paths.

## Correctness and measurement protocol

Before timing, a seeded 64×64 check compares real GPU BF16 GEMM with CPU FP32
matmul of the **identical BF16-quantized inputs**. This separates input rounding
from the operation's numerical error. Both outputs must be finite. The relative
Frobenius/L2 error must be at most 0.01, and maximum absolute error divided by the
reference's maximum absolute value must be at most 0.02. These are explicit probe
tolerances, not a guarantee of accuracy for arbitrary matrices or model outputs.
The measured error norms and tolerance values are retained in the JSON.

The fixed GEMM sizes are 512, 2048, 4096 and 8192. Each uses two untimed warmups
and five measured multiplications of the same random BF16 inputs, with a
preallocated output. Every final large output is checked for finite values, but
these large matrices are not compared elementwise with an FP32 reference.

Each sample records two different intervals:

- **Device time:** elapsed GPU time between PyTorch CUDA events. ROCm uses this
  PyTorch device API as well.
- **Synchronized wall time:** host time surrounding event recording, GEMM
  launch and device synchronization. It includes launch/synchronization overhead.

Input construction, allocation and warmup are outside both measured intervals.
The output includes raw samples, minimum/maximum/median milliseconds and median
TFLOPs/s for each interval. The FLOP convention is `2 * n^3`; the conversion is
`2 * n^3 / (milliseconds * 1e9)`. Zero, negative and nonfinite timing values are
rejected. Device and wall measurements must be reported separately, rather than
substituting device time for application latency.

The parent validates the complete report and publishes a fully written JSON file
atomically. Evidence must remain **strictly smaller than 1 MiB** and must not
contain nonfinite JSON values. Partial files in its temporary directory are
removed on failure. Store the verified result alongside model benchmark,
training, runtime and backup evidence when preserving a temporary GPU campaign.

GPU clocks, virtualization, other programs that ignore the advisory lease,
temperature and GEMM shape can all affect results. Warmed dense GEMM is a narrow
compute measurement. Measure actual model prefill, decode, end-to-end latency,
peak memory and task quality independently before making deployment decisions.

The MI300X product specification lists 192 GB HBM3 and up to 5.3 TB/s memory
bandwidth; these are vendor hardware specifications, not measured results from
this probe. See the official [MI300X specification](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html),
[PyTorch event API](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.Event.html)
and [PyTorch synchronization API](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.synchronize.html).
