# Q8_0 model preparation

The Qwen3-8B Hugging Face checkpoint contains 15.26 GiB of BF16 tensor data. It does
not leave enough space for a stable, fully offloaded run on the 16 GB RX 9070 XT. The
Q8 campaign therefore converts that checkpoint once and treats the resulting GGUF as
an immutable experiment input.

The pinned converter dry-run reports 399 tensors and an estimated 8.7 GB output. This
leaves space for the KV cache and runtime buffers in the single-request decode workload.

## Validate without writing

Run the preparation tool from this repository:

```bash
python tools/prepare_q8_model.py \
  --model-dir /workspace/math-rule-loop/models/qwen3-8b-hf \
  --llama-repo /workspace/math-rule-loop/tools/llama.cpp-source \
  --python /workspace/math-rule-loop/.venv-lighteval/bin/python \
  --output /workspace/math-rule-loop/models/qwen3-8b-q8/Qwen3-8B-Q8_0.gguf \
  --dry-run
```

Dry-run hashes every converter input, verifies that the llama.cpp checkout has no
tracked changes, records the converter commit and script hash, and calls the real
converter with `--dry-run`. It does not create the output directory, GGUF, manifest,
partial file, or lock file.

## Create and freeze Q8_0

Remove only the final `--dry-run` flag from the command above. Preparation writes to a
fixed `.partial` path and publishes the GGUF only after conversion succeeds and its
SHA-256 is calculated. It then writes:

```text
/workspace/math-rule-loop/models/qwen3-8b-q8/Qwen3-8B-Q8_0.gguf
/workspace/math-rule-loop/models/qwen3-8b-q8/Qwen3-8B-Q8_0.gguf.preparation.json
```

The manifest contains the sorted input file manifest, input manifest hash, converter
commit and hash, exact argv, output size, and output hash. Repeating the same command
reuses the result only when all of these coordinates and the current output hash match.
The tool never overwrites a mismatched output, manifest, or stale partial file.

If conversion completed before the preparation tool could write its manifest, use the
same command with `--adopt-existing`. Adoption does not convert or overwrite the GGUF.
It verifies the current input manifest and converter identity, runs the converter's
dry-run, checks the GGUF version, architecture, tensor count and Q8_0 file type, then
hashes the stable file and writes only the missing manifest. Adoption records that the
original conversion argv is unknown instead of claiming false provenance.

The current example pins `model.sha256` to the adopted local output. If the model is
ever regenerated, update that field from `output.sha256` in the new preparation
manifest before running the task. The GGUF hash must remain fixed for baseline and
every candidate experiment.

The benchmark must select the discrete GPU explicitly with `-dev ROCm0 -mg 0 -ngl
999`. Preserve llama-bench's built-in warmup. A run that does not offload every model
layer to gfx1201 is not a valid baseline.
