# Imatrix-guided mixed-bit model preparation

`tools/prepare_mixed_bit_models.py` produces an importance matrix and independent
Q5_K_M and Q4_K_M GGUFs from one frozen BF16 GGUF. Both quantizations read the BF16
source directly; the tool never passes llama.cpp's `--allow-requantize` option and
rejects a source whose `general.file_type` or tensor histogram is already quantized.

First inspect the complete plan without creating an output directory or running
llama.cpp:

```bash
python tools/prepare_mixed_bit_models.py \
  --bf16-model /path/to/Qwen3-8B-BF16.gguf \
  --calibration fixtures/q8-runtime-quality/math500-ppl.txt \
  --llama-bin-dir /path/to/llama.cpp/build/bin \
  --output-dir /path/to/qwen3-8b-mixed-k \
  --dry-run
```

Remove `--dry-run` to run `llama-imatrix`, then generate Q5_K_M and Q4_K_M. Commands
write fixed `.partial` files under an exclusive preparation lock. The tool validates
the imatrix identity and both output GGUFs before atomically publishing each artifact;
the manifest is published last. Existing outputs are never overwritten.

The preparation manifest records SHA-256 values for the BF16 source, calibration
corpus, both llama.cpp executables, the imatrix, and both models. It also records exact
argv/cwd coordinates, command-output hashes, GGUF identities, and tensor-type
histograms. A repeated invocation reuses the preparation only when the current inputs,
tools, command plan, coordinates, paths, and artifact hashes still match.

If all three artifacts exist from an earlier run but the manifest is missing, add
`--adopt-existing`. Adoption validates and hashes the stable files and writes only the
manifest. It explicitly records that the original commands are unknown.

For the bounded mixed-bit quality comparison, add `--math-limit 100` to
`tools/q8_runtime_quality_eval.py`. The option selects 12 abstract-algebra, 12 college,
32 high-school, and 44 elementary questions. It ranks each subject by
`sha256(str(20260815) + NUL + item_id)`, then globally orders the selected questions by
the same rank. The result protocol records the seed, quotas, source and selected
counts, selection algorithm, and ordered selected-ID hash. Omitting the option retains
the full 848-question fixture.
