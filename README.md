# AMD Inference Optimization

Test LLM performance on AMD GPUs, compare optimizations, and review results.

- **RX 9070 XT + llama.cpp**: local model optimization and testing.
- **MI300X + vLLM**: task setup, status, and next steps. Automatic execution is still in development.

## Install

Requires Python 3.11 or later. Run from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Running models also requires ROCm and either llama.cpp or vLLM.

## Try a demo

No GPU or model download needed:

```bash
gpuopt replay examples/q4-rdna-recorded.yaml
```

This walks through a saved optimization example.

## Use a local model

Prepare a ROCm-enabled llama.cpp build, a GGUF model, and ROCm Issue Agent.
Replace the paths below with your own:

```bash
gpuopt config init \
  --model-root /path/to/model-library \
  --runtime-repo /path/to/llama.cpp \
  --runtime-build /path/to/llama.cpp/build \
  --mcp-command /path/to/rocm-agent-mcp

gpuopt config doctor
gpuopt model scan
gpuopt model list
```

Place GGUF files under `model-library/derived/MODEL/QUANTIZATION/model.gguf`.
Download the test questions once, then follow the prompts to select a model:

```bash
pip install -e '.[eval-prep]'
gpuopt eval prepare general-100.v1
gpuopt optimize
```

When the task needs your input, follow the printed `next_action`, then continue:

```bash
gpuopt resume TASK_ID --store .gpuopt/store
```

Replace `TASK_ID` with the ID shown when the task was created. Adjust `--store` if you use a custom folder.

## Use MI300X

Follow the [MI300X quick start](docs/vllm-mi300x.md) to configure your model and runtime.

```bash
gpuopt vllm --help
```

## View results in your browser

```bash
gpuopt ui --store main=.gpuopt/store --port 4561
```

Open the [local dashboard](http://127.0.0.1:4561) to browse tasks, models, and results.

More: [Local setup](docs/guided-cli-and-evaluation.md) ·
[MI300X architecture](docs/mi300x-architecture.md) ·
[Examples](examples/) · [Third-party notices](THIRD_PARTY_NOTICES.md)
