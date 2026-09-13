# MI300X + vLLM guide

Use this guide to configure a task for one MI300X and check its status and next step.
The CLI does not yet start vLLM or run the full GPU test workflow automatically.

## Before you start

Follow the [README](../README.md) to install this project. You also need ROCm, a compatible vLLM installation, and a local model.
Run the commands below from the project directory.

## Create a task

Copy the example configuration:

```bash
mkdir -p .gpuopt/configs
cp examples/vllm-mi300x-task.yaml .gpuopt/configs/mi300x.yaml
```

Set your model directory, Python path, software versions, and GPU details.
Replace all placeholders. Expand the first-time setup section below for details.

Check the configuration, then create the task:

```bash
gpuopt vllm probe-plan --config .gpuopt/configs/mi300x.yaml
gpuopt vllm create --config .gpuopt/configs/mi300x.yaml --store .gpuopt/store
```

Run the create command only after the check returns `accepted: true`.

## Check a task

Replace `TASK_ID` with the ID shown when you created the task:

```bash
gpuopt vllm status TASK_ID --store .gpuopt/store
gpuopt vllm next TASK_ID --store .gpuopt/store
```

`status` shows the current state; `next` shows what to do next. For these tasks, `gpuopt resume` also only shows the next step.

<details>
<summary>First-time setup details</summary>

Edit `.gpuopt/configs/mi300x.yaml`. The template needs these changes before use:

- `task.runtime`: set the actual Python path, software versions, and file checksums. Remove both `image` and `image_digest` if you are not using a container.
- `model` and `task.model`: set the model name, local directory, and commit.
- `device`: set your GPU details. Use the same HIP UUID for every `ROCR_VISIBLE_DEVICES` value.
- `serving`, `task.workload`, and `task.benchmark`: keep the model, paths, input/output lengths, and concurrency settings consistent.
- `quality_protocol`: provide your own quality test script; the example script is not included. Keep `profile.mode: unavailable` for the initial setup.

`.gpuopt/` is ignored by Git and can hold local configuration and task data.

### Generate model and environment files

Replace the model path and both commit placeholders below. Save the generated JSON outside the model directory:

```bash
python tools/capture_vllm_model_snapshot.py \
  --model-dir /path/to/models/Qwen3-8B \
  --model-id Qwen/Qwen3-8B \
  --revision REPLACE_WITH_MODEL_COMMIT \
  --tokenizer-revision REPLACE_WITH_TOKENIZER_COMMIT \
  --output "$PWD/.gpuopt/configs/model-snapshot.json"
```

Set `model.snapshot_digest` to the output value and `model.snapshot_manifest_path` to the JSON file's absolute path.

Install this project in the Python environment that runs vLLM, then use that interpreter to generate the environment file. Replace the Python path below:

```bash
/path/to/vllm/python -I tools/capture_vllm_environment.py \
  --output "$PWD/.gpuopt/configs/vllm-environment.json"
```

Set `task.runtime.environment_manifest_sha256` to the output value. Replace the other example checksums with those of your actual files.

After editing `serving`, run this code and copy the output into `task.benchmark.protocol_hash`:

```bash
python - <<'PYTHON'
from pathlib import Path
import yaml
from amd_inference_opt.vllm_models import VLLMServingProtocol
config = yaml.safe_load(Path(".gpuopt/configs/mi300x.yaml").read_text())
print(VLLMServingProtocol.model_validate(config["serving"]).coordinate_sha256)
PYTHON
```

The `gpuopt vllm probe-plan` command above only generates a check plan. If it reports a checksum mismatch, copy each command's `request_sha256` from its output into the matching field: `device.amd_smi_command_sha256` or `device.hip_probe_command_sha256`. Run `probe-plan` again until it returns `accepted: true`.

</details>

For implementation details and execution limits, see [MI300X architecture](mi300x-architecture.md).
