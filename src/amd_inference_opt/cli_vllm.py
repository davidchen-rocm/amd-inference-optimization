"""Non-executing MI300X/vLLM CLI commands and probe planning."""

# Typer intentionally declares CLI metadata through calls in parameter defaults.
# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from .cli_common import _default_store, _document, _fail, _json
from .store import ExperimentStore, StoreError

vllm_app = typer.Typer(
    no_args_is_help=True,
    help="Manage the non-executing vLLM + MI300X control plane.",
)

_VLLM_PROBE_TIMEOUT_SECONDS = 60.0


def _vllm_execution_boundary() -> dict[str, Any]:
    """Describe the deliberate boundary of the minimal vLLM CLI."""

    return {
        "mode": "control_plane_only",
        "gpu_execution_started": False,
        "next_action_executed": False,
        "can_execute_next_action": False,
        "reason": (
            "this CLI does not wire paid-GPU execution; use a reviewed production "
            "composition with a verified runtime inspector, a concrete external "
            "quality implementation, and an independent watchdog"
        ),
        "required_external_components": [
            "verified vLLM runtime inspector",
            "concrete MI300X inspection and telemetry composition",
            "concrete baseline/candidate quality command",
            "independent paid-GPU watchdog and cleanup supervisor",
        ],
    }


def _vllm_action_payload(coordinator: Any, record: Any) -> dict[str, Any]:
    return {
        "schema": "gpuopt.vllm-control-plane.v1",
        "task_id": record.task_id,
        "campaign_kind": "vllm_mi300x",
        "config_sha256": record.config_sha256,
        "record": record,
        "next_action": coordinator.next_action(record),
        "execution": _vllm_execution_boundary(),
    }


def _vllm_probe_plan(configured: Any, *, source: dict[str, str]) -> dict[str, Any]:
    """Build the exact read-only MI300X inspection requests without running them."""

    # The helper binds the probe implementation digest as well as argv/cwd/env.
    # This is the same public function used by the concrete inspection port.
    from .mi300x_observation import (
        observation_environment,
        observation_request_sha256,
        observation_unset_environment,
    )

    runtime = configured.task.runtime
    executable = runtime.executable
    device_uuid = configured.device.device_uuid
    cwd = configured.serving.cwd.resolve()
    environment = observation_environment(device_uuid)
    unset_environment = observation_unset_environment()
    commands = (
        (
            "amd_smi_static",
            [
                executable,
                "-I",
                "-m",
                "amd_inference_opt.mi300x_observation",
                "--mode",
                "static",
                "--uuid",
                device_uuid,
            ],
            configured.device.amd_smi_command_sha256,
        ),
        (
            "hip_logical_device",
            [
                executable,
                "-I",
                "-m",
                "amd_inference_opt.mi300x_observation",
                "--mode",
                "hip",
            ],
            configured.device.hip_probe_command_sha256,
        ),
    )
    planned_commands = []
    for name, argv, expected_hash in commands:
        request_hash = observation_request_sha256(
            argv,
            cwd=cwd,
            device_uuid=device_uuid,
            timeout_seconds=_VLLM_PROBE_TIMEOUT_SECONDS,
        )
        planned_commands.append(
            {
                "name": name,
                "argv": argv,
                "request_sha256": request_hash,
                "configured_request_sha256": expected_hash,
                "hash_matches": request_hash == expected_hash,
            }
        )
    hashes_match = all(item["hash_matches"] for item in planned_commands)
    return {
        "schema": "gpuopt.vllm-mi300x-probe-plan.v1",
        "task_id": configured.task.id,
        "source": source,
        "accepted": hashes_match,
        "execution_started": False,
        "cwd": str(cwd),
        "timeout_seconds": _VLLM_PROBE_TIMEOUT_SECONDS,
        "environment": {
            "set": environment,
            "unset": list(unset_environment),
        },
        "commands": planned_commands,
        "preconditions": [
            {
                "id": "rocm_minimum",
                "requirement": "ROCm>=6.4",
                "configured_value": runtime.rocm_version,
                "observed_by_plan": False,
            },
            {
                "id": "deployment_observability",
                "requirement": (
                    "bare-metal AMD SMI/HIP visibility"
                    if runtime.image_digest is None
                    else "same-container RepoDigest and cgroup-v2 process observability"
                ),
                "observed_by_plan": False,
            },
            {
                "id": "rocr_uuid_selector",
                "requirement": (
                    "ROCR_VISIBLE_DEVICES must use the AMD SMI enumeration HIP UUID"
                ),
                "configured_value": device_uuid,
                "observed_by_plan": False,
            },
        ],
    }


@vllm_app.command("create")
def vllm_create(
    config: Path = typer.Option(
        ..., "--config", exists=True, dir_okay=False, readable=True
    ),
    store_root: Path = typer.Option(
        _default_store(), "--store", help="Experiment store root."
    ),
) -> None:
    """Create an immutable vLLM control-plane task without executing a GPU."""

    try:
        from .vllm_models import VLLMCampaignConfig
        from .vllm_workflow import VLLMWorkflowCoordinator

        configured = VLLMCampaignConfig.model_validate(_document(config))
        coordinator = VLLMWorkflowCoordinator(ExperimentStore(store_root))
        record = coordinator.create(configured)
        payload = _vllm_action_payload(coordinator, record)
        payload["created"] = True
    except (StoreError, ValidationError, ValueError, RuntimeError, OSError) as exc:
        _fail(str(exc))
    typer.echo(_json(payload))


@vllm_app.command("status")
def vllm_status(
    task_id: str,
    store_root: Path = typer.Option(
        _default_store(), "--store", help="Experiment store root."
    ),
) -> None:
    """Verify and print persisted vLLM state; never run its next action."""

    try:
        from .vllm_workflow import VLLMWorkflowCoordinator

        coordinator = VLLMWorkflowCoordinator(ExperimentStore(store_root))
        payload = _vllm_action_payload(coordinator, coordinator.load(task_id))
    except (StoreError, ValidationError, ValueError, RuntimeError, OSError) as exc:
        _fail(str(exc))
    typer.echo(_json(payload))


@vllm_app.command("next")
def vllm_next(
    task_id: str,
    store_root: Path = typer.Option(
        _default_store(), "--store", help="Experiment store root."
    ),
) -> None:
    """Print one durable next action and explicitly refuse GPU execution."""

    try:
        from .vllm_workflow import VLLMWorkflowCoordinator

        coordinator = VLLMWorkflowCoordinator(ExperimentStore(store_root))
        payload = _vllm_action_payload(coordinator, coordinator.load(task_id))
        payload.pop("record")
    except (StoreError, ValidationError, ValueError, RuntimeError, OSError) as exc:
        _fail(str(exc))
    typer.echo(_json(payload))


@vllm_app.command("probe-plan")
def vllm_probe_plan(
    task_id: str | None = typer.Argument(
        None, help="Persisted vLLM task; mutually exclusive with --config."
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Unpersisted config to hash before vllm create.",
    ),
    store_root: Path = typer.Option(
        _default_store(), "--store", help="Experiment store root."
    ),
) -> None:
    """Plan hash-bound MI300X probes without importing AMD SMI or touching a GPU."""

    try:
        from .vllm_models import VLLMCampaignConfig
        from .vllm_workflow import VLLMWorkflowCoordinator

        if (task_id is None) == (config is None):
            raise ValueError("provide exactly one of TASK_ID or --config")
        if config is not None:
            configured = VLLMCampaignConfig.model_validate(_document(config))
            source = {"kind": "config", "path": str(config.resolve())}
        else:
            assert task_id is not None
            coordinator = VLLMWorkflowCoordinator(ExperimentStore(store_root))
            configured = coordinator.load_config(task_id)
            source = {
                "kind": "persisted_task",
                "store": str(Path(store_root).resolve()),
            }
        payload = _vllm_probe_plan(configured, source=source)
    except (StoreError, ValidationError, ValueError, RuntimeError, OSError) as exc:
        _fail(str(exc))
    typer.echo(_json(payload))
    if not payload["accepted"]:
        _fail(
            "probe request hash mismatch; no probe was executed. Copy the computed "
            "request_sha256 values into the immutable config and review the diff."
        )
