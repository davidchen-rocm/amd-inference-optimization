#!/usr/bin/env python3
"""Request or execute one approval-bound Q4_RDNA kernel-timing profile."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

from amd_inference_opt.llama_cpp import sha256_file
from amd_inference_opt.protocol import BASELINE_UNSET_ENVIRONMENT, amd_runtime_environment
from amd_inference_opt.rocm_mcp import (
    MCPServerConfig,
    RocmIssueAgentClient,
    approval_request,
)


def _write_atomic(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _binding(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, str]]:
    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    sidecar = args.sidecar.resolve(strict=True)
    extra = {"LLAMA_Q4_RDNA_SIDECAR": str(sidecar)}
    if args.arm == "candidate-paired":
        extra["LLAMA_Q4_RDNA_GATE_PAIR"] = "1"
    runtime = amd_runtime_environment(
        rocm_library_paths=(
            str(binary.parent),
            "/opt/rocm/core-7.14/lib",
            "/opt/rocm/lib",
        ),
        extra=extra,
    )
    unset = tuple(name for name in BASELINE_UNSET_ENVIRONMENT if name not in runtime)
    command = [
        str(binary),
        "-m",
        str(model),
        "-p",
        "0",
        "-n",
        "8",
        "-b",
        "2048",
        "-ub",
        "512",
        "-t",
        "12",
        "-r",
        "1",
        "-ngl",
        "999",
        "-mg",
        "0",
        "-dev",
        "ROCm0",
        "-o",
        "json",
        "-oe",
        "none",
    ]
    arguments = RocmIssueAgentClient.profile_arguments(
        command,
        preset="kernel-timing",
        cwd=binary.parent,
        timeout_seconds=300,
        max_trace_bytes=200_000_000,
        max_trace_files=64,
        max_events_per_type=10_000,
        max_percentile_samples_per_kernel=1_000,
    )
    context: dict[str, object] = {
        "attempt_id": args.attempt_id,
        "arm": args.arm,
        "runtime_environment": dict(sorted(runtime.items())),
        "unset_environment": list(unset),
        "binary_sha256": sha256_file(binary),
        "model_sha256": sha256_file(model),
        "sidecar_sha256": sha256_file(sidecar),
        "profile_coordinate": "decode-n8-r1-full-shape-specific",
    }
    return {"arguments": arguments, "context": context}, runtime


async def _execute(
    *,
    args: argparse.Namespace,
    binding: dict[str, object],
    runtime: dict[str, str],
    approval_sha256: str,
) -> dict[str, object]:
    server_env = {"ROCM_AGENT_HOME": str(args.agent_home.resolve())}
    server_env.update(runtime)
    config = MCPServerConfig(
        command=str(args.mcp_command.resolve(strict=True)),
        env=server_env,
        cwd=str(args.mcp_cwd.resolve(strict=True)),
    )
    context = binding["context"]
    arguments = binding["arguments"]
    assert isinstance(context, dict) and isinstance(arguments, dict)
    async with RocmIssueAgentClient(config, approval_context=context) as client:
        connection = client.connection_evidence
        profile = await client.call_tool(
            "rocm_profile_workload",
            arguments,
            approval_sha256=approval_sha256,
        )
        case_id = None
        if isinstance(profile.structured_content, dict):
            value = profile.structured_content.get("case_id")
            case_id = value if isinstance(value, str) else None
        summary = await client.get_case_summary(case_id) if case_id else None
        return {
            "schema": "gpuopt.q4rdna-pair-profile-envelope.v1",
            "connection": connection.to_dict() if connection else None,
            "profile": profile.to_dict(),
            "case_summary": summary.to_dict() if summary else None,
            "contract_error": profile.contract_error,
            "case_id": case_id,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("baseline", "candidate-paired"), required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--mcp-command",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mcp-cwd",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--agent-home", type=Path, default=Path.home() / ".rocm-agent"
    )
    parser.add_argument("--approval-sha256")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    binding, runtime = _binding(args)
    context = binding["context"]
    arguments = binding["arguments"]
    assert isinstance(context, dict) and isinstance(arguments, dict)
    approval = approval_request(
        "rocm_profile_workload",
        arguments,
        execution_context=context,
    )
    request = {
        "schema": "gpuopt.mcp-approval-request.v1",
        "request_id": args.attempt_id,
        **approval.to_dict(),
    }
    request_path = root / "approval-request.json"
    if request_path.exists():
        existing = json.loads(request_path.read_text(encoding="utf-8"))
        if existing != request:
            raise SystemExit("existing approval request has a different binding")
    else:
        _write_atomic(request_path, request)
    if args.approval_sha256 is None:
        print(json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    receipt_path = root / "approval-consumed.json"
    if receipt_path.exists():
        raise SystemExit("approval was already consumed for this attempt")
    if args.approval_sha256 != approval.request_sha256:
        raise SystemExit("approval SHA-256 does not match the exact request")
    _write_atomic(
        receipt_path,
        {
            "schema": "gpuopt.mcp-approval-consumption.v1",
            "request_id": args.attempt_id,
            "request_sha256": approval.request_sha256,
            "consumed_before_transport": True,
        },
    )
    envelope = asyncio.run(
        _execute(
            args=args,
            binding=binding,
            runtime=runtime,
            approval_sha256=args.approval_sha256,
        )
    )
    _write_atomic(root / "evidence.json", envelope)
    print(json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
