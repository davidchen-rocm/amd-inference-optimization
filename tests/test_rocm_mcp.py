from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from amd_inference_opt.profile_contract import ProfileContract
from amd_inference_opt.rocm_mcp import (
    ApprovalAlreadyConsumed,
    ApprovalRequired,
    MCPServerConfig,
    RocmIssueAgentClient,
    RocmMCPError,
    approval_request,
)


class _FakeModel:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        for key, value in payload.items():
            setattr(self, key, value)

    def model_dump(self, **_: object) -> dict[str, Any]:
        return self.payload.copy()


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self) -> _FakeModel:
        return _FakeModel(
            {"protocolVersion": "2025-06-18", "serverInfo": {"name": "ROCm Issue Agent"}}
        )

    async def list_tools(self) -> _FakeModel:
        names = [
            "rocm_snapshot",
            "rocm_hip_capabilities",
            "rocm_observation_capabilities",
            "rocm_profile_workload",
            "rocm_get_case_summary",
        ]
        return _FakeModel({"tools": [_FakeModel({"name": name}) for name in names]})

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FakeModel:
        self.calls.append((name, arguments))
        schemas = {
            "rocm_snapshot": "rocm.mcp-snapshot.v1",
            "rocm_hip_capabilities": "rocm.hip-capabilities.v1",
            "rocm_observation_capabilities": "rocm.observation-capabilities.v1",
            "rocm_profile_workload": "rocm.mcp-kernel-evidence.v1",
            "rocm_get_case_summary": "rocm.mcp-case-summary.v1",
        }
        structured = {
            "schema": schemas[name],
            "case_id": "case-1",
            "nested": {"kernel_evidence": [{"name": "gemv", "duration_us": 98.5}]},
        }
        return _FakeModel(
            {
                "isError": False,
                "content": [{"type": "text", "text": "summary"}],
                "structuredContent": structured,
            }
        )


def _factory(session: _FakeSession):
    @asynccontextmanager
    async def factory(_: MCPServerConfig) -> AsyncIterator[_FakeSession]:
        yield session

    return factory


def test_connect_records_protocol_inventory_and_exact_structured_response(tmp_path: Path) -> None:
    async def exercise() -> None:
        session = _FakeSession()
        config = MCPServerConfig(
            command="/opt/tools/rocm-agent-mcp",
            env={"ROCM_AGENT_HOME": str(tmp_path)},
        )
        async with RocmIssueAgentClient(config, session_factory=_factory(session)) as client:
            evidence = client.connection_evidence
            assert evidence is not None
            assert evidence.protocol_result["protocolVersion"] == "2025-06-18"
            assert evidence.server_env_keys == ("ROCM_AGENT_HOME",)
            calls = await client.inspect_environment()
            assert calls[0].structured_content == {
                "schema": "rocm.mcp-snapshot.v1",
                "case_id": "case-1",
                "nested": {"kernel_evidence": [{"name": "gemv", "duration_us": 98.5}]},
            }
            assert calls[0].raw_result["structuredContent"] == calls[0].structured_content
            assert calls[0].contract_error is None

    asyncio.run(exercise())


def test_contract_error_preserves_but_flags_unknown_schema(tmp_path: Path) -> None:
    async def exercise() -> None:
        session = _FakeSession()
        async with RocmIssueAgentClient(
            MCPServerConfig(command="/opt/tools/rocm-agent-mcp"),
            session_factory=_factory(session),
        ) as client:
            call = await client.call_tool("rocm_snapshot")
            assert call.contract_error is None
            object.__setattr__(call, "structured_content", {"schema": "future.v2"})
            assert "expected 'rocm.mcp-snapshot.v1'" in (call.contract_error or "")

    asyncio.run(exercise())


def test_profile_requires_approval_for_exact_arguments(tmp_path: Path) -> None:
    async def exercise() -> None:
        session = _FakeSession()
        config = MCPServerConfig(command="/opt/tools/rocm-agent-mcp")
        async with RocmIssueAgentClient(config, session_factory=_factory(session)) as client:
            arguments = client.profile_arguments(
                ["./llama-bench", "-n", "128"],
                cwd=tmp_path,
                timeout_seconds=60,
            )
            request = client.make_approval_request("rocm_profile_workload", arguments)
            with pytest.raises(ApprovalRequired) as mismatch:
                await client.profile_workload(
                    ["./llama-bench", "-n", "512"],
                    cwd=tmp_path,
                    timeout_seconds=60,
                    approval_sha256=request.request_sha256,
                )
            assert mismatch.value.request.request_sha256 != request.request_sha256

            result = await client.profile_workload(
                ["./llama-bench", "-n", "128"],
                cwd=tmp_path,
                timeout_seconds=60,
                approval_sha256=request.request_sha256,
            )
            assert not result.is_error
            assert session.calls[-1] == ("rocm_profile_workload", arguments)
            with pytest.raises(ApprovalAlreadyConsumed):
                await client.profile_workload(
                    ["./llama-bench", "-n", "128"],
                    cwd=tmp_path,
                    timeout_seconds=60,
                    approval_sha256=request.request_sha256,
                )

    asyncio.run(exercise())


def test_approval_hash_is_canonical_and_argument_bound() -> None:
    first = approval_request("rocm_profile_workload", {"preset": "kernel-timing", "command": ["x"]})
    reordered = approval_request(
        "rocm_profile_workload", {"command": ["x"], "preset": "kernel-timing"}
    )
    changed = approval_request(
        "rocm_profile_workload", {"command": ["y"], "preset": "kernel-timing"}
    )

    assert first.request_sha256 == reordered.request_sha256
    assert first.request_sha256 != changed.request_sha256


def test_profile_capture_limits_are_validated_and_approval_bound() -> None:
    limited = RocmIssueAgentClient.profile_arguments(
        ["/opt/llama-bench", "-n", "128"],
        max_trace_bytes=123_000_000,
        max_trace_files=12,
        max_events_per_type=4_000,
        max_percentile_samples_per_kernel=16,
    )
    changed = {**limited, "max_trace_bytes": 124_000_000}

    assert limited["max_trace_files"] == 12
    assert limited["max_percentile_samples_per_kernel"] == 16
    assert approval_request(
        "rocm_profile_workload", limited
    ).request_sha256 != approval_request(
        "rocm_profile_workload", changed
    ).request_sha256
    with pytest.raises(RocmMCPError, match="max_events_per_type"):
        RocmIssueAgentClient.profile_arguments(
            ["/opt/llama-bench"], max_events_per_type=0
        )


def test_approval_hash_binds_stdio_execution_environment_and_attempt() -> None:
    arguments = {"command": ["/opt/llama-bench"], "preset": "kernel-timing"}
    first = approval_request(
        "rocm_profile_workload",
        arguments,
        execution_context={
            "attempt_id": "baseline-attempt-0001",
            "runtime_environment": {"HSA_VISIBLE_DEVICES": "0"},
        },
    )
    retry = approval_request(
        "rocm_profile_workload",
        arguments,
        execution_context={
            "attempt_id": "baseline-attempt-0002",
            "runtime_environment": {"HSA_VISIBLE_DEVICES": "0"},
        },
    )
    changed_environment = approval_request(
        "rocm_profile_workload",
        arguments,
        execution_context={
            "attempt_id": "baseline-attempt-0001",
            "runtime_environment": {"HSA_VISIBLE_DEVICES": "1"},
        },
    )

    assert first.request_sha256 != retry.request_sha256
    assert first.request_sha256 != changed_environment.request_sha256


def test_server_config_converts_domain_command_to_program_and_args(tmp_path: Path) -> None:
    domain = SimpleNamespace(
        command=["/opt/rocm/bin/rocm-agent-mcp", "--debug"],
        env={"ROCM_AGENT_HOME": "/var/tmp/rocm-agent"},
        cwd=tmp_path,
    )

    config = MCPServerConfig.from_domain(domain)

    assert config.command == "/opt/rocm/bin/rocm-agent-mcp"
    assert config.args == ("--debug",)
    assert config.cwd == str(tmp_path.resolve())


def test_live_profile_discovery_and_drift_do_not_consume_execution_approval() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "rocm-profile-contract.json").read_text()
    )

    class ContractSession(_FakeSession):
        async def list_tools(self) -> _FakeModel:
            return _FakeModel({"tools": [_FakeModel({
                "name": "rocm_profile_workload", "inputSchema": fixture["input_schema"],
            }), _FakeModel({"name": "rocm_observation_capabilities"})]})

        async def call_tool(self, name, arguments):
            if name == "rocm_observation_capabilities":
                self.calls.append((name, arguments))
                return _FakeModel({
                    "isError": False,
                    "structuredContent": fixture["observation_capabilities"],
                })
            return await super().call_tool(name, arguments)

    async def exercise() -> None:
        session = ContractSession()
        config = MCPServerConfig(command="/opt/rocm-agent-mcp")
        async with RocmIssueAgentClient(config, session_factory=_factory(session)) as client:
            contract = await client.profile_contract()
            assert isinstance(contract, ProfileContract)
            arguments = client.profile_arguments(["/usr/bin/python3"], max_trace_bytes=200_000_000)
            contract.validate(arguments)
            client._approval_context = {"profile_contract_sha256": contract.sha256}
            approval = client.make_approval_request("rocm_profile_workload", arguments)
            fixture["observation_capabilities"]["capabilities"]["kernel_timing"]["version"] = "7.3"
            with pytest.raises(RocmMCPError, match="changed after approval"):
                await client.call_tool(
                    "rocm_profile_workload", arguments, approval_sha256=approval.request_sha256
                )
            assert approval.request_sha256 not in client._consumed_approval_hashes
            assert all(name != "rocm_profile_workload" for name, _ in session.calls)
            fixture["observation_capabilities"]["capabilities"]["kernel_timing"]["version"] = "7.2"
            await client.call_tool(
                "rocm_profile_workload", arguments, approval_sha256=approval.request_sha256
            )
            assert approval.request_sha256 in client._consumed_approval_hashes

    asyncio.run(exercise())
