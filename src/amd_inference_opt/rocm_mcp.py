"""Official MCP stdio integration for the existing ROCm Issue Agent.

This module deliberately does not parse ROCm profiler artifacts.  It records the
MCP protocol result and its ``structuredContent`` exactly, leaving evidence
interpretation to the optimization agent and workflow.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .command import validate_argv
from .profile_contract import ProfileContract, ProfileContractError

READ_ONLY_TOOLS = frozenset(
    {
        "rocm_snapshot",
        "rocm_hip_capabilities",
        "rocm_observation_capabilities",
        "rocm_get_case_summary",
        "rocm_get_trace_summary",
    }
)
EXECUTING_TOOLS = frozenset(
    {
        "rocm_run_workload",
        "rocm_hip_smoke_test",
        "rocm_trace_workload",
        "rocm_profile_workload",
        "rocm_run_validation",
    }
)

EXPECTED_SCHEMAS: dict[str, str] = {
    "rocm_snapshot": "rocm.mcp-snapshot.v1",
    "rocm_hip_capabilities": "rocm.hip-capabilities.v1",
    "rocm_observation_capabilities": "rocm.observation-capabilities.v1",
    "rocm_run_workload": "rocm.mcp-diagnosis.v1",
    "rocm_trace_workload": "rocm.mcp-trace-result.v1",
    "rocm_profile_workload": "rocm.mcp-kernel-evidence.v1",
    "rocm_get_trace_summary": "rocm.mcp-trace-summary.v1",
    "rocm_run_validation": "rocm.mcp-validation-result.v1",
    "rocm_get_case_summary": "rocm.mcp-case-summary.v1",
}


class RocmMCPError(RuntimeError):
    """MCP configuration, connection, or contract error."""


class ApprovalRequired(RocmMCPError):
    def __init__(self, request: MCPApprovalRequest) -> None:
        super().__init__(
            f"explicit approval required for {request.tool_name}; "
            f"approve request sha256={request.request_sha256}"
        )
        self.request = request


class ApprovalAlreadyConsumed(RocmMCPError):
    """Raised when an execution approval is replayed."""


@dataclass(frozen=True)
class MCPServerConfig:
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None

    def __post_init__(self) -> None:
        command_path = Path(self.command)
        if not command_path.is_absolute():
            raise RocmMCPError("MCP server command must be an absolute path")
        validate_argv((self.command, *self.args))
        if self.cwd is not None and not Path(self.cwd).is_absolute():
            raise RocmMCPError("MCP server cwd must be an absolute path")
        for key, value in self.env.items():
            if not isinstance(key, str) or not key or not isinstance(value, str):
                raise RocmMCPError("MCP environment must map non-empty strings to strings")

    @classmethod
    def from_domain(cls, config: Any) -> MCPServerConfig:
        """Convert the public ``models.MCPConfig`` without coupling to Pydantic."""

        command = validate_argv(config.command)
        cwd = str(Path(config.cwd).resolve()) if config.cwd is not None else None
        return cls(
            command=command[0],
            args=command[1:],
            env=dict(config.env),
            cwd=cwd,
        )


@dataclass(frozen=True)
class MCPApprovalRequest:
    tool_name: str
    arguments: dict[str, Any]
    execution_context: dict[str, Any]
    request_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MCPToolCall:
    tool_name: str
    arguments: dict[str, Any]
    called_at: str
    is_error: bool
    structured_content: Any
    raw_result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def schema_name(self) -> str | None:
        if not isinstance(self.structured_content, dict):
            return None
        value = self.structured_content.get("schema")
        return value if isinstance(value, str) else None

    @property
    def contract_error(self) -> str | None:
        if self.is_error:
            return f"{self.tool_name} returned MCP isError=true"
        expected = EXPECTED_SCHEMAS.get(self.tool_name)
        if expected is None:
            return None
        if not isinstance(self.structured_content, dict):
            return f"{self.tool_name} returned no structured object"
        if self.schema_name != expected:
            return (
                f"{self.tool_name} returned schema {self.schema_name!r}; "
                f"expected {expected!r}"
            )
        return None


@dataclass(frozen=True)
class MCPConnectionEvidence:
    protocol_result: dict[str, Any]
    tool_inventory: tuple[dict[str, Any], ...]
    server_command: str
    server_args: tuple[str, ...]
    server_cwd: str | None
    server_env_keys: tuple[str, ...]
    server_env_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dump_raw(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=False)
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return copy.deepcopy(value)


def approval_request(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    execution_context: Mapping[str, Any] | None = None,
) -> MCPApprovalRequest:
    """Create an approval bound to the tool call and its execution context.

    ROCm Issue Agent deliberately does not accept environment injection in tool
    arguments.  Workload environment is instead fixed when its stdio server is
    spawned.  Binding that environment (and a persistent attempt id) here keeps
    the human approval exact without weakening the MCP server's interface.
    """

    frozen_arguments = copy.deepcopy(dict(arguments))
    frozen_context = copy.deepcopy(dict(execution_context or {}))
    try:
        encoded = json.dumps(
            {
                "tool_name": tool_name,
                "arguments": frozen_arguments,
                "execution_context": frozen_context,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RocmMCPError("MCP tool arguments must be JSON serializable") from error
    return MCPApprovalRequest(
        tool_name=tool_name,
        arguments=frozen_arguments,
        execution_context=frozen_context,
        request_sha256=hashlib.sha256(encoded).hexdigest(),
    )


@asynccontextmanager
async def _official_stdio_session(config: MCPServerConfig) -> AsyncIterator[Any]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise RocmMCPError("the official 'mcp' package is required for ROCm integration") from error

    parameters = StdioServerParameters(
        command=config.command,
        args=list(config.args),
        env=dict(config.env) or None,
        cwd=config.cwd,
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            yield session


SessionFactory = Callable[[MCPServerConfig], AbstractAsyncContextManager[Any]]


class RocmIssueAgentClient:
    """One initialized stdio MCP session with exact-call approval enforcement."""

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        session_factory: SessionFactory | None = None,
        approval_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self._session_factory = session_factory or _official_stdio_session
        self._context: AbstractAsyncContextManager[Any] | None = None
        self._session: Any | None = None
        self.connection_evidence: MCPConnectionEvidence | None = None
        self._tool_names: set[str] = set()
        self._consumed_approval_hashes: set[str] = set()
        self._approval_context = copy.deepcopy(dict(approval_context or {}))

    async def __aenter__(self) -> RocmIssueAgentClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def connect(self) -> MCPConnectionEvidence:
        if self._session is not None:
            if self.connection_evidence is None:  # defensive invariant
                raise RocmMCPError("connected MCP client has no initialization evidence")
            return self.connection_evidence
        context = self._session_factory(self.config)
        entered = False
        try:
            session = await context.__aenter__()
            entered = True
            initialized = await session.initialize()
            tools_result = await session.list_tools()
            tools = list(getattr(tools_result, "tools", []))
            seen_cursors: set[str] = set()
            while cursor := getattr(tools_result, "nextCursor", None):
                if cursor in seen_cursors:
                    raise RocmMCPError("MCP tool inventory pagination repeated a cursor")
                seen_cursors.add(cursor)
                tools_result = await session.list_tools(cursor=cursor)
                tools.extend(getattr(tools_result, "tools", []))
        except BaseException as error:
            if entered:
                await context.__aexit__(type(error), error, error.__traceback__)
            raise
        self._context = context
        self._session = session
        inventory = tuple(_dump_raw(tool) for tool in tools)
        self._tool_names = {
            str(item.get("name"))
            for item in inventory
            if isinstance(item, dict) and item.get("name")
        }
        self.connection_evidence = MCPConnectionEvidence(
            protocol_result=_dump_raw(initialized),
            tool_inventory=inventory,
            server_command=self.config.command,
            server_args=self.config.args,
            server_cwd=self.config.cwd,
            server_env_keys=tuple(sorted(self.config.env)),
            server_env_sha256=hashlib.sha256(
                json.dumps(
                    dict(self.config.env),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )
        return self.connection_evidence

    async def close(self) -> None:
        context = self._context
        self._context = None
        self._session = None
        self.connection_evidence = None
        self._tool_names.clear()
        if context is not None:
            await context.__aexit__(None, None, None)

    def make_approval_request(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> MCPApprovalRequest:
        if tool_name not in EXECUTING_TOOLS:
            raise RocmMCPError(f"tool does not require execution approval: {tool_name}")
        return approval_request(
            tool_name,
            arguments,
            execution_context=self._approval_context,
        )

    async def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        approval_sha256: str | None = None,
    ) -> MCPToolCall:
        if self._session is None:
            raise RocmMCPError("MCP client is not connected; use 'async with' or connect()")
        selected_arguments = copy.deepcopy(dict(arguments or {}))
        if self._tool_names and tool_name not in self._tool_names:
            raise RocmMCPError(f"ROCm Issue Agent does not expose required tool: {tool_name}")
        if tool_name in EXECUTING_TOOLS:
            request = approval_request(
                tool_name,
                selected_arguments,
                execution_context=self._approval_context,
            )
            if approval_sha256 != request.request_sha256:
                raise ApprovalRequired(request)
            if request.request_sha256 in self._consumed_approval_hashes:
                raise ApprovalAlreadyConsumed(
                    f"approval has already been consumed: {request.request_sha256}"
                )
            if tool_name == "rocm_profile_workload" and (
                expected_contract := self._approval_context.get("profile_contract_sha256")
            ):
                await self.check_profile_contract(
                    selected_arguments, expected_sha256=expected_contract
                )
            # Consume before transport. A failed or cancelled execution may still have
            # reached the local server and must never make the receipt reusable.
            self._consumed_approval_hashes.add(request.request_sha256)
        result = await self._session.call_tool(tool_name, selected_arguments)
        raw = _dump_raw(result)
        if not isinstance(raw, dict):
            raise RocmMCPError(f"unexpected MCP result type for {tool_name}")
        # Keep structuredContent independent from any projections consumers may build.
        structured = copy.deepcopy(getattr(result, "structuredContent", None))
        if structured is None and "structuredContent" in raw:
            structured = copy.deepcopy(raw["structuredContent"])
        return MCPToolCall(
            tool_name=tool_name,
            arguments=selected_arguments,
            called_at=datetime.now(UTC).isoformat(),
            is_error=bool(getattr(result, "isError", raw.get("isError", False))),
            structured_content=structured,
            raw_result=raw,
        )

    async def inspect_environment(self) -> tuple[MCPToolCall, ...]:
        return (
            await self.call_tool("rocm_snapshot"),
            await self.call_tool("rocm_hip_capabilities"),
            await self.call_tool("rocm_observation_capabilities"),
        )

    async def profile_contract(self) -> ProfileContract:
        """Read the live API and tool capabilities without executing a workload."""

        if self.connection_evidence is None:
            raise RocmMCPError("MCP client is not connected")
        inventory = self.connection_evidence.tool_inventory
        profile_tools = [item for item in inventory if item.get("name") == "rocm_profile_workload"]
        if len(profile_tools) != 1 or not isinstance(
            profile_tools[0].get("inputSchema"), dict
        ):
            raise RocmMCPError("installed ROCm profile tool has no unique inputSchema")
        call = await self.call_tool("rocm_observation_capabilities")
        if call.contract_error:
            raise RocmMCPError(call.contract_error)
        return ProfileContract(
            input_schema=copy.deepcopy(profile_tools[0]["inputSchema"]),
            observation_capabilities=copy.deepcopy(call.structured_content),
            connection=self.connection_evidence.to_dict(),
            observation_call=call.to_dict(),
        )

    async def check_profile_contract(
        self, arguments: Mapping[str, Any], *, expected_sha256: str | None = None
    ) -> ProfileContract:
        contract = await self.profile_contract()
        try:
            contract.validate(arguments)
        except ProfileContractError as error:
            raise RocmMCPError(str(error)) from error
        if expected_sha256 is not None and contract.sha256 != expected_sha256:
            raise RocmMCPError(
                "ROCm profile contract changed after approval; request a new approval"
            )
        return contract

    @staticmethod
    def profile_arguments(
        command: Sequence[str],
        *,
        preset: str = "kernel-timing",
        cwd: str | Path | None = None,
        timeout_seconds: float | None = None,
        python_executable: str | None = None,
        max_trace_bytes: int | None = None,
        max_trace_files: int | None = None,
        max_events_per_type: int | None = None,
        max_percentile_samples_per_kernel: int | None = None,
    ) -> dict[str, Any]:
        if preset not in {"kernel-basic", "kernel-timing", "kernel-metadata"}:
            raise RocmMCPError(f"unsupported ROCm profile preset: {preset}")
        arguments: dict[str, Any] = {"command": list(validate_argv(command)), "preset": preset}
        if cwd is not None:
            arguments["cwd"] = str(Path(cwd).resolve())
        if timeout_seconds is not None:
            if timeout_seconds <= 0:
                raise RocmMCPError("timeout_seconds must be greater than zero")
            arguments["timeout_seconds"] = timeout_seconds
        if python_executable is not None:
            arguments["python_executable"] = python_executable
        capture_limits = {
            "max_trace_bytes": max_trace_bytes,
            "max_trace_files": max_trace_files,
            "max_events_per_type": max_events_per_type,
            "max_percentile_samples_per_kernel": max_percentile_samples_per_kernel,
        }
        for name, value in capture_limits.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RocmMCPError(f"{name} must be a positive integer")
            arguments[name] = value
        return arguments

    async def profile_workload(
        self,
        command: Sequence[str],
        *,
        preset: str = "kernel-timing",
        cwd: str | Path | None = None,
        timeout_seconds: float | None = None,
        python_executable: str | None = None,
        max_trace_bytes: int | None = None,
        max_trace_files: int | None = None,
        max_events_per_type: int | None = None,
        max_percentile_samples_per_kernel: int | None = None,
        approval_sha256: str | None = None,
    ) -> MCPToolCall:
        arguments = self.profile_arguments(
            command,
            preset=preset,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
            max_trace_bytes=max_trace_bytes,
            max_trace_files=max_trace_files,
            max_events_per_type=max_events_per_type,
            max_percentile_samples_per_kernel=max_percentile_samples_per_kernel,
        )
        return await self.call_tool(
            "rocm_profile_workload",
            arguments,
            approval_sha256=approval_sha256,
        )

    async def get_case_summary(self, case_id: str) -> MCPToolCall:
        if not case_id:
            raise RocmMCPError("case_id must not be empty")
        return await self.call_tool("rocm_get_case_summary", {"case_id": case_id})
