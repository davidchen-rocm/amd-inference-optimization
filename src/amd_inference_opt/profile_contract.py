"""Validate profiling against the installed server's advertised contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class ProfileContractError(RuntimeError):
    """The installed profiler cannot satisfy the requested evidence contract."""


_LIMITS = {
    "timeout_seconds",
    "max_trace_bytes",
    "max_trace_files",
    "max_events_per_type",
    "max_percentile_samples_per_kernel",
}


def _has_bounded_number(schema: Mapping[str, Any]) -> bool:
    """Require an advertised bound on every non-null numeric alternative."""

    if not isinstance(schema, Mapping):
        return False
    alternatives = schema.get("anyOf", [schema])
    numeric = [
        item for item in alternatives
        if isinstance(item, Mapping) and item.get("type") != "null"
    ]
    return bool(numeric) and all(
        item.get("type") in {"integer", "number"}
        and not isinstance(item.get("maximum"), bool)
        and isinstance(item.get("maximum"), (int, float))
        and math.isfinite(item["maximum"])
        and item["maximum"] > 0
        for item in numeric
    )


@dataclass(frozen=True)
class ProfileContract:
    input_schema: dict[str, Any]
    observation_capabilities: dict[str, Any]
    connection: dict[str, Any]
    observation_call: dict[str, Any]

    @property
    def sha256(self) -> str:
        # Probe timestamps and diagnostics vary between sessions. Bind the
        # installed API and usable profiler identity, preserving raw probes below.
        capabilities = self.observation_capabilities.get("capabilities", {})
        selected = {
            name: {
                key: capability.get(key)
                for key in ("status", "integrated", "tool", "executable_path", "version")
            }
            for name in ("kernel_trace", "kernel_timing", "kernel_metadata")
            if isinstance(capability := capabilities.get(name), Mapping)
        }
        payload = {
            "input_schema": self.input_schema,
            "capabilities": selected,
            "supported_profiler_options": self.observation_capabilities.get(
                "supported_profiler_options"
            ),
            "profiler_help_complete": self.observation_capabilities.get("profiler_help_complete"),
            "server_info": self.connection.get("protocol_result", {}).get("serverInfo"),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy({
            "schema": "gpuopt.rocm-profile-contract.v1",
            "contract_sha256": self.sha256,
            "input_schema": self.input_schema,
            "connection": self.connection,
            "observation_call": self.observation_call,
        })

    def validate(self, arguments: Mapping[str, Any]) -> None:
        schema = self.input_schema
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            raise ProfileContractError("installed ROCm profile inputSchema is invalid") from error
        pending: list[Any] = [schema]
        while pending:
            node = pending.pop()
            if isinstance(node, Mapping):
                if any(
                    key in node and not node[key].startswith("#")
                    for key in ("$ref", "$dynamicRef")
                ):
                    raise ProfileContractError(
                        "ROCm profile schema references an external resource"
                    )
                pending.extend(node.values())
            elif isinstance(node, list):
                pending.extend(node)
        properties = schema.get("properties")
        if schema.get("type") != "object" or not isinstance(properties, Mapping):
            raise ProfileContractError("ROCm profile tool has no usable inputSchema")
        unknown = set(arguments) - set(properties)
        if unknown:
            raise ProfileContractError(
                "installed ROCm profile tool does not support arguments: "
                + ", ".join(sorted(unknown))
            )
        for name in sorted(_LIMITS & set(arguments)):
            value = arguments[name]
            if value is None:
                continue
            if not _has_bounded_number(properties[name]):
                raise ProfileContractError(
                    f"installed ROCm profile schema does not advertise a hard maximum for {name}; "
                    "update ROCm Issue Agent before requesting profiling"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
                value
            ):
                raise ProfileContractError(f"profile {name} must be a finite number")
        errors = list(Draft202012Validator(schema).iter_errors(dict(arguments)))
        if errors:
            # Errors under anyOf carry the actionable numeric-limit failure in context.
            error = errors[0]
            detail = "; ".join(
                child.message for child in error.context if child.validator != "type"
            ) or error.message
            location = ".".join(map(str, error.absolute_path)) or "arguments"
            raise ProfileContractError(f"ROCm profile contract rejects {location}: {detail}")
        if (
            "profiler_help_complete" in self.observation_capabilities
            and self.observation_capabilities["profiler_help_complete"] is not True
        ):
            raise ProfileContractError("installed ROCm profiler help was incomplete or failed")
        preset = arguments.get(
            "preset", properties.get("preset", {}).get("default", "kernel-timing")
        )
        required = ["kernel_trace"]
        if preset in {"kernel-timing", "kernel-metadata"}:
            required.append("kernel_timing")
        if preset == "kernel-metadata":
            required.append("kernel_metadata")
        capabilities = self.observation_capabilities.get("capabilities", {})
        if not isinstance(capabilities, Mapping):
            raise ProfileContractError("ROCm observation capabilities are malformed")
        for name in required:
            capability = capabilities.get(name, {})
            if not isinstance(capability, Mapping):
                raise ProfileContractError(f"ROCm observation capability is malformed: {name}")
            if capability.get("status") != "available" or capability.get("integrated") is not True:
                raise ProfileContractError(
                    f"installed ROCm profiler lacks usable {name}: "
                    f"{capability.get('status', 'unknown')}"
                )
