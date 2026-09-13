from __future__ import annotations

import json
from pathlib import Path

import pytest

from amd_inference_opt.profile_contract import ProfileContract, ProfileContractError


@pytest.fixture
def contract() -> ProfileContract:
    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "rocm-profile-contract.json").read_text()
    )
    return ProfileContract(**payload)


@pytest.mark.parametrize("name,value", [
    ("max_trace_bytes", 500_000_000),
    ("max_trace_files", 65),
    ("max_events_per_type", 10_001),
    ("max_percentile_samples_per_kernel", 10_001),
    ("timeout_seconds", 3601),
    ("timeout_seconds", float("nan")),
])
def test_rejects_limits_installed_server_cannot_honor(
    contract: ProfileContract, name: str, value: float
) -> None:
    with pytest.raises(ProfileContractError, match=name):
        contract.validate({"command": ["/usr/bin/python3"], name: value})


def test_unknown_server_limits_are_not_guessed(contract: ProfileContract) -> None:
    del contract.input_schema["properties"]["max_trace_bytes"]["anyOf"][0]["maximum"]
    with pytest.raises(ProfileContractError, match="does not advertise a hard maximum"):
        contract.validate({"command": ["/usr/bin/python3"], "max_trace_bytes": 1})


def test_unsupported_arguments_and_metadata_fail_before_execution(
    contract: ProfileContract,
) -> None:
    with pytest.raises(ProfileContractError, match="max_trace_size_mb"):
        contract.validate({"command": ["python"], "max_trace_size_mb": 500})
    with pytest.raises(ProfileContractError, match="kernel_metadata"):
        contract.validate({"command": ["python"], "preset": "kernel-metadata"})


def test_detected_but_unintegrated_timing_is_not_available(contract: ProfileContract) -> None:
    contract.observation_capabilities["capabilities"]["kernel_timing"]["integrated"] = False
    with pytest.raises(ProfileContractError, match="kernel_timing"):
        contract.validate({"command": ["python"], "preset": "kernel-timing"})


def test_omitted_preset_checks_the_installed_server_default(contract: ProfileContract) -> None:
    contract.input_schema["properties"]["preset"]["default"] = "kernel-timing"
    contract.observation_capabilities["capabilities"]["kernel_timing"]["status"] = "tool_missing"
    with pytest.raises(ProfileContractError, match="kernel_timing"):
        contract.validate({"command": ["python"]})
    contract.input_schema["properties"]["preset"]["default"] = "kernel-basic"
    contract.validate({"command": ["python"]})


def test_contract_hash_ignores_probe_time_but_binds_budget_and_profiler(contract: ProfileContract):
    first = contract.sha256
    contract.observation_capabilities["collected_at"] = "2026-09-06T01:00:00+00:00"
    assert contract.sha256 == first
    contract.observation_capabilities["capabilities"]["kernel_trace"]["version"] = "7.3"
    second = contract.sha256
    assert second != first
    contract.input_schema["properties"]["max_trace_bytes"]["anyOf"][0]["maximum"] = 100
    assert contract.sha256 != second


def test_contract_binds_profiler_flags_when_version_is_unchanged(contract: ProfileContract):
    contract.observation_capabilities.update({
        "supported_profiler_options": ["--kernel-trace", "--hip-trace"],
        "profiler_help_complete": True,
    })
    original = contract.sha256
    contract.observation_capabilities["supported_profiler_options"].append("--memory-copy-trace")
    assert contract.sha256 != original
    original = contract.sha256
    contract.observation_capabilities["profiler_help_complete"] = False
    assert contract.sha256 != original


def test_malformed_bound_schema_is_an_actionable_contract_error(contract: ProfileContract):
    contract.input_schema["properties"]["max_trace_bytes"] = True
    with pytest.raises(ProfileContractError, match="hard maximum"):
        contract.validate({"command": ["python"], "max_trace_bytes": 1})


def test_external_schema_references_are_not_fetched(contract: ProfileContract):
    contract.input_schema["properties"]["command"] = {"$ref": "https://example.com/schema"}
    with pytest.raises(ProfileContractError, match="external resource"):
        contract.validate({"command": ["python"]})


def test_failed_help_cannot_support_available_timing_claim(contract: ProfileContract):
    contract.observation_capabilities["profiler_help_complete"] = False
    with pytest.raises(ProfileContractError, match="help was incomplete or failed"):
        contract.validate({"command": ["python"], "preset": "kernel-timing"})
