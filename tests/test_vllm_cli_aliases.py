from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from test_vllm_workflow import _config

from amd_inference_opt.vllm_adapter import VLLMSpecError, build_bench_serve_argv
from amd_inference_opt.vllm_models import VLLMCampaignConfig


def _with_benchmark_argv(
    config: VLLMCampaignConfig, argv: list[str]
) -> VLLMCampaignConfig:
    serving = config.serving.model_copy(update={"benchmark_argv": argv})
    task = config.task.model_copy(
        update={
            "benchmark": config.task.benchmark.model_copy(
                update={
                    "benchmark_command": argv,
                    "protocol_hash": serving.coordinate_sha256,
                }
            )
        }
    )
    return VLLMCampaignConfig.model_validate(
        {**config.model_dump(), "serving": serving, "task": task}
    )


def _replace_option(argv: list[str], old: str, replacement: list[str]) -> list[str]:
    index = argv.index(old)
    return [*argv[:index], *replacement, *argv[index + 2 :]]


@pytest.mark.parametrize("input_flag", ["--input-len", "--random-input-len"])
@pytest.mark.parametrize("output_flag", ["--output-len", "--random-output-len"])
@pytest.mark.parametrize("equals", [False, True])
def test_serving_accepts_one_observed_length_alias_per_coordinate(
    tmp_path: Path, input_flag: str, output_flag: str, equals: bool
) -> None:
    config = _config(tmp_path)
    argv = config.serving.benchmark_argv
    for old, selected, value in (
        ("--random-input-len", input_flag, "128"),
        ("--random-output-len", output_flag, "32"),
    ):
        replacement = [f"{selected}={value}"] if equals else [selected, value]
        argv = _replace_option(argv, old, replacement)
    validated = _with_benchmark_argv(config, argv)
    assert validated.serving.input_tokens == 128
    assert validated.serving.output_tokens == 32
    assert validated.serving.benchmark_argv == argv


@pytest.mark.parametrize("coordinate,value", [("input", "128"), ("output", "32")])
@pytest.mark.parametrize("conflicting", [False, True])
@pytest.mark.parametrize("equals", [False, True])
@pytest.mark.parametrize("primary_first", [False, True])
def test_serving_rejects_mixed_length_aliases_even_when_values_agree(
    tmp_path: Path,
    coordinate: str,
    value: str,
    conflicting: bool,
    equals: bool,
    primary_first: bool,
) -> None:
    config = _config(tmp_path)
    primary = f"--{coordinate}-len"
    extra_value = "4096" if conflicting else value
    extra = [f"{primary}={extra_value}"] if equals else [primary, extra_value]
    argv = config.serving.benchmark_argv
    if primary_first:
        index = argv.index(f"--random-{coordinate}-len")
        argv = [*argv[:index], *extra, *argv[index:]]
    else:
        argv = [*argv, *extra]
    with pytest.raises(ValidationError, match=f"benchmark {coordinate}_tokens exactly once"):
        _with_benchmark_argv(config, argv)


@pytest.mark.parametrize("coordinate", ["input", "output"])
def test_serving_rejects_primary_length_alias_with_wrong_shape(
    tmp_path: Path, coordinate: str
) -> None:
    config = _config(tmp_path)
    argv = _replace_option(
        config.serving.benchmark_argv,
        f"--random-{coordinate}-len",
        [f"--{coordinate}-len=4096"],
    )
    with pytest.raises(ValidationError, match=f"benchmark {coordinate}_tokens exactly once"):
        _with_benchmark_argv(config, argv)


@pytest.mark.parametrize("flag", ["--input-len", "--output-len"])
@pytest.mark.parametrize("equals", [False, True])
@pytest.mark.parametrize("location", ["extra_args", "vllm_argv"])
def test_benchmark_builder_rejects_primary_alias_override(
    flag: str, equals: bool, location: str
) -> None:
    override = (f"{flag}=4096",) if equals else (flag, "4096")
    kwargs = {location: ("vllm", *override) if location == "vllm_argv" else override}
    with pytest.raises(VLLMSpecError, match="locked benchmark flags"):
        build_bench_serve_argv(
            model="org/model",
            base_url="http://127.0.0.1:8000",
            random_input_len=128,
            random_output_len=32,
            **kwargs,
        )
