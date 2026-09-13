#!/usr/bin/env python3
"""Compare two llama.cpp runtime/model arms with bounded process evidence.

This tool never synthesizes quality values. It executes each runtime's llama-cli
and llama-perplexity, and emits a complete result only after greedy, PPL, and
multiple-choice math evaluation all succeed for both runtimes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import selectors
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "gpuopt.q8-runtime-quality.v1"
MATH_SCHEMA = "gpuopt.q8-runtime-quality-fixture.v1"
PPL_PATTERN = re.compile(
    r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*\+/-\s*([0-9]+(?:\.[0-9]+)?)"
)
MATH_PATTERN = re.compile(
    r"Final result:\s*([0-9]+(?:\.[0-9]+)?)\s*\+/-\s*([0-9]+(?:\.[0-9]+)?)"
)
MATH_PROGRESS_PATTERN = re.compile(
    r"^\s*([0-9]+)[\t ]+([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE
)
ANSWER_PATTERN = re.compile(r"^[ABCD](?:,[ABCD])*$")
SUBJECTS = frozenset(
    {
        "abstract_algebra",
        "college_mathematics",
        "high_school_mathematics",
        "elementary_mathematics",
    }
)
LOCAL_LIBRARY_PREFIXES = ("libggml", "libllama")
REQUIRED_RUNTIME_LIBRARIES = ("libggml-hip.so", "libggml.so", "libllama.so")
MATH_SELECTION_SEED = 20260815
GREEDY_CANARY_SEED = 20260817
GREEDY_GENERATION_SEED = 20260815
FULL_MATH_TOTAL = 848
MAX_PPL_REGRESSION_FRACTION = 0.005
MAX_MATH_DROP_FRACTION = 0.01
MCNEMAR_ALPHA = 0.05
MATH_100_QUOTAS = {
    "abstract_algebra": 12,
    "college_mathematics": 12,
    "high_school_mathematics": 32,
    "elementary_mathematics": 44,
}
RUNTIME_ARG_ALIASES = {
    "-fa": "-fa",
    "--flash-attn": "-fa",
    "-ctk": "-ctk",
    "--cache-type-k": "-ctk",
    "-ctv": "-ctv",
    "--cache-type-v": "-ctv",
}
RUNTIME_ARG_VALUES = {
    "-fa": frozenset({"on", "off", "auto"}),
    "-ctk": frozenset({"f16", "q8_0", "q4_0"}),
    "-ctv": frozenset({"f16", "q8_0", "q4_0"}),
}
RUNTIME_ARG_ORDER = ("-fa", "-ctk", "-ctv")


class EvaluationError(RuntimeError):
    """The requested runtime evaluation is incomplete or invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_regular_file(path: Path, *, label: str, executable: bool = False) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise EvaluationError(f"{label} must be a regular, non-symlink file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise EvaluationError(f"{label} is not executable: {resolved}")
    return resolved


def resolve_tools(binary: Path, *, label: str) -> tuple[Path, Path]:
    resolved = binary.resolve()
    bin_dir = resolved if resolved.is_dir() else resolved.parent
    if resolved.is_file() and resolved.name == "llama-cli":
        cli = resolved
    else:
        cli = bin_dir / "llama-cli"
    perplexity = bin_dir / "llama-perplexity"
    return (
        require_regular_file(cli, label=f"{label} llama-cli", executable=True),
        require_regular_file(
            perplexity, label=f"{label} llama-perplexity", executable=True
        ),
    )


def parse_runtime_libraries(output: str, *, expected_dir: Path) -> dict[str, dict[str, Any]]:
    libraries: dict[str, dict[str, Any]] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if " => " not in line:
            continue
        name, remainder = line.split(" => ", 1)
        if not name.startswith(LOCAL_LIBRARY_PREFIXES):
            continue
        rendered_path = remainder.split(" (", 1)[0].strip()
        if rendered_path == "not found":
            raise EvaluationError(f"runtime library is not found: {name}")
        path = Path(rendered_path).resolve()
        if path.parent != expected_dir:
            raise EvaluationError(
                f"runtime library pollution: {name} resolved to {path}, expected {expected_dir}"
            )
        if not path.is_file():
            raise EvaluationError(f"runtime library is not a file: {path}")
        libraries[name] = {"path": str(path), "sha256": sha256_file(path)}
    for required in REQUIRED_RUNTIME_LIBRARIES:
        if not any(name.startswith(required) for name in libraries):
            raise EvaluationError(f"runtime library audit is missing {required}")
    return libraries


def audit_runtime_libraries(binary: Path) -> dict[str, Any]:
    if binary.read_bytes()[:4] != b"\x7fELF":
        return {"status": "not-elf-test-fixture", "libraries": {}}
    result = subprocess.run(
        ["ldd", str(binary)],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        shell=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise EvaluationError(f"ldd failed for {binary}: {result.stderr.strip()[-1000:]}")
    libraries = parse_runtime_libraries(result.stdout, expected_dir=binary.parent.resolve())
    return {"status": "verified", "libraries": libraries}


def audit_toolchain(benchmark: Path, cli: Path, perplexity: Path) -> dict[str, Any]:
    return {
        "benchmark": audit_runtime_libraries(benchmark),
        "cli": audit_runtime_libraries(cli),
        "perplexity": audit_runtime_libraries(perplexity),
    }


def normalize_runtime_args(values: list[str], *, arm: str) -> tuple[str, ...]:
    """Validate and canonicalize the narrow per-arm llama.cpp runtime controls."""

    parsed: dict[str, str] = {}
    index = 0
    while index < len(values):
        token = values[index]
        if "=" in token:
            option, value = token.split("=", 1)
            index += 1
        else:
            option = token
            if index + 1 >= len(values):
                raise EvaluationError(f"{arm} runtime argument {option!r} has no value")
            value = values[index + 1]
            index += 2
        canonical = RUNTIME_ARG_ALIASES.get(option)
        if canonical is None:
            raise EvaluationError(
                f"{arm} runtime argument {option!r} is not allowed; model, device, "
                "offload, input, and output overrides are forbidden"
            )
        if value not in RUNTIME_ARG_VALUES[canonical]:
            allowed = ", ".join(sorted(RUNTIME_ARG_VALUES[canonical]))
            raise EvaluationError(
                f"{arm} runtime argument {canonical} has invalid value {value!r}; "
                f"allowed: {allowed}"
            )
        if canonical in parsed:
            raise EvaluationError(f"{arm} runtime argument {canonical} is duplicated")
        parsed[canonical] = value
    return tuple(
        part
        for option in RUNTIME_ARG_ORDER
        if option in parsed
        for part in (option, parsed[option])
    )


def load_math_items(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(f"invalid math JSONL line {line_number}") from error
        if not isinstance(raw, dict):
            raise EvaluationError(f"math item {line_number} is not an object")
        choices = raw.get("choices")
        answer = raw.get("answer_index")
        if (
            not isinstance(raw.get("id"), str)
            or not isinstance(raw.get("subject"), str)
            or raw["subject"] not in SUBJECTS
            or not isinstance(raw.get("question"), str)
            or not raw["question"]
            or not isinstance(choices, list)
            or len(choices) != 4
            or any(not isinstance(choice, str) or not choice for choice in choices)
            or not isinstance(answer, int)
            or isinstance(answer, bool)
            or not 0 <= answer < 4
        ):
            raise EvaluationError(f"math item {line_number} has an invalid shape")
        items.append(raw)
    if not items:
        raise EvaluationError("math JSONL contains no items")
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise EvaluationError("math JSONL contains duplicate IDs")
    return items


def _math_item_rank(item_id: str, *, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{item_id}".encode()).digest()


def select_greedy_canary(
    items: list[dict[str, Any]], count: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a stable canary from the complete fixture, independent of math limits."""

    if count <= 0:
        raise EvaluationError("greedy count must be positive")
    if count > len(items):
        raise EvaluationError("greedy count exceeds the complete math fixture")
    selected = sorted(
        items,
        key=lambda item: (
            _math_item_rank(item["id"], seed=GREEDY_CANARY_SEED),
            item["id"],
        ),
    )[:count]
    ids = [item["id"] for item in selected]
    answers = ["ABCD"[item["answer_index"]] for item in selected]
    pairs = [
        {"id": item_id, "answer": answer}
        for item_id, answer in zip(ids, answers, strict=True)
    ]
    return selected, {
        "mode": "fixture_wide_sha256_rank_v1",
        "seed": GREEDY_CANARY_SEED,
        "rank_input": "utf8(str(seed) + NUL + item_id)",
        "source_total": len(items),
        "total": len(selected),
        "ids": ids,
        "ids_sha256": canonical_sha256(ids),
        "answers": answers,
        "answers_sha256": canonical_sha256(answers),
        "canary_sha256": canonical_sha256(pairs),
    }


def select_math_items(
    items: list[dict[str, Any]], math_limit: int | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the reviewed, deterministic 100-question math subset when requested."""

    if math_limit is None:
        selected = list(items)
        return selected, {
            "mode": "fixture_order_full",
            "source_total": len(items),
            "selected_total": len(selected),
            "selected_ids_sha256": canonical_sha256(
                [item["id"] for item in selected]
            ),
        }
    if math_limit != sum(MATH_100_QUOTAS.values()):
        raise EvaluationError("--math-limit currently supports only 100")

    selected: list[dict[str, Any]] = []
    for subject, quota in MATH_100_QUOTAS.items():
        available = [item for item in items if item["subject"] == subject]
        if len(available) < quota:
            raise EvaluationError(
                f"math fixture has {len(available)} {subject} items, needs {quota}"
            )
        available.sort(
            key=lambda item: (_math_item_rank(item["id"], seed=MATH_SELECTION_SEED), item["id"])
        )
        selected.extend(available[:quota])
    selected.sort(
        key=lambda item: (_math_item_rank(item["id"], seed=MATH_SELECTION_SEED), item["id"])
    )
    selected_ids = [item["id"] for item in selected]
    return selected, {
        "mode": "stratified_sha256_rank_v1",
        "seed": MATH_SELECTION_SEED,
        "rank_input": "utf8(str(seed) + NUL + item_id)",
        "source_total": len(items),
        "selected_total": len(selected),
        "quotas": dict(MATH_100_QUOTAS),
        "selected_ids_sha256": canonical_sha256(selected_ids),
    }


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def serialize_multiple_choice(items: list[dict[str, Any]]) -> bytes:
    """Serialize llama-perplexity's documented multiple-choice input format."""

    tasks: list[bytes] = []
    for item in items:
        labels = "ABCD"
        choices_text = "\n".join(
            f"{labels[index]}. {choice}" for index, choice in enumerate(item["choices"])
        )
        question = f"Question: {item['question']}\n{choices_text}\nAnswer:"
        task = bytearray(_pack_string(question))
        task.extend(struct.pack("<I", 4))
        for label in labels:
            task.extend(_pack_string(label))
        task.extend(
            struct.pack("<4i", *(int(index == item["answer_index"]) for index in range(4)))
        )
        # mc2 is unused, but the llama.cpp format requires an empty second answer set.
        task.extend(struct.pack("<I", 0))
        tasks.append(bytes(task))

    header_size = 4 + 4 * len(tasks)
    positions: list[int] = []
    position = header_size
    for task in tasks:
        positions.append(position)
        position += len(task)
    return (
        struct.pack("<I", len(tasks))
        + struct.pack(f"<{len(positions)}I", *positions)
        + b"".join(tasks)
    )


def run_command(
    argv: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
    max_output_bytes: int = 16 * 1024 * 1024,
) -> tuple[str, str, float]:
    started = time.monotonic()
    if max_output_bytes <= 0:
        raise EvaluationError("command output limit must be positive")
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        start_new_session=True,
    )
    streams = selectors.DefaultSelector()
    assert process.stdout is not None and process.stderr is not None
    streams.register(process.stdout, selectors.EVENT_READ, "stdout")
    streams.register(process.stderr, selectors.EVENT_READ, "stderr")
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    failure: str | None = None
    deadline = started + timeout_seconds
    while streams.get_map():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failure = f"command timed out after {timeout_seconds:g}s: {argv[0]}"
            break
        for key, _ in streams.select(min(remaining, 0.25)):
            chunk = os.read(key.fileobj.fileno(), 64 * 1024)
            if not chunk:
                streams.unregister(key.fileobj)
                continue
            captured[key.data].extend(chunk)
            if sum(len(value) for value in captured.values()) > max_output_bytes:
                failure = (
                    f"command output exceeded {max_output_bytes} bytes: {argv[0]}"
                )
                break
        if failure is not None:
            break
    if failure is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    else:
        process.wait(timeout=5)
    stdout = captured["stdout"].decode("utf-8", errors="replace")
    stderr = captured["stderr"].decode("utf-8", errors="replace")
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    if failure is not None:
        raise EvaluationError(failure)
    if process.returncode != 0:
        tail = stderr.strip()[-2000:]
        raise EvaluationError(
            f"command failed ({process.returncode}): {argv[0]}\n{tail}"
        )
    return stdout, stderr, time.monotonic() - started


def _parse_metric(
    pattern: re.Pattern[str], stdout: str, stderr: str, *, label: str
) -> tuple[float, float]:
    matches = pattern.findall(f"{stdout}\n{stderr}")
    if len(matches) != 1:
        raise EvaluationError(f"{label} output must contain exactly one final metric")
    value, uncertainty = (float(part) for part in matches[0])
    if not math.isfinite(value) or value < 0:
        raise EvaluationError(f"{label} value is invalid")
    return value, uncertainty


def parse_math_correctness(
    stdout: str,
    stderr: str,
    *,
    total: int,
    final_percent: float,
) -> tuple[bool, ...]:
    """Recover each task result from llama-perplexity's cumulative accuracy log."""

    if total <= 0:
        raise EvaluationError("math total must be positive")
    matches = MATH_PROGRESS_PATTERN.findall(f"{stdout}\n{stderr}")
    if len(matches) != total:
        raise EvaluationError(
            f"math output contains {len(matches)} progress rows; expected {total}"
        )
    correctness: list[bool] = []
    previous_correct = 0
    for expected_index, (raw_index, raw_percent) in enumerate(matches, 1):
        index = int(raw_index)
        percent = float(raw_percent)
        if index != expected_index:
            raise EvaluationError(
                "math progress rows must be contiguous and ordered from one"
            )
        if not math.isfinite(percent) or not 0 <= percent <= 100:
            raise EvaluationError("math progress accuracy is outside [0, 100]")
        scaled_correct = percent * index / 100.0
        cumulative_correct = int(math.floor(scaled_correct + 0.5))
        rendered_percent = 100.0 * cumulative_correct / index
        if abs(percent - rendered_percent) > 1e-6:
            raise EvaluationError(
                f"math progress row {index} cannot represent an integer correct count"
            )
        delta = cumulative_correct - previous_correct
        if delta not in {0, 1}:
            raise EvaluationError(f"math progress row {index} has an invalid increment")
        correctness.append(bool(delta))
        previous_correct = cumulative_correct

    reconstructed_percent = 100.0 * previous_correct / total
    # The final line is rendered to four decimals from a float in llama.cpp.
    if abs(final_percent - reconstructed_percent) > 1e-4:
        raise EvaluationError(
            "math final accuracy does not match the per-task progress evidence"
        )
    return tuple(correctness)


def exact_two_sided_mcnemar_p(baseline_only: int, candidate_only: int) -> float:
    """Return the exact two-sided binomial McNemar p-value for discordant pairs."""

    if baseline_only < 0 or candidate_only < 0:
        raise EvaluationError("McNemar discordant counts cannot be negative")
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, candidate_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (
        1 << discordant
    )
    return min(1.0, 2.0 * probability)


def compare_math_correctness(
    baseline_bits: str, candidate_bits: str
) -> dict[str, Any]:
    if (
        not baseline_bits
        or len(baseline_bits) != len(candidate_bits)
        or any(bit not in "01" for bit in baseline_bits + candidate_bits)
    ):
        raise EvaluationError("paired math correctness vectors are invalid")
    both_correct = baseline_only = candidate_only = both_wrong = 0
    for baseline_bit, candidate_bit in zip(
        baseline_bits, candidate_bits, strict=True
    ):
        if baseline_bit == candidate_bit == "1":
            both_correct += 1
        elif baseline_bit == "1":
            baseline_only += 1
        elif candidate_bit == "1":
            candidate_only += 1
        else:
            both_wrong += 1
    return {
        "total": len(baseline_bits),
        "both_correct": both_correct,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "both_wrong": both_wrong,
        "candidate_net_correct": candidate_only - baseline_only,
        "mcnemar_exact_two_sided_p": exact_two_sided_mcnemar_p(
            baseline_only, candidate_only
        ),
    }


def corrected_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    math_comparison: dict[str, Any],
    math_selection: dict[str, Any],
) -> dict[str, Any]:
    """Apply the fixed, full-fixture Q8 acceptance policy deterministically."""

    full_math = (
        math_comparison["total"] == FULL_MATH_TOTAL
        and math_selection.get("mode") == "fixture_order_full"
        and math_selection.get("source_total") == FULL_MATH_TOTAL
        and math_selection.get("selected_total") == FULL_MATH_TOTAL
    )
    ppl_regression = candidate["perplexity"] / baseline["perplexity"] - 1.0
    math_drop = baseline["math_accuracy"] - candidate["math_accuracy"]
    candidate_net_regression = math_comparison["baseline_only"] > math_comparison[
        "candidate_only"
    ]
    paired_regression_allowed = (
        not candidate_net_regression
        or math_comparison["mcnemar_exact_two_sided_p"] > MCNEMAR_ALPHA
    )
    checks = {
        "full_math_848": full_math,
        "perplexity_regression_within_0_5_percent": (
            ppl_regression <= MAX_PPL_REGRESSION_FRACTION + 1e-12
        ),
        "math_drop_within_1_percentage_point": (
            math_drop <= MAX_MATH_DROP_FRACTION + 1e-12
        ),
        "net_math_regression_not_significant": paired_regression_allowed,
        "greedy_accuracy_not_lower": (
            candidate["greedy_accuracy"] >= baseline["greedy_accuracy"]
        ),
    }
    applicable = full_math
    passed = applicable and all(checks.values())
    return {
        "schema": "gpuopt.q8-runtime-quality-corrected-gate.v1",
        "applicable": applicable,
        "outcome": "PASS" if passed else ("FAIL" if applicable else "NOT_APPLICABLE"),
        "thresholds": {
            "required_math_total": FULL_MATH_TOTAL,
            "max_perplexity_regression_fraction": MAX_PPL_REGRESSION_FRACTION,
            "max_math_accuracy_drop_fraction": MAX_MATH_DROP_FRACTION,
            "mcnemar_alpha": MCNEMAR_ALPHA,
            "greedy_accuracy_drop_allowed": False,
        },
        "measurements": {
            "perplexity_regression_fraction": ppl_regression,
            "math_accuracy_drop_fraction": math_drop,
            "candidate_net_math_correct": math_comparison["candidate_net_correct"],
            "mcnemar_exact_two_sided_p": math_comparison[
                "mcnemar_exact_two_sided_p"
            ],
            "baseline_greedy_accuracy": baseline["greedy_accuracy"],
            "candidate_greedy_accuracy": candidate["greedy_accuracy"],
        },
        "checks": checks,
    }


def greedy_prompt(items: list[dict[str, Any]], count: int) -> tuple[str, str, int]:
    selected = items[:count]
    labels = "ABCD"
    sections = []
    for index, item in enumerate(selected, 1):
        choices = "\n".join(
            f"{labels[choice_index]}. {choice}"
            for choice_index, choice in enumerate(item["choices"])
        )
        sections.append(f"{index}. {item['question']}\n{choices}")
    prompt = (
        "Answer each independent multiple-choice math question. Return only the answer "
        "letters in order, separated by commas.\n\n"
        + "\n\n".join(sections)
        + "\n\nAnswers:"
    )
    grammar = 'root ::= answer' + ' "," answer' * (count - 1) + '\nanswer ::= "A" | "B" | "C" | "D"'
    return prompt, grammar, count


def evaluate_runtime(
    *,
    name: str,
    benchmark_binary: Path,
    cli: Path,
    perplexity: Path,
    runtime_libraries: dict[str, Any],
    runtime_args: tuple[str, ...],
    model: Path,
    ppl_file: Path,
    math_binary: Path,
    math_total: int,
    greedy_items: list[dict[str, Any]],
    artifact_dir: Path,
    ngl: int,
    threads: int,
    context_size: int,
    batch_size: int,
    ubatch_size: int,
    ppl_chunks: int,
    greedy_count: int,
    timeout_seconds: float,
    greedy_context_size: int | None = None,
    general_binary: Path | None = None,
    general_cases: list[Any] | None = None,
) -> dict[str, Any]:
    runtime_dir = artifact_dir / name
    runtime_dir.mkdir(parents=True, exist_ok=True)
    prompt, grammar, answer_count = greedy_prompt(greedy_items, greedy_count)
    prompt_path = runtime_dir / "greedy-prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    common = [
        "-m",
        str(model),
        "-ngl",
        str(ngl),
        "-t",
        str(threads),
        "-c",
        str(context_size),
        "-b",
        str(batch_size),
        "-ub",
        str(ubatch_size),
        *runtime_args,
    ]
    greedy_common = list(common)
    greedy_common[greedy_common.index("-c") + 1] = str(
        greedy_context_size or context_size
    )
    cli_argv = [
        str(cli),
        *greedy_common,
        "-f",
        str(prompt_path),
        "-n",
        str(answer_count * 3 + 4),
        "--temp",
        "0",
        "--seed",
        str(GREEDY_GENERATION_SEED),
        "--grammar",
        grammar,
        "--no-conversation",
        "--single-turn",
        "--no-display-prompt",
        "--no-show-timings",
        "--simple-io",
        "--color",
        "off",
        "--log-disable",
    ]
    greedy_stdout, greedy_stderr, greedy_seconds = run_command(
        cli_argv,
        cwd=runtime_dir,
        timeout_seconds=timeout_seconds,
        stdout_path=runtime_dir / "greedy.stdout",
        stderr_path=runtime_dir / "greedy.stderr",
    )
    answer_lines = [
        line.strip()
        for line in greedy_stdout.splitlines()
        if ANSWER_PATTERN.fullmatch(line.strip())
        and len(line.strip().split(",")) == answer_count
    ]
    if len(answer_lines) != 1:
        raise EvaluationError(f"{name} greedy output is not the required answer sequence")
    greedy_answers = answer_lines[0].split(",")
    expected_greedy_answers = [
        "ABCD"[item["answer_index"]] for item in greedy_items[:greedy_count]
    ]
    greedy_correctness = [
        actual == expected
        for actual, expected in zip(
            greedy_answers, expected_greedy_answers, strict=True
        )
    ]
    greedy_correct = sum(greedy_correctness)

    ppl_argv = [
        str(perplexity),
        *common,
        "-f",
        str(ppl_file),
        "--chunks",
        str(ppl_chunks),
    ]
    ppl_stdout, ppl_stderr, ppl_seconds = run_command(
        ppl_argv,
        cwd=runtime_dir,
        timeout_seconds=timeout_seconds,
        stdout_path=runtime_dir / "perplexity.stdout",
        stderr_path=runtime_dir / "perplexity.stderr",
    )
    ppl, ppl_uncertainty = _parse_metric(
        PPL_PATTERN, ppl_stdout, ppl_stderr, label=f"{name} perplexity"
    )
    if ppl <= 0:
        raise EvaluationError(f"{name} perplexity must be positive")

    math_argv = [
        str(perplexity),
        *common,
        "-np",
        "32",
        "-f",
        str(math_binary),
        "--multiple-choice",
    ]
    math_stdout, math_stderr, math_seconds = run_command(
        math_argv,
        cwd=runtime_dir,
        timeout_seconds=timeout_seconds,
        stdout_path=runtime_dir / "math.stdout",
        stderr_path=runtime_dir / "math.stderr",
    )
    math_percent, math_uncertainty_percent = _parse_metric(
        MATH_PATTERN, math_stdout, math_stderr, label=f"{name} math"
    )
    math_correctness = parse_math_correctness(
        math_stdout,
        math_stderr,
        total=math_total,
        final_percent=math_percent,
    )
    math_correctness_bits = "".join("1" if correct else "0" for correct in math_correctness)
    math_correct = sum(math_correctness)
    math_accuracy = math_correct / math_total

    general: dict[str, Any] | None = None
    general_seconds: float | None = None
    general_argv: list[str] | None = None
    general_stderr = ""
    if general_binary is not None:
        if not general_cases:
            raise EvaluationError("general suite binary has no case coordinates")
        general_argv = [
            str(perplexity),
            *common,
            "-np",
            "32",
            "-f",
            str(general_binary),
            "--multiple-choice",
        ]
        general_stdout, general_stderr, general_seconds = run_command(
            general_argv,
            cwd=runtime_dir,
            timeout_seconds=timeout_seconds,
            stdout_path=runtime_dir / "general.stdout",
            stderr_path=runtime_dir / "general.stderr",
        )
        general_percent, general_uncertainty = _parse_metric(
            MATH_PATTERN,
            general_stdout,
            general_stderr,
            label=f"{name} general suite",
        )
        correctness = parse_math_correctness(
            general_stdout,
            general_stderr,
            total=len(general_cases),
            final_percent=general_percent,
        )
        grouped: dict[str, list[bool]] = {}
        item_results: list[dict[str, Any]] = []
        for case, correct in zip(general_cases, correctness, strict=True):
            group = str(case.group)
            grouped.setdefault(group, []).append(correct)
            item_results.append({"case_id": case.case_id, "correct": correct})
        general = {
            "accuracy": sum(correctness) / len(correctness),
            "correct": sum(correctness),
            "total": len(correctness),
            "uncertainty": general_uncertainty / 100.0,
            "correctness_bits": "".join("1" if value else "0" for value in correctness),
            "items": item_results,
            "groups": {
                group: {
                    "correct": sum(values),
                    "total": len(values),
                    "accuracy": sum(values) / len(values),
                }
                for group, values in sorted(grouped.items())
            },
        }

    accuracies = {
        "math_accuracy": math_accuracy,
        "greedy_accuracy": greedy_correct / answer_count,
    }
    if general is not None:
        accuracies["general_accuracy"] = general["accuracy"]
        accuracies.update(
            {
                f"{group}_accuracy": values["accuracy"]
                for group, values in general["groups"].items()
            }
        )

    result = {
        "benchmark_binary_sha256": sha256_file(benchmark_binary),
        "binary_sha256": sha256_file(cli),
        "perplexity_binary_sha256": sha256_file(perplexity),
        "runtime_libraries": runtime_libraries,
        "runtime_args": list(runtime_args),
        "perplexity": ppl,
        "perplexity_uncertainty": ppl_uncertainty,
        "math_accuracy": math_accuracy,
        "math_accuracy_uncertainty": math_uncertainty_percent / 100.0,
        "math_correct": math_correct,
        "math_total": math_total,
        "math_correctness_bits": math_correctness_bits,
        "math_correctness_sha256": hashlib.sha256(
            math_correctness_bits.encode("ascii")
        ).hexdigest(),
        "accuracies": accuracies,
        "greedy_tokens": greedy_answers,
        "greedy_tokens_sha256": canonical_sha256(greedy_answers),
        "greedy_correct": greedy_correct,
        "greedy_total": answer_count,
        "greedy_accuracy": greedy_correct / answer_count,
        "greedy_correctness_bits": "".join(
            "1" if correct else "0" for correct in greedy_correctness
        ),
        "seconds": {
            "greedy": greedy_seconds,
            "perplexity": ppl_seconds,
            "math": math_seconds,
        },
        "commands": {
            "greedy": cli_argv,
            "perplexity": ppl_argv,
            "math": math_argv,
        },
        "stderr_nonempty": {
            "greedy": bool(greedy_stderr),
            "perplexity": bool(ppl_stderr),
            "math": bool(math_stderr),
        },
    }
    if general is not None:
        result["general"] = general
        result["seconds"]["general"] = general_seconds
        result["commands"]["general"] = general_argv
        result["stderr_nonempty"]["general"] = bool(general_stderr)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="baseline llama.cpp bin dir or llama-cli")
    parser.add_argument(
        "--candidate", required=True, help="candidate llama.cpp bin dir or llama-cli"
    )
    parser.add_argument("--model", help="one model used by both runtime arms")
    parser.add_argument("--baseline-model")
    parser.add_argument("--candidate-model")
    parser.add_argument("--fixture-manifest", required=True)
    parser.add_argument(
        "--general-suite-dir",
        help="frozen gpuopt general-100.v1 directory (optional)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-baseline-model-sha256")
    parser.add_argument("--expected-candidate-model-sha256")
    parser.add_argument(
        "--baseline-runtime-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="repeat one baseline runtime token; supports -fa, -ctk, and -ctv",
    )
    parser.add_argument(
        "--candidate-runtime-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="repeat one candidate runtime token; supports -fa, -ctk, and -ctv",
    )
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--ppl-chunks", type=int, default=32)
    parser.add_argument("--greedy-count", type=int, default=8)
    parser.add_argument(
        "--math-limit",
        type=int,
        help="use the reviewed deterministic stratified subset (currently only 100)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output).resolve()
    failure: dict[str, Any] = {"schema": SCHEMA, "status": "failed"}
    try:
        if args.ngl < 0 or args.threads <= 0 or args.context_size < 32:
            raise EvaluationError("ngl, threads, or context size is invalid")
        if args.batch_size <= 0 or args.ubatch_size <= 0 or args.ppl_chunks <= 0:
            raise EvaluationError("batch, ubatch, and PPL chunks must be positive")
        if args.greedy_count <= 0 or args.timeout_seconds <= 0:
            raise EvaluationError("greedy count and timeout must be positive")
        baseline_runtime_args = normalize_runtime_args(
            args.baseline_runtime_arg, arm="baseline"
        )
        candidate_runtime_args = normalize_runtime_args(
            args.candidate_runtime_arg, arm="candidate"
        )

        if args.model:
            if args.baseline_model or args.candidate_model:
                raise EvaluationError("--model cannot be combined with per-arm model paths")
            baseline_model_arg = candidate_model_arg = args.model
        else:
            if not args.baseline_model or not args.candidate_model:
                raise EvaluationError(
                    "provide --model or both --baseline-model and --candidate-model"
                )
            baseline_model_arg = args.baseline_model
            candidate_model_arg = args.candidate_model
        baseline_model = require_regular_file(
            Path(baseline_model_arg), label="baseline model"
        )
        candidate_model = require_regular_file(
            Path(candidate_model_arg), label="candidate model"
        )
        baseline_model_sha256 = sha256_file(baseline_model)
        candidate_model_sha256 = sha256_file(candidate_model)
        same_model = baseline_model_sha256 == candidate_model_sha256
        failure["baseline_model_sha256"] = baseline_model_sha256
        failure["candidate_model_sha256"] = candidate_model_sha256
        if same_model:
            failure["model_sha256"] = baseline_model_sha256
        expected_baseline = (
            args.expected_baseline_model_sha256 or args.expected_model_sha256
        )
        expected_candidate = (
            args.expected_candidate_model_sha256 or args.expected_model_sha256
        )
        if expected_baseline and baseline_model_sha256 != expected_baseline:
            raise EvaluationError(
                "baseline model SHA-256 does not match the expected identity"
            )
        if expected_candidate and candidate_model_sha256 != expected_candidate:
            raise EvaluationError(
                "candidate model SHA-256 does not match the expected identity"
            )

        manifest_path = require_regular_file(
            Path(args.fixture_manifest), label="quality fixture manifest"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema") != MATH_SCHEMA:
            raise EvaluationError("quality fixture manifest has an unsupported schema")
        math_meta = manifest.get("math")
        ppl_meta = manifest.get("perplexity")
        if not isinstance(math_meta, dict) or not isinstance(ppl_meta, dict):
            raise EvaluationError("quality fixture manifest is incomplete")
        math_path = require_regular_file(
            manifest_path.parent / str(math_meta.get("path")), label="math fixture"
        )
        ppl_path = require_regular_file(
            manifest_path.parent / str(ppl_meta.get("path")), label="perplexity fixture"
        )
        if sha256_file(math_path) != math_meta.get("sha256"):
            raise EvaluationError("math fixture SHA-256 does not match its manifest")
        if sha256_file(ppl_path) != ppl_meta.get("sha256"):
            raise EvaluationError("perplexity fixture SHA-256 does not match its manifest")

        source_items = load_math_items(math_path)
        if len(source_items) != math_meta.get("total"):
            raise EvaluationError("math fixture item count does not match its manifest")
        greedy_items, greedy_canary = select_greedy_canary(
            source_items, args.greedy_count
        )
        items, math_selection = select_math_items(source_items, args.math_limit)

        baseline_benchmark = require_regular_file(
            Path(args.baseline), label="baseline llama-bench", executable=True
        )
        candidate_benchmark = require_regular_file(
            Path(args.candidate), label="candidate llama-bench", executable=True
        )
        baseline_cli, baseline_ppl = resolve_tools(baseline_benchmark, label="baseline")
        candidate_cli, candidate_ppl = resolve_tools(candidate_benchmark, label="candidate")
        baseline_libraries = audit_toolchain(
            baseline_benchmark, baseline_cli, baseline_ppl
        )
        candidate_libraries = audit_toolchain(
            candidate_benchmark, candidate_cli, candidate_ppl
        )
        artifact_dir = output.parent / f"{output.stem}.raw"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        math_binary = artifact_dir / "mmlu-math.bin"
        math_binary.write_bytes(serialize_multiple_choice(items))

        general_binary: Path | None = None
        general_cases: list[Any] | None = None
        general_coordinate: dict[str, Any] | None = None
        if args.general_suite_dir:
            try:
                from amd_inference_opt.eval_suites import (
                    GENERAL_100_SUITE_ID,
                    load_frozen_suite,
                    multiple_choice_scorer_protocol_sha256,
                    quality_suite_coordinate,
                    serialize_llama_multiple_choice,
                )
            except ImportError as error:
                raise EvaluationError(
                    "general suite evaluation requires amd_inference_opt to be installed"
                ) from error
            general_manifest, loaded_cases = load_frozen_suite(
                Path(args.general_suite_dir)
            )
            if general_manifest.suite_id != GENERAL_100_SUITE_ID:
                raise EvaluationError("general suite has an unsupported suite_id")
            general_cases = list(loaded_cases)
            coordinate = quality_suite_coordinate(Path(args.general_suite_dir))
            general_binary = artifact_dir / "general-100.bin"
            general_binary.write_bytes(
                serialize_llama_multiple_choice(general_cases)
            )
            general_coordinate = {
                "suite_id": general_manifest.suite_id,
                "manifest_sha256": coordinate.manifest_sha256,
                "selected_ids_sha256": (
                    general_manifest.selection.selected_ids_sha256
                ),
                "total": general_manifest.total_cases,
                "group_counts": general_manifest.group_counts,
                "scorer_protocol_sha256": (
                    multiple_choice_scorer_protocol_sha256()
                ),
            }

        protocol = {
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "baseline_model_sha256": baseline_model_sha256,
            "candidate_model_sha256": candidate_model_sha256,
            "math_fixture_sha256": math_meta["sha256"],
            "math_total": len(items),
            "math_selection": math_selection,
            "greedy_canary": greedy_canary,
            "perplexity_fixture_sha256": ppl_meta["sha256"],
            "coordinates": {
                "ngl": args.ngl,
                "threads": args.threads,
                "context_size": args.context_size,
                "batch_size": args.batch_size,
                "ubatch_size": args.ubatch_size,
                "ppl_chunks": args.ppl_chunks,
                "greedy_count": args.greedy_count,
                "greedy_generation_seed": GREEDY_GENERATION_SEED,
                "sampling": "greedy",
                "runtime_args": {
                    "baseline": list(baseline_runtime_args),
                    "candidate": list(candidate_runtime_args),
                },
            },
            "limitations": manifest.get("limitations", []),
        }
        if general_coordinate is not None:
            protocol["general_suite"] = general_coordinate
        protocol_hash = canonical_sha256(protocol)
        failure["protocol_hash"] = protocol_hash

        baseline = evaluate_runtime(
            name="baseline",
            benchmark_binary=baseline_benchmark,
            cli=baseline_cli,
            perplexity=baseline_ppl,
            runtime_libraries=baseline_libraries,
            runtime_args=baseline_runtime_args,
            model=baseline_model,
            ppl_file=ppl_path,
            math_binary=math_binary,
            math_total=len(items),
            greedy_items=greedy_items,
            artifact_dir=artifact_dir,
            ngl=args.ngl,
            threads=args.threads,
            context_size=args.context_size,
            batch_size=args.batch_size,
            ubatch_size=args.ubatch_size,
            ppl_chunks=args.ppl_chunks,
            greedy_count=args.greedy_count,
            timeout_seconds=args.timeout_seconds,
            general_binary=general_binary,
            general_cases=general_cases,
        )
        candidate = evaluate_runtime(
            name="candidate",
            benchmark_binary=candidate_benchmark,
            cli=candidate_cli,
            perplexity=candidate_ppl,
            runtime_libraries=candidate_libraries,
            runtime_args=candidate_runtime_args,
            model=candidate_model,
            ppl_file=ppl_path,
            math_binary=math_binary,
            math_total=len(items),
            greedy_items=greedy_items,
            artifact_dir=artifact_dir,
            ngl=args.ngl,
            threads=args.threads,
            context_size=args.context_size,
            batch_size=args.batch_size,
            ubatch_size=args.ubatch_size,
            ppl_chunks=args.ppl_chunks,
            greedy_count=args.greedy_count,
            timeout_seconds=args.timeout_seconds,
            general_binary=general_binary,
            general_cases=general_cases,
        )
        math_comparison = compare_math_correctness(
            baseline["math_correctness_bits"],
            candidate["math_correctness_bits"],
        )
        comparison = {
            "math": math_comparison,
            "greedy_tokens_equal": (
                baseline["greedy_tokens_sha256"]
                == candidate["greedy_tokens_sha256"]
            ),
            "greedy_accuracy_delta": (
                candidate["greedy_accuracy"] - baseline["greedy_accuracy"]
            ),
            "perplexity_regression_fraction": (
                candidate["perplexity"] / baseline["perplexity"] - 1.0
            ),
            "math_accuracy_delta": (
                candidate["math_accuracy"] - baseline["math_accuracy"]
            ),
        }
        if general_coordinate is not None:
            comparison["general"] = compare_math_correctness(
                baseline["general"]["correctness_bits"],
                candidate["general"]["correctness_bits"],
            )
            comparison["general_accuracy_delta"] = (
                candidate["general"]["accuracy"]
                - baseline["general"]["accuracy"]
            )
        gate = corrected_gate(
            baseline,
            candidate,
            math_comparison,
            math_selection,
        )
        document = {
            "schema": SCHEMA,
            "status": "complete",
            "protocol_hash": protocol_hash,
            "baseline_model_sha256": baseline_model_sha256,
            "candidate_model_sha256": candidate_model_sha256,
            "protocol": protocol,
            "baseline": baseline,
            "candidate": candidate,
            "comparison": comparison,
            "corrected_gate": gate,
            # Kept as a legacy diagnostic. It is not an acceptance condition.
            "greedy_tokens_equal": comparison["greedy_tokens_equal"],
            "artifact_directory": str(artifact_dir),
        }
        if same_model:
            document["model_sha256"] = baseline_model_sha256
        write_json_atomic(output, document)
        print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (EvaluationError, OSError, json.JSONDecodeError) as error:
        failure["error"] = str(error)
        write_json_atomic(output, failure)
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
