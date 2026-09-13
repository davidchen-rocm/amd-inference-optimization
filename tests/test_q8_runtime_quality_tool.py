import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from amd_inference_opt.eval_suites import DatasetSourceV1, prepare_general_100

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools/q8_runtime_quality_eval.py"


def _evaluator_module():
    spec = importlib.util.spec_from_file_location("q8_runtime_quality_eval", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_runtime(
    root: Path,
    *,
    greedy: str,
    perplexity: float,
    accuracy_percent: float,
    math_correctness: tuple[bool, ...] | None = None,
    fail_cli: bool = False,
) -> Path:
    root.mkdir()
    benchmark = root / "llama-bench"
    benchmark.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli = root / "llama-cli"
    cli.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "assert '--grammar' in sys.argv\n"
        "assert '--no-conversation' in sys.argv\n"
        "assert '--single-turn' in sys.argv\n"
        + ("raise SystemExit(7)\n" if fail_cli else f"print({greedy!r})\n"),
        encoding="utf-8",
    )
    ppl = root / "llama-perplexity"
    ppl.write_text(
        "#!/usr/bin/env python3\n"
        "import struct\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"configured = {list(math_correctness) if math_correctness is not None else None!r}\n"
        "if '--multiple-choice' in sys.argv:\n"
        "    fixture = Path(sys.argv[sys.argv.index('-f') + 1])\n"
        "    total = struct.unpack('<I', fixture.read_bytes()[:4])[0]\n"
        f"    target_correct = round({accuracy_percent!r} * total / 100.0)\n"
        "    correctness = configured or "
        "([True] * target_correct + [False] * (total - target_correct))\n"
        "    assert len(correctness) == total\n"
        "    correct = 0\n"
        "    for index, item_correct in enumerate(correctness, 1):\n"
        "        correct += int(item_correct)\n"
        "        print(f'{index}\\t{100.0 * correct / index:.8f}', file=sys.stderr)\n"
        "    print(f'Final result: {100.0 * correct / total:.4f} +/- 1.0000', file=sys.stderr)\n"
        "else:\n"
        f"    print('Final estimate: PPL = {perplexity:.4f} +/- 0.0100', file=sys.stderr)\n",
        encoding="utf-8",
    )
    benchmark.chmod(0o755)
    cli.chmod(0o755)
    ppl.chmod(0o755)
    return benchmark


def _write_fixture(root: Path) -> Path:
    root.mkdir()
    math_path = root / "math.jsonl"
    records = []
    for index in range(8):
        records.append(
            {
                "id": f"abstract_algebra-{index}",
                "subject": "abstract_algebra",
                "question": f"What is {index} + 1?",
                "choices": [str(index + 1), str(index + 2), str(index + 3), str(index + 4)],
                "answer_index": 0,
            }
        )
    math_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    ppl_path = root / "ppl.txt"
    ppl_path.write_text(("a reproducible perplexity corpus " * 200) + "\n", encoding="utf-8")
    manifest = {
        "schema": "gpuopt.q8-runtime-quality-fixture.v1",
        "math": {"path": math_path.name, "sha256": _sha256(math_path), "total": 8},
        "perplexity": {"path": ppl_path.name, "sha256": _sha256(ppl_path)},
        "limitations": ["test fixture"],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _general_rows(source: DatasetSourceV1) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(30):
        if source.source_id == "mmlu_general":
            row = {
                "id": f"mmlu-{index}",
                "question": f"MMLU {index}?",
                "choices": ["a", "b", "c", "d"],
                "answer": index % 4,
            }
        elif source.source_id == "arc_challenge":
            row = {
                "id": f"arc-{index}",
                "question": f"ARC {index}?",
                "choices": {
                    "label": ["A", "B", "C", "D"],
                    "text": ["a", "b", "c", "d"],
                },
                "answerKey": "ABCD"[index % 4],
            }
        elif source.source_id == "hellaswag":
            row = {
                "ind": index,
                "ctx": f"HellaSwag {index}",
                "endings": ["a", "b", "c", "d"],
                "label": str(index % 4),
            }
        else:
            row = {
                "id": f"wino-{index}",
                "sentence": f"Person {index} chose _.",
                "option1": "Alice",
                "option2": "Bob",
                "answer": str(index % 2 + 1),
            }
        rows.append(row)
    return rows


def _run(
    tmp_path: Path,
    *,
    baseline_greedy: str = "A,B,C,D,A,B,C,D",
    candidate_greedy: str = "A,B,C,D,A,B,C,D",
    fail_candidate: bool = False,
    distinct_candidate_model: bool = False,
    runtime_arguments: tuple[str, ...] = (),
    baseline_math_correctness: tuple[bool, ...] | None = None,
    candidate_math_correctness: tuple[bool, ...] | None = None,
    include_general: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    baseline = _write_runtime(
        tmp_path / "baseline",
        greedy=baseline_greedy,
        perplexity=3.1000,
        accuracy_percent=62.5,
        math_correctness=baseline_math_correctness,
    )
    candidate = _write_runtime(
        tmp_path / "candidate",
        greedy=candidate_greedy,
        perplexity=3.1020,
        accuracy_percent=62.5,
        math_correctness=candidate_math_correctness,
        fail_cli=fail_candidate,
    )
    model = tmp_path / "model-q8.gguf"
    model.write_bytes(b"GGUF-q8-fixture")
    candidate_model = tmp_path / "model-q6.gguf"
    candidate_model.write_bytes(b"GGUF-q6-fixture")
    manifest = _write_fixture(tmp_path / "fixture")
    general = tmp_path / "general-100.v1"
    if include_general:
        prepare_general_100(general, loader=_general_rows)
    output = tmp_path / "quality.json"
    model_arguments = (
        [
            "--baseline-model",
            str(model),
            "--candidate-model",
            str(candidate_model),
        ]
        if distinct_candidate_model
        else ["--model", str(model)]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            *model_arguments,
            *runtime_arguments,
            "--fixture-manifest",
            str(manifest),
            *(["--general-suite-dir", str(general)] if include_general else []),
            "--output",
            str(output),
            "--ppl-chunks",
            "1",
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    return result, output, model


def test_executes_both_runtimes_and_emits_strict_complete_schema(tmp_path: Path) -> None:
    result, output, model = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema"] == "gpuopt.q8-runtime-quality.v1"
    assert document["status"] == "complete"
    assert document["model_sha256"] == _sha256(model)
    assert len(document["protocol_hash"]) == 64
    encoded_protocol = json.dumps(
        document["protocol"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert document["protocol_hash"] == hashlib.sha256(encoded_protocol).hexdigest()
    assert document["baseline"]["perplexity"] == pytest.approx(3.1)
    assert document["candidate"]["perplexity"] == pytest.approx(3.102)
    assert document["baseline"]["math_accuracy"] == pytest.approx(0.625)
    assert document["candidate"]["math_total"] == 8
    assert document["baseline"]["math_correct"] == 5
    assert document["baseline"]["math_correctness_bits"] == "11111000"
    assert len(document["baseline"]["math_correctness_sha256"]) == 64
    assert document["comparison"]["math"] == {
        "total": 8,
        "both_correct": 5,
        "baseline_only": 0,
        "candidate_only": 0,
        "both_wrong": 3,
        "candidate_net_correct": 0,
        "mcnemar_exact_two_sided_p": 1.0,
    }
    assert document["protocol"]["math_selection"]["mode"] == "fixture_order_full"
    assert document["protocol"]["math_selection"]["selected_total"] == 8
    assert document["protocol"]["greedy_canary"]["source_total"] == 8
    assert document["protocol"]["greedy_canary"]["answers"] == ["A"] * 8
    assert len(document["protocol"]["greedy_canary"]["canary_sha256"]) == 64
    assert document["baseline"]["greedy_correct"] == 2
    assert document["baseline"]["greedy_total"] == 8
    assert document["baseline"]["greedy_accuracy"] == 0.25
    assert document["corrected_gate"]["outcome"] == "NOT_APPLICABLE"
    assert document["greedy_tokens_equal"] is True
    assert document["baseline"]["greedy_tokens_sha256"] == document["candidate"][
        "greedy_tokens_sha256"
    ]
    assert (tmp_path / "quality.raw/baseline/math.stderr").is_file()
    assert (tmp_path / "quality.raw/candidate/perplexity.stderr").is_file()


def test_executes_frozen_general_100_and_emits_per_group_accuracies(
    tmp_path: Path,
) -> None:
    result, output, _ = _run(tmp_path, include_general=True)

    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["protocol"]["general_suite"]["total"] == 100
    assert document["baseline"]["general"]["total"] == 100
    assert document["baseline"]["general"]["correct"] == 62
    assert document["baseline"]["accuracies"]["general_accuracy"] == 0.62
    assert set(document["baseline"]["general"]["groups"]) == {
        "arc_challenge",
        "hellaswag",
        "mmlu_general",
        "winogrande",
    }
    assert document["comparison"]["general"]["total"] == 100
    assert (tmp_path / "quality.raw/candidate/general.stderr").is_file()


def test_records_real_greedy_mismatch_instead_of_declaring_correctness(tmp_path: Path) -> None:
    result, output, _ = _run(tmp_path, candidate_greedy="A,B,C,D,A,B,C,A")

    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["greedy_tokens_equal"] is False
    assert document["baseline"]["greedy_tokens_sha256"] != document["candidate"][
        "greedy_tokens_sha256"
    ]


def test_evaluates_distinct_model_representation_for_each_runtime_arm(
    tmp_path: Path,
) -> None:
    result, output, baseline_model = _run(
        tmp_path,
        distinct_candidate_model=True,
    )

    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["baseline_model_sha256"] == _sha256(baseline_model)
    assert document["candidate_model_sha256"] == _sha256(
        tmp_path / "model-q6.gguf"
    )
    assert document["baseline_model_sha256"] != document["candidate_model_sha256"]
    assert "model_sha256" not in document


def test_failed_runtime_emits_failed_schema_without_fake_metrics(tmp_path: Path) -> None:
    result, output, _ = _run(tmp_path, fail_candidate=True)

    assert result.returncode == 1
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema"] == "gpuopt.q8-runtime-quality.v1"
    assert document["status"] == "failed"
    assert "candidate" not in document
    assert "perplexity" not in document
    assert "command failed (7)" in document["error"]


def test_runtime_library_audit_rejects_cross_build_pollution(tmp_path: Path) -> None:
    module = _evaluator_module()
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    for name in ("libllama.so.0", "libggml.so.0", "libggml-hip.so.0"):
        (baseline / name).write_bytes(name.encode())
        (candidate / name).write_bytes(name.encode())
    output = "\n".join(
        [
            f"libllama.so.0 => {candidate / 'libllama.so.0'} (0x1)",
            f"libggml.so.0 => {candidate / 'libggml.so.0'} (0x2)",
            f"libggml-hip.so.0 => {baseline / 'libggml-hip.so.0'} (0x3)",
        ]
    )

    with pytest.raises(module.EvaluationError, match="runtime library pollution"):
        module.parse_runtime_libraries(output, expected_dir=candidate)


def test_command_output_is_bounded_and_partial_evidence_is_preserved(
    tmp_path: Path,
) -> None:
    module = _evaluator_module()
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    with pytest.raises(module.EvaluationError, match="output exceeded"):
        module.run_command(
            [sys.executable, "-c", "print('x' * 200000)"],
            cwd=tmp_path,
            timeout_seconds=10,
            stdout_path=stdout,
            stderr_path=stderr,
            max_output_bytes=4096,
        )

    assert 4096 < stdout.stat().st_size < 100000
    assert stderr.is_file()


def test_math_limit_100_uses_reviewed_stratified_selection() -> None:
    module = _evaluator_module()
    fixture = ROOT / "fixtures/q8-runtime-quality/mmlu-math.jsonl"
    items = module.load_math_items(fixture)

    selected, provenance = module.select_math_items(items, 100)
    repeated, repeated_provenance = module.select_math_items(list(reversed(items)), 100)

    assert [item["id"] for item in selected] == [item["id"] for item in repeated]
    assert provenance == repeated_provenance
    assert Counter(item["subject"] for item in selected) == {
        "abstract_algebra": 12,
        "college_mathematics": 12,
        "high_school_mathematics": 32,
        "elementary_mathematics": 44,
    }
    assert provenance["seed"] == 20260815
    assert provenance["selected_ids_sha256"] == (
        "11670b5040127d9713742b935a743fe932edbe3cc83f53202962e22929365841"
    )


def test_greedy_canary_is_fixed_and_independent_of_math_subset() -> None:
    module = _evaluator_module()
    fixture = ROOT / "fixtures/q8-runtime-quality/mmlu-math.jsonl"
    items = module.load_math_items(fixture)

    selected, provenance = module.select_greedy_canary(items, 8)
    repeated, repeated_provenance = module.select_greedy_canary(
        list(reversed(items)), 8
    )
    math_subset, _ = module.select_math_items(items, 100)
    subset_canary, _ = module.select_greedy_canary(math_subset, 8)

    assert [item["id"] for item in selected] == [item["id"] for item in repeated]
    assert provenance == repeated_provenance
    assert [item["id"] for item in selected] != [item["id"] for item in subset_canary]
    assert provenance["seed"] == 20260817
    assert provenance["ids"] == [
        "high_school_mathematics-0041",
        "college_mathematics-0002",
        "high_school_mathematics-0251",
        "elementary_mathematics-0369",
        "elementary_mathematics-0043",
        "high_school_mathematics-0152",
        "elementary_mathematics-0053",
        "high_school_mathematics-0198",
    ]
    assert provenance["ids_sha256"] == (
        "3b6f3af3654ed12d8ae3421e3a8419cf665c40afa3f195d02d01710383986ec2"
    )
    assert provenance["answers"] == ["B", "B", "B", "D", "C", "D", "A", "A"]
    assert provenance["canary_sha256"] == (
        "102ba6c7b1aa188c5d7663a2ce614c39ab7fa8647f34371ddb85d2b0084eef88"
    )


def test_math_progress_is_reconstructed_to_paired_correctness() -> None:
    module = _evaluator_module()
    progress = "\n".join(
        [
            "1\t100.00000000",
            "2\t50.00000000",
            "3\t66.66666667",
            "4\t50.00000000",
        ]
    )

    correctness = module.parse_math_correctness(
        "", progress, total=4, final_percent=50.0
    )

    assert correctness == (True, False, True, False)
    with pytest.raises(module.EvaluationError, match="contiguous"):
        module.parse_math_correctness(
            "", progress.replace("2\t", "3\t", 1), total=4, final_percent=50.0
        )


def test_paired_comparison_and_exact_two_sided_mcnemar() -> None:
    module = _evaluator_module()

    comparison = module.compare_math_correctness("1" * 10, "0" * 10)

    assert comparison == {
        "total": 10,
        "both_correct": 0,
        "baseline_only": 10,
        "candidate_only": 0,
        "both_wrong": 0,
        "candidate_net_correct": -10,
        "mcnemar_exact_two_sided_p": pytest.approx(0.001953125),
    }


def test_corrected_gate_is_full_848_only_and_uses_paired_regression() -> None:
    module = _evaluator_module()
    baseline = {
        "perplexity": 3.0,
        "math_accuracy": 0.6,
        "greedy_accuracy": 0.75,
    }
    candidate = {
        "perplexity": 3.015,
        "math_accuracy": 0.59,
        "greedy_accuracy": 0.75,
    }
    selection = {
        "mode": "fixture_order_full",
        "source_total": 848,
        "selected_total": 848,
    }
    comparison = {
        "total": 848,
        "baseline_only": 10,
        "candidate_only": 2,
        "candidate_net_correct": -8,
        "mcnemar_exact_two_sided_p": 0.0386,
    }

    gate = module.corrected_gate(baseline, candidate, comparison, selection)

    assert gate["applicable"] is True
    assert gate["outcome"] == "FAIL"
    assert gate["checks"]["perplexity_regression_within_0_5_percent"] is True
    assert gate["checks"]["math_drop_within_1_percentage_point"] is True
    assert gate["checks"]["net_math_regression_not_significant"] is False

    comparison["mcnemar_exact_two_sided_p"] = 0.0501
    assert module.corrected_gate(baseline, candidate, comparison, selection)[
        "outcome"
    ] == "PASS"
    selection["selected_total"] = 100
    comparison["total"] = 100
    assert module.corrected_gate(baseline, candidate, comparison, selection)[
        "outcome"
    ] == "NOT_APPLICABLE"


def test_math_limit_rejects_unreviewed_sizes() -> None:
    module = _evaluator_module()

    with pytest.raises(module.EvaluationError, match="supports only 100"):
        module.select_math_items([], 99)


def test_arm_runtime_args_are_canonicalized_hashed_and_applied_to_all_commands(
    tmp_path: Path,
) -> None:
    result, output, _ = _run(
        tmp_path,
        runtime_arguments=(
            "--baseline-runtime-arg=--flash-attn=on",
            "--baseline-runtime-arg=-ctk",
            "--baseline-runtime-arg=f16",
            "--candidate-runtime-arg=-ctv=q8_0",
            "--candidate-runtime-arg=-fa",
            "--candidate-runtime-arg=on",
        ),
    )

    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["protocol"]["coordinates"]["runtime_args"] == {
        "baseline": ["-fa", "on", "-ctk", "f16"],
        "candidate": ["-fa", "on", "-ctv", "q8_0"],
    }
    assert document["baseline"]["runtime_args"] == ["-fa", "on", "-ctk", "f16"]
    assert document["candidate"]["runtime_args"] == ["-fa", "on", "-ctv", "q8_0"]
    for command in document["baseline"]["commands"].values():
        assert command[command.index("-fa") + 1] == "on"
        assert command[command.index("-ctk") + 1] == "f16"
    for command in document["candidate"]["commands"].values():
        assert command[command.index("-fa") + 1] == "on"
        assert command[command.index("-ctv") + 1] == "q8_0"

    plain_result, plain_output, _ = _run(tmp_path / "plain")
    assert plain_result.returncode == 0, plain_result.stderr
    plain = json.loads(plain_output.read_text(encoding="utf-8"))
    assert plain["protocol_hash"] != document["protocol_hash"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--candidate-runtime-arg=-m", "--candidate-runtime-arg=other.gguf"), "not allowed"),
        (("--candidate-runtime-arg=-ctk=q6_k",), "invalid value"),
        (
            (
                "--candidate-runtime-arg=-fa=on",
                "--candidate-runtime-arg=--flash-attn=off",
            ),
            "duplicated",
        ),
    ],
)
def test_arm_runtime_args_reject_overrides_invalid_values_and_duplicates(
    tmp_path: Path, arguments: tuple[str, ...], message: str
) -> None:
    result, output, _ = _run(tmp_path, runtime_arguments=arguments)

    assert result.returncode == 1
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert message in document["error"]
    assert "baseline" not in document
    assert "candidate" not in document
