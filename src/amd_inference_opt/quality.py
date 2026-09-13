"""Deterministic quality commands and result normalization."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .command import StageCommand
from .models import QualityResult, RunStatus


class QualityAdapterError(ValueError):
    """A quality command or raw three-way result is incomplete or incompatible."""


@dataclass(frozen=True)
class Q4ThreeWayQualityProtocol:
    math_rule_loop_dir: str
    hf_model_path: str
    q4_k_model_path: str
    output_path: str
    python_path: str | None = None
    script_path: str | None = None
    pythonpath: str | None = None
    perplexity_tokens: int = 16384
    full_math: bool = True
    timeout_seconds: float = 14400

    def __post_init__(self) -> None:
        root = Path(self.math_rule_loop_dir).resolve()
        python = (
            Path(self.python_path).resolve()
            if self.python_path
            else root / ".venv-lighteval/bin/python"
        )
        script = (
            Path(self.script_path).resolve()
            if self.script_path
            else root / "tools/q4rdna_threeway_quality_eval.py"
        )
        dependency_path = (
            Path(self.pythonpath).resolve()
            if self.pythonpath
            else root / "tools/kernel-anvil-deps"
        )
        if self.perplexity_tokens < 2:
            raise QualityAdapterError("perplexity_tokens must be at least two")
        if self.timeout_seconds <= 0:
            raise QualityAdapterError("quality timeout_seconds must be positive")
        for label, path in {
            "math_rule_loop_dir": root,
            "python": python,
            "quality script": script,
            "HF model": Path(self.hf_model_path).resolve(),
            "Q4_K model": Path(self.q4_k_model_path).resolve(),
            "PYTHONPATH": dependency_path,
        }.items():
            if not path.exists():
                raise QualityAdapterError(f"{label} does not exist: {path}")
        object.__setattr__(self, "math_rule_loop_dir", str(root))
        object.__setattr__(self, "python_path", str(python))
        object.__setattr__(self, "script_path", str(script))
        object.__setattr__(self, "pythonpath", str(dependency_path))
        object.__setattr__(self, "hf_model_path", str(Path(self.hf_model_path).resolve()))
        object.__setattr__(self, "q4_k_model_path", str(Path(self.q4_k_model_path).resolve()))
        object.__setattr__(self, "output_path", str(Path(self.output_path).resolve()))

    @property
    def argv(self) -> tuple[str, ...]:
        command = (
            str(self.python_path),
            str(self.script_path),
            "--model",
            self.hf_model_path,
            "--q4-k",
            self.q4_k_model_path,
            "--ppl-tokens",
            str(self.perplexity_tokens),
            "--variants",
            "bf16,q4_k_m,q4_rdna",
            "--output",
            self.output_path,
        )
        return command + (("--full-math",) if self.full_math else ())

    def command(self) -> StageCommand:
        return StageCommand(
            name="quality",
            argv=self.argv,
            cwd=self.math_rule_loop_dir,
            env={"PYTHONPATH": str(self.pythonpath)},
            timeout_seconds=self.timeout_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["argv"] = list(self.argv)
        return result


@dataclass(frozen=True)
class Q8RuntimeQualityProtocol:
    """Pairwise runtime quality command for immutable baseline/candidate GGUFs."""

    command_template: tuple[str, ...]
    baseline_binary: str
    candidate_binary: str
    model_path: str
    output_path: str
    cwd: str
    environment: dict[str, str]
    candidate_model_path: str | None = None
    timeout_seconds: float = 14400

    def __post_init__(self) -> None:
        if not self.command_template:
            raise QualityAdapterError("Q8 quality command template must not be empty")
        if self.timeout_seconds <= 0:
            raise QualityAdapterError("Q8 quality timeout must be positive")
        for label, path in {
            "baseline binary": Path(self.baseline_binary),
            "candidate binary": Path(self.candidate_binary),
            "baseline model": Path(self.model_path),
            "candidate model": Path(self.candidate_model_path or self.model_path),
            "quality cwd": Path(self.cwd),
        }.items():
            if not path.resolve().exists():
                raise QualityAdapterError(f"{label} does not exist: {path.resolve()}")

    @property
    def argv(self) -> tuple[str, ...]:
        coordinates = {
            "baseline_binary": str(Path(self.baseline_binary).resolve()),
            "candidate_binary": str(Path(self.candidate_binary).resolve()),
            "model": str(Path(self.model_path).resolve()),
            "baseline_model": str(Path(self.model_path).resolve()),
            "candidate_model": str(
                Path(self.candidate_model_path or self.model_path).resolve()
            ),
            "output": str(Path(self.output_path).resolve()),
        }
        try:
            return tuple(item.format_map(coordinates) for item in self.command_template)
        except KeyError as error:
            raise QualityAdapterError(
                f"unknown Q8 quality command placeholder: {error.args[0]}"
            ) from error

    def command(self) -> StageCommand:
        return StageCommand(
            name="quality",
            argv=self.argv,
            cwd=str(Path(self.cwd).resolve()),
            env=self.environment,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass(frozen=True)
class NormalizedQualityPair:
    baseline: QualityResult
    candidate: QualityResult
    protocol_hash: str
    raw_status: str

    def to_command_output(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.model_dump(mode="json"),
            "candidate": self.candidate.model_dump(mode="json"),
            "protocol_hash": self.protocol_hash,
            "raw_status": self.raw_status,
        }


def _load_document(payload: str | bytes | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, Path):
        decoded: object = json.loads(payload.read_text(encoding="utf-8"))
    elif isinstance(payload, bytes):
        decoded = json.loads(payload.decode("utf-8"))
    elif isinstance(payload, str):
        stripped = payload.lstrip()
        if stripped.startswith(("{", "[")) or "\n" in payload:
            decoded = json.loads(payload)
        else:
            possible_path = Path(payload)
            decoded = json.loads(possible_path.read_text(encoding="utf-8"))
    else:
        decoded = payload
    if not isinstance(decoded, dict):
        raise QualityAdapterError("three-way quality result must be a JSON object")
    return decoded


def _variant(document: dict[str, Any], key: str) -> tuple[float, float, int]:
    raw = document.get(key)
    if not isinstance(raw, dict):
        raise QualityAdapterError(f"three-way quality result is missing {key}")
    perplexity = raw.get("perplexity")
    math_result = raw.get("math")
    if not isinstance(perplexity, dict) or not isinstance(math_result, dict):
        raise QualityAdapterError(f"{key} is missing perplexity or math results")
    ppl = perplexity.get("perplexity")
    correct = math_result.get("correct")
    total = math_result.get("total")
    accuracy = math_result.get("accuracy")
    if (
        not isinstance(ppl, (int, float))
        or isinstance(ppl, bool)
        or not math.isfinite(ppl)
        or ppl <= 0
    ):
        raise QualityAdapterError(f"{key} perplexity must be positive")
    if (
        not isinstance(correct, int)
        or isinstance(correct, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total <= 0
        or correct < 0
        or correct > total
    ):
        raise QualityAdapterError(f"{key} math counts are invalid")
    computed_accuracy = correct / total
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
        raise QualityAdapterError(f"{key} math accuracy is invalid")
    if abs(float(accuracy) - computed_accuracy) > 1e-12:
        raise QualityAdapterError(f"{key} math accuracy does not match correct/total")
    return float(ppl), computed_accuracy, total


def normalize_q4_threeway_quality(
    payload: str | bytes | Path | dict[str, Any],
    *,
    baseline_representation_hash: str | None = None,
    candidate_representation_hash: str | None = None,
) -> NormalizedQualityPair:
    """Normalize Q4_K_M and Q4_RDNA from the complete three-way artifact."""

    document = _load_document(payload)
    status = document.get("status")
    if status != "complete":
        raise QualityAdapterError(f"three-way quality result status is {status!r}, not 'complete'")
    protocol = document.get("protocol")
    if not isinstance(protocol, dict):
        raise QualityAdapterError("three-way quality result is missing protocol")
    protocol_hash = protocol.get("hash")
    if (
        not isinstance(protocol_hash, str)
        or len(protocol_hash) != 64
        or any(character not in "0123456789abcdef" for character in protocol_hash)
    ):
        raise QualityAdapterError("quality protocol hash must be a lowercase SHA-256 digest")
    baseline_ppl, baseline_accuracy, baseline_total = _variant(document, "q4_k_m")
    candidate_ppl, candidate_accuracy, candidate_total = _variant(document, "q4_rdna")
    if baseline_total != candidate_total:
        raise QualityAdapterError("baseline and candidate math totals do not match")
    baseline = QualityResult(
        status=RunStatus.SUCCEEDED,
        correctness_passed=True,
        perplexity=baseline_ppl,
        accuracies={"math_accuracy": baseline_accuracy},
        coordinate_hash=protocol_hash,
        representation_hash=baseline_representation_hash,
    )
    candidate = QualityResult(
        status=RunStatus.SUCCEEDED,
        correctness_passed=True,
        perplexity=candidate_ppl,
        accuracies={"math_accuracy": candidate_accuracy},
        coordinate_hash=protocol_hash,
        representation_hash=candidate_representation_hash,
    )
    return NormalizedQualityPair(
        baseline=baseline,
        candidate=candidate,
        protocol_hash=protocol_hash,
        raw_status=status,
    )


def normalize_q8_runtime_quality(
    payload: str | bytes | Path | dict[str, Any],
    *,
    expected_model_sha256: str,
    expected_candidate_model_sha256: str | None = None,
) -> NormalizedQualityPair:
    """Validate a baseline/candidate result produced by the Q8 runtime evaluator."""

    document = _load_document(payload)
    if document.get("schema") != "gpuopt.q8-runtime-quality.v1":
        raise QualityAdapterError("Q8 quality result has an unsupported schema")
    if document.get("status") != "complete":
        raise QualityAdapterError("Q8 quality result is not complete")
    protocol_hash = document.get("protocol_hash")
    model_sha256 = document.get("model_sha256")
    baseline_model_sha256 = document.get("baseline_model_sha256", model_sha256)
    candidate_model_sha256 = document.get("candidate_model_sha256", model_sha256)
    if not isinstance(protocol_hash, str) or len(protocol_hash) != 64:
        raise QualityAdapterError("Q8 quality protocol hash is invalid")
    expected_candidate = expected_candidate_model_sha256 or expected_model_sha256
    if baseline_model_sha256 != expected_model_sha256:
        raise QualityAdapterError("Q8 quality result used a different model for baseline")
    if candidate_model_sha256 != expected_candidate:
        raise QualityAdapterError("Q8 quality result used a different model for candidate")

    def result(
        key: str, representation_hash: str
    ) -> tuple[QualityResult, str, float | None]:
        raw = document.get(key)
        if not isinstance(raw, dict):
            raise QualityAdapterError(f"Q8 quality result is missing {key}")
        perplexity = raw.get("perplexity")
        accuracy = raw.get("math_accuracy")
        token_hash = raw.get("greedy_tokens_sha256")
        if (
            not isinstance(perplexity, (int, float))
            or isinstance(perplexity, bool)
            or not math.isfinite(perplexity)
            or perplexity <= 0
        ):
            raise QualityAdapterError(f"{key} perplexity is invalid")
        if (
            not isinstance(accuracy, (int, float))
            or isinstance(accuracy, bool)
            or not math.isfinite(accuracy)
            or not 0 <= accuracy <= 1
        ):
            raise QualityAdapterError(f"{key} math_accuracy is invalid")
        if (
            not isinstance(token_hash, str)
            or len(token_hash) != 64
            or any(character not in "0123456789abcdef" for character in token_hash)
        ):
            raise QualityAdapterError(f"{key} greedy token hash is invalid")
        greedy_correct = raw.get("greedy_correct")
        greedy_total = raw.get("greedy_total")
        greedy_accuracy = raw.get("greedy_accuracy")
        greedy_values = (greedy_correct, greedy_total, greedy_accuracy)
        if any(value is not None for value in greedy_values):
            if (
                not isinstance(greedy_correct, int)
                or isinstance(greedy_correct, bool)
                or not isinstance(greedy_total, int)
                or isinstance(greedy_total, bool)
                or greedy_total <= 0
                or greedy_correct < 0
                or greedy_correct > greedy_total
                or not isinstance(greedy_accuracy, (int, float))
                or isinstance(greedy_accuracy, bool)
                or not math.isfinite(greedy_accuracy)
                or abs(float(greedy_accuracy) - greedy_correct / greedy_total) > 1e-12
            ):
                raise QualityAdapterError(f"{key} greedy correctness metrics are invalid")
            normalized_greedy_accuracy: float | None = float(greedy_accuracy)
        else:
            normalized_greedy_accuracy = None
        raw_accuracies = raw.get("accuracies", {})
        if not isinstance(raw_accuracies, dict):
            raise QualityAdapterError(f"{key} accuracies must be an object")
        accuracies: dict[str, float] = {}
        for metric, value in raw_accuracies.items():
            if (
                not isinstance(metric, str)
                or not metric.strip()
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise QualityAdapterError(f"{key} contains an invalid accuracy metric")
            accuracies[metric] = float(value)
        if "math_accuracy" in accuracies and not math.isclose(
            accuracies["math_accuracy"], float(accuracy), abs_tol=1e-12
        ):
            raise QualityAdapterError(
                f"{key} math_accuracy differs from the accuracies mapping"
            )
        accuracies["math_accuracy"] = float(accuracy)
        if normalized_greedy_accuracy is not None:
            if "greedy_accuracy" in accuracies and not math.isclose(
                accuracies["greedy_accuracy"],
                normalized_greedy_accuracy,
                abs_tol=1e-12,
            ):
                raise QualityAdapterError(
                    f"{key} greedy_accuracy differs from the accuracies mapping"
                )
            accuracies["greedy_accuracy"] = normalized_greedy_accuracy
        return (
            QualityResult(
                status=RunStatus.SUCCEEDED,
                correctness_passed=True,
                perplexity=float(perplexity),
                accuracies=accuracies,
                coordinate_hash=protocol_hash,
                representation_hash=representation_hash,
            ),
            token_hash,
            normalized_greedy_accuracy,
        )

    baseline, baseline_tokens, baseline_greedy_accuracy = result(
        "baseline", expected_model_sha256
    )
    candidate, candidate_tokens, candidate_greedy_accuracy = result(
        "candidate", expected_candidate
    )
    if (baseline_greedy_accuracy is None) != (candidate_greedy_accuracy is None):
        raise QualityAdapterError(
            "baseline and candidate must use the same greedy correctness protocol"
        )
    if baseline_greedy_accuracy is None:
        # Compatibility with artifacts written before ground-truth canary scoring.
        candidate.correctness_passed = baseline_tokens == candidate_tokens
    else:
        assert candidate_greedy_accuracy is not None
        candidate.correctness_passed = (
            candidate_greedy_accuracy >= baseline_greedy_accuracy
        )
    return NormalizedQualityPair(
        baseline=baseline,
        candidate=candidate,
        protocol_hash=protocol_hash,
        raw_status="complete",
    )


__all__ = [
    "NormalizedQualityPair",
    "Q4ThreeWayQualityProtocol",
    "Q8RuntimeQualityProtocol",
    "QualityAdapterError",
    "normalize_q4_threeway_quality",
    "normalize_q8_runtime_quality",
]
