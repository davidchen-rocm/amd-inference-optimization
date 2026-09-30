"""CPU-only validation and immutable public model registry for MacFit jobs."""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from typing import Any

MODELS = {
    "qwen3-0-6b": {
        "id": "qwen3-0-6b",
        "repo_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "name": "Qwen3 0.6B",
    },
    "qwen3-1-7b": {
        "id": "qwen3-1-7b",
        "repo_id": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "name": "Qwen3 1.7B",
    },
    "qwen3-4b": {
        "id": "qwen3-4b",
        "repo_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "name": "Qwen3 4B",
    },
    "qwen3-8b": {
        "id": "qwen3-8b",
        "repo_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "name": "Qwen3 8B",
    },
}
# Revisions resolved from the official huggingface.co/api/models endpoints, 2026-09-30.
TASKS = ("answers", "writing", "format", "classify", "custom")
LANGUAGES = ("en", "zh", "multi")
LIMITS = {
    "max_body_bytes": 3 * 1024 * 1024,
    "min_training_rows": 3,
    "max_training_rows": 500,
    "min_evaluation_rows": 3,
    "max_evaluation_rows": 50,
    "max_seq_length": 2048,
    "max_steps": 500,
    "max_epochs": 3,
    "walltime_seconds": 3600,
    "max_goal_chars": 6000,
    "max_source_chars": 12000,
}
COMMON_TRAINING = {
    "method": "lora_sft",
    "dtype": "bfloat16",
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "max_seq_length": 2048,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_grad_norm": 1.0,
    "seed": 42,
    "walltime_seconds": 3600,
    "eval_max_new_tokens": 256,
}
PRESETS = {
    "quick": {**COMMON_TRAINING, "epochs": 1, "max_steps": 50},
    "standard": {**COMMON_TRAINING, "epochs": 3, "max_steps": 300},
}
GENERATION_CONFIG = {
    "max_input_tokens": 4096,
    "max_new_tokens": 768,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_attempts_per_example": 3,
    "walltime_seconds": 3600,
    "seed": 42,
}


class InputError(ValueError):
    """A request is outside the bounded public training contract."""


def capabilities() -> dict[str, Any]:
    """Static contract only: the service must set availability from worker health."""
    return {
        "available": False,
        "auth": {"required": True, "provider": "firebase"},
        "models": copy.deepcopy(list(MODELS.values())),
        "tasks": list(TASKS),
        "languages": list(LANGUAGES),
        "presets": list(PRESETS),
        "limits": copy.deepcopy(LIMITS),
        "generation": True,
        "training": True,
        "evaluation": True,
        "cancel": True,
        "online_hosting": False,
        "artifact_format": "peft_lora_adapter",
    }


def normalized_question(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _object(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise InputError(f"{label} must be an object.")
    extra = set(value) - allowed
    if extra:
        raise InputError(f"{label} contains unsupported fields: {', '.join(sorted(extra))}.")
    return value


def _text(value: Any, limit: int, label: str, *, required: bool = True) -> str:
    if not isinstance(value, str) or "\x00" in value or len(value) > limit:
        raise InputError(f"{label} must be text of at most {limit} characters.")
    result = value.strip()
    if required and not result:
        raise InputError(f"{label} is required.")
    return value


def _rows(value: Any, minimum: int, maximum: int, label: str) -> list[Any]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise InputError(f"{label} must contain {minimum} to {maximum} rows.")
    return value


def _identity(row: dict[str, Any], seen: set[str], label: str) -> str:
    value = _text(row.get("id"), 128, f"{label} id").strip()
    if value in seen:
        raise InputError(f"{label} ids must be unique.")
    seen.add(value)
    return value


def _reviewed(row: dict[str, Any], label: str) -> None:
    if row.get("approved") is not True:
        raise InputError(f"Every {label} must be reviewed and approved.")


def _example(row: Any, seen: set[str], label: str) -> dict[str, Any]:
    row = _object(row, {"id", "question", "answer", "system", "approved"}, label)
    _reviewed(row, label)
    out = {
        "id": _identity(row, seen, label),
        "question": _text(row.get("question"), 12000, f"{label} question"),
        "answer": _text(row.get("answer"), 24000, f"{label} answer"),
        "approved": True,
    }
    if "system" in row:
        out["system"] = _text(row["system"], 6000, f"{label} system", required=False)
    return out


def validate_job_input(kind: str, value: Any) -> dict[str, Any]:
    """Return JSON-safe input with pinned registry/preset; never accepts caller paths."""
    common = {"model_id", "task", "goal", "language", "source", "base_model"}
    fields = (
        {"purpose", "seeds", "target_count", "generation_config"}
        if kind == "generation"
        else {"training", "evaluation", "preset", "training_config"}
    )
    if not isinstance(kind, str) or kind not in {"generation", "training"}:
        raise InputError("Job kind must be generation or training.")
    value = _object(value, common | fields, "Job input")
    try:
        # Server-derived metadata adds a small amount to the private snapshot. Bound
        # caller data independently so revalidating a near-limit snapshot is stable.
        client_value = {
            key: item
            for key, item in value.items()
            if key not in {"base_model", "training_config", "generation_config"}
        }
        encoded = json.dumps(
            client_value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise InputError("Job input must be valid JSON text.") from exc
    if len(encoded) > LIMITS["max_body_bytes"]:
        raise InputError("Job input exceeds the 3 MiB limit.")
    model_id = value.get("model_id")
    if not isinstance(model_id, str) or model_id not in MODELS:
        raise InputError("Choose one of the available training models.")
    if value.get("task") not in TASKS or value.get("language") not in LANGUAGES:
        raise InputError("Choose a supported task and language.")
    out = {
        "model_id": model_id,
        "base_model": copy.deepcopy(MODELS[model_id]),
        "task": value["task"],
        "language": value["language"],
        "goal": _text(value.get("goal"), 6000, "Goal"),
        "source": _text(value.get("source", ""), 12000, "Source", required=False),
    }
    if len(out["goal"].strip()) < 12:
        raise InputError("Describe your goal in at least 12 characters.")
    if "base_model" in value and value["base_model"] != out["base_model"]:
        raise InputError("The base model identity must match the pinned registry.")
    if kind == "generation":
        purpose = value.get("purpose")
        count = value.get("target_count")
        if (
            not isinstance(purpose, str)
            or purpose not in {"preview", "dataset"}
            or type(count) is not int
        ):
            raise InputError("Choose preview or dataset and an integer target count.")
        if (purpose == "preview" and count != 3) or (
            purpose == "dataset" and count not in (12, 24, 48)
        ):
            raise InputError("Preview produces 3 examples; datasets contain 12, 24 or 48.")
        seed_count = 3 if purpose == "dataset" else 0
        raw = _rows(value.get("seeds", []), seed_count, seed_count, "Seeds")
        seen: set[str] = set()
        seeds = [_example(row, seen, "Seed") for row in raw]
        if len({normalized_question(row["question"]) for row in seeds}) != len(seeds):
            raise InputError("Corrected seed questions must be distinct.")
        out.update(
            purpose=purpose,
            seeds=seeds,
            target_count=count,
            generation_config=copy.deepcopy(GENERATION_CONFIG),
        )
        if "generation_config" in value and value["generation_config"] != out["generation_config"]:
            raise InputError("Generation settings must match the server configuration.")
        return out
    preset = value.get("preset", "quick")
    if not isinstance(preset, str) or preset not in PRESETS:
        raise InputError("Choose the quick or standard training preset.")
    training, seen, question_keys = [], set(), set()
    for row in _rows(value.get("training"), 3, 500, "Training data"):
        row = _object(row, {"id", "messages", "approved"}, "Training row")
        _reviewed(row, "training row")
        messages = []
        for message in _rows(row.get("messages"), 2, 3, "Messages"):
            message = _object(message, {"role", "content"}, "Message")
            role = message.get("role")
            if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
                raise InputError("Only system, user and assistant messages are supported.")
            maximum = {"system": 6000, "user": 12000, "assistant": 24000}[role]
            messages.append(
                {"role": role, "content": _text(message.get("content"), maximum, "Message")}
            )
        roles = [message["role"] for message in messages]
        if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
            raise InputError(
                "Use one user message and one answer, with an optional system message."
            )
        question = normalized_question(messages[-2]["content"])
        if question in question_keys:
            raise InputError(
                "Training questions must be distinct; resolve duplicate answers first."
            )
        question_keys.add(question)
        training.append(
            {"id": _identity(row, seen, "Training"), "messages": messages, "approved": True}
        )
    evaluation, seen, test_keys = [], set(), set()
    for row in _rows(value.get("evaluation"), 3, 50, "Held-out evaluation"):
        row = _object(row, {"id", "question", "expected", "system", "approved"}, "Evaluation row")
        _reviewed(row, "test question")
        item = {
            "id": _identity(row, seen, "Evaluation"),
            "approved": True,
            "question": _text(row.get("question"), 12000, "Test question"),
            "expected": _text(row.get("expected"), 24000, "Expected answer"),
        }
        if "system" in row:
            item["system"] = _text(row["system"], 6000, "Test system", required=False)
        question = normalized_question(item["question"])
        if question in question_keys or question in test_keys:
            raise InputError("Test questions must be distinct and kept out of training.")
        test_keys.add(question)
        evaluation.append(item)
    out.update(
        training=training,
        evaluation=evaluation,
        preset=preset,
        training_config=copy.deepcopy(PRESETS[preset]),
    )
    if "training_config" in value and value["training_config"] != out["training_config"]:
        raise InputError("Training settings must match the server preset.")
    return out
