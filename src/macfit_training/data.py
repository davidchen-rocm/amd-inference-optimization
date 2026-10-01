"""Assistant-only examples; no GPU libraries are imported by validation."""

from __future__ import annotations

from typing import Any

from .config import InputError


def system_prompt(job: dict[str, Any]) -> str:
    prompt = job["goal"]
    if job.get("source"):
        prompt += "\n\nReference material supplied by the user:\n" + job["source"]
    return prompt


def with_system(messages: list[dict[str, str]], default_system: str) -> list[dict[str, str]]:
    if messages[0]["role"] == "system":
        return [dict(message) for message in messages]
    return [{"role": "system", "content": default_system}, *messages]


def prompt_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    # Qwen3's official template supplies a closed, empty thinking block here.
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    if not isinstance(ids, list) or not ids or not all(isinstance(x, int) for x in ids):
        raise InputError("The model tokenizer did not produce a valid chat prompt.")
    return ids


def encode_training_row(
    tokenizer: Any, messages: list[dict[str, str]], *, max_length: int, row_id: str = "row"
) -> dict[str, list[int]]:
    """Mask all prompt tokens and supervise only the final assistant answer + EOS."""
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise InputError("A training example must end with an assistant answer.")
    prefix = prompt_ids(tokenizer, messages[:-1])
    answer = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    if not answer or tokenizer.eos_token_id is None:
        raise InputError("The tokenizer could not encode an assistant answer and stop token.")
    target = [*answer, tokenizer.eos_token_id]
    ids = [*prefix, *target]
    if len(ids) > max_length:
        raise InputError(
            f"Example {row_id!r} needs {len(ids)} tokens; the limit is {max_length}. "
            "Shorten the question, system instructions or answer; no text was truncated."
        )
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prefix) + target,
    }


def prepare_training_data(tokenizer: Any, job: dict[str, Any]) -> list[dict[str, list[int]]]:
    return [
        encode_training_row(
            tokenizer,
            with_system(row["messages"], system_prompt(job)),
            max_length=job["training_config"]["max_seq_length"],
            row_id=row["id"],
        )
        for row in job["training"]
    ]


def evaluation_messages(row: dict[str, Any], job: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": row.get("system") or system_prompt(job)},
        {"role": "user", "content": row["question"]},
    ]


def tensor_batch(encoded: dict[str, list[int]], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: torch.tensor([value], dtype=torch.long, device=device)
        for key, value in encoded.items()
    }
