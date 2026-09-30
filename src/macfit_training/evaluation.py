"""Measured held-out inference and expected-answer loss before and after SFT."""

from __future__ import annotations

import math
import time
from typing import Any

from .config import InputError, normalized_question
from .data import encode_training_row, evaluation_messages, prompt_ids, tensor_batch


def generate_text(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_input_tokens: int,
    max_new_tokens: int,
    sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict[str, Any]:
    import torch

    ids = prompt_ids(tokenizer, messages)
    if len(ids) > max_input_tokens:
        raise InputError("The prompt exceeds this job's context limit. Shorten the reference text.")
    inputs = tensor_batch({"input_ids": ids, "attention_mask": [1] * len(ids)}, model.device)
    model.eval()
    torch.cuda.synchronize()
    started = time.monotonic()
    options = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if sample:
        options.update(temperature=temperature, top_p=top_p)
    else:
        options.update(temperature=None, top_p=None, top_k=None)
    with torch.inference_mode():
        output = model.generate(**inputs, **options)[0, len(ids) :]
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    result = {
        "text": tokenizer.decode(output, skip_special_tokens=True).strip(),
        "input_tokens": len(ids),
        "output_tokens": int(output.numel()),
        "elapsed_seconds": elapsed,
        "stopped_by_limit": bool(
            output.numel() >= max_new_tokens and output[-1].item() != tokenizer.eos_token_id
        ),
    }
    del inputs, output
    return result


def evaluate_model(
    model: Any, tokenizer: Any, job: dict[str, Any], *, stage: str, emit: Any
) -> dict[str, Any]:
    import torch

    model.eval()
    samples, total_loss, total_tokens = [], 0.0, 0
    for index, row in enumerate(job["evaluation"]):
        messages = evaluation_messages(row, job)
        encoded = encode_training_row(
            tokenizer,
            [*messages, {"role": "assistant", "content": row["expected"]}],
            max_length=job["training_config"]["max_seq_length"],
            row_id=row["id"],
        )
        count = sum(label != -100 for label in encoded["labels"][1:])
        batch = tensor_batch(encoded, model.device)
        with torch.inference_mode():
            loss = float(model(**batch).loss.detach().float().cpu())
        if not math.isfinite(loss):
            raise RuntimeError("Held-out expected-answer loss is not finite.")
        prediction = generate_text(
            model,
            tokenizer,
            messages,
            max_input_tokens=job["training_config"]["max_seq_length"],
            max_new_tokens=job["training_config"]["eval_max_new_tokens"],
        )
        total_loss += loss * count
        total_tokens += count
        samples.append(
            {
                "id": row["id"],
                "question": row["question"],
                "expected": row["expected"],
                **prediction,
                "expected_answer_loss": loss,
                "exact_match": normalized_question(prediction["text"])
                == normalized_question(row["expected"]),
            }
        )
        emit(stage, completed=index + 1, total=len(job["evaluation"]), unit="questions")
        del batch
    return {
        "samples": samples,
        "expected_answer_loss": total_loss / total_tokens,
        "loss_tokens": total_tokens,
        "exact_match_rate": sum(row["exact_match"] for row in samples) / len(samples),
    }


def comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if [row["id"] for row in before["samples"]] != [row["id"] for row in after["samples"]]:
        raise ValueError("Before and after evaluation identities differ.")
    samples = [
        {
            "id": pre["id"],
            "question": pre["question"],
            "expected": pre["expected"],
            "before": pre["text"],
            "after": post["text"],
            "before_measurements": {
                k: v for k, v in pre.items() if k not in {"id", "question", "expected", "text"}
            },
            "after_measurements": {
                k: v for k, v in post.items() if k not in {"id", "question", "expected", "text"}
            },
        }
        for pre, post in zip(before["samples"], after["samples"], strict=True)
    ]
    return {
        "samples": samples,
        "metrics": {
            "before_expected_answer_loss": before["expected_answer_loss"],
            "after_expected_answer_loss": after["expected_answer_loss"],
            "before_exact_match_rate": before["exact_match_rate"],
            "after_exact_match_rate": after["exact_match_rate"],
        },
        "decoding": {"mode": "greedy", "enable_thinking": False},
        "limitations": "Loss and exact matching do not establish factual or semantic quality. "
        "Review the actual held-out outputs; this small test set is user supplied.",
    }
