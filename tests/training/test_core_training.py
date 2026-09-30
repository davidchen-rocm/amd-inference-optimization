import copy
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from macfit_training.artifacts import canonical_sha256, describe_artifact, finish_artifacts
from macfit_training.config import LIMITS, InputError, capabilities, validate_job_input
from macfit_training.data import encode_training_row, prepare_training_data
from macfit_training.evaluation import comparison
from macfit_training.generation import GenerationError, build_examples, parse_generated_example


def request(kind="training"):
    common = {
        "model_id": "qwen3-0-6b",
        "task": "answers",
        "goal": "Answer support questions clearly.",
        "language": "en",
        "source": "Orders ship within two business days.",
    }
    if kind == "generation":
        return {**common, "purpose": "preview", "seeds": [], "target_count": 3}
    return {
        **common,
        "preset": "quick",
        "training": [
            {
                "id": str(i),
                "approved": True,
                "messages": [
                    {"role": "user", "content": f"Training question {i}?"},
                    {"role": "assistant", "content": f"Corrected answer {i}."},
                ],
            }
            for i in range(3)
        ],
        "evaluation": [
            {
                "id": str(i),
                "approved": True,
                "question": f"New test {i}?",
                "expected": f"Expected answer {i}.",
            }
            for i in range(3)
        ],
    }


def test_public_registry_is_pinned_and_validation_is_idempotent():
    cap = capabilities()
    assert {m["id"] for m in cap["models"]} == {"qwen3-0-6b", "qwen3-1-7b", "qwen3-4b", "qwen3-8b"}
    assert all(re.fullmatch(r"[a-f0-9]{40}", m["revision"]) for m in cap["models"])
    assert cap["auth"]["required"] is True
    assert cap["presets"] == ["quick", "standard"]
    for kind in ("generation", "training"):
        raw = request(kind)
        original = copy.deepcopy(raw)
        clean = validate_job_input(kind, raw)
        assert clean == validate_job_input(kind, clean)
        assert raw == original
        clean["base_model"]["revision"] = "a" * 40
        with pytest.raises(InputError, match="pinned registry"):
            validate_job_input(kind, clean)


def test_normalized_snapshot_near_size_limit_remains_valid(monkeypatch):
    raw = request()
    size = len(json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode())
    monkeypatch.setitem(LIMITS, "max_body_bytes", size)
    clean = validate_job_input("training", raw)
    assert validate_job_input("training", clean) == clean
    raw["source"] += "a"
    with pytest.raises(InputError, match="3 MiB"):
        validate_job_input("training", raw)


def test_malformed_kind_is_a_validation_error():
    with pytest.raises(InputError, match="kind"):
        validate_job_input([], request())


@pytest.mark.parametrize(
    "patch",
    [
        {"model_id": "someone/unreviewed-model"},
        {"model_id": []},
        {"owner_uid": "other-user"},
        {"command": "arbitrary"},
        {"preset": "unbounded"},
        {"training": []},
        {"evaluation": []},
        {"training_config": {"max_steps": 99999}},
    ],
)
def test_public_training_input_rejects_unsupported_and_unbounded_fields(patch):
    with pytest.raises(InputError):
        validate_job_input("training", {**request(), **patch})


def test_training_requires_approval_and_disjoint_normalized_heldout_questions():
    raw = request()
    raw["training"][0]["approved"] = "true"
    with pytest.raises(InputError, match="reviewed"):
        validate_job_input("training", raw)
    raw = request()
    raw["evaluation"][0]["question"] = " TRAINING   QUESTION 0? "
    with pytest.raises(InputError, match="kept out of training"):
        validate_job_input("training", raw)
    raw = request()
    raw["training"][0]["messages"][0]["role"] = {}
    with pytest.raises(InputError):
        validate_job_input("training", raw)


class Tokenizer:
    eos_token_id = 1000

    def apply_chat_template(self, messages, **options):
        assert options == {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return [10, 11, 12, 13]

    def encode(self, text, **options):
        assert options == {"add_special_tokens": False}
        return [ord(c) for c in text]


def test_only_assistant_answer_and_eos_have_loss_labels():
    messages = [
        {"role": "system", "content": "Private instructions"},
        {"role": "user", "content": "A question"},
        {"role": "assistant", "content": "Yes"},
    ]
    encoded = encode_training_row(Tokenizer(), messages, max_length=8)
    assert encoded["input_ids"] == [10, 11, 12, 13, ord("Y"), ord("e"), ord("s"), 1000]
    assert encoded["labels"] == [-100, -100, -100, -100, ord("Y"), ord("e"), ord("s"), 1000]
    assert encoded["attention_mask"] == [1] * 8
    with pytest.raises(InputError, match="no text was truncated"):
        encode_training_row(Tokenizer(), messages, max_length=7)
    job = validate_job_input("training", request())
    assert len(prepare_training_data(Tokenizer(), job)) == 3


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"question":"x"}',
        '{"question":"x","answer":false}',
        '{"question":"x","answer":"y","command":"run"}',
    ],
)
def test_generated_examples_must_be_valid_text_objects(text):
    with pytest.raises(GenerationError):
        parse_generated_example(text)


def test_model_generation_preserves_corrected_seeds_and_never_falls_back_to_templates():
    raw = request("generation")
    raw.update(
        purpose="dataset",
        target_count=12,
        seeds=[
            {
                "id": f"seed-{i}",
                "question": f"Seed {i}?",
                "answer": f"  exact\nanswer {i}\n",
                "approved": True,
            }
            for i in range(3)
        ],
    )
    job = validate_job_input("generation", raw)
    counter = 0

    def infer(_messages):
        nonlocal counter
        counter += 1
        return {
            "text": json.dumps({"question": f"Generated {counter}?", "answer": "Real output."}),
            "output_tokens": 18,
        }

    examples, measurements = build_examples(job, infer, lambda *a, **k: None)
    assert len(examples) == 12 and len(measurements) == 9
    assert examples[0]["answer"] == raw["seeds"][0]["answer"]
    assert all(row["approved"] for row in examples[:3])
    assert all(
        row["origin"] == "gpu-generated" and row["approved"] is False for row in examples[3:]
    )
    calls = []

    def invalid(_messages):
        calls.append(1)
        return {"text": "broken JSON"}

    with pytest.raises(GenerationError, match="corrected seeds are unchanged"):
        build_examples(job, invalid, lambda *a, **k: None)
    assert len(calls) == 3
    assert job["seeds"] == raw["seeds"]


def test_generation_retry_explains_duplicate_and_keeps_distinct_task_intents():
    job = validate_job_input("generation", request("generation"))
    calls = []

    def infer(messages):
        specification = json.loads(messages[-1]["content"])
        calls.append(specification)
        if len(calls) <= 2:
            question = "When will my order ship?"
        elif len(calls) == 3:
            feedback = specification["retry_feedback"]
            assert "repeated" in feedback["rejection_reason"]
            assert "When will my order ship?" in feedback["rejected_response"]
            assert specification["avoid_these_questions"] == ["When will my order ship?"]
            question = "Can a weekend order leave before Tuesday?"
        else:
            assert "retry_feedback" not in specification
            question = "Is expedited shipping available?"
        return {"text": json.dumps({"question": question, "answer": "A proposed answer."})}

    examples, _ = build_examples(job, infer, lambda *a, **k: None)
    assert len(calls) == 4
    assert len({row["question"] for row in examples}) == 3
    assert len({calls[index]["new_example_type"] for index in (0, 1, 3)}) == 3
    assert all(row["approved"] is False for row in examples)


def test_before_after_comparison_contains_actual_outputs_and_rejects_identity_mismatch():
    before = {
        "samples": [{"id": "a", "question": "q", "expected": "expected", "text": "actual base"}],
        "expected_answer_loss": 2.1,
        "exact_match_rate": 0,
    }
    after = {
        **before,
        "samples": [{**before["samples"][0], "text": "actual adapter"}],
        "expected_answer_loss": 1.9,
    }
    result = comparison(before, after)
    assert result["samples"][0]["before"] == "actual base"
    assert result["samples"][0]["after"] == "actual adapter"
    assert "do not establish" in result["limitations"]
    after["samples"][0]["id"] = "different"
    with pytest.raises(ValueError, match="identities differ"):
        comparison(before, after)


def test_adapter_artifacts_are_hash_bound_portable_and_explicitly_not_standalone(tmp_path):
    job = validate_job_input("training", request())
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"test adapter bytes")
    (adapter / "adapter_config.json").write_text("{}")
    records = finish_artifacts(
        tmp_path, job, {"evaluation": {"samples": []}, "provenance": {}}, adapter_dir=adapter
    )
    assert len(records) == 4
    for record in records:
        path = tmp_path / "artifacts" / record["name"]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((tmp_path / "artifacts/manifest.json").read_text())
    assert manifest["input_sha256"] == canonical_sha256(job)
    assert manifest["adapter_is_standalone_model"] is False
    with tarfile.open(tmp_path / "artifacts/adapter.tar.gz") as archive:
        assert all(member.name.startswith("adapter/") and member.isfile() for member in archive)
        usage = archive.extractfile("adapter/USAGE.md").read().decode()
        assert job["base_model"]["revision"] in usage
    (tmp_path / "link").symlink_to(tmp_path / "artifacts/manifest.json")
    with pytest.raises(ValueError, match="regular files"):
        describe_artifact(tmp_path / "link", "manifest")


def test_framework_modules_do_not_import_torch_on_cpu():
    root = Path(__file__).resolve().parents[2] / "src"
    script = (
        "import sys;"
        + "".join(
            f"__import__('macfit_training.{module}');"
            for module in (
                "config",
                "data",
                "trainer",
                "generation",
                "evaluation",
                "artifacts",
                "export",
                "worker",
                "cli",
            )
        )
        + "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], env={"PYTHONPATH": str(root)}, check=True)
