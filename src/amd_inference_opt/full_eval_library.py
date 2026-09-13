"""Frozen, full-size English evaluation data for final model validation.

The compact suites in :mod:`amd_inference_opt.eval_suites` remain the fast
optimization gates.  This module downloads complete upstream test splits and
publishes them as an immutable, hash-bound local library.  It deliberately does
not pretend that heterogeneous tasks share one scorer.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ENGLISH_FULL_SUITE_ID = "english-full.v1"
ENGLISH_FULL_SCHEMA = "gpuopt.full-evaluation-library.v1"

LONG_BENCH_TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)


class FullEvaluationError(ValueError):
    """The full evaluation library could not be prepared or verified."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FullEvaluationSourceV1(_StrictModel):
    source_id: str
    category: Literal[
        "knowledge",
        "commonsense",
        "math",
        "instruction_following",
        "code",
        "long_context",
    ]
    dataset: str
    config: str | None = None
    split: str
    revision: str
    license: str
    scorer: str
    path: str
    records: Annotated[int, Field(gt=0)]
    bytes: Annotated[int, Field(gt=0)]
    sha256: str

    @field_validator(
        "source_id",
        "dataset",
        "split",
        "revision",
        "license",
        "scorer",
        "path",
    )
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("full evaluation source text fields must not be empty")
        return value

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("full evaluation paths must be relative")
        return value

    @field_validator("sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return value


class FullEvaluationManifestV1(_StrictModel):
    schema_version: Literal["gpuopt.full-evaluation-library.v1"] = ENGLISH_FULL_SCHEMA
    suite_id: Literal["english-full.v1"] = ENGLISH_FULL_SUITE_ID
    language: Literal["en"] = "en"
    purpose: Literal["final_validation"] = "final_validation"
    total_records: Annotated[int, Field(gt=0)]
    sources: list[FullEvaluationSourceV1]
    limitations: list[str]

    @model_validator(mode="after")
    def consistent(self) -> FullEvaluationManifestV1:
        if not self.sources or len({source.source_id for source in self.sources}) != len(
            self.sources
        ):
            raise ValueError("full evaluation sources must be present and unique")
        if self.total_records != sum(source.records for source in self.sources):
            raise ValueError("total_records does not match source record counts")
        return self


SOURCE_COORDINATES: tuple[dict[str, str | None], ...] = (
    {
        "source_id": "mmlu_pro",
        "category": "knowledge",
        "dataset": "TIGER-Lab/MMLU-Pro",
        "config": None,
        "split": "test",
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "license": "MIT",
        "scorer": "official-mmlu-pro-choice",
    },
    {
        "source_id": "arc_challenge",
        "category": "knowledge",
        "dataset": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "license": "CC-BY-SA-4.0",
        "scorer": "normalized-logprob-multiple-choice",
    },
    {
        "source_id": "hellaswag",
        "category": "commonsense",
        "dataset": "Rowan/hellaswag",
        "config": None,
        "split": "validation",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "license": "MIT",
        "scorer": "normalized-logprob-multiple-choice",
    },
    {
        "source_id": "winogrande",
        "category": "commonsense",
        "dataset": "allenai/winogrande",
        "config": "winogrande_xl",
        "split": "validation",
        "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "license": "Apache-2.0",
        "scorer": "normalized-logprob-multiple-choice",
    },
    {
        "source_id": "math_500",
        "category": "math",
        "dataset": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "license": "source-card-unspecified",
        "scorer": "boxed-exact-answer-generation",
    },
    {
        "source_id": "gsm8k",
        "category": "math",
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "license": "MIT",
        "scorer": "numeric-exact-answer-generation",
    },
    {
        "source_id": "ifeval",
        "category": "instruction_following",
        "dataset": "google/IFEval",
        "config": None,
        "split": "train",
        "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "license": "Apache-2.0",
        "scorer": "google-ifeval-strict-and-loose",
    },
    {
        "source_id": "livecodebench",
        "category": "code",
        "dataset": "livecodebench/code_generation_lite",
        "config": "release_v6",
        "split": "test",
        "revision": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        "license": "MIT (dataset script metadata)",
        "scorer": "official-livecodebench-code-execution",
    },
    {
        "source_id": "longbench_en",
        "category": "long_context",
        "dataset": "THUDM/LongBench",
        "config": "16 English and code tasks",
        "split": "test",
        "revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
        "license": "source-card-unspecified; constituent-task licenses apply",
        "scorer": "official-longbench-task-specific",
    },
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record_identity(source_id: str, row: Mapping[str, Any]) -> str:
    fields = {
        "mmlu_pro": ("question_id",),
        "arc_challenge": ("id",),
        "math_500": ("unique_id",),
        "ifeval": ("key",),
        "livecodebench": ("question_id",),
        "longbench_en": ("benchmark_task", "_id"),
    }.get(source_id, ())
    values = [str(row.get(field, "")).strip() for field in fields]
    if fields and all(values):
        return ":".join(values)
    return hashlib.sha256(_canonical_json(dict(row))).hexdigest()


def _write_records(path: Path, source_id: str, rows: Iterable[Mapping[str, Any]]) -> int:
    seen: dict[str, bytes] = {}
    count = 0
    with path.open("wb") as output:
        for raw in rows:
            row = dict(raw)
            identity = _record_identity(source_id, row)
            row["gpuopt_case_id"] = f"{source_id}:{identity}"
            payload = _canonical_json(row) + b"\n"
            payload_digest = hashlib.sha256(payload).digest()
            existing = seen.get(identity)
            if existing is not None:
                if existing != payload_digest:
                    raise FullEvaluationError(
                        f"{source_id} has conflicting duplicate id {identity}"
                    )
                continue
            seen[identity] = payload_digest
            output.write(payload)
            count += 1
    if count == 0:
        raise FullEvaluationError(f"{source_id} produced no records")
    return count


def _hub_rows(coordinate: Mapping[str, str | None]) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset  # type: ignore[import-not-found]

    dataset = str(coordinate["dataset"])
    config = coordinate["config"]
    arguments = (dataset, str(config)) if config is not None else (dataset,)
    rows = load_dataset(
        *arguments,
        split=str(coordinate["split"]),
        revision=str(coordinate["revision"]),
    )
    return (dict(row) for row in rows)


def _livecodebench_rows(coordinate: Mapping[str, str | None]) -> Iterable[Mapping[str, Any]]:
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    names = (
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    )
    for name in names:
        path = hf_hub_download(
            str(coordinate["dataset"]),
            name,
            repo_type="dataset",
            revision=str(coordinate["revision"]),
        )
        with Path(path).open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def _longbench_rows(coordinate: Mapping[str, str | None]) -> Iterable[Mapping[str, Any]]:
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    archive = hf_hub_download(
        str(coordinate["dataset"]),
        "data.zip",
        repo_type="dataset",
        revision=str(coordinate["revision"]),
    )
    with ZipFile(archive) as bundle:
        for task in LONG_BENCH_TASKS:
            member = f"data/{task}.jsonl"
            if member not in bundle.namelist():
                raise FullEvaluationError(f"LongBench archive is missing {member}")
            with bundle.open(member) as source:
                for raw in source:
                    if raw.strip():
                        row = json.loads(raw)
                        row["benchmark_task"] = task
                        yield row


def _default_rows(coordinate: Mapping[str, str | None]) -> Iterable[Mapping[str, Any]]:
    source_id = coordinate["source_id"]
    if source_id == "livecodebench":
        return _livecodebench_rows(coordinate)
    if source_id == "longbench_en":
        return _longbench_rows(coordinate)
    return _hub_rows(coordinate)


def load_english_full(output_dir: str | Path) -> FullEvaluationManifestV1:
    root = Path(output_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FullEvaluationError(f"full evaluation manifest is missing: {manifest_path}")
    manifest = FullEvaluationManifestV1.model_validate_json(manifest_path.read_bytes())
    for source in manifest.sources:
        path = root / source.path
        if not path.is_file() or path.is_symlink():
            raise FullEvaluationError(f"full evaluation data file is missing: {path}")
        if path.stat().st_size != source.bytes or _file_sha256(path) != source.sha256:
            raise FullEvaluationError(f"full evaluation data hash/size mismatch: {path}")
        with path.open("rb") as payload:
            records = sum(1 for line in payload if line.strip())
        if records != source.records:
            raise FullEvaluationError(f"full evaluation record count mismatch: {path}")
    return manifest


def prepare_english_full(
    output_dir: str | Path,
    *,
    records_by_source: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> FullEvaluationManifestV1:
    """Download and freeze every configured complete English evaluation split."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        return load_english_full(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    sources: list[FullEvaluationSourceV1] = []
    try:
        data_root = staging / "data"
        data_root.mkdir()
        for coordinate in SOURCE_COORDINATES:
            source_id = str(coordinate["source_id"])
            if records_by_source is not None:
                if source_id not in records_by_source:
                    raise FullEvaluationError(f"records omitted source {source_id}")
                rows = records_by_source[source_id]
            else:
                rows = _default_rows(coordinate)
            relative = f"data/{source_id}.jsonl"
            path = staging / relative
            records = _write_records(path, source_id, rows)
            sources.append(
                FullEvaluationSourceV1(
                    **coordinate,
                    path=relative,
                    records=records,
                    bytes=path.stat().st_size,
                    sha256=_file_sha256(path),
                )
            )
        manifest = FullEvaluationManifestV1(
            total_records=sum(source.records for source in sources),
            sources=sources,
            limitations=[
                "This library freezes data only; each source requires its declared scorer.",
                "Compact optimization gates should run before this expensive final suite.",
                (
                    "LongBench includes only its 14 English and two code tasks; "
                    "Chinese tasks are excluded."
                ),
                (
                    "Upstream source cards do not declare one aggregate license "
                    "for MATH-500 or LongBench."
                ),
            ],
        )
        (staging / "manifest.json").write_bytes(
            json.dumps(
                manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        os.replace(staging, destination)
        return load_english_full(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
