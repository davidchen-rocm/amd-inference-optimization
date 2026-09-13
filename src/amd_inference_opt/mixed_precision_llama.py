"""llama.cpp materialization adapter for exact mixed-precision policies."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .mixed_precision import Precision, PrecisionPolicy
from .models import ArtifactRef, StrictModel

_TYPE_ARGUMENT = {
    Precision.Q4_K: "q4_k",
    Precision.Q5_K: "q5_k",
    Precision.Q6_K: "q6_k",
    Precision.Q8_0: "q8_0",
    Precision.F16: "f16",
    Precision.BF16: "bf16",
    Precision.F32: "f32",
}


class MixedPrecisionPackSpec(StrictModel):
    schema_name: Literal["gpuopt.mixed-precision-pack-spec.v1"] = Field(
        default="gpuopt.mixed-precision-pack-spec.v1", alias="schema"
    )
    candidate_id: str
    policy_sha256: str
    source_model: ArtifactRef
    importance_matrix: ArtifactRef
    quantizer_binary: ArtifactRef
    output_path: Path
    argv: list[str] = Field(min_length=1)
    command_sha256: str

    @model_validator(mode="after")
    def command_hash_matches(self) -> MixedPrecisionPackSpec:
        encoded = json.dumps(self.argv, separators=(",", ":")).encode()
        if hashlib.sha256(encoded).hexdigest() != self.command_sha256:
            raise ValueError("mixed-precision pack command hash mismatch")
        return self


def _selector(names: list[str]) -> str:
    if not names:
        raise ValueError("cannot build an empty tensor selector")
    return "^(?:" + "|".join(re.escape(name) for name in sorted(names)) + ")$"


def build_llama_quantize_pack_spec(
    policy: PrecisionPolicy,
    *,
    quantizer_path: str | Path,
    quantizer_binary: ArtifactRef,
    source_model_path: str | Path,
    source_model: ArtifactRef,
    imatrix_path: str | Path,
    importance_matrix: ArtifactRef,
    output_path: str | Path,
    threads: int,
    maximum_names_per_rule: int = 64,
) -> MixedPrecisionPackSpec:
    if policy.provider != "llama_cpp" or policy.quantizer_format is None:
        raise ValueError("only llama_cpp policies can use llama-quantize")
    if threads < 1 or maximum_names_per_rule < 1:
        raise ValueError("threads and maximum_names_per_rule must be positive")
    by_precision: dict[Precision, list[str]] = {}
    for assignment in policy.assignments:
        if assignment.precision == policy.base_precision:
            continue
        if assignment.precision not in _TYPE_ARGUMENT:
            raise ValueError(
                f"llama-quantize does not support policy precision {assignment.precision}"
            )
        by_precision.setdefault(assignment.precision, []).append(assignment.tensor_name)
    argv = [str(Path(quantizer_path)), "--imatrix", str(Path(imatrix_path))]
    for precision in sorted(by_precision, key=lambda item: item.value):
        names = sorted(by_precision[precision])
        for offset in range(0, len(names), maximum_names_per_rule):
            pattern = _selector(names[offset : offset + maximum_names_per_rule])
            argv.extend(("--tensor-type", f"{pattern}={_TYPE_ARGUMENT[precision]}"))
    argv.extend(
        (
            str(Path(source_model_path)),
            str(Path(output_path)),
            policy.quantizer_format,
            str(threads),
        )
    )
    digest = hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()
    return MixedPrecisionPackSpec(
        candidate_id=policy.id,
        policy_sha256=policy.assignment_sha256,
        source_model=source_model,
        importance_matrix=importance_matrix,
        quantizer_binary=quantizer_binary,
        output_path=Path(output_path),
        argv=argv,
        command_sha256=digest,
    )


__all__ = ["MixedPrecisionPackSpec", "build_llama_quantize_pack_spec"]
