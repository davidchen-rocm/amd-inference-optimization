"""Pure planning adapters for mixed-bit and shape-kernel campaigns.

The objects in this module describe candidate intent and the evidence needed to
validate it.  They deliberately do not prepare models, edit llama.cpp, build a
binary, or launch a workload.  Execution remains the campaign runner's job.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ArtifactProvenance",
    "CampaignAdapterError",
    "KernelDispatchValidation",
    "MixedBitCampaignAdapter",
    "MixedBitCandidateSpec",
    "MixedBitProvenance",
    "ModelGeometry",
    "ShapeKernelCampaignAdapter",
    "ShapeKernelCandidateSpec",
    "ShapePatchKnob",
    "TensorRegexValidation",
    "TensorTypeOverride",
    "plan_mixed_bit_candidates",
    "plan_precision_reference_candidates",
    "plan_shape_kernel_candidates",
]


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TARGET_SOURCE = "ggml/src/ggml-cuda/mmvq.cu"


class CampaignAdapterError(ValueError):
    """A planning coordinate or validation descriptor is unsafe."""


def _positive(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CampaignAdapterError(f"{field} must be a positive integer")


def _optional_bounded_integer(
    value: int | None,
    field: str,
    *,
    maximum: int,
    power_of_two: bool = False,
) -> None:
    if value is None:
        return
    _positive(value, field)
    if value > maximum:
        raise CampaignAdapterError(f"{field} must be at most {maximum}")
    if power_of_two and value & (value - 1):
        raise CampaignAdapterError(f"{field} must be a power of two")


@dataclass(frozen=True)
class ModelGeometry:
    """The model dimensions needed by the two campaign planners."""

    block_count: int
    hidden_size: int
    intermediate_size: int
    attention_head_count: int
    kv_head_count: int
    vocab_size: int
    head_dim: int | None = None
    architecture: str = "qwen3"
    full_attention_layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        for field in (
            "block_count",
            "hidden_size",
            "intermediate_size",
            "attention_head_count",
            "kv_head_count",
            "vocab_size",
        ):
            _positive(getattr(self, field), field)
        if not self.architecture.strip():
            raise CampaignAdapterError("architecture must not be empty")
        if self.hidden_size % self.attention_head_count:
            raise CampaignAdapterError("hidden_size must be divisible by attention_head_count")
        resolved_head_dim = self.hidden_size // self.attention_head_count
        if self.head_dim is None:
            object.__setattr__(self, "head_dim", resolved_head_dim)
        else:
            _positive(self.head_dim, "head_dim")
            if self.head_dim != resolved_head_dim:
                raise CampaignAdapterError(
                    "head_dim must equal hidden_size / attention_head_count"
                )
        if self.kv_head_count > self.attention_head_count:
            raise CampaignAdapterError("kv_head_count cannot exceed attention_head_count")
        if self.attention_head_count % self.kv_head_count:
            raise CampaignAdapterError(
                "attention_head_count must be divisible by kv_head_count"
            )
        if self.full_attention_layers is not None:
            layers = tuple(self.full_attention_layers)
            if not layers or len(layers) != len(set(layers)):
                raise CampaignAdapterError(
                    "full_attention_layers must be non-empty and unique"
                )
            if tuple(sorted(layers)) != layers or any(
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or not 0 <= layer < self.block_count
                for layer in layers
            ):
                raise CampaignAdapterError(
                    "full_attention_layers must be sorted valid block indices"
                )
            object.__setattr__(self, "full_attention_layers", layers)

    @property
    def kv_width(self) -> int:
        """Output width of a key or value projection."""

        assert self.head_dim is not None
        return self.kv_head_count * self.head_dim

    @property
    def attention_block_indices(self) -> tuple[int, ...]:
        return self.full_attention_layers or tuple(range(self.block_count))

    @property
    def kv_cache_layer_count(self) -> int:
        return len(self.attention_block_indices)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> ModelGeometry:
        """Build geometry from common GGUF/model-config field names.

        The aliases make the planning boundary explicit without coupling it to
        one metadata reader.  Missing or conflicting values fail closed.
        """

        aliases = {
            "block_count": ("block_count", "num_hidden_layers"),
            "hidden_size": ("hidden_size", "embedding_length"),
            "intermediate_size": ("intermediate_size", "feed_forward_length"),
            "attention_head_count": ("attention_head_count", "num_attention_heads"),
            "kv_head_count": ("kv_head_count", "attention_head_count_kv", "num_key_value_heads"),
            "vocab_size": ("vocab_size", "token_count"),
            "head_dim": ("head_dim", "attention_key_length"),
            "architecture": ("architecture", "model_type"),
        }

        def select(field: str, *, required: bool = True) -> object | None:
            found = [metadata[name] for name in aliases[field] if name in metadata]
            if not found:
                if required:
                    raise CampaignAdapterError(f"model metadata is missing {field}")
                return None
            if any(value != found[0] for value in found[1:]):
                raise CampaignAdapterError(f"model metadata has conflicting {field} values")
            return found[0]

        numeric_fields = {
            field: select(field)
            for field in (
                "block_count",
                "hidden_size",
                "intermediate_size",
                "attention_head_count",
                "kv_head_count",
                "vocab_size",
            )
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in numeric_fields.values()
        ):
            raise CampaignAdapterError("model geometry metadata fields must be integers")
        head_dim = select("head_dim", required=False)
        if head_dim is not None and (isinstance(head_dim, bool) or not isinstance(head_dim, int)):
            raise CampaignAdapterError("head_dim metadata must be an integer")
        architecture = select("architecture", required=False)
        if architecture is not None and not isinstance(architecture, str):
            raise CampaignAdapterError("architecture metadata must be a string")
        raw_full_attention = metadata.get("full_attention_layers")
        layer_types = metadata.get("layer_types")
        if raw_full_attention is not None and layer_types is not None:
            raise CampaignAdapterError(
                "model metadata cannot provide both full_attention_layers and layer_types"
            )
        if layer_types is not None:
            if (
                isinstance(layer_types, (str, bytes))
                or not isinstance(layer_types, (list, tuple))
                or len(layer_types) != numeric_fields["block_count"]
                or any(not isinstance(value, str) for value in layer_types)
            ):
                raise CampaignAdapterError(
                    "layer_types must contain one string for every model block"
                )
            raw_full_attention = tuple(
                index
                for index, value in enumerate(layer_types)
                if value == "full_attention"
            )
        if raw_full_attention is not None and (
            isinstance(raw_full_attention, (str, bytes))
            or not isinstance(raw_full_attention, (list, tuple))
        ):
            raise CampaignAdapterError("full_attention_layers must be an integer array")
        return cls(
            **numeric_fields,  # type: ignore[arg-type]
            head_dim=head_dim,
            architecture=architecture or "qwen3",
            full_attention_layers=(
                tuple(raw_full_attention) if raw_full_attention is not None else None
            ),
        )


@dataclass(frozen=True)
class ArtifactProvenance:
    """Immutable identity for one preparation input or tool."""

    path: str
    sha256: str
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise CampaignAdapterError("artifact path must not be empty")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise CampaignAdapterError("artifact sha256 must be 64 lowercase hexadecimal digits")
        if self.size_bytes is not None:
            _positive(self.size_bytes, "artifact size_bytes")

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass(frozen=True)
class MixedBitProvenance:
    """Inputs proving every mixed-bit arm starts independently from BF16."""

    source_model: ArtifactProvenance
    calibration_corpus: ArtifactProvenance
    importance_matrix: ArtifactProvenance
    quantizer_binary: ArtifactProvenance
    llama_cpp_commit: str
    source_quantization: str = "BF16"
    direct_from_source: bool = True

    def __post_init__(self) -> None:
        for field in (
            "source_model",
            "calibration_corpus",
            "importance_matrix",
            "quantizer_binary",
        ):
            if not isinstance(getattr(self, field), ArtifactProvenance):
                raise CampaignAdapterError(f"{field} must be ArtifactProvenance")
        if self.source_quantization.upper() != "BF16":
            raise CampaignAdapterError("mixed-bit candidates must be quantized from BF16")
        if self.direct_from_source is not True:
            raise CampaignAdapterError("requantizing an existing quantized GGUF is forbidden")
        if not self.llama_cpp_commit.strip():
            raise CampaignAdapterError("llama_cpp_commit must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TensorRegexValidation:
    """An anchored selector plus the complete tensor names it may match."""

    pattern: str
    expected_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.pattern.startswith("^") or not self.pattern.endswith("$"):
            raise CampaignAdapterError("tensor regex must be anchored with ^ and $")
        try:
            compiled = re.compile(self.pattern)
        except re.error as error:
            raise CampaignAdapterError(f"invalid tensor regex: {error}") from error
        if not self.expected_names:
            raise CampaignAdapterError("tensor regex must have expected names")
        if len(set(self.expected_names)) != len(self.expected_names):
            raise CampaignAdapterError("expected tensor names must be unique")
        missed = [name for name in self.expected_names if compiled.fullmatch(name) is None]
        if missed:
            raise CampaignAdapterError(
                f"tensor regex does not match its expected names: {', '.join(missed[:3])}"
            )

    @property
    def expected_match_count(self) -> int:
        return len(self.expected_names)

    def matching_names(self, inventory: Iterable[str]) -> tuple[str, ...]:
        """Return matches only when they equal the locked expected set."""

        compiled = re.compile(self.pattern)
        matches = tuple(name for name in inventory if compiled.fullmatch(name) is not None)
        if len(set(matches)) != len(matches):
            raise CampaignAdapterError("tensor inventory contains duplicate matching names")
        expected = set(self.expected_names)
        actual = set(matches)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing[:3]))
            if unexpected:
                detail.append("unexpected " + ", ".join(unexpected[:3]))
            raise CampaignAdapterError("tensor selector validation failed: " + "; ".join(detail))
        return tuple(name for name in self.expected_names if name in actual)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "expected_names": list(self.expected_names),
            "expected_match_count": self.expected_match_count,
        }


@dataclass(frozen=True)
class TensorTypeOverride:
    """One llama-quantize ``--tensor-type`` rule and its validator."""

    operator_group: str
    tensor_type: str
    selector: TensorRegexValidation

    def __post_init__(self) -> None:
        if not self.operator_group.strip():
            raise CampaignAdapterError("operator_group must not be empty")
        if self.tensor_type not in {"Q4_K", "Q5_K"}:
            raise CampaignAdapterError("custom FFN tensor type must be Q4_K or Q5_K")
        if not isinstance(self.selector, TensorRegexValidation):
            raise CampaignAdapterError("selector must be TensorRegexValidation")

    @property
    def cli_value(self) -> str:
        return f"{self.selector.pattern}={self.tensor_type.lower()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_group": self.operator_group,
            "tensor_type": self.tensor_type,
            "selector": self.selector.to_dict(),
            "cli_value": self.cli_value,
        }


@dataclass(frozen=True)
class MixedBitCandidateSpec:
    """One independently prepared mixed-bit model candidate."""

    candidate_id: str
    order: int
    base_quantization: str
    output_filename: str
    provenance: MixedBitProvenance
    tensor_overrides: tuple[TensorTypeOverride, ...] = ()
    reference_arm: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.candidate_id):
            raise CampaignAdapterError("candidate_id must be a stable lowercase identifier")
        _positive(self.order, "candidate order")
        if self.base_quantization not in {"Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}:
            raise CampaignAdapterError("unsupported mixed-bit base quantization")
        if (
            not self.output_filename.endswith(".gguf")
            or Path(self.output_filename).name != self.output_filename
        ):
            raise CampaignAdapterError("output_filename must be a leaf .gguf name")
        if not isinstance(self.provenance, MixedBitProvenance):
            raise CampaignAdapterError("provenance must be MixedBitProvenance")
        if self.tensor_overrides and self.base_quantization != "Q6_K":
            raise CampaignAdapterError("custom FFN policies must keep a Q6_K base")
        if self.tensor_overrides and self.reference_arm:
            raise CampaignAdapterError("reference precision arms cannot override tensors")
        if (
            self.base_quantization == "Q6_K"
            and not self.tensor_overrides
            and not self.reference_arm
        ):
            raise CampaignAdapterError("a custom Q6_K policy requires tensor overrides")
        if self.base_quantization == "Q8_0" and not self.reference_arm:
            raise CampaignAdapterError("Q8_0 is supported only as a reference precision arm")

    @property
    def quantizer_args(self) -> tuple[str, ...]:
        """Declarative quantizer fragment; it is never executed here."""

        args: list[str] = ["--imatrix", self.provenance.importance_matrix.path]
        for override in self.tensor_overrides:
            args.extend(("--tensor-type", override.cli_value))
        args.append(self.base_quantization)
        return tuple(args)

    @property
    def preserves_non_ffn_q6(self) -> bool:
        return self.base_quantization == "Q6_K"

    def validate_tensor_inventory(self, inventory: Iterable[str]) -> None:
        names = tuple(inventory)
        for override in self.tensor_overrides:
            override.selector.matching_names(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "order": self.order,
            "base_quantization": self.base_quantization,
            "output_filename": self.output_filename,
            "provenance": self.provenance.to_dict(),
            "tensor_overrides": [override.to_dict() for override in self.tensor_overrides],
            "quantizer_args": list(self.quantizer_args),
            "preserves_non_ffn_q6": self.preserves_non_ffn_q6,
            "reference_arm": self.reference_arm,
        }


def _layer_names(geometry: ModelGeometry, operators: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"blk.{block}.{operator}.weight"
        for block in range(geometry.block_count)
        for operator in operators
    )


def _attention_layer_names(
    geometry: ModelGeometry, operators: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        f"blk.{block}.{operator}.weight"
        for block in geometry.attention_block_indices
        for operator in operators
    )


def _ffn_override(
    geometry: ModelGeometry,
    *,
    operators: tuple[str, ...],
    tensor_type: str,
) -> TensorTypeOverride:
    alternatives = "|".join(operator.removeprefix("ffn_") for operator in operators)
    suffix = alternatives if len(operators) == 1 else f"({alternatives})"
    return TensorTypeOverride(
        operator_group="/".join(operator.removeprefix("ffn_") for operator in operators),
        tensor_type=tensor_type,
        selector=TensorRegexValidation(
            pattern=rf"^blk\.[0-9]+\.ffn_{suffix}\.weight$",
            expected_names=_layer_names(geometry, operators),
        ),
    )


def plan_mixed_bit_candidates(
    geometry: ModelGeometry,
    provenance: MixedBitProvenance,
) -> tuple[MixedBitCandidateSpec, ...]:
    """Return the five approved mixed-bit candidates in fixed order."""

    all_ffn = ("ffn_gate", "ffn_up", "ffn_down")
    gate_up = ("ffn_gate", "ffn_up")
    down = ("ffn_down",)
    candidates = (
        MixedBitCandidateSpec(
            "q5_k_m",
            1,
            "Q5_K_M",
            "Q5_K_M-imatrix.gguf",
            provenance,
        ),
        MixedBitCandidateSpec(
            "q4_k_m",
            2,
            "Q4_K_M",
            "Q4_K_M-imatrix.gguf",
            provenance,
        ),
        MixedBitCandidateSpec(
            "q6_ffn_q5_k",
            3,
            "Q6_K",
            "Q6-FFN-Q5_K-imatrix.gguf",
            provenance,
            (_ffn_override(geometry, operators=all_ffn, tensor_type="Q5_K"),),
        ),
        MixedBitCandidateSpec(
            "q6_gate_up_q4_k_down_q5_k",
            4,
            "Q6_K",
            "Q6-GateUp-Q4_K-Down-Q5_K-imatrix.gguf",
            provenance,
            (
                _ffn_override(geometry, operators=gate_up, tensor_type="Q4_K"),
                _ffn_override(geometry, operators=down, tensor_type="Q5_K"),
            ),
        ),
        MixedBitCandidateSpec(
            "q6_ffn_q4_k",
            5,
            "Q6_K",
            "Q6-FFN-Q4_K-imatrix.gguf",
            provenance,
            (_ffn_override(geometry, operators=all_ffn, tensor_type="Q4_K"),),
        ),
    )
    if tuple(candidate.order for candidate in candidates) != tuple(range(1, 6)):
        raise AssertionError("mixed-bit candidate order drifted")
    return candidates


def plan_precision_reference_candidates(
    provenance: MixedBitProvenance,
    quantizations: tuple[str, ...],
) -> tuple[MixedBitCandidateSpec, ...]:
    """Plan a config-selected stock precision sweep from one frozen BF16 source."""

    supported = {"Q5_K_M", "Q6_K", "Q8_0"}
    if not quantizations:
        raise CampaignAdapterError("reference precision sweep cannot be empty")
    if len(quantizations) != len(set(quantizations)):
        raise CampaignAdapterError("reference precision sweep cannot repeat a quantization")
    unsupported = sorted(set(quantizations) - supported)
    if unsupported:
        raise CampaignAdapterError(
            "unsupported reference precision: " + ", ".join(unsupported)
        )
    return tuple(
        MixedBitCandidateSpec(
            candidate_id=quantization.lower(),
            order=index,
            base_quantization=quantization,
            output_filename=f"{quantization}-imatrix.gguf",
            provenance=provenance,
            reference_arm=True,
        )
        for index, quantization in enumerate(quantizations, 1)
    )


@dataclass(frozen=True)
class ShapePatchKnob:
    """Bounded tuning knobs for one exact N/K specialization.

    ``None`` means that a candidate does not alter that coordinate.  The
    defaults encode the first small-k/8-rows hypothesis, while callers can
    describe other experiments without generating arbitrary kernel source.
    """

    architecture: str
    n: int
    k: int
    quantization: str = "Q6_K"
    ncols_dst: int = 1
    small_k: bool | None = True
    split_k: int | None = None
    waves_per_block: int | None = None
    rows_per_wave: int | None = None
    rows_per_block: int | None = 8
    vector_load_bytes: int | None = None
    vgpr_limit: int | None = None
    fuse_gate_up: bool | None = None
    source_path: str = _TARGET_SOURCE

    def __post_init__(self) -> None:
        if self.architecture != "gfx1201":
            raise CampaignAdapterError("shape-kernel candidates are locked to gfx1201")
        _positive(self.n, "shape N")
        _positive(self.k, "shape K")
        if self.quantization != "Q6_K" or self.ncols_dst != 1:
            raise CampaignAdapterError("shape-kernel candidates require Q6_K ncols_dst=1")
        if self.small_k is not None and not isinstance(self.small_k, bool):
            raise CampaignAdapterError("small_k must be bool or None")
        if self.fuse_gate_up is not None and not isinstance(self.fuse_gate_up, bool):
            raise CampaignAdapterError("fuse_gate_up must be bool or None")
        _optional_bounded_integer(self.split_k, "split_k", maximum=64, power_of_two=True)
        _optional_bounded_integer(
            self.waves_per_block,
            "waves_per_block",
            maximum=16,
            power_of_two=True,
        )
        _optional_bounded_integer(
            self.rows_per_wave,
            "rows_per_wave",
            maximum=32,
            power_of_two=True,
        )
        _optional_bounded_integer(
            self.rows_per_block,
            "rows_per_block",
            maximum=64,
            power_of_two=True,
        )
        _optional_bounded_integer(
            self.vector_load_bytes,
            "vector_load_bytes",
            maximum=32,
            power_of_two=True,
        )
        _optional_bounded_integer(self.vgpr_limit, "vgpr_limit", maximum=256)
        if (
            self.rows_per_wave is not None
            and self.rows_per_block is not None
            and (
                self.rows_per_wave > self.rows_per_block
                or self.rows_per_block % self.rows_per_wave
            )
        ):
            raise CampaignAdapterError(
                "rows_per_wave must divide rows_per_block and cannot exceed it"
            )
        coordinates = (
            self.small_k,
            self.split_k,
            self.waves_per_block,
            self.rows_per_wave,
            self.rows_per_block,
            self.vector_load_bytes,
            self.vgpr_limit,
            self.fuse_gate_up,
        )
        if all(value is None for value in coordinates):
            raise CampaignAdapterError("a shape patch must select at least one tuning knob")
        if self.source_path != _TARGET_SOURCE:
            raise CampaignAdapterError(f"shape patch may only target {_TARGET_SOURCE}")


@dataclass(frozen=True)
class KernelDispatchValidation:
    """Profiler and benchmark evidence required for a shape patch."""

    baseline_grid: tuple[int, int, int]
    candidate_grid: tuple[int, int, int]
    workgroup: tuple[int, int, int] = (32, 8, 1)
    kernel_name_contains: str = "mul_mat_vec_q"
    changed_paths: tuple[str, ...] = (_TARGET_SOURCE,)
    independent_clean_base: bool = True
    generation_only: bool = True
    benchmark_token_lengths: tuple[int, ...] = (128, 512)
    max_cv_percent: float = 2.0
    min_improvement_percent_each: float = 1.0

    def __post_init__(self) -> None:
        for field in ("baseline_grid", "candidate_grid", "workgroup"):
            dimensions = getattr(self, field)
            if len(dimensions) != 3:
                raise CampaignAdapterError(f"{field} must contain three dimensions")
            for dimension in dimensions:
                _positive(dimension, field)
        if self.changed_paths != (_TARGET_SOURCE,):
            raise CampaignAdapterError("shape-kernel patch must change only mmvq.cu")
        if self.independent_clean_base is not True:
            raise CampaignAdapterError("each shape patch must start from an independent clean base")
        if self.generation_only is not True:
            raise CampaignAdapterError("shape benchmark gate is generation-only")
        if self.benchmark_token_lengths != (128, 512):
            raise CampaignAdapterError("shape benchmark lengths must be exactly tg128 and tg512")
        if self.max_cv_percent != 2.0 or self.min_improvement_percent_each != 1.0:
            raise CampaignAdapterError(
                "shape benchmark gate must be CV<=2% and >=1% at both lengths"
            )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for field in (
            "baseline_grid",
            "candidate_grid",
            "workgroup",
            "changed_paths",
            "benchmark_token_lengths",
        ):
            result[field] = list(getattr(self, field))
        return result


@dataclass(frozen=True)
class ShapeKernelCandidateSpec:
    """One exact model shape and its declarative patch/evidence contract."""

    candidate_id: str
    order: int
    operator_group: str
    tensor_selector: TensorRegexValidation
    patch_knob: ShapePatchKnob
    validation: KernelDispatchValidation

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.candidate_id):
            raise CampaignAdapterError("shape candidate_id must be a stable lowercase identifier")
        _positive(self.order, "shape candidate order")
        if not self.operator_group.strip():
            raise CampaignAdapterError("shape operator_group must not be empty")
        if self.patch_knob.fuse_gate_up is True and self.candidate_id != "ffn_gate_up":
            raise CampaignAdapterError("fuse_gate_up is only valid for the FFN gate/up shape")
        expected_baseline = (self.patch_knob.n * 32, 8, 1)
        if self.validation.baseline_grid != expected_baseline:
            raise CampaignAdapterError("baseline dispatch grid does not match shape N")
        if self.patch_knob.rows_per_block is not None:
            rows = self.patch_knob.rows_per_block
            expected_candidate = (
                ((self.patch_knob.n + rows - 1) // rows) * 32,
                self.validation.workgroup[1],
                self.validation.workgroup[2],
            )
            if self.validation.candidate_grid != expected_candidate:
                raise CampaignAdapterError(
                    "candidate dispatch grid does not match rows_per_block"
                )

    @property
    def n(self) -> int:
        return self.patch_knob.n

    @property
    def k(self) -> int:
        return self.patch_knob.k

    @property
    def split_k(self) -> int | None:
        return self.patch_knob.split_k

    @property
    def waves_per_block(self) -> int | None:
        return self.patch_knob.waves_per_block

    @property
    def rows_per_wave(self) -> int | None:
        return self.patch_knob.rows_per_wave

    @property
    def rows_per_block(self) -> int | None:
        return self.patch_knob.rows_per_block

    @property
    def vector_load_bytes(self) -> int | None:
        return self.patch_knob.vector_load_bytes

    @property
    def vgpr_limit(self) -> int | None:
        return self.patch_knob.vgpr_limit

    @property
    def fuse_gate_up(self) -> bool | None:
        return self.patch_knob.fuse_gate_up

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "order": self.order,
            "operator_group": self.operator_group,
            "N": self.n,
            "K": self.k,
            "tensor_selector": self.tensor_selector.to_dict(),
            "patch_knob": asdict(self.patch_knob),
            "validation": self.validation.to_dict(),
        }


def _shape_candidate(
    *,
    candidate_id: str,
    order: int,
    operator_group: str,
    n: int,
    k: int,
    selector: TensorRegexValidation,
    architecture: str,
) -> ShapeKernelCandidateSpec:
    knob = ShapePatchKnob(architecture=architecture, n=n, k=k)
    assert knob.rows_per_block is not None
    validation = KernelDispatchValidation(
        baseline_grid=(n * 32, 8, 1),
        candidate_grid=(((n + knob.rows_per_block - 1) // knob.rows_per_block) * 32, 8, 1),
    )
    return ShapeKernelCandidateSpec(
        candidate_id=candidate_id,
        order=order,
        operator_group=operator_group,
        tensor_selector=selector,
        patch_knob=knob,
        validation=validation,
    )


def plan_shape_kernel_candidates(
    geometry: ModelGeometry,
    *,
    architecture: str = "gfx1201",
) -> tuple[ShapeKernelCandidateSpec, ...]:
    """Derive the four approved real Qwen projection shapes.

    This returns descriptors only.  In particular, it is not a generic patch
    generator and it performs no source, build, profiler, or GPU operation.
    """

    candidates = (
        _shape_candidate(
            candidate_id="ffn_gate_up",
            order=1,
            operator_group="FFN gate/up projections",
            n=geometry.intermediate_size,
            k=geometry.hidden_size,
            selector=TensorRegexValidation(
                r"^blk\.[0-9]+\.ffn_(gate|up)\.weight$",
                _layer_names(geometry, ("ffn_gate", "ffn_up")),
            ),
            architecture=architecture,
        ),
        _shape_candidate(
            candidate_id="attention_query_output",
            order=2,
            operator_group="self-attention query/output projections",
            n=geometry.hidden_size,
            k=geometry.hidden_size,
            selector=TensorRegexValidation(
                r"^blk\.[0-9]+\.attn_(q|output)\.weight$",
                _attention_layer_names(geometry, ("attn_q", "attn_output")),
            ),
            architecture=architecture,
        ),
        _shape_candidate(
            candidate_id="attention_key_value",
            order=3,
            operator_group="self-attention key/value projections",
            n=geometry.kv_width,
            k=geometry.hidden_size,
            selector=TensorRegexValidation(
                r"^blk\.[0-9]+\.attn_(k|v)\.weight$",
                _attention_layer_names(geometry, ("attn_k", "attn_v")),
            ),
            architecture=architecture,
        ),
        _shape_candidate(
            candidate_id="vocab_output_head",
            order=4,
            operator_group="vocabulary output head",
            n=geometry.vocab_size,
            k=geometry.hidden_size,
            selector=TensorRegexValidation(r"^output\.weight$", ("output.weight",)),
            architecture=architecture,
        ),
    )
    return candidates


@dataclass(frozen=True)
class MixedBitCampaignAdapter:
    """Adapter exposing the fixed mixed-bit candidate plan."""

    geometry: ModelGeometry
    provenance: MixedBitProvenance

    def candidates(self) -> tuple[MixedBitCandidateSpec, ...]:
        return plan_mixed_bit_candidates(self.geometry, self.provenance)


@dataclass(frozen=True)
class ShapeKernelCampaignAdapter:
    """Adapter exposing geometry-derived shape-kernel candidate specs."""

    geometry: ModelGeometry
    architecture: str = "gfx1201"

    def candidates(self) -> tuple[ShapeKernelCandidateSpec, ...]:
        return plan_shape_kernel_candidates(self.geometry, architecture=self.architecture)
