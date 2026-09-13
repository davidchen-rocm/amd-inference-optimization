"""Frozen long-context KV-cache benchmark coordinates and acceptance gates.

This module is deliberately independent from the workflow models.  It describes
one llama-bench arm at a time because passing multiple values to both ``-ctk``
and ``-ctv`` creates a Cartesian product, including unwanted mixed K/V types.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .command import StageCommand, argv_sha256, validate_argv

__all__ = [
    "KV_CACHE_DEPTHS",
    "KVCampaignPlan",
    "KVCacheArm",
    "KVCacheArmPlan",
    "KVCacheBenchmarkProtocol",
    "KVCacheBenchmarkRecord",
    "KVCacheBenchmarkResult",
    "KVCacheCanaryPlan",
    "KVCacheError",
    "KVCacheGateResult",
    "KVCacheScoredPlan",
    "KVCacheType",
    "KVModelGeometry",
    "KVRuntimeCapabilities",
    "QWEN3_8B_KV_BYTES",
    "build_kv_campaign_plan",
    "evaluate_kv_cache_gate",
    "kv_cache_bytes_for_geometry",
    "parse_kv_cache_bench_json",
    "parse_kv_cache_benchmark_json",
    "qwen3_8b_kv_cache_bytes",
]


KV_CACHE_DEPTHS = (4096, 16384, 28672)
_LONG_DEPTHS = (16384, 28672)
_QWEN3_8B_LAYERS = 36
_QWEN3_8B_KV_HEADS = 8
_QWEN3_8B_HEAD_DIM = 128


class KVCacheError(ValueError):
    """A KV-cache coordinate or llama-bench result is invalid."""


class KVCacheType(StrEnum):
    """KV-cache types supported by the fixed experiment."""

    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"


@dataclass(frozen=True)
class KVCacheArm:
    """One matching K/V cache-type arm with flash attention forced on."""

    cache_type: KVCacheType
    flash_attn: bool = True

    def __post_init__(self) -> None:
        try:
            cache_type = KVCacheType(self.cache_type)
        except (TypeError, ValueError) as error:
            raise KVCacheError(f"unsupported KV-cache type: {self.cache_type!r}") from error
        object.__setattr__(self, "cache_type", cache_type)
        if self.flash_attn is not True:
            raise KVCacheError("KV-cache benchmark arms require flash attention")

    @property
    def type_k(self) -> KVCacheType:
        return self.cache_type

    @property
    def type_v(self) -> KVCacheType:
        return self.cache_type


def _canonical_sha256(document: object) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class KVCacheBenchmarkProtocol:
    """One exact, model-identified long-context llama-bench coordinate."""

    llama_bench_path: str
    model_path: str
    arm: KVCacheArm
    model_quantization: str = "Q6_K"
    depths: tuple[int, ...] = KV_CACHE_DEPTHS
    generation_tokens: int = 128
    repetitions: int = 5
    batch_size: int = 2048
    ubatch_size: int = 512
    threads: int = 12
    gpu_layers: int = 999
    device_id: int = 0
    device_selector: str = "ROCm0"
    warmup: bool = True
    timeout_seconds: float = 1800
    extra_args: tuple[str, ...] = ()
    cwd: str = "."
    environment: Mapping[str, str] = field(default_factory=dict)
    unset_environment: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.arm, KVCacheArm):
            raise KVCacheError("arm must be a KVCacheArm")
        if not self.model_quantization.strip():
            raise KVCacheError("model_quantization must not be empty")
        if self.depths != KV_CACHE_DEPTHS:
            raise KVCacheError(
                "KV-cache protocol depths must be exactly 4096, 16384, and 28672"
            )
        if self.generation_tokens != 128:
            raise KVCacheError("KV-cache protocol requires exactly 128 generation tokens")
        if self.repetitions not in {5, 12}:
            raise KVCacheError("KV-cache protocol requires five samples or a 12-sample retry")
        if self.batch_size < 1 or self.ubatch_size < 1:
            raise KVCacheError("batch sizes must be positive")
        if self.ubatch_size > self.batch_size:
            raise KVCacheError("ubatch_size cannot exceed batch_size")
        if self.threads < 1:
            raise KVCacheError("threads must be positive")
        if self.gpu_layers < 0 or self.device_id != 0:
            raise KVCacheError("KV-cache protocol requires GPU device zero")
        if self.device_selector != "ROCm0":
            raise KVCacheError("KV-cache protocol requires device selector ROCm0")
        if self.warmup is not True:
            raise KVCacheError("KV-cache protocol requires llama-bench warmup")
        if self.timeout_seconds <= 0:
            raise KVCacheError("timeout_seconds must be positive")
        if self.extra_args:
            validate_argv(self.extra_args)
        overlap = set(self.environment) & set(self.unset_environment)
        if overlap:
            raise KVCacheError(
                "protocol environment cannot set and unset the same names: "
                + ", ".join(sorted(overlap))
            )

    @property
    def argv(self) -> tuple[str, ...]:
        argv = (
            str(Path(self.llama_bench_path).resolve()),
            "-m",
            str(Path(self.model_path).resolve()),
            "-p",
            "0",
            "-d",
            ",".join(str(depth) for depth in self.depths),
            "-n",
            str(self.generation_tokens),
            "-b",
            str(self.batch_size),
            "-ub",
            str(self.ubatch_size),
            "-t",
            str(self.threads),
            "-r",
            str(self.repetitions),
            "-ctk",
            self.arm.type_k.value,
            "-ctv",
            self.arm.type_v.value,
            "-fa",
            "on",
            "-ngl",
            str(self.gpu_layers),
            "-mg",
            "0",
            "-dev",
            self.device_selector,
            "-o",
            "json",
            "-oe",
            "none",
        )
        return argv + tuple(self.extra_args)

    @property
    def argv_hash(self) -> str:
        return argv_sha256(self.argv)

    @property
    def semantic_hash(self) -> str:
        """Hash semantics while excluding binary/model paths and runtime deltas."""

        return _canonical_sha256(
            {
                "schema_version": 1,
                "model_quantization": self.model_quantization,
                "type_k": self.arm.type_k.value,
                "type_v": self.arm.type_v.value,
                "flash_attn": self.arm.flash_attn,
                "depths": self.depths,
                "generation_tokens": self.generation_tokens,
                "repetitions": self.repetitions,
                "batch_size": self.batch_size,
                "ubatch_size": self.ubatch_size,
                "threads": self.threads,
                "gpu_layers": self.gpu_layers,
                "device_id": self.device_id,
                "device_selector": self.device_selector,
                "warmup": self.warmup,
                "extra_args": self.extra_args,
            }
        )

    @property
    def protocol_hash(self) -> str:
        """Compatibility name for the semantic protocol hash."""

        return self.semantic_hash

    def command(self, *, name: str | None = None) -> StageCommand:
        return StageCommand(
            name=name or f"kv-{self.arm.cache_type.value}",
            argv=self.argv,
            cwd=self.cwd,
            env=self.environment,
            unset_env=self.unset_environment,
            timeout_seconds=self.timeout_seconds,
        )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "argv": list(self.argv),
                "argv_hash": self.argv_hash,
                "semantic_hash": self.semantic_hash,
                "protocol_hash": self.protocol_hash,
            }
        )
        return result


@dataclass(frozen=True)
class KVCacheBenchmarkRecord:
    """Validated throughput samples for one cache type and depth."""

    depth: int
    cache_type: KVCacheType
    mean_tokens_per_second: float
    stddev_tokens_per_second: float
    samples_tokens_per_second: tuple[float, ...]
    coefficient_of_variation_percent: float
    raw: dict[str, Any]

    @property
    def sample_count(self) -> int:
        return len(self.samples_tokens_per_second)

    @property
    def cv_percent(self) -> float:
        return self.coefficient_of_variation_percent

    @property
    def test_id(self) -> str:
        return f"{self.cache_type.value}-d{self.depth}"


@dataclass(frozen=True)
class KVCacheBenchmarkResult:
    """A complete three-depth result for one matching KV-cache arm."""

    arm: KVCacheArm
    records: tuple[KVCacheBenchmarkRecord, ...]
    raw: tuple[dict[str, Any], ...]

    def by_depth(self) -> dict[int, KVCacheBenchmarkRecord]:
        return {record.depth: record for record in self.records}

    @property
    def max_cv_percent(self) -> float:
        return max(record.cv_percent for record in self.records)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _decode_payload(
    payload: str | bytes | Path | list[object] | dict[str, object],
) -> object:
    try:
        if isinstance(payload, Path):
            return json.loads(payload.read_text(encoding="utf-8"))
        if isinstance(payload, bytes):
            return json.loads(payload.decode("utf-8"))
        if isinstance(payload, str):
            return json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KVCacheError(f"invalid llama-bench JSON: {error}") from error
    return payload


def _strict_int(row: Mapping[str, object], field: str, row_index: int) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise KVCacheError(f"llama-bench row {row_index} field {field!r} must be an integer")
    return value


def _positive_number(value: object, field: str, row_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KVCacheError(f"llama-bench row {row_index} field {field!r} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise KVCacheError(
            f"llama-bench row {row_index} field {field!r} must be finite and positive"
        )
    return result


def parse_kv_cache_benchmark_json(
    payload: str | bytes | Path | list[object] | dict[str, object],
    *,
    arm: KVCacheArm | None = None,
    protocol: KVCacheBenchmarkProtocol | None = None,
    depths: tuple[int, ...] = KV_CACHE_DEPTHS,
    generation_tokens: int = 128,
    repetitions: int = 5,
    max_cv_percent: float = 2.0,
) -> KVCacheBenchmarkResult:
    """Strictly parse one complete matching-type, flash-attention benchmark arm."""

    if protocol is not None:
        if arm is not None and arm != protocol.arm:
            raise KVCacheError("arm disagrees with protocol.arm")
        arm = protocol.arm
        depths = protocol.depths
        generation_tokens = protocol.generation_tokens
        repetitions = protocol.repetitions
    if arm is None:
        raise KVCacheError("an expected KVCacheArm or protocol is required")
    if depths != KV_CACHE_DEPTHS:
        raise KVCacheError("expected depths must be exactly 4096, 16384, and 28672")
    if generation_tokens != 128 or repetitions not in {5, 12}:
        raise KVCacheError("parser requires n_gen=128 and five or 12 samples per depth")
    if not math.isfinite(max_cv_percent) or max_cv_percent < 0:
        raise KVCacheError("max_cv_percent must be finite and non-negative")

    decoded = _decode_payload(payload)
    if isinstance(decoded, dict):
        rows: list[object] = [decoded]
    elif isinstance(decoded, list):
        rows = decoded
    else:
        raise KVCacheError("llama-bench JSON must be an object or array")
    if len(rows) != len(depths):
        raise KVCacheError(
            f"llama-bench JSON must contain exactly {len(depths)} rows; got {len(rows)}"
        )

    raw_rows: list[dict[str, Any]] = []
    records_by_depth: dict[int, KVCacheBenchmarkRecord] = {}
    for index, untyped in enumerate(rows):
        if not isinstance(untyped, dict):
            raise KVCacheError(f"llama-bench row {index} must be an object")
        row = dict(untyped)
        depth = _strict_int(row, "n_depth", index)
        if depth not in depths:
            raise KVCacheError(f"llama-bench row {index} has unexpected n_depth {depth}")
        if depth in records_by_depth:
            raise KVCacheError(f"llama-bench JSON repeats n_depth {depth}")
        if _strict_int(row, "n_prompt", index) != 0:
            raise KVCacheError(f"llama-bench row {index} must have n_prompt=0")
        if _strict_int(row, "n_gen", index) != generation_tokens:
            raise KVCacheError(
                f"llama-bench row {index} must have n_gen={generation_tokens}"
            )
        if row.get("type_k") != arm.type_k.value:
            raise KVCacheError(
                f"llama-bench row {index} type_k does not match {arm.type_k.value}"
            )
        if row.get("type_v") != arm.type_v.value:
            raise KVCacheError(
                f"llama-bench row {index} type_v does not match {arm.type_v.value}"
            )
        if _strict_int(row, "flash_attn", index) != 1:
            raise KVCacheError(f"llama-bench row {index} did not force flash attention on")

        raw_samples = row.get("samples_ts")
        if not isinstance(raw_samples, list) or len(raw_samples) != repetitions:
            raise KVCacheError(
                f"llama-bench row {index} must contain exactly {repetitions} samples_ts"
            )
        samples = tuple(
            _positive_number(sample, "samples_ts", index) for sample in raw_samples
        )
        sample_mean = statistics.fmean(samples)
        sample_stddev = statistics.stdev(samples)
        cv_percent = sample_stddev / sample_mean * 100
        if cv_percent > max_cv_percent:
            raise KVCacheError(
                f"llama-bench row {index} CV {cv_percent:.4f}% exceeds "
                f"{max_cv_percent:.4f}%"
            )

        reported_mean = _positive_number(row.get("avg_ts"), "avg_ts", index)
        # samples_ts is printed to three decimals while avg_ts is printed to six.
        if not math.isclose(reported_mean, sample_mean, rel_tol=1e-5, abs_tol=1e-3):
            raise KVCacheError(
                f"llama-bench row {index} avg_ts is inconsistent with samples_ts"
            )
        raw_rows.append(row)
        records_by_depth[depth] = KVCacheBenchmarkRecord(
            depth=depth,
            cache_type=arm.cache_type,
            mean_tokens_per_second=reported_mean,
            stddev_tokens_per_second=sample_stddev,
            samples_tokens_per_second=samples,
            coefficient_of_variation_percent=cv_percent,
            raw=row,
        )

    missing = set(depths) - set(records_by_depth)
    if missing:  # pragma: no cover - row count and duplicate checks normally report first
        raise KVCacheError("llama-bench JSON is missing depths: " + ", ".join(map(str, missing)))
    records = tuple(records_by_depth[depth] for depth in depths)
    return KVCacheBenchmarkResult(arm=arm, records=records, raw=tuple(raw_rows))


# A shorter spelling is convenient at call sites and remains explicit about the
# llama-bench-specific input format.
parse_kv_cache_bench_json = parse_kv_cache_benchmark_json


def qwen3_8b_kv_cache_bytes(depth: int, cache_type: KVCacheType) -> int:
    """Return exact packed K+V bytes for Qwen3-8B at ``depth`` tokens.

    Qwen3-8B has 36 layers, 8 KV heads, and a head dimension of 128.  F16 stores
    two bytes per element.  llama.cpp Q8_0 and Q4_0 store blocks of 32 elements
    in 34 and 18 bytes respectively, including their FP16 scale.
    """

    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise KVCacheError("depth must be a positive integer")
    try:
        resolved_type = KVCacheType(cache_type)
    except (TypeError, ValueError) as error:
        raise KVCacheError(f"unsupported KV-cache type: {cache_type!r}") from error
    elements_per_tensor = (
        depth * _QWEN3_8B_LAYERS * _QWEN3_8B_KV_HEADS * _QWEN3_8B_HEAD_DIM
    )
    if resolved_type == KVCacheType.F16:
        bytes_per_tensor = elements_per_tensor * 2
    else:
        block_bytes = 34 if resolved_type == KVCacheType.Q8_0 else 18
        if elements_per_tensor % 32:  # pragma: no cover - fixed Qwen geometry divides evenly
            raise KVCacheError("Qwen3-8B KV element count is not quantization-block aligned")
        bytes_per_tensor = elements_per_tensor // 32 * block_bytes
    return bytes_per_tensor * 2  # K and V


QWEN3_8B_KV_BYTES: Mapping[KVCacheType, Mapping[int, int]] = {
    cache_type: {
        depth: qwen3_8b_kv_cache_bytes(depth, cache_type) for depth in KV_CACHE_DEPTHS
    }
    for cache_type in KVCacheType
}


@dataclass(frozen=True)
class KVCacheGateResult:
    """Auditable result of the fixed long-context performance/stability gate."""

    passed: bool
    improvements_percent: Mapping[int, float]
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.passed


def evaluate_kv_cache_gate(
    baseline: KVCacheBenchmarkResult,
    candidate: KVCacheBenchmarkResult,
    *,
    max_cv_percent: float = 2.0,
) -> KVCacheGateResult:
    """Compare a quantized-cache arm with F16 using the fixed acceptance budget."""

    if baseline.arm.cache_type != KVCacheType.F16:
        raise KVCacheError("KV-cache gate baseline must be the F16 arm")
    if candidate.arm.cache_type == KVCacheType.F16:
        raise KVCacheError("KV-cache gate candidate must be a quantized cache arm")
    if not math.isfinite(max_cv_percent) or max_cv_percent < 0:
        raise KVCacheError("max_cv_percent must be finite and non-negative")
    baseline_by_depth = baseline.by_depth()
    candidate_by_depth = candidate.by_depth()
    expected = set(KV_CACHE_DEPTHS)
    if set(baseline_by_depth) != expected or set(candidate_by_depth) != expected:
        raise KVCacheError("baseline and candidate must each contain the three fixed depths")

    improvements = {
        depth: (
            candidate_by_depth[depth].mean_tokens_per_second
            / baseline_by_depth[depth].mean_tokens_per_second
            - 1
        )
        * 100
        for depth in KV_CACHE_DEPTHS
    }

    def meets(actual: float, minimum: float) -> bool:
        # Ratio arithmetic can put an exact boundary a few ulps below its
        # mathematical value (for example, 97/100 may report -3.0000000000000027%).
        return actual >= minimum or math.isclose(actual, minimum, abs_tol=1e-12)

    stability = all(
        record.cv_percent <= max_cv_percent
        for result in (baseline, candidate)
        for record in result.records
    )
    checks = {
        "cv_at_most_2_percent": stability,
        "d4096_at_least_minus_3_percent": meets(improvements[4096], -3),
        "d16384_non_regressing": meets(improvements[16384], 0),
        "d28672_non_regressing": meets(improvements[28672], 0),
        "one_long_depth_at_least_5_percent": any(
            meets(improvements[depth], 5) for depth in _LONG_DEPTHS
        ),
    }
    reasons: list[str] = []
    if not stability:
        reasons.append(f"a baseline or candidate sample CV exceeds {max_cv_percent:.4f}%")
    if not checks["d4096_at_least_minus_3_percent"]:
        reasons.append("d4096 regresses by more than 3%")
    if not checks["d16384_non_regressing"]:
        reasons.append("d16384 regresses")
    if not checks["d28672_non_regressing"]:
        reasons.append("d28672 regresses")
    if not checks["one_long_depth_at_least_5_percent"]:
        reasons.append("neither d16384 nor d28672 improves by at least 5%")
    if not reasons:
        reasons.append("all KV-cache performance and stability gates passed")
    return KVCacheGateResult(
        passed=all(checks.values()),
        improvements_percent=improvements,
        checks=checks,
        reasons=tuple(reasons),
    )


def _mapping_positive_int(
    value: Mapping[str, object],
    names: tuple[str, ...],
    label: str,
    *,
    default: int | None = None,
) -> int:
    raw: object = default
    for name in names:
        if name in value:
            raw = value[name]
            break
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise KVCacheError(f"{label} must be a positive integer")
    return raw


@dataclass(frozen=True)
class KVModelGeometry:
    """Model fields needed to derive safe KV-cache depths and byte counts."""

    layers: int
    kv_heads: int
    head_dim: int
    max_context_tokens: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise KVCacheError(f"model geometry {name} must be a positive integer")
        if self.head_dim % 32:
            raise KVCacheError("model head_dim must be divisible by the 32-element KV block")

    @classmethod
    def from_value(
        cls,
        value: KVModelGeometry | Mapping[str, object],
    ) -> KVModelGeometry:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise KVCacheError("model_geometry must be an object")
        return cls(
            layers=_mapping_positive_int(
                value, ("layers", "n_layers", "block_count"), "layers"
            ),
            kv_heads=_mapping_positive_int(
                value,
                ("kv_heads", "n_kv_heads", "attention_head_count_kv"),
                "kv_heads",
            ),
            head_dim=_mapping_positive_int(
                value,
                ("head_dim", "attention_head_dim", "attention_key_length"),
                "head_dim",
            ),
            max_context_tokens=_mapping_positive_int(
                value,
                ("max_context_tokens", "context_length", "n_ctx_train"),
                "max_context_tokens",
            ),
        )


@dataclass(frozen=True)
class KVRuntimeCapabilities:
    """Runtime facts required before emitting all three campaign arms."""

    supported_cache_types: tuple[KVCacheType, ...]
    flash_attention: bool
    max_context_tokens: int
    minimum_depth: int = 1
    device_selector: str = "ROCm0"

    def __post_init__(self) -> None:
        normalized: list[KVCacheType] = []
        try:
            for cache_type in self.supported_cache_types:
                resolved = KVCacheType(cache_type)
                if resolved not in normalized:
                    normalized.append(resolved)
        except (TypeError, ValueError) as error:
            raise KVCacheError("runtime has an unsupported cache type") from error
        object.__setattr__(self, "supported_cache_types", tuple(normalized))
        if self.flash_attention is not True:
            raise KVCacheError("KV-cache campaign requires flash-attention support")
        if (
            isinstance(self.max_context_tokens, bool)
            or not isinstance(self.max_context_tokens, int)
            or self.max_context_tokens <= 0
        ):
            raise KVCacheError("runtime max_context_tokens must be a positive integer")
        if (
            isinstance(self.minimum_depth, bool)
            or not isinstance(self.minimum_depth, int)
            or self.minimum_depth <= 0
        ):
            raise KVCacheError("runtime minimum_depth must be a positive integer")
        if self.device_selector != "ROCm0":
            raise KVCacheError("KV-cache campaign requires device selector ROCm0")

    @classmethod
    def from_value(
        cls,
        value: KVRuntimeCapabilities | Mapping[str, object],
        *,
        default_max_context_tokens: int,
    ) -> KVRuntimeCapabilities:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise KVCacheError("capabilities must be an object")
        raw_types = value.get(
            "supported_cache_types",
            value.get("cache_types", tuple(KVCacheType)),
        )
        if isinstance(raw_types, (str, bytes)) or not isinstance(raw_types, Sequence):
            raise KVCacheError("supported_cache_types must be a sequence")
        flash_attention = value.get(
            "flash_attention",
            value.get("flash_attn", value.get("supports_flash_attention")),
        )
        if not isinstance(flash_attention, bool):
            raise KVCacheError("flash_attention capability must be boolean")
        device = value.get("device_selector", "ROCm0")
        if not isinstance(device, str):
            raise KVCacheError("device_selector must be a string")
        return cls(
            supported_cache_types=tuple(raw_types),  # type: ignore[arg-type]
            flash_attention=flash_attention,
            max_context_tokens=_mapping_positive_int(
                value,
                ("max_context_tokens", "context_length", "max_context"),
                "runtime max_context_tokens",
                default=default_max_context_tokens,
            ),
            minimum_depth=_mapping_positive_int(
                value,
                ("minimum_depth", "min_depth"),
                "runtime minimum_depth",
                default=1,
            ),
            device_selector=device,
        )


@dataclass(frozen=True)
class KVCacheCanaryPlan:
    """The minimum-work check run before a costly scored arm."""

    depth: int
    generation_tokens: int = 1
    repetitions: int = 1


@dataclass(frozen=True)
class KVCacheScoredPlan:
    """The fixed scored 4K/16K/28K run for one arm."""

    depths: tuple[int, ...]
    generation_tokens: int = 128
    repetitions: int = 5
    noisy_retry_repetitions: int = 12
    max_cv_percent: float = 2.0


@dataclass(frozen=True)
class KVCacheArmPlan:
    """Serializable canary and scored coordinates for one matching cache type."""

    arm: KVCacheArm
    canary: KVCacheCanaryPlan
    scored: KVCacheScoredPlan
    kv_bytes_by_depth: Mapping[int, int]

    @property
    def cache_type(self) -> KVCacheType:
        return self.arm.cache_type

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_type": self.cache_type.value,
            "type_k": self.arm.type_k.value,
            "type_v": self.arm.type_v.value,
            "flash_attn": self.arm.flash_attn,
            "canary": asdict(self.canary),
            "scored": asdict(self.scored),
            "kv_bytes_by_depth": {
                str(depth): byte_count
                for depth, byte_count in self.kv_bytes_by_depth.items()
            },
        }


@dataclass(frozen=True)
class KVCampaignPlan:
    """Pure campaign-adapter plan consumed by orchestration code."""

    schema: str
    model_geometry: KVModelGeometry
    capabilities: KVRuntimeCapabilities
    effective_context_tokens: int
    arms: tuple[KVCacheArmPlan, ...]

    @property
    def benchmark_depths(self) -> tuple[int, ...]:
        return self.arms[0].scored.depths

    @property
    def canary_depth(self) -> int:
        return self.arms[0].canary.depth

    @property
    def semantic_hash(self) -> str:
        return _canonical_sha256(self._document())

    @property
    def plan_hash(self) -> str:
        return self.semantic_hash

    def _document(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "model_geometry": asdict(self.model_geometry),
            "capabilities": {
                **asdict(self.capabilities),
                "supported_cache_types": [
                    cache_type.value
                    for cache_type in self.capabilities.supported_cache_types
                ],
            },
            "effective_context_tokens": self.effective_context_tokens,
            "arms": [arm.to_dict() for arm in self.arms],
        }

    def to_dict(self) -> dict[str, object]:
        document = self._document()
        document["semantic_hash"] = self.semantic_hash
        document["plan_hash"] = self.plan_hash
        return document


def kv_cache_bytes_for_geometry(
    geometry: KVModelGeometry | Mapping[str, object],
    depth: int,
    cache_type: KVCacheType,
) -> int:
    """Return packed K+V bytes for an arbitrary block-aligned model geometry."""

    resolved_geometry = KVModelGeometry.from_value(geometry)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise KVCacheError("depth must be a positive integer")
    try:
        resolved_type = KVCacheType(cache_type)
    except (TypeError, ValueError) as error:
        raise KVCacheError(f"unsupported KV-cache type: {cache_type!r}") from error
    elements_per_tensor = (
        depth
        * resolved_geometry.layers
        * resolved_geometry.kv_heads
        * resolved_geometry.head_dim
    )
    if resolved_type == KVCacheType.F16:
        bytes_per_tensor = elements_per_tensor * 2
    else:
        block_bytes = 34 if resolved_type == KVCacheType.Q8_0 else 18
        if elements_per_tensor % 32:  # pragma: no cover - geometry validates head alignment
            raise KVCacheError("KV geometry is not 32-element block aligned")
        bytes_per_tensor = elements_per_tensor // 32 * block_bytes
    return bytes_per_tensor * 2


def build_kv_campaign_plan(
    model_geometry: KVModelGeometry | Mapping[str, object],
    capabilities: KVRuntimeCapabilities | Mapping[str, object],
) -> KVCampaignPlan:
    """Build f16/q8_0/q4_0 canary and scored plans from declared capabilities."""

    geometry = KVModelGeometry.from_value(model_geometry)
    runtime = KVRuntimeCapabilities.from_value(
        capabilities,
        default_max_context_tokens=geometry.max_context_tokens,
    )
    required_types = tuple(KVCacheType)
    missing = [
        cache_type.value
        for cache_type in required_types
        if cache_type not in runtime.supported_cache_types
    ]
    if missing:
        raise KVCacheError(
            "runtime does not support required KV-cache arms: " + ", ".join(missing)
        )
    effective_context = min(geometry.max_context_tokens, runtime.max_context_tokens)
    depths = tuple(kib * 1024 for kib in (4, 16, 28))
    generation_tokens = 128
    if depths[-1] + generation_tokens > effective_context:
        raise KVCacheError(
            "model/runtime context cannot fit the required d28672 + tg128 coordinate"
        )
    if runtime.minimum_depth + 1 > effective_context:
        raise KVCacheError("model/runtime context cannot fit the minimum-depth canary")
    canary = KVCacheCanaryPlan(depth=runtime.minimum_depth)
    scored = KVCacheScoredPlan(depths=depths)
    arms = tuple(
        KVCacheArmPlan(
            arm=KVCacheArm(cache_type),
            canary=canary,
            scored=scored,
            kv_bytes_by_depth={
                depth: kv_cache_bytes_for_geometry(geometry, depth, cache_type)
                for depth in depths
            },
        )
        for cache_type in required_types
    )
    return KVCampaignPlan(
        schema="gpuopt.kv-cache-campaign-plan.v1",
        model_geometry=geometry,
        capabilities=runtime,
        effective_context_tokens=effective_context,
        arms=arms,
    )
