"""Bounded GGUF tensor inventory and llama-imatrix statistics adapters."""

from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path
from typing import BinaryIO

from .mixed_precision import (
    Precision,
    SensitivityEvidence,
    TensorInventory,
    TensorInventoryEntry,
    TensorNamingConfig,
    TensorSensitivityScore,
)
from .models import ArtifactRef

_TYPE_LAYOUT: dict[int, tuple[Precision, int, int]] = {
    0: (Precision.F32, 1, 4),
    1: (Precision.F16, 1, 2),
    8: (Precision.Q8_0, 32, 34),
    12: (Precision.Q4_K, 256, 144),
    13: (Precision.Q5_K, 256, 176),
    14: (Precision.Q6_K, 256, 210),
    30: (Precision.BF16, 1, 2),
}


class GGUFInventoryError(RuntimeError):
    pass


def _exact(source: BinaryIO, size: int) -> bytes:
    data = source.read(size)
    if len(data) != size:
        raise GGUFInventoryError("truncated GGUF")
    return data


def _string(source: BinaryIO) -> str:
    length = struct.unpack("<Q", _exact(source, 8))[0]
    if length > 64 * 1024 * 1024:
        raise GGUFInventoryError("unreasonable GGUF string length")
    try:
        return _exact(source, length).decode("utf-8")
    except UnicodeDecodeError as error:
        raise GGUFInventoryError("invalid GGUF UTF-8") from error


def _skip_value(source: BinaryIO, value_type: int) -> None:
    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if value_type in scalar_sizes:
        _exact(source, scalar_sizes[value_type])
        return
    if value_type == 8:
        _string(source)
        return
    if value_type == 9:
        element_type = struct.unpack("<I", _exact(source, 4))[0]
        count = struct.unpack("<Q", _exact(source, 8))[0]
        if count > 100_000_000:
            raise GGUFInventoryError("unreasonable GGUF array length")
        for _ in range(count):
            _skip_value(source, element_type)
        return
    raise GGUFInventoryError(f"unsupported GGUF metadata type: {value_type}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_gguf_inventory(
    model_path: str | Path,
    *,
    source_artifact: ArtifactRef,
    naming: TensorNamingConfig,
) -> TensorInventory:
    """Read names, shapes, storage types, and logical payload sizes from a GGUF."""

    candidate = Path(model_path).expanduser()
    if candidate.is_symlink():
        raise GGUFInventoryError("GGUF must be a regular non-symlink file")
    path = candidate.resolve(strict=True)
    if not path.is_file():
        raise GGUFInventoryError("GGUF must be a regular non-symlink file")
    if path.stat().st_size != source_artifact.size or _sha256(path) != source_artifact.sha256:
        raise GGUFInventoryError("GGUF does not match source ArtifactRef")
    with path.open("rb") as source:
        if _exact(source, 4) != b"GGUF":
            raise GGUFInventoryError("invalid GGUF magic")
        version = struct.unpack("<I", _exact(source, 4))[0]
        if version not in {2, 3}:
            raise GGUFInventoryError(f"unsupported GGUF version: {version}")
        tensor_count = struct.unpack("<Q", _exact(source, 8))[0]
        metadata_count = struct.unpack("<Q", _exact(source, 8))[0]
        if tensor_count < 1 or tensor_count > 10_000_000 or metadata_count > 10_000_000:
            raise GGUFInventoryError("unreasonable GGUF header counts")
        for _ in range(metadata_count):
            _string(source)
            _skip_value(source, struct.unpack("<I", _exact(source, 4))[0])
        entries: list[TensorInventoryEntry] = []
        for _ in range(tensor_count):
            name = _string(source)
            dimensions = struct.unpack("<I", _exact(source, 4))[0]
            if not 1 <= dimensions <= 16:
                raise GGUFInventoryError("unreasonable tensor rank")
            shape = [struct.unpack("<Q", _exact(source, 8))[0] for _ in range(dimensions)]
            tensor_type = struct.unpack("<I", _exact(source, 4))[0]
            _exact(source, 8)  # relative tensor data offset
            precision, block_elements, block_bytes = _TYPE_LAYOUT.get(
                tensor_type, (Precision.OTHER, 1, 0)
            )
            elements = 1
            for dimension in shape:
                elements *= dimension
            storage_bytes = (
                (elements + block_elements - 1) // block_elements * block_bytes
                if block_bytes
                else 0
            )
            group, protected = naming.classify(name)
            entries.append(
                TensorInventoryEntry(
                    name=name,
                    shape=shape,
                    elements=elements,
                    storage_bytes=storage_bytes,
                    precision=precision,
                    operator_group=group,
                    protected=protected,
                    quantizable=dimensions >= 2,
                )
            )
    return TensorInventory(
        model_sha256=source_artifact.sha256,
        entries=entries,
        source_artifact=source_artifact,
    )


_IMATRIX_ROW = re.compile(
    r"^\s*(?P<layer>-|[0-9]+)\s+(?P<tensor>[A-Za-z0-9_.-]+)\s+"
    r"(?P<score>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s+"
)


def parse_imatrix_statistics(
    output: str,
    inventory: TensorInventory,
    *,
    calibration_artifact: ArtifactRef,
    quality_ablation_artifacts: list[ArtifactRef] | None = None,
) -> SensitivityEvidence:
    """Normalize the stable leading columns from llama-imatrix --show-statistics."""

    by_name = {entry.name: entry for entry in inventory.entries}
    scores: dict[str, float] = {}
    for line in output.splitlines():
        match = _IMATRIX_ROW.match(line)
        if not match:
            continue
        layer = match.group("layer")
        tensor = match.group("tensor")
        candidates = []
        if layer != "-":
            candidates.extend(
                (
                    f"blk.{layer}.{tensor}.weight",
                    f"blk.{layer}.{tensor}",
                )
            )
        candidates.extend((tensor, f"{tensor}.weight"))
        name = next((candidate for candidate in candidates if candidate in by_name), None)
        if name is None:
            suffix = f".{tensor}.weight"
            matches = [
                entry.name
                for entry in inventory.entries
                if entry.name.endswith(suffix)
                and (layer == "-" or entry.name.startswith(f"blk.{layer}."))
            ]
            if len(matches) == 1:
                name = matches[0]
        if name is not None:
            scores[name] = float(match.group("score"))
    if not scores:
        raise GGUFInventoryError("llama-imatrix output contained no recognized tensor rows")
    missing = sorted(
        entry.name
        for entry in inventory.entries
        if entry.quantizable
        and entry.precision in {Precision.BF16, Precision.F16, Precision.F32}
        and entry.operator_group != "other"
        and entry.name not in scores
    )
    return SensitivityEvidence(
        model_sha256=inventory.model_sha256,
        scores=[
            TensorSensitivityScore(
                tensor_name=name,
                score=score,
                source_metric="imatrix.sum_activation_squared",
            )
            for name, score in sorted(scores.items())
        ],
        calibration_artifacts=[calibration_artifact],
        quality_ablation_artifacts=quality_ablation_artifacts or [],
        missing_tensors=missing,
    )


__all__ = [
    "GGUFInventoryError",
    "inspect_gguf_inventory",
    "parse_imatrix_statistics",
]
