"""Safe, local-only discovery and provenance catalog for model artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .project_config import ProjectConfig, ProjectConfigError, load_project_config

CATALOG_RELATIVE_PATH = Path(".gpuopt/model-catalog.json")
LINKS_RELATIVE_PATH = Path(".gpuopt/model-links.json")
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_HASH_CHUNK_BYTES = 4 * 1024 * 1024


class ModelLibraryError(RuntimeError):
    """Raised for unsafe libraries, ambiguous references, or invalid catalogs."""


class ModelRole(StrEnum):
    ORIGIN = "ORIGIN"
    DERIVED = "DERIVED"


class ModelFormat(StrEnum):
    HF_SAFETENSORS = "HF_SAFETENSORS"
    GGUF = "GGUF"
    PYTORCH_BIN = "PYTORCH_BIN"


class ProvenanceStatus(StrEnum):
    ORIGIN = "ORIGIN"
    VERIFIED = "VERIFIED"
    DECLARED = "DECLARED"
    INFERRED = "INFERRED"
    MISSING = "MISSING"


class CatalogFile(BaseModel):
    """Content identity plus stable-stat coordinates used for hash-cache reuse."""

    model_config = ConfigDict(extra="forbid")

    path: str
    size: int = Field(ge=0)
    sha256: str
    device: int = Field(ge=0)
    inode: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    ctime_ns: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        portable = PurePosixPath(value)
        if portable.is_absolute() or ".." in portable.parts or str(portable) in {"", "."}:
            raise ValueError("catalog file path must be a safe relative path")
        return str(portable)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value


class ModelCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    aliases: list[str]
    role: ModelRole
    model_name: str
    variant: str | None = None
    path: Path
    model_path: Path
    format: ModelFormat
    architecture: str | None = None
    quantization: str | None = None
    sha256: str
    size: int = Field(ge=0)
    files: list[CatalogFile]
    source_model_id: str | None = None
    origin_model_id: str | None = None
    provenance_status: ProvenanceStatus
    preparation_manifest: Path | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.startswith("model-") or len(value) != 26:
            raise ValueError("model id must use the stable model-<20 hex> form")
        if any(character not in "0123456789abcdef" for character in value[6:]):
            raise ValueError("model id must use the stable model-<20 hex> form")
        return value

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, values: list[str]) -> list[str]:
        invalid = (
            not values
            or len(values) != len(set(values))
            or any(not value.strip() for value in values)
        )
        if invalid:
            raise ValueError("model aliases must be non-empty and unique")
        for value in values:
            portable = PurePosixPath(value)
            if portable.is_absolute() or ".." in portable.parts:
                raise ValueError("model aliases must be safe relative references")
        return values

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return CatalogFile.validate_sha256(value)


class ModelLibraryCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_name: Literal["gpuopt.model-library-catalog.v1"] = Field(
        default="gpuopt.model-library-catalog.v1", alias="schema"
    )
    generated_at: datetime
    model_root: Path
    entries: list[ModelCatalogEntry]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity(self) -> ModelLibraryCatalog:
        identifiers = [entry.id for entry in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("catalog model ids must be unique")
        aliases = [alias for entry in self.entries for alias in entry.aliases]
        duplicates = sorted(alias for alias, count in Counter(aliases).items() if count > 1)
        if duplicates:
            raise ValueError(f"catalog aliases must be unique: {duplicates}")
        return self

    def resolve(self, reference: str) -> ModelCatalogEntry:
        matches = [
            entry for entry in self.entries if reference == entry.id or reference in entry.aliases
        ]
        if not matches:
            raise ModelLibraryError(f"model reference not found: {reference}")
        if len(matches) != 1:
            raise ModelLibraryError(f"model reference is ambiguous: {reference}")
        return matches[0]


class ModelLinkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    derived_id: str
    derived_alias: str
    derived_sha256: str
    origin_id: str
    origin_alias: str
    origin_sha256: str
    linked_at: datetime

    @field_validator("derived_sha256", "origin_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return CatalogFile.validate_sha256(value)


class ModelLinks(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_name: Literal["gpuopt.model-links.v1"] = Field(
        default="gpuopt.model-links.v1", alias="schema"
    )
    links: list[ModelLinkRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_derived_aliases(self) -> ModelLinks:
        aliases = [link.derived_alias for link in self.links]
        if len(aliases) != len(set(aliases)):
            raise ValueError("only one explicit origin link is allowed per derived alias")
        return self


def _regular_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ModelLibraryError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ModelLibraryError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise ModelLibraryError(f"{label} is not a directory: {resolved}")
    return resolved


def _safe_child(parent: Path, child: Path, *, label: str) -> Path:
    if child.is_symlink():
        raise ModelLibraryError(f"{label} must not be a symlink: {child}")
    try:
        resolved = child.resolve(strict=True)
    except OSError as exc:
        raise ModelLibraryError(f"{label} does not exist: {child}") from exc
    if not resolved.is_relative_to(parent):
        raise ModelLibraryError(f"{label} escapes model root: {child}")
    return resolved


def _read_small_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ModelLibraryError(f"{label} must be a regular non-symlink file: {path}")
    if path.stat().st_size > _MAX_METADATA_BYTES:
        raise ModelLibraryError(f"{label} is too large: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelLibraryError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ModelLibraryError(f"{label} must contain a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: BaseModel) -> None:
    if path.is_symlink():
        raise ModelLibraryError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


@contextmanager
def _catalog_lock(project_root: Path):
    control = project_root / ".gpuopt"
    if control.is_symlink():
        raise ModelLibraryError(f"control directory must not be a symlink: {control}")
    try:
        control.mkdir(exist_ok=True)
    except OSError as exc:
        raise ModelLibraryError(f"cannot create model catalog directory: {control}") from exc
    if not control.is_dir():
        raise ModelLibraryError(f"model catalog directory is not a directory: {control}")
    lock = control / ".model-catalog.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        current = lock.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
        ):
            raise ModelLibraryError("model catalog lock is not a stable regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        current = lock.stat(follow_symlinks=False)
        if opened.st_dev != current.st_dev or opened.st_ino != current.st_ino:
            raise ModelLibraryError("model catalog lock was replaced while waiting")
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _previous_file_cache(catalog: ModelLibraryCatalog | None) -> dict[Path, CatalogFile]:
    if catalog is None:
        return {}
    return {
        (entry.path / file.path).resolve(strict=False): file
        for entry in catalog.entries
        for file in entry.files
    }


def _hash_file(path: Path, base: Path, previous: CatalogFile | None) -> CatalogFile:
    path = _safe_child(base, path, label="model file")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ModelLibraryError(f"cannot open model file safely: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ModelLibraryError(f"model file is not regular: {path}")
        coordinates = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if previous is not None and coordinates == (
            previous.device,
            previous.inode,
            previous.size,
            previous.mtime_ns,
            previous.ctime_ns,
        ):
            digest = previous.sha256
        else:
            hasher = hashlib.sha256()
            while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        after = os.fstat(descriptor)
        if coordinates != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ModelLibraryError(f"model file changed while scanning: {path}")
    finally:
        os.close(descriptor)
    current = path.stat(follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
        raise ModelLibraryError(f"model file was replaced while scanning: {path}")
    return CatalogFile(
        path=path.relative_to(base).as_posix(),
        size=before.st_size,
        sha256=digest,
        device=before.st_dev,
        inode=before.st_ino,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
    )


def _aggregate_sha(files: list[CatalogFile]) -> str:
    hasher = hashlib.sha256()
    for file in sorted(files, key=lambda item: item.path):
        hasher.update(file.path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(file.size).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(file.sha256.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _stable_id(role: ModelRole, alias: str, content_sha256: str) -> str:
    identity = f"{role.value}\0{alias}\0{content_sha256}".encode()
    return "model-" + hashlib.sha256(identity).hexdigest()[:20]


def _origin_paths(directory: Path) -> tuple[list[Path], ModelFormat]:
    index = directory / "model.safetensors.index.json"
    if index.exists() or index.is_symlink():
        document = _read_small_json(index, label="safetensors index")
        weight_map = document.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelLibraryError(f"safetensors index has no weight map: {index}")
        names: set[str] = set()
        for raw in weight_map.values():
            if not isinstance(raw, str):
                raise ModelLibraryError(f"safetensors index has an invalid shard: {index}")
            portable = PurePosixPath(raw)
            if portable.is_absolute() or ".." in portable.parts:
                raise ModelLibraryError(f"safetensors index has an unsafe shard: {raw}")
            names.add(str(portable))
        return [directory / name for name in sorted(names)], ModelFormat.HF_SAFETENSORS
    safetensors = sorted(directory.glob("*.safetensors"))
    if safetensors:
        return safetensors, ModelFormat.HF_SAFETENSORS
    gguf = sorted(directory.glob("*.gguf"))
    if gguf:
        return gguf, ModelFormat.GGUF
    pytorch = sorted(directory.glob("pytorch_model*.bin"))
    if pytorch:
        return pytorch, ModelFormat.PYTORCH_BIN
    return [], ModelFormat.HF_SAFETENSORS


def _origin_metadata(directory: Path) -> tuple[str | None, str | None]:
    config_path = directory / "config.json"
    if not config_path.exists() and not config_path.is_symlink():
        return None, None
    config = _read_small_json(config_path, label="model config")
    architecture = config.get("model_type")
    dtype = config.get("torch_dtype") or config.get("dtype")
    quantization = None
    if isinstance(dtype, str):
        quantization = {
            "bfloat16": "BF16",
            "float16": "F16",
            "float32": "F32",
        }.get(dtype.lower(), dtype.upper())
    return str(architecture) if architecture else None, quantization


def _primary_alias(entry: ModelCatalogEntry) -> str:
    return entry.aliases[0]


def _load_previous(project_root: Path) -> ModelLibraryCatalog | None:
    path = project_root / CATALOG_RELATIVE_PATH
    if not path.exists():
        return None
    try:
        return load_model_catalog(project_root)
    except ModelLibraryError:
        return None


def _load_links(project_root: Path) -> ModelLinks:
    path = project_root / LINKS_RELATIVE_PATH
    if not path.exists():
        return ModelLinks()
    if path.is_symlink() or not path.is_file():
        raise ModelLibraryError(f"model links file is unsafe: {path}")
    try:
        return ModelLinks.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ModelLibraryError(f"invalid model links file: {path}") from exc


def _manifest_candidates(entry: ModelCatalogEntry) -> list[Path]:
    model_path = entry.model_path
    candidates = [
        model_path.with_name(model_path.name + ".preparation.json"),
        entry.path / "preparation.json",
        entry.path / "preparation-manifest.json",
    ]
    return list(dict.fromkeys(candidates))


def _output_record(document: dict[str, Any], entry: ModelCatalogEntry) -> dict[str, Any] | None:
    records: list[dict[str, Any]] = []
    output = document.get("output")
    if isinstance(output, dict):
        records.append(output)
    artifacts = document.get("artifacts")
    if isinstance(artifacts, dict):
        records.extend(value for value in artifacts.values() if isinstance(value, dict))
    for record in records:
        raw_path = record.get("path")
        sha256 = record.get("sha256")
        size = record.get("size")
        if not isinstance(raw_path, str) or not isinstance(sha256, str):
            continue
        if Path(raw_path).name != entry.model_path.name:
            continue
        if sha256 == entry.sha256 and (size is None or size == entry.size):
            return record
    return None


def _match_manifest_source(
    document: dict[str, Any], entries: list[ModelCatalogEntry]
) -> ModelCatalogEntry | None:
    source = document.get("source")
    if isinstance(source, dict):
        raw_path, sha256 = source.get("path"), source.get("sha256")
        if isinstance(raw_path, str) and isinstance(sha256, str):
            candidate_path = Path(raw_path).expanduser().resolve(strict=False)
            for entry in entries:
                if entry.model_path == candidate_path and entry.sha256 == sha256:
                    return entry
    source_directory = document.get("source_model_dir")
    input_manifest = document.get("input_manifest")
    if isinstance(source_directory, str) and isinstance(input_manifest, dict):
        candidate_path = Path(source_directory).expanduser().resolve(strict=False)
        claimed = {
            value.get("path"): value
            for value in input_manifest.get("files", [])
            if isinstance(value, dict) and isinstance(value.get("path"), str)
        }
        for entry in entries:
            if entry.path != candidate_path or entry.role != ModelRole.ORIGIN:
                continue
            if all(
                file.path in claimed
                and claimed[file.path].get("sha256") == file.sha256
                and claimed[file.path].get("size") == file.size
                for file in entry.files
            ):
                return entry
    return None


def _apply_provenance(
    entries: list[ModelCatalogEntry], links: ModelLinks, warnings: list[str]
) -> list[ModelCatalogEntry]:
    origins_by_name = {
        entry.model_name: entry for entry in entries if entry.role == ModelRole.ORIGIN
    }
    links_by_alias = {link.derived_alias: link for link in links.links}
    resolved: list[ModelCatalogEntry] = []
    for entry in entries:
        if entry.role == ModelRole.ORIGIN:
            resolved.append(entry)
            continue
        origin = origins_by_name.get(entry.model_name)
        update: dict[str, Any] = {
            "origin_model_id": origin.id if origin else None,
            "provenance_status": (
                ProvenanceStatus.INFERRED if origin else ProvenanceStatus.MISSING
            ),
        }
        link = links_by_alias.get(_primary_alias(entry))
        if link is not None:
            linked_origin = next((item for item in entries if item.id == link.origin_id), None)
            if (
                link.derived_id == entry.id
                and link.derived_sha256 == entry.sha256
                and linked_origin is not None
                and linked_origin.role == ModelRole.ORIGIN
                and linked_origin.sha256 == link.origin_sha256
            ):
                update.update(
                    origin_model_id=linked_origin.id,
                    source_model_id=linked_origin.id,
                    provenance_status=ProvenanceStatus.DECLARED,
                )
            else:
                warnings.append(f"stale explicit model link ignored: {link.derived_alias}")
        for manifest in _manifest_candidates(entry):
            if not manifest.exists() and not manifest.is_symlink():
                continue
            try:
                document = _read_small_json(manifest, label="preparation manifest")
            except ModelLibraryError as exc:
                warnings.append(str(exc))
                continue
            source = _match_manifest_source(document, entries)
            if _output_record(document, entry) is not None and source is not None:
                source_origin = (
                    source.id
                    if source.role == ModelRole.ORIGIN
                    else source.origin_model_id or update.get("origin_model_id")
                )
                update.update(
                    source_model_id=source.id,
                    origin_model_id=source_origin,
                    provenance_status=ProvenanceStatus.VERIFIED,
                    preparation_manifest=manifest.resolve(strict=True),
                )
                break
        resolved.append(entry.model_copy(update=update))
    return resolved


def scan_model_library(
    project_root: str | Path, *, config: ProjectConfig | None = None
) -> ModelLibraryCatalog:
    """Scan one configured model library and atomically replace its JSON cache."""

    try:
        root = Path(project_root).expanduser().resolve(strict=True)
        selected_config = config or load_project_config(root)
    except (OSError, ProjectConfigError) as exc:
        raise ModelLibraryError("cannot resolve project configuration") from exc
    if not root.is_dir():
        raise ModelLibraryError(f"project root is not a directory: {root}")
    configured_model_root = selected_config.model_root.expanduser()
    if not configured_model_root.is_absolute():
        configured_model_root = root / configured_model_root
    model_root = _regular_directory(configured_model_root, label="model root")
    previous = _load_previous(root)
    file_cache = _previous_file_cache(previous)
    warnings: list[str] = []
    entries: list[ModelCatalogEntry] = []

    origin_root = model_root / "origin"
    if origin_root.exists() or origin_root.is_symlink():
        origin_root = _safe_child(model_root, origin_root, label="origin model directory")
        if not origin_root.is_dir():
            raise ModelLibraryError(f"origin path is not a directory: {origin_root}")
        for directory in sorted(origin_root.iterdir(), key=lambda value: value.name.lower()):
            if directory.is_symlink():
                warnings.append(f"symlinked origin model skipped: {directory}")
                continue
            if not directory.is_dir():
                continue
            directory = _safe_child(model_root, directory, label="origin model")
            try:
                paths, model_format = _origin_paths(directory)
                if not paths:
                    warnings.append(f"origin model has no recognized weight files: {directory}")
                    continue
                files = [
                    _hash_file(path, directory, file_cache.get(path.resolve(strict=False)))
                    for path in paths
                ]
                aggregate = _aggregate_sha(files)
                alias = f"origin/{directory.name}"
                architecture, quantization = _origin_metadata(directory)
                entries.append(
                    ModelCatalogEntry(
                        id=_stable_id(ModelRole.ORIGIN, alias, aggregate),
                        aliases=[alias],
                        role=ModelRole.ORIGIN,
                        model_name=directory.name,
                        path=directory,
                        model_path=directory,
                        format=model_format,
                        architecture=architecture,
                        quantization=quantization,
                        sha256=aggregate,
                        size=sum(file.size for file in files),
                        files=files,
                        provenance_status=ProvenanceStatus.ORIGIN,
                    )
                )
            except ModelLibraryError as exc:
                warnings.append(str(exc))

    derived_root = model_root / "derived"
    if derived_root.exists() or derived_root.is_symlink():
        derived_root = _safe_child(model_root, derived_root, label="derived model directory")
        if not derived_root.is_dir():
            raise ModelLibraryError(f"derived path is not a directory: {derived_root}")
        for model_directory in sorted(derived_root.iterdir(), key=lambda value: value.name.lower()):
            if model_directory.is_symlink():
                warnings.append(f"symlinked derived model skipped: {model_directory}")
                continue
            if not model_directory.is_dir():
                continue
            for variant_directory in sorted(
                model_directory.iterdir(), key=lambda value: value.name.lower()
            ):
                if variant_directory.is_symlink():
                    warnings.append(f"symlinked derived variant skipped: {variant_directory}")
                    continue
                if not variant_directory.is_dir():
                    continue
                gguf_paths = sorted(variant_directory.glob("*.gguf"))
                safe_paths = [path for path in gguf_paths if not path.is_symlink()]
                for path in gguf_paths:
                    if path.is_symlink():
                        warnings.append(f"symlinked derived model skipped: {path}")
                for model_path in safe_paths:
                    try:
                        file = _hash_file(
                            model_path,
                            variant_directory,
                            file_cache.get(model_path.resolve(strict=False)),
                        )
                    except ModelLibraryError as exc:
                        warnings.append(str(exc))
                        continue
                    short_alias = f"derived/{model_directory.name}/{variant_directory.name}"
                    alias = (
                        short_alias if len(safe_paths) == 1 else f"{short_alias}/{model_path.name}"
                    )
                    entries.append(
                        ModelCatalogEntry(
                            id=_stable_id(ModelRole.DERIVED, alias, file.sha256),
                            aliases=[alias],
                            role=ModelRole.DERIVED,
                            model_name=model_directory.name,
                            variant=variant_directory.name,
                            path=variant_directory.resolve(strict=True),
                            model_path=model_path.resolve(strict=True),
                            format=ModelFormat.GGUF,
                            quantization=variant_directory.name,
                            sha256=file.sha256,
                            size=file.size,
                            files=[file],
                            provenance_status=ProvenanceStatus.MISSING,
                        )
                    )

    links = _load_links(root)
    entries = _apply_provenance(entries, links, warnings)
    # Add a short name only where it cannot hide an origin/derived ambiguity.
    name_counts = Counter(entry.model_name for entry in entries)
    entries = [
        entry.model_copy(update={"aliases": [*entry.aliases, entry.model_name]})
        if entry.role == ModelRole.ORIGIN and name_counts[entry.model_name] == 1
        else entry
        for entry in entries
    ]
    catalog = ModelLibraryCatalog(
        generated_at=datetime.now(UTC),
        model_root=model_root,
        entries=sorted(entries, key=lambda item: (_primary_alias(item).lower(), item.id)),
        warnings=sorted(set(warnings)),
    )
    with _catalog_lock(root):
        _atomic_json(root / CATALOG_RELATIVE_PATH, catalog)
    return catalog


def load_model_catalog(project_root: str | Path) -> ModelLibraryCatalog:
    try:
        root = Path(project_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ModelLibraryError(f"project root does not exist: {project_root}") from exc
    path = root / CATALOG_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise ModelLibraryError(f"model catalog not found or unsafe: {path}")
    try:
        catalog = ModelLibraryCatalog.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ModelLibraryError(f"invalid model catalog: {path}") from exc
    model_root = _regular_directory(catalog.model_root, label="catalog model root")
    for entry in catalog.entries:
        for candidate, expected_directory in (
            (entry.path, True),
            (entry.model_path, entry.role == ModelRole.ORIGIN),
        ):
            if candidate.is_symlink():
                raise ModelLibraryError(f"catalog model path is a symlink: {candidate}")
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                raise ModelLibraryError(f"catalog model path is missing: {candidate}") from exc
            if not resolved.is_relative_to(model_root):
                raise ModelLibraryError(f"catalog model path escapes model root: {candidate}")
            if expected_directory and not resolved.is_dir():
                raise ModelLibraryError(f"catalog model directory is invalid: {candidate}")
            if not expected_directory and not resolved.is_file():
                raise ModelLibraryError(f"catalog model file is invalid: {candidate}")
    return catalog


def list_models(
    catalog: ModelLibraryCatalog,
    *,
    role: ModelRole | None = None,
    quantization: str | None = None,
) -> list[ModelCatalogEntry]:
    normalized_quantization = quantization.upper() if quantization else None
    return [
        entry
        for entry in catalog.entries
        if (role is None or entry.role == role)
        and (
            normalized_quantization is None
            or (entry.quantization or "").upper() == normalized_quantization
        )
    ]


def show_model(catalog: ModelLibraryCatalog, reference: str) -> ModelCatalogEntry:
    return catalog.resolve(reference)


def link_model(
    project_root: str | Path, derived_reference: str, origin_reference: str
) -> ModelLinkRecord:
    """Record a user-declared relation without claiming conversion verification."""

    root = Path(project_root).expanduser().resolve(strict=True)
    try:
        catalog = load_model_catalog(root)
    except ModelLibraryError:
        catalog = scan_model_library(root)
    derived = catalog.resolve(derived_reference)
    origin = catalog.resolve(origin_reference)
    if derived.role != ModelRole.DERIVED:
        raise ModelLibraryError("the linked model must be a DERIVED model")
    if origin.role != ModelRole.ORIGIN:
        raise ModelLibraryError("the link target must be an ORIGIN model")
    record = ModelLinkRecord(
        derived_id=derived.id,
        derived_alias=_primary_alias(derived),
        derived_sha256=derived.sha256,
        origin_id=origin.id,
        origin_alias=_primary_alias(origin),
        origin_sha256=origin.sha256,
        linked_at=datetime.now(UTC),
    )
    with _catalog_lock(root):
        links = _load_links(root)
        retained = [link for link in links.links if link.derived_alias != record.derived_alias]
        _atomic_json(root / LINKS_RELATIVE_PATH, ModelLinks(links=[*retained, record]))
    scan_model_library(root)
    return record


__all__ = [
    "CATALOG_RELATIVE_PATH",
    "LINKS_RELATIVE_PATH",
    "CatalogFile",
    "ModelCatalogEntry",
    "ModelFormat",
    "ModelLibraryCatalog",
    "ModelLibraryError",
    "ModelLinkRecord",
    "ModelLinks",
    "ModelRole",
    "ProvenanceStatus",
    "link_model",
    "list_models",
    "load_model_catalog",
    "scan_model_library",
    "show_model",
]
