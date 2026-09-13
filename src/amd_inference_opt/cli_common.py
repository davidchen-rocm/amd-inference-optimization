"""Shared CLI configuration, JSON rendering, and error handling."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

import typer
import yaml


def _default_store() -> Path:
    configured = os.environ.get("GPUOPT_STORE")
    return Path(configured) if configured else Path.cwd() / ".gpuopt"


def _document(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise typer.BadParameter(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise typer.BadParameter(f"{path} must contain a mapping")
    return value


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    def encode(item: Any) -> Any:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if isinstance(item, (date, datetime)):
            return item.isoformat()
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Enum):
            return item.value
        raise TypeError(f"{type(item).__name__} is not JSON serializable")

    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=encode,
    )


def _fail(message: str, code: int = 2) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code)
