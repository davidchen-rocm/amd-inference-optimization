"""Dependency-free, read-only HTTP server for the local control plane."""

from __future__ import annotations

import ipaddress
import json
import mimetypes
import secrets
from collections.abc import Callable
from datetime import date, datetime
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import ValidationError

from .frontend_api import ApiErrorV1, ControlPlaneReader, FrontendReadError
from .frontend_control import (
    DraftOptionsV1,
    DraftStore,
    ExperimentBuilderMetaV1,
    OptimizationDraftRequestV1,
    build_model_catalog,
    frontend_control_schema,
    validate_draft_model,
)

DEFAULT_UI_HOST = "127.0.0.1"
DEFAULT_UI_PORT = 4561

_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
)
_STATIC_FILES = {
    "/assets/app.js": "app.js",
    "/assets/styles.css": "styles.css",
}


def is_loopback_host(host: str) -> bool:
    """Return whether a bind address is explicitly local-only."""

    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _json_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True)

    def encode(item: Any) -> Any:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json", by_alias=True)
        if isinstance(item, (date, datetime)):
            return item.isoformat()
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Enum):
            return item.value
        raise TypeError(f"{type(item).__name__} is not JSON serializable")

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=encode,
    ).encode("utf-8")


def _handler_factory(
    reader: ControlPlaneReader,
    draft_store: DraftStore | None = None,
    csrf_token: str | None = None,
) -> type[BaseHTTPRequestHandler]:
    class ControlPlaneHandler(BaseHTTPRequestHandler):
        server_version = "gpuopt-control-plane/1"
        sys_version = ""

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", _CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            super().end_headers()

        def log_message(self, format: str, *args: object) -> None:
            # Keep the CLI quiet; operators can inspect persisted workflow events.
            return

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _error(self, status: HTTPStatus, code: str, message: str) -> None:
            self._send(
                status,
                _json_bytes(ApiErrorV1(code=code, message=message)),
                "application/json; charset=utf-8",
            )

        def _api(self, path: str, query: dict[str, list[str]]) -> None:
            segments = [unquote(value) for value in path.strip("/").split("/")]
            try:
                if segments == ["api", "v1", "meta"]:
                    payload: Any = reader.meta()
                elif segments == ["api", "v1", "schema"]:
                    payload = reader.schema()
                elif segments == ["api", "v1", "runs"]:
                    payload = reader.runs()
                elif segments == ["api", "v1", "catalog", "models"]:
                    payload = build_model_catalog(reader)
                elif (
                    len(segments) == 5
                    and segments[:4] == ["api", "v1", "catalog", "models"]
                ):
                    payload = next(
                        (
                            item
                            for item in build_model_catalog(reader).items
                            if item.id == segments[4]
                        ),
                        None,
                    )
                    if payload is None:
                        self._error(
                            HTTPStatus.NOT_FOUND, "model_not_found", "model family not found"
                        )
                        return
                elif segments == ["api", "v1", "builder", "meta"]:
                    payload = ExperimentBuilderMetaV1(
                        draft_writes_enabled=draft_store is not None,
                        csrf_token=csrf_token if draft_store is not None else None,
                    )
                elif segments == ["api", "v1", "builder", "options"]:
                    payload = DraftOptionsV1()
                elif segments == ["api", "v1", "builder", "schema"]:
                    payload = frontend_control_schema()
                elif segments == ["api", "v1", "drafts"]:
                    if draft_store is None:
                        self._error(
                            HTTPStatus.NOT_FOUND,
                            "draft_store_disabled",
                            "draft database is not configured",
                        )
                        return
                    payload = draft_store.list()
                elif len(segments) == 4 and segments[:3] == ["api", "v1", "drafts"]:
                    if draft_store is None:
                        self._error(
                            HTTPStatus.NOT_FOUND,
                            "draft_store_disabled",
                            "draft database is not configured",
                        )
                        return
                    payload = draft_store.get(segments[3])
                elif len(segments) >= 5 and segments[:3] == ["api", "v1", "runs"]:
                    source_id, run_id = segments[3:5]
                    detail = reader.source(source_id).detail(run_id)
                    if len(segments) == 5:
                        payload = detail
                    elif segments[5:] == ["optimization-map"]:
                        payload = reader.source(source_id).optimization_map(run_id)
                    elif segments[5:] == ["events"]:
                        payload = {
                            "schema": "gpuopt.event-list.v1",
                            "items": detail.recent_events,
                        }
                    elif segments[5:] == ["artifacts"]:
                        payload = {
                            "schema": "gpuopt.artifact-list.v1",
                            "items": detail.artifacts,
                        }
                    elif len(segments) == 7 and segments[5] == "experiments":
                        payload = reader.source(source_id).experiment(run_id, segments[6])
                    elif segments[5:] == ["artifacts", "preview"]:
                        artifact_paths = query.get("path", [])
                        if len(artifact_paths) != 1 or not artifact_paths[0]:
                            self._error(
                                HTTPStatus.BAD_REQUEST,
                                "invalid_artifact_path",
                                "exactly one non-empty artifact path is required",
                            )
                            return
                        payload = reader.source(source_id).preview(run_id, artifact_paths[0])
                    else:
                        self._error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")
                        return
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")
                    return
            except (FrontendReadError, KeyError) as error:
                self._error(HTTPStatus.NOT_FOUND, "read_error", str(error))
                return
            except (OSError, ValueError) as error:
                self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_record", str(error))
                return
            self._send(HTTPStatus.OK, _json_bytes(payload), "application/json; charset=utf-8")

        def _static(self, path: str) -> None:
            filename = _STATIC_FILES[path]
            content = resources.files("amd_inference_opt.frontend").joinpath(filename).read_bytes()
            media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            if media_type.startswith("text/") or media_type == "application/javascript":
                media_type += "; charset=utf-8"
            self._send(HTTPStatus.OK, content, media_type)

        def _handle_read(self) -> None:
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if path.startswith("/api/"):
                self._api(path, parse_qs(parsed.query, keep_blank_values=True))
                return
            if path in _STATIC_FILES:
                self._static(path)
                return
            if path == "/" or path.startswith(("/runs/", "/models", "/builder", "/drafts")):
                content = resources.files("amd_inference_opt.frontend").joinpath(
                    "index.html"
                ).read_bytes()
                self._send(HTTPStatus.OK, content, "text/html; charset=utf-8")
                return
            self._error(HTTPStatus.NOT_FOUND, "not_found", "resource not found")

        def do_GET(self) -> None:  # noqa: N802
            self._handle_read()

        def do_HEAD(self) -> None:  # noqa: N802
            self._handle_read()

        def do_POST(self) -> None:  # noqa: N802
            path = unquote(urlsplit(self.path).path)
            if path != "/api/v1/drafts" or draft_store is None:
                self._read_only()
                return
            if self.headers.get("X-GPUOPT-CSRF") != csrf_token:
                self._error(HTTPStatus.FORBIDDEN, "invalid_csrf", "invalid CSRF token")
                return
            media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if media_type != "application/json":
                self._error(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "invalid_content_type",
                    "draft requests must use application/json",
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length <= 0 or length > 128 * 1024:
                self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "invalid_body_size",
                    "draft request must be between 1 byte and 128 KiB",
                )
                return
            try:
                payload = json.loads(self.rfile.read(length))
                request = OptimizationDraftRequestV1.model_validate(payload)
                validate_draft_model(reader, request)
                record = draft_store.create(request)
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                self._error(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "invalid_draft",
                    str(error),
                )
                return
            except FrontendReadError as error:
                self._error(HTTPStatus.NOT_FOUND, "read_error", str(error))
                return
            self._send(
                HTTPStatus.CREATED,
                _json_bytes(record),
                "application/json; charset=utf-8",
            )

        def _read_only(self) -> None:
            self._error(HTTPStatus.METHOD_NOT_ALLOWED, "read_only", "control plane is read-only")

        do_PUT = _read_only
        do_PATCH = _read_only
        do_DELETE = _read_only
        do_OPTIONS = _read_only

    return ControlPlaneHandler


class ControlPlaneHTTPServer(ThreadingHTTPServer):
    """Threaded local server carrying an immutable ControlPlaneReader reference."""

    daemon_threads = True


def create_control_plane_server(
    reader: ControlPlaneReader,
    *,
    host: str = DEFAULT_UI_HOST,
    port: int = DEFAULT_UI_PORT,
    draft_store: DraftStore | None = None,
) -> ControlPlaneHTTPServer:
    if not is_loopback_host(host):
        raise ValueError("UI host must be a loopback address")
    if not 0 <= port <= 65535:
        raise ValueError("UI port must be between 0 and 65535")
    token = secrets.token_urlsafe(32) if draft_store is not None else None
    return ControlPlaneHTTPServer(
        (host, port), _handler_factory(reader, draft_store, token)
    )


def serve_control_plane(
    reader: ControlPlaneReader,
    *,
    host: str = DEFAULT_UI_HOST,
    port: int = DEFAULT_UI_PORT,
    draft_store: DraftStore | None = None,
    announce: Callable[[str], None] = print,
) -> None:
    """Serve until interrupted; this function never mutates a configured store."""

    server = create_control_plane_server(
        reader, host=host, port=port, draft_store=draft_store
    )
    bound_host, bound_port = server.server_address[:2]
    announce(f"gpuopt control-plane UI: http://{bound_host}:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
        pass
    finally:
        server.server_close()


__all__ = [
    "ControlPlaneHTTPServer",
    "DEFAULT_UI_HOST",
    "DEFAULT_UI_PORT",
    "create_control_plane_server",
    "is_loopback_host",
    "serve_control_plane",
]
