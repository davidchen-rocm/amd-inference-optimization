from __future__ import annotations

import http.client
import importlib.util
import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "deploy/macfit-training/bridge.py"
SPEC = importlib.util.spec_from_file_location("macfit_test_bridge", PATH)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


@contextmanager
def serving(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def upstream(records, *, status=200, payload=None, disconnect=False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self):
            size = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(size) if size else b""
            records.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": body,
                }
            )
            if disconnect:
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            content = json.dumps(payload or {"items": []}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(content)

        do_GET = do_HEAD = do_POST = respond

    return Handler


@pytest.fixture
def configured(tmp_path, monkeypatch):
    token = tmp_path / "gateway-token"
    token.write_text("synthetic-private-gateway-secret\n")
    monkeypatch.setattr(bridge, "TOKEN_FILE", token)
    monkeypatch.setattr(bridge, "UPSTREAM_HOST", "127.0.0.1")
    monkeypatch.setattr(bridge, "ARCHIVE_HOST", "127.0.0.1")
    return monkeypatch


def request(port, method, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request(
            method,
            path,
            body=body,
            headers={
                "Authorization": "Bearer synthetic-user-token",
                "X-Training-Gateway": "untrusted-browser-value",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_safe_reads_fall_back_on_upstream_connection_refused(configured, method):
    records = []
    with socket.socket() as refused:
        refused.bind(
            ("127.0.0.1", 0)
        )  # Allocate an unused port, then close it to force connection refusal.
        configured.setattr(bridge, "UPSTREAM_PORT", refused.getsockname()[1])
        refused.close()
        with serving(upstream(records)) as archive_port:
            configured.setattr(bridge, "ARCHIVE_PORT", archive_port)
            with serving(bridge.Bridge) as bridge_port:
                status, body = request(bridge_port, method, "/api/training/jobs?project_id=test")
    assert status == 200
    assert body == b"" if method == "HEAD" else json.loads(body) == {"items": []}
    assert len(records) == 1 and records[0]["method"] == method
    assert records[0]["headers"]["Authorization"] == "Bearer synthetic-user-token"
    assert records[0]["headers"]["X-Training-Gateway"] == "synthetic-private-gateway-secret"


@pytest.mark.parametrize("path", ["/api/training/jobs", "/api/training/jobs/test/cancel"])
def test_post_is_never_replayed_after_ambiguous_upstream_disconnect(configured, path):
    primary, archive = [], []
    with serving(upstream(primary, disconnect=True)) as primary_port:
        with serving(upstream(archive)) as archive_port:
            configured.setattr(bridge, "UPSTREAM_PORT", primary_port)
            configured.setattr(bridge, "ARCHIVE_PORT", archive_port)
            with serving(bridge.Bridge) as bridge_port:
                status, body = request(bridge_port, "POST", path, b'{"request_id":"stable"}')
    assert status == 503 and json.loads(body)["error"]["code"] == "training_unavailable"
    assert len(primary) == 1 and primary[0]["body"] == b'{"request_id":"stable"}'
    assert archive == []


def test_health_remains_ready_when_gpu_admission_is_closed(configured):
    records = []
    payload = {"available": False, "auth": {"required": True, "provider": "firebase"}}
    with serving(upstream(records, payload=payload)) as primary_port:
        configured.setattr(bridge, "UPSTREAM_PORT", primary_port)
        with serving(bridge.Bridge) as bridge_port:
            status, body = request(bridge_port, "GET", "/healthz")
    assert status == 200 and json.loads(body) == {"status": "ok"}
    assert records[0]["path"] == "/api/training/capabilities"
    assert "Authorization" not in records[0]["headers"]


def test_health_uses_reachable_archive_when_gpu_connection_is_gone(configured):
    records = []
    payload = {"available": False, "auth": {"required": True, "provider": "firebase"}}
    with socket.socket() as refused:
        refused.bind(("127.0.0.1", 0))
        configured.setattr(bridge, "UPSTREAM_PORT", refused.getsockname()[1])
        refused.close()
        with serving(upstream(records, payload=payload)) as archive_port:
            configured.setattr(bridge, "ARCHIVE_PORT", archive_port)
            with serving(bridge.Bridge) as bridge_port:
                status, body = request(bridge_port, "GET", "/healthz")
    assert status == 200 and json.loads(body) == {"status": "ok"}
    assert len(records) == 1


def test_upstream_http_error_does_not_trigger_archive_replay(configured):
    primary, archive = [], []
    with serving(
        upstream(primary, status=503, payload={"error": {"code": "authentication_unavailable"}})
    ) as primary_port:
        with serving(upstream(archive)) as archive_port:
            configured.setattr(bridge, "UPSTREAM_PORT", primary_port)
            configured.setattr(bridge, "ARCHIVE_PORT", archive_port)
            with serving(bridge.Bridge) as bridge_port:
                status, body = request(bridge_port, "GET", "/api/training/jobs")
    assert status == 503 and json.loads(body)["error"]["code"] == "authentication_unavailable"
    assert len(primary) == 1 and archive == []


def test_malformed_health_shape_returns_503_instead_of_connection_reset(configured):
    records = []
    with serving(upstream(records, payload=["unexpected-array"])) as primary_port:
        configured.setattr(bridge, "UPSTREAM_PORT", primary_port)
        with serving(bridge.Bridge) as bridge_port:
            status, body = request(bridge_port, "GET", "/healthz")
    assert status == 503 and json.loads(body)["error"]["code"] == "training_unavailable"
