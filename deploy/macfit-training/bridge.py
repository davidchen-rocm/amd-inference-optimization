"""Bounded HTTP bridge over a private outbound SSH tunnel.

This process never authenticates browser-supplied user IDs. Firebase bearer
verification and job ownership are enforced by the GPU service.
"""

from __future__ import annotations

import http.client
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

UPSTREAM_HOST = os.environ.get("TRAINING_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("TRAINING_UPSTREAM_PORT", "8792"))
ARCHIVE_HOST = os.environ.get("TRAINING_ARCHIVE_HOST", "127.0.0.1")
ARCHIVE_PORT = int(os.environ.get("TRAINING_ARCHIVE_PORT", "8793"))
TOKEN_FILE = Path(
    os.environ.get("TRAINING_GATEWAY_TOKEN_FILE", "/run/training-secrets/gateway-token")
)
MAX_BODY = 3 * 1024 * 1024
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class Bridge(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(20)

    def log_message(self, format, *args):
        # Never log bearer tokens, query strings or user content.
        return

    def error_json(self, status, code, message):
        body = json.dumps(
            {"error": {"code": code, "message": message, "retryable": status >= 500}}
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.close_connection = True

    def do_GET(self):
        self.forward()

    def do_HEAD(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        health = urlsplit(self.path).path == "/healthz"
        if not health and not self.path.startswith("/api/training/"):
            self.error_json(404, "not_found", "This endpoint is not available.")
            return
        if self.headers.get("Transfer-Encoding"):
            self.error_json(400, "invalid_request", "A Content-Length header is required.")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.error_json(400, "invalid_request", "Invalid Content-Length.")
            return
        if length < 0 or length > MAX_BODY:
            self.error_json(413, "request_too_large", "Training requests must fit within 3 MiB.")
            return
        try:
            body = self.rfile.read(length) if length else None
            if body is not None and len(body) != length:
                raise ValueError("incomplete request")
            token = TOKEN_FILE.read_text().strip()
            if not token:
                raise ValueError("missing gateway configuration")
        except (OSError, ValueError):
            self.error_json(
                503, "training_unavailable", "The training connection is unavailable. Please retry."
            )
            return
        # Both health attempts must fit the five-second Kubernetes probe budget.
        timeout = 1.5 if health else 5
        connection = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=timeout)
        headers = {
            "X-Training-Gateway": token,
            "Accept": "application/json",
            "Content-Type": self.headers.get("Content-Type", "application/json"),
        }
        if not health and self.headers.get("Authorization"):
            headers["Authorization"] = self.headers["Authorization"]
        started = False
        try:
            path = "/api/training/capabilities" if health else self.path
            try:
                connection.request(
                    "GET" if health else self.command, path, body=body, headers=headers
                )
                response = connection.getresponse()
            except (OSError, http.client.HTTPException):
                connection.close()
                # Never replay submissions/cancellation after an ambiguous upstream failure.
                # The archive service independently verifies Firebase ownership for every read.
                if not health and self.command not in {"GET", "HEAD"}:
                    raise
                connection = http.client.HTTPConnection(ARCHIVE_HOST, ARCHIVE_PORT, timeout=timeout)
                connection.request("GET" if health else self.command, path, headers=headers)
                response = connection.getresponse()
            if health:
                data = response.read(MAX_BODY)
                state = json.loads(data) if response.status == 200 else {}
                if (
                    response.status != 200
                    or not isinstance(state, dict)
                    or not isinstance(state.get("auth"), dict)
                ):
                    self.error_json(
                        503, "training_unavailable", "The training service is not reachable."
                    )
                    return
                result = b'{"status":"ok"}\n'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(result)
                return
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS | {"server", "date"}:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            started = True
            if self.command != "HEAD":
                while chunk := response.read(64 * 1024):
                    self.wfile.write(chunk)
        except (OSError, http.client.HTTPException, ValueError, json.JSONDecodeError):
            if not started:
                self.error_json(
                    503, "training_unavailable", "The GPU connection is unavailable. Please retry."
                )
        finally:
            connection.close()
            self.close_connection = True


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8080), Bridge)
    server.daemon_threads = True
    server.serve_forever()
