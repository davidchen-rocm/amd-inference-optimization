"""Serve isolated writable copies of verified backups; never mutate retained versions."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from backup_pull import (
    MAX_ENTRY_BYTES,
    PUBLISH_SCHEMA,
    archive_lock,
    checked_json,
    current_version,
    quick_check,
    regular_file,
)

SERVING_NAME = re.compile(r"v\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{8}-[0-9a-f]{8}\Z")
MAX_UNPACKED_BYTES = 32 * 1024**3


def prepare_copy(root: Path) -> tuple[Path, dict] | None:
    """The publisher's lock prevents retention from pruning a source while it is copied."""
    with archive_lock(root):
        version = current_version(root)
        if version is None:
            return None
        source = root / "versions" / version
        with regular_file(source / "metadata.json", 65536) as handle:
            receipt = checked_json(handle.read(65537))
        if (
            receipt.get("schema") != PUBLISH_SCHEMA
            or receipt.get("verified") is not True
            or receipt.get("version") != version
            or receipt.get("archive_only") is not True
            or receipt.get("sqlite_quick_check") != "ok"
        ):
            raise ValueError("Latest archive has no verified receipt")
        serving = root / "serving"
        serving.mkdir(mode=0o700, exist_ok=True)
        if serving.is_symlink():
            raise ValueError("Serving directory cannot be a symlink")
        needed = receipt["unpacked_size_bytes"]
        if type(needed) is not int or not 0 < needed <= MAX_UNPACKED_BYTES:
            raise ValueError("Invalid snapshot size")
        if shutil.disk_usage(root).free < needed + 1024**3:
            raise ValueError("Not enough space to preserve both serving copies")
        name = "v" + version + "-" + uuid.uuid4().hex[:8]
        pending, final = serving / ("." + name + ".pending"), serving / name
        pending.mkdir(mode=0o700)
        total, count = 0, 0
        try:
            source_data = source / "data"
            if source_data.is_symlink():
                raise ValueError("Snapshot data cannot be a symlink")
            for directory, children, filenames in os.walk(source_data):
                directory = Path(directory)
                relative = directory.relative_to(source_data)
                if relative == Path("."):
                    children[:] = [name for name in children if name == "jobs"]
                    filenames = [
                        name for name in filenames if name in {"jobs.sqlite3", "archive-mode.json"}
                    ]
                for child in children:
                    if (directory / child).is_symlink():
                        raise ValueError("Snapshot contains a linked directory")
                target_dir = pending / relative
                target_dir.mkdir(mode=0o700, exist_ok=True)
                for filename in filenames:
                    count += 1
                    if count > 20000:
                        raise ValueError("Serving snapshot has too many files")
                    with regular_file(directory / filename, MAX_ENTRY_BYTES) as incoming:
                        with (target_dir / filename).open("xb") as outgoing:
                            while chunk := incoming.read(1024 * 1024):
                                total += len(chunk)
                                if total > MAX_UNPACKED_BYTES:
                                    raise ValueError("Serving snapshot exceeds its storage bound")
                                outgoing.write(chunk)
            if (
                json.loads((pending / "archive-mode.json").read_text()).get("archive_only")
                is not True
            ):
                raise ValueError("Serving snapshot is not archive-only")
            quick_check(pending, receipt["jobs"])
            (pending / "serving-receipt.json").write_text(json.dumps(receipt))
            pending.rename(final)
            return final, receipt
        finally:
            if pending.exists():
                shutil.rmtree(pending)


def archive_app(data: Path):
    from macfit_training.service.api import create_app
    from macfit_training.service.settings import Settings

    settings = Settings(
        data_dir=data,
        archive_only=True,
        gateway_secret=os.environ["MACFIT_TRAINING_GATEWAY_SECRET"],
        firebase_project_id=os.environ.get("MACFIT_FIREBASE_PROJECT_ID", "macfit-example"),
    )
    app = create_app(settings)

    @app.middleware("http")
    async def read_only(request, call_next):
        if request.method not in {"GET", "HEAD"}:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                {
                    "error": {
                        "code": "archive_read_only",
                        "message": "Saved results are read-only.",
                        "retryable": False,
                    }
                },
                status_code=503,
                headers={"Cache-Control": "no-store"},
            )
        response = await call_next(request)
        response.headers["X-MacFit-Archive"] = "true"
        return response

    return app


def serve(data: Path) -> None:
    import uvicorn

    uvicorn.run(
        archive_app(data),
        host="0.0.0.0",
        port=8793,
        workers=1,
        access_log=False,
        proxy_headers=False,
        limit_concurrency=32,
        server_header=False,
    )


def ready() -> bool:
    try:
        request = urllib.request.Request(
            "http://127.0.0.1:8793/api/training/capabilities",
            headers={"X-Training-Gateway": os.environ["MACFIT_TRAINING_GATEWAY_SECRET"].strip()},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            result = json.loads(response.read(65536))
        return result.get("available") is False and result.get("auth", {}).get("required") is True
    except (OSError, ValueError):
        return False


def stop(child: subprocess.Popen | None) -> None:
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=25)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


def previous_copy(root: Path) -> tuple[Path, dict] | None:
    serving = root / "serving"
    if not serving.is_dir() or serving.is_symlink():
        return None
    for directory in sorted(serving.iterdir(), reverse=True):
        if (
            not SERVING_NAME.fullmatch(directory.name)
            or directory.is_symlink()
            or not directory.is_dir()
        ):
            continue
        try:
            with regular_file(directory / "serving-receipt.json", 65536) as handle:
                receipt = checked_json(handle.read(65537))
            if (
                receipt.get("schema") != PUBLISH_SCHEMA
                or receipt.get("verified") is not True
                or receipt.get("archive_only") is not True
                or not directory.name.startswith("v" + receipt.get("version", "") + "-")
            ):
                continue
            quick_check(directory, receipt["jobs"])
            return directory, receipt
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            continue
    return None


def watch(root: Path) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    ending = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_args: ending.set())
    signal.signal(signal.SIGINT, lambda *_args: ending.set())
    child, current, current_receipt = None, None, None
    with (root / ".serving.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        serving = root / "serving"
        if serving.is_dir() and not serving.is_symlink():
            for directory in serving.iterdir():
                stem = directory.name.removeprefix(".").removesuffix(".pending")
                if (
                    directory.name == "." + stem + ".pending"
                    and SERVING_NAME.fullmatch(stem)
                    and directory.is_dir()
                    and not directory.is_symlink()
                ):
                    shutil.rmtree(directory)
        prior = previous_copy(root)
        if prior is not None:
            current, current_receipt = prior
            child = subprocess.Popen(
                [sys.executable, __file__, "serve", "--data-dir", str(current)]
            )
        try:
            while not ending.is_set():
                candidate = None
                try:
                    if current is not None and (child is None or child.poll() is not None):
                        child = subprocess.Popen(
                            [sys.executable, __file__, "serve", "--data-dir", str(current)]
                        )
                    try:
                        version = current_version(root)
                    except (OSError, ValueError):
                        version = None
                    changed = version and (
                        current_receipt is None or version != current_receipt["version"]
                    )
                    candidate = prepare_copy(root) if changed else None
                    if candidate is not None or (current is not None and child.poll() is not None):
                        selected, receipt = candidate or (current, current_receipt)
                        stop(child)
                        child = subprocess.Popen(
                            [sys.executable, __file__, "serve", "--data-dir", str(selected)]
                        )
                        deadline = time.monotonic() + 30
                        while (
                            child.poll() is None
                            and time.monotonic() < deadline
                            and not ending.is_set()
                        ):
                            if ready():
                                current, current_receipt = selected, receipt
                                print(
                                    json.dumps(
                                        {
                                            "archive": "serving",
                                            "version": receipt["version"],
                                            "source_created_at": receipt["source_created_at"],
                                        }
                                    ),
                                    flush=True,
                                )
                                break
                            ending.wait(0.5)
                        else:
                            stop(child)
                            if current is not None and current != selected:
                                child = subprocess.Popen(
                                    [sys.executable, __file__, "serve", "--data-dir", str(current)]
                                )
                            raise RuntimeError("New archive service did not become ready")
                        for directory in (root / "serving").iterdir():
                            if (
                                directory != current
                                and SERVING_NAME.fullmatch(directory.name)
                                and directory.is_dir()
                                and not directory.is_symlink()
                            ):
                                shutil.rmtree(directory)
                except (OSError, ValueError, RuntimeError, KeyError, TypeError, sqlite3.Error):
                    if candidate is not None and candidate[0] != current:
                        with contextlib.suppress(OSError):
                            shutil.rmtree(candidate[0])
                    # No private data, bearer token or dataset content enters diagnostic logs.
                    print('{"archive":"waiting_or_retaining_last_good"}', flush=True)
                ending.wait(10)
        finally:
            stop(child)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("watch", "serve"))
    parser.add_argument("--archive-root", type=Path, default=Path("/archive"))
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "watch":
        watch(args.archive_root)
    elif args.data_dir is None:
        parser.error("serve requires --data-dir")
    else:
        serve(args.data_dir)
