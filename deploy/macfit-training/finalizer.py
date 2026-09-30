"""Verify the final snapshot, then suspend only the two explicitly scoped CronJobs."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import ssl
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/opt/training")
from backup_pull import PUBLISH_SCHEMA, VERSION_NAME, archive_lock, checked_json, regular_file


class ClusterClient:
    def __init__(
        self,
        namespace: str,
        names: tuple[str, str],
        *,
        opener=urllib.request.urlopen,
        credentials: Path = Path("/run/finalizer-api"),
    ):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", namespace):
            raise ValueError("Invalid finalizer namespace")
        if len(set(names)) != 2 or any(
            not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", name) for name in names
        ):
            raise ValueError("Invalid finalizer CronJob scope")
        self.namespace, self.names, self.opener = namespace, names, opener
        self.credentials = credentials

    def suspend(self, name: str) -> None:
        if name not in self.names:
            raise ValueError("CronJob is outside finalizer scope")
        root = self.credentials
        token = (root / "token").read_text().strip()
        if not token or len(token) > 16384:
            raise ValueError("Invalid projected API token")
        host = os.environ["KUBERNETES_SERVICE_HOST"]
        if ":" in host:
            host = "[" + host + "]"
        port = int(os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443"))
        request = urllib.request.Request(
            f"https://{host}:{port}/apis/batch/v1/namespaces/{self.namespace}/cronjobs/{name}",
            data=b'{"spec":{"suspend":true}}',
            method="PATCH",
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/merge-patch+json",
            },
        )
        context = ssl.create_default_context(cafile=root / "ca.crt")
        with self.opener(request, context=context, timeout=5) as response:
            data = response.read(1024 * 1024 + 1)
            if response.status != 200 or len(data) > 1024 * 1024:
                raise ValueError("The cluster did not confirm schedule suspension")
            value = json.loads(data)
        if (
            not isinstance(value, dict)
            or value.get("metadata", {}).get("name") != name
            or value.get("spec", {}).get("suspend") is not True
        ):
            raise ValueError("The cluster did not confirm schedule suspension")


def verify_fresh_terminal(root: Path, receipt: dict, started: float, *, now: float) -> None:
    if not math.isfinite(started) or not started - 30 <= now <= started + 600:
        raise ValueError("Finalization is outside its bounded execution window")
    version = receipt.get("version")
    if (
        not isinstance(version, str)
        or not VERSION_NAME.fullmatch(version)
        or receipt.get("schema") != PUBLISH_SCHEMA
        or receipt.get("verified") is not True
        or receipt.get("archive_only") is not True
        or receipt.get("sqlite_quick_check") != "ok"
    ):
        raise ValueError("Final backup has no verified receipt")
    created = datetime.fromisoformat(receipt["source_created_at"].replace("Z", "+00:00"))
    if created.tzinfo is None or not started - 60 <= created.timestamp() <= now + 30:
        raise ValueError("Final backup is stale or has an invalid timestamp")
    with archive_lock(root):
        directory = root / "versions" / version
        if directory.is_symlink():
            raise ValueError("Final snapshot directory must be real")
        with regular_file(directory / "metadata.json", 65536) as handle:
            if checked_json(handle.read(65537)) != receipt:
                raise ValueError("Final backup receipt does not match the published version")
        database = directory / "data/jobs.sqlite3"
        with regular_file(database, 8 * 1024**3):
            pass
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            connection.execute("PRAGMA query_only=ON")
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("Final database integrity check failed")
            rows = connection.execute("SELECT status,error,worker_pid FROM jobs").fetchall()
        if len(rows) != receipt.get("jobs"):
            raise ValueError("Final job count differs from its receipt")
        for status, error, worker_pid in rows:
            if status not in {"succeeded", "failed", "cancelled"} or worker_pid is not None:
                raise ValueError("Final backup contains unfinished work")
            if error is not None:
                parsed = json.loads(error)
                if not isinstance(parsed, dict) or parsed.get("code") == "archive_interrupted":
                    raise ValueError("A restored terminal status masks unfinished source work")


def verify_transfer(root: Path, incoming: Path) -> dict:
    deadline = time.monotonic() + 75
    while True:
        result = subprocess.run(
            [
                sys.executable,
                "/opt/training/backup_pull.py",
                "verify",
                "--archive",
                str(incoming),
                "--archive-root",
                str(root),
                "--max-archive-gib",
                "25",
                "--max-unpacked-gib",
                "32",
                "--keep",
                "3",
            ],
            capture_output=True,
            timeout=90,
            check=False,
        )
        if result.returncode == 0 and len(result.stdout) <= 65536:
            receipt = checked_json(result.stdout)
            if isinstance(receipt, dict):
                return receipt
        if time.monotonic() >= deadline:
            raise ValueError("Final snapshot verification did not complete")
        time.sleep(5)


def finalize(
    root: Path,
    incoming: Path,
    started: float,
    fetched: bool,
    client: ClusterClient,
    *,
    verifier=verify_transfer,
    clock=time.time,
    pause=time.sleep,
) -> int:
    success, receipt, suspended = False, None, []
    # Pause first: a verifier timeout must not leave the expired schedules running.
    for name in client.names:
        for attempt in range(3):
            try:
                client.suspend(name)
                suspended.append(name)
                break
            except (OSError, ValueError, KeyError):
                if attempt < 2:
                    pause(1)
    try:
        if not fetched:
            raise ValueError("Final transfer failed")
        receipt = verifier(root, incoming)
        for attempt in range(6):
            try:
                verify_fresh_terminal(root, receipt, started, now=clock())
                break
            except BlockingIOError:
                if attempt == 5:
                    raise
                pause(1)
        success = True
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, subprocess.SubprocessError):
        print(
            "Final backup was not confirmed; the last verified archive is retained.",
            file=sys.stderr,
        )
    report = {
        "schema": "macfit-finalization.v1",
        "verified": success,
        "finished_at": datetime.fromtimestamp(clock(), UTC).isoformat(),
        "suspended_cronjobs": suspended,
        "source_created_at": receipt.get("source_created_at") if success else None,
        "version": receipt.get("version") if success else None,
        "last_verified_archive_retained": True,
    }
    temporary = root / (".finalization-" + uuid.uuid4().hex + ".json")
    try:
        temporary.write_text(json.dumps(report, sort_keys=True) + "\n")
        os.replace(temporary, root / "finalization.json")
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    if len(suspended) != len(client.names):
        print(
            "Schedule suspension was not confirmed; operator intervention is required.",
            file=sys.stderr,
        )
        return 2
    return 0 if success else 1


def main() -> int:
    root = Path("/archive")
    uid = str(uuid.UUID(os.environ["POD_UID"]))
    incoming_dir = root / "final-incoming" / uid
    client = ClusterClient(
        os.environ["POD_NAMESPACE"], ("macfit-training-backup", "macfit-training-finalizer")
    )
    try:
        started = float(Path("/final-state/started-at").read_text())
        planned = datetime.fromisoformat(
            os.environ["FINAL_PLANNED_AT"].replace("Z", "+00:00")
        ).timestamp()
        fetched = Path("/final-state/fetch-status").read_text().strip() == "0"
        fetched = fetched and planned - 60 <= started <= planned + 180
    except (OSError, ValueError, KeyError):
        started, fetched = time.time(), False
    try:
        return finalize(root, incoming_dir / "snapshot.tgz", started, fetched, client)
    finally:
        if incoming_dir.is_dir() and not incoming_dir.is_symlink():
            shutil.rmtree(incoming_dir)


if __name__ == "__main__":
    os.umask(0o077)
    try:
        result = main()
    except Exception:
        print(
            "Finalization was not confirmed; inspect the retained archive and schedule status.",
            file=sys.stderr,
        )
        result = 2
    raise SystemExit(result)
