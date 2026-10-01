"""Owner-scoped SQLite WAL queue; immutable inputs live beside the database."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .settings import Settings

ACTIVE = ("queued", "running", "cancelling")
TERMINAL = ("succeeded", "failed", "cancelled")
PUBLIC_FIELDS = (
    "id",
    "request_id",
    "project_id",
    "kind",
    "status",
    "stage",
    "progress",
    "created_at",
    "updated_at",
    "input_hash",
    "base_model",
    "error",
    "result",
    "artifacts",
)
JSON_FIELDS = ("progress", "base_model", "error", "result", "artifacts")


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.status, self.code, self.message, self.retryable = status, code, message, retryable


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def public_job(job: dict) -> dict:
    result = {name: job[name] for name in PUBLIC_FIELDS}
    result["artifacts"] = [
        {name: artifact[name] for name in ("id", "type", "name", "size_bytes", "sha256")}
        for artifact in job["artifacts"]
    ]
    return result


class JobStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.data_dir
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(exist_ok=True, mode=0o700)
        if self.jobs_dir.is_symlink():
            raise ValueError("The private jobs directory must not be a symbolic link.")
        self.path = self.root / "jobs.sqlite3"
        if self.path.is_symlink():
            raise ValueError("The job database must not be a symbolic link.")
        self._lock = threading.RLock()
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner_uid TEXT NOT NULL,
                    request_id TEXT NOT NULL, project_id TEXT NOT NULL, kind TEXT NOT NULL,
                    status TEXT NOT NULL, stage TEXT NOT NULL, progress TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    input_hash TEXT NOT NULL, input_bytes INTEGER NOT NULL,
                    base_model TEXT NOT NULL, error TEXT, result TEXT,
                    artifacts TEXT NOT NULL DEFAULT '[]', artifact_bytes INTEGER NOT NULL DEFAULT 0,
                    worker_pid INTEGER, worker_start_ticks INTEGER,
                    UNIQUE(owner_uid, request_id)
                );
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS jobs_owner_project
                    ON jobs(owner_uid, project_id, created_at);
            """)
        os.chmod(self.path, 0o600)

    @contextmanager
    def connection(self):
        with self._lock:
            connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("PRAGMA busy_timeout=10000")
                yield connection
            finally:
                connection.close()

    @staticmethod
    def decode(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        for name in JSON_FIELDS:
            result[name] = json.loads(result[name]) if result[name] is not None else None
        return result

    def directory(self, job_id: str) -> Path:
        # IDs only originate from UUID generation or a previously fetched database row.
        if str(uuid.UUID(job_id)) != job_id:
            raise ValueError("Invalid private job identity")
        return self.jobs_dir / job_id

    def get(self, job_id: str, owner_uid: str | None = None) -> dict:
        with self.connection() as connection:
            if owner_uid is None:
                row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM jobs WHERE id=? AND owner_uid=?", (job_id, owner_uid)
                ).fetchone()
        if row is None:
            raise ServiceError(404, "not_found", "This job could not be found.")
        return self.decode(row)

    def list(
        self, owner_uid: str, *, project_id: str | None = None, request_id: str | None = None
    ) -> list[dict]:
        where, args = ["owner_uid=?"], [owner_uid]
        if project_id:
            where.append("project_id=?")
            args.append(project_id)
        if request_id:
            where.append("request_id=?")
            args.append(request_id)
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE "
                + " AND ".join(where)
                + " ORDER BY created_at DESC, id DESC LIMIT 50",
                args,
            ).fetchall()
        return [self.decode(row) for row in rows]

    def create(
        self, owner_uid: str, request_id: str, project_id: str, kind: str, job_input: dict
    ) -> tuple[dict, bool]:
        encoded = canonical(job_input)
        digest = hashlib.sha256(
            canonical({"project_id": project_id, "kind": kind, "input": job_input})
        ).hexdigest()
        job_id, now = str(uuid.uuid4()), utcnow()
        directory = self.directory(job_id)
        made_directory = committed = False
        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                prior = self.decode(
                    connection.execute(
                        "SELECT * FROM jobs WHERE owner_uid=? AND request_id=?",
                        (owner_uid, request_id),
                    ).fetchone()
                )
                if prior:
                    connection.rollback()
                    if prior["input_hash"] != digest:
                        raise ServiceError(
                            409,
                            "request_conflict",
                            "This request ID already belongs to different inputs. "
                            "Create a new request.",
                        )
                    return prior, False
                active = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE owner_uid=? "
                    "AND status IN ('queued','running','cancelling')",
                    (owner_uid,),
                ).fetchone()[0]
                if active:
                    raise ServiceError(
                        409,
                        "owner_busy",
                        "Finish or cancel your current job before submitting another.",
                    )
                queued = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running','cancelling')"
                ).fetchone()[0]
                if queued >= self.settings.max_queue_jobs:
                    raise ServiceError(
                        429,
                        "queue_full",
                        "The training queue is full. Please try again later.",
                        True,
                    )
                if not self.settings.accepts_new(queued=queued):
                    raise ServiceError(
                        503,
                        "gpu_window_closed",
                        "New GPU jobs are paused. Your saved jobs and downloads remain available.",
                    )
                count = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                owned = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE owner_uid=?", (owner_uid,)
                ).fetchone()[0]
                if count >= self.settings.max_total_jobs or owned >= self.settings.max_owner_jobs:
                    raise ServiceError(
                        429,
                        "job_quota",
                        "The saved-job limit has been reached. Contact the site owner.",
                    )
                stored = connection.execute(
                    "SELECT COALESCE(SUM(input_bytes + artifact_bytes),0) FROM jobs"
                ).fetchone()[0]
                if (
                    stored + len(encoded) >= self.settings.max_storage_bytes
                    or shutil.disk_usage(self.root).free < self.settings.min_free_bytes
                ):
                    raise ServiceError(
                        503,
                        "storage_full",
                        "There is not enough storage for another job. Please try again later.",
                        True,
                    )
                directory.mkdir(mode=0o700)
                made_directory = True
                descriptor = os.open(
                    directory / "input.json",
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                    0o400,
                )
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                # Commit metadata only after its immutable input snapshot is on disk.
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                connection.execute(
                    "INSERT INTO jobs (id,owner_uid,request_id,project_id,kind,status,stage,"
                    "created_at,updated_at,input_hash,input_bytes,base_model) "
                    "VALUES (?,?,?,?,?,'queued','queued',?,?,?,?,?)",
                    (
                        job_id,
                        owner_uid,
                        request_id,
                        project_id,
                        kind,
                        now,
                        now,
                        digest,
                        len(encoded),
                        canonical(job_input["base_model"]).decode(),
                    ),
                )
                connection.commit()
                committed = True
            return self.get(job_id), True
        except Exception:
            if made_directory and not committed:
                shutil.rmtree(directory)
            raise

    def claim(self) -> dict | None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM jobs WHERE status IN ('running','cancelling') LIMIT 1"
            ).fetchone():
                connection.rollback()
                return None
            row = connection.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1"
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            if not self.settings.can_finish(self.settings.max_walltime_seconds):
                connection.execute(
                    "UPDATE jobs SET status='failed',stage='failed',updated_at=?,error=? "
                    "WHERE status='queued'",
                    (
                        utcnow(),
                        canonical(
                            {
                                "code": "gpu_window_closed",
                                "message": "The GPU availability window ended before this job "
                                "could start. Your inputs are saved.",
                                "retryable": True,
                            }
                        ).decode(),
                    ),
                )
                connection.commit()
                return None
            connection.execute(
                "UPDATE jobs SET status='running',stage='starting',updated_at=? WHERE id=?",
                (utcnow(), row["id"]),
            )
            connection.commit()
            return self.get(row["id"])

    def set_worker(self, job_id: str, pid: int, start_ticks: int | None):
        with self.connection() as connection:
            connection.execute(
                "UPDATE jobs SET worker_pid=?,worker_start_ticks=?,updated_at=? WHERE id=?",
                (pid, start_ticks, utcnow(), job_id),
            )

    def progress(self, job_id: str, stage: str, progress: dict | None):
        with self.connection() as connection:
            connection.execute(
                "UPDATE jobs SET stage=?,progress=?,updated_at=? WHERE id=? AND status='running'",
                (
                    stage,
                    canonical(progress).decode() if progress is not None else None,
                    utcnow(),
                    job_id,
                ),
            )

    def cancel(self, job_id: str, owner_uid: str) -> dict:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = self.decode(
                connection.execute(
                    "SELECT * FROM jobs WHERE id=? AND owner_uid=?", (job_id, owner_uid)
                ).fetchone()
            )
            if job is None:
                raise ServiceError(404, "not_found", "This job could not be found.")
            if job["status"] == "queued":
                connection.execute(
                    "UPDATE jobs SET status='cancelled',stage='cancelled',updated_at=? WHERE id=?",
                    (utcnow(), job_id),
                )
            elif job["status"] == "running":
                connection.execute(
                    "UPDATE jobs SET status='cancelling',stage='cancel_requested',updated_at=? "
                    "WHERE id=?",
                    (utcnow(), job_id),
                )
            connection.commit()
        return self.get(job_id, owner_uid)

    def finish(
        self,
        job_id: str,
        status: str,
        *,
        error: dict | None = None,
        result: dict | None = None,
        artifacts: list | None = None,
    ):
        if status not in TERMINAL:
            raise ValueError("Only terminal statuses may finish a job")
        artifacts = artifacts or []
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = self.decode(
                connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            )
            if job is None or job["status"] in TERMINAL:
                connection.rollback()
                return
            # A user cancellation observed before success commits wins the race.
            if status == "succeeded" and job["status"] == "cancelling":
                status, result, artifacts = "cancelled", None, []
            total = sum(a["size_bytes"] for a in artifacts)
            stored = connection.execute(
                "SELECT COALESCE(SUM(input_bytes + artifact_bytes),0) FROM jobs WHERE id<>?",
                (job_id,),
            ).fetchone()[0]
            if (
                status == "succeeded"
                and stored + job["input_bytes"] + total > self.settings.max_storage_bytes
            ):
                status, result, artifacts, total = "failed", None, [], 0
                error = {
                    "code": "storage_full",
                    "message": "The result exceeded available artifact storage.",
                    "retryable": True,
                }
            connection.execute(
                "UPDATE jobs SET status=?,stage=?,updated_at=?,error=?,result=?,artifacts=?,"
                "artifact_bytes=?,worker_pid=NULL,worker_start_ticks=NULL WHERE id=?",
                (
                    status,
                    status,
                    utcnow(),
                    canonical(error).decode() if error else None,
                    canonical(result).decode() if result is not None else None,
                    canonical(artifacts).decode(),
                    total,
                    job_id,
                ),
            )
            connection.commit()

    def interrupted(self) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('running','cancelling')"
            ).fetchall()
        return [self.decode(row) for row in rows]

    def queue_size(self) -> int:
        with self.connection() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running','cancelling')"
            ).fetchone()[0]
