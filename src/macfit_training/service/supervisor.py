"""One durable queue consumer, one worker session, bounded cleanup before completion."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from amd_inference_opt.process_group import (
    ProcessGroupError,
    ProcessGroupLease,
    read_process,
    session_has_live_members,
    terminate_uncaptured_child,
)

from .outputs import open_regular, read_json, verify_outputs
from .settings import Settings
from .store import JobStore, canonical


class WorkerLimitError(ValueError):
    pass


class Supervisor:
    def __init__(
        self,
        store: JobStore,
        settings: Settings,
        *,
        readiness: Callable[[], bool] | None = None,
        command_factory: Callable[[Path, str], list[str]] | None = None,
    ):
        self.store, self.settings = store, settings
        self.readiness = readiness or settings.gpu_ready
        self.command_factory = command_factory or (
            lambda directory, kind: [
                sys.executable,
                "-m",
                "macfit_training.worker",
                "--job-dir",
                str(directory),
                "--kind",
                kind,
            ]
        )
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock_fd: int | None = None
        self.heartbeat = 0.0
        self.fatal: str | None = None
        self.active_pid: int | None = None

    @property
    def healthy(self) -> bool:
        return bool(
            self.thread
            and self.thread.is_alive()
            and not self.fatal
            and time.monotonic() - self.heartbeat < 30
        )

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.lock_fd = os.open(
            self.store.root / ".service.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._recover()
        except Exception:
            os.close(self.lock_fd)
            self.lock_fd = None
            raise
        self.stop_event.clear()
        self.heartbeat = time.monotonic()
        self.thread = threading.Thread(target=self._loop, name="macfit-gpu-supervisor", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.wake_event.set()
        if self.thread:
            self.thread.join(timeout=self.settings.termination_grace_seconds + 15)
            if self.thread.is_alive():
                self.fatal = "worker_cleanup_required"
                raise RuntimeError("Worker cleanup is not confirmed; keep the service stopped.")
        if self.lock_fd is not None:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            os.close(self.lock_fd)
            self.lock_fd = None

    def notify(self):
        self.wake_event.set()

    def _recover(self):
        for job in self.store.interrupted():
            pid, ticks = job["worker_pid"], job["worker_start_ticks"]
            if pid:
                if sys.platform == "linux":
                    identity = read_process(pid)
                    if identity is not None and identity.start_ticks == ticks:
                        lease = ProcessGroupLease(pid, ticks)
                        lease.terminate(grace_seconds=self.settings.termination_grace_seconds)
                    elif session_has_live_members(pid):
                        raise ProcessGroupError(
                            "An interrupted worker session needs operator cleanup."
                        )
                else:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise ProcessGroupError(
                            "A previous worker must be stopped before recovery."
                        )
            self.store.finish(
                job["id"],
                "failed",
                error={
                    "code": "service_restarted",
                    "message": "The service restarted before this job finished. "
                    "Submit a new request to retry.",
                    "retryable": True,
                },
            )
            self._remove_unpublished(job["id"])

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                self.heartbeat = time.monotonic()
                if not self.settings.can_finish(self.settings.max_walltime_seconds):
                    self.store.claim()  # Expire queued work that cannot finish before shutdown.
                if self.readiness():
                    job = self.store.claim()
                    if job:
                        self._run(job)
                        continue
                self.wake_event.wait(self.settings.poll_seconds)
                self.wake_event.clear()
        except Exception:
            # Fail closed: do not claim another GPU job after uncertain supervision.
            self.fatal = "worker_cleanup_required"

    def _environment(self) -> dict[str, str]:
        allowed = {
            "PATH",
            "HOME",
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "HF_HOME",
            "HF_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "XDG_CACHE_HOME",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "PYTORCH_HIP_ALLOC_CONF",
            "PYTORCH_CUDA_ALLOC_CONF",
            "MACFIT_GPU_LOCK",
            "MACFIT_MODEL_CACHE",
            "TOKENIZERS_PARALLELISM",
        }
        result = {name: value for name, value in os.environ.items() if name in allowed}
        # Worker cwd is a private job directory, so relative PYTHONPATH entries are unsafe.
        import_roots = [str(Path(__file__).resolve().parents[2])]
        import_roots.extend(
            str(Path(item).resolve())
            for item in result.get("PYTHONPATH", "").split(os.pathsep)
            if item
        )
        result["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(import_roots))
        result["PYTHONUNBUFFERED"] = "1"
        result["MACFIT_GPU_LOCK"] = result.get("MACFIT_GPU_LOCK", str(self.store.root / "gpu.lock"))
        return result

    def _cleanup(self, process, lease):
        if lease is not None:
            lease.terminate(grace_seconds=self.settings.termination_grace_seconds)
            process.wait(timeout=5)
            if lease.has_live_members():
                raise ProcessGroupError("Worker session did not stop")
        else:
            # Portable test/development path; production GPU supervision uses /proc identities.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=self.settings.termination_grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    def _drain_log(self, stream, path):
        kept = 0
        descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as log:
            try:
                while chunk := stream.read(65536):
                    remaining = self.settings.max_log_bytes - kept
                    if remaining > 0:
                        log.write(chunk[:remaining])
                        kept += min(remaining, len(chunk))
            finally:
                stream.close()

    def _events(self, directory, cursor):
        path = directory / "events.jsonl"
        if not path.exists() and not path.is_symlink():
            return cursor
        with open_regular(path, self.settings.max_event_bytes) as events:
            events.seek(cursor)
            while True:
                line = events.readline(16385)
                if not line:
                    break
                if len(line) > 16384:
                    raise WorkerLimitError("Worker event exceeded its limit")
                if not line.endswith(b"\n"):
                    break  # A partial append will be read on the next poll.
                cursor = events.tell()
                try:
                    event = json.loads(line)
                    stage, progress = event.get("stage"), event.get("progress")
                    if not isinstance(stage, str) or not re.fullmatch(
                        r"[a-z][a-z0-9_-]{0,63}", stage
                    ):
                        raise ValueError("Invalid event stage")
                    if progress is not None:
                        if not isinstance(progress, dict) or set(progress) != {
                            "completed",
                            "total",
                            "unit",
                        }:
                            raise ValueError("Invalid progress")
                        completed, total = progress["completed"], progress["total"]
                        if type(completed) not in (int, float) or type(total) not in (int, float):
                            raise ValueError("Invalid progress counts")
                        if (
                            not math.isfinite(completed)
                            or not math.isfinite(total)
                            or not 0 <= completed <= total
                            or total <= 0
                        ):
                            raise ValueError("Invalid progress counts")
                        if not isinstance(progress["unit"], str) or not re.fullmatch(
                            r"[A-Za-z][A-Za-z _-]{0,31}", progress["unit"]
                        ):
                            raise ValueError("Invalid progress unit")
                    self.store.progress(directory.name, stage, progress)
                except (ValueError, TypeError, AttributeError) as error:
                    raise WorkerLimitError("Invalid worker progress event") from error
        return cursor

    def _check_outputs_size(self, directory):
        total, count = 0, 0
        for parent, dirs, files in os.walk(directory, followlinks=False):
            for name in dirs + files:
                path = Path(parent) / name
                try:
                    item = path.lstat()
                except FileNotFoundError:
                    continue  # Atomic worker publication may rename a temporary file.
                if stat.S_ISLNK(item.st_mode) or not (
                    stat.S_ISREG(item.st_mode) or stat.S_ISDIR(item.st_mode)
                ):
                    raise WorkerLimitError("Unexpected worker output file")
                if stat.S_ISREG(item.st_mode):
                    total += item.st_size
                count += 1
                if (
                    count > 256
                    or total
                    > self.settings.max_job_artifact_bytes
                    + self.settings.max_result_bytes
                    + self.settings.max_log_bytes
                    + self.settings.max_event_bytes
                    + self.settings.max_body_bytes
                ):
                    raise WorkerLimitError("Worker storage limit exceeded")
        if shutil.disk_usage(self.store.root).free < self.settings.min_free_bytes:
            raise WorkerLimitError("Worker storage is full")

    def _worker_error(self, directory):
        default = {
            "code": "worker_failed",
            "message": "The GPU task could not finish. Review the inputs and try again.",
            "retryable": True,
        }
        try:
            error = read_json(directory / "error.json", 16384)
            code = error.get("code")
            if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code):
                default["code"] = code
                default["message"] = {
                    "invalid_input": "The saved inputs are not valid for this model. "
                    "Review them and submit a new request.",
                    "generation_invalid": "The model could not generate a valid dataset. "
                    "Adjust your examples or instructions and retry.",
                    "time_limit": "This job reached its GPU time limit. "
                    "Try the quick preset or a smaller dataset.",
                    "out_of_memory": "This job exceeded GPU memory. "
                    "Try a smaller model or the quick preset.",
                }.get(code, default["message"])
            # Detailed exceptions remain in private logs, never in an HTTP response.
            default["retryable"] = error.get("retryable") is not False
        except (OSError, ValueError):
            pass
        return default

    def _remove_unpublished(self, job_id):
        directory = self.store.directory(job_id) / "artifacts"
        if directory.is_symlink():
            directory.unlink()
        elif directory.exists():
            shutil.rmtree(directory)

    def _run(self, job):
        directory = self.store.directory(job["id"])
        process = lease = log_thread = None
        read_fd = write_fd = None
        cleaned = False
        try:
            job_input = read_json(directory / "input.json", self.settings.max_body_bytes + 65536)
            actual_hash = hashlib.sha256(
                canonical(
                    {"project_id": job["project_id"], "kind": job["kind"], "input": job_input}
                )
            ).hexdigest()
            if actual_hash != job["input_hash"]:
                raise WorkerLimitError("The immutable input changed")
            if self.store.get(job["id"])["status"] == "cancelling":
                self.store.finish(job["id"], "cancelled")
                return
            read_fd, write_fd = os.pipe()
            command = [
                sys.executable,
                "-m",
                "macfit_training.service.launcher",
                "--gate-fd",
                str(read_fd),
                "--",
                *self.command_factory(directory, job["kind"]),
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(read_fd,),
                env=self._environment(),
                cwd=directory,
            )
            os.close(read_fd)
            read_fd = None
            self.active_pid = process.pid
            ticks = None
            if sys.platform == "linux":
                identity = read_process(process.pid)
                if identity is None:
                    raise ProcessGroupError("Worker identity was lost before launch")
                ticks = identity.start_ticks
                lease = ProcessGroupLease(process.pid, ticks)
            self.store.set_worker(job["id"], process.pid, ticks)
            log_thread = threading.Thread(
                target=self._drain_log, args=(process.stdout, directory / "worker.log"), daemon=True
            )
            log_thread.start()
            os.write(write_fd, b"1")
            os.close(write_fd)
            write_fd = None
            config = job_input.get(
                "training_config" if job["kind"] == "training" else "generation_config", {}
            )
            walltime = min(
                self.settings.max_walltime_seconds,
                int(config.get("walltime_seconds", self.settings.max_walltime_seconds)),
            )
            deadline, cursor, next_size_check = time.monotonic() + walltime, 0, 0.0
            if self.settings.gpu_deadline is not None:
                deadline = min(
                    deadline,
                    time.monotonic()
                    + max(
                        0,
                        self.settings.gpu_deadline
                        - time.time()
                        - self.settings.deadline_reserve_seconds,
                    ),
                )
            reason = None
            while True:
                self.heartbeat = time.monotonic()
                cursor = self._events(directory, cursor)
                state = self.store.get(job["id"])["status"]
                if state == "cancelling":
                    reason = "cancelled"
                    break
                if self.stop_event.is_set():
                    reason = "service_stopped"
                    break
                if time.monotonic() >= deadline:
                    reason = "worker_timeout"
                    break
                if time.monotonic() >= next_size_check:
                    self._check_outputs_size(directory)
                    next_size_check = time.monotonic() + 1
                # Inspect the session before reaping its leader so ownership remains anchored.
                if lease is not None:
                    live = lease.has_live_members()
                    leader = read_process(process.pid)
                    finished = leader is None or leader.state in {"Z", "X"}
                    if finished:
                        if live:
                            self._cleanup(process, lease)
                        else:
                            process.wait(timeout=5)
                        cleaned = True
                        break
                elif process.poll() is not None:
                    cleaned = True
                    break
                self.wake_event.wait(self.settings.poll_seconds)
                self.wake_event.clear()
            if not cleaned:
                self._cleanup(process, lease)
                cleaned = True
            log_thread.join(timeout=5)
            if log_thread.is_alive():
                raise ProcessGroupError("Worker output pipe remains open after cleanup")
            self._events(directory, cursor)
            if reason == "cancelled":
                self.store.finish(job["id"], "cancelled")
            elif reason:
                self.store.finish(
                    job["id"],
                    "failed",
                    error={
                        "code": reason,
                        "message": "The job stopped before completion. "
                        "Submit a new request to retry.",
                        "retryable": True,
                    },
                )
            elif process.returncode != 0:
                self.store.finish(job["id"], "failed", error=self._worker_error(directory))
            else:
                result, artifacts = verify_outputs(directory, self.settings)
                self.store.finish(job["id"], "succeeded", result=result, artifacts=artifacts)
        except Exception as error:
            if process is not None and not cleaned:
                try:
                    if sys.platform == "linux" and lease is None:
                        terminate_uncaptured_child(process)
                    else:
                        self._cleanup(process, lease)
                    cleaned = True
                except Exception:
                    self.fatal = "worker_cleanup_required"
                    # Keep the durable running/cancelling state for startup recovery.
                    raise ProcessGroupError("Cannot confirm worker cleanup") from error
            if isinstance(error, ProcessGroupError):
                self.fatal = "worker_cleanup_required"
                raise
            self.store.finish(
                job["id"],
                "failed",
                error={
                    "code": "invalid_worker_output" if process else "invalid_job_input",
                    "message": "The job could not produce a verified result. "
                    "Submit a new request to retry.",
                    "retryable": True,
                },
            )
        finally:
            for descriptor in (read_fd, write_fd):
                if descriptor is not None:
                    os.close(descriptor)
            if log_thread:
                log_thread.join(timeout=5)
            self.active_pid = None
            if self.store.get(job["id"])["status"] != "succeeded" and (process is None or cleaned):
                self._remove_unpublished(job["id"])
