"""One supervised, private-directory job per process; no GPU imports on module load."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import finish_artifacts, write_json
from .config import LIMITS, InputError, validate_job_input


class JobCancelled(RuntimeError):
    pass


class JobTimeout(RuntimeError):
    pass


class EventWriter:
    def __init__(self, path: Path):
        self.path = path
        self.count = 0
        self.stream = path.open("x", encoding="utf-8")
        os.chmod(path, 0o600)

    def __call__(
        self,
        stage: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        unit: str | None = None,
        message: str | None = None,
    ) -> None:
        if self.count >= 1000:
            raise RuntimeError("The worker event limit was exceeded.")
        event: dict[str, Any] = {"stage": stage, "timestamp": datetime.now(UTC).isoformat()}
        if completed is not None:
            if total is None or not 0 <= completed <= total or not unit:
                raise ValueError("Progress must describe actual completed work.")
            event["progress"] = {"completed": completed, "total": total, "unit": unit}
        if message:
            event["message"] = message[:240]
        self.stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        self.count += 1

    def close(self) -> None:
        self.stream.close()


def read_input(path: Path) -> dict[str, Any]:
    maximum = LIMITS["max_body_bytes"] + 65536  # Bounded server-derived registry/preset metadata.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise InputError("The private input must be a bounded regular JSON file.")
        contents = stream.read(maximum + 1)
    if len(contents) > maximum:
        raise InputError("The job input is too large.")
    try:
        return json.loads(contents)
    except (ValueError, UnicodeError) as exc:
        raise InputError("The job input is not valid JSON.") from exc


def error_record(error: Exception) -> dict[str, Any]:
    from .generation import GenerationError

    if isinstance(error, InputError):
        return {"code": "invalid_input", "message": str(error)[:500], "retryable": False}
    if isinstance(error, GenerationError):
        return {"code": "generation_invalid", "message": str(error)[:500], "retryable": True}
    if isinstance(error, JobCancelled):
        return {"code": "cancelled", "message": "The job was cancelled.", "retryable": True}
    if isinstance(error, JobTimeout):
        return {
            "code": "time_limit",
            "message": "The job reached its time limit. Try a smaller model or dataset.",
            "retryable": True,
        }
    if "out of memory" in str(error).lower():
        return {
            "code": "out_of_memory",
            "message": "The GPU ran out of memory. Try a smaller model.",
            "retryable": True,
        }
    return {
        "code": "worker_failed",
        "message": "The GPU job could not finish. Your source data is unchanged. Please retry.",
        "retryable": True,
    }


def execute_job(job_dir: Path, kind: str, *, runners: dict[str, Any] | None = None) -> int:
    """Injection is for CPU behavior tests; the CLI always uses the real GPU runners."""
    from amd_inference_opt.resource_lock import exclusive_gpu_lock

    if job_dir.is_symlink() or not job_dir.is_dir():
        raise ValueError("The job directory must be a real existing private directory.")
    job_dir = job_dir.resolve()
    for name in ("result.json", "error.json", "events.jsonl", "artifacts", "adapter"):
        if (job_dir / name).exists() or (job_dir / name).is_symlink():
            raise ValueError(
                "This job directory has already been used; refusing to duplicate its work."
            )
    emit = EventWriter(job_dir / "events.jsonl")
    try:
        job = validate_job_input(kind, read_input(job_dir / "input.json"))
        emit("waiting_for_gpu", message="Waiting for exclusive GPU access.")
        lock_path = Path(os.environ.get("MACFIT_GPU_LOCK", "/tmp/macfit-training-gpu.lock"))
        with exclusive_gpu_lock(lock_path):
            if runners is None:
                from .generation import run_generation
                from .trainer import run_training

                runners = {"generation": run_generation, "training": run_training}
            result = runners[kind](job, job_dir, emit)
            emit("packaging_artifacts", message="Verifying downloadable artifacts.")
            result["artifacts"] = finish_artifacts(
                job_dir,
                job,
                result,
                adapter_dir=job_dir / "adapter" if kind == "training" else None,
            )
            write_json(job_dir / "result.json", result)
        emit("succeeded")
        return 0
    except Exception as error:
        write_json(job_dir / "error.json", error_record(error))
        # Detailed diagnostic is private supervisor output; public message is bounded above.
        traceback.print_exc()
        return 2
    finally:
        emit.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one bounded MacFit GPU job.")
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("generation", "training"))
    args = parser.parse_args(argv)
    os.umask(0o077)

    def cancel(_signum: int, _frame: Any) -> None:
        raise JobCancelled()

    def timeout(_signum: int, _frame: Any) -> None:
        raise JobTimeout()

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, LIMITS["walltime_seconds"])
    try:
        return execute_job(args.job_dir, args.kind)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    raise SystemExit(main())
