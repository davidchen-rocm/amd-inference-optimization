from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    gateway_secret: str = field(repr=False)
    firebase_project_id: str = "macfit-example"
    gpu_ready_file: Path | None = None
    max_body_bytes: int = 3 * 1024 * 1024
    max_queue_jobs: int = 8
    max_owner_jobs: int = 100
    max_total_jobs: int = 1000
    max_storage_bytes: int = 20 * 1024**3
    min_free_bytes: int = 2 * 1024**3
    max_log_bytes: int = 2 * 1024 * 1024
    max_event_bytes: int = 4 * 1024 * 1024
    max_result_bytes: int = 8 * 1024 * 1024
    max_artifact_bytes: int = 1024**3
    max_job_artifact_bytes: int = 2 * 1024**3
    max_artifacts: int = 16
    max_walltime_seconds: int = 3600
    poll_seconds: float = 0.25
    termination_grace_seconds: float = 5.0
    gpu_deadline: float | None = None
    stop_accepting_at: float | None = None
    deadline_reserve_seconds: int = 120
    archive_only: bool = False

    def can_finish(self, walltime_seconds: float, *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return not self.archive_only and (
            self.gpu_deadline is None
            or now + walltime_seconds + self.deadline_reserve_seconds < self.gpu_deadline
        )

    def accepts_new(self, *, queued: int = 0, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return (self.stop_accepting_at is None or now < self.stop_accepting_at) and self.can_finish(
            (queued + 1) * self.max_walltime_seconds, now=now
        )

    def __post_init__(self):
        object.__setattr__(self, "data_dir", Path(self.data_dir).resolve())
        object.__setattr__(self, "gateway_secret", self.gateway_secret.strip())
        if (self.data_dir / "archive-mode.json").exists():
            object.__setattr__(self, "archive_only", True)
        if len(self.gateway_secret.encode()) < 32:
            raise ValueError("A private gateway secret of at least 32 bytes is required.")
        if self.gpu_ready_file is not None:
            object.__setattr__(self, "gpu_ready_file", Path(self.gpu_ready_file).resolve())

    def gpu_ready(self) -> bool:
        if self.archive_only or self.gpu_ready_file is None:
            return False
        try:
            if self.gpu_ready_file.is_symlink() or self.gpu_ready_file.stat().st_size > 4096:
                return False
            data = json.loads(self.gpu_ready_file.read_text())
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            return data.get("ready") is True and data.get("boot_id") == boot_id
        except (OSError, ValueError, AttributeError):
            return False

    @classmethod
    def from_env(cls):
        def instant(name):
            value = os.environ.get(name)
            if not value:
                return None
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError(f"{name} must include a timezone")
            return parsed.timestamp()

        return cls(
            data_dir=Path(os.environ.get("MACFIT_TRAINING_DATA", "/srv/macfit-training/data")),
            gateway_secret=os.environ.get("MACFIT_TRAINING_GATEWAY_SECRET", ""),
            firebase_project_id=os.environ.get(
                "MACFIT_FIREBASE_PROJECT_ID", "macfit-example"
            ),
            gpu_ready_file=Path(
                os.environ.get("MACFIT_GPU_READY_FILE", "/srv/macfit-training/gpu-ready.json")
            ),
            gpu_deadline=instant("MACFIT_GPU_DEADLINE"),
            stop_accepting_at=instant("MACFIT_STOP_ACCEPTING_AT"),
            archive_only=os.environ.get("MACFIT_ARCHIVE_ONLY", "0") == "1",
        )
