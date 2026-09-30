from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

import pytest

from macfit_training.service.settings import Settings
from macfit_training.service.store import JobStore
from macfit_training.service.supervisor import Supervisor

SUCCESS = """
import hashlib,json,os
from pathlib import Path
p=Path('.')
(p/'events.jsonl').write_text(json.dumps({'stage':'generating','progress':{'completed':3,'total':3,'unit':'examples'}})+'\\n')
a=p/'artifacts'; a.mkdir()
f=a/'generation.json'; f.write_text(json.dumps({'examples':[{'input':'A','output':'B'}]}))
b=f.read_bytes()
d={'id':'generation-json','type':'dataset','name':f.name,'size_bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
(p/'result.json').write_text(json.dumps({'method':'model_generation','artifacts':[d]}))
print('x'*100000)
"""


def make_store(tmp_path, **changes):
    settings = Settings(
        tmp_path,
        "synthetic-gateway-secret-test-value",
        min_free_bytes=0,
        poll_seconds=0.02,
        termination_grace_seconds=0.1,
        **changes,
    )
    return JobStore(settings), settings


def queue(store, owner="alice"):
    return store.create(
        owner,
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        "generation",
        {
            "base_model": {"id": "synthetic", "revision": "a" * 40},
            "generation_config": {"walltime_seconds": 3600},
        },
    )[0]


def wait_for(store, job_id, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get(job_id)
        if predicate(job):
            return job
        time.sleep(0.02)
    pytest.fail(f"Job never reached expected state: {store.get(job_id)}")


def test_supervisor_runs_process_and_publishes_verified_outputs(tmp_path, monkeypatch):
    store, settings = make_store(tmp_path, max_log_bytes=1024)
    monkeypatch.setenv("MACFIT_TRAINING_GATEWAY_SECRET", "never-copy-this-to-worker")
    supervisor = Supervisor(
        store,
        settings,
        readiness=lambda: True,
        command_factory=lambda *_: [sys.executable, "-c", SUCCESS],
    )
    assert "MACFIT_TRAINING_GATEWAY_SECRET" not in supervisor._environment()
    job = queue(store)
    supervisor.start()
    try:
        completed = wait_for(store, job["id"], lambda j: j["status"] in ("failed", "succeeded"))
        assert completed["status"] == "succeeded"
        assert completed["result"] == {"method": "model_generation"}
        assert completed["progress"] == {"completed": 3, "total": 3, "unit": "examples"}
        assert completed["artifacts"][0]["size_bytes"] > 0
        assert completed["worker_pid"] is None
        assert (store.directory(job["id"]) / "worker.log").stat().st_size == 1024
    finally:
        supervisor.stop()


def test_cancellation_waits_for_process_cleanup(tmp_path):
    store, settings = make_store(tmp_path)
    supervisor = Supervisor(
        store,
        settings,
        readiness=lambda: True,
        command_factory=lambda *_: [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    job = queue(store)
    supervisor.start()
    try:
        running = wait_for(store, job["id"], lambda j: j["worker_pid"] is not None)
        assert store.cancel(job["id"], "alice")["status"] == "cancelling"
        supervisor.notify()
        result = wait_for(store, job["id"], lambda j: j["status"] == "cancelled")
        assert result["artifacts"] == [] and result["worker_pid"] is None
        with pytest.raises(ProcessLookupError):
            os.kill(running["worker_pid"], 0)
    finally:
        supervisor.stop()


def test_timeout_failure_does_not_publish_outputs(tmp_path):
    store, settings = make_store(tmp_path, max_walltime_seconds=1)
    supervisor = Supervisor(
        store,
        settings,
        readiness=lambda: True,
        command_factory=lambda *_: [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    job = queue(store)
    supervisor.start()
    try:
        result = wait_for(store, job["id"], lambda j: j["status"] == "failed")
        assert result["error"]["code"] == "worker_timeout"
        assert not (store.directory(job["id"]) / "artifacts").exists()
    finally:
        supervisor.stop()


def test_restart_never_replays_interrupted_worker_and_flock_prevents_two_consumers(tmp_path):
    store, settings = make_store(tmp_path)
    interrupted = queue(store)
    store.claim()
    pending = queue(store, "bob")
    supervisor = Supervisor(store, settings, readiness=lambda: False)
    duplicate = Supervisor(store, settings, readiness=lambda: False)
    supervisor.start()
    try:
        assert store.get(interrupted["id"])["error"]["code"] == "service_restarted"
        assert store.get(pending["id"])["status"] == "queued"
        with pytest.raises(BlockingIOError):
            duplicate.start()
    finally:
        supervisor.stop()


def test_bad_hash_fails_without_exposing_or_preserving_unverified_artifacts(tmp_path):
    store, settings = make_store(tmp_path)
    code = SUCCESS.replace("'sha256':hashlib.sha256(b).hexdigest()", "'sha256':'0'*64")
    supervisor = Supervisor(
        store,
        settings,
        readiness=lambda: True,
        command_factory=lambda *_: [sys.executable, "-c", code],
    )
    job = queue(store)
    supervisor.start()
    try:
        result = wait_for(store, job["id"], lambda j: j["status"] == "failed")
        assert result["error"]["code"] == "invalid_worker_output"
        assert result["artifacts"] == []
        assert not (store.directory(job["id"]) / "artifacts").exists()
    finally:
        supervisor.stop()


def test_launcher_never_executes_worker_if_supervisor_dies_before_permission(tmp_path):
    read_fd, write_fd = os.pipe()
    marker = tmp_path / "should-not-exist"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "macfit_training.service.launcher",
            "--gate-fd",
            str(read_fd),
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        pass_fds=(read_fd,),
    )
    os.close(read_fd)
    os.close(write_fd)
    assert process.wait(timeout=5) == 125
    assert not marker.exists()
