"""Bounded cleanup of a Linux session without equating leader exit with completion."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class ProcessGroupError(RuntimeError):
    """The managed process group could not be proven stopped."""


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    group: int
    session: int
    state: str


def read_process(pid: int, proc_root: Path = Path("/proc")) -> ProcessIdentity | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as error:
        raise ProcessGroupError(f"cannot inspect managed process {pid}") from error
    tail = raw[raw.rfind(")") + 1 :].split()
    try:
        return ProcessIdentity(pid, int(tail[19]), int(tail[2]), int(tail[3]), tail[0])
    except (IndexError, ValueError) as error:
        raise ProcessGroupError(f"invalid process identity for {pid}") from error


def session_has_live_members(pid: int, proc_root: Path = Path("/proc")) -> bool:
    """Read-only recovery check; never signal a session with a lost leader."""
    try:
        entries = list(proc_root.iterdir())
    except OSError as error:
        raise ProcessGroupError("cannot enumerate managed session") from error
    for entry in entries:
        if entry.name.isdecimal():
            identity = read_process(int(entry.name), proc_root)
            if (
                identity is not None
                and identity.session == pid
                and identity.state not in {"Z", "X"}
            ):
                return True
    return False


def terminate_uncaptured_child(
    process: subprocess.Popen, *, proc_root: Path = Path("/proc"),
) -> None:
    """Emergency rollback before a freshly owned Popen child has been reaped.

    The parent still owns this PID, so its initial process-group number cannot
    be reused. Without /proc identity we cannot safely discover other groups.
    Stop the known initial group, then require separate session verification.
    """
    if process.returncode is not None:
        raise ProcessGroupError("uncaptured child was already reaped; cleanup ownership is lost")
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as error:
        raise ProcessGroupError("cannot roll back uncaptured child process group") from error
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        raise ProcessGroupError("uncaptured child survived emergency termination") from error
    deadline = time.monotonic() + 5
    while session_has_live_members(process.pid, proc_root):
        if time.monotonic() >= deadline:
            raise ProcessGroupError(
                "initial group stopped but other session workers remain unverified"
            )
        time.sleep(0.05)


class ProcessGroupLease:
    """Track exact members of a session created with ``start_new_session=True``.

    Zombies cannot execute GPU work and do not delay cleanup.  Identity checks
    still retain them as anchors until the caller reaps its Popen child.
    This is a local session boundary, not a Docker/cgroup cleanup implementation.
    """

    def __init__(self, pid: int, start_ticks: int, *, proc_root: Path = Path("/proc")):
        self.pid = pid
        self.proc_root = proc_root
        leader = read_process(pid, proc_root)
        if (
            leader is None
            or leader.start_ticks != start_ticks
            or leader.group != pid
            or leader.session != pid
        ):
            raise ProcessGroupError("managed session leader identity is unavailable or changed")
        self.members = {pid: leader}
        self._refresh()

    def _refresh(self) -> list[ProcessIdentity]:
        current: dict[int, ProcessIdentity] = {}
        try:
            entries = list(self.proc_root.iterdir())
        except OSError as error:
            raise ProcessGroupError("cannot enumerate managed process group") from error
        for entry in entries:
            if entry.name.isdecimal():
                identity = read_process(int(entry.name), self.proc_root)
                if identity is not None:
                    current[identity.pid] = identity
        retained = {
            pid: identity
            for pid, previous in self.members.items()
            if (identity := current.get(pid)) is not None
            and identity.start_ticks == previous.start_ticks
        }
        group = {
            pid: identity for pid, identity in current.items()
            if identity.session == self.pid
        }
        if group and not any(item.session == self.pid for item in retained.values()):
            raise ProcessGroupError(
                "process group has members but its ownership cannot be verified"
            )
        self.members = {**retained, **group}
        return [identity for identity in self.members.values() if identity.state not in {"Z", "X"}]

    def _signal(self, members: list[ProcessIdentity], sig: int) -> None:
        for previous in members:
            descriptor: int | None = None
            try:
                # A pidfd binds the signal to the opened process even if its
                # numeric PID is reused between inspection and delivery.
                descriptor = os.pidfd_open(previous.pid)
                current = read_process(previous.pid, self.proc_root)
                if current is None or current.start_ticks != previous.start_ticks:
                    continue
                signal.pidfd_send_signal(descriptor, sig)
            except ProcessLookupError:
                pass
            except (OSError, AttributeError) as error:
                raise ProcessGroupError(f"cannot signal managed process {previous.pid}") from error
            finally:
                if descriptor is not None:
                    os.close(descriptor)

    def has_live_members(self) -> bool:
        return bool(self._refresh())

    def terminate(self, *, grace_seconds: float, kill_seconds: float = 5.0) -> None:
        members = self._refresh()
        self._signal(members, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while members := self._refresh():
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if not members:
            return
        deadline = time.monotonic() + kill_seconds
        while members := self._refresh():
            self._signal(members, signal.SIGKILL)
            if time.monotonic() >= deadline:
                raise ProcessGroupError("managed process group survived bounded termination")
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
