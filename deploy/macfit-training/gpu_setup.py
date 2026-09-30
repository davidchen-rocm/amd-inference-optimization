#!/usr/bin/env python3
"""Idempotent GPU-host service provisioning. Dry run unless --apply is explicit.

Run only after any root-owned smoke worker has finished:
  python3 gpu_setup.py --env-file /srv/macfit-training/gpu.env --apply --start
No packages, SSH keys, GPU jobs, or readiness markers are installed by this script.
"""

from __future__ import annotations

import argparse
import grp
import json
import os
import pwd
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

USER = "macfit-train"
UNIT_NAME = "macfit-training.service"
ENV_DESTINATION = Path("/etc/macfit-training/gpu.env")
UNIT_DESTINATION = Path("/etc/systemd/system") / UNIT_NAME


def quote(value: str | Path) -> str:
    value = str(value)
    if any(character in value for character in ("\n", "\r", "\x00", "%")):
        raise ValueError(
            "Deployment paths must not contain control characters or systemd specifiers"
        )
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def directive_path(value: Path) -> str:
    """WorkingDirectory/EnvironmentFile parse a literal path, not an ExecStart argument."""
    path = str(value)
    if not value.is_absolute() or any(char.isspace() or char in '\x00%"\\' for char in path):
        raise ValueError("Systemd deployment paths must be absolute and contain no whitespace")
    return path


def render_unit(base: Path) -> str:
    base = base.absolute()
    writable_paths = " ".join(directive_path(base / name) for name in ("data", "cache", "tmp"))
    # EnvironmentFile overrides these defaults; validate_env checks owned path overrides.
    return f"""[Unit]
Description=MacFit private GPU training API
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User={USER}
Group={USER}
WorkingDirectory={directive_path(base / "repo")}
Environment=PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
Environment={quote("PYTHONPATH=" + str(base / "repo/src"))}
Environment={quote("MACFIT_TRAINING_DATA=" + str(base / "data"))}
Environment={quote("MACFIT_MODEL_CACHE=" + str(base / "cache"))}
Environment={quote("MACFIT_GPU_LOCK=" + str(base / "data/gpu.lock"))}
Environment={quote("MACFIT_GPU_READY_FILE=" + str(base / "gpu-ready.json"))}
Environment={quote("HF_HOME=" + str(base / "cache/huggingface"))}
Environment={quote("TMPDIR=" + str(base / "tmp"))}
Environment=HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
EnvironmentFile={directive_path(ENV_DESTINATION)}
ExecStart={quote(base / "venv/bin/python")} -m macfit_training.service
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
KillMode=control-group
SendSIGKILL=yes
UMask=0077
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ProtectControlGroups=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
LockPersonality=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
RemoveIPC=yes
CapabilityBoundingSet=
AmbientCapabilities=
ReadWritePaths={writable_paths}
LimitMEMLOCK=infinity
LimitNOFILE=4096
TasksMax=1024
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""


def load_env(path: Path, base: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 16384 or info.st_mode & 0o077:
            raise ValueError("The environment source must be a private regular file (mode 0600)")
        content = stream.read(16385)
    if len(content) > 16384 or b"\x00" in content:
        raise ValueError("Invalid environment source")
    values = {}
    for line in content.decode("utf-8").splitlines():
        tokens = shlex.split(line, comments=True, posix=True)
        if not tokens:
            continue
        if len(tokens) != 1 or not re.fullmatch(r"[A-Z][A-Z0-9_]*=.*", tokens[0]):
            raise ValueError("Use one KEY=value assignment per environment line")
        key, value = tokens[0].split("=", 1)
        if key in values:
            raise ValueError("Duplicate environment assignment")
        values[key] = value
    if len(values.get("MACFIT_TRAINING_GATEWAY_SECRET", "").encode()) < 32:
        raise ValueError("A gateway secret of at least 32 bytes is required")
    expected = {
        "MACFIT_TRAINING_DATA": str(base / "data"),
        "MACFIT_MODEL_CACHE": str(base / "cache"),
        "MACFIT_GPU_LOCK": str(base / "data/gpu.lock"),
        "MACFIT_GPU_READY_FILE": str(base / "gpu-ready.json"),
    }
    for name, value in expected.items():
        if name in values and values[name] != value:
            raise ValueError(f"{name} does not match this deployment")
    if not re.fullmatch(r"[a-z][a-z0-9-]*", values.get("MACFIT_FIREBASE_PROJECT_ID", "")):
        raise ValueError("MACFIT_FIREBASE_PROJECT_ID must be a configured Firebase project ID")
    for name in ("MACFIT_GPU_DEADLINE", "MACFIT_STOP_ACCEPTING_AT"):
        if name not in values:
            raise ValueError(f"{name} is required for this temporary GPU deployment")
        parsed = datetime.fromisoformat(values[name].replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError(f"{name} must include a timezone")
    return content.rstrip(b"\n") + b"\n"


def atomic_write(path: Path, content: bytes, mode: int) -> bool:
    if path.is_symlink():
        raise ValueError("Refusing to replace a symbolic-link deployment file")
    if path.exists() and path.read_bytes() == content:
        os.chmod(path, mode)
        return False
    descriptor, temporary = tempfile.mkstemp(prefix=".macfit-install-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True


def private_tree(path: Path, uid: int, gid: int):
    if path.is_symlink():
        raise ValueError("Managed data/cache directories cannot be symbolic links")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    for parent, directories, filenames in os.walk(path, followlinks=False):
        directory = Path(parent)
        os.chown(directory, uid, gid, follow_symlinks=False)
        os.chmod(directory, 0o700)
        for name in directories + filenames:
            child = directory / name
            info = child.lstat()
            os.chown(child, uid, gid, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                continue  # Hugging Face snapshots use links to blobs; never follow them.
            if stat.S_ISREG(info.st_mode):
                # Preserve read-only immutable inputs; remove access for other users.
                os.chmod(child, (stat.S_IMODE(info.st_mode) & 0o700) | 0o400)
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError("Unexpected device or special file in managed data")


def run(*arguments: str, check: bool = True):
    return subprocess.run(
        arguments, check=check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def apply(base: Path, env_file: Path, *, start: bool):
    if sys.platform != "linux" or os.geteuid() != 0:
        raise ValueError("Provisioning requires Linux root")
    if base.is_symlink() or not (base / "repo/src/macfit_training/service/api.py").is_file():
        raise ValueError("The checked deployment source is missing")
    if not os.access(base / "venv/bin/python", os.X_OK):
        raise ValueError("The prepared service virtual environment is missing")
    content = load_env(env_file, base)
    try:
        group = grp.getgrnam(USER)
    except KeyError:
        run("groupadd", "--system", USER)
        group = grp.getgrnam(USER)
    try:
        account = pwd.getpwnam(USER)
    except KeyError:
        run(
            "useradd",
            "--system",
            "--gid",
            USER,
            "--home-dir",
            str(base),
            "--shell",
            "/usr/sbin/nologin",
            USER,
        )
        account = pwd.getpwnam(USER)
    if account.pw_uid == 0:
        raise ValueError("The service account must not be root")
    if account.pw_gid != group.gr_gid:
        raise ValueError("The existing service account has an unexpected primary group")
    groups = []
    for name in ("render", "video"):
        try:
            grp.getgrnam(name)
            groups.append(name)
        except KeyError:
            pass
    if groups:
        run("usermod", "--append", "--groups", ",".join(groups), USER)
    for name in ("data", "cache", "tmp"):
        private_tree(base / name, account.pw_uid, account.pw_gid)
    ENV_DESTINATION.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(ENV_DESTINATION.parent, 0o700)
    changed = atomic_write(ENV_DESTINATION, content, 0o600)
    changed = atomic_write(UNIT_DESTINATION, render_unit(base).encode(), 0o644) or changed
    run("systemd-analyze", "verify", str(UNIT_DESTINATION))
    # CPU import only; no model/GPU load, and no credentials are needed for this check.
    run(
        "runuser",
        "-u",
        USER,
        "--",
        "env",
        "PYTHONPATH=" + str(base / "repo/src"),
        str(base / "venv/bin/python"),
        "-c",
        "import macfit_training.service.api",
    )
    run("systemctl", "daemon-reload")
    run("systemctl", "enable", UNIT_NAME)
    if start:
        active = run("systemctl", "is-active", "--quiet", UNIT_NAME, check=False).returncode == 0
        run("systemctl", "restart" if active and changed else "start", UNIT_NAME)
        run("systemctl", "is-active", "--quiet", UNIT_NAME)
    return {"applied": True, "changed": changed, "started": start, "service": UNIT_NAME}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/srv/macfit-training"))
    parser.add_argument("--env-file", type=Path, default=Path("/srv/macfit-training/gpu.env"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--print-unit", action="store_true")
    args = parser.parse_args(argv)
    base = args.base.absolute()
    try:
        unit = render_unit(base)
        if args.print_unit:
            print(unit, end="")
        elif args.apply:
            print(json.dumps(apply(base, args.env_file, start=args.start), sort_keys=True))
        else:
            print(
                json.dumps(
                    {
                        "apply": False,
                        "user": USER,
                        "service": UNIT_NAME,
                        "base": str(base),
                        "environment_source": str(args.env_file),
                        "environment_destination": str(ENV_DESTINATION),
                        "listener": "127.0.0.1:8791",
                        "start_requested": args.start,
                    },
                    sort_keys=True,
                )
            )
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        print(
            "GPU provisioning did not complete. Check the private environment file, "
            "prepared source/venv, directory ownership, and systemd status.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
