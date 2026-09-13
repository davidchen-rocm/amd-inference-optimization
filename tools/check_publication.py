#!/usr/bin/env python3
"""Check staged publication content for local artifacts and common private coordinates.

This limited privacy gate supplements a dedicated secret scanner such as Gitleaks.
It reads immutable index object IDs, including partially staged files, and never
prints matched content. It does not inspect history or unstaged files.
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import subprocess
from pathlib import Path, PurePosixPath

MAX_BLOB_BYTES = 5 * 1024 * 1024
Finding = tuple[str, str, int]
_HOME = re.compile(r"/(?:home|Users)/([^/\s\"'`<>]+)|[A-Za-z]:[\\/]+Users[\\/]+([^\\/\s\"'`<>]+)")
_PLACEHOLDERS = {"USER", "USERNAME", "YOUR_USER", "YOUR_USERNAME", "$USER", "${USER}"}
_EMAIL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
_EXAMPLE_MAIL_DOMAINS = {"example.com", "example.org", "example.net", "noreply.github.com"}
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_EXAMPLE_NETWORKS = tuple(
    ipaddress.IPv4Network(value)
    for value in ("127.0.0.0/8", "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
)
_CREDENTIAL_NAMES = {
    ".netrc",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "secrets.json",
    "service-account.json",
    "service_account.json",
}


def _git(repo: Path, *args: str, input_data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "--no-replace-objects", "-C", str(repo), *args],
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=30,
    ).stdout


def _path_category(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    root = parts[0]
    if root.startswith((".gpuopt", ".venv")) or root in {
        ".qwen35-work",
        "work",
        "live-decisions",
        "artifacts",
        "tmp",
    }:
        return "local-artifact"
    for part in parts:
        name = part.lower()
        if (
            name in _CREDENTIAL_NAMES
            or name.endswith((".pem", ".key"))
            or (name == ".env" or name.startswith(".env."))
            and name != ".env.example"
        ):
            return "credential-path"
    return None


def _content_findings(content: bytes) -> list[tuple[str, int]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return [("unscannable-content", 0)]
    if any(ord(value) < 32 and value not in "\t\r\n" for value in text):
        return [("unscannable-content", 0)]
    findings: list[tuple[str, int]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if any((match[1] or match[2]) not in _PLACEHOLDERS for match in _HOME.finditer(line)):
            findings.append(("personal-home", number))
        for match in _EMAIL.finditer(line):
            domain = match[1].lower()
            if not (
                domain in _EXAMPLE_MAIL_DOMAINS
                or domain.endswith((".invalid", ".test"))
                or domain == "users.noreply.github.com"
                or any(domain.endswith("." + example) for example in _EXAMPLE_MAIL_DOMAINS)
            ):
                findings.append(("personal-email", number))
                break
        for match in _IPV4.finditer(line):
            try:
                address = ipaddress.IPv4Address(match[0])
            except ipaddress.AddressValueError:
                continue
            if address != ipaddress.IPv4Address("0.0.0.0") and not any(
                address in network for network in _EXAMPLE_NETWORKS
            ):
                findings.append(("ipv4-endpoint", number))
                break
    return findings


def scan_index(repo: Path, *, max_blob_bytes: int = MAX_BLOB_BYTES) -> list[Finding]:
    """Return category/path/line findings; a read failure always blocks publication."""

    if max_blob_bytes < 1:
        raise ValueError("max_blob_bytes must be positive")
    findings: list[Finding] = []
    entries: list[tuple[str, str]] = []
    try:
        for row in _git(repo, "ls-files", "--stage", "-z").split(b"\0"):
            if not row:
                continue
            metadata, path_bytes = row.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split()
            path = path_bytes.decode("utf-8")
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_id):
                raise ValueError("invalid index object ID")
            category = (
                "unresolved-index"
                if stage != "0"
                else "nested-gitlink"
                if mode == "160000"
                else "unsupported-index-mode"
                if mode not in {"100644", "100755", "120000"}
                else _path_category(path)
            )
            if category:
                findings.append((category, path, 0))
            else:
                entries.append((object_id, path))
        if not entries:
            return sorted(set(findings))
        objects = list(dict.fromkeys(object_id for object_id, _ in entries))
        metadata = (
            _git(
                repo,
                "cat-file",
                "--batch-check",
                input_data=("\n".join(objects) + "\n").encode("ascii"),
            )
            .decode("ascii")
            .splitlines()
        )
        if len(metadata) != len(objects):
            raise ValueError("incomplete object metadata")
        sizes: dict[str, int] = {}
        for expected, line in zip(objects, metadata, strict=True):
            object_id, kind, size = line.split()
            if object_id != expected or kind != "blob" or int(size) < 0:
                raise ValueError("invalid index blob metadata")
            sizes[object_id] = int(size)
        cached: dict[str, list[tuple[str, int]]] = {}
        for object_id, path in entries:
            if sizes[object_id] > max_blob_bytes:
                findings.append(("oversized-content", path, 0))
                continue
            if object_id not in cached:
                content = _git(repo, "cat-file", "blob", object_id)
                if len(content) != sizes[object_id]:
                    raise ValueError("incomplete index blob")
                cached[object_id] = _content_findings(content)
            findings.extend((category, path, line) for category, line in cached[object_id])
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        findings.append(("index-read-error", ".", 0))
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    findings = scan_index(args.repo)
    for category, path, line in findings:
        # Quote control characters in unusual Git filenames without printing content.
        safe_path = path.encode("unicode_escape").decode("ascii")
        print(f"{category} {safe_path}:{line}")
    if not findings:
        print("Publication index privacy checks passed (dedicated secret scan still required).")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
