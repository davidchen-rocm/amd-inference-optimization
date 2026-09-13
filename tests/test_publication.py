from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_publication.py"
_SPEC = importlib.util.spec_from_file_location("check_publication", TOOL)
assert _SPEC is not None and _SPEC.loader is not None
publication = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(publication)


def _git(repo: Path, *args: str, content: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=content,
        capture_output=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "--quiet")
    return tmp_path


def _stage(repo: Path, path: str, content: bytes) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    _git(repo, "add", "--", path)


def _private_coordinates() -> bytes:
    return (
        "first line\n"
        + "/".join(("", "home", "private-person", "model"))
        + "\n"
        + "private.person"
        + "@"
        + "private-provider"
        + ".com\n"
        + ".".join(("198", "18", "0", "1"))
        + "\n"
    ).encode()


def test_reads_staged_bytes_in_both_directions_after_partial_staging(repo: Path) -> None:
    _stage(repo, "report.md", _private_coordinates())
    (repo / "report.md").write_text("already sanitized in working tree\n")

    assert publication.scan_index(repo) == [
        ("ipv4-endpoint", "report.md", 4),
        ("personal-email", "report.md", 3),
        ("personal-home", "report.md", 2),
    ]

    _git(repo, "add", "report.md")
    (repo / "report.md").write_bytes(_private_coordinates())
    assert publication.scan_index(repo) == []


def test_example_coordinates_and_public_noreply_addresses_are_allowed(repo: Path) -> None:
    _stage(
        repo,
        "examples.md",
        (
            b"/home/USER/model\n/Users/USERNAME/project\nC:\\Users\\USER\\project\n"
            b"tests@example.com tests@example.org tests@example.net tests@example.invalid\n"
            b"test@users.noreply.github.com\n"
            b"127.0.0.1 0.0.0.0 192.0.2.10 198.51.100.8 203.0.113.20\n"
        ),
    )
    assert publication.scan_index(repo) == []


@pytest.mark.parametrize(
    "path",
    [
        ".gpuopt-live/state.json",
        ".qwen35-work/data.txt",
        "work/job.txt",
        "live-decisions/run.json",
        "artifacts/result.json",
        "tmp/run.log",
        ".venv-vllm/config.txt",
    ],
)
def test_rejects_local_generated_roots(repo: Path, path: str) -> None:
    _stage(repo, path, b"local data")
    assert publication.scan_index(repo) == [("local-artifact", path, 0)]


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.production",
        "config/credentials.json",
        "keys/id_ed25519",
        "keys/private.pem",
        "keys/private.key",
    ],
)
def test_rejects_credential_filenames(repo: Path, path: str) -> None:
    _stage(repo, path, b"placeholder")
    assert publication.scan_index(repo) == [("credential-path", path, 0)]


@pytest.mark.parametrize("content", [b"binary\x00payload", b"invalid\xffutf8"])
def test_rejects_unscannable_blobs(repo: Path, content: bytes) -> None:
    _stage(repo, "data.bin", content)
    assert publication.scan_index(repo) == [("unscannable-content", "data.bin", 0)]


def test_rejects_oversized_blobs(repo: Path) -> None:
    _stage(repo, "large.txt", b"a" * 33)
    assert publication.scan_index(repo, max_blob_bytes=32) == [
        ("oversized-content", "large.txt", 0)
    ]


def test_unresolved_index_entries_block_publication(repo: Path) -> None:
    object_id = _git(repo, "hash-object", "-w", "--stdin", content=b"conflict")
    _git(
        repo,
        "update-index",
        "--index-info",
        content=(
            b"100644 " + object_id + b" 1\tconflict.txt\n"
            b"100644 " + object_id + b" 2\tconflict.txt\n"
        ),
    )
    assert publication.scan_index(repo) == [("unresolved-index", "conflict.txt", 0)]


def test_nested_gitlink_blocks_publication(repo: Path) -> None:
    tree = _git(repo, "write-tree").decode()
    commit = _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit-tree",
        tree,
        content=b"fixture\n",
    ).decode()
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{commit},vendor/runtime")
    assert publication.scan_index(repo) == [("nested-gitlink", "vendor/runtime", 0)]


def test_git_replace_cannot_hide_staged_private_content(repo: Path) -> None:
    _stage(repo, "report.md", _private_coordinates())
    original = _git(repo, "rev-parse", ":report.md").decode()
    replacement = _git(repo, "hash-object", "-w", "--stdin", content=b"sanitized").decode()
    _git(repo, "replace", original, replacement)

    assert len(publication.scan_index(repo)) == 3


def test_cli_reports_locations_without_echoing_private_values(repo: Path) -> None:
    private = _private_coordinates()
    _stage(repo, "report.md", private)
    result = subprocess.run(
        [sys.executable, str(TOOL), "--repo", str(repo)],
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == b""
    assert result.stdout == (
        b"ipv4-endpoint report.md:4\npersonal-email report.md:3\npersonal-home report.md:2\n"
    )
    for value in private.splitlines()[1:]:
        assert value not in result.stdout


def test_git_read_failure_blocks_publication(tmp_path: Path) -> None:
    assert publication.scan_index(tmp_path) == [("index-read-error", ".", 0)]
