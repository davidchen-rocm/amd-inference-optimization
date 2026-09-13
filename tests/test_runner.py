from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from amd_inference_opt.runner import (
    ChangeSet,
    ExperimentRunner,
    ExperimentSpec,
    ExperimentStep,
    runner_spec_from_domain,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Tests")
    (repo / "value.txt").write_text("old\n", encoding="utf-8")
    _git(repo, "add", "value.txt")
    _git(repo, "commit", "-m", "initial")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_source_patch_uses_detached_worktree_and_runs_steps(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    patch = tmp_path / "change.patch"
    patch.write_text(
        "diff --git a/value.txt b/value.txt\n"
        "--- a/value.txt\n"
        "+++ b/value.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n",
        encoding="utf-8",
    )
    worktree = tmp_path / "worktrees" / "experiment-1"
    artifacts = tmp_path / "artifacts" / "experiment-1"
    spec = ExperimentSpec(
        experiment_id="experiment-1",
        source_repo=str(repo),
        base_commit=commit,
        worktree_path=str(worktree),
        artifact_dir=str(artifacts),
        change=ChangeSet(kind="source_patch", patch_path=str(patch)),
        steps=(
            ExperimentStep(
                name="build",
                argv=(
                    sys.executable,
                    "-c",
                    "from pathlib import Path; assert Path('value.txt').read_text() == 'new\\n'",
                ),
            ),
            ExperimentStep(
                name="smoke",
                argv=(sys.executable, "-c", "print('smoke-ok')"),
            ),
        ),
    )

    result = ExperimentRunner().execute(spec)

    assert result.succeeded
    assert result.build_status == "built"
    assert (worktree / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert (repo / "value.txt").read_text(encoding="utf-8") == "old\n"
    assert _git(worktree, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    assert (artifacts / "smoke.stdout").read_text(encoding="utf-8") == "smoke-ok\n"
    manifest = json.loads((artifacts / "runner-result.json").read_text(encoding="utf-8"))
    assert manifest["change"]["patch_sha256"]


def test_runtime_config_reuses_build_and_is_injected_only_at_runtime(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    worktree = tmp_path / "worktrees" / "experiment-config"
    artifacts = tmp_path / "artifacts" / "experiment-config"
    spec = ExperimentSpec(
        experiment_id="experiment-config",
        source_repo=str(repo),
        base_commit=commit,
        worktree_path=str(worktree),
        artifact_dir=str(artifacts),
        change=ChangeSet(kind="runtime_config", environment={"MAPPING": "split-k"}),
        steps=(
            ExperimentStep(
                name="e2e",
                argv=(
                    sys.executable,
                    "-c",
                    "import os; print(os.environ['MAPPING'])",
                ),
            ),
        ),
    )

    result = ExperimentRunner().execute(spec)

    assert result.succeeded
    assert result.build_status == "reused"
    assert result.worktree_create is None
    assert not worktree.exists()
    assert result.binary is not None
    assert result.binary.path == str(Path(sys.executable).resolve())
    assert result.step_results["e2e"].stdout == "split-k\n"


def test_runtime_config_rejects_wrong_reused_binary_hash(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    spec = ExperimentSpec(
        experiment_id="experiment-binary-mismatch",
        source_repo=str(repo),
        base_commit=commit,
        worktree_path=str(tmp_path / "must-not-exist"),
        artifact_dir=str(tmp_path / "artifacts"),
        change=ChangeSet(kind="runtime_config", environment={"MAPPING": "old"}),
        steps=(ExperimentStep(name="e2e", argv=(sys.executable, "-c", "print('x')")),),
        binary_path=sys.executable,
        expected_binary_sha256="0" * 64,
    )

    result = ExperimentRunner().execute(spec)

    assert result.failure_stage == "binary_identity"
    assert result.binary is not None
    assert result.binary.expected_sha256_matches is False
    assert result.step_results == {}


def test_patch_freeze_verifies_sha_and_exact_file_set(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    patch = tmp_path / "change.patch"
    patch.write_text(
        "diff --git a/value.txt b/value.txt\n"
        "--- a/value.txt\n"
        "+++ b/value.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    spec = ExperimentSpec(
        experiment_id="frozen-patch",
        source_repo=str(repo),
        base_commit=commit,
        worktree_path=str(tmp_path / "worktree"),
        artifact_dir=str(tmp_path / "artifacts"),
        change=ChangeSet(
            kind="source_patch",
            patch_path=str(patch),
            expected_patch_sha256=digest,
            allowed_patch_files=("value.txt",),
        ),
        steps=(ExperimentStep(name="build", argv=(sys.executable, "-c", "pass")),),
    )

    result = ExperimentRunner().execute(spec)

    assert result.succeeded
    assert result.change is not None
    assert result.change.patch_sha256 == digest
    assert result.change.changed_files == ("value.txt",)


def test_source_patch_runtime_environment_is_not_visible_to_build(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    patch = tmp_path / "change.patch"
    patch.write_text(
        "diff --git a/value.txt b/value.txt\n"
        "--- a/value.txt\n"
        "+++ b/value.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n",
        encoding="utf-8",
    )
    spec = ExperimentSpec(
        experiment_id="patch-runtime-coordinate",
        source_repo=str(repo),
        base_commit=commit,
        worktree_path=str(tmp_path / "worktree"),
        artifact_dir=str(tmp_path / "artifacts"),
        change=ChangeSet(
            kind="source_patch",
            patch_path=str(patch),
            environment={"Q4_RUNTIME_ONLY": "candidate"},
        ),
        steps=(
            ExperimentStep(
                name="build",
                argv=(
                    sys.executable,
                    "-c",
                    "import os; assert 'Q4_RUNTIME_ONLY' not in os.environ",
                ),
            ),
            ExperimentStep(
                name="smoke",
                argv=(
                    sys.executable,
                    "-c",
                    "import os; assert os.environ['Q4_RUNTIME_ONLY'] == 'candidate'",
                ),
            ),
        ),
    )

    result = ExperimentRunner().execute(spec)

    assert result.succeeded


def test_runtime_config_can_restore_default_by_unsetting_override(
    tmp_path: Path, monkeypatch: object
) -> None:
    repo, commit = _repo(tmp_path)
    monkeypatch.setenv("MAPPING", "old")  # type: ignore[attr-defined]
    spec = ExperimentSpec(
        experiment_id="experiment-unset",
        source_repo=str(repo),
        base_commit=commit,
        worktree_path=str(tmp_path / "worktrees" / "experiment-unset"),
        artifact_dir=str(tmp_path / "artifacts" / "experiment-unset"),
        change=ChangeSet(kind="runtime_config", unset_environment=("MAPPING",)),
        steps=(
            ExperimentStep(
                name="e2e",
                argv=(
                    sys.executable,
                    "-c",
                    "import os; print(os.getenv('MAPPING', 'default'))",
                ),
            ),
        ),
    )

    result = ExperimentRunner().execute(spec)

    assert result.succeeded
    assert result.step_results["e2e"].stdout == "default\n"


def test_runtime_failure_stops_later_steps_without_creating_worktree(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    worktree = tmp_path / "worktrees" / "experiment-fail"
    artifacts = tmp_path / "artifacts" / "experiment-fail"
    spec = ExperimentSpec(
        experiment_id="experiment-fail",
        source_repo=str(repo),
        base_commit=commit,
        worktree_path=str(worktree),
        artifact_dir=str(artifacts),
        change=ChangeSet(kind="runtime_config", environment={"MAPPING": "old"}),
        steps=(
            ExperimentStep(name="smoke", argv=(sys.executable, "-c", "raise SystemExit(2)")),
            ExperimentStep(name="e2e", argv=(sys.executable, "-c", "print('must not run')")),
        ),
    )

    result = ExperimentRunner().execute(spec)

    assert result.failure_stage == "smoke"
    assert set(result.step_results) == {"smoke"}
    assert not worktree.exists()
    assert result.worktree_create is None
    assert not (artifacts / "e2e.stdout").exists()


def test_projects_public_domain_models_into_runner_spec(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    patch = tmp_path / "change.patch"
    patch.write_text("patch evidence", encoding="utf-8")
    task = SimpleNamespace(
        runtime=SimpleNamespace(repo_path=repo, base_commit=commit),
        budgets=SimpleNamespace(build_timeout_seconds=99),
        benchmark=SimpleNamespace(timeout_seconds=33),
    )
    model_spec = SimpleNamespace(
        id="domain-experiment",
        worktree_path=tmp_path / "explicit-worktree",
        change=SimpleNamespace(
            kind="source_patch",
            patch_path=patch,
            env={},
        ),
        commands={
            "e2e": [sys.executable, "-c", "print('e2e')"],
            "build": [sys.executable, "-c", "print('build')"],
        },
    )

    spec = runner_spec_from_domain(task, model_spec, tmp_path / "artifacts")

    assert spec.base_commit == commit
    assert [step.name for step in spec.steps] == ["build", "e2e"]
    assert spec.steps[0].timeout_seconds == 99
    assert spec.steps[1].timeout_seconds == 33


def test_projection_prefers_typed_stage_commands_and_prepared_binary(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    binary = prepared / "llama-bench"
    binary.write_bytes(b"binary")
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    task = SimpleNamespace(
        runtime=SimpleNamespace(
            repo_path=repo,
            base_commit=commit,
            prepared_binary_path=binary,
            prepared_binary_sha256=binary_sha,
        ),
        budgets=SimpleNamespace(build_timeout_seconds=99),
        benchmark=SimpleNamespace(timeout_seconds=33),
        metadata={"prepared_source_path": str(prepared)},
    )
    model_spec = SimpleNamespace(
        id="typed",
        worktree_path=None,
        change=SimpleNamespace(
            kind="runtime_config",
            patch_path=None,
            patch_sha256=None,
            env={"MAPPING": "old"},
            unset_env=[],
        ),
        commands={"e2e": ["must-not-run"]},
        stage_commands={
            "e2e": SimpleNamespace(
                argv=[str(binary), "-n", "128,512"],
                cwd=Path("."),
                env={"HSA_VISIBLE_DEVICES": "0"},
                unset_env=["SMITHY_CONFIG"],
                timeout_seconds=77,
            )
        },
    )

    spec = runner_spec_from_domain(task, model_spec, tmp_path / "artifacts")

    assert spec.steps[0].argv[0] == str(binary)
    assert spec.steps[0].env == {"HSA_VISIBLE_DEVICES": "0"}
    assert spec.steps[0].unset_env == ("SMITHY_CONFIG",)
    assert spec.steps[0].timeout_seconds == 77
    assert spec.execution_root == str(prepared)
    assert spec.binary_path == str(binary)
    assert spec.expected_binary_sha256 == binary_sha
