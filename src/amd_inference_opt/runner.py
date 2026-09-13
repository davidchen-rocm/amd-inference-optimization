"""Isolated git-worktree experiment execution.

The runner executes a declared experiment.  It does not decide whether the
experiment is good; ACCEPT/REJECT/INCONCLUSIVE belongs to the Gate engine.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .command import CommandResult, CommandRunner, StageCommand


class ExperimentRunnerError(RuntimeError):
    """An invalid experiment specification or unsafe path was supplied."""


@dataclass(frozen=True)
class ChangeSet:
    kind: Literal["source_patch", "runtime_config"]
    patch_path: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    unset_environment: tuple[str, ...] = ()
    expected_patch_sha256: str | None = None
    allowed_patch_files: tuple[str, ...] = ()
    candidate_model_path: str | None = None
    candidate_model_sha256: str | None = None
    runtime_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "source_patch":
            if not self.patch_path:
                raise ExperimentRunnerError("source_patch requires patch_path")
        elif self.kind == "runtime_config":
            if self.patch_path is not None:
                raise ExperimentRunnerError("runtime_config cannot include patch_path")
            if (
                not self.environment
                and not self.unset_environment
                and self.candidate_model_path is None
                and not self.runtime_args
            ):
                raise ExperimentRunnerError(
                    "runtime_config requires an environment or candidate model change"
                )
        else:
            raise ExperimentRunnerError(f"unsupported change kind: {self.kind}")
        invalid = [
            key
            for key in self.unset_environment
            if not key or "=" in key or "\x00" in key
        ]
        if invalid:
            raise ExperimentRunnerError("invalid environment name in unset_environment")
        overlap = set(self.environment) & set(self.unset_environment)
        if overlap:
            raise ExperimentRunnerError(
                "environment values cannot be both set and unset: "
                + ", ".join(sorted(overlap))
            )
        if self.expected_patch_sha256 is not None and (
            len(self.expected_patch_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.expected_patch_sha256)
        ):
            raise ExperimentRunnerError("expected_patch_sha256 must be a lowercase SHA-256")
        if any(
            Path(path).is_absolute() or ".." in Path(path).parts
            for path in self.allowed_patch_files
        ):
            raise ExperimentRunnerError("allowed_patch_files must be repository-relative paths")
        if any(not value or "\x00" in value for value in self.runtime_args):
            raise ExperimentRunnerError("runtime_args must be non-empty NUL-free values")


ExperimentStep = StageCommand


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    source_repo: str
    base_commit: str
    worktree_path: str
    artifact_dir: str
    change: ChangeSet
    steps: tuple[ExperimentStep, ...]
    base_environment: Mapping[str, str] = field(default_factory=dict)
    base_unset_environment: tuple[str, ...] = ()
    execution_root: str | None = None
    binary_path: str | None = None
    expected_binary_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_id or self.experiment_id in {".", ".."}:
            raise ExperimentRunnerError("experiment_id must not be empty")
        names = [step.name for step in self.steps]
        if len(names) != len(set(names)):
            raise ExperimentRunnerError("experiment step names must be unique")
        if self.change.kind == "source_patch" and "build" not in names:
            raise ExperimentRunnerError("source_patch experiments require a build step")
        if self.change.kind == "runtime_config" and "build" in names:
            raise ExperimentRunnerError("runtime_config experiments must reuse the existing binary")
        if self.expected_binary_sha256 is not None and (
            len(self.expected_binary_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.expected_binary_sha256)
        ):
            raise ExperimentRunnerError("expected_binary_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True)
class PatchApplication:
    kind: str
    patch_path: str | None
    patch_sha256: str | None
    expected_patch_sha256: str | None
    changed_files: tuple[str, ...]
    verification_error: str | None
    check: CommandResult | None
    apply: CommandResult | None

    @property
    def succeeded(self) -> bool:
        if self.kind == "runtime_config":
            return True
        return bool(
            self.verification_error is None
            and self.check
            and self.check.succeeded
            and self.apply
            and self.apply.succeeded
        )


@dataclass(frozen=True)
class BinaryIdentity:
    path: str
    sha256: str
    expected_sha256: str | None
    expected_sha256_matches: bool | None


@dataclass(frozen=True)
class ExperimentRunResult:
    experiment_id: str
    worktree_path: str
    worktree_create: CommandResult | None
    execution_root: str
    change: PatchApplication | None
    binary: BinaryIdentity | None
    step_results: Mapping[str, CommandResult]
    build_status: Literal["built", "reused", "failed", "not_run"]
    failure_stage: str | None

    @property
    def succeeded(self) -> bool:
        return self.failure_stage is None and all(
            result.succeeded for result in self.step_results.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _get(raw: Any, name: str, default: Any = None) -> Any:
    return raw.get(name, default) if isinstance(raw, Mapping) else getattr(raw, name, default)


def _stage_from_domain(
    name: str,
    raw: Any,
    *,
    default_timeout: float,
) -> ExperimentStep:
    if isinstance(raw, (list, tuple)):
        return ExperimentStep(name=name, argv=tuple(raw), timeout_seconds=default_timeout)
    argv = _get(raw, "argv")
    if argv is None:
        raise ExperimentRunnerError(f"typed command {name!r} is missing argv")
    return ExperimentStep(
        name=name,
        argv=tuple(argv),
        cwd=str(_get(raw, "cwd", ".")),
        env=dict(_get(raw, "env", {})),
        unset_env=tuple(_get(raw, "unset_env", _get(raw, "unset_environment", ()))),
        timeout_seconds=float(_get(raw, "timeout_seconds", default_timeout)),
    )


def _infer_binary_path(steps: Sequence[ExperimentStep]) -> str | None:
    for preferred in ("e2e", "microbench", "smoke"):
        for step in steps:
            if step.name == preferred and Path(step.argv[0]).is_absolute():
                return str(Path(step.argv[0]).resolve())
    return None


def runner_spec_from_domain(
    task: Any,
    model_spec: Any,
    artifact_dir: str | Path,
) -> ExperimentSpec:
    """Create an execution plan from the public task/experiment Pydantic models.

    The domain model owns persisted intent; this projection supplies runner-only
    locations and timeouts. Command names remain explicit workflow stages.
    """

    artifacts = Path(artifact_dir).resolve()
    configured_worktree = getattr(model_spec, "worktree_path", None)
    worktree = (
        Path(configured_worktree).resolve()
        if configured_worktree is not None
        else artifacts.parent / f"{model_spec.id}-worktree"
    )
    change = model_spec.change
    selected_change = ChangeSet(
        kind=str(change.kind),
        patch_path=str(Path(change.patch_path).resolve()) if change.patch_path else None,
        environment=dict(change.env),
        unset_environment=tuple(getattr(change, "unset_env", ())),
        expected_patch_sha256=getattr(change, "patch_sha256", None),
        allowed_patch_files=tuple(getattr(change, "allowed_patch_files", ())),
        candidate_model_path=(
            str(Path(change.candidate_model_path).resolve())
            if getattr(change, "candidate_model_path", None)
            else None
        ),
        candidate_model_sha256=getattr(change, "candidate_model_sha256", None),
        runtime_args=tuple(getattr(change, "runtime_args", ())),
    )
    typed_commands = getattr(model_spec, "stage_commands", {}) or {}
    raw_commands = typed_commands or model_spec.commands
    preferred_order = ("configure", "build", "smoke", "microbench", "e2e", "quality")
    command_names = list(raw_commands)
    ordered_names = [name for name in preferred_order if name in raw_commands]
    ordered_names.extend(name for name in command_names if name not in ordered_names)
    build_timeout = float(task.budgets.build_timeout_seconds)
    benchmark_timeout = float(task.benchmark.timeout_seconds)
    steps = tuple(
        _stage_from_domain(
            name,
            raw_commands[name],
            default_timeout=(
                build_timeout
                if name in {"configure", "build"}
                else 14400.0
                if name == "quality"
                else benchmark_timeout
            ),
        )
        for name in ordered_names
    )
    metadata = getattr(model_spec, "metadata", {}) or {}
    execution_root = getattr(model_spec, "execution_root", None) or _get(
        metadata, "execution_root"
    )
    task_metadata = getattr(task, "metadata", {}) or {}
    if execution_root is None:
        execution_root = _get(task_metadata, "prepared_source_path")
    binary_path = getattr(model_spec, "binary_path", None) or _get(metadata, "binary_path")
    if binary_path is None:
        binary_path = getattr(task.runtime, "prepared_binary_path", None)
    if binary_path is None:
        binary_path = _infer_binary_path(steps)
    expected_binary_sha256 = getattr(model_spec, "binary_sha256", None) or _get(
        metadata, "binary_sha256"
    )
    if expected_binary_sha256 is None:
        expected_binary_sha256 = getattr(task.runtime, "prepared_binary_sha256", None)
    return ExperimentSpec(
        experiment_id=model_spec.id,
        source_repo=str(Path(task.runtime.repo_path).resolve()),
        base_commit=task.runtime.base_commit,
        worktree_path=str(worktree),
        artifact_dir=str(artifacts),
        change=selected_change,
        steps=steps,
        base_environment=dict(getattr(model_spec, "base_environment", {})),
        base_unset_environment=tuple(getattr(model_spec, "base_unset_environment", ())),
        execution_root=str(Path(execution_root).resolve()) if execution_root else None,
        binary_path=str(Path(binary_path).resolve()) if binary_path else None,
        expected_binary_sha256=expected_binary_sha256,
    )


class ExperimentRunner:
    """Create a detached worktree, apply one change, and execute declared steps."""

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self.commands = command_runner or CommandRunner()

    def create_worktree(
        self,
        source_repo: str | Path,
        worktree_path: str | Path,
        base_commit: str,
        *,
        artifact_dir: str | Path | None = None,
    ) -> CommandResult:
        repo = Path(source_repo).resolve()
        worktree = Path(worktree_path).resolve()
        if not repo.is_dir() or not (repo / ".git").exists():
            raise ExperimentRunnerError(f"source_repo is not a git repository: {repo}")
        if not base_commit or base_commit.startswith("-"):
            raise ExperimentRunnerError("base_commit must be a commit identifier")
        if worktree == repo or _is_relative_to(worktree, repo):
            raise ExperimentRunnerError("worktree must be outside source_repo")
        if worktree.exists():
            raise ExperimentRunnerError(f"worktree target already exists: {worktree}")

        verify = self.commands.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", f"{base_commit}^{{commit}}"],
            cwd=repo,
        )
        if not verify.succeeded:
            raise ExperimentRunnerError(verify.stderr.strip() or f"unknown commit: {base_commit}")
        resolved_commit = verify.stdout.strip()
        worktree.parent.mkdir(parents=True, exist_ok=True)
        logs = Path(artifact_dir).resolve() if artifact_dir else None
        return self.commands.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(worktree), resolved_commit],
            cwd=repo,
            timeout_seconds=300,
            stdout_path=logs / "worktree.stdout" if logs else None,
            stderr_path=logs / "worktree.stderr" if logs else None,
        )

    def apply_change(
        self,
        worktree_path: str | Path,
        change: ChangeSet,
        *,
        artifact_dir: str | Path | None = None,
    ) -> PatchApplication:
        if change.kind == "runtime_config":
            return PatchApplication(
                kind=change.kind,
                patch_path=None,
                patch_sha256=None,
                expected_patch_sha256=None,
                changed_files=(),
                verification_error=None,
                check=None,
                apply=None,
            )

        worktree = Path(worktree_path).resolve()
        if not worktree.is_dir():
            raise ExperimentRunnerError(f"worktree does not exist: {worktree}")

        patch = Path(change.patch_path or "").resolve()
        if not patch.is_file():
            raise ExperimentRunnerError(f"patch file does not exist: {patch}")
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        logs = Path(artifact_dir).resolve() if artifact_dir else None
        verification_error: str | None = None
        if change.expected_patch_sha256 and digest != change.expected_patch_sha256:
            verification_error = (
                f"patch SHA-256 mismatch: expected {change.expected_patch_sha256}, got {digest}"
            )
        numstat = self.commands.run(
            ["git", "-C", str(worktree), "apply", "--numstat", "--", str(patch)],
            cwd=worktree,
            timeout_seconds=120,
            stdout_path=logs / "patch-numstat.stdout" if logs else None,
            stderr_path=logs / "patch-numstat.stderr" if logs else None,
        )
        changed_files = tuple(
            line.split("\t", 2)[-1]
            for line in numstat.stdout.splitlines()
            if line.strip() and "\t" in line
        )
        if not numstat.succeeded and verification_error is None:
            verification_error = numstat.stderr.strip() or "could not enumerate patch contents"
        if change.allowed_patch_files:
            unexpected = sorted(set(changed_files) - set(change.allowed_patch_files))
            missing = sorted(set(change.allowed_patch_files) - set(changed_files))
            if (unexpected or missing) and verification_error is None:
                verification_error = (
                    f"patch file set mismatch; unexpected={unexpected}, missing={missing}"
                )
        check = self.commands.run(
            ["git", "-C", str(worktree), "apply", "--check", "--", str(patch)],
            cwd=worktree,
            timeout_seconds=120,
            stdout_path=logs / "patch-check.stdout" if logs else None,
            stderr_path=logs / "patch-check.stderr" if logs else None,
        )
        applied: CommandResult | None = None
        if check.succeeded and verification_error is None:
            applied = self.commands.run(
                ["git", "-C", str(worktree), "apply", "--", str(patch)],
                cwd=worktree,
                timeout_seconds=120,
                stdout_path=logs / "patch-apply.stdout" if logs else None,
                stderr_path=logs / "patch-apply.stderr" if logs else None,
            )
        return PatchApplication(
            kind=change.kind,
            patch_path=str(patch),
            patch_sha256=digest,
            expected_patch_sha256=change.expected_patch_sha256,
            changed_files=changed_files,
            verification_error=verification_error,
            check=check,
            apply=applied,
        )

    def execute(self, spec: ExperimentSpec) -> ExperimentRunResult:
        repo = Path(spec.source_repo).resolve()
        worktree = Path(spec.worktree_path).resolve()
        artifacts = Path(spec.artifact_dir).resolve()
        artifacts.mkdir(parents=True, exist_ok=True)

        worktree_create: CommandResult | None = None
        if spec.change.kind == "source_patch":
            execution_root = worktree
            worktree_create = self.create_worktree(
                repo,
                worktree,
                spec.base_commit,
                artifact_dir=artifacts,
            )
            if not worktree_create.succeeded:
                result = ExperimentRunResult(
                    experiment_id=spec.experiment_id,
                    worktree_path=str(worktree),
                    execution_root=str(execution_root),
                    worktree_create=worktree_create,
                    change=None,
                    binary=None,
                    step_results={},
                    build_status="not_run",
                    failure_stage="create_worktree",
                )
                self._write_manifest(artifacts, result)
                return result
        else:
            # Runtime coordinates reuse a prepared binary directly. Creating a
            # detached source worktree here would imply a source mutation that did
            # not happen and waste substantial disk space.
            execution_root = Path(spec.execution_root or repo).resolve()
            if not execution_root.is_dir():
                raise ExperimentRunnerError(
                    f"runtime_config execution_root is not a directory: {execution_root}"
                )

        change_result = self.apply_change(execution_root, spec.change, artifact_dir=artifacts)
        if not change_result.succeeded:
            result = ExperimentRunResult(
                experiment_id=spec.experiment_id,
                worktree_path=str(worktree),
                execution_root=str(execution_root),
                worktree_create=worktree_create,
                change=change_result,
                binary=None,
                step_results={},
                build_status="not_run",
                failure_stage="apply_patch",
            )
            self._write_manifest(artifacts, result)
            return result

        binary: BinaryIdentity | None = None
        binary_path = spec.binary_path or _infer_binary_path(spec.steps)
        if spec.change.kind == "runtime_config" and binary_path is None:
            raise ExperimentRunnerError(
                "runtime_config requires binary_path or an inferable argv[0]"
            )
        if binary_path is not None and spec.change.kind == "runtime_config":
            binary = self._binary_identity(binary_path, spec.expected_binary_sha256)
            if binary.expected_sha256_matches is False:
                result = ExperimentRunResult(
                    experiment_id=spec.experiment_id,
                    worktree_path=str(worktree),
                    execution_root=str(execution_root),
                    worktree_create=worktree_create,
                    change=change_result,
                    binary=binary,
                    step_results={},
                    build_status="reused",
                    failure_stage="binary_identity",
                )
                self._write_manifest(artifacts, result)
                return result

        results: dict[str, CommandResult] = {}
        failure_stage: str | None = None
        build_reused = spec.change.kind == "runtime_config"
        build_status: Literal["built", "reused", "failed", "not_run"] = (
            "reused" if build_reused else "not_run"
        )
        for step in spec.steps:
            step_cwd = step.resolved_cwd(execution_root)
            if not step_cwd.is_dir():
                raise ExperimentRunnerError(
                    f"step {step.name!r} cwd does not exist: {step_cwd}"
                )
            if (
                spec.change.kind == "source_patch"
                and step.name != "quality"
                and not _is_relative_to(step_cwd, execution_root)
            ):
                raise ExperimentRunnerError(
                    f"source-patch step {step.name!r} cwd must be inside its worktree"
                )
            step_env = dict(spec.base_environment)
            step_env.update(step.env)
            step_unset = tuple(dict.fromkeys((*spec.base_unset_environment, *step.unset_env)))
            if step.name in {"smoke", "microbench", "e2e", "profile"}:
                step_env.update(spec.change.environment)
                step_unset = tuple(
                    dict.fromkeys((*step_unset, *spec.change.unset_environment))
                )
            # A narrower stage or candidate coordinate may explicitly set a
            # variable that the clean base coordinate unsets.
            step_unset = tuple(name for name in step_unset if name not in step_env)
            command_result = self.commands.run(
                step.argv,
                cwd=step_cwd,
                env=step_env,
                unset_env=step_unset,
                timeout_seconds=step.timeout_seconds,
                stdout_path=artifacts / f"{step.name}.stdout",
                stderr_path=artifacts / f"{step.name}.stderr",
            )
            results[step.name] = command_result
            if step.name == "build":
                build_status = "built" if command_result.succeeded else "failed"
            if not command_result.succeeded:
                failure_stage = step.name
                break

        if binary_path is not None and (
            spec.change.kind == "source_patch" or binary is None
        ):
            binary = self._binary_identity(binary_path, spec.expected_binary_sha256)
            if binary.expected_sha256_matches is False and failure_stage is None:
                failure_stage = "binary_identity"

        result = ExperimentRunResult(
            experiment_id=spec.experiment_id,
            worktree_path=str(worktree),
            execution_root=str(execution_root),
            worktree_create=worktree_create,
            change=change_result,
            binary=binary,
            step_results=results,
            build_status=build_status,
            failure_stage=failure_stage,
        )
        self._write_manifest(artifacts, result)
        return result

    @staticmethod
    def _binary_identity(
        binary_path: str | Path,
        expected_sha256: str | None,
    ) -> BinaryIdentity:
        path = Path(binary_path).resolve()
        if not path.is_file():
            raise ExperimentRunnerError(f"declared binary does not exist: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return BinaryIdentity(
            path=str(path),
            sha256=digest,
            expected_sha256=expected_sha256,
            expected_sha256_matches=(digest == expected_sha256 if expected_sha256 else None),
        )

    @staticmethod
    def _write_manifest(artifact_dir: Path, result: ExperimentRunResult) -> None:
        destination = artifact_dir / "runner-result.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
