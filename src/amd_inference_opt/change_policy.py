"""Deterministic enforcement of task-owned optimization boundaries."""

from __future__ import annotations

from .models import ChangeKind, ExperimentSpec, OptimizationTask


class ChangePolicyError(ValueError):
    """An experiment attempts to modify a coordinate the task locked."""


def _runtime_flag_names(
    argv: list[str],
    *,
    arity_by_flag: dict[str, int] | None = None,
) -> tuple[str, ...]:
    """Parse only task-declared options; arity defaults to one for compatibility."""

    arities = arity_by_flag or {}
    flags: list[str] = []
    index = 0
    while index < len(argv):
        flag = argv[index]
        if not flag.startswith("-") or flag in {"-", "--"} or "=" in flag:
            raise ChangePolicyError("runtime_args must use explicit option names")
        arity = arities.get(flag, 1)
        if arity not in {0, 1}:
            raise ChangePolicyError(f"unsupported runtime argument arity for {flag}")
        if arity == 1:
            if index + 1 >= len(argv):
                raise ChangePolicyError(f"runtime argument {flag} requires a value")
            value = argv[index + 1]
            if not value or "\x00" in value:
                raise ChangePolicyError(
                    "runtime_args values must be non-empty and NUL-free"
                )
            index += 1
        flags.append(flag)
        index += 1
    if len(flags) != len(set(flags)):
        raise ChangePolicyError("runtime_args cannot repeat an option")
    return tuple(flags)


def validate_experiment_change(
    task: OptimizationTask,
    spec: ExperimentSpec,
) -> None:
    """Reject scope expansion before a worktree, process, or artifact is created."""

    policy = task.change_policy
    change = spec.change
    if change.kind not in policy.allowed_change_kinds:
        raise ChangePolicyError(
            f"change kind {change.kind} is outside this task's allowed change kinds"
        )

    locked = set(policy.locked_coordinates)
    if change.candidate_model_path is not None:
        conflicts = locked & {"model_sha256", "model_path", "quantization"}
        if conflicts:
            raise ChangePolicyError(
                "candidate model changes locked coordinates: "
                + ", ".join(sorted(conflicts))
            )

    if change.kind == ChangeKind.SOURCE_PATCH and "binary_sha256" in locked:
        raise ChangePolicyError("source patches cannot preserve a locked binary_sha256")

    if (change.env or change.unset_env) and "runtime_environment" in locked:
        raise ChangePolicyError("runtime environment is locked for this task")

    flags = (
        _runtime_flag_names(
            change.runtime_args,
            arity_by_flag=policy.runtime_arg_arity,
        )
        if change.runtime_args
        else ()
    )
    allowed_flags = set(policy.allowed_runtime_args)
    unexpected = sorted(set(flags) - allowed_flags)
    if unexpected:
        raise ChangePolicyError(
            "runtime arguments are outside this task's allow-list: "
            + ", ".join(unexpected)
        )


__all__ = ["ChangePolicyError", "validate_experiment_change"]
