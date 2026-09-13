"""Private module entry point for one durable quality execution attempt."""

from __future__ import annotations

import argparse

from .quality_execution import run_quality_worker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    return run_quality_worker(
        args.store,
        args.task,
        args.experiment,
        args.attempt,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
