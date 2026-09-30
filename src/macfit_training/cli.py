"""Reusable command line entrypoints; GPU execution is explicit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import capabilities, validate_job_input


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MacFit bounded ROCm generation and LoRA SFT.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities")
    validate = commands.add_parser("validate")
    validate.add_argument("--kind", choices=("generation", "training"), required=True)
    validate.add_argument("--input", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--kind", choices=("generation", "training"), required=True)
    run.add_argument("--job-dir", type=Path, required=True)
    export = commands.add_parser(
        "export-merged", help="Merge a verified completed LoRA job offline on CPU."
    )
    export.add_argument("--job-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--base-model-dir", type=Path, help="Relocated copy of the pinned snapshot."
    )
    args = parser.parse_args(argv)
    if args.command == "capabilities":
        print(json.dumps(capabilities(), indent=2))
        return 0
    if args.command == "validate":
        from .worker import read_input

        print(json.dumps(validate_job_input(args.kind, read_input(args.input)), indent=2))
        return 0
    if args.command == "export-merged":
        from .export import export_merged

        manifest = export_merged(args.job_dir, args.output, base_model_dir=args.base_model_dir)
        print(json.dumps(manifest, indent=2))
        return 0
    from .worker import main as worker_main

    return worker_main(["--kind", args.kind, "--job-dir", str(args.job_dir)])


if __name__ == "__main__":
    raise SystemExit(main())
