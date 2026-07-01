from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKERS = min(16, max(1, (os.cpu_count() or 8) // 2))


def run_command(parts: list[str]) -> None:
    subprocess.run(parts, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the event-egress simulator and experiment suite. "
            "Defaults to the full experiment suite when no command is given."
        )
    )
    subparsers = parser.add_subparsers(dest="command")
    parser.set_defaults(command="full")

    subparsers.add_parser("verify", help="Run tests and a smoke simulation.")

    baseline = subparsers.add_parser("baseline", help="Run the baseline comparison.")
    baseline.add_argument("--seeds", type=int, default=20)
    baseline.add_argument("--workers", type=int, default=1)

    subparsers.add_parser("full", help="Run the full experiment suite.")

    subparsers.add_parser(
        "figures",
        help=(
            "Validate the frozen result tables and render the IEEE-style figures."
        ),
    )
    subparsers.add_parser(
        "check-figures",
        help="Check that IEEE figure tables match the frozen reference protocol.",
    )

    figure_pipeline = subparsers.add_parser(
        "figure-pipeline",
        help="Generate required tables and render the IEEE-style figures.",
    )
    figure_pipeline.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    runtime = subparsers.add_parser("runtime", help="Run the runtime benchmark.")
    runtime.add_argument("--repeats", type=int, default=25)
    runtime.add_argument("--seed", type=int, default=20_260_622)

    args = parser.parse_args()

    if args.command == "verify":
        run_command([sys.executable, str(ROOT / "scripts" / "verify.py")])
    elif args.command == "baseline":
        run_command(
            [
                sys.executable,
                str(ROOT / "experiments" / "run_baselines.py"),
                "--seeds",
                str(args.seeds),
                "--workers",
                str(args.workers),
            ]
        )
    elif args.command == "full":
        run_command([sys.executable, str(ROOT / "experiments" / "run_all.py")])
    elif args.command == "figures":
        run_command([sys.executable, str(ROOT / "experiments" / "render_ieee_figures.py")])
    elif args.command == "check-figures":
        run_command([sys.executable, str(ROOT / "scripts" / "check_ieee_figure_data.py")])
    elif args.command == "figure-pipeline":
        run_command(
            [
                sys.executable,
                str(ROOT / "experiments" / "run_ieee_figure_pipeline.py"),
                "--workers",
                str(args.workers),
            ]
        )
    elif args.command == "runtime":
        run_command(
            [
                sys.executable,
                str(ROOT / "experiments" / "run_runtime_benchmark.py"),
                "--repeats",
                str(args.repeats),
                "--seed",
                str(args.seed),
            ]
        )


if __name__ == "__main__":
    main()
