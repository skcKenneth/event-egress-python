from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent
DEFAULT_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 1))


def run_command(parts: list[str]) -> None:
    print("+", " ".join(parts))
    subprocess.run(parts, cwd=ROOT, check=True)


def _experiment_command(script: str, seeds: int, workers: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments" / script),
        "--seeds",
        str(seeds),
        "--workers",
        str(workers),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the event-egress simulator and reproducible experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="Run unit tests and a smoke simulation.")

    baseline = subparsers.add_parser("baseline", help="Run a quick baseline comparison.")
    baseline.add_argument("--seeds", type=int, default=20)
    baseline.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    final = subparsers.add_parser(
        "final-validation",
        help="Run the confirmatory held-out center-only validation.",
    )
    final.add_argument("--seeds", type=int, default=200)
    final.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    estimator = subparsers.add_parser(
        "estimator",
        help="Run capacity-estimator convergence diagnostics.",
    )
    estimator.add_argument("--seeds", type=int, default=50)
    estimator.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    topology = subparsers.add_parser(
        "topology",
        help="Run the network-family generalization study.",
    )
    topology.add_argument("--seeds", type=int, default=50)
    topology.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    sensor = subparsers.add_parser(
        "sensor",
        help="Run sensor noise, delay, dropout, and bias diagnostics.",
    )
    sensor.add_argument("--seeds", type=int, default=30)
    sensor.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    subparsers.add_parser(
        "full",
        help="Run the broader exploratory suite with moderate seed counts.",
    )
    subparsers.add_parser(
        "figures",
        help="Validate frozen tables and render the three paper figures.",
    )
    subparsers.add_parser(
        "check-figures",
        help="Validate frozen table protocol and SHA-256 manifest.",
    )

    pipeline = subparsers.add_parser(
        "figure-pipeline",
        help="Regenerate the complete frozen paper experiment suite.",
    )
    pipeline.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    pipeline.add_argument(
        "--refresh-artifacts",
        action="store_true",
        help="Replace tracked paper_artifacts after successful validation.",
    )

    runtime = subparsers.add_parser("runtime", help="Run the runtime benchmark.")
    runtime.add_argument("--repeats", type=int, default=25)
    runtime.add_argument("--seed", type=int, default=20_260_622)

    diagnostic = subparsers.add_parser(
        "decision-diagnostics", help="Run the separate PIC2026 action diagnostics."
    )
    diagnostic.add_argument("--seed-start", type=int, default=110000)
    diagnostic.add_argument("--seeds", type=int, default=30)
    diagnostic.add_argument("--output", type=Path, default=ROOT / "outputs" / "pic2026_revision")

    comparison = subparsers.add_parser(
        "runtime-comparison", help="Time reactive and matched mean controllers on shared snapshots."
    )
    comparison.add_argument("--seed-start", type=int, default=110000)
    comparison.add_argument("--seeds", type=int, default=5)
    comparison.add_argument("--warmups", type=int, default=5)
    comparison.add_argument("--repetitions", type=int, default=30)
    comparison.add_argument("--output", type=Path, default=ROOT / "outputs" / "pic2026_revision")
    subparsers.add_parser("revision-artifacts", help="Render the separate revision paper artifacts.")
    subparsers.add_parser("check-revision", help="Check revision provenance and primary-seed replay.")

    args = parser.parse_args()

    if args.command == "verify":
        run_command([sys.executable, str(ROOT / "scripts" / "verify.py")])
    elif args.command == "baseline":
        run_command(_experiment_command("run_baselines.py", args.seeds, args.workers))
    elif args.command == "final-validation":
        run_command(
            _experiment_command(
                "run_information_value_validation.py",
                args.seeds,
                args.workers,
            )
        )
    elif args.command == "estimator":
        run_command(
            _experiment_command(
                "run_estimator_diagnostics.py",
                args.seeds,
                args.workers,
            )
        )
    elif args.command == "topology":
        run_command(
            _experiment_command(
                "run_topology_generalization.py",
                args.seeds,
                args.workers,
            )
        )
    elif args.command == "sensor":
        run_command(
            _experiment_command(
                "run_sensor_ablation.py",
                args.seeds,
                args.workers,
            )
        )
    elif args.command == "full":
        run_command([sys.executable, str(ROOT / "experiments" / "run_all.py")])
    elif args.command == "figures":
        run_command(
            [sys.executable, str(ROOT / "experiments" / "render_ieee_figures.py")]
        )
    elif args.command == "check-figures":
        run_command(
            [sys.executable, str(ROOT / "scripts" / "check_ieee_figure_data.py")]
        )
    elif args.command == "figure-pipeline":
        command = [
            sys.executable,
            str(ROOT / "experiments" / "run_ieee_figure_pipeline.py"),
            "--workers",
            str(args.workers),
        ]
        if args.refresh_artifacts:
            command.append("--refresh-artifacts")
        run_command(command)
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
    elif args.command == "decision-diagnostics":
        run_command([
            sys.executable, str(ROOT / "experiments" / "run_decision_diagnostics.py"),
            "--seed-start", str(args.seed_start), "--seeds", str(args.seeds),
            "--output", str(args.output),
        ])
    elif args.command == "runtime-comparison":
        run_command([
            sys.executable, str(ROOT / "experiments" / "run_runtime_comparison.py"),
            "--seed-start", str(args.seed_start), "--seeds", str(args.seeds),
            "--warmups", str(args.warmups), "--repetitions", str(args.repetitions),
            "--output", str(args.output),
        ])
    elif args.command == "revision-artifacts":
        run_command([sys.executable, str(ROOT / "experiments" / "render_revision_artifacts.py")])
    elif args.command == "check-revision":
        run_command([sys.executable, str(ROOT / "scripts" / "check_revision.py")])


if __name__ == "__main__":
    main()
