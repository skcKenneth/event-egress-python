from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import ROOT

from egress_sim.analysis import aggregate_results, ensure_output_directories, run_policy_suite
from egress_sim.config import load_spec
from egress_sim.policies import DeterministicLoadBalancePolicy, RobustRollingHorizonPolicy


LEVELS = (0.0, 0.15, 0.30, 0.45)


def run(seed_count: int = 30, workers: int = 4) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, figures = ensure_output_directories(ROOT)
    frames = []
    for level in LEVELS:
        frame = run_policy_suite(
            spec,
            range(10_000, 10_000 + seed_count),
            policy_classes=(DeterministicLoadBalancePolicy, RobustRollingHorizonPolicy),
            capacity_log_sigma=level,
            workers=workers,
        )
        frame["capacity_log_sigma"] = level
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    summary = aggregate_results(raw, ["capacity_log_sigma", "policy"])
    raw.to_csv(tables / "capacity_stress_runs.csv", index=False)
    summary.to_csv(tables / "capacity_stress_summary.csv", index=False)
    from egress_sim.plotting import plot_sensitivity

    plot_sensitivity(
        summary,
        "capacity_log_sigma",
        "Capacity log-standard deviation",
        figures / "capacity_stress.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.seeds, args.workers)
