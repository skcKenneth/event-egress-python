from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import ROOT

from egress_sim.analysis import aggregate_results, ensure_output_directories, run_policy_suite
from egress_sim.config import load_spec
from egress_sim.policies import DeterministicLoadBalancePolicy, RobustRollingHorizonPolicy


LEVELS = (0.35, 0.50, 0.68, 0.85)


def run(seed_count: int = 30, workers: int = 4) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, figures = ensure_output_directories(ROOT)
    frames = []
    for level in LEVELS:
        frame = run_policy_suite(
            spec,
            range(10_000, 10_000 + seed_count),
            policy_classes=(DeterministicLoadBalancePolicy, RobustRollingHorizonPolicy),
            compliance_mean=level,
            workers=workers,
        )
        frame["compliance_mean"] = level
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    summary = aggregate_results(raw, ["compliance_mean", "policy"])
    raw.to_csv(tables / "compliance_sensitivity_runs.csv", index=False)
    summary.to_csv(tables / "compliance_sensitivity_summary.csv", index=False)
    from egress_sim.plotting import plot_sensitivity

    plot_sensitivity(
        summary,
        "compliance_mean",
        "Actual mean compliance",
        figures / "compliance_sensitivity.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.seeds, args.workers)
