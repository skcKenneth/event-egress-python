from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT

from egress_sim.analysis import aggregate_results, ensure_output_directories, run_policy_suite
from egress_sim.config import load_spec
from egress_sim.policies import (
    DeterministicLoadBalancePolicy,
    OracleCapacityMeanPolicy,
    OracleCapacityPolicy,
    RobustRollingHorizonPolicy,
    ScenarioMeanPolicy,
    StaticNearestPolicy,
    UnconditionedMeanPolicy,
    UnconditionedRobustPolicy,
)
from egress_sim.simulation import run_simulation


def run(seed_count: int = 50, workers: int = 4) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, figures = ensure_output_directories(ROOT)
    seeds = range(10_000, 10_000 + seed_count)
    raw = run_policy_suite(
        spec,
        seeds,
        policy_classes=(
            StaticNearestPolicy,
            DeterministicLoadBalancePolicy,
            UnconditionedMeanPolicy,
            ScenarioMeanPolicy,
            UnconditionedRobustPolicy,
            RobustRollingHorizonPolicy,
            OracleCapacityMeanPolicy,
            OracleCapacityPolicy,
        ),
        workers=workers,
    )
    summary = aggregate_results(raw, ["policy"])
    raw.to_csv(tables / "baseline_runs.csv", index=False)
    summary.to_csv(tables / "baseline_summary.csv", index=False)
    from egress_sim.plotting import plot_baseline_comparison, plot_trajectory

    plot_baseline_comparison(raw, figures / "baseline_comparison.png")

    histories = {}
    for policy_class in (
        StaticNearestPolicy,
        DeterministicLoadBalancePolicy,
        RobustRollingHorizonPolicy,
    ):
        result = run_simulation(spec, policy_class(), seed=10_000)
        histories[result.policy] = pd.DataFrame(result.history)
    plot_trajectory(histories, figures / "representative_trajectory.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.seeds, args.workers)
