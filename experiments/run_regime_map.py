from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import ROOT
from _stats import paired_effects

from egress_sim.analysis import ensure_output_directories, run_policy_suite
from egress_sim.config import load_spec
from egress_sim.policies import DeterministicLoadBalancePolicy, RobustRollingHorizonPolicy


CAPACITY_LEVELS = (0.0, 0.15, 0.30, 0.45)
COMPLIANCE_LEVELS = (0.35, 0.50, 0.68, 0.85)


def run(seed_count: int = 20, workers: int = 4) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    frames = []
    for sigma in CAPACITY_LEVELS:
        for compliance in COMPLIANCE_LEVELS:
            frame = run_policy_suite(
                spec,
                range(10_000, 10_000 + seed_count),
                policy_classes=(DeterministicLoadBalancePolicy, RobustRollingHorizonPolicy),
                capacity_log_sigma=sigma,
                compliance_mean=compliance,
                workers=workers,
            )
            frame["capacity_log_sigma"] = sigma
            frame["compliance_mean"] = compliance
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(tables / "regime_map_runs.csv", index=False)

    rows = []
    for (sigma, compliance), subset in raw.groupby(
        ["capacity_log_sigma", "compliance_mean"]
    ):
        reactive_clearance = subset.loc[
            subset.policy == "deterministic_load_balance", "clearance_time_seconds"
        ]
        robust_clearance = subset.loc[
            subset.policy == "robust_rolling_horizon", "clearance_time_seconds"
        ]
        reactive_density = subset.loc[
            subset.policy == "deterministic_load_balance", "peak_density_per_m2"
        ]
        robust_density = subset.loc[
            subset.policy == "robust_rolling_horizon", "peak_density_per_m2"
        ]
        reactive_p90 = reactive_clearance.quantile(0.90)
        robust_p90 = robust_clearance.quantile(0.90)
        effects = paired_effects(
            subset,
            "robust_rolling_horizon",
            "deterministic_load_balance",
            label=f"regime-{sigma}-{compliance}",
        )
        rows.append(
            {
                "capacity_log_sigma": sigma,
                "compliance_mean": compliance,
                "runs": seed_count,
                "reactive_p90_s": reactive_p90,
                "robust_p90_s": robust_p90,
                "p90_improvement_percent": 100.0
                * (reactive_p90 - robust_p90)
                / reactive_p90,
                "reactive_density_mean": reactive_density.mean(),
                "robust_density_mean": robust_density.mean(),
                "density_improvement_percent": 100.0
                * (reactive_density.mean() - robust_density.mean())
                / reactive_density.mean(),
                **effects,
            }
        )
    pd.DataFrame(rows).to_csv(tables / "regime_map_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.seeds, args.workers)
