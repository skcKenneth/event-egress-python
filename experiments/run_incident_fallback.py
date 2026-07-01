from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import ROOT
from _stats import paired_effects

from egress_sim.analysis import PolicyVariant, ensure_output_directories, run_policy_variants
from egress_sim.config import load_spec


def run(seed_count: int = 50, workers: int = 4) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    variants = (
        PolicyVariant("deterministic_load_balance", "deterministic_load_balance", {}),
        PolicyVariant("robust_rolling_horizon", "robust_rolling_horizon", {}),
        PolicyVariant("change_triggered_fallback", "change_triggered_fallback", {}),
    )
    conditions = (
        ("stationary", {}),
        (
            "temporary_primary_exit_capacity_55pct",
            {
                "temporary_exit_index": 0,
                "temporary_capacity_factor": 0.55,
                "temporary_start_seconds": 240.0,
                "temporary_end_seconds": 600.0,
            },
        ),
    )
    frames = []
    effects = []
    for condition, simulation_kwargs in conditions:
        frame = run_policy_variants(
            spec,
            range(30_000, 30_000 + seed_count),
            variants,
            capacity_log_sigma=0.45,
            workers=workers,
            simulation_kwargs=simulation_kwargs,
        )
        frame["condition"] = condition
        frames.append(frame)
        for proposed, baseline in (
            ("robust_rolling_horizon", "deterministic_load_balance"),
            ("change_triggered_fallback", "deterministic_load_balance"),
            ("change_triggered_fallback", "robust_rolling_horizon"),
        ):
            effects.append(
                {
                    "condition": condition,
                    **paired_effects(
                        frame,
                        proposed,
                        baseline,
                        label=f"incident-{condition}-{proposed}-vs-{baseline}",
                    ),
                }
            )
    pd.concat(frames, ignore_index=True).to_csv(
        tables / "incident_fallback_runs.csv", index=False
    )
    pd.DataFrame(effects).to_csv(
        tables / "incident_fallback_effects.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.seeds, args.workers)
