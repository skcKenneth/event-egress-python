from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import ROOT
from _stats import paired_effects

from egress_sim.analysis import PolicyVariant, ensure_output_directories, run_policy_variants
from egress_sim.config import load_spec


def run(seed_count: int = 20, workers: int = 8) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    variants = (
        PolicyVariant("robust_rolling_horizon", "robust_rolling_horizon", {}),
        PolicyVariant(
            "change_triggered_fallback",
            "fallback_t10_h2",
            {"drop_threshold": 0.10, "hold_updates": 2},
        ),
        PolicyVariant(
            "change_triggered_fallback",
            "fallback_t15_h2",
            {"drop_threshold": 0.15, "hold_updates": 2},
        ),
        PolicyVariant(
            "change_triggered_fallback",
            "fallback_t15_h3",
            {"drop_threshold": 0.15, "hold_updates": 3},
        ),
        PolicyVariant(
            "change_triggered_fallback",
            "fallback_t20_h2",
            {"drop_threshold": 0.20, "hold_updates": 2},
        ),
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
            range(35_000, 35_000 + seed_count),
            variants,
            capacity_log_sigma=0.45,
            workers=workers,
            simulation_kwargs=simulation_kwargs,
        )
        frame["condition"] = condition
        frames.append(frame)
        for proposed in (variant.name for variant in variants[1:]):
            effects.append(
                {
                    "condition": condition,
                    **paired_effects(
                        frame,
                        proposed,
                        "robust_rolling_horizon",
                        label=f"fallback-dev-{condition}-{proposed}",
                    ),
                }
            )
    pd.concat(frames, ignore_index=True).to_csv(
        tables / "fallback_sensitivity_runs.csv", index=False
    )
    pd.DataFrame(effects).to_csv(
        tables / "fallback_sensitivity_effects.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.seeds, args.workers)
