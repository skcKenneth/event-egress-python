from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _bootstrap import ROOT
from _stats import paired_effects

from egress_sim.analysis import PolicyVariant, ensure_output_directories, run_policy_variants
from egress_sim.config import load_spec


def run(seed_count: int = 20, workers: int = 4) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    travel_multiplier = np.ones_like(spec.free_flow_time)
    travel_multiplier[:, 0] = 0.80
    conditions = [
        ("matched", {}, {}),
        ("travel_time_primary_route_minus_20pct", {"travel_time_multiplier": travel_multiplier}, {}),
        ("assumed_residual_sigma_008", {"assumed_residual_sigma": 0.08}, {}),
        ("assumed_compliance_085", {"assumed_compliance_mean": 0.85}, {}),
        (
            "temporary_primary_exit_capacity_55pct",
            {},
            {
                "temporary_exit_index": 0,
                "temporary_capacity_factor": 0.55,
                "temporary_start_seconds": 240.0,
                "temporary_end_seconds": 600.0,
            },
        ),
    ]
    frames = []
    effect_rows = []
    for condition, robust_kwargs, simulation_kwargs in conditions:
        variants = (
            PolicyVariant(
                "deterministic_load_balance", "deterministic_load_balance", {}
            ),
            PolicyVariant(
                "robust_rolling_horizon", "robust_rolling_horizon", robust_kwargs
            ),
        )
        frame = run_policy_variants(
            spec,
            range(10_000, 10_000 + seed_count),
            variants,
            capacity_log_sigma=0.45,
            workers=workers,
            simulation_kwargs=simulation_kwargs,
        )
        frame["condition"] = condition
        frames.append(frame)
        effect_rows.append(
            {
                "condition": condition,
                **paired_effects(
                    frame,
                    "robust_rolling_horizon",
                    "deterministic_load_balance",
                    label=f"mismatch-{condition}",
                ),
            }
        )
    pd.concat(frames, ignore_index=True).to_csv(
        tables / "model_mismatch_runs.csv", index=False
    )
    pd.DataFrame(effect_rows).to_csv(tables / "model_mismatch_effects.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.seeds, args.workers)
