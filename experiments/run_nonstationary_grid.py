from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import ROOT
from _stats import paired_effects

from egress_sim.analysis import PolicyVariant, ensure_output_directories, run_policy_variants
from egress_sim.config import load_spec


CONDITIONS = (
    (
        "abrupt_early_long_55_both",
        {
            "temporary_exit_index": 0,
            "temporary_capacity_factor": 0.55,
            "temporary_start_seconds": 240.0,
            "temporary_end_seconds": 600.0,
            "temporary_scope": "both",
            "temporary_profile": "abrupt",
        },
    ),
    (
        "abrupt_late_short_55_both",
        {
            "temporary_exit_index": 0,
            "temporary_capacity_factor": 0.55,
            "temporary_start_seconds": 480.0,
            "temporary_end_seconds": 720.0,
            "temporary_scope": "both",
            "temporary_profile": "abrupt",
        },
    ),
    (
        "abrupt_early_long_70_both",
        {
            "temporary_exit_index": 0,
            "temporary_capacity_factor": 0.70,
            "temporary_start_seconds": 240.0,
            "temporary_end_seconds": 600.0,
            "temporary_scope": "both",
            "temporary_profile": "abrupt",
        },
    ),
    (
        "gradual_early_long_55_both",
        {
            "temporary_exit_index": 0,
            "temporary_capacity_factor": 0.55,
            "temporary_start_seconds": 240.0,
            "temporary_end_seconds": 600.0,
            "temporary_scope": "both",
            "temporary_profile": "gradual",
        },
    ),
    (
        "recovery_early_long_55_both",
        {
            "temporary_exit_index": 0,
            "temporary_capacity_factor": 0.55,
            "temporary_start_seconds": 240.0,
            "temporary_end_seconds": 600.0,
            "temporary_scope": "both",
            "temporary_profile": "recovery",
        },
    ),
    (
        "piecewise_early_long_55_both",
        {
            "temporary_exit_index": 0,
            "temporary_capacity_factor": 0.55,
            "temporary_start_seconds": 240.0,
            "temporary_end_seconds": 600.0,
            "temporary_scope": "both",
            "temporary_profile": "piecewise",
        },
    ),
    (
        "path_only_early_long_55",
        {
            "temporary_exit_index": 0,
            "temporary_capacity_factor": 0.55,
            "temporary_start_seconds": 240.0,
            "temporary_end_seconds": 600.0,
            "temporary_scope": "path_only",
            "temporary_profile": "abrupt",
        },
    ),
    (
        "exit_only_early_long_55",
        {
            "temporary_exit_index": 0,
            "temporary_capacity_factor": 0.55,
            "temporary_start_seconds": 240.0,
            "temporary_end_seconds": 600.0,
            "temporary_scope": "exit_only",
            "temporary_profile": "abrupt",
        },
    ),
)


def run(seed_count: int = 30, workers: int = 8) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    variants = (
        PolicyVariant("deterministic_load_balance", "deterministic_load_balance", {}),
        PolicyVariant("scenario_mean_only", "scenario_mean_only", {}),
        PolicyVariant("robust_rolling_horizon", "robust_rolling_horizon", {}),
        PolicyVariant("change_triggered_fallback", "change_triggered_fallback", {}),
    )
    frames = []
    effects = []
    for condition, simulation_kwargs in CONDITIONS:
        frame = run_policy_variants(
            spec,
            range(60_000, 60_000 + seed_count),
            variants,
            capacity_log_sigma=0.45,
            compliance_mean=0.68,
            workers=workers,
            simulation_kwargs=simulation_kwargs,
        )
        frame["condition"] = condition
        frames.append(frame)
        for proposed, baseline in (
            ("scenario_mean_only", "deterministic_load_balance"),
            ("robust_rolling_horizon", "deterministic_load_balance"),
            ("change_triggered_fallback", "deterministic_load_balance"),
            ("change_triggered_fallback", "robust_rolling_horizon"),
        ):
            effects.append(
                {
                    "condition": condition,
                    "proposed_policy": proposed,
                    "baseline_policy": baseline,
                    **paired_effects(
                        frame,
                        proposed,
                        baseline,
                        label=f"nonstationary-{condition}-{proposed}-vs-{baseline}",
                    ),
                }
            )
    pd.concat(frames, ignore_index=True).to_csv(
        tables / "nonstationary_grid_runs.csv", index=False
    )
    pd.DataFrame(effects).to_csv(tables / "nonstationary_grid_effects.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.seeds, args.workers)
