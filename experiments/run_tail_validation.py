from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import ROOT
from _stats import paired_effects

from egress_sim.analysis import PolicyVariant, ensure_output_directories, run_policy_variants
from egress_sim.config import load_spec


def run(seed_count: int = 200, workers: int = 4) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    frames = []
    effects = []
    conditions = (
        (
            "default",
            0.22,
            0.68,
            {},
            (
                PolicyVariant("deterministic_load_balance", "deterministic_load_balance", {}),
                PolicyVariant("scenario_mean_only", "scenario_mean_only", {}),
                PolicyVariant("robust_rolling_horizon", "robust_rolling_horizon", {}),
            ),
        ),
        (
            "high_compliance_strong_uncertainty",
            0.45,
            0.85,
            {},
            (
                PolicyVariant("deterministic_load_balance", "deterministic_load_balance", {}),
                PolicyVariant("scenario_mean_only", "scenario_mean_only", {}),
                PolicyVariant("robust_rolling_horizon", "robust_rolling_horizon", {}),
            ),
        ),
        (
            "temporary_primary_exit_capacity_55pct",
            0.45,
            0.68,
            {
                "temporary_exit_index": 0,
                "temporary_capacity_factor": 0.55,
                "temporary_start_seconds": 240.0,
                "temporary_end_seconds": 600.0,
            },
            (
                PolicyVariant("deterministic_load_balance", "deterministic_load_balance", {}),
                PolicyVariant("robust_rolling_horizon", "robust_rolling_horizon", {}),
                PolicyVariant("change_triggered_fallback", "change_triggered_fallback", {}),
            ),
        ),
    )
    for condition, sigma, compliance, simulation_kwargs, variants in conditions:
        frame = run_policy_variants(
            spec,
            range(40_000, 40_000 + seed_count),
            variants,
            capacity_log_sigma=sigma,
            compliance_mean=compliance,
            workers=workers,
            simulation_kwargs=simulation_kwargs,
        )
        frame["condition"] = condition
        frames.append(frame)
        comparisons = (
            (
                (
                    ("change_triggered_fallback", "deterministic_load_balance"),
                    ("change_triggered_fallback", "robust_rolling_horizon"),
                    ("robust_rolling_horizon", "deterministic_load_balance"),
                )
                if condition == "temporary_primary_exit_capacity_55pct"
                else (
                    ("robust_rolling_horizon", "deterministic_load_balance"),
                    ("robust_rolling_horizon", "scenario_mean_only"),
                )
            )
        )
        for proposed, baseline in comparisons:
            effects.append(
                {
                    "condition": condition,
                    "proposed_policy": proposed,
                    "baseline_policy": baseline,
                    **paired_effects(
                        frame,
                        proposed,
                        baseline,
                        label=f"tail-{condition}-{proposed}-vs-{baseline}",
                    ),
                }
            )
    pd.concat(frames, ignore_index=True).to_csv(
        tables / "tail_validation_runs.csv", index=False
    )
    pd.DataFrame(effects).to_csv(
        tables / "tail_validation_effects.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.seeds, args.workers)
