from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import DEFAULT_WORKERS, ROOT
from _stats import paired_effects

from egress_sim.analysis import (
    PolicyVariant,
    aggregate_results,
    ensure_output_directories,
    run_policy_variants,
)
from egress_sim.config import load_spec


FINAL_SEED_START = 90_000


def run(seed_count: int = 200, workers: int = DEFAULT_WORKERS) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    seeds = range(FINAL_SEED_START, FINAL_SEED_START + seed_count)
    variants = (
        PolicyVariant("static_nearest", "static_nearest", {}),
        PolicyVariant("deterministic_load_balance", "deterministic_load_balance", {}),
        PolicyVariant("unconditioned_mean_only", "unconditioned_mean_only", {}),
        PolicyVariant("scenario_mean_only", "scenario_mean_only", {}),
        PolicyVariant("oracle_capacity_mean_only", "oracle_capacity_mean_only", {}),
        PolicyVariant(
            "robust_without_sensor_conditioning",
            "unconditioned_mean_cvar",
            {},
        ),
        PolicyVariant("robust_rolling_horizon", "conditioned_mean_cvar", {}),
    )
    raw = run_policy_variants(spec, seeds, variants, workers=workers)
    raw.to_csv(tables / "information_value_validation_runs.csv", index=False)
    aggregate_results(raw, ["policy"]).to_csv(
        tables / "information_value_validation_summary.csv", index=False
    )

    comparisons = (
        ("deterministic_load_balance", "static_nearest", "dynamic-vs-static"),
        (
            "unconditioned_mean_only",
            "deterministic_load_balance",
            "planning-vs-reactive",
        ),
        (
            "scenario_mean_only",
            "unconditioned_mean_only",
            "primary-sensing-value",
        ),
        (
            "oracle_capacity_mean_only",
            "scenario_mean_only",
            "oracle-residual-information",
        ),
        (
            "unconditioned_mean_cvar",
            "unconditioned_mean_only",
            "cvar-without-conditioning",
        ),
        (
            "conditioned_mean_cvar",
            "scenario_mean_only",
            "cvar-with-conditioning",
        ),
        (
            "conditioned_mean_cvar",
            "unconditioned_mean_cvar",
            "sensing-value-with-cvar",
        ),
        (
            "scenario_mean_only",
            "deterministic_load_balance",
            "secondary-system-comparison",
        ),
    )
    effects = []
    for proposed, baseline, label in comparisons:
        effects.append(
            {
                "comparison": label,
                "proposed_policy": proposed,
                "baseline_policy": baseline,
                "seed_start": FINAL_SEED_START,
                "seed_end": FINAL_SEED_START + seed_count - 1,
                **paired_effects(
                    raw,
                    proposed,
                    baseline,
                    label=f"information-final-{label}",
                ),
            }
        )
    pd.DataFrame(effects).to_csv(
        tables / "information_value_validation_effects.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    run(args.seeds, args.workers)
