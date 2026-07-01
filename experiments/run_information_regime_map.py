from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import DEFAULT_WORKERS, ROOT
from _stats import paired_effects

from egress_sim.analysis import (
    PolicyVariant,
    ensure_output_directories,
    execute_policy_variant_tasks,
)
from egress_sim.config import load_spec


SEED_START = 96_000
CAPACITY_LEVELS = (0.0, 0.15, 0.30, 0.45)
COMPLIANCE_LEVELS = (0.35, 0.50, 0.68, 0.85)
POLICY_SEED_OFFSET = 1_000_003


def run(seed_count: int = 50, workers: int = DEFAULT_WORKERS) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    seeds = range(SEED_START, SEED_START + seed_count)
    variants = (
        PolicyVariant("unconditioned_mean_only", "unconditioned_mean_only", {}),
        PolicyVariant("scenario_mean_only", "scenario_mean_only", {}),
    )
    tasks = [
        (
            spec,
            int(seed),
            int(seed) + POLICY_SEED_OFFSET,
            variant,
            sigma,
            compliance,
            {},
            {},
        )
        for sigma in CAPACITY_LEVELS
        for compliance in COMPLIANCE_LEVELS
        for seed in seeds
        for variant in variants
    ]
    raw = execute_policy_variant_tasks(tasks, workers=workers)
    rows = []
    for (sigma, compliance), subset in raw.groupby(
        ["capacity_log_sigma", "compliance_mean"]
    ):
        rows.append(
            {
                "capacity_log_sigma": sigma,
                "compliance_mean": compliance,
                "runs": seed_count,
                **paired_effects(
                    subset,
                    "scenario_mean_only",
                    "unconditioned_mean_only",
                    label=f"info-regime-{sigma}-{compliance}",
                ),
            }
        )
    raw.to_csv(tables / "information_regime_map_runs.csv", index=False)
    pd.DataFrame(rows).to_csv(
        tables / "information_regime_map_summary.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    run(args.seeds, args.workers)
