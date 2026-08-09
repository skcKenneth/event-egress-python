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


SEED_START = 94_000
CAPACITY_SIGMAS = (0.0, 0.15, 0.22, 0.45)
POLICY_SEED_OFFSET = 1_000_003


def residual_modes(environmental_sigma: float) -> tuple[tuple[str, float], ...]:
    return (
        ("zero", 0.0),
        ("shrunk", max(0.05, 0.45 * environmental_sigma)),
        ("matched", environmental_sigma),
        ("overdispersed", max(0.05, 1.50 * environmental_sigma)),
    )


def run(seed_count: int = 50, workers: int = DEFAULT_WORKERS) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    seeds = range(SEED_START, SEED_START + seed_count)
    tasks = []
    for sigma in CAPACITY_SIGMAS:
        for seed in seeds:
            tasks.append(
                (
                    spec,
                    int(seed),
                    int(seed) + POLICY_SEED_OFFSET,
                    PolicyVariant(
                        "unconditioned_mean_only",
                        "unconditioned_mean_only",
                        {},
                    ),
                    sigma,
                    None,
                    {},
                    {},
                )
            )
            for mode, assumed_sigma in residual_modes(sigma):
                tasks.append(
                    (
                        spec,
                        int(seed),
                        int(seed) + POLICY_SEED_OFFSET,
                        PolicyVariant(
                            "conditioned_mean_equal_spread",
                            f"conditioned_mean_{mode}",
                            {"assumed_residual_sigma": assumed_sigma},
                        ),
                        sigma,
                        None,
                        {},
                        {"residual_mode": mode, "residual_sigma": assumed_sigma},
                    )
                )
    raw = execute_policy_variant_tasks(tasks, workers=workers)
    effects = []
    for sigma in CAPACITY_SIGMAS:
        subset = raw.loc[raw["capacity_log_sigma"] == sigma]
        for mode, assumed_sigma in residual_modes(sigma):
            proposed = f"conditioned_mean_{mode}"
            effects.append(
                {
                    "capacity_log_sigma": sigma,
                    "residual_mode": mode,
                    "residual_sigma": assumed_sigma,
                    "proposed_policy": proposed,
                    "baseline_policy": "unconditioned_mean_only",
                    **paired_effects(
                        subset,
                        proposed,
                        "unconditioned_mean_only",
                        label=f"residual-{sigma}-{mode}",
                    ),
                }
            )
    raw.to_csv(tables / "residual_spread_ablation_runs.csv", index=False)
    pd.DataFrame(effects).to_csv(
        tables / "residual_spread_ablation_effects.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    run(args.seeds, args.workers)
