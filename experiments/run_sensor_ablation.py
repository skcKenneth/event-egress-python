from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import ROOT
from _stats import paired_effects

from egress_sim.analysis import (
    PolicyVariant,
    aggregate_results,
    ensure_output_directories,
    run_policy_variants,
)
from egress_sim.config import load_spec


VARIANTS = (
    PolicyVariant(
        "robust_without_sensor_conditioning",
        "robust_without_sensor_conditioning",
        {},
    ),
    PolicyVariant("robust_rolling_horizon", "conditioned_clean", {}),
    PolicyVariant(
        "robust_rolling_horizon",
        "conditioned_noise_015",
        {"sensor_noise_sigma": 0.15},
    ),
    PolicyVariant(
        "robust_rolling_horizon",
        "conditioned_delay_60s",
        {"sensor_delay_steps": 6},
    ),
    PolicyVariant(
        "robust_rolling_horizon",
        "conditioned_dropout_20pct",
        {"sensor_dropout": 0.20},
    ),
    PolicyVariant(
        "robust_rolling_horizon",
        "conditioned_bias_plus_10pct",
        {"sensor_bias": 1.10},
    ),
)


def run(seed_count: int = 30, workers: int = 4) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, figures = ensure_output_directories(ROOT)
    raw = run_policy_variants(
        spec,
        range(10_000, 10_000 + seed_count),
        VARIANTS,
        capacity_log_sigma=0.45,
        workers=workers,
    )
    summary = aggregate_results(raw, ["policy"])
    raw.to_csv(tables / "sensor_ablation_runs.csv", index=False)
    summary.to_csv(tables / "sensor_ablation_summary.csv", index=False)
    from egress_sim.plotting import plot_ablation

    plot_ablation(summary, figures / "sensor_ablation.png")

    rows = []
    baseline = "robust_without_sensor_conditioning"
    for variant in VARIANTS[1:]:
        rows.append(
            {
                "condition": variant.name,
                **paired_effects(
                    raw,
                    variant.name,
                    baseline,
                    label=f"sensor-{variant.name}",
                ),
            }
        )
    pd.DataFrame(rows).to_csv(tables / "sensor_condition_effects.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.seeds, args.workers)
