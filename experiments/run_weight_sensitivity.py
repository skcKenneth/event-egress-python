from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from _bootstrap import DEFAULT_WORKERS, ROOT
from _stats import paired_effects

from egress_sim.analysis import PolicyVariant, ensure_output_directories, run_policy_variants
from egress_sim.config import load_spec


LEVELS = {
    "mean_delay_weight": (0.0, 0.25, 0.50),
    "density_weight": (2.0, 6.0, 12.0),
    "switch_weight": (0.02, 0.08, 0.20),
    "estimator_alpha": (0.10, 0.30, 0.60),
}


def run(seed_count: int = 30, workers: int = DEFAULT_WORKERS) -> None:
    base_spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    frames = []
    effect_rows = []
    for factor, levels in LEVELS.items():
        for level in levels:
            spec = replace(
                base_spec,
                robust_control=replace(
                    base_spec.robust_control,
                    **{factor: level},
                ),
            )
            variants = (
                PolicyVariant(
                    "unconditioned_mean_only",
                    "unconditioned_mean_only",
                    {},
                ),
                PolicyVariant(
                    "conditioned_mean_equal_spread",
                    "conditioned_mean_equal_spread",
                    {},
                ),
            )
            frame = run_policy_variants(
                spec,
                range(10_000, 10_000 + seed_count),
                variants,
                capacity_log_sigma=0.45,
                compliance_mean=0.85,
                workers=workers,
            )
            frame["factor"] = factor
            frame["level"] = level
            frames.append(frame)
            effect_rows.append(
                {
                    "factor": factor,
                    "level": level,
                    **paired_effects(
                        frame,
                        "conditioned_mean_equal_spread",
                        "unconditioned_mean_only",
                        label=f"matched-sensitivity-{factor}-{level}",
                    ),
                }
            )
    pd.concat(frames, ignore_index=True).to_csv(
        tables / "weight_sensitivity_runs.csv", index=False
    )
    pd.DataFrame(effect_rows).to_csv(
        tables / "weight_sensitivity_effects.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    run(args.seeds, args.workers)
