from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from _bootstrap import ROOT
from _stats import paired_effects

from egress_sim.analysis import PolicyVariant, ensure_output_directories, run_policy_variants
from egress_sim.config import load_spec


def run(seed_count: int = 50, workers: int = 8) -> None:
    base_spec = load_spec(ROOT / "config" / "default.json")
    contextual_time = base_spec.free_flow_time.copy()
    contextual_time[:, 0] = np.maximum(contextual_time[:, 0], 300.0)
    specs = (
        ("synthetic_base", base_spec),
        ("primary_route_floor_300s", replace(base_spec, free_flow_time=contextual_time)),
    )
    variants = (
        PolicyVariant("deterministic_load_balance", "deterministic_load_balance", {}),
        PolicyVariant("robust_rolling_horizon", "robust_rolling_horizon", {}),
    )
    tables, _ = ensure_output_directories(ROOT)
    frames = []
    effects = []
    for condition, spec in specs:
        frame = run_policy_variants(
            spec,
            range(50_000, 50_000 + seed_count),
            variants,
            workers=workers,
        )
        frame["condition"] = condition
        frames.append(frame)
        effects.append(
            {
                "condition": condition,
                **paired_effects(
                    frame,
                    "robust_rolling_horizon",
                    "deterministic_load_balance",
                    label=f"travel-time-context-{condition}",
                ),
            }
        )
    pd.concat(frames, ignore_index=True).to_csv(
        tables / "travel_time_context_runs.csv", index=False
    )
    pd.DataFrame(effects).to_csv(
        tables / "travel_time_context_effects.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.seeds, args.workers)
