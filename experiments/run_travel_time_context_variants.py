from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from _bootstrap import ROOT
from _stats import paired_effects

from egress_sim.analysis import PolicyVariant, ensure_output_directories, run_policy_variants
from egress_sim.config import load_spec


def _variant_specs(base_spec) -> tuple[tuple[str, object], ...]:
    floor_180 = base_spec.free_flow_time.copy()
    floor_180[:, 0] = np.maximum(floor_180[:, 0], 180.0)
    floor_300 = base_spec.free_flow_time.copy()
    floor_300[:, 0] = np.maximum(floor_300[:, 0], 300.0)
    floor_420 = base_spec.free_flow_time.copy()
    floor_420[:, 0] = np.maximum(floor_420[:, 0], 420.0)
    additive_300 = base_spec.free_flow_time.copy()
    additive_300[:, 0] = additive_300[:, 0] + 300.0
    return (
        ("synthetic_base", base_spec),
        ("primary_route_floor_180s", replace(base_spec, free_flow_time=floor_180)),
        ("primary_route_floor_300s", replace(base_spec, free_flow_time=floor_300)),
        ("primary_route_floor_420s", replace(base_spec, free_flow_time=floor_420)),
        ("primary_route_additive_300s", replace(base_spec, free_flow_time=additive_300)),
    )


def run(seed_count: int = 50, workers: int = 8) -> None:
    base_spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    variants = (
        PolicyVariant("deterministic_load_balance", "deterministic_load_balance", {}),
        PolicyVariant("robust_rolling_horizon", "robust_rolling_horizon", {}),
    )
    frames = []
    effects = []
    for condition, spec in _variant_specs(base_spec):
        frame = run_policy_variants(
            spec,
            range(51_000, 51_000 + seed_count),
            variants,
            workers=workers,
        )
        frame["condition"] = condition
        frames.append(frame)
        effects.append(
            {
                "condition": condition,
                "proposed_policy": "robust_rolling_horizon",
                "baseline_policy": "deterministic_load_balance",
                **paired_effects(
                    frame,
                    "robust_rolling_horizon",
                    "deterministic_load_balance",
                    label=f"travel-time-context-variant-{condition}",
                ),
            }
        )
    pd.concat(frames, ignore_index=True).to_csv(
        tables / "travel_time_context_variant_runs.csv", index=False
    )
    pd.DataFrame(effects).to_csv(
        tables / "travel_time_context_variant_effects.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.seeds, args.workers)
