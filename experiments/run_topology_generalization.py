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


SEED_START = 98_000
TOPOLOGIES = (
    ("base_asymmetric", "default.json"),
    ("symmetric", "topology_symmetric.json"),
    ("path_bottleneck", "topology_path_bottleneck.json"),
    ("exit_bottleneck", "topology_exit_bottleneck.json"),
    ("expanded_venue", "topology_expanded_venue.json"),
)
STRUCTURAL_SIGMA = 0.30
STRUCTURAL_COMPLIANCE = 0.68


def run(seed_count: int = 50, workers: int = DEFAULT_WORKERS) -> None:
    tables, _ = ensure_output_directories(ROOT)
    variants = (
        PolicyVariant("unconditioned_mean_only", "unconditioned_mean_only", {}),
        PolicyVariant(
            "conditioned_mean_equal_spread",
            "conditioned_mean_equal_spread",
            {},
        ),
    )
    frames: list[pd.DataFrame] = []
    effects: list[dict[str, float | str]] = []
    for topology, config_name in TOPOLOGIES:
        spec = load_spec(ROOT / "config" / config_name)
        frame = run_policy_variants(
            spec,
            range(SEED_START, SEED_START + seed_count),
            variants,
            capacity_log_sigma=STRUCTURAL_SIGMA,
            compliance_mean=STRUCTURAL_COMPLIANCE,
            workers=workers,
        )
        frame["topology"] = topology
        frame["config_file"] = config_name
        frames.append(frame)
        effects.append(
            {
                "topology": topology,
                "config_file": config_name,
                "zones": spec.zone_count,
                "corridors": spec.exit_count,
                "total_population": spec.total_population,
                "capacity_log_sigma": STRUCTURAL_SIGMA,
                "compliance_mean": STRUCTURAL_COMPLIANCE,
                **paired_effects(
                    frame,
                    "conditioned_mean_equal_spread",
                    "unconditioned_mean_only",
                    label=f"topology-{topology}",
                ),
            }
        )

    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(tables / "topology_generalization_runs.csv", index=False)
    aggregate_results(raw, ["topology", "policy"]).to_csv(
        tables / "topology_generalization_summary.csv", index=False
    )
    pd.DataFrame(effects).to_csv(
        tables / "topology_generalization_effects.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    run(args.seeds, args.workers)
