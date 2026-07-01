from __future__ import annotations

import argparse

from _bootstrap import DEFAULT_WORKERS, ROOT
from render_ieee_figures import figure1, figure2, figure3, validate_inputs
from run_clipped_lognormal_diagnostic import run as run_clipped_lognormal
from run_information_regime_map import run as run_information_map
from run_information_value_validation import run as run_final_validation
from run_residual_spread_ablation import run as run_residual_ablation
from run_time_step_sensitivity import run as run_time_step_sensitivity


def run(
    *,
    workers: int,
    validation_seeds: int,
    map_seeds: int,
    time_step_seeds: int,
    residual_seeds: int,
    lognormal_samples: int,
) -> None:
    """Generate all CSV inputs and render the three IEEE-style figures."""
    run_final_validation(seed_count=validation_seeds, workers=workers)
    run_information_map(seed_count=map_seeds, workers=workers)
    run_time_step_sensitivity(seed_count=time_step_seeds, workers=workers)
    run_residual_ablation(seed_count=residual_seeds, workers=workers)
    run_clipped_lognormal(samples=lognormal_samples)

    tables = ROOT / "outputs" / "tables"
    output = ROOT / "outputs" / "figures" / "ieee"
    output.mkdir(parents=True, exist_ok=True)
    validate_inputs(tables)
    figure1(tables, output)
    figure2(tables, output)
    figure3(tables, output)
    print(f"Wrote IEEE-style figures to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the result tables and render the IEEE-style figure set."
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--validation-seeds", type=int, default=200)
    parser.add_argument("--map-seeds", type=int, default=50)
    parser.add_argument("--time-step-seeds", type=int, default=50)
    parser.add_argument("--residual-seeds", type=int, default=50)
    parser.add_argument("--lognormal-samples", type=int, default=1_000_000)
    args = parser.parse_args()
    run(
        workers=args.workers,
        validation_seeds=args.validation_seeds,
        map_seeds=args.map_seeds,
        time_step_seeds=args.time_step_seeds,
        residual_seeds=args.residual_seeds,
        lognormal_samples=args.lognormal_samples,
    )
