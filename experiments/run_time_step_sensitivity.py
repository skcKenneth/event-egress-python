from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from _bootstrap import DEFAULT_WORKERS, ROOT
from _stats import paired_effects

from egress_sim.analysis import (
    PolicyVariant,
    aggregate_results,
    ensure_output_directories,
    execute_policy_variant_tasks,
)
from egress_sim.config import EgressSpec, load_spec


SEED_START = 92_000
TIME_STEPS = (10.0, 5.0, 2.5)
POLICY_SEED_OFFSET = 1_000_003


def spec_with_time_step(spec: EgressSpec, dt: float) -> EgressSpec:
    """Refine the physical time step without changing controller time scales."""

    dt = float(dt)
    base_dt = float(spec.time_step_seconds)
    base_alpha = float(spec.robust_control.estimator_alpha)
    # Preserve the same exponential-smoothing decay over a fixed physical time:
    # (1 - alpha_dt) ** (base_dt / dt) = 1 - alpha_base.
    estimator_alpha = 1.0 - (1.0 - base_alpha) ** (dt / base_dt)
    robust = replace(
        spec.robust_control,
        rollout_steps=int(round(300.0 / dt)),
        estimator_alpha=estimator_alpha,
    )
    return replace(
        spec,
        time_step_seconds=dt,
        control_interval_steps=int(round(120.0 / dt)),
        robust_control=robust,
    )


def run(seed_count: int = 30, workers: int = DEFAULT_WORKERS) -> None:
    base_spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    variants = (
        PolicyVariant("deterministic_load_balance", "deterministic_load_balance", {}),
        PolicyVariant("unconditioned_mean_only", "unconditioned_mean_only", {}),
        PolicyVariant("conditioned_mean_equal_spread", "conditioned_mean_equal_spread", {}),
    )
    seeds = range(SEED_START, SEED_START + seed_count)
    tasks = []
    for dt in TIME_STEPS:
        spec = spec_with_time_step(base_spec, dt)
        metadata = {
            "time_step_seconds": dt,
            "control_interval_steps": spec.control_interval_steps,
            "rollout_steps": spec.robust_control.rollout_steps,
            "estimator_alpha": spec.robust_control.estimator_alpha,
        }
        for seed in seeds:
            for variant in variants:
                tasks.append(
                    (
                        spec,
                        int(seed),
                        int(seed) + POLICY_SEED_OFFSET,
                        variant,
                        None,
                        None,
                        {},
                        metadata,
                    )
                )
    raw = execute_policy_variant_tasks(tasks, workers=workers)
    summaries = []
    effects = []
    for dt, subset in raw.groupby("time_step_seconds"):
        spec = spec_with_time_step(base_spec, float(dt))
        summaries.append(
            aggregate_results(subset, ["policy"]).assign(
                time_step_seconds=dt,
                estimator_alpha=spec.robust_control.estimator_alpha,
            )
        )
        for proposed, baseline, label in (
            (
                "conditioned_mean_equal_spread",
                "unconditioned_mean_only",
                "sensing-vs-unconditioned",
            ),
            (
                "conditioned_mean_equal_spread",
                "deterministic_load_balance",
                "conditioned-vs-reactive",
            ),
        ):
            effects.append(
                {
                    "time_step_seconds": dt,
                    "comparison": label,
                    "proposed_policy": proposed,
                    "baseline_policy": baseline,
                    "clearance_quantization_seconds": dt,
                    "control_interval_seconds": spec.control_interval_steps * dt,
                    "rollout_horizon_seconds": spec.robust_control.rollout_steps * dt,
                    "estimator_alpha": spec.robust_control.estimator_alpha,
                    **paired_effects(
                        subset,
                        proposed,
                        baseline,
                        label=f"dt-{dt}-{label}",
                    ),
                }
            )
    raw.to_csv(tables / "time_step_sensitivity_runs.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(
        tables / "time_step_sensitivity_summary.csv", index=False
    )
    pd.DataFrame(effects).to_csv(
        tables / "time_step_sensitivity_effects.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    run(args.seeds, args.workers)
