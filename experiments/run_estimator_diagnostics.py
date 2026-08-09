from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from _bootstrap import DEFAULT_WORKERS, ROOT

from egress_sim.analysis import spec_with_uncertainty_overrides
from egress_sim.config import load_spec
from egress_sim.policies import build_policy
from egress_sim.simulation import run_simulation


SEED_START = 97_000
REGIMES = (
    ("default", 0.22, 0.68),
    ("high_information", 0.45, 0.85),
)


def _worker(task: tuple[str, float, float, int]) -> tuple[list[dict], dict]:
    regime, sigma, compliance, seed = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    spec = spec_with_uncertainty_overrides(
        load_spec(ROOT / "config" / "default.json"),
        capacity_log_sigma=sigma,
        compliance_mean=compliance,
    )
    result = run_simulation(
        spec,
        build_policy("conditioned_mean_equal_spread"),
        seed=seed,
    )
    history = [
        {
            "regime": regime,
            "capacity_log_sigma": sigma,
            "compliance_mean": compliance,
            "seed": seed,
            "scenario_seed": result.scenario_seed,
            "policy_seed": result.policy_seed,
            "policy": result.policy,
            **row,
        }
        for row in result.history
    ]
    final = {
        "regime": regime,
        "capacity_log_sigma": sigma,
        "compliance_mean": compliance,
        **result.summary(),
    }
    return history, final


def _quantile(q: float):
    return lambda values: values.quantile(q)


def run(seed_count: int = 50, workers: int = DEFAULT_WORKERS) -> None:
    tables = ROOT / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    tasks = [
        (regime, sigma, compliance, seed)
        for regime, sigma, compliance in REGIMES
        for seed in range(SEED_START, SEED_START + seed_count)
    ]
    if workers <= 1:
        results = [_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_worker, tasks, chunksize=max(1, len(tasks) // (workers * 4))))

    history_rows = [row for histories, _ in results for row in histories]
    final_rows = [final for _, final in results]
    history = pd.DataFrame(history_rows)
    final = pd.DataFrame(final_rows)
    history.to_csv(tables / "estimator_diagnostics_runs.csv", index=False)
    final.to_csv(tables / "estimator_diagnostics_final.csv", index=False)

    summary = (
        history.groupby(["regime", "capacity_log_sigma", "compliance_mean", "time_seconds"], as_index=False)
        .agg(
            runs=("scenario_seed", "nunique"),
            capacity_mae_mean=("capacity_mae", "mean"),
            capacity_mae_median=("capacity_mae", "median"),
            capacity_mae_q10=("capacity_mae", _quantile(0.10)),
            capacity_mae_q90=("capacity_mae", _quantile(0.90)),
            path_capacity_mae_mean=("path_capacity_mae", "mean"),
            exit_capacity_mae_mean=("exit_capacity_mae", "mean"),
            capacity_observed_fraction_mean=("capacity_observed_fraction", "mean"),
            path_observed_fraction_mean=("path_observed_fraction", "mean"),
            exit_observed_fraction_mean=("exit_observed_fraction", "mean"),
            path_saturation_fraction_mean=("path_saturation_fraction", "mean"),
            exit_saturation_fraction_mean=("exit_saturation_fraction", "mean"),
        )
    )
    summary.to_csv(tables / "estimator_diagnostics_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    run(args.seeds, args.workers)
