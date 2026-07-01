from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EgressSpec
from .policies import (
    DeterministicLoadBalancePolicy,
    RobustRollingHorizonPolicy,
    StaticNearestPolicy,
    build_policy,
)
from .simulation import run_simulation


POLICY_CLASSES = (
    StaticNearestPolicy,
    DeterministicLoadBalancePolicy,
    RobustRollingHorizonPolicy,
)


@dataclass(frozen=True)
class PolicyVariant:
    kind: str
    name: str
    kwargs: dict


def spec_with_uncertainty_overrides(
    spec: EgressSpec,
    *,
    capacity_log_sigma: float | None = None,
    compliance_mean: float | None = None,
) -> EgressSpec:
    if capacity_log_sigma is None and compliance_mean is None:
        return spec
    uncertainty = replace(
        spec.uncertainty,
        capacity_log_sigma=(
            spec.uncertainty.capacity_log_sigma
            if capacity_log_sigma is None
            else float(capacity_log_sigma)
        ),
        compliance_mean=(
            spec.uncertainty.compliance_mean
            if compliance_mean is None
            else float(compliance_mean)
        ),
    )
    return replace(spec, uncertainty=uncertainty)


def _worker_init() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _executor_chunksize(task_count: int, workers: int) -> int:
    if task_count <= workers:
        return 1
    return max(1, task_count // (workers * 4))


def _normalize_task(task: tuple) -> tuple:
    if len(task) == 7:
        return (*task, {})
    if len(task) != 8:
        raise ValueError(f"expected a 7- or 8-item task tuple, got {len(task)} items")
    return task


def _run_variant_task(task: tuple) -> dict:
    (
        spec,
        scenario_seed,
        policy_seed,
        variant,
        capacity_log_sigma,
        compliance_mean,
        simulation_kwargs,
        metadata,
    ) = _normalize_task(task)
    run_spec = spec_with_uncertainty_overrides(
        spec,
        capacity_log_sigma=capacity_log_sigma,
        compliance_mean=compliance_mean,
    )
    policy = build_policy(variant.kind, name=variant.name, **variant.kwargs)
    result = run_simulation(
        run_spec,
        policy,
        seed=scenario_seed,
        policy_seed=policy_seed,
        **simulation_kwargs,
    )
    summary = result.summary()
    if capacity_log_sigma is not None:
        summary["capacity_log_sigma"] = capacity_log_sigma
    if compliance_mean is not None:
        summary["compliance_mean"] = compliance_mean
    summary.update(metadata)
    return summary


def execute_policy_variant_tasks(
    tasks: list[tuple],
    *,
    workers: int = 1,
) -> pd.DataFrame:
    if not tasks:
        return pd.DataFrame()
    if workers <= 1:
        rows = [_run_variant_task(task) for task in tasks]
    else:
        chunksize = _executor_chunksize(len(tasks), workers)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
        ) as executor:
            rows = list(
                executor.map(_run_variant_task, tasks, chunksize=chunksize)
            )
    return pd.DataFrame(rows)


def run_policy_variants(
    spec: EgressSpec,
    seeds: Iterable[int],
    variants: tuple[PolicyVariant, ...],
    *,
    capacity_log_sigma: float | None = None,
    compliance_mean: float | None = None,
    workers: int = 1,
    policy_seed_offset: int = 1_000_003,
    simulation_kwargs: dict | None = None,
) -> pd.DataFrame:
    simulation_kwargs = simulation_kwargs or {}
    tasks = [
        (
            spec,
            int(seed),
            int(seed) + policy_seed_offset,
            variant,
            capacity_log_sigma,
            compliance_mean,
            simulation_kwargs,
            {},
        )
        for seed in seeds
        for variant in variants
    ]
    return execute_policy_variant_tasks(tasks, workers=workers)


def run_policy_suite(
    spec: EgressSpec,
    seeds: Iterable[int],
    *,
    policy_classes=POLICY_CLASSES,
    capacity_log_sigma: float | None = None,
    compliance_mean: float | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    variants = tuple(
        PolicyVariant(kind=policy_class.name, name=policy_class.name, kwargs={})
        for policy_class in policy_classes
    )
    return run_policy_variants(
        spec,
        seeds,
        variants,
        capacity_log_sigma=capacity_log_sigma,
        compliance_mean=compliance_mean,
        workers=workers,
    )


def aggregate_results(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, as_index=False)
    return grouped.agg(
        runs=("seed", "count"),
        completion_rate=("completed", "mean"),
        clearance_mean_s=("clearance_time_seconds", "mean"),
        clearance_std_s=("clearance_time_seconds", "std"),
        clearance_p90_s=("clearance_time_seconds", lambda values: values.quantile(0.90)),
        mean_person_time_s=("mean_clearance_time_seconds", "mean"),
        peak_density_mean=("peak_density_per_m2", "mean"),
        peak_density_max=("peak_density_per_m2", "max"),
        mass_balance_error_max=("mass_balance_error", "max"),
    )


def ensure_output_directories(root: Path) -> tuple[Path, Path]:
    tables = root / "outputs" / "tables"
    figures = root / "outputs" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def relative_change(proposed: float, baseline: float) -> float:
    return 100.0 * (proposed - baseline) / baseline if baseline else np.nan
