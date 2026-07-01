from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class UncertaintySpec:
    capacity_log_sigma: float
    compliance_mean: float
    compliance_concentration: float
    capacity_multiplier_min: float
    capacity_multiplier_max: float


@dataclass(frozen=True)
class RobustControlSpec:
    prediction_scenarios: int
    candidate_count: int
    rollout_steps: int
    cvar_alpha: float
    risk_weight: float
    mean_delay_weight: float
    density_weight: float
    switch_weight: float
    safe_density_per_m2: float
    estimator_alpha: float
    perturbation_scale: float


@dataclass(frozen=True)
class EgressSpec:
    time_step_seconds: float
    control_interval_steps: int
    maximum_duration_seconds: float
    zones: tuple[str, ...]
    exits: tuple[str, ...]
    population: np.ndarray
    zone_release_rate: np.ndarray
    path_capacity: np.ndarray
    free_flow_time: np.ndarray
    path_queue_area: np.ndarray
    exit_capacity: np.ndarray
    exit_queue_area: np.ndarray
    habitual_exit_index: np.ndarray
    uncertainty: UncertaintySpec
    robust_control: RobustControlSpec

    @property
    def zone_count(self) -> int:
        return len(self.zones)

    @property
    def exit_count(self) -> int:
        return len(self.exits)

    @property
    def total_population(self) -> float:
        return float(np.sum(self.population))

    @property
    def habitual_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.zone_count, self.exit_count), dtype=float)
        matrix[np.arange(self.zone_count), self.habitual_exit_index] = 1.0
        return matrix

    @property
    def delay_steps(self) -> np.ndarray:
        return np.maximum(1, np.ceil(self.free_flow_time / self.time_step_seconds).astype(int))


def _array(data: dict, key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(data[key], dtype=float)
    if value.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {value.shape}")
    if np.any(value <= 0.0):
        raise ValueError(f"{key} must contain positive values")
    return value


def load_spec(path: str | Path) -> EgressSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    zones = tuple(data["zones"])
    exits = tuple(data["exits"])
    z_count, e_count = len(zones), len(exits)
    habitual = np.asarray(data["habitual_exit_index"], dtype=int)
    if habitual.shape != (z_count,) or np.any(habitual < 0) or np.any(habitual >= e_count):
        raise ValueError("habitual_exit_index must contain one valid exit index per zone")

    uncertainty = UncertaintySpec(**data["uncertainty"])
    robust = RobustControlSpec(**data["robust_control"])
    if not 0.0 < uncertainty.compliance_mean < 1.0:
        raise ValueError("compliance_mean must be in (0, 1)")
    if not 0.0 < robust.cvar_alpha < 1.0:
        raise ValueError("cvar_alpha must be in (0, 1)")
    if not 0.0 < robust.estimator_alpha <= 1.0:
        raise ValueError("estimator_alpha must be in (0, 1]")
    if not 0.0 <= robust.perturbation_scale <= 1.0:
        raise ValueError("perturbation_scale must be in [0, 1]")

    return EgressSpec(
        time_step_seconds=float(data["time_step_seconds"]),
        control_interval_steps=int(data["control_interval_steps"]),
        maximum_duration_seconds=float(data["maximum_duration_seconds"]),
        zones=zones,
        exits=exits,
        population=_array(data, "population", (z_count,)),
        zone_release_rate=_array(data, "zone_release_rate_per_second", (z_count,)),
        path_capacity=_array(data, "path_capacity_per_second", (z_count, e_count)),
        free_flow_time=_array(data, "free_flow_time_seconds", (z_count, e_count)),
        path_queue_area=_array(data, "path_queue_area_m2", (z_count, e_count)),
        exit_capacity=_array(data, "exit_capacity_per_second", (e_count,)),
        exit_queue_area=_array(data, "exit_queue_area_m2", (e_count,)),
        habitual_exit_index=habitual,
        uncertainty=uncertainty,
        robust_control=robust,
    )
