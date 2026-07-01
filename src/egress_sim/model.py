from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EgressSpec


ESTIMATE_HISTORY_STEPS = 13


@dataclass(frozen=True)
class Scenario:
    path_capacity_multiplier: np.ndarray
    exit_capacity_multiplier: np.ndarray
    compliance: np.ndarray


@dataclass
class EgressState:
    remaining: np.ndarray
    path_queue: np.ndarray
    transit: np.ndarray
    exit_queue: np.ndarray
    cleared: float
    elapsed_seconds: float
    person_seconds: float
    peak_density: float
    last_recommendation: np.ndarray
    recommendation_changes: float
    path_capacity_estimate: np.ndarray
    exit_capacity_estimate: np.ndarray
    path_capacity_estimate_history: np.ndarray
    exit_capacity_estimate_history: np.ndarray

    def clone(self) -> "EgressState":
        return EgressState(
            remaining=self.remaining.copy(),
            path_queue=self.path_queue.copy(),
            transit=self.transit.copy(),
            exit_queue=self.exit_queue.copy(),
            cleared=float(self.cleared),
            elapsed_seconds=float(self.elapsed_seconds),
            person_seconds=float(self.person_seconds),
            peak_density=float(self.peak_density),
            last_recommendation=self.last_recommendation.copy(),
            recommendation_changes=float(self.recommendation_changes),
            path_capacity_estimate=self.path_capacity_estimate.copy(),
            exit_capacity_estimate=self.exit_capacity_estimate.copy(),
            path_capacity_estimate_history=self.path_capacity_estimate_history.copy(),
            exit_capacity_estimate_history=self.exit_capacity_estimate_history.copy(),
        )

    @property
    def active_population(self) -> float:
        return float(
            np.sum(self.remaining)
            + np.sum(self.path_queue)
            + np.sum(self.transit)
            + np.sum(self.exit_queue)
        )


def initial_state(spec: EgressSpec) -> EgressState:
    max_delay = int(np.max(spec.delay_steps))
    return EgressState(
        remaining=spec.population.copy(),
        path_queue=np.zeros((spec.zone_count, spec.exit_count), dtype=float),
        transit=np.zeros((spec.zone_count, spec.exit_count, max_delay), dtype=float),
        exit_queue=np.zeros(spec.exit_count, dtype=float),
        cleared=0.0,
        elapsed_seconds=0.0,
        person_seconds=0.0,
        peak_density=0.0,
        last_recommendation=spec.habitual_matrix.copy(),
        recommendation_changes=0.0,
        path_capacity_estimate=np.ones_like(spec.path_capacity),
        exit_capacity_estimate=np.ones_like(spec.exit_capacity),
        path_capacity_estimate_history=np.ones(
            (ESTIMATE_HISTORY_STEPS, *spec.path_capacity.shape), dtype=float
        ),
        exit_capacity_estimate_history=np.ones(
            (ESTIMATE_HISTORY_STEPS, *spec.exit_capacity.shape), dtype=float
        ),
    )


def normalize_recommendation(recommendation: np.ndarray, spec: EgressSpec) -> np.ndarray:
    recommendation = np.asarray(recommendation, dtype=float)
    if recommendation.shape != (spec.zone_count, spec.exit_count):
        raise ValueError("recommendation has incorrect shape")
    recommendation = np.clip(recommendation, 0.0, None)
    row_sums = recommendation.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("each recommendation row must contain positive mass")
    return recommendation / row_sums


def current_peak_density(state: EgressState, spec: EgressSpec) -> float:
    path_density = np.divide(
        state.path_queue,
        spec.path_queue_area,
        out=np.zeros_like(state.path_queue),
        where=spec.path_queue_area > 0,
    )
    exit_density = np.divide(
        state.exit_queue,
        spec.exit_queue_area,
        out=np.zeros_like(state.exit_queue),
        where=spec.exit_queue_area > 0,
    )
    return float(max(np.max(path_density), np.max(exit_density)))


def advance_state(
    state: EgressState,
    spec: EgressSpec,
    recommendation: np.ndarray,
    scenario: Scenario,
) -> dict[str, float]:
    recommendation = normalize_recommendation(recommendation, spec)
    dt = spec.time_step_seconds
    active_before = state.active_population
    state.person_seconds += active_before * dt

    arrivals = state.transit[:, :, 0].copy()
    state.transit[:, :, :-1] = state.transit[:, :, 1:]
    state.transit[:, :, -1] = 0.0
    state.exit_queue += arrivals.sum(axis=0)

    actual_choice = (
        scenario.compliance[:, None] * recommendation
        + (1.0 - scenario.compliance[:, None]) * spec.habitual_matrix
    )
    released = np.minimum(state.remaining, spec.zone_release_rate * dt)
    state.remaining -= released
    state.path_queue += released[:, None] * actual_choice

    path_step_capacity = spec.path_capacity * scenario.path_capacity_multiplier * dt
    path_demand = state.path_queue.copy()
    path_outflow = np.minimum(state.path_queue, path_step_capacity)
    state.path_queue -= path_outflow
    path_saturated = path_demand >= path_step_capacity - 1e-9
    observed_path_multiplier = np.divide(
        path_outflow,
        spec.path_capacity * dt,
        out=state.path_capacity_estimate.copy(),
        where=spec.path_capacity > 0,
    )
    state.path_capacity_estimate[path_saturated] = (
        (1.0 - spec.robust_control.estimator_alpha)
        * state.path_capacity_estimate[path_saturated]
        + spec.robust_control.estimator_alpha
        * observed_path_multiplier[path_saturated]
    )
    for zone in range(spec.zone_count):
        for exit_index in range(spec.exit_count):
            delay_index = int(spec.delay_steps[zone, exit_index]) - 1
            state.transit[zone, exit_index, delay_index] += path_outflow[zone, exit_index]

    exit_step_capacity = spec.exit_capacity * scenario.exit_capacity_multiplier * dt
    exit_demand = state.exit_queue.copy()
    exit_outflow = np.minimum(state.exit_queue, exit_step_capacity)
    state.exit_queue -= exit_outflow
    exit_saturated = exit_demand >= exit_step_capacity - 1e-9
    observed_exit_multiplier = np.divide(
        exit_outflow,
        spec.exit_capacity * dt,
        out=state.exit_capacity_estimate.copy(),
        where=spec.exit_capacity > 0,
    )
    state.exit_capacity_estimate[exit_saturated] = (
        (1.0 - spec.robust_control.estimator_alpha)
        * state.exit_capacity_estimate[exit_saturated]
        + spec.robust_control.estimator_alpha
        * observed_exit_multiplier[exit_saturated]
    )
    state.path_capacity_estimate_history[:-1] = state.path_capacity_estimate_history[1:]
    state.path_capacity_estimate_history[-1] = state.path_capacity_estimate
    state.exit_capacity_estimate_history[:-1] = state.exit_capacity_estimate_history[1:]
    state.exit_capacity_estimate_history[-1] = state.exit_capacity_estimate
    state.cleared += float(np.sum(exit_outflow))

    state.elapsed_seconds += dt
    state.peak_density = max(state.peak_density, current_peak_density(state, spec))
    change = float(np.sum(np.abs(recommendation - state.last_recommendation)))
    state.recommendation_changes += change
    state.last_recommendation = recommendation.copy()

    return {
        "released": float(np.sum(released)),
        "path_outflow": float(np.sum(path_outflow)),
        "exit_outflow": float(np.sum(exit_outflow)),
        "active_population": state.active_population,
        "density": current_peak_density(state, spec),
        "mass_balance": state.active_population + state.cleared,
    }
