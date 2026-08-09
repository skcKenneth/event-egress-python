from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EgressSpec
from .model import EgressState, Scenario, advance_state, initial_state
from .uncertainty import sample_scenario


@dataclass(frozen=True)
class SimulationResult:
    policy: str
    seed: int
    scenario_seed: int
    policy_seed: int
    completed: bool
    clearance_time_seconds: float
    mean_clearance_time_seconds: float
    peak_density_per_m2: float
    recommendation_changes: float
    fallback_decisions: int
    mass_balance_error: float
    capacity_mae: float
    path_capacity_mae: float
    exit_capacity_mae: float
    capacity_observed_fraction: float
    path_observed_fraction: float
    exit_observed_fraction: float
    history: tuple[dict[str, float], ...]

    def summary(self) -> dict[str, float | int | str | bool]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "scenario_seed": self.scenario_seed,
            "policy_seed": self.policy_seed,
            "completed": self.completed,
            "clearance_time_seconds": self.clearance_time_seconds,
            "mean_clearance_time_seconds": self.mean_clearance_time_seconds,
            "peak_density_per_m2": self.peak_density_per_m2,
            "recommendation_changes": self.recommendation_changes,
            "fallback_decisions": self.fallback_decisions,
            "mass_balance_error": self.mass_balance_error,
            "capacity_mae": self.capacity_mae,
            "path_capacity_mae": self.path_capacity_mae,
            "exit_capacity_mae": self.exit_capacity_mae,
            "capacity_observed_fraction": self.capacity_observed_fraction,
            "path_observed_fraction": self.path_observed_fraction,
            "exit_observed_fraction": self.exit_observed_fraction,
        }


def _capacity_diagnostics(
    state: EgressState,
    scenario: Scenario,
) -> dict[str, float]:
    path_error = np.abs(
        state.path_capacity_estimate - scenario.path_capacity_multiplier
    )
    exit_error = np.abs(
        state.exit_capacity_estimate - scenario.exit_capacity_multiplier
    )
    path_observed = state.path_saturated_observation_count > 0
    exit_observed = state.exit_saturated_observation_count > 0
    total_links = path_error.size + exit_error.size
    return {
        "path_capacity_mae": float(np.mean(path_error)),
        "exit_capacity_mae": float(np.mean(exit_error)),
        "capacity_mae": float(
            (float(np.sum(path_error)) + float(np.sum(exit_error))) / total_links
        ),
        "path_observed_fraction": float(np.mean(path_observed)),
        "exit_observed_fraction": float(np.mean(exit_observed)),
        "capacity_observed_fraction": float(
            (int(np.sum(path_observed)) + int(np.sum(exit_observed))) / total_links
        ),
    }


def run_simulation(
    spec: EgressSpec,
    policy,
    *,
    seed: int,
    policy_seed: int | None = None,
    scenario: Scenario | None = None,
    capacity_log_sigma: float | None = None,
    compliance_mean: float | None = None,
    temporary_exit_index: int | None = None,
    temporary_capacity_factor: float = 1.0,
    temporary_start_seconds: float = 240.0,
    temporary_end_seconds: float = 600.0,
    temporary_scope: str = "both",
    temporary_profile: str = "abrupt",
) -> SimulationResult:
    scenario_rng = np.random.default_rng(seed)
    actual_policy_seed = int(seed + 1_000_003 if policy_seed is None else policy_seed)
    policy_rng = np.random.default_rng(actual_policy_seed)
    actual_scenario = scenario or sample_scenario(
        spec,
        scenario_rng,
        capacity_log_sigma=capacity_log_sigma,
        compliance_mean=compliance_mean,
    )
    if hasattr(policy, "set_oracle_scenario"):
        policy.set_oracle_scenario(actual_scenario)
    state: EgressState = initial_state(spec)
    recommendation = state.last_recommendation.copy()
    history: list[dict[str, float]] = []
    max_steps = int(np.ceil(spec.maximum_duration_seconds / spec.time_step_seconds))
    last_step_scenario = actual_scenario

    for step in range(max_steps):
        if step % spec.control_interval_steps == 0:
            recommendation = policy.decide(state, spec, policy_rng)
        step_scenario = actual_scenario
        if (
            temporary_exit_index is not None
            and temporary_start_seconds <= state.elapsed_seconds < temporary_end_seconds
        ):
            duration = max(
                temporary_end_seconds - temporary_start_seconds,
                spec.time_step_seconds,
            )
            progress = min(
                1.0,
                max(
                    0.0,
                    (state.elapsed_seconds - temporary_start_seconds) / duration,
                ),
            )
            if temporary_profile == "abrupt":
                factor = temporary_capacity_factor
            elif temporary_profile == "gradual":
                factor = 1.0 + (temporary_capacity_factor - 1.0) * progress
            elif temporary_profile == "recovery":
                factor = temporary_capacity_factor + (
                    1.0 - temporary_capacity_factor
                ) * progress
            elif temporary_profile == "piecewise":
                factor = (
                    temporary_capacity_factor
                    if progress < 0.5
                    else (1.0 + temporary_capacity_factor) / 2.0
                )
            else:
                raise ValueError(f"unknown temporary_profile: {temporary_profile}")
            path_multiplier = actual_scenario.path_capacity_multiplier.copy()
            exit_multiplier = actual_scenario.exit_capacity_multiplier.copy()
            if temporary_scope in ("both", "path_only"):
                path_multiplier[:, temporary_exit_index] *= factor
            if temporary_scope in ("both", "exit_only"):
                exit_multiplier[temporary_exit_index] *= factor
            if temporary_scope not in ("both", "path_only", "exit_only"):
                raise ValueError(f"unknown temporary_scope: {temporary_scope}")
            step_scenario = Scenario(
                path_multiplier,
                exit_multiplier,
                actual_scenario.compliance,
            )
        last_step_scenario = step_scenario
        diagnostics = advance_state(state, spec, recommendation, step_scenario)
        capacity = _capacity_diagnostics(state, step_scenario)
        history.append(
            {
                "time_seconds": state.elapsed_seconds,
                "remaining": float(np.sum(state.remaining)),
                "path_queue": float(np.sum(state.path_queue)),
                "in_transit": float(np.sum(state.transit)),
                "exit_queue": float(np.sum(state.exit_queue)),
                "cleared": state.cleared,
                "active_population": float(diagnostics["active_population"]),
                "density_per_m2": float(diagnostics["density"]),
                "path_saturation_fraction": float(
                    diagnostics["path_saturation_fraction"]
                ),
                "exit_saturation_fraction": float(
                    diagnostics["exit_saturation_fraction"]
                ),
                **capacity,
            }
        )
        if state.active_population <= 1e-8:
            break

    completed = state.active_population <= 1e-8
    mass_error = abs(state.active_population + state.cleared - spec.total_population)
    mean_clearance = state.person_seconds / max(spec.total_population, 1.0)
    final_capacity = _capacity_diagnostics(state, last_step_scenario)
    return SimulationResult(
        policy=policy.name,
        seed=seed,
        scenario_seed=seed,
        policy_seed=actual_policy_seed,
        completed=completed,
        clearance_time_seconds=state.elapsed_seconds,
        mean_clearance_time_seconds=mean_clearance,
        peak_density_per_m2=state.peak_density,
        recommendation_changes=state.recommendation_changes,
        fallback_decisions=int(getattr(policy, "fallback_decisions", 0)),
        mass_balance_error=mass_error,
        history=tuple(history),
        **final_capacity,
    )
