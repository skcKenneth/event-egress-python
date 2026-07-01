from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import EgressSpec
from .model import (
    EgressState,
    Scenario,
    advance_state,
    current_peak_density,
    normalize_recommendation,
)
from .uncertainty import sample_conditioned_scenario, sample_scenario


def _softmax_rows(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = -scores / max(temperature, 1e-9)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(scaled)
    return weights / weights.sum(axis=1, keepdims=True)


class StaticNearestPolicy:
    name = "static_nearest"

    def decide(
        self, state: EgressState, spec: EgressSpec, rng: np.random.Generator
    ) -> np.ndarray:
        del state, rng
        return spec.habitual_matrix.copy()


class DeterministicLoadBalancePolicy:
    name = "deterministic_load_balance"

    def decide(
        self, state: EgressState, spec: EgressSpec, rng: np.random.Generator
    ) -> np.ndarray:
        del rng
        path_wait = np.divide(
            state.path_queue,
            spec.path_capacity,
            out=np.zeros_like(state.path_queue),
            where=spec.path_capacity > 0,
        )
        exit_wait = np.divide(
            state.exit_queue,
            spec.exit_capacity,
            out=np.zeros_like(state.exit_queue),
            where=spec.exit_capacity > 0,
        )
        score_seconds = spec.free_flow_time + path_wait + exit_wait[None, :]
        return _softmax_rows(score_seconds, temperature=24.0)


@dataclass
class RobustPolicyDiagnostics:
    objective: float = float("nan")
    mean_cost: float = float("nan")
    cvar_cost: float = float("nan")
    candidate_count: int = 0


class RobustRollingHorizonPolicy:
    name = "robust_rolling_horizon"

    def __init__(
        self,
        *,
        name_override: str | None = None,
        risk_weight_override: float | None = None,
        sensor_noise_sigma: float = 0.0,
        sensor_bias: float = 1.0,
        sensor_dropout: float = 0.0,
        sensor_delay_steps: int = 0,
        assumed_residual_sigma: float | None = None,
        assumed_compliance_mean: float | None = None,
        travel_time_multiplier: np.ndarray | None = None,
    ) -> None:
        if name_override is not None:
            self.name = name_override
        self.risk_weight_override = risk_weight_override
        self.sensor_noise_sigma = sensor_noise_sigma
        self.sensor_bias = sensor_bias
        self.sensor_dropout = sensor_dropout
        self.sensor_delay_steps = sensor_delay_steps
        self.assumed_residual_sigma = assumed_residual_sigma
        self.assumed_compliance_mean = assumed_compliance_mean
        self.travel_time_multiplier = travel_time_multiplier
        self._last_path_observation: np.ndarray | None = None
        self._last_exit_observation: np.ndarray | None = None
        self.last_diagnostics = RobustPolicyDiagnostics()

    def _controller_travel_time(self, spec: EgressSpec) -> np.ndarray:
        if self.travel_time_multiplier is None:
            return spec.free_flow_time
        multiplier = np.asarray(self.travel_time_multiplier, dtype=float)
        if multiplier.shape != spec.free_flow_time.shape:
            raise ValueError("travel_time_multiplier has incorrect shape")
        return spec.free_flow_time * multiplier

    def _decision_state(
        self, state: EgressState, spec: EgressSpec, rng: np.random.Generator
    ) -> EgressState:
        if (
            self.sensor_noise_sigma == 0.0
            and self.sensor_bias == 1.0
            and self.sensor_dropout == 0.0
            and self.sensor_delay_steps == 0
        ):
            return state
        delay = min(max(int(self.sensor_delay_steps), 0), len(state.path_capacity_estimate_history) - 1)
        path = state.path_capacity_estimate_history[-1 - delay].copy()
        exit_values = state.exit_capacity_estimate_history[-1 - delay].copy()
        if self.sensor_noise_sigma > 0.0:
            sigma = self.sensor_noise_sigma
            path *= np.exp(rng.normal(-0.5 * sigma**2, sigma, path.shape))
            exit_values *= np.exp(rng.normal(-0.5 * sigma**2, sigma, exit_values.shape))
        path *= self.sensor_bias
        exit_values *= self.sensor_bias
        if self._last_path_observation is None:
            self._last_path_observation = path.copy()
            self._last_exit_observation = exit_values.copy()
        if self.sensor_dropout > 0.0:
            path_mask = rng.random(path.shape) < self.sensor_dropout
            exit_mask = rng.random(exit_values.shape) < self.sensor_dropout
            path[path_mask] = self._last_path_observation[path_mask]
            exit_values[exit_mask] = self._last_exit_observation[exit_mask]
        lower = spec.uncertainty.capacity_multiplier_min
        upper = spec.uncertainty.capacity_multiplier_max
        path = np.clip(path, lower, upper)
        exit_values = np.clip(exit_values, lower, upper)
        self._last_path_observation = path.copy()
        self._last_exit_observation = exit_values.copy()
        observed = state.clone()
        observed.path_capacity_estimate = path
        observed.exit_capacity_estimate = exit_values
        return observed

    def _candidate_recommendations(
        self, state: EgressState, spec: EgressSpec, rng: np.random.Generator
    ) -> list[np.ndarray]:
        dynamic = DeterministicLoadBalancePolicy().decide(state, spec, rng)
        capacity_score = np.divide(
            spec.path_capacity * spec.exit_capacity[None, :],
            self._controller_travel_time(spec),
        )
        capacity_weighted = capacity_score / capacity_score.sum(axis=1, keepdims=True)
        equal = np.full((spec.zone_count, spec.exit_count), 1.0 / spec.exit_count)
        candidates = [
            spec.habitual_matrix.copy(),
            dynamic,
            0.75 * dynamic + 0.25 * equal,
            0.75 * dynamic + 0.25 * capacity_weighted,
            0.75 * dynamic + 0.25 * state.last_recommendation,
        ]
        target_count = max(spec.robust_control.candidate_count, len(candidates))
        while len(candidates) < target_count:
            anchor = dynamic if len(candidates) % 2 == 0 else capacity_weighted
            concentration = 8.0 + 48.0 * anchor
            perturbation = np.vstack([rng.dirichlet(row) for row in concentration])
            scale = spec.robust_control.perturbation_scale
            candidate = (1.0 - scale) * dynamic + scale * perturbation
            candidates.append(candidate)
        return [normalize_recommendation(candidate, spec) for candidate in candidates]

    def _scenario_cost(
        self,
        state: EgressState,
        spec: EgressSpec,
        recommendation: np.ndarray,
        scenario,
    ) -> float:
        rollout = state.clone()
        rollout.peak_density = current_peak_density(state, spec)
        controller_spec = replace(spec, free_flow_time=self._controller_travel_time(spec))
        start_people = max(rollout.active_population, 1.0)
        start_person_seconds = rollout.person_seconds
        for _ in range(spec.robust_control.rollout_steps):
            if rollout.active_population <= 1e-8:
                break
            advance_state(rollout, controller_spec, recommendation, scenario)
        simulated_seconds = rollout.elapsed_seconds - state.elapsed_seconds
        actual_choice = (
            scenario.compliance[:, None] * recommendation
            + (1.0 - scenario.compliance[:, None]) * spec.habitual_matrix
        )
        future_path_work = rollout.path_queue + rollout.remaining[:, None] * actual_choice
        path_capacity = controller_spec.path_capacity * scenario.path_capacity_multiplier
        path_stage_seconds = float(np.max(future_path_work / np.maximum(path_capacity, 1e-9)))
        exit_work = (
            rollout.exit_queue
            + np.sum(rollout.transit, axis=(0, 2))
            + np.sum(future_path_work, axis=0)
        )
        exit_capacity = controller_spec.exit_capacity * scenario.exit_capacity_multiplier
        exit_stage_seconds = float(np.max(exit_work / np.maximum(exit_capacity, 1e-9)))
        occupied_paths = future_path_work > 1e-8
        free_flow_seconds = float(
            np.max(controller_spec.free_flow_time[occupied_paths])
            if np.any(occupied_paths)
            else 0.0
        )
        residual_seconds = path_stage_seconds + exit_stage_seconds + free_flow_seconds
        estimated_clearance_minutes = (simulated_seconds + residual_seconds) / 60.0
        mean_delay_minutes = (
            (rollout.person_seconds - start_person_seconds) / start_people / 60.0
        )
        excess_density = max(
            0.0, rollout.peak_density - spec.robust_control.safe_density_per_m2
        )
        return float(
            estimated_clearance_minutes
            + spec.robust_control.mean_delay_weight * mean_delay_minutes
            + spec.robust_control.density_weight * excess_density**2
        )

    def decide(
        self, state: EgressState, spec: EgressSpec, rng: np.random.Generator
    ) -> np.ndarray:
        decision_state = self._decision_state(state, spec, rng)
        candidates = self._candidate_recommendations(decision_state, spec, rng)
        scenarios = self._prediction_scenarios(decision_state, spec, rng)
        alpha = spec.robust_control.cvar_alpha
        tail_count = max(1, int(np.ceil((1.0 - alpha) * len(scenarios))))
        best_objective = float("inf")
        best = candidates[0]
        best_mean = float("nan")
        best_cvar = float("nan")
        for candidate in candidates:
            costs = np.asarray(
                [self._scenario_cost(state, spec, candidate, scenario) for scenario in scenarios]
            )
            mean_cost = float(np.mean(costs))
            cvar_cost = float(np.mean(np.sort(costs)[-tail_count:]))
            switch_cost = spec.robust_control.switch_weight * float(
                np.mean(np.abs(candidate - decision_state.last_recommendation))
            )
            risk_weight = (
                spec.robust_control.risk_weight
                if self.risk_weight_override is None
                else self.risk_weight_override
            )
            objective = mean_cost + risk_weight * cvar_cost + switch_cost
            if objective < best_objective:
                best_objective = objective
                best = candidate
                best_mean = mean_cost
                best_cvar = cvar_cost
        self.last_diagnostics = RobustPolicyDiagnostics(
            objective=best_objective,
            mean_cost=best_mean,
            cvar_cost=best_cvar,
            candidate_count=len(candidates),
        )
        return best.copy()

    def _prediction_scenarios(
        self, state: EgressState, spec: EgressSpec, rng: np.random.Generator
    ) -> list:
        return [
            sample_conditioned_scenario(
                spec,
                rng,
                state.path_capacity_estimate,
                state.exit_capacity_estimate,
                residual_sigma=self.assumed_residual_sigma,
                compliance_mean=self.assumed_compliance_mean,
            )
            for _ in range(spec.robust_control.prediction_scenarios)
        ]


class UnconditionedRobustPolicy(RobustRollingHorizonPolicy):
    name = "robust_without_sensor_conditioning"

    def _prediction_scenarios(
        self, state: EgressState, spec: EgressSpec, rng: np.random.Generator
    ) -> list:
        del state
        return [
            sample_scenario(spec, rng)
            for _ in range(spec.robust_control.prediction_scenarios)
        ]


class UnconditionedMeanPolicy(UnconditionedRobustPolicy):
    name = "unconditioned_mean_only"

    def __init__(self, **kwargs) -> None:
        super().__init__(risk_weight_override=0.0, **kwargs)


class OracleCapacityPolicy(RobustRollingHorizonPolicy):
    """Non-deployable diagnostic using the realized fixed capacity multipliers."""

    name = "oracle_capacity_rolling_horizon"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._oracle_scenario: Scenario | None = None

    def set_oracle_scenario(self, scenario: Scenario) -> None:
        self._oracle_scenario = Scenario(
            scenario.path_capacity_multiplier.copy(),
            scenario.exit_capacity_multiplier.copy(),
            scenario.compliance.copy(),
        )

    def _prediction_scenarios(
        self, state: EgressState, spec: EgressSpec, rng: np.random.Generator
    ) -> list:
        del state
        if self._oracle_scenario is None:
            raise RuntimeError("oracle scenario must be attached before simulation")
        uncertainty = spec.uncertainty
        alpha = max(uncertainty.compliance_mean * uncertainty.compliance_concentration, 1e-6)
        beta = max(
            (1.0 - uncertainty.compliance_mean) * uncertainty.compliance_concentration,
            1e-6,
        )
        return [
            Scenario(
                self._oracle_scenario.path_capacity_multiplier.copy(),
                self._oracle_scenario.exit_capacity_multiplier.copy(),
                np.clip(rng.beta(alpha, beta, spec.zone_count), 0.02, 0.98),
            )
            for _ in range(spec.robust_control.prediction_scenarios)
        ]


class OracleCapacityMeanPolicy(OracleCapacityPolicy):
    name = "oracle_capacity_mean_only"

    def __init__(self, **kwargs) -> None:
        super().__init__(risk_weight_override=0.0, **kwargs)


class ScenarioMeanPolicy(RobustRollingHorizonPolicy):
    name = "scenario_mean_only"

    def __init__(self, **kwargs) -> None:
        super().__init__(risk_weight_override=0.0, **kwargs)


class ChangeTriggeredFallbackPolicy(RobustRollingHorizonPolicy):
    """Use reactive balancing briefly after a sharp capacity-estimate drop."""

    name = "change_triggered_fallback"

    def __init__(
        self,
        *,
        drop_threshold: float = 0.10,
        hold_updates: int = 2,
        minimum_detection_seconds: float = 240.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 < drop_threshold < 1.0:
            raise ValueError("drop_threshold must lie in (0, 1)")
        if hold_updates < 1:
            raise ValueError("hold_updates must be positive")
        self.drop_threshold = float(drop_threshold)
        self.hold_updates = int(hold_updates)
        self.minimum_detection_seconds = float(minimum_detection_seconds)
        self._fallback_updates_remaining = 0
        self.fallback_decisions = 0

    def _sharp_capacity_drop(self, state: EgressState, spec: EgressSpec) -> bool:
        if state.elapsed_seconds < self.minimum_detection_seconds:
            return False
        lag = min(spec.control_interval_steps, len(state.path_capacity_estimate_history) - 1)
        previous_path = state.path_capacity_estimate_history[-1 - lag]
        previous_exit = state.exit_capacity_estimate_history[-1 - lag]
        path_ratio = np.divide(
            state.path_capacity_estimate,
            previous_path,
            out=np.ones_like(state.path_capacity_estimate),
            where=previous_path > 1e-9,
        )
        exit_ratio = np.divide(
            state.exit_capacity_estimate,
            previous_exit,
            out=np.ones_like(state.exit_capacity_estimate),
            where=previous_exit > 1e-9,
        )
        threshold = 1.0 - self.drop_threshold
        coordinated_drop = (exit_ratio < threshold) & (
            np.median(path_ratio, axis=0) < threshold
        )
        return bool(np.any(coordinated_drop))

    def decide(
        self, state: EgressState, spec: EgressSpec, rng: np.random.Generator
    ) -> np.ndarray:
        if self._sharp_capacity_drop(state, spec):
            self._fallback_updates_remaining = max(
                self._fallback_updates_remaining, self.hold_updates
            )
        if self._fallback_updates_remaining > 0:
            self._fallback_updates_remaining -= 1
            self.fallback_decisions += 1
            return DeterministicLoadBalancePolicy().decide(state, spec, rng)
        return super().decide(state, spec, rng)


def build_policy(kind: str, *, name: str | None = None, **kwargs):
    if kind == "static_nearest":
        policy = StaticNearestPolicy()
    elif kind == "deterministic_load_balance":
        policy = DeterministicLoadBalancePolicy()
    elif kind == "scenario_mean_only":
        policy = ScenarioMeanPolicy(**kwargs)
    elif kind == "unconditioned_mean_only":
        policy = UnconditionedMeanPolicy(**kwargs)
    elif kind == "robust_without_sensor_conditioning":
        policy = UnconditionedRobustPolicy(**kwargs)
    elif kind == "oracle_capacity_rolling_horizon":
        policy = OracleCapacityPolicy(**kwargs)
    elif kind == "oracle_capacity_mean_only":
        policy = OracleCapacityMeanPolicy(**kwargs)
    elif kind == "robust_rolling_horizon":
        policy = RobustRollingHorizonPolicy(**kwargs)
    elif kind == "change_triggered_fallback":
        policy = ChangeTriggeredFallbackPolicy(**kwargs)
    else:
        raise ValueError(f"unknown policy kind: {kind}")
    if name is not None:
        policy.name = name
    return policy


POLICIES = {
    StaticNearestPolicy.name: StaticNearestPolicy,
    DeterministicLoadBalancePolicy.name: DeterministicLoadBalancePolicy,
    RobustRollingHorizonPolicy.name: RobustRollingHorizonPolicy,
    UnconditionedRobustPolicy.name: UnconditionedRobustPolicy,
    UnconditionedMeanPolicy.name: UnconditionedMeanPolicy,
    OracleCapacityPolicy.name: OracleCapacityPolicy,
    OracleCapacityMeanPolicy.name: OracleCapacityMeanPolicy,
    ScenarioMeanPolicy.name: ScenarioMeanPolicy,
    ChangeTriggeredFallbackPolicy.name: ChangeTriggeredFallbackPolicy,
}
