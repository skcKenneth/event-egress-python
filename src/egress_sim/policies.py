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
from .uncertainty import (
    sample_centered_scenario,
    sample_conditioned_scenario,
    sample_scenario,
)


def _softmax_rows(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = -scores / max(temperature, 1e-9)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(scaled)
    return weights / weights.sum(axis=1, keepdims=True)


def _decision_substreams(
    rng: np.random.Generator,
) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """Create observation, candidate, and scenario RNGs for one update.

    The parent generator consumes exactly three integers per control update.
    Consequently, matched policies with the same policy seed receive identical
    scenario substream seeds even after their state-dependent candidate
    generators begin consuming different numbers of variates.
    """

    seeds = rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=3,
        dtype=np.int64,
    )
    generators = tuple(np.random.default_rng(int(seed)) for seed in seeds)
    return generators[0], generators[1], generators[2]


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


@dataclass(frozen=True)
class DecisionTrace:
    """Optional copies from the actual decision evaluation; no new draws."""

    candidates: np.ndarray
    scenarios: tuple[Scenario, ...]
    scenario_costs: np.ndarray
    clearance_minutes: np.ndarray
    mean_delay_minutes: np.ndarray
    density_penalty: np.ndarray
    mean_cost: np.ndarray
    cvar_cost: np.ndarray
    switch_cost: np.ndarray
    objective: np.ndarray
    selected_index: int


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
        capture_trace: bool = False,
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
        self.capture_trace = capture_trace
        self.last_trace: DecisionTrace | None = None
        self._trace_components: tuple[np.ndarray, ...] | None = None

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

    def _batched_scenario_costs(
        self,
        state: EgressState,
        spec: EgressSpec,
        candidates: list[np.ndarray],
        scenarios: list[Scenario],
    ) -> np.ndarray:
        """Evaluate all candidate-scenario rollouts in one vectorized batch.

        The scalar :meth:`_scenario_cost` remains the reference implementation
        used by validation tests and surrogate diagnostics. Capacity-estimate
        updates inside a rollout are omitted because the recommendation is
        frozen and those estimates are never read during the rollout.
        """

        recommendations = np.asarray(candidates, dtype=float)
        scenario_path = np.asarray(
            [scenario.path_capacity_multiplier for scenario in scenarios],
            dtype=float,
        )
        scenario_exit = np.asarray(
            [scenario.exit_capacity_multiplier for scenario in scenarios],
            dtype=float,
        )
        compliance = np.asarray(
            [scenario.compliance for scenario in scenarios],
            dtype=float,
        )
        candidate_count = recommendations.shape[0]
        scenario_count = scenario_path.shape[0]
        controller_spec = replace(
            spec,
            free_flow_time=self._controller_travel_time(spec),
        )
        z_count = spec.zone_count
        e_count = spec.exit_count
        max_delay = state.transit.shape[2]
        batch_shape = (candidate_count, scenario_count)

        remaining = np.broadcast_to(
            state.remaining,
            (*batch_shape, z_count),
        ).copy()
        path_queue = np.broadcast_to(
            state.path_queue,
            (*batch_shape, z_count, e_count),
        ).copy()
        transit = np.broadcast_to(
            state.transit,
            (*batch_shape, z_count, e_count, max_delay),
        ).copy()
        exit_queue = np.broadcast_to(
            state.exit_queue,
            (*batch_shape, e_count),
        ).copy()
        person_seconds = np.full(batch_shape, state.person_seconds, dtype=float)
        elapsed_seconds = np.full(batch_shape, state.elapsed_seconds, dtype=float)

        current_path_density = np.divide(
            state.path_queue,
            spec.path_queue_area,
            out=np.zeros_like(state.path_queue),
            where=spec.path_queue_area > 0,
        )
        current_exit_density = np.divide(
            state.exit_queue,
            spec.exit_queue_area,
            out=np.zeros_like(state.exit_queue),
            where=spec.exit_queue_area > 0,
        )
        initial_peak = float(
            max(np.max(current_path_density), np.max(current_exit_density))
        )
        peak_density = np.full(batch_shape, initial_peak, dtype=float)

        actual_choice = (
            compliance[None, :, :, None] * recommendations[:, None, :, :]
            + (1.0 - compliance[None, :, :, None])
            * spec.habitual_matrix[None, None, :, :]
        )
        path_capacity = controller_spec.path_capacity[None, :, :] * scenario_path
        exit_capacity = controller_spec.exit_capacity[None, :] * scenario_exit
        dt = spec.time_step_seconds
        release_per_step = spec.zone_release_rate * dt

        for _ in range(spec.robust_control.rollout_steps):
            active = (
                np.sum(remaining, axis=-1)
                + np.sum(path_queue, axis=(-2, -1))
                + np.sum(transit, axis=(-3, -2, -1))
                + np.sum(exit_queue, axis=-1)
            )
            alive = active > 1e-8
            if not np.any(alive):
                break
            person_seconds += active * dt * alive
            elapsed_seconds += dt * alive

            arrivals = transit[..., 0].copy()
            transit[..., :-1] = transit[..., 1:]
            transit[..., -1] = 0.0
            exit_queue += np.sum(arrivals, axis=-2)

            released = np.minimum(remaining, release_per_step)
            remaining -= released
            path_queue += released[..., None] * actual_choice

            path_outflow = np.minimum(
                path_queue,
                path_capacity[None, :, :, :] * dt,
            )
            path_queue -= path_outflow
            for zone in range(z_count):
                for exit_index in range(e_count):
                    delay_index = int(
                        controller_spec.delay_steps[zone, exit_index]
                    ) - 1
                    transit[:, :, zone, exit_index, delay_index] += path_outflow[
                        :, :, zone, exit_index
                    ]

            exit_outflow = np.minimum(
                exit_queue,
                exit_capacity[None, :, :] * dt,
            )
            exit_queue -= exit_outflow

            path_density = path_queue / spec.path_queue_area[None, None, :, :]
            exit_density = exit_queue / spec.exit_queue_area[None, None, :]
            peak_density = np.maximum(
                peak_density,
                np.maximum(
                    np.max(path_density, axis=(-2, -1)),
                    np.max(exit_density, axis=-1),
                ),
            )

        simulated_seconds = elapsed_seconds - state.elapsed_seconds
        future_path_work = path_queue + remaining[..., None] * actual_choice
        path_stage_seconds = np.max(
            future_path_work
            / np.maximum(path_capacity[None, :, :, :], 1e-9),
            axis=(-2, -1),
        )
        exit_work = (
            exit_queue
            + np.sum(transit, axis=(2, 4))
            + np.sum(future_path_work, axis=2)
        )
        exit_stage_seconds = np.max(
            exit_work / np.maximum(exit_capacity[None, :, :], 1e-9),
            axis=-1,
        )
        occupied_paths = future_path_work > 1e-8
        free_flow_seconds = np.max(
            np.where(
                occupied_paths,
                controller_spec.free_flow_time[None, None, :, :],
                0.0,
            ),
            axis=(-2, -1),
        )
        estimated_clearance_minutes = (
            simulated_seconds
            + path_stage_seconds
            + exit_stage_seconds
            + free_flow_seconds
        ) / 60.0
        start_people = max(state.active_population, 1.0)
        mean_delay_minutes = (
            person_seconds - state.person_seconds
        ) / start_people / 60.0
        excess_density = np.maximum(
            0.0,
            peak_density - spec.robust_control.safe_density_per_m2,
        )
        costs = (
            estimated_clearance_minutes
            + spec.robust_control.mean_delay_weight * mean_delay_minutes
            + spec.robust_control.density_weight * excess_density**2
        )
        if self.capture_trace:
            self._trace_components = (
                estimated_clearance_minutes.copy(),
                mean_delay_minutes.copy(),
                (spec.robust_control.density_weight * excess_density**2).copy(),
            )
        return costs

    def decide(
        self, state: EgressState, spec: EgressSpec, rng: np.random.Generator
    ) -> np.ndarray:
        observation_rng, candidate_rng, scenario_rng = _decision_substreams(rng)
        decision_state = self._decision_state(state, spec, observation_rng)
        candidates = self._candidate_recommendations(
            decision_state, spec, candidate_rng
        )
        scenarios = self._prediction_scenarios(
            decision_state, spec, scenario_rng
        )
        costs = self._batched_scenario_costs(state, spec, candidates, scenarios)
        alpha = spec.robust_control.cvar_alpha
        tail_count = max(1, int(np.ceil((1.0 - alpha) * len(scenarios))))
        mean_cost = np.mean(costs, axis=1)
        cvar_cost = np.mean(np.sort(costs, axis=1)[:, -tail_count:], axis=1)
        candidate_array = np.asarray(candidates, dtype=float)
        switch_cost = spec.robust_control.switch_weight * np.mean(
            np.abs(
                candidate_array
                - decision_state.last_recommendation[None, :, :]
            ),
            axis=(1, 2),
        )
        risk_weight = (
            spec.robust_control.risk_weight
            if self.risk_weight_override is None
            else self.risk_weight_override
        )
        objective = mean_cost + risk_weight * cvar_cost + switch_cost
        best_index = int(np.argmin(objective))
        self.last_diagnostics = RobustPolicyDiagnostics(
            objective=float(objective[best_index]),
            mean_cost=float(mean_cost[best_index]),
            cvar_cost=float(cvar_cost[best_index]),
            candidate_count=len(candidates),
        )
        self.last_trace = None
        if self.capture_trace:
            assert self._trace_components is not None
            clearance, mean_delay, density_penalty = self._trace_components
            self.last_trace = DecisionTrace(
                candidates=candidate_array.copy(),
                scenarios=tuple(
                    Scenario(s.path_capacity_multiplier.copy(),
                             s.exit_capacity_multiplier.copy(), s.compliance.copy())
                    for s in scenarios
                ),
                scenario_costs=costs.copy(),
                clearance_minutes=clearance,
                mean_delay_minutes=mean_delay,
                density_penalty=density_penalty,
                mean_cost=mean_cost.copy(),
                cvar_cost=cvar_cost.copy(),
                switch_cost=switch_cost.copy(),
                objective=objective.copy(),
                selected_index=best_index,
            )
        return candidates[best_index].copy()

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
        return [
            sample_centered_scenario(
                spec,
                rng,
                self._oracle_scenario.path_capacity_multiplier,
                self._oracle_scenario.exit_capacity_multiplier,
                capacity_log_sigma=0.0,
            )
            for _ in range(spec.robust_control.prediction_scenarios)
        ]


class OracleCapacityMeanPolicy(OracleCapacityPolicy):
    name = "oracle_capacity_mean_only"

    def __init__(self, **kwargs) -> None:
        super().__init__(risk_weight_override=0.0, **kwargs)


class ConditionedMeanPolicy(RobustRollingHorizonPolicy):
    """Mean-only controller centered on online capacity estimates.

    The default pre-clipping prediction log-standard deviation is the same
    environmental value used by :class:`UnconditionedMeanPolicy`, so the
    implemented primary treatment changes only the capacity-scenario center
    parameter under the same clipping rule and aligned scenario substreams.
    """

    name = "conditioned_mean_equal_spread"

    def __init__(self, **kwargs) -> None:
        super().__init__(risk_weight_override=0.0, **kwargs)


class ScenarioMeanPolicy(ConditionedMeanPolicy):
    """Backward-compatible alias for the conditioned mean controller."""

    name = "scenario_mean_only"


class ContractedSpreadMeanPolicy(RobustRollingHorizonPolicy):
    """Secondary diagnostic using the legacy contracted prediction spread.

    The primary paper comparison uses :class:`ConditionedMeanPolicy`, whose
    scenario log-standard deviation matches the unconditioned prior.  This
    class retains ``max(0.05, 0.45 * sigma)`` only to estimate the separate
    contribution of narrowing the scenario distribution.
    """

    name = "conditioned_mean_contracted_spread"

    def __init__(self, **kwargs) -> None:
        super().__init__(risk_weight_override=0.0, **kwargs)

    def _prediction_scenarios(
        self, state: EgressState, spec: EgressSpec, rng: np.random.Generator
    ) -> list:
        residual_sigma = max(0.05, 0.45 * spec.uncertainty.capacity_log_sigma)
        return [
            sample_conditioned_scenario(
                spec,
                rng,
                state.path_capacity_estimate,
                state.exit_capacity_estimate,
                residual_sigma=residual_sigma,
                compliance_mean=self.assumed_compliance_mean,
            )
            for _ in range(spec.robust_control.prediction_scenarios)
        ]


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
    elif kind == "conditioned_mean_equal_spread":
        policy = ConditionedMeanPolicy(**kwargs)
    elif kind == "scenario_mean_only":
        policy = ScenarioMeanPolicy(**kwargs)
    elif kind == "conditioned_mean_contracted_spread":
        policy = ContractedSpreadMeanPolicy(**kwargs)
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
    ConditionedMeanPolicy.name: ConditionedMeanPolicy,
    ScenarioMeanPolicy.name: ScenarioMeanPolicy,
    ContractedSpreadMeanPolicy.name: ContractedSpreadMeanPolicy,
    ChangeTriggeredFallbackPolicy.name: ChangeTriggeredFallbackPolicy,
}
