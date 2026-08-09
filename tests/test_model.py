from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from egress_sim.config import RobustControlSpec, load_spec
from egress_sim.analysis import spec_with_uncertainty_overrides
from egress_sim.model import advance_state, initial_state
from egress_sim.policies import (
    ChangeTriggeredFallbackPolicy,
    ConditionedMeanPolicy,
    ContractedSpreadMeanPolicy,
    DeterministicLoadBalancePolicy,
    OracleCapacityMeanPolicy,
    OracleCapacityPolicy,
    RobustRollingHorizonPolicy,
    ScenarioMeanPolicy,
    StaticNearestPolicy,
    UnconditionedMeanPolicy,
    UnconditionedRobustPolicy,
    build_policy,
)
from egress_sim.simulation import run_simulation
from egress_sim.uncertainty import nominal_scenario
from run_time_step_sensitivity import spec_with_time_step


class EgressModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_spec(ROOT / "config" / "default.json")

    def test_mass_is_conserved_each_step(self) -> None:
        state = initial_state(self.spec)
        scenario = nominal_scenario(self.spec)
        recommendation = self.spec.habitual_matrix
        for _ in range(40):
            diagnostics = advance_state(state, self.spec, recommendation, scenario)
            self.assertAlmostEqual(
                diagnostics["mass_balance"], self.spec.total_population, places=8
            )
            self.assertTrue(np.all(state.remaining >= -1e-10))
            self.assertTrue(np.all(state.path_queue >= -1e-10))
            self.assertTrue(np.all(state.transit >= -1e-10))
            self.assertTrue(np.all(state.exit_queue >= -1e-10))

    def test_policy_rows_are_probability_distributions(self) -> None:
        state = initial_state(self.spec)
        rng = np.random.default_rng(3)
        for policy in (StaticNearestPolicy(), DeterministicLoadBalancePolicy()):
            recommendation = policy.decide(state, self.spec, rng)
            np.testing.assert_allclose(recommendation.sum(axis=1), 1.0)
            self.assertTrue(np.all(recommendation >= 0.0))

    def test_robust_policy_rows_are_probability_distributions(self) -> None:
        small_control = RobustControlSpec(
            prediction_scenarios=3,
            candidate_count=6,
            rollout_steps=4,
            cvar_alpha=0.8,
            risk_weight=0.45,
            mean_delay_weight=0.25,
            density_weight=6.0,
            switch_weight=0.08,
            safe_density_per_m2=3.5,
            estimator_alpha=0.3,
            perturbation_scale=0.3,
        )
        spec = replace(self.spec, robust_control=small_control)
        recommendation = RobustRollingHorizonPolicy().decide(
            initial_state(spec), spec, np.random.default_rng(5)
        )
        np.testing.assert_allclose(recommendation.sum(axis=1), 1.0)
        self.assertTrue(np.all(recommendation >= 0.0))

    def test_batched_rollout_costs_match_scalar_reference(self) -> None:
        small_control = replace(
            self.spec.robust_control,
            prediction_scenarios=3,
            candidate_count=6,
            rollout_steps=8,
        )
        spec = replace(self.spec, robust_control=small_control)
        state = initial_state(spec)
        actual = nominal_scenario(spec)
        for _ in range(10):
            advance_state(state, spec, spec.habitual_matrix, actual)
        policy = ConditionedMeanPolicy()
        candidates = policy._candidate_recommendations(
            state, spec, np.random.default_rng(1011)
        )
        scenarios = policy._prediction_scenarios(
            state, spec, np.random.default_rng(1012)
        )
        scalar = np.asarray(
            [
                [policy._scenario_cost(state, spec, candidate, scenario) for scenario in scenarios]
                for candidate in candidates
            ]
        )
        batched = policy._batched_scenario_costs(
            state, spec, candidates, scenarios
        )
        np.testing.assert_allclose(batched, scalar, rtol=1e-12, atol=1e-12)

    def test_candidate_generation_is_normalized_and_ordered(self) -> None:
        state = initial_state(self.spec)
        rng = np.random.default_rng(23)
        policy = RobustRollingHorizonPolicy()
        candidates = policy._candidate_recommendations(state, self.spec, rng)
        self.assertEqual(len(candidates), self.spec.robust_control.candidate_count)
        np.testing.assert_allclose(candidates[0], self.spec.habitual_matrix)
        reactive = DeterministicLoadBalancePolicy().decide(
            state, self.spec, np.random.default_rng(23)
        )
        np.testing.assert_allclose(candidates[1], reactive)
        for candidate in candidates:
            np.testing.assert_allclose(candidate.sum(axis=1), 1.0)
            self.assertTrue(np.all(candidate >= 0.0))

    def test_conditioned_and_unconditioned_mean_share_candidates(self) -> None:
        state = initial_state(self.spec)
        state.path_capacity_estimate[:] = 0.65
        state.exit_capacity_estimate[:] = 0.70
        conditioned = ScenarioMeanPolicy()
        unconditioned = UnconditionedMeanPolicy()
        cond_candidates = conditioned._candidate_recommendations(
            state, self.spec, np.random.default_rng(101)
        )
        uncond_candidates = unconditioned._candidate_recommendations(
            state, self.spec, np.random.default_rng(101)
        )
        self.assertEqual(len(cond_candidates), len(uncond_candidates))
        for cond, uncond in zip(cond_candidates, uncond_candidates, strict=True):
            np.testing.assert_allclose(cond, uncond)

    def test_primary_predictions_change_only_the_capacity_center(self) -> None:
        uncertainty = replace(
            self.spec.uncertainty,
            capacity_log_sigma=0.05,
            capacity_multiplier_min=0.01,
            capacity_multiplier_max=10.0,
        )
        spec = replace(self.spec, uncertainty=uncertainty)
        state = initial_state(spec)
        state.path_capacity_estimate[:] = 0.80
        state.exit_capacity_estimate[:] = 0.90
        conditioned = ConditionedMeanPolicy()._prediction_scenarios(
            state, spec, np.random.default_rng(7301)
        )
        unconditioned = UnconditionedMeanPolicy()._prediction_scenarios(
            state, spec, np.random.default_rng(7301)
        )
        for cond, uncond in zip(conditioned, unconditioned, strict=True):
            np.testing.assert_allclose(
                cond.path_capacity_multiplier / state.path_capacity_estimate,
                uncond.path_capacity_multiplier,
            )
            np.testing.assert_allclose(
                cond.exit_capacity_multiplier / state.exit_capacity_estimate,
                uncond.exit_capacity_multiplier,
            )
            np.testing.assert_allclose(cond.compliance, uncond.compliance)

    def test_decision_substreams_keep_scenario_draws_aligned_under_sensor_noise(self) -> None:
        uncertainty = replace(
            self.spec.uncertainty,
            capacity_log_sigma=0.05,
            capacity_multiplier_min=0.01,
            capacity_multiplier_max=10.0,
        )
        small_control = replace(
            self.spec.robust_control,
            prediction_scenarios=3,
            candidate_count=6,
            rollout_steps=1,
        )
        spec = replace(
            self.spec,
            uncertainty=uncertainty,
            robust_control=small_control,
        )

        class CapturingPolicy(ConditionedMeanPolicy):
            def _decision_state(self, state, spec, rng):
                observed = super()._decision_state(state, spec, rng)
                self.captured_path_center = observed.path_capacity_estimate.copy()
                self.captured_exit_center = observed.exit_capacity_estimate.copy()
                return observed

            def _prediction_scenarios(self, state, spec, rng):
                scenarios = super()._prediction_scenarios(state, spec, rng)
                self.captured_scenarios = scenarios
                return scenarios

        clean = CapturingPolicy()
        noisy = CapturingPolicy(sensor_noise_sigma=0.15)
        clean.decide(initial_state(spec), spec, np.random.default_rng(4401))
        noisy.decide(initial_state(spec), spec, np.random.default_rng(4401))

        for clean_scenario, noisy_scenario in zip(
            clean.captured_scenarios, noisy.captured_scenarios, strict=True
        ):
            np.testing.assert_allclose(
                clean_scenario.path_capacity_multiplier
                / clean.captured_path_center,
                noisy_scenario.path_capacity_multiplier
                / noisy.captured_path_center,
            )
            np.testing.assert_allclose(
                clean_scenario.exit_capacity_multiplier
                / clean.captured_exit_center,
                noisy_scenario.exit_capacity_multiplier
                / noisy.captured_exit_center,
            )
            np.testing.assert_allclose(
                clean_scenario.compliance, noisy_scenario.compliance
            )

    def test_decision_substreams_keep_scenarios_aligned_for_different_states(self) -> None:
        small_control = replace(
            self.spec.robust_control,
            prediction_scenarios=3,
            candidate_count=10,
            rollout_steps=1,
        )
        spec = replace(self.spec, robust_control=small_control)

        class CapturingPolicy(ConditionedMeanPolicy):
            def _prediction_scenarios(self, state, spec, rng):
                scenarios = super()._prediction_scenarios(state, spec, rng)
                self.captured_scenarios = scenarios
                return scenarios

        state_a = initial_state(spec)
        state_b = initial_state(spec)
        state_b.path_queue[:] = np.arange(state_b.path_queue.size).reshape(
            state_b.path_queue.shape
        ) * 25.0
        state_b.exit_queue[:] = np.array([300.0, 50.0, 600.0])
        first = CapturingPolicy()
        second = CapturingPolicy()
        first.decide(state_a, spec, np.random.default_rng(9917))
        second.decide(state_b, spec, np.random.default_rng(9917))
        for left, right in zip(
            first.captured_scenarios, second.captured_scenarios, strict=True
        ):
            np.testing.assert_allclose(
                left.path_capacity_multiplier, right.path_capacity_multiplier
            )
            np.testing.assert_allclose(
                left.exit_capacity_multiplier, right.exit_capacity_multiplier
            )
            np.testing.assert_allclose(left.compliance, right.compliance)

    def test_full_matched_policies_are_identical_without_capacity_uncertainty(self) -> None:
        small_control = replace(
            self.spec.robust_control,
            prediction_scenarios=2,
            candidate_count=6,
            rollout_steps=3,
        )
        zero_uncertainty = replace(
            self.spec.uncertainty,
            capacity_log_sigma=0.0,
        )
        spec = replace(
            self.spec,
            robust_control=small_control,
            uncertainty=zero_uncertainty,
        )
        conditioned = run_simulation(spec, ConditionedMeanPolicy(), seed=8123)
        unconditioned = run_simulation(spec, UnconditionedMeanPolicy(), seed=8123)
        for field in (
            "clearance_time_seconds",
            "mean_clearance_time_seconds",
            "peak_density_per_m2",
            "recommendation_changes",
            "capacity_mae",
            "capacity_observed_fraction",
        ):
            self.assertAlmostEqual(
                float(getattr(conditioned, field)),
                float(getattr(unconditioned, field)),
                places=10,
            )

    def test_primary_predictions_are_identical_at_nominal_center(self) -> None:
        state = initial_state(self.spec)
        conditioned = ConditionedMeanPolicy()._prediction_scenarios(
            state, self.spec, np.random.default_rng(811)
        )
        unconditioned = UnconditionedMeanPolicy()._prediction_scenarios(
            state, self.spec, np.random.default_rng(811)
        )
        for cond, uncond in zip(conditioned, unconditioned, strict=True):
            np.testing.assert_allclose(
                cond.path_capacity_multiplier,
                uncond.path_capacity_multiplier,
            )
            np.testing.assert_allclose(
                cond.exit_capacity_multiplier,
                uncond.exit_capacity_multiplier,
            )
            np.testing.assert_allclose(cond.compliance, uncond.compliance)

    def test_saturated_observation_counts_are_recorded(self) -> None:
        state = initial_state(self.spec)
        state.path_queue[:] = 1000.0
        state.exit_queue[:] = 1000.0
        scenario = nominal_scenario(self.spec)
        advance_state(state, self.spec, self.spec.habitual_matrix, scenario)
        self.assertTrue(np.all(state.path_saturated_observation_count == 1))
        self.assertTrue(np.all(state.exit_saturated_observation_count == 1))

    def test_time_step_refinement_preserves_physical_controller_scales(self) -> None:
        base_decay = 1.0 - self.spec.robust_control.estimator_alpha
        for dt in (10.0, 5.0, 2.5):
            refined = spec_with_time_step(self.spec, dt)
            self.assertAlmostEqual(
                refined.control_interval_steps * dt, 120.0, places=10
            )
            self.assertAlmostEqual(
                refined.robust_control.rollout_steps * dt, 300.0, places=10
            )
            refined_decay_over_base_step = (
                1.0 - refined.robust_control.estimator_alpha
            ) ** (self.spec.time_step_seconds / dt)
            self.assertAlmostEqual(
                refined_decay_over_base_step, base_decay, places=12
            )

    def test_topology_configurations_are_shape_valid(self) -> None:
        for name in (
            "topology_symmetric.json",
            "topology_path_bottleneck.json",
            "topology_exit_bottleneck.json",
        ):
            spec = load_spec(ROOT / "config" / name)
            self.assertEqual(spec.path_capacity.shape, (spec.zone_count, spec.exit_count))
            self.assertEqual(spec.exit_capacity.shape, (spec.exit_count,))
            self.assertAlmostEqual(spec.total_population, 5000.0)

    def test_conditioning_only_changes_capacity_prediction_center(self) -> None:
        state = initial_state(self.spec)
        conditioned = ConditionedMeanPolicy()
        unconditioned = UnconditionedMeanPolicy()
        cond_scenarios = conditioned._prediction_scenarios(
            state, self.spec, np.random.default_rng(7)
        )
        uncond_scenarios = unconditioned._prediction_scenarios(
            state, self.spec, np.random.default_rng(7)
        )
        # At the nominal center the scenario arrays are identical, proving
        # common-random-number alignment and an equal pre-clipping log-spread parameter.
        for cond, uncond in zip(cond_scenarios, uncond_scenarios, strict=True):
            np.testing.assert_allclose(
                cond.path_capacity_multiplier,
                uncond.path_capacity_multiplier,
            )
            np.testing.assert_allclose(
                cond.exit_capacity_multiplier,
                uncond.exit_capacity_multiplier,
            )
            np.testing.assert_allclose(cond.compliance, uncond.compliance)

        state.path_capacity_estimate[:] = 0.80
        state.exit_capacity_estimate[:] = 0.85
        shifted = conditioned._prediction_scenarios(
            state, self.spec, np.random.default_rng(7)
        )
        self.assertFalse(
            np.allclose(
                shifted[0].path_capacity_multiplier,
                uncond_scenarios[0].path_capacity_multiplier,
            )
        )

    def test_mean_and_cvar_variants_differ_only_by_risk_weight(self) -> None:
        self.assertEqual(ScenarioMeanPolicy().risk_weight_override, 0.0)
        self.assertIsNone(RobustRollingHorizonPolicy().risk_weight_override)
        self.assertEqual(UnconditionedMeanPolicy().risk_weight_override, 0.0)
        self.assertIsNone(UnconditionedRobustPolicy().risk_weight_override)

    def test_build_policy_includes_matched_information_variants(self) -> None:
        expected = {
            "conditioned_mean_equal_spread": ConditionedMeanPolicy,
            "conditioned_mean_contracted_spread": ContractedSpreadMeanPolicy,
            "unconditioned_mean_only": UnconditionedMeanPolicy,
            "scenario_mean_only": ScenarioMeanPolicy,
            "robust_without_sensor_conditioning": UnconditionedRobustPolicy,
            "robust_rolling_horizon": RobustRollingHorizonPolicy,
            "oracle_capacity_mean_only": OracleCapacityMeanPolicy,
        }
        for kind, cls in expected.items():
            self.assertIsInstance(build_policy(kind), cls)

    def test_oracle_capacity_policy_runs_as_diagnostic(self) -> None:
        result = run_simulation(self.spec, OracleCapacityPolicy(), seed=31)
        self.assertTrue(result.completed)
        self.assertEqual(result.policy, "oracle_capacity_rolling_horizon")
        self.assertLess(result.mass_balance_error, 1e-7)

    def test_oracle_capacity_scenarios_preserve_matched_compliance_draws(self) -> None:
        state = initial_state(self.spec)
        realized = nominal_scenario(self.spec)
        realized.path_capacity_multiplier[:] = 0.80
        realized.exit_capacity_multiplier[:] = 0.90

        oracle = OracleCapacityMeanPolicy()
        oracle.set_oracle_scenario(realized)
        oracle_scenarios = oracle._prediction_scenarios(
            state, self.spec, np.random.default_rng(771)
        )

        state.path_capacity_estimate[:] = 0.80
        state.exit_capacity_estimate[:] = 0.90
        conditioned = ConditionedMeanPolicy(assumed_residual_sigma=0.0)
        conditioned_scenarios = conditioned._prediction_scenarios(
            state, self.spec, np.random.default_rng(771)
        )
        unconditioned = UnconditionedMeanPolicy()._prediction_scenarios(
            state, self.spec, np.random.default_rng(771)
        )
        for oracle_scenario, conditioned_scenario, unconditioned_scenario in zip(
            oracle_scenarios,
            conditioned_scenarios,
            unconditioned,
            strict=True,
        ):
            np.testing.assert_allclose(
                oracle_scenario.path_capacity_multiplier,
                conditioned_scenario.path_capacity_multiplier,
            )
            np.testing.assert_allclose(
                oracle_scenario.exit_capacity_multiplier,
                conditioned_scenario.exit_capacity_multiplier,
            )
            np.testing.assert_allclose(
                oracle_scenario.compliance, conditioned_scenario.compliance
            )
            np.testing.assert_allclose(
                oracle_scenario.compliance, unconditioned_scenario.compliance
            )

    def test_oracle_capacity_mean_policy_runs_as_diagnostic(self) -> None:
        result = run_simulation(self.spec, OracleCapacityMeanPolicy(), seed=32)
        self.assertTrue(result.completed)
        self.assertEqual(result.policy, "oracle_capacity_mean_only")
        self.assertLess(result.mass_balance_error, 1e-7)

    def test_nonstationary_scope_and_profile_options_run(self) -> None:
        for scope, profile in (("path_only", "gradual"), ("exit_only", "recovery")):
            result = run_simulation(
                self.spec,
                DeterministicLoadBalancePolicy(),
                seed=37,
                temporary_exit_index=0,
                temporary_capacity_factor=0.70,
                temporary_start_seconds=240.0,
                temporary_end_seconds=360.0,
                temporary_scope=scope,
                temporary_profile=profile,
            )
            self.assertTrue(result.completed)
            self.assertLess(result.mass_balance_error, 1e-7)

    def test_change_triggered_policy_falls_back_after_sharp_drop(self) -> None:
        state = initial_state(self.spec)
        state.elapsed_seconds = 360.0
        state.path_capacity_estimate_history[:] = 1.0
        state.exit_capacity_estimate_history[:] = 1.0
        state.path_capacity_estimate[:, 0] = 0.70
        state.exit_capacity_estimate[0] = 0.70
        state.path_capacity_estimate_history[-1, :, 0] = 0.70
        state.exit_capacity_estimate_history[-1, 0] = 0.70
        policy = ChangeTriggeredFallbackPolicy()
        rng = np.random.default_rng(17)
        recommendation = policy.decide(state, self.spec, rng)
        reactive = DeterministicLoadBalancePolicy().decide(
            state, self.spec, np.random.default_rng(17)
        )
        np.testing.assert_allclose(recommendation, reactive)
        self.assertEqual(policy.fallback_decisions, 1)

    def test_change_triggered_policy_ignores_stationary_estimate(self) -> None:
        state = initial_state(self.spec)
        state.elapsed_seconds = 360.0
        state.path_capacity_estimate[:] = 0.70
        state.exit_capacity_estimate[:] = 0.75
        state.path_capacity_estimate_history[:] = 0.70
        state.exit_capacity_estimate_history[:] = 0.75
        policy = ChangeTriggeredFallbackPolicy()
        small_control = replace(
            self.spec.robust_control,
            prediction_scenarios=2,
            candidate_count=5,
            rollout_steps=2,
        )
        spec = replace(self.spec, robust_control=small_control)
        recommendation = policy.decide(state, spec, np.random.default_rng(19))
        np.testing.assert_allclose(recommendation.sum(axis=1), 1.0)
        self.assertEqual(policy.fallback_decisions, 0)

    def test_simulation_is_reproducible(self) -> None:
        first = run_simulation(self.spec, DeterministicLoadBalancePolicy(), seed=11)
        second = run_simulation(self.spec, DeterministicLoadBalancePolicy(), seed=11)
        self.assertEqual(first.summary(), second.summary())

    def test_scenario_and_policy_seeds_are_recorded_separately(self) -> None:
        result = run_simulation(
            self.spec,
            DeterministicLoadBalancePolicy(),
            seed=101,
            policy_seed=202,
        )
        self.assertEqual(result.scenario_seed, 101)
        self.assertEqual(result.policy_seed, 202)
        self.assertEqual(result.seed, result.scenario_seed)

    def test_uncertainty_overrides_update_policy_visible_spec(self) -> None:
        run_spec = spec_with_uncertainty_overrides(
            self.spec,
            capacity_log_sigma=0.45,
            compliance_mean=0.85,
        )
        self.assertAlmostEqual(run_spec.uncertainty.capacity_log_sigma, 0.45)
        self.assertAlmostEqual(run_spec.uncertainty.compliance_mean, 0.85)
        self.assertAlmostEqual(self.spec.uncertainty.capacity_log_sigma, 0.22)
        self.assertAlmostEqual(self.spec.uncertainty.compliance_mean, 0.68)

    def test_capacity_history_retains_lagged_estimates(self) -> None:
        state = initial_state(self.spec)
        state.path_queue[:] = 1000.0
        state.exit_queue[:] = 1000.0
        scenario = nominal_scenario(self.spec)
        scenario.path_capacity_multiplier[:] = 0.5
        scenario.exit_capacity_multiplier[:] = 0.6
        advance_state(state, self.spec, self.spec.habitual_matrix, scenario)
        np.testing.assert_allclose(
            state.path_capacity_estimate_history[-1], state.path_capacity_estimate
        )
        np.testing.assert_allclose(state.path_capacity_estimate_history[-2], 1.0)

    def test_static_simulation_finishes_and_conserves_mass(self) -> None:
        result = run_simulation(self.spec, StaticNearestPolicy(), seed=2)
        self.assertTrue(result.completed)
        self.assertLess(result.mass_balance_error, 1e-7)
        self.assertGreater(result.clearance_time_seconds, 0.0)
        self.assertGreater(result.peak_density_per_m2, 0.0)

    def test_saturated_flow_updates_capacity_estimate(self) -> None:
        state = initial_state(self.spec)
        state.path_queue[:] = 1000.0
        state.exit_queue[:] = 1000.0
        scenario = nominal_scenario(self.spec)
        scenario.path_capacity_multiplier[:] = 0.5
        scenario.exit_capacity_multiplier[:] = 0.6
        advance_state(state, self.spec, self.spec.habitual_matrix, scenario)
        expected_path = 0.7 * 1.0 + 0.3 * 0.5
        expected_exit = 0.7 * 1.0 + 0.3 * 0.6
        np.testing.assert_allclose(state.path_capacity_estimate, expected_path)
        np.testing.assert_allclose(state.exit_capacity_estimate, expected_exit)
        self.assertTrue(np.all(state.path_saturated_observation_count == 1))
        self.assertTrue(np.all(state.exit_saturated_observation_count == 1))

    def test_all_topology_configs_load(self) -> None:
        for config_name in (
            "default.json",
            "topology_symmetric.json",
            "topology_path_bottleneck.json",
            "topology_exit_bottleneck.json",
            "topology_expanded_venue.json",
        ):
            spec = load_spec(ROOT / "config" / config_name)
            self.assertGreaterEqual(spec.zone_count, 4)
            self.assertGreaterEqual(spec.exit_count, 3)
            self.assertAlmostEqual(spec.total_population, 5000.0)


if __name__ == "__main__":
    unittest.main()
