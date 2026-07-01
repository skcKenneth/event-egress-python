from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egress_sim.config import RobustControlSpec, load_spec
from egress_sim.analysis import spec_with_uncertainty_overrides
from egress_sim.model import advance_state, initial_state
from egress_sim.policies import (
    ChangeTriggeredFallbackPolicy,
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

    def test_conditioning_only_changes_capacity_prediction_center(self) -> None:
        state = initial_state(self.spec)
        state.path_capacity_estimate[:] = 0.55
        state.exit_capacity_estimate[:] = 0.60
        conditioned = ScenarioMeanPolicy(assumed_residual_sigma=0.0)
        unconditioned = UnconditionedMeanPolicy()
        cond_scenarios = conditioned._prediction_scenarios(
            state, self.spec, np.random.default_rng(7)
        )
        uncond_scenarios = unconditioned._prediction_scenarios(
            state, self.spec, np.random.default_rng(7)
        )
        np.testing.assert_allclose(cond_scenarios[0].path_capacity_multiplier, 0.55)
        np.testing.assert_allclose(cond_scenarios[0].exit_capacity_multiplier, 0.60)
        self.assertFalse(
            np.allclose(
                cond_scenarios[0].path_capacity_multiplier,
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
        self.assertTrue(np.all(state.path_capacity_estimate < 1.0))
        self.assertTrue(np.all(state.exit_capacity_estimate < 1.0))


if __name__ == "__main__":
    unittest.main()
