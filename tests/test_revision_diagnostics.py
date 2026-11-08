from __future__ import annotations

import copy
import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from egress_sim.config import load_spec
from egress_sim.model import advance_state, current_peak_density, initial_state
from egress_sim.policies import build_policy
from egress_sim.revision_diagnostics import (
    POLICY_KINDS, SUMMARY_METRICS, collect_snapshots, continue_branch,
    equivalent_recommendations, evaluate_snapshot, summarize_decisions,
)
from egress_sim.simulation import run_simulation
from egress_sim.uncertainty import nominal_scenario
from run_surrogate_validation import _surrogate_clearance_minutes


class RevisionDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = load_spec(ROOT / "config" / "default.json")
        cls.spec = replace(base, robust_control=replace(
            base.robust_control, prediction_scenarios=3,
            candidate_count=6, rollout_steps=8,
        ))

    def assertStateEqual(self, left, right):
        for field in fields(left):
            np.testing.assert_array_equal(getattr(left, field.name), getattr(right, field.name))

    def test_trace_preserves_decisions_rng_and_complete_simulations(self):
        for kind in POLICY_KINDS:
            plain, traced = build_policy(kind), build_policy(kind, capture_trace=True)
            left_rng, right_rng = np.random.default_rng(95), np.random.default_rng(95)
            state = initial_state(self.spec)
            for _ in range(3):
                left = plain.decide(state, self.spec, left_rng)
                right = traced.decide(state, self.spec, right_rng)
                np.testing.assert_array_equal(left, right)
                self.assertEqual(left_rng.bit_generator.state, right_rng.bit_generator.state)
                self.assertEqual(plain.last_diagnostics, traced.last_diagnostics)
                for _ in range(12):
                    advance_state(state, self.spec, left, nominal_scenario(self.spec))
            a = run_simulation(self.spec, build_policy(kind), seed=543)
            b = run_simulation(self.spec, build_policy(kind, capture_trace=True), seed=543)
            self.assertEqual(a, b)

    def test_traced_cost_components_match_independent_scalar_rollouts(self):
        state = initial_state(self.spec)
        for _ in range(12):
            advance_state(state, self.spec, state.last_recommendation, nominal_scenario(self.spec))
        policy = build_policy(POLICY_KINDS[1], capture_trace=True)
        policy.decide(state, self.spec, np.random.default_rng(17))
        trace = policy.last_trace
        for i, candidate in enumerate(trace.candidates):
            for j, scenario in enumerate(trace.scenarios):
                expected_clearance = _surrogate_clearance_minutes(state, self.spec, policy, candidate, scenario)
                rollout = state.clone()
                rollout.peak_density = current_peak_density(state, self.spec)
                for _ in range(self.spec.robust_control.rollout_steps):
                    if rollout.active_population <= 1e-8:
                        break
                    advance_state(rollout, self.spec, candidate, scenario)
                expected_delay = (rollout.person_seconds - state.person_seconds) / max(state.active_population, 1.0) / 60
                excess = max(0., rollout.peak_density - self.spec.robust_control.safe_density_per_m2)
                expected_density = self.spec.robust_control.density_weight * excess**2
                self.assertAlmostEqual(trace.clearance_minutes[i, j], expected_clearance, places=11)
                self.assertAlmostEqual(trace.mean_delay_minutes[i, j], expected_delay, places=11)
                self.assertAlmostEqual(trace.density_penalty[i, j], expected_density, places=11)
                self.assertAlmostEqual(trace.scenario_costs[i, j], policy._scenario_cost(state, self.spec, candidate, scenario), places=11)

    def test_snapshots_replay_exact_decision_without_aliasing(self):
        snap = collect_snapshots(self.spec, 77, POLICY_KINDS[1], (120.,))[0]
        before = snap.state.clone()
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(snap.rng_state)
        selected = copy.deepcopy(snap.policy).decide(snap.state.clone(), self.spec, rng)
        np.testing.assert_array_equal(selected, snap.selected_recommendation)
        for mode in ("fixed", "reactive_continuation"):
            left = continue_branch(snap.state, self.spec, snap.scenario, selected, mode)
            right = continue_branch(snap.state, self.spec, snap.scenario, selected.copy(), mode)
            self.assertEqual(left, right)
            self.assertTrue(left.completed)
            self.assertStateEqual(before, snap.state)
        self.assertFalse(np.shares_memory(snap.state.path_capacity_estimate, before.path_capacity_estimate))

    def test_reactive_updates_follow_first_action_and_censoring_uses_absolute_limit(self):
        snap = collect_snapshots(self.spec, 88, POLICY_KINDS[0], (120.,))[0]
        outcome = continue_branch(snap.state, self.spec, snap.scenario, snap.selected_recommendation, "reactive_continuation")
        self.assertTrue(outcome.reactive_update_seconds)
        self.assertEqual(outcome.reactive_update_seconds[0], 240.)
        np.testing.assert_array_equal(np.diff(outcome.reactive_update_seconds), 120.)
        limited = replace(self.spec, maximum_duration_seconds=130.)
        censored = continue_branch(snap.state, limited, snap.scenario, snap.selected_recommendation, "reactive_continuation")
        self.assertFalse(censored.completed)
        self.assertTrue(np.isnan(censored.residual_clearance_seconds))
        self.assertEqual(censored.elapsed_observed_seconds, 10.)
        self.assertGreater(censored.remaining_population, 0.)
        self.assertFalse(censored.reactive_update_seconds)

    def test_cleared_snapshots_are_explicitly_skipped_without_replacement(self):
        tiny = replace(self.spec, population=np.full(self.spec.zone_count, .01))
        snaps = collect_snapshots(tiny, 12, POLICY_KINDS[0], (360., 600.))
        self.assertEqual([s.snapshot_seconds for s in snaps], [360., 600.])
        self.assertTrue(all(not s.eligible and s.skip_reason == "cleared_before_snapshot" for s in snaps))

    def test_equivalent_recommendations_and_action_pair_records(self):
        snap = collect_snapshots(self.spec, 91, POLICY_KINDS[0], (120.,))[0]
        self.assertTrue(equivalent_recommendations(snap.trace.candidates[1], snap.trace.candidates[1].copy()))
        decision, candidates, branches = evaluate_snapshot(snap, self.spec)
        self.assertEqual(len(candidates), 6)
        self.assertEqual(len(branches), 4)
        self.assertLessEqual(decision["objective_gap_minutes"], 1e-12)
        self.assertTrue(all(b["completed"] for b in branches))

    def test_seed_weighting_does_not_count_repeated_decisions_as_seeds(self):
        rows = []
        for seed, n, value in ((1, 3, 0.), (2, 1, 1.)):
            for _ in range(n):
                rows.append({"seed": seed, "policy": "test", "snapshot_seconds": 120.,
                             **{metric: value for metric in SUMMARY_METRICS}})
        per_seed, summary = summarize_decisions(pd.DataFrame(rows), bootstrap_samples=50)
        row = summary[(summary.policy == "all") & (summary.metric == "reactive_selected")].iloc[0]
        self.assertEqual(row.estimate, .5)
        self.assertEqual(row.valid_seeds, 2)
        self.assertEqual(len(per_seed[per_seed.policy == "all"]), 2 * len(SUMMARY_METRICS))


if __name__ == "__main__":
    unittest.main()
