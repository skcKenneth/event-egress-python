from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from egress_sim.config import load_spec
from egress_sim.model import initial_state
from egress_sim.policies import ConditionedMeanPolicy
import run_runtime_comparison as runtime


class RuntimeComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_spec(ROOT / "config" / "default.json")
        self.state = initial_state(self.spec)
        self.state.elapsed_seconds = 120.0
        self.state.path_queue[:] = 4.0
        self.state.path_capacity_estimate[:] = 0.8
        self.state.path_capacity_estimate_history[:] = 0.8
        self.snapshot = SimpleNamespace(
            seed=900_001,
            policy_kind="conditioned_mean_equal_spread",
            snapshot_seconds=120.0,
            eligible=True,
            skip_reason="",
            state=self.state,
            policy=ConditionedMeanPolicy(),
            rng_state=deepcopy(np.random.default_rng(1234).bit_generator.state),
        )

    def test_repetitions_restore_state_policy_and_rng_outside_timer(self) -> None:
        observed = []
        events = []

        class MutatingPolicy:
            def decide(self, state, spec, rng):
                events.append("decide")
                observed.append((state.path_queue.copy(), state.path_capacity_estimate.copy(), rng.random()))
                state.path_queue[:] = -99.0
                state.path_capacity_estimate[:] = -99.0
                return spec.habitual_matrix

        def prepare(snapshot, policy_kind):
            events.append("prepare")
            rng = np.random.default_rng()
            rng.bit_generator.state = deepcopy(snapshot.rng_state)
            return snapshot.state.clone(), MutatingPolicy(), rng

        tick = 0
        def timer():
            nonlocal tick
            events.append("timer")
            tick += 1_000_000
            return tick

        with patch.object(runtime, "_prepare_decision", side_effect=prepare):
            frame = runtime.benchmark_snapshots(
                self.spec, [self.snapshot], warmups=1, repetitions=3, timer_ns=timer
            )
        self.assertEqual(len(observed), 12)
        self.assertEqual(len(frame), 9)
        self.assertTrue((frame["elapsed_ms"] == 1.0).all())
        for queue, estimate, draw in observed:
            np.testing.assert_array_equal(queue, self.state.path_queue)
            np.testing.assert_array_equal(estimate, self.state.path_capacity_estimate)
            self.assertEqual(draw, observed[0][2])
        self.assertTrue(np.all(self.state.path_queue == 4.0))
        timed_events = events[6:]
        self.assertEqual(timed_events, ["prepare", "timer", "decide", "timer"] * 9)
        for policy in runtime.POLICY_KINDS:
            subset = frame.loc[frame["policy"] == policy]
            self.assertEqual(set(subset["order_position"]), {0, 1, 2})

    def test_policy_preparation_preserves_cache_but_disables_trace(self) -> None:
        self.snapshot.policy.capture_trace = True
        self.snapshot.policy._last_path_observation = np.full_like(self.spec.path_capacity, 0.7)
        for kind in runtime.POLICY_KINDS:
            state, policy, rng = runtime._prepare_decision(self.snapshot, kind)
            np.testing.assert_array_equal(state.path_capacity_estimate_history, self.state.path_capacity_estimate_history)
            self.assertFalse(getattr(policy, "capture_trace", False))
            self.assertIsNot(state.path_queue, self.state.path_queue)
            self.assertEqual(rng.bit_generator.state, self.snapshot.rng_state)
            if hasattr(policy, "_last_path_observation"):
                np.testing.assert_array_equal(policy._last_path_observation, self.snapshot.policy._last_path_observation)
                self.assertIsNot(policy._last_path_observation, self.snapshot.policy._last_path_observation)
        self.assertTrue(self.snapshot.policy.capture_trace)

    def test_skipped_snapshots_are_not_timed_and_summary_retains_phase(self) -> None:
        skipped = SimpleNamespace(eligible=False)
        with patch.object(runtime, "_prepare_decision") as prepare:
            frame = runtime.benchmark_snapshots(self.spec, [skipped], warmups=0, repetitions=1)
        prepare.assert_not_called()
        self.assertTrue(frame.empty)
        self.assertTrue(runtime.summarize_timings(frame, 120.0).empty)

        ticks = iter(range(0, 12_000_000, 1_000_000))
        frame = runtime.benchmark_snapshots(
            self.spec, [self.snapshot], warmups=0, repetitions=2,
            timer_ns=lambda: next(ticks),
        )
        summary = runtime.summarize_timings(frame, 120.0)
        self.assertEqual(set(summary["phase"]), {"overall", "early"})
        self.assertTrue((summary["snapshot_count"] == 1).all())
        self.assertTrue((summary["timed_decisions"] == 2).all())
        self.assertTrue((summary["median_ms"] == 1.0).all())
        self.assertTrue((summary["p95_ms"] == 1.0).all())

    def test_invalid_repetitions_are_rejected(self) -> None:
        for warmups, repetitions in ((-1, 1), (0, 0)):
            with self.assertRaises(ValueError):
                runtime.benchmark_snapshots(self.spec, [], warmups=warmups, repetitions=repetitions)


if __name__ == "__main__":
    unittest.main()
