from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egress_sim.analysis import aggregate_results
sys.path.insert(0, str(ROOT / "experiments"))
from _stats import paired_effects


class OutputTests(unittest.TestCase):
    def test_aggregation_reports_tail_and_safety_metrics(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "policy": "a",
                    "seed": 0,
                    "completed": True,
                    "clearance_time_seconds": 100.0,
                    "mean_clearance_time_seconds": 60.0,
                    "peak_density_per_m2": 2.0,
                    "recommendation_changes": 0.0,
                    "mass_balance_error": 0.0,
                },
                {
                    "policy": "a",
                    "seed": 1,
                    "completed": True,
                    "clearance_time_seconds": 120.0,
                    "mean_clearance_time_seconds": 70.0,
                    "peak_density_per_m2": 3.0,
                    "recommendation_changes": 0.0,
                    "mass_balance_error": 1e-12,
                },
            ]
        )
        summary = aggregate_results(frame, ["policy"])
        self.assertAlmostEqual(float(summary.loc[0, "clearance_mean_s"]), 110.0)
        self.assertIn("clearance_p90_s", summary.columns)
        self.assertAlmostEqual(float(summary.loc[0, "peak_density_max"]), 3.0)

    def test_paired_effects_report_median_p90_and_win_rate(self) -> None:
        rows = []
        for seed, baseline, proposed in ((1, 100.0, 90.0), (2, 120.0, 120.0), (3, 140.0, 130.0)):
            rows.extend(
                [
                    {
                        "scenario_seed": seed,
                        "policy": "baseline",
                        "clearance_time_seconds": baseline,
                    },
                    {
                        "scenario_seed": seed,
                        "policy": "proposed",
                        "clearance_time_seconds": proposed,
                    },
                ]
            )
        effects = paired_effects(
            pd.DataFrame(rows), "proposed", "baseline", samples=500, label="unit-test"
        )
        self.assertEqual(effects["paired_runs"], 3)
        self.assertIn("median_difference", effects)
        self.assertIn("p90_difference", effects)
        self.assertIn("loss_rate_percent", effects)
        self.assertIn("paired_difference_q10", effects)
        self.assertIn("paired_difference_q90", effects)
        self.assertAlmostEqual(effects["win_rate_percent"], 200.0 / 3.0)
        self.assertAlmostEqual(effects["loss_rate_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
