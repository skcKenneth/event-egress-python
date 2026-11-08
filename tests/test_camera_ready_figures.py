"""Protect the statistical meaning of camera-ready figure refinements."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer
from matplotlib.text import Text
import numpy as np
import pandas as pd

from camera_ready_common import WIDTH_IN, effect
from camera_ready_fig13 import figure1, figure3
from camera_ready_fig2 import figure2
from camera_ready_fig4 import C, U, figure4
from render_ieee_figures import bootstrap_mean


TABLES = ROOT / "paper_artifacts" / "tables"


class CameraReadyFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = pd.read_csv(TABLES / "information_value_validation_runs.csv")
        cls.effects = pd.read_csv(TABLES / "information_value_validation_effects.csv").set_index("comparison")
        cls.summary = pd.read_csv(ROOT / "revision_artifacts/2026-11/decision_diagnostics_summary.csv")
        cls.fig1, cls.data1 = figure1(TABLES)
        cls.fig2, cls.data2 = figure2(TABLES)
        cls.fig3, cls.data3 = figure3(TABLES)
        cls.fig4, cls.data4 = figure4(cls.summary)

    @classmethod
    def tearDownClass(cls) -> None:
        for fig in (cls.fig1, cls.fig2, cls.fig3, cls.fig4):
            plt.close(fig)

    def test_heatmap_values_and_zero_inclusion_symbols_preserve_paired_effects(self) -> None:
        source = pd.read_csv(TABLES / "information_regime_map_summary.csv")
        text = self.fig1.axes[2].texts
        index = 0
        for i, sigma in enumerate(self.data1["capacity_log_sigma"]):
            for j, rho in enumerate(self.data1["compliance_mean"]):
                row = source[np.isclose(source.capacity_log_sigma, sigma)
                             & np.isclose(source.compliance_mean, rho)].iloc[0]
                self.assertAlmostEqual(self.data1["mean_difference_percent"][i][j], row.mean_difference_percent)
                self.assertAlmostEqual(self.data1["mean_ci95_low_seconds"][i][j], row.mean_ci95_low)
                self.assertAlmostEqual(self.data1["mean_ci95_high_seconds"][i][j], row.mean_ci95_high)
                includes_zero = row.mean_ci95_low <= 0 <= row.mean_ci95_high
                self.assertEqual(self.data1["interval_includes_zero"][i][j], includes_zero)
                self.assertEqual(text[index].get_text().endswith("*"), includes_zero)
                if sigma == 0:
                    self.assertEqual((row.mean_ci95_low, row.mean_ci95_high), (0, 0))
                    self.assertTrue(text[index].get_text().endswith("*"))
                index += 1

    def test_robustness_numeric_positions_and_all_intervals_match_the_frozen_tables(self) -> None:
        np.testing.assert_array_equal(self.fig3.axes[0].get_xticks(), [10, 5, 2.5])
        self.assertEqual([text.get_text() for text in self.fig3.axes[0].get_xticklabels()], ["10", "5", "2.5"])
        self.assertGreater(self.fig3.axes[0].get_xlim()[0], self.fig3.axes[0].get_xlim()[1])
        dt = pd.read_csv(TABLES / "time_step_sensitivity_effects.csv")
        for displayed in self.data3["time_step"]:
            expected = dt[dt.comparison == displayed["comparison"]].sort_values("time_step_seconds")
            for key in ("time_step_seconds", "mean_difference", "mean_ci95_low", "mean_ci95_high"):
                np.testing.assert_allclose(displayed[key], expected[key].to_numpy(), rtol=0, atol=1e-12)
        topology = pd.read_csv(TABLES / "topology_generalization_effects.csv").set_index("topology")
        for displayed in self.data3["topology"]:
            expected = topology.loc[displayed["topology"]]
            for key in ("mean_difference", "mean_ci95_low", "mean_ci95_high"):
                self.assertAlmostEqual(displayed[key], expected[key], places=12)
        faults = pd.read_csv(TABLES / "sensor_condition_effects.csv")
        for displayed in self.data3["estimate_faults"]:
            expected = faults[faults.regime == displayed["regime"]].set_index("fault").loc[displayed["fault"]]
            for key in ("mean_difference", "mean_ci95_low", "mean_ci95_high"):
                np.testing.assert_allclose(displayed[key], expected[key].to_numpy(), rtol=0, atol=1e-12)

    def test_marginal_policy_intervals_and_histogram_remain_unchanged(self) -> None:
        for index, row in enumerate(self.data2["panel_a"]):
            values = self.raw.loc[self.raw.policy == row["policy"], "clearance_time_seconds"].to_numpy()
            expected = bootstrap_mean(values, 100 + index)
            np.testing.assert_allclose([row["mean"], row["ci95_low"], row["ci95_high"]], expected,
                                       rtol=0, atol=1e-12)
        axis = next(ax for ax in self.fig2.axes if ax.get_label() == "fig2a")
        self.assertTrue(all(line.get_linestyle() in ("None", "none", "", None)
                            for line in axis.lines), "Categorical policy means must not be connected")
        pivot = self.raw.pivot(index="scenario_seed", columns="policy", values="clearance_time_seconds")
        differences = (pivot[C] - pivot[U]).to_numpy()
        expected_edges = np.arange(10*np.floor(differences.min()/10)-10,
                                   10*np.ceil(differences.max()/10)+20, 10)
        counts, edges = np.histogram(differences, bins=expected_edges)
        np.testing.assert_array_equal(self.data2["panel_b"]["histogram_counts"], counts)
        np.testing.assert_array_equal(self.data2["panel_b"]["histogram_edges"], edges)
        self.assertFalse(self.data2["panel_b"]["quantile_shading"])
        hist_axis = next(ax for ax in self.fig2.axes if ax.get_label() == "fig2b")
        self.assertEqual(len(hist_axis.patches), len(counts), "No extra quantile-shading patch is allowed")
        np.testing.assert_allclose([patch.get_height() for patch in hist_axis.patches], counts)

    def test_principal_paired_effects_and_fill_semantics_match_frozen_intervals(self) -> None:
        self.assertEqual([row["comparison"] for row in self.data2["panel_c"][:2]],
                         ["primary-sensing-value", "secondary-system-comparison"])
        axis = next(ax for ax in self.fig2.axes if ax.get_label() == "fig2c")
        points = [container.lines[0] for container in axis.containers
                  if isinstance(container, ErrorbarContainer)]
        self.assertEqual(len(points), len(self.data2["panel_c"]))
        for row, point in zip(self.data2["panel_c"], points):
            expected = self.effects.loc[row["comparison"]]
            np.testing.assert_allclose([row["estimate"], row["ci95_low"], row["ci95_high"]],
                                       [expected.mean_difference, expected.mean_ci95_low, expected.mean_ci95_high],
                                       rtol=0, atol=1e-12)
            contains_zero = expected.mean_ci95_low <= 0 <= expected.mean_ci95_high
            self.assertEqual(row["contains_zero"], contains_zero)
            self.assertEqual(to_rgba(point.get_markerfacecolor()) == to_rgba("white"), contains_zero)

    def test_estimator_band_is_the_descriptive_run_quantiles_not_a_confidence_interval(self) -> None:
        source = pd.read_csv(TABLES / "estimator_diagnostics_summary.csv")
        source = source[(source.regime == "default") & (source.time_seconds <= 600)].sort_values("time_seconds")
        for key in ("time_seconds", "capacity_mae_mean", "capacity_mae_q10", "capacity_mae_q90",
                    "capacity_observed_fraction_mean", "runs"):
            np.testing.assert_allclose(self.data2["panel_d"][key], source[key].to_numpy(), rtol=0, atol=1e-12)
        self.assertTrue((source.runs == 50).all())
        axis = next(ax for ax in self.fig2.axes if ax.get_label() == "fig2d")
        bands = [collection for collection in axis.collections if isinstance(collection, PolyCollection)]
        self.assertEqual(len(bands), 1)
        vertices = bands[0].get_paths()[0].vertices
        for row in source.itertuples():
            ys = vertices[np.isclose(vertices[:, 0], row.time_seconds), 1]
            self.assertTrue(np.any(np.isclose(ys, row.capacity_mae_q10)))
            self.assertTrue(np.any(np.isclose(ys, row.capacity_mae_q90)))

    def test_collector_rates_and_action_effects_keep_their_original_sampling_units(self) -> None:
        self.assertFalse(self.data4["u_minus_c_contrast_added"])
        self.assertEqual([row["strict_reversals"] for row in self.data4["panel_a"]], [26, 13])
        for row in self.data4["panel_a"]:
            self.assertEqual(row["completed_decisions"], 90)
            self.assertEqual(row["environmental_seeds"], 30)
            self.assertAlmostEqual(row["estimate"], row["strict_reversals"]/90)
        axis = next(ax for ax in self.fig4.axes if ax.get_label() == "fig4b")
        points = [container.lines[0] for container in axis.containers
                  if isinstance(container, ErrorbarContainer)]
        self.assertEqual(len(points), 4)
        for row, point in zip(self.data4["panel_b"], points):
            source = self.summary[(self.summary.policy == row["policy"])
                                  & (self.summary.snapshot_seconds == -1)
                                  & (self.summary.metric == row["metric"])].iloc[0]
            np.testing.assert_allclose([row["estimate"], row["ci95_low"], row["ci95_high"]],
                                       [source.estimate, source.ci95_low, source.ci95_high], rtol=0, atol=1e-12)
            self.assertTrue(row["contains_zero"])
            self.assertEqual(to_rgba(point.get_markerfacecolor()), to_rgba("white"))
            self.assertEqual(point.get_marker(), "s" if row["continuation"] == "fixed" else "o")
            self.assertEqual(len(point.get_xdata()), 1, "No line joining collector state sets is allowed")

    def test_zero_endpoint_uses_hollow_marker_and_fixed_canvas_keeps_readable_text(self) -> None:
        fig, ax = plt.subplots()
        try:
            effect(ax, .1, 0, .2, 0)
            point = next(c for c in ax.containers if isinstance(c, ErrorbarContainer)).lines[0]
            self.assertEqual(to_rgba(point.get_markerfacecolor()), to_rgba("white"))
        finally:
            plt.close(fig)
        for fig in (self.fig1, self.fig2, self.fig3, self.fig4):
            self.assertAlmostEqual(fig.get_size_inches()[0], WIDTH_IN)
            texts = [item for item in fig.findobj(Text) if item.get_visible() and item.get_text().strip()]
            self.assertTrue(texts)
            for text in texts:
                self.assertGreaterEqual(text.get_fontsize(), 7.0, text.get_text())

    def test_heatmap_annotations_have_readable_foreground_background_contrast(self) -> None:
        axis = self.fig1.axes[2]
        image = axis.images[0]
        def luminance(rgba):
            rgb = np.asarray(rgba[:3], dtype=float)
            linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
            return float(linear @ [.2126, .7152, .0722])
        for text, value in zip(axis.texts, image.get_array().ravel()):
            foreground = luminance(to_rgba(text.get_color()))
            background = luminance(image.cmap(image.norm(value)))
            ratio = (max(foreground, background)+.05)/(min(foreground, background)+.05)
            self.assertGreaterEqual(ratio, 4.5, text.get_text())


if __name__ == "__main__":
    unittest.main()
