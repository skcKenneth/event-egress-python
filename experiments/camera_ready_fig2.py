"""Camera-ready presentation of the unchanged primary evidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from camera_ready_common import (
    WIDTH_IN, BLUE, BLUE_LIGHT, ORANGE, INK, GRID, NEUTRAL,
    use_style, despine, zero, effect,
)
from render_ieee_figures import bootstrap_mean


POLICIES = (
    ("static_nearest", "Static"),
    ("deterministic_load_balance", "Reactive"),
    ("unconditioned_mean_only", "Unconditioned"),
    ("conditioned_mean_equal_spread", "Conditioned"),
    ("oracle_capacity_mean_only", "Oracle"),
)
COMPARISONS = (
    ("primary-sensing-value", "C − U"),
    ("secondary-system-comparison", "C − Reactive"),
    ("planning-vs-reactive", "U − Reactive"),
    ("oracle-residual-information", "Oracle − C"),
)


def _panel_heading(fig: plt.Figure, x: float, y: float,
                   letter: str, text: str) -> None:
    fig.text(x, y, f"({letter}) {text}", fontsize=8.5, fontweight="bold",
             color=INK, ha="left", va="top")


def figure2(tables: Path) -> tuple[plt.Figure, dict]:
    """Draw four panels, retaining every original statistic and sampling rule.

    The returned data describe the actual plotted numbers for independent
    parity checks. Policy-mean intervals are marginal bootstrap intervals;
    panel (c) reads the pre-existing paired-effect intervals directly.
    """
    use_style()
    tables = Path(tables)
    raw = pd.read_csv(tables / "information_value_validation_runs.csv")
    effects = pd.read_csv(tables / "information_value_validation_effects.csv").set_index("comparison")
    fig = plt.figure(figsize=(WIDTH_IN, 3.60))
    # Panel envelopes are approximately 45:55 in the top row and 60:40 below.
    # Internal margins reserve full-size category labels and right-hand axes.
    a = fig.add_axes((.155, .665, .270, .250), label="fig2a")
    b = fig.add_axes((.575, .665, .405, .250), label="fig2b")
    c = fig.add_axes((.155, .140, .405, .315), label="fig2c")
    d = fig.add_axes((.725, .140, .205, .260), label="fig2d")
    _panel_heading(fig, .020, .982, "a", "Policy means")
    _panel_heading(fig, .535, .982, "b", "Paired clearance differences")
    _panel_heading(fig, .020, .550, "c", "Matched effects (C: conditioned; U: unconditioned)")
    _panel_heading(fig, .650, .550, "d", "Estimator diagnostics")

    policy_data = []
    for index, (policy, label) in enumerate(POLICIES):
        values = raw.loc[raw.policy == policy, "clearance_time_seconds"].to_numpy()
        mean, low, high = bootstrap_mean(values, 100 + index)
        y = 4 - index
        color = BLUE if policy == "conditioned_mean_equal_spread" else NEUTRAL
        a.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt="o",
                   linestyle="none", color=color, markerfacecolor=color,
                   markeredgecolor=color, capsize=2.8, markersize=4.8,
                   markeredgewidth=1.1, linewidth=1.0, zorder=3)
        a.text(1.035, y, f"{mean:.2f}", transform=a.get_yaxis_transform(),
               ha="left", va="center", fontsize=8.0,
               color=BLUE if policy == "conditioned_mean_equal_spread" else INK,
               fontweight="bold" if policy == "conditioned_mean_equal_spread" else "normal")
        policy_data.append(dict(policy=policy, label=label, mean=mean,
                                ci95_low=low, ci95_high=high,
                                bootstrap_seed=100+index))
    a.set_yticks(range(4, -1, -1), [label for _, label in POLICIES])
    a.set_ylim(-.55, 4.55)
    a.set_xlim(650, 1040)
    a.set_xticks((700, 850, 1000))
    a.set_xlabel("Mean clearance (s)", labelpad=3)
    a.grid(axis="x", color=GRID, linewidth=.45)
    despine(a)

    pivot = raw.pivot(index="scenario_seed", columns="policy", values="clearance_time_seconds")
    differences = (pivot["conditioned_mean_equal_spread"] - pivot["unconditioned_mean_only"]).to_numpy()
    low_edge = 10.0*np.floor(differences.min()/10.0)-10.0
    high_edge = 10.0*np.ceil(differences.max()/10.0)+10.0
    edges = np.arange(low_edge, high_edge+10.0, 10.0)
    counts, _, _ = b.hist(differences, bins=edges, color=BLUE_LIGHT,
                          edgecolor=BLUE, linewidth=.65)
    mean_difference = float(differences.mean())
    wins = float(np.mean(differences < 0)*100)
    ties = float(np.mean(differences == 0)*100)
    losses = float(np.mean(differences > 0)*100)
    zero(b)
    b.axvline(mean_difference, color=BLUE, linewidth=1.3)
    b.text(.025, .95, f"Mean {mean_difference:.2f} s", transform=b.transAxes,
           va="top", ha="left", fontsize=8.0, color=BLUE, fontweight="bold")
    b.text(.025, .72, f"Faster / tied / slower\n{wins:g}% / {ties:g}% / {losses:g}%",
           transform=b.transAxes, va="top", ha="left", fontsize=7.2,
           linespacing=1.05, color=INK)
    b.set_xlabel("Paired C − U clearance (s)", labelpad=3)
    b.set_ylabel("Seed count", labelpad=3)
    b.set_xticks((-200, -100, 0, 50))
    b.set_yticks((0, 40, 80))
    b.grid(axis="y", color=GRID, linewidth=.45)
    despine(b)

    contrast_data = []
    for index, (key, label) in enumerate(COMPARISONS):
        row = effects.loc[key]
        estimate, low, high = (float(row.mean_difference),
                               float(row.mean_ci95_low), float(row.mean_ci95_high))
        y = 3-index
        color = BLUE if index < 2 else NEUTRAL
        effect(c, estimate, low, high, y, color=color,
               marker="o" if index < 2 else "s")
        # A separate line of values keeps the full intervals legible without
        # using the location of the whiskers as a text-placement constraint.
        c.text(.985, y+.27, f"{estimate:+.2f} [{low:+.2f}, {high:+.2f}] s",
               transform=c.get_yaxis_transform(), ha="right", va="bottom",
               fontsize=8.0 if index < 2 else 7.2,
               color=BLUE if index < 2 else "#4A5260",
               bbox={"facecolor": "white", "edgecolor": "none", "pad": .2},
               fontweight="bold" if index < 2 else "normal")
        contrast_data.append(dict(comparison=key, label=label,
                                  estimate=estimate, ci95_low=low, ci95_high=high,
                                  contains_zero=bool(low <= 0 <= high), primary=index < 2))
    c.set_yticks(range(3, -1, -1), [label for _, label in COMPARISONS])
    c.set_ylim(-.5, 3.9)
    c.set_xlim(-25, 25)
    c.set_xticks((-20, 0, 20))
    c.set_xlabel("Paired clearance difference (s)", labelpad=3)
    zero(c)
    c.grid(axis="x", color=GRID, linewidth=.45)
    despine(c)

    summary = pd.read_csv(tables / "estimator_diagnostics_summary.csv")
    subset = summary[(summary.regime == "default") & (summary.time_seconds <= 600)].sort_values("time_seconds")
    time = subset.time_seconds.to_numpy(dtype=float)
    mean = subset.capacity_mae_mean.to_numpy(dtype=float)
    q10 = subset.capacity_mae_q10.to_numpy(dtype=float)
    q90 = subset.capacity_mae_q90.to_numpy(dtype=float)
    observed = subset.capacity_observed_fraction_mean.to_numpy(dtype=float)
    d.fill_between(time, q10, q90, color=BLUE_LIGHT, alpha=.75, linewidth=0)
    d.plot(time, mean, color=BLUE, linewidth=1.2)
    d.set_xlim(0, 600)
    d.set_ylim(0, .21)
    d.set_xticks((0, 300, 600))
    d.set_yticks((0, .1, .2))
    d.set_xlabel("Elapsed time (s)", labelpad=3)
    d.set_ylabel("Capacity MAE", labelpad=3)
    d.grid(color=GRID, linewidth=.45)
    observed_ax = d.twinx()
    observed_ax.set_label("fig2d_observed")
    observed_ax.plot(time, 100*observed, color=ORANGE, linestyle="--", linewidth=1.1)
    observed_ax.set_ylim(0, 105)
    observed_ax.set_yticks((0, 50, 100))
    observed_ax.set_ylabel("Observed (%)", labelpad=3)
    observed_ax.spines["top"].set_visible(False)
    d.spines["top"].set_visible(False)
    d.legend(handles=[Line2D([], [], color=BLUE, label="Mean MAE"),
                       Patch(facecolor=BLUE_LIGHT, edgecolor="none", alpha=.75,
                             label="MAE 10th–90th pct."),
                       Line2D([], [], color=ORANGE, linestyle="--", label="Ever observed links")],
             frameon=False, fontsize=7.2, loc="lower left", bbox_to_anchor=(-.13, 1.02),
             borderaxespad=0, labelspacing=.15, handlelength=1.6, handletextpad=.5)

    plot_data = {
        "panel_a": policy_data,
        "panel_b": {"scenario_seeds": pivot.index.tolist(),
                    "paired_differences": differences.tolist(),
                    "histogram_edges": edges.tolist(), "histogram_counts": counts.tolist(),
                    "mean": mean_difference, "win_rate_percent": wins,
                    "tie_rate_percent": ties, "loss_rate_percent": losses,
                    "quantile_shading": False},
        "panel_c": contrast_data,
        "panel_d": {"time_seconds": time.tolist(), "capacity_mae_mean": mean.tolist(),
                    "capacity_mae_q10": q10.tolist(), "capacity_mae_q90": q90.tolist(),
                    "capacity_observed_fraction_mean": observed.tolist(),
                    "runs": subset.runs.astype(int).tolist(),
                    "shade_definition": "10th–90th percentiles of run-level MAE across 50 seeds; descriptive spread, not a confidence band"},
    }
    return fig, plot_data
