"""Same-state diagnostics with explicit collector and continuation encodings."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from camera_ready_common import (
    BLUE, ORANGE, INK, WIDTH_IN, despine, effect, title, use_style, zero,
)

U = "unconditioned_mean_only"
C = "conditioned_mean_equal_spread"


def _row(summary: pd.DataFrame, policy: str, metric: str) -> pd.Series:
    selected = summary[(summary.policy == policy)
                       & (summary.snapshot_seconds == -1)
                       & (summary.metric == metric)]
    if len(selected) != 1:
        raise ValueError(f"Expected one overall diagnostic row: {policy}, {metric}")
    row = selected.iloc[0]
    numbers = np.asarray([row.estimate, row.ci95_low, row.ci95_high], dtype=float)
    if not np.all(np.isfinite(numbers)) or not numbers[1] <= numbers[0] <= numbers[2]:
        raise ValueError(f"Invalid diagnostic interval: {policy}, {metric}")
    return row


def figure4(summary: pd.DataFrame) -> tuple[plt.Figure, dict]:
    """Return the refined figure and the unchanged numeric inputs it displays.

    Each collector's summaries describe its own trajectory states. No U-minus-C
    contrast or connecting line is introduced. Action-effect markers encode
    continuation by shape and inclusion of zero by fill.
    """
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.20))
    ax_rate, ax_effect = axes
    ax_rate.set_label("fig4a")
    ax_effect.set_label("fig4b")
    colors = {U: ORANGE, C: BLUE}
    labels = {U: "Unconditioned", C: "Conditioned"}
    rate_data = []
    effect_data = []
    for index, policy in enumerate((U, C)):
        row = _row(summary, policy, "clearance_rank_disagreement")
        estimate, low, high = (float(row.estimate), float(row.ci95_low), float(row.ci95_high))
        effect(ax_rate, estimate * 100, low * 100, high * 100,
               index, color=colors[policy], horizontal=False)
        ax_rate.text(index, high * 100 + 2.4, f"{estimate * 100:.1f}%",
                     ha="center", va="bottom", fontsize=8.0, color=INK)
        rate_data.append({
            "policy": policy, "estimate": estimate, "ci95_low": low,
            "ci95_high": high, "display_percent": estimate * 100,
            "completed_decisions": int(row.valid_decisions),
            "environmental_seeds": int(row.valid_seeds),
            "strict_reversals": int(round(estimate * int(row.valid_decisions))),
        })
        for offset, metric, mode, marker in (
            (-.10, "fixed_difference_seconds", "fixed", "s"),
            (.10, "reactive_continuation_difference_seconds", "reactive_continuation", "o"),
        ):
            row = _row(summary, policy, metric)
            estimate, low, high = (float(row.estimate), float(row.ci95_low), float(row.ci95_high))
            effect(ax_effect, estimate, low, high, index + offset,
                   color=colors[policy], marker=marker, horizontal=False)
            effect_data.append({
                "policy": policy, "metric": metric, "continuation": mode,
                "estimate": estimate, "ci95_low": low, "ci95_high": high,
                "contains_zero": low <= 0 <= high, "marker": marker,
                "completed_decisions": int(row.valid_decisions),
                "environmental_seeds": int(row.valid_seeds),
            })

    title(ax_rate, "a", "Fixed-action rank reversals")
    ax_rate.set_ylabel("Completed decisions (%)")
    ax_rate.set_ylim(0, 46)
    ax_rate.set_yticks([0, 10, 20, 30, 40])
    title(ax_effect, "b", "Same-state action effects")
    ax_effect.set_ylabel("Residual clearance difference (s)")
    zero(ax_effect, horizontal=True)
    ax_effect.set_ylim(-10.3, 10.2)
    ax_effect.set_yticks([-8, -4, 0, 4, 8])
    ax_effect.legend(handles=[
        Line2D([], [], color=INK, marker="s", markerfacecolor="white",
               linestyle="none", label="Fixed to clearance"),
        Line2D([], [], color=INK, marker="o", markerfacecolor="white",
               linestyle="none", label="120 s, then reactive"),
    ], frameon=False, fontsize=7.2, loc="lower left", handlelength=1.0,
        borderaxespad=.3, labelspacing=.35)
    for ax in axes:
        ax.set_xticks([0, 1], [labels[U], labels[C]])
        ax.set_xlim(-.50, 1.50)
        ax.grid(axis="y")
        despine(ax)
    fig.subplots_adjust(left=.075, right=.985, bottom=.19, top=.84, wspace=.35)
    data = {
        "panel_a": rate_data, "panel_b": effect_data,
        "collector_state_sets": "different on-policy state sets",
        "u_minus_c_contrast_added": False,
        "rate_denominator": "all completed decisions per collector; ties are not reversals",
        "fill_rule": "clearance-effect markers are hollow if the 95% interval includes zero",
        "continuation_markers": {"fixed": "s", "reactive_continuation": "o"},
    }
    return fig, data
