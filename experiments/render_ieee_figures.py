from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from _bootstrap import ROOT

from egress_sim.figure_protocol import validate_reference_figure_inputs
from egress_sim.ieee_style import (
    BLUE,
    BLUE_LIGHT,
    GRID,
    IEEE_DOUBLE_COLUMN_IN,
    INK,
    MUTED,
    NEUTRAL,
    NEUTRAL_LIGHT,
    ORANGE,
    ORANGE_LIGHT,
    save_ieee,
    use_ieee_style,
)


def _despine(ax: plt.Axes, *, right: bool = True, top: bool = True) -> None:
    if right:
        ax.spines["right"].set_visible(False)
    if top:
        ax.spines["top"].set_visible(False)


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(
        f"({letter}) {title}",
        loc="left",
        pad=4,
        fontweight="bold",
        color=INK,
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 6.0,
    weight: str = "bold",
    color: str = INK,
    zorder: int = 3,
    linespacing: float = 1.0,
    auto_fit: bool = False,
    safety_pad: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float, float, float]:
    x, y = xy
    w, h = wh
    center_x = x + w / 2
    center_y = y + h / 2
    label = ax.text(
        center_x,
        center_y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=linespacing,
        zorder=zorder + 1,
        clip_on=False,
    )
    if auto_fit:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        bounds = label.get_window_extent(renderer=renderer)
        inv = ax.transData.inverted()
        (text_x0, text_y0), (text_x1, text_y1) = inv.transform(
            ((bounds.x0, bounds.y0), (bounds.x1, bounds.y1))
        )
        pad_x, pad_y = safety_pad
        w = max(w, (text_x1 - text_x0) + 2.0 * pad_x)
        h = max(h, (text_y1 - text_y0) + 2.0 * pad_y)
        x = center_x - w / 2
        y = center_y - h / 2
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.8,
            zorder=zorder,
            clip_on=False,
        )
    )
    return x, y, w, h


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    lw: float = 0.9,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=lw,
            color=color,
            shrinkA=1,
            shrinkB=1,
            zorder=2,
        )
    )


def draw_network_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.93, "sources", ha="left", fontsize=5.8, color=MUTED)
    ax.text(0.75, 0.93, "corridors", ha="left", fontsize=5.8, color=MUTED)
    zones = [
        ("Z1\n$n=1400$", 0.78),
        ("Z2\n$n=1200$", 0.60),
        ("Z3\n$n=1000$", 0.40),
        ("Z4\n$n=1400$", 0.22),
    ]
    exits = [("Corridor A", 0.75), ("Corridor B", 0.50), ("Corridor C", 0.25)]
    for label, y in zones:
        rounded_box(
            ax,
            (0.03, y - 0.047),
            (0.20, 0.094),
            label,
            face=ORANGE_LIGHT,
            edge=ORANGE,
            fontsize=5.25,
            linespacing=1.08,
        )
    rounded_box(
        ax,
        (0.35, 0.20),
        (0.26, 0.58),
        "queue\nstate\n$R,Q,Y,W,D$",
        face="#F7F8FA",
        edge="#C8CDD7",
        fontsize=5.8,
        linespacing=1.06,
    )
    exit_boxes: dict[str, tuple[float, float, float, float]] = {}
    for label, y in exits:
        exit_boxes[label] = rounded_box(
            ax,
            (0.77, y - 0.047),
            (0.18, 0.094),
            label,
            face=BLUE_LIGHT,
            edge=BLUE,
            fontsize=5.7,
            auto_fit=True,
            safety_pad=(0.035, 0.014),
        )
    for _, y in zones:
        arrow(
            ax,
            (0.23, y),
            (0.35, y),
            color=ORANGE if y in (0.60, 0.40) else BLUE,
            lw=1.0,
        )
    for label, y in exits:
        arrow(
            ax,
            (0.61, y),
            (exit_boxes[label][0], y),
            color=BLUE if label != "Corridor C" else ORANGE,
            lw=1.15,
        )
    ax.text(
        0.48,
        0.075,
        "conserved queue state",
        ha="center",
        fontsize=5.4,
        color=MUTED,
    )


def draw_information_loop(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("Observe", "queues + saturated\nthroughput", BLUE_LIGHT, BLUE),
        ("Estimate", "persistent capacity", ORANGE_LIGHT, ORANGE),
        ("Predict", "same spread; prior OR\nestimated center", BLUE_LIGHT, BLUE),
        ("Choose", "best of 16\ncandidates", ORANGE_LIGHT, ORANGE),
        ("Guide", "route shares\nfor 120 s", BLUE_LIGHT, BLUE),
    ]
    y_positions = [0.84, 0.66, 0.48, 0.30, 0.12]
    for index, ((head, detail, face, edge), y) in enumerate(
        zip(steps, y_positions, strict=True)
    ):
        rounded_box(
            ax,
            (0.13, y - 0.055),
            (0.66, 0.11),
            f"{head}\n{detail}",
            face=face,
            edge=edge,
            fontsize=5.35,
            linespacing=0.98,
        )
        if index < len(steps) - 1:
            arrow(
                ax,
                (0.46, y - 0.057),
                (0.46, y_positions[index + 1] + 0.057),
                color=INK,
                lw=0.75,
            )
    ax.plot([0.82, 0.90, 0.90], [0.12, 0.12, 0.84], color=NEUTRAL, lw=0.75)
    arrow(ax, (0.90, 0.84), (0.79, 0.84), color=NEUTRAL, lw=0.75)
    ax.text(
        0.945,
        0.48,
        "feedback",
        ha="center",
        va="center",
        rotation=90,
        fontsize=5.0,
        color=MUTED,
    )


def plot_information_map(ax: plt.Axes, frame: pd.DataFrame) -> None:
    row_values = sorted(frame["capacity_log_sigma"].unique(), reverse=True)
    col_values = sorted(frame["compliance_mean"].unique())
    matrix = np.empty((len(row_values), len(col_values)))
    low = np.empty_like(matrix)
    high = np.empty_like(matrix)
    for row_index, sigma in enumerate(row_values):
        for col_index, compliance in enumerate(col_values):
            row = frame[
                np.isclose(frame["capacity_log_sigma"], sigma)
                & np.isclose(frame["compliance_mean"], compliance)
            ].iloc[0]
            matrix[row_index, col_index] = float(row["mean_difference_percent"])
            low[row_index, col_index] = float(row["mean_ci95_low"])
            high[row_index, col_index] = float(row["mean_ci95_high"])
    limit = max(5.0, float(np.ceil(np.max(np.abs(matrix)))))
    cmap = LinearSegmentedColormap.from_list(
        "info_value", [BLUE, BLUE_LIGHT, "#FFFFFF", ORANGE_LIGHT, ORANGE]
    )
    image = ax.imshow(
        matrix,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    del image
    ax.set_xticks(np.arange(len(col_values)), [f"{value:.2g}" for value in col_values])
    ax.set_yticks(np.arange(len(row_values)), [f"{value:.2g}" for value in row_values])
    ax.set_xlabel("Mean compliance")
    ax.set_ylabel(r"Capacity uncertainty, $\sigma$")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            unresolved = low[row_index, col_index] <= 0.0 <= high[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]:+.1f}{'*' if unresolved else ''}",
                ha="center",
                va="center",
                fontsize=6.0,
                color=(
                    "white"
                    if abs(matrix[row_index, col_index]) >= 0.58 * limit
                    else INK
                ),
            )
            if high[row_index, col_index] < 0.0:
                ax.add_patch(
                    Rectangle(
                        (col_index - 0.5, row_index - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=BLUE,
                        linewidth=1.35,
                    )
                )
    ax.set_xticks(np.arange(-0.5, len(col_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_values), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(
        0,
        -0.20,
        r"Values are $\Delta T$ (%); blue favors conditioning; * interval crosses 0.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        color=MUTED,
        clip_on=False,
    )


def figure1(tables: Path, out: Path) -> None:
    use_ieee_style()
    fig = plt.figure(figsize=(IEEE_DOUBLE_COLUMN_IN, 3.26))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.02, 0.98, 1.30), wspace=0.56)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    draw_network_panel(ax0)
    draw_information_loop(ax1)
    plot_information_map(
        ax2,
        pd.read_csv(tables / "information_regime_map_summary.csv"),
    )
    panel_title(ax0, "a", "Queue benchmark")
    panel_title(ax1, "b", "Center-only ablation")
    panel_title(ax2, "c", "Where information helps")
    fig.subplots_adjust(left=0.025, right=0.992, top=0.90, bottom=0.20)
    save_ieee(fig, out / "fig1_system_overview")


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(5000, len(values)))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def forest(
    ax: plt.Axes,
    frame: pd.DataFrame,
    labels: list[str],
    *,
    xlim: tuple[float, float] | None = None,
    annotate: bool = True,
) -> None:
    y = np.arange(len(frame))[::-1]
    ax.axvline(0, color=INK, linestyle=":", linewidth=0.8)
    for yy, row in zip(y, frame.itertuples(), strict=True):
        estimate = float(row.mean_difference)
        low = float(row.mean_ci95_low)
        high = float(row.mean_ci95_high)
        color = BLUE if high < 0 else ORANGE if low > 0 else NEUTRAL
        ax.errorbar(
            estimate,
            yy,
            xerr=[[estimate - low], [high - estimate]],
            fmt="s",
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            capsize=2.1,
            linewidth=0.85,
            zorder=3,
        )
        if annotate:
            ax.text(
                estimate,
                yy + 0.18,
                f"{estimate:+.1f}s",
                ha="center",
                va="bottom",
                fontsize=5.15,
                color=INK,
            )
    ax.set_yticks(y, labels)
    if xlim is None:
        all_low = frame["mean_ci95_low"].astype(float).min()
        all_high = frame["mean_ci95_high"].astype(float).max()
        span = max(10.0, all_high - all_low)
        xlim = (all_low - 0.12 * span, all_high + 0.12 * span)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    _despine(ax)


def _plot_estimator_panel(ax: plt.Axes, tables: Path) -> None:
    summary = pd.read_csv(tables / "estimator_diagnostics_summary.csv")
    subset = summary[summary["regime"] == "default"].sort_values("time_seconds")
    subset = subset[subset["time_seconds"] <= 600.0]
    time = subset["time_seconds"].astype(float).to_numpy()
    mean = subset["capacity_mae_mean"].astype(float).to_numpy()
    q10 = subset["capacity_mae_q10"].astype(float).to_numpy()
    q90 = subset["capacity_mae_q90"].astype(float).to_numpy()
    observed = subset["capacity_observed_fraction_mean"].astype(float).to_numpy()
    ax.fill_between(time, q10, q90, color=BLUE_LIGHT, alpha=0.75, linewidth=0)
    ax.plot(time, mean, color=BLUE, linewidth=1.15, label="capacity MAE")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Multiplier MAE")
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, linewidth=0.45)
    ax2 = ax.twinx()
    ax2.plot(
        time,
        100.0 * observed,
        color=ORANGE,
        linewidth=1.0,
        linestyle="--",
        label="ever observed",
    )
    ax2.set_ylabel("Links observed (%)", labelpad=2)
    ax2.set_ylim(0, 105)
    ax2.spines["top"].set_visible(False)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        loc="upper right",
        fontsize=5.1,
        handlelength=1.4,
    )
    _despine(ax, right=False)


def figure2(tables: Path, out: Path) -> None:
    use_ieee_style()
    raw = pd.read_csv(tables / "information_value_validation_runs.csv")
    effects = pd.read_csv(tables / "information_value_validation_effects.csv")
    fig = plt.figure(figsize=(IEEE_DOUBLE_COLUMN_IN, 3.55))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.0, 1.0),
        width_ratios=(1.22, 1.02, 1.08),
        hspace=0.64,
        wspace=0.56,
    )
    ax_ladder = fig.add_subplot(grid[:, 0])
    ax_primary = fig.add_subplot(grid[0, 1:])
    ax_decomp = fig.add_subplot(grid[1, 1])
    ax_estimator = fig.add_subplot(grid[1, 2])

    ladder = [
        ("static_nearest", "Static", NEUTRAL),
        ("deterministic_load_balance", "Reactive", "#A9B0BD"),
        ("unconditioned_mean_only", "Uncond.\nmean", BLUE_LIGHT),
        ("conditioned_mean_equal_spread", "Cond.\nmean", BLUE),
        ("oracle_capacity_mean_only", "Oracle\nmean", ORANGE_LIGHT),
    ]
    xs = np.arange(len(ladder))
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for index, (policy, _, _) in enumerate(ladder):
        values = raw.loc[
            raw["policy"] == policy,
            "clearance_time_seconds",
        ].to_numpy()
        mean, low, high = bootstrap_mean(values, 100 + index)
        means.append(mean)
        lows.append(low)
        highs.append(high)
    ax_ladder.plot(xs, means, color=INK, linewidth=0.8, zorder=1)
    for x, mean, low, high, (_, label, color) in zip(
        xs, means, lows, highs, ladder, strict=True
    ):
        ax_ladder.errorbar(
            x,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="none",
            color=INK,
            ecolor=INK,
            capsize=2.2,
        )
        ax_ladder.scatter(
            x,
            mean,
            s=62,
            color=color,
            edgecolor=INK,
            linewidth=0.7,
            zorder=3,
        )
        ax_ladder.text(
            x,
            mean + 32,
            f"{mean:.0f}s",
            ha="center",
            fontsize=5.35,
            color=INK,
        )
    ax_ladder.set_xticks(xs, [label for _, label, _ in ladder])
    ax_ladder.set_ylabel("Mean clearance (s)")
    y_min = min(means) - 60
    y_max = max(means) + 75
    ax_ladder.set_ylim(y_min, y_max)
    ax_ladder.grid(axis="y", color=GRID, linewidth=0.45)
    _despine(ax_ladder)
    panel_title(ax_ladder, "a", "Confirmatory policy ladder")

    pivot = raw.pivot(
        index="scenario_seed",
        columns="policy",
        values="clearance_time_seconds",
    )
    difference = (
        pivot["conditioned_mean_equal_spread"]
        - pivot["unconditioned_mean_only"]
    )
    low_edge = 10.0 * np.floor(difference.min() / 10.0) - 10.0
    high_edge = 10.0 * np.ceil(difference.max() / 10.0) + 10.0
    bins = np.arange(low_edge, high_edge + 10.0, 10.0)
    ax_primary.hist(
        difference,
        bins=bins,
        color=BLUE_LIGHT,
        edgecolor=BLUE,
        linewidth=0.6,
    )
    mean_difference = float(difference.mean())
    q10, q90 = np.quantile(difference, [0.10, 0.90])
    ax_primary.axvline(0, color=INK, linestyle=":", linewidth=0.9)
    ax_primary.axvline(mean_difference, color=BLUE, linewidth=1.5)
    ax_primary.axvspan(q10, q90, color=BLUE, alpha=0.08)
    wins = float((difference < 0).mean() * 100.0)
    ties = float((difference == 0).mean() * 100.0)
    ax_primary.text(
        0.035,
        0.90,
        f"mean {mean_difference:.1f} s\n{wins:.1f}% cond. faster\n{ties:.1f}% tied",
        transform=ax_primary.transAxes,
        ha="left",
        va="top",
        fontsize=5.75,
        color=BLUE,
        fontweight="bold",
        linespacing=1.03,
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": BLUE,
            "linewidth": 0.55,
            "alpha": 0.94,
        },
    )
    ax_primary.set_xlabel(
        r"Paired $\Delta T_{\rm sense}$, conditioned $-$ unconditioned (s)"
    )
    ax_primary.set_ylabel("Seed count")
    ax_primary.grid(axis="y", color=GRID, linewidth=0.45)
    _despine(ax_primary)
    panel_title(ax_primary, "b", "Center-only information effect")

    decomp_keys = [
        "planning-vs-reactive",
        "primary-sensing-value",
        "secondary-system-comparison",
        "oracle-residual-information",
    ]
    decomp = effects.set_index("comparison").loc[decomp_keys].reset_index()
    forest(
        ax_decomp,
        decomp,
        [
            "Uncond. mean\n- reactive",
            "Cond. mean\n- uncond.",
            "Cond. mean\n- reactive",
            "Oracle mean\n- cond.",
        ],
    )
    ax_decomp.set_xlabel(r"$\Delta T$ (s)")
    panel_title(ax_decomp, "c", "Component effects")

    _plot_estimator_panel(ax_estimator, tables)
    panel_title(ax_estimator, "d", "Estimator convergence")

    fig.subplots_adjust(left=0.065, right=0.962, top=0.90, bottom=0.17)
    save_ieee(fig, out / "fig2_performance_landscape")


def _plot_time_step(ax: plt.Axes, tables: Path) -> None:
    frame = pd.read_csv(tables / "time_step_sensitivity_effects.csv")
    colors = {
        "sensing-vs-unconditioned": BLUE,
        "conditioned-vs-reactive": ORANGE,
    }
    labels = {
        "sensing-vs-unconditioned": "matched information",
        "conditioned-vs-reactive": "vs. reactive",
    }
    for key, subset in frame.groupby("comparison"):
        subset = subset.sort_values("time_step_seconds")
        x = subset["time_step_seconds"].astype(float).to_numpy()
        y = subset["mean_difference"].astype(float).to_numpy()
        low = subset["mean_ci95_low"].astype(float).to_numpy()
        high = subset["mean_ci95_high"].astype(float).to_numpy()
        ax.errorbar(
            x,
            y,
            yerr=[y - low, high - y],
            marker="o",
            color=colors[key],
            capsize=2.2,
            linewidth=0.95,
            label=labels[key],
        )
    ax.axhline(0, color=INK, linestyle=":", linewidth=0.8)
    ax.set_xlim(10.8, 1.7)
    ax.set_xlabel(r"Finer time step $\Delta t$ (s)")
    ax.set_ylabel(r"Paired effect $\Delta T$ (s)")
    ax.legend(frameon=False, loc="lower left", fontsize=5.25, handlelength=1.3)
    ax.grid(color=GRID, linewidth=0.45)
    _despine(ax)


def _plot_topologies(ax: plt.Axes, tables: Path) -> None:
    frame = pd.read_csv(tables / "topology_generalization_effects.csv")
    order = [
        "base_asymmetric",
        "symmetric",
        "path_bottleneck",
        "exit_bottleneck",
        "expanded_venue",
    ]
    labels = [
        "Base asymmetric",
        "Symmetric",
        "Path bottleneck",
        "Exit bottleneck",
        "Expanded 5x4",
    ]
    ordered = frame.set_index("topology").loc[order].reset_index()
    forest(ax, ordered, labels)
    ax.set_xlabel(r"Cond. $-$ uncond. mean (s)")


def _plot_sensor_faults(ax: plt.Axes, tables: Path) -> None:
    frame = pd.read_csv(tables / "sensor_condition_effects.csv")
    order = [
        "clean",
        "noise_sigma_015",
        "delay_60s",
        "dropout_20pct",
        "bias_plus_10pct",
    ]
    labels = ["Clean", "Noise 0.15", "Delay 60 s", "Dropout 20%", "Bias +10%"]
    y = np.arange(len(order))[::-1]
    offsets = {"default": 0.12, "high_information": -0.12}
    colors = {"default": ORANGE, "high_information": BLUE}
    markers = {"default": "o", "high_information": "s"}
    display = {"default": "default", "high_information": "high-info"}
    ax.axvline(0, color=INK, linestyle=":", linewidth=0.8)
    for regime in ("default", "high_information"):
        subset = frame[frame["regime"] == regime].set_index("fault").loc[order]
        estimate = subset["mean_difference"].astype(float).to_numpy()
        low = subset["mean_ci95_low"].astype(float).to_numpy()
        high = subset["mean_ci95_high"].astype(float).to_numpy()
        ax.errorbar(
            estimate,
            y + offsets[regime],
            xerr=[estimate - low, high - estimate],
            fmt=markers[regime],
            color=colors[regime],
            markerfacecolor="white",
            capsize=2.0,
            linewidth=0.85,
            label=display[regime],
        )
    ax.set_yticks(y, labels)
    all_low = frame["mean_ci95_low"].astype(float).min()
    all_high = frame["mean_ci95_high"].astype(float).max()
    span = max(20.0, all_high - all_low)
    ax.set_xlim(all_low - 0.10 * span, all_high + 0.10 * span)
    ax.set_xlabel(r"Cond. $-$ uncond. mean (s)")
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.legend(frameon=False, loc="lower left", fontsize=5.2, handlelength=1.2)
    _despine(ax)


def figure3(tables: Path, out: Path) -> None:
    use_ieee_style()
    fig = plt.figure(figsize=(IEEE_DOUBLE_COLUMN_IN, 2.72))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.00, 1.05, 1.16), wspace=0.58)
    ax_dt = fig.add_subplot(grid[0, 0])
    ax_topology = fig.add_subplot(grid[0, 1])
    ax_sensor = fig.add_subplot(grid[0, 2])
    _plot_time_step(ax_dt, tables)
    _plot_topologies(ax_topology, tables)
    _plot_sensor_faults(ax_sensor, tables)
    panel_title(ax_dt, "a", "Time-step robustness")
    panel_title(ax_topology, "b", "Network-family transfer")
    panel_title(ax_sensor, "c", "Estimate faults")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.87, bottom=0.22)
    save_ieee(fig, out / "fig3_context_validation")


def render_all(tables: Path, output: Path, *, strict: bool = True) -> None:
    tables = Path(tables)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if strict:
        validate_reference_figure_inputs(tables)
    figure1(tables, output)
    figure2(tables, output)
    figure3(tables, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render the three compact IEEE-style paper figures."
    )
    parser.add_argument(
        "--tables",
        type=Path,
        default=ROOT / "paper_artifacts" / "tables",
        help="Directory containing the frozen paper CSV tables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "figures" / "ieee",
        help="Output directory for vector PDF and 600-dpi PNG figures.",
    )
    parser.add_argument(
        "--allow-nonreference",
        action="store_true",
        help="Render exploratory tables without frozen-protocol validation.",
    )
    args = parser.parse_args()
    render_all(args.tables, args.output, strict=not args.allow_nonreference)
    print(f"Wrote IEEE figures to {args.output}")
