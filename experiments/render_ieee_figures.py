from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from _bootstrap import ROOT

from egress_sim.ieee_style import (
    BLUE,
    BLUE_LIGHT,
    GRID,
    IEEE_DOUBLE_COLUMN_IN,
    INK,
    MUTED,
    NEUTRAL,
    ORANGE,
    ORANGE_LIGHT,
    save_ieee,
    use_ieee_style,
)


REQUIRED_TABLES = (
    "information_regime_map_summary.csv",
    "information_value_validation_runs.csv",
    "information_value_validation_effects.csv",
    "time_step_sensitivity_effects.csv",
    "residual_spread_ablation_effects.csv",
    "clipped_lognormal_diagnostic.csv",
)


def validate_inputs(tables: Path) -> None:
    missing = [name for name in REQUIRED_TABLES if not (tables / name).exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Missing figure-data tables:\n  - " + joined +
            "\nRun experiments/run_ieee_figure_pipeline.py first, "
            "or generate the listed experiments individually."
        )


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", pad=4, fontweight="bold", color=INK)


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
) -> None:
    x, y = xy
    w, h = wh
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
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=linespacing,
        zorder=zorder + 1,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str, lw: float = 0.9) -> None:
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

    zones = [("Z1\n$n=1400$", 0.78), ("Z2\n$n=1200$", 0.60), ("Z3\n$n=1000$", 0.40), ("Z4\n$n=1400$", 0.22)]
    exits = [("Corridor A", 0.75), ("Corridor B", 0.50), ("Corridor C", 0.25)]
    for label, y in zones:
        rounded_box(ax, (0.03, y - 0.047), (0.20, 0.094), label, face=ORANGE_LIGHT, edge=ORANGE, fontsize=5.25, linespacing=1.08)
    rounded_box(ax, (0.35, 0.20), (0.26, 0.58), "queue\nstate\n$R,Q,Y,W,D$", face="#F7F8FA", edge="#C8CDD7", fontsize=5.8, linespacing=1.06)
    for label, y in exits:
        rounded_box(ax, (0.77, y - 0.047), (0.18, 0.094), label, face=BLUE_LIGHT, edge=BLUE, fontsize=5.7)

    for _, y in zones:
        arrow(ax, (0.23, y), (0.35, y), color=ORANGE if y in (0.60, 0.40) else BLUE, lw=1.0)
    for label, y in exits:
        arrow(ax, (0.61, y), (0.77, y), color=BLUE if label != "Corridor C" else ORANGE, lw=1.15)

    ax.text(0.48, 0.075, "conserved queue state", ha="center", fontsize=5.4, color=MUTED)


def draw_information_loop(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("Observe", "queues + saturated\nthroughput", BLUE_LIGHT, BLUE),
        ("Estimate", "effective capacity", ORANGE_LIGHT, ORANGE),
        ("Predict", "prior OR\nconditioned scenarios", BLUE_LIGHT, BLUE),
        ("Choose", "best of 16\ncandidates", ORANGE_LIGHT, ORANGE),
        ("Guide", "route shares\nfor 120 s", BLUE_LIGHT, BLUE),
    ]
    y_positions = [0.84, 0.66, 0.48, 0.30, 0.12]
    for index, ((head, detail, face, edge), y) in enumerate(zip(steps, y_positions, strict=True)):
        rounded_box(ax, (0.15, y - 0.055), (0.60, 0.11), f"{head}\n{detail}", face=face, edge=edge, fontsize=5.55, linespacing=0.98)
        if index < len(steps) - 1:
            arrow(ax, (0.47, y - 0.057), (0.47, y_positions[index + 1] + 0.057), color=INK, lw=0.75)
    ax.plot([0.80, 0.89, 0.89], [0.12, 0.12, 0.84], color=NEUTRAL, lw=0.75)
    arrow(ax, (0.89, 0.84), (0.75, 0.84), color=NEUTRAL, lw=0.75)
    ax.text(0.935, 0.48, "feedback", ha="center", va="center", rotation=90, fontsize=5.0, color=MUTED)


def plot_information_map(ax: plt.Axes, frame: pd.DataFrame) -> None:
    matrix = frame.pivot(index="capacity_log_sigma", columns="compliance_mean", values="mean_difference_percent").sort_index(ascending=False)
    low = frame.pivot(index="capacity_log_sigma", columns="compliance_mean", values="mean_ci95_low").sort_index(ascending=False)
    high = frame.pivot(index="capacity_log_sigma", columns="compliance_mean", values="mean_ci95_high").sort_index(ascending=False)
    annot = matrix.copy().astype(object)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            star = "*" if low.iloc[r, c] <= 0.0 <= high.iloc[r, c] else ""
            annot.iloc[r, c] = f"{matrix.iloc[r, c]:+.1f}{star}"
    limit = 10.0
    cmap = LinearSegmentedColormap.from_list("info_value", [BLUE, BLUE_LIGHT, "#FFFFFF", ORANGE_LIGHT, ORANGE])
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 6.1},
        linewidths=0.8,
        linecolor="white",
        cbar=False,
    )
    ax.set_xlabel("Mean compliance")
    ax.set_ylabel(r"Capacity uncertainty, $\sigma$")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if float(high.iloc[r, c]) < 0.0:
                ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=BLUE, linewidth=1.45))
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
    grid = fig.add_gridspec(1, 3, width_ratios=(1.02, 0.96, 1.30), wspace=0.56)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    draw_network_panel(ax0)
    draw_information_loop(ax1)
    plot_information_map(ax2, pd.read_csv(tables / "information_regime_map_summary.csv"))
    panel_title(ax0, "a", "Queue benchmark")
    panel_title(ax1, "b", "Matched controller")
    panel_title(ax2, "c", "Where information helps")
    fig.subplots_adjust(left=0.025, right=0.992, top=0.90, bottom=0.20)
    save_ieee(fig, out / "fig1_system_overview")


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(5000, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def forest(ax: plt.Axes, frame: pd.DataFrame, labels: list[str], *, xlim: tuple[float, float], annotate_side: str = "right") -> None:
    y = np.arange(len(frame))[::-1]
    ax.axvline(0, color=INK, linestyle=":", linewidth=0.8)
    for yy, row in zip(y, frame.itertuples(), strict=True):
        est = float(row.mean_difference)
        low = float(row.mean_ci95_low)
        high = float(row.mean_ci95_high)
        color = BLUE if high < 0 else ORANGE if low > 0 else NEUTRAL
        ax.errorbar(
            est,
            yy,
            xerr=[[est - low], [high - est]],
            fmt="s",
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            capsize=2.1,
            linewidth=0.85,
            zorder=3,
        )
        ha = "left" if annotate_side == "right" else "right"
        dx = 4 if annotate_side == "right" else -4
        ax.text(est + dx, yy + 0.16, f"{est:+.1f}s", ha=ha, va="bottom", fontsize=5.4, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    sns.despine(ax=ax, left=True)


def figure2(tables: Path, out: Path) -> None:
    use_ieee_style()
    raw = pd.read_csv(tables / "information_value_validation_runs.csv")
    effects = pd.read_csv(tables / "information_value_validation_effects.csv")
    fig = plt.figure(figsize=(IEEE_DOUBLE_COLUMN_IN, 3.05))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.0), width_ratios=(1.22, 1.06, 1.02), hspace=0.62, wspace=0.52)
    ax_ladder = fig.add_subplot(grid[:, 0])
    ax_primary = fig.add_subplot(grid[0, 1:])
    ax_decomp = fig.add_subplot(grid[1, 1])
    ax_cvar = fig.add_subplot(grid[1, 2])

    ladder = [
        ("static_nearest", "Static", NEUTRAL),
        ("deterministic_load_balance", "Reactive", "#A9B0BD"),
        ("unconditioned_mean_only", "Uncond.\nmean", BLUE_LIGHT),
        ("scenario_mean_only", "Cond.\nmean", BLUE),
        ("oracle_capacity_mean_only", "Oracle\nmean", ORANGE_LIGHT),
    ]
    xs = np.arange(len(ladder))
    means, lows, highs = [], [], []
    for idx, (policy, _, _) in enumerate(ladder):
        vals = raw.loc[raw.policy == policy, "clearance_time_seconds"].to_numpy()
        mean, low, high = bootstrap_mean(vals, 100 + idx)
        means.append(mean)
        lows.append(low)
        highs.append(high)
    ax_ladder.plot(xs, means, color=INK, linewidth=0.8, zorder=1)
    for x, mean, low, high, (_, label, color) in zip(xs, means, lows, highs, ladder, strict=True):
        ax_ladder.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt="o", color=INK, ecolor=INK, capsize=2.2, markersize=0)
        ax_ladder.scatter(x, mean, s=62, color=color, edgecolor=INK, linewidth=0.7, zorder=3)
        ax_ladder.text(x, mean + 34, f"{mean:.0f}s", ha="center", fontsize=5.4, color=INK)
    ax_ladder.set_xticks(xs, [label for _, label, _ in ladder])
    ax_ladder.set_ylabel("Mean clearance (s)")
    ax_ladder.set_ylim(650, 1040)
    ax_ladder.grid(axis="y", color=GRID, linewidth=0.45)
    sns.despine(ax=ax_ladder)
    panel_title(ax_ladder, "a", "Final-validation ladder")

    pivot = raw.pivot(index="scenario_seed", columns="policy", values="clearance_time_seconds")
    diff = pivot["scenario_mean_only"] - pivot["unconditioned_mean_only"]
    bins = np.arange(-250, 61, 20)
    ax_primary.hist(diff, bins=bins, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=0.6)
    mean = diff.mean()
    q10, q90 = np.quantile(diff, [0.10, 0.90])
    ax_primary.axvline(0, color=INK, linestyle=":", linewidth=0.9)
    ax_primary.axvline(mean, color=BLUE, linewidth=1.5)
    ax_primary.axvspan(q10, q90, color=BLUE, alpha=0.08)
    wins = float((diff < 0).mean() * 100.0)
    ties = float((diff == 0).mean() * 100.0)
    ax_primary.text(
        0.035,
        0.90,
        f"mean {mean:.1f} s\n{wins:.1f}% cond. faster\n{ties:.1f}% tied",
        transform=ax_primary.transAxes,
        ha="left",
        va="top",
        fontsize=5.75,
        color=BLUE,
        fontweight="bold",
        linespacing=1.03,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=BLUE, linewidth=0.55, alpha=0.94),
    )
    ax_primary.set_xlim(-250, 45)
    ax_primary.set_xlabel(r"Paired $\Delta T_{\rm sense}$, cond. $-$ uncond. (s)")
    ax_primary.set_ylabel("Seed count")
    ax_primary.grid(axis="y", color=GRID, linewidth=0.45)
    sns.despine(ax=ax_primary)
    panel_title(ax_primary, "b", "Primary matched information effect")

    decomp_keys = ["planning-vs-reactive", "primary-sensing-value", "oracle-residual-information"]
    decomp = effects.set_index("comparison").loc[decomp_keys].reset_index()
    forest(
        ax_decomp,
        decomp,
        ["Uncond. mean\n- reactive", "Cond. mean\n- uncond. mean", "Oracle mean\n- cond. mean"],
        xlim=(-34, 20),
    )
    ax_decomp.set_xlabel(r"$\Delta T$ (s)")
    panel_title(ax_decomp, "c", "Component effects")

    cvar = effects.set_index("comparison").loc[["cvar-without-conditioning", "cvar-with-conditioning"]].reset_index()
    forest(ax_cvar, cvar, ["CVaR - mean\n(no cond.)", "CVaR - mean\n(cond.)"], xlim=(-4, 10))
    ax_cvar.set_xlabel(r"$\Delta T$ (s)")
    panel_title(ax_cvar, "d", "CVaR adds little")

    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.17)
    save_ieee(fig, out / "fig2_performance_landscape")


def figure3(tables: Path, out: Path) -> None:
    use_ieee_style()
    dt = pd.read_csv(tables / "time_step_sensitivity_effects.csv")
    residual = pd.read_csv(tables / "residual_spread_ablation_effects.csv")
    lognorm = pd.read_csv(tables / "clipped_lognormal_diagnostic.csv")
    fig = plt.figure(figsize=(IEEE_DOUBLE_COLUMN_IN, 2.60))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.02, 0.95, 1.04), wspace=0.43)
    ax_dt = fig.add_subplot(grid[0, 0])
    ax_res = fig.add_subplot(grid[0, 1])
    ax_clip = fig.add_subplot(grid[0, 2])

    colors = {"sensing-vs-unconditioned": BLUE, "conditioned-vs-reactive": ORANGE}
    labels = {"sensing-vs-unconditioned": "matched information", "conditioned-vs-reactive": "vs. reactive"}
    for key, subset in dt.groupby("comparison"):
        subset = subset.sort_values("time_step_seconds")
        x = subset.time_step_seconds.astype(float).to_numpy()
        y = subset.mean_difference.astype(float).to_numpy()
        low = subset.mean_ci95_low.astype(float).to_numpy()
        high = subset.mean_ci95_high.astype(float).to_numpy()
        ax_dt.errorbar(
            x,
            y,
            yerr=[y - low, high - y],
            marker="o",
            color=colors[key],
            capsize=2.2,
            linewidth=0.95,
            label=labels[key],
        )
    ax_dt.axhline(0, color=INK, linestyle=":", linewidth=0.8)
    ax_dt.set_xlim(10.8, 1.7)
    ax_dt.set_xlabel(r"Finer time step $\Delta t$ (s)")
    ax_dt.set_ylabel(r"Paired effect $\Delta T$ (s)")
    ax_dt.legend(frameon=False, loc="lower left", fontsize=5.4, handlelength=1.3)
    ax_dt.grid(color=GRID, linewidth=0.45)
    sns.despine(ax=ax_dt)
    panel_title(ax_dt, "a", "Not a one-step artifact")

    heat = residual.pivot(index="residual_sigma", columns="capacity_log_sigma", values="mean_difference").sort_index()
    heat.index = [f"{value:.2f}" for value in heat.index]
    annot = heat.map(lambda v: f"{v:.0f}")
    sns.heatmap(
        heat,
        ax=ax_res,
        cmap=LinearSegmentedColormap.from_list("residual", [BLUE, BLUE_LIGHT, "#FFFFFF"]),
        vmin=-65,
        vmax=0,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 6.0},
        linewidths=0.7,
        linecolor="white",
        cbar=False,
    )
    ax_res.set_xlabel(r"Environmental $\sigma$")
    ax_res.set_ylabel(r"Scenario residual $\sigma_r$")
    ax_res.tick_params(axis="x", rotation=0)
    ax_res.tick_params(axis="y", rotation=0)
    ax_res.text(0, -0.24, "entries: cond. - uncond. mean (s)", transform=ax_res.transAxes, ha="left", va="top", fontsize=5.0, color=MUTED)
    panel_title(ax_res, "b", "Residual-spread sensitivity")

    x = lognorm.capacity_log_sigma.astype(float).to_numpy()
    mean = lognorm.post_clip_mean.astype(float).to_numpy()
    lower_mass = lognorm.mass_at_lower_clip_percent.astype(float).to_numpy()
    upper_mass = lognorm.mass_at_upper_clip_percent.astype(float).to_numpy()
    ax_clip.plot(x, mean, marker="o", color=BLUE, linewidth=1.0)
    ax_clip.axhline(1.0, color=INK, linestyle=":", linewidth=0.8)
    ax_clip.set_xlabel(r"Capacity log-$\sigma$")
    ax_clip.set_ylabel("Post-clipping mean")
    ax_clip.set_ylim(0.90, 1.015)
    ax_clip.grid(color=GRID, linewidth=0.45)
    ax2 = ax_clip.twinx()
    ax2.bar(x - 0.012, lower_mass, width=0.018, color=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=0.45, label="lower clip")
    ax2.bar(x + 0.012, upper_mass, width=0.018, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=0.45, label="upper clip")
    ax2.set_ylabel("Clipped mass (%)", labelpad=2)
    ax2.set_ylim(0, 22)
    sns.despine(ax=ax_clip, right=False)
    panel_title(ax_clip, "c", "Clipping shifts high-$\\sigma$ mean")

    fig.subplots_adjust(left=0.07, right=0.955, top=0.86, bottom=0.24)
    save_ieee(fig, out / "fig3_context_validation")


if __name__ == "__main__":
    tables = ROOT / "outputs" / "tables"
    output = ROOT / "outputs" / "figures" / "ieee"
    output.mkdir(parents=True, exist_ok=True)
    validate_inputs(tables)
    figure1(tables, output)
    figure2(tables, output)
    figure3(tables, output)
    print(f"Wrote IEEE figures to {output}")
