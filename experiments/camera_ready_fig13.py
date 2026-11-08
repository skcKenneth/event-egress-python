"""Targeted, fixed-canvas refinements of the benchmark and robustness figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from camera_ready_common import (
    BLUE, BLUE_LIGHT, GRID, INK, NEUTRAL, ORANGE, ORANGE_LIGHT,
    WIDTH_IN, despine, effect, title, use_style, zero,
)


def _box(ax, x, y, w, h, text, *, face, edge, fontsize=7.2):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor=face, edgecolor=edge, linewidth=.8, clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, linespacing=.98, clip_on=False)
    return patch


def _arrow(ax, start, end, *, color=INK, linewidth=.9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=5.5, linewidth=linewidth,
                                color=color, shrinkA=.5, shrinkB=.5))


def _network(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.02, .97, "Sources", fontsize=7.2, va="top", color=INK)
    ax.text(.68, .97, "Corridors", fontsize=7.2, va="top", color=INK)
    zones = [("Z1\n$n=1400$", .80), ("Z2\n$n=1200$", .60),
             ("Z3\n$n=1000$", .40), ("Z4\n$n=1400$", .20)]
    for label, y in zones:
        _box(ax, .02, y-.07, .24, .14, label, face=ORANGE_LIGHT, edge=ORANGE)
        _arrow(ax, (.27, y), (.34, y), color=ORANGE)
    _box(ax, .35, .16, .25, .68, "Queue\nstate\n$R,Q,Y$\n$W,D$",
         face="#F7F8FA", edge="#A7ADBA")
    for label, y in [("Corridor A", .75), ("Corridor B", .50), ("Corridor C", .25)]:
        _box(ax, .69, y-.07, .29, .14, label, face=BLUE_LIGHT, edge=BLUE)
        _arrow(ax, (.61, y), (.68, y), color=BLUE)


def _information_loop(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    steps = [
        ("Observe\nqueues + saturated\nthroughput", BLUE_LIGHT, BLUE),
        ("Estimate\npersistent capacity", ORANGE_LIGHT, ORANGE),
        ("Predict\ncommon spread; prior\nor estimated center", BLUE_LIGHT, BLUE),
        ("Choose\nbest of 16 candidates", ORANGE_LIGHT, ORANGE),
        ("Guide\nroute shares for 120 s", BLUE_LIGHT, BLUE),
    ]
    ys = [.84, .66, .48, .30, .12]
    for i, ((label, face, edge), y) in enumerate(zip(steps, ys, strict=True)):
        _box(ax, .015, y-.074, .85, .148, label, face=face, edge=edge)
        if i < len(ys)-1:
            _arrow(ax, (.44, y-.078), (.44, ys[i+1]+.078), linewidth=.8)
    ax.plot([.88, .94, .94], [.12, .12, .84], color=NEUTRAL, linewidth=.8)
    _arrow(ax, (.94, .84), (.875, .84), color=NEUTRAL, linewidth=.8)
    ax.text(.985, .48, "Feedback", fontsize=7.2, color=INK, rotation=90,
            ha="center", va="center")


def _luminance(rgb):
    rgb = np.asarray(rgb[:3])
    linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
    return float(linear @ np.asarray([.2126, .7152, .0722]))


def figure1(tables: Path):
    """Preserve the three original views, with legible labels and exact data."""
    use_style()
    fig = plt.figure(figsize=(WIDTH_IN, 2.88))
    network = fig.add_axes([.008, .09, .293, .79])
    loop = fig.add_axes([.322, .09, .255, .79])
    heatmap = fig.add_axes([.680, .17, .301, .68])
    _network(network)
    _information_loop(loop)
    title(network, "a", "Queue benchmark")
    title(loop, "b", "Center-only ablation")
    title(heatmap, "c", "Information value")

    frame = pd.read_csv(tables / "information_regime_map_summary.csv")
    sigmas = sorted(frame.capacity_log_sigma.unique().tolist(), reverse=True)
    compliance = sorted(frame.compliance_mean.unique().tolist())
    values = np.empty((len(sigmas), len(compliance)))
    low = np.empty_like(values)
    high = np.empty_like(values)
    for i, sigma in enumerate(sigmas):
        for j, rho in enumerate(compliance):
            row = frame[np.isclose(frame.capacity_log_sigma, sigma)
                        & np.isclose(frame.compliance_mean, rho)].iloc[0]
            values[i, j] = row.mean_difference_percent
            low[i, j] = row.mean_ci95_low
            high[i, j] = row.mean_ci95_high
    limit = max(5., float(np.ceil(np.max(np.abs(values)))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0., vmax=limit)
    cmap = LinearSegmentedColormap.from_list(
        "camera_ready_info", [BLUE, BLUE_LIGHT, "#FFFFFF", ORANGE_LIGHT, ORANGE])
    heatmap.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    unresolved = (low <= 0) & (high >= 0)
    below_zero = high < 0
    for i in range(len(sigmas)):
        for j in range(len(compliance)):
            rgb = cmap(norm(values[i, j]))
            luminance = _luminance(rgb)
            # Select the higher-contrast text color, rather than a value cutoff.
            color = "white" if luminance < .179 else "#000000"
            heatmap.text(j, i, f"{values[i,j]:+.1f}{'*' if unresolved[i,j] else ''}",
                         ha="center", va="center", fontsize=8.0, color=color)
            if below_zero[i, j]:
                heatmap.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False,
                                           edgecolor=BLUE, linewidth=1.3))
    heatmap.set_xticks(np.arange(len(compliance)), [f"{x:.2g}" for x in compliance])
    heatmap.set_yticks(np.arange(len(sigmas)), [f"{x:.2g}" for x in sigmas])
    heatmap.set_xlabel("Mean compliance", labelpad=4)
    heatmap.set_ylabel(r"Capacity uncertainty, $\sigma$", labelpad=4)
    heatmap.set_xticks(np.arange(-.5, len(compliance), 1), minor=True)
    heatmap.set_yticks(np.arange(-.5, len(sigmas), 1), minor=True)
    heatmap.grid(which="minor", color="white", linewidth=.8)
    heatmap.tick_params(which="minor", bottom=False, left=False)
    data = {"capacity_log_sigma": sigmas, "compliance_mean": compliance,
            "mean_difference_percent": values.tolist(),
            "mean_ci95_low_seconds": low.tolist(), "mean_ci95_high_seconds": high.tolist(),
            "interval_includes_zero": unresolved.tolist(),
            "interval_entirely_below_zero": below_zero.tolist(),
            "color_limits_percent": [-limit, limit]}
    return fig, data


def figure3(tables: Path):
    """Preserve every robustness effect and CI on a compact, fixed canvas."""
    use_style()
    fig = plt.figure(figsize=(WIDTH_IN, 2.70))
    dt_ax = fig.add_axes([.070, .20, .255, .57])
    topo_ax = fig.add_axes([.490, .20, .155, .57])
    fault_ax = fig.add_axes([.790, .20, .195, .57])
    title(dt_ax, "a", "Time-step sensitivity")
    fig.text(.395, .93, "(b) Network families", fontsize=8.5, weight="bold", color=INK)
    title(fault_ax, "c", "Estimate faults")
    # Move (a)/(c) titles to the same top baseline as the compact middle panel.
    dt_ax.set_title("", loc="left")
    fault_ax.set_title("", loc="left")
    fig.text(.070, .93, "(a) Time-step sensitivity", fontsize=8.5, weight="bold", color=INK)
    fig.text(.745, .93, "(c) Estimate faults", fontsize=8.5, weight="bold", color=INK)

    dt = pd.read_csv(tables / "time_step_sensitivity_effects.csv")
    dt_data = []
    series = [("sensing-vs-unconditioned", BLUE, "o", "-", "C − U"),
              ("conditioned-vs-reactive", ORANGE, "s", "--", "C − reactive")]
    for key, color, marker, linestyle, label in series:
        subset = dt[dt.comparison == key].sort_values("time_step_seconds")
        xs = subset.time_step_seconds.to_numpy(dtype=float)
        dt_ax.plot(xs, subset.mean_difference, color=color, linestyle=linestyle, linewidth=.9)
        for row in subset.itertuples():
            effect(dt_ax, row.mean_difference, row.mean_ci95_low, row.mean_ci95_high,
                   row.time_step_seconds, color=color, marker=marker, horizontal=False)
        dt_data.append({"comparison": key, "time_step_seconds": xs.tolist(),
                        "mean_difference": subset.mean_difference.tolist(),
                        "mean_ci95_low": subset.mean_ci95_low.tolist(),
                        "mean_ci95_high": subset.mean_ci95_high.tolist()})
    zero(dt_ax, horizontal=True)
    dt_ax.set(xlim=(10.8, 1.7), ylim=(-55, 6), ylabel=r"Paired effect $\Delta T$ (s)")
    dt_ax.set_xticks([10., 5., 2.5], ["10", "5", "2.5"])
    dt_ax.set_yticks([-40, -20, 0])
    dt_ax.set_xlabel(r"Time step $\Delta t$ (s)", labelpad=4)
    handles = [Line2D([], [], color=color, marker=marker, linestyle=style, label=label,
                      linewidth=.9, markersize=4.)
               for _, color, marker, style, label in series]
    dt_ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-.01, 1.015),
                 frameon=False, ncol=1, borderaxespad=0, handlelength=1.5,
                 labelspacing=.25, handletextpad=.5)
    dt_ax.grid(axis="y", color=GRID)
    despine(dt_ax)

    topology = pd.read_csv(tables / "topology_generalization_effects.csv")
    order = ["base_asymmetric", "symmetric", "path_bottleneck", "exit_bottleneck", "expanded_venue"]
    labels = ["Base asymmetric", "Symmetric", "Path bottleneck", "Exit bottleneck", "Expanded 5×4"]
    topology = topology.set_index("topology").loc[order].reset_index()
    ys = np.arange(len(topology))[::-1]
    zero(topo_ax)
    for y, row in zip(ys, topology.itertuples(), strict=True):
        effect(topo_ax, row.mean_difference, row.mean_ci95_low, row.mean_ci95_high,
               float(y), color=BLUE)
        if row.topology in ("path_bottleneck", "exit_bottleneck"):
            topo_ax.text(.02, y+.28,
                         f"{row.mean_difference:.1f} [{row.mean_ci95_low:.1f}, {row.mean_ci95_high:.1f}]",
                         transform=topo_ax.get_yaxis_transform(), fontsize=7.2,
                         ha="left", va="bottom", color=INK)
    topo_ax.set_yticks(ys, labels)
    topo_ax.set(xlim=(-54, 4), ylim=(-.55, 4.6))
    topo_ax.set_xticks([-40, -20, 0])
    topo_ax.set_xlabel("C − U (s)", labelpad=4)
    topo_ax.grid(axis="x", color=GRID)
    despine(topo_ax)

    faults = pd.read_csv(tables / "sensor_condition_effects.csv")
    fault_order = ["clean", "noise_sigma_015", "delay_60s", "dropout_20pct", "bias_plus_10pct"]
    fault_labels = ["Clean", "Noise 0.15", "Delay 60 s", "Dropout 20%", "Bias +10%"]
    fault_data = []
    zero(fault_ax)
    regimes = [("default", ORANGE, "o", .18, "Default"),
               ("high_information", BLUE, "s", -.18, "High-info")]
    for regime, color, marker, offset, label in regimes:
        subset = faults[faults.regime == regime].set_index("fault").loc[fault_order].reset_index()
        for y, row in zip(np.arange(5)[::-1], subset.itertuples(), strict=True):
            effect(fault_ax, row.mean_difference, row.mean_ci95_low, row.mean_ci95_high,
                   float(y)+offset, color=color, marker=marker)
        fault_data.append({"regime": regime, "fault": fault_order,
                           "mean_difference": subset.mean_difference.tolist(),
                           "mean_ci95_low": subset.mean_ci95_low.tolist(),
                           "mean_ci95_high": subset.mean_ci95_high.tolist()})
    fault_ax.set_yticks(np.arange(5)[::-1], fault_labels)
    fault_ax.set(xlim=(-110, 3), ylim=(-.6, 4.65))
    fault_ax.set_xticks([-100, -50, 0])
    fault_ax.set_xlabel("C − U (s)", labelpad=4)
    handles = [Line2D([], [], color=color, marker=marker, linestyle="", label=label,
                      markerfacecolor=color, markersize=4.4)
               for _, color, marker, _, label in regimes]
    fault_ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-.02, 1.015),
                    frameon=False, ncol=1, borderaxespad=0, handlelength=1.2,
                    labelspacing=.25, handletextpad=.5)
    fault_ax.grid(axis="x", color=GRID)
    despine(fault_ax)
    data = {"time_step": dt_data,
            "topology": topology[["topology", "mean_difference", "mean_ci95_low", "mean_ci95_high"]].to_dict("records"),
            "estimate_faults": fault_data}
    return fig, data
