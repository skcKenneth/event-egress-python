from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "static_nearest": "#7f7f7f",
    "deterministic_load_balance": "#0072B2",
    "robust_rolling_horizon": "#D55E00",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def plot_baseline_comparison(frame: pd.DataFrame, output: Path) -> None:
    _style()
    policies = list(COLORS)
    labels = ["Static nearest", "Reactive balance", "Robust rolling horizon"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    clearance = [
        frame.loc[frame.policy == policy, "clearance_time_seconds"].to_numpy() / 60.0
        for policy in policies
    ]
    density = [
        frame.loc[frame.policy == policy, "peak_density_per_m2"].to_numpy()
        for policy in policies
    ]
    boxes = axes[0].boxplot(clearance, labels=labels, patch_artist=True, showmeans=True)
    for patch, policy in zip(boxes["boxes"], policies):
        patch.set_facecolor(COLORS[policy])
        patch.set_alpha(0.72)
    boxes = axes[1].boxplot(density, labels=labels, patch_artist=True, showmeans=True)
    for patch, policy in zip(boxes["boxes"], policies):
        patch.set_facecolor(COLORS[policy])
        patch.set_alpha(0.72)
    axes[0].set_ylabel("Clearance time (min)")
    axes[1].set_ylabel(r"Peak queue-density proxy (persons/m$^2$)")
    axes[1].axhline(3.5, color="black", linestyle="--", linewidth=1, label="Design threshold")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(
    summary: pd.DataFrame,
    parameter: str,
    x_label: str,
    output: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for policy, subset in summary.groupby("policy"):
        subset = subset.sort_values(parameter)
        color = COLORS.get(policy, "black")
        axes[0].plot(
            subset[parameter],
            subset["clearance_p90_s"] / 60.0,
            marker="o",
            color=color,
            label=policy.replace("_", " "),
        )
        axes[1].plot(
            subset[parameter],
            subset["peak_density_mean"],
            marker="o",
            color=color,
            label=policy.replace("_", " "),
        )
    axes[0].set_ylabel("90th-percentile clearance (min)")
    axes[1].set_ylabel(r"Mean peak queue-density proxy (persons/m$^2$)")
    for axis in axes:
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(histories: dict[str, pd.DataFrame], output: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for policy, frame in histories.items():
        color = COLORS[policy]
        axes[0].plot(
            frame.time_seconds / 60.0,
            frame.active_population,
            color=color,
            label=policy.replace("_", " "),
        )
        axes[1].plot(
            frame.time_seconds / 60.0,
            frame.density_per_m2,
            color=color,
            label=policy.replace("_", " "),
        )
    axes[0].set_xlabel("Time (min)")
    axes[0].set_ylabel("People remaining in system")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel(r"Queue-density proxy (persons/m$^2$)")
    axes[1].axhline(3.5, color="black", linestyle="--", linewidth=1)
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(summary: pd.DataFrame, output: Path) -> None:
    _style()
    label_map = {
        "robust_rolling_horizon": "Sensor-conditioned",
        "robust_without_sensor_conditioning": "Unconditioned ablation",
    }
    labels = [label_map.get(name, name.replace("_", " ")) for name in summary.policy]
    colors = [
        COLORS["robust_rolling_horizon"]
        if name == "robust_rolling_horizon"
        else "#999999"
        for name in summary.policy
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    axes[0].bar(x, summary.clearance_p90_s / 60.0, color=colors)
    axes[1].bar(x, summary.peak_density_mean, color=colors)
    axes[0].set_ylabel("90th-percentile clearance (min)")
    axes[1].set_ylabel(r"Mean peak queue-density proxy (persons/m$^2$)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=16)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
