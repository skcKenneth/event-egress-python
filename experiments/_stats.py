from __future__ import annotations

import zlib

import numpy as np
import pandas as pd


def paired_effects(
    frame: pd.DataFrame,
    proposed: str,
    baseline: str,
    *,
    outcome: str = "clearance_time_seconds",
    samples: int = 5000,
    label: str = "paired-effect",
) -> dict[str, float]:
    seed_column = "scenario_seed" if "scenario_seed" in frame.columns else "seed"
    pivot = frame.pivot(index=seed_column, columns="policy", values=outcome).dropna()
    proposed_values = pivot[proposed].to_numpy()
    baseline_values = pivot[baseline].to_numpy()
    differences = proposed_values - baseline_values
    n = len(differences)
    if n == 0:
        raise ValueError(f"no paired runs for {proposed} versus {baseline}")
    seed = 20260621 + zlib.crc32(label.encode("utf-8"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(samples, n))
    sampled_differences = differences[draws]
    boot_mean = sampled_differences.mean(axis=1)
    boot_median = np.median(sampled_differences, axis=1)
    boot_p90 = np.quantile(proposed_values[draws], 0.90, axis=1) - np.quantile(
        baseline_values[draws], 0.90, axis=1
    )
    baseline_mean = float(np.mean(baseline_values))
    proposed_mean = float(np.mean(proposed_values))
    baseline_p90 = float(np.quantile(baseline_values, 0.90))
    proposed_p90 = float(np.quantile(proposed_values, 0.90))
    return {
        "paired_runs": n,
        "baseline_mean": baseline_mean,
        "proposed_mean": proposed_mean,
        "mean_difference": float(np.mean(differences)),
        "mean_difference_percent": float(100.0 * np.mean(differences) / baseline_mean),
        "mean_ci95_low": float(np.quantile(boot_mean, 0.025)),
        "mean_ci95_high": float(np.quantile(boot_mean, 0.975)),
        "median_difference": float(np.median(differences)),
        "median_ci95_low": float(np.quantile(boot_median, 0.025)),
        "median_ci95_high": float(np.quantile(boot_median, 0.975)),
        "baseline_p90": baseline_p90,
        "proposed_p90": proposed_p90,
        "p90_difference": proposed_p90 - baseline_p90,
        "p90_difference_percent": float(
            100.0 * (proposed_p90 - baseline_p90) / baseline_p90
        ),
        "p90_ci95_low": float(np.quantile(boot_p90, 0.025)),
        "p90_ci95_high": float(np.quantile(boot_p90, 0.975)),
        "win_rate_percent": float(100.0 * np.mean(differences < 0.0)),
        "tie_rate_percent": float(100.0 * np.mean(differences == 0.0)),
        "loss_rate_percent": float(100.0 * np.mean(differences > 0.0)),
        "paired_difference_q10": float(np.quantile(differences, 0.10)),
        "paired_difference_q90": float(np.quantile(differences, 0.90)),
        "worst_observed_difference": float(np.max(differences)),
        "best_observed_difference": float(np.min(differences)),
    }
