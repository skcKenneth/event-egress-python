from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class FigureProtocolError(ValueError):
    """Raised when figure inputs do not match the frozen reference protocol."""


@dataclass(frozen=True)
class FigureProtocol:
    validation_seeds: int = 200
    map_seeds_per_cell: int = 50
    time_step_seeds: int = 50
    residual_seeds_per_cell: int = 50
    lognormal_samples: int = 1_000_000


REFERENCE_PROTOCOL = FigureProtocol()

REQUIRED_TABLES = (
    "information_regime_map_runs.csv",
    "information_regime_map_summary.csv",
    "information_value_validation_runs.csv",
    "information_value_validation_effects.csv",
    "information_value_validation_summary.csv",
    "time_step_sensitivity_runs.csv",
    "time_step_sensitivity_effects.csv",
    "residual_spread_ablation_runs.csv",
    "residual_spread_ablation_effects.csv",
    "clipped_lognormal_diagnostic.csv",
)

# Rounded values printed in the three bundled IEEE figures.  Keeping the
# reference here prevents a quick or partial experiment from silently replacing
# the validated figure data.
REFERENCE_MAP_PERCENT = {
    (0.45, 0.35): -1.5,
    (0.45, 0.50): -3.3,
    (0.45, 0.68): -6.7,
    (0.45, 0.85): -9.6,
    (0.30, 0.35): -1.4,
    (0.30, 0.50): -2.3,
    (0.30, 0.68): -4.5,
    (0.30, 0.85): -6.4,
    (0.15, 0.35): -0.1,
    (0.15, 0.50): -0.3,
    (0.15, 0.68): -0.7,
    (0.15, 0.85): -1.2,
    (0.00, 0.35): -0.0,
    (0.00, 0.50): +0.2,
    (0.00, 0.68): -0.1,
    (0.00, 0.85): +0.3,
}

REFERENCE_POLICY_MEANS = {
    "static_nearest": 966.65,
    "deterministic_load_balance": 715.60,
    "unconditioned_mean_only": 722.65,
    "scenario_mean_only": 702.95,
    "oracle_capacity_mean_only": 699.15,
    "unconditioned_mean_cvar": 726.80,
    "conditioned_mean_cvar": 702.70,
}

REFERENCE_TIME_STEP_EFFECTS = {
    (10.0, "sensing-vs-unconditioned"): -22.80,
    (5.0, "sensing-vs-unconditioned"): -26.90,
    (2.5, "sensing-vs-unconditioned"): -25.95,
    (10.0, "conditioned-vs-reactive"): -7.40,
    (5.0, "conditioned-vs-reactive"): -12.70,
    (2.5, "conditioned-vs-reactive"): -12.05,
}

REFERENCE_RESIDUAL_ROUNDED = {
    (0.00, 0.00): 0,
    (0.15, 0.00): -16,
    (0.22, 0.00): -31,
    (0.45, 0.00): -90,
    (0.00, 0.05): 1,
    (0.15, 0.05): -16,
    (0.22, 0.05): -29,
    (0.45, 0.05): -85,
    (0.00, 0.10): 1,
    (0.15, 0.10): -16,
    (0.22, 0.10): -28,
    (0.45, 0.10): -85,
    (0.00, 0.20): 4,
    (0.15, 0.20): -11,
    (0.22, 0.20): -23,
    (0.45, 0.20): -77,
}

REFERENCE_CLIPPED_MEANS = {
    0.00: 1.000,
    0.15: 0.998,
    0.22: 0.990,
    0.30: 0.971,
    0.45: 0.927,
}


def _read(tables: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(tables / name)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureProtocolError(message)


def _close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0.0):
        raise FigureProtocolError(
            f"{label}: expected {expected}, found {actual}. "
            "The table was probably generated with a non-reference seed count "
            "or a changed configuration."
        )


def validate_reference_figure_inputs(tables: Path) -> None:
    """Validate seed counts and printed values for the frozen IEEE figure set."""
    tables = Path(tables)
    missing = [name for name in REQUIRED_TABLES if not (tables / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing IEEE figure-data tables:\n  - " + "\n  - ".join(missing)
        )

    protocol = REFERENCE_PROTOCOL

    # Final validation: seven policies, each on the same 200 seeds.
    final_runs = _read(tables, "information_value_validation_runs.csv")
    final_counts = final_runs.groupby("policy")["scenario_seed"].nunique()
    _require(
        set(final_counts.index) == set(REFERENCE_POLICY_MEANS),
        "Final-validation policy set does not match the reference figure set.",
    )
    _require(
        bool((final_counts == protocol.validation_seeds).all()),
        f"Final-validation figures require {protocol.validation_seeds} seeds per policy.",
    )
    _require(
        set(final_runs["scenario_seed"].unique()) == set(range(90_000, 90_200)),
        "Final-validation seed range must be 90000--90199.",
    )
    final_summary = _read(tables, "information_value_validation_summary.csv").set_index("policy")
    for policy, expected in REFERENCE_POLICY_MEANS.items():
        _close(
            float(final_summary.loc[policy, "clearance_mean_s"]),
            expected,
            1e-9,
            f"Final mean for {policy}",
        )

    # Information-region map: 50 paired seeds for every one of 16 cells.
    map_runs = _read(tables, "information_regime_map_runs.csv")
    map_counts = map_runs.groupby(
        ["capacity_log_sigma", "compliance_mean", "policy"]
    )["scenario_seed"].nunique()
    _require(
        bool((map_counts == protocol.map_seeds_per_cell).all()),
        f"Information map requires {protocol.map_seeds_per_cell} seeds per cell and policy.",
    )
    _require(
        set(map_runs["scenario_seed"].unique()) == set(range(96_000, 96_050)),
        "Information-map seed range must be 96000--96049.",
    )
    map_summary = _read(tables, "information_regime_map_summary.csv")
    _require(len(map_summary) == 16, "Information map must contain exactly 16 cells.")
    _require(
        bool((map_summary["runs"] == protocol.map_seeds_per_cell).all()),
        f"Information-map summary must report {protocol.map_seeds_per_cell} runs per cell.",
    )
    for (sigma, compliance), expected in REFERENCE_MAP_PERCENT.items():
        row = map_summary[
            np.isclose(map_summary["capacity_log_sigma"], sigma)
            & np.isclose(map_summary["compliance_mean"], compliance)
        ]
        _require(len(row) == 1, f"Missing information-map cell ({sigma}, {compliance}).")
        actual = round(float(row.iloc[0]["mean_difference_percent"]), 1)
        _close(actual, expected, 1e-12, f"Information-map cell ({sigma}, {compliance})")

    # Time-step sensitivity.
    time_runs = _read(tables, "time_step_sensitivity_runs.csv")
    time_counts = time_runs.groupby(["time_step_seconds", "policy"])["scenario_seed"].nunique()
    _require(
        bool((time_counts == protocol.time_step_seeds).all()),
        f"Time-step figure requires {protocol.time_step_seeds} seeds per step and policy.",
    )
    time_effects = _read(tables, "time_step_sensitivity_effects.csv")
    for (step, comparison), expected in REFERENCE_TIME_STEP_EFFECTS.items():
        row = time_effects[
            np.isclose(time_effects["time_step_seconds"], step)
            & (time_effects["comparison"] == comparison)
        ]
        _require(len(row) == 1, f"Missing time-step effect ({step}, {comparison}).")
        _close(float(row.iloc[0]["mean_difference"]), expected, 1e-9, f"Time-step effect ({step}, {comparison})")

    # Residual-spread heat map.
    residual = _read(tables, "residual_spread_ablation_effects.csv")
    _require(
        bool((residual["paired_runs"] == protocol.residual_seeds_per_cell).all()),
        f"Residual-spread figure requires {protocol.residual_seeds_per_cell} paired seeds per cell.",
    )
    for (sigma, residual_sigma), expected in REFERENCE_RESIDUAL_ROUNDED.items():
        row = residual[
            np.isclose(residual["capacity_log_sigma"], sigma)
            & np.isclose(residual["residual_sigma"], residual_sigma)
        ]
        _require(len(row) == 1, f"Missing residual-spread cell ({sigma}, {residual_sigma}).")
        actual = int(np.rint(float(row.iloc[0]["mean_difference"])))
        _require(actual == expected, f"Residual-spread cell ({sigma}, {residual_sigma}): expected {expected}, found {actual}.")

    # Clipped-lognormal diagnostic.
    clipped = _read(tables, "clipped_lognormal_diagnostic.csv")
    _require(
        bool((clipped["samples"] == protocol.lognormal_samples).all()),
        f"Clipped-lognormal figure requires {protocol.lognormal_samples:,} samples.",
    )
    for sigma, expected in REFERENCE_CLIPPED_MEANS.items():
        row = clipped[np.isclose(clipped["capacity_log_sigma"], sigma)]
        _require(len(row) == 1, f"Missing clipped-lognormal row for sigma={sigma}.")
        actual = round(float(row.iloc[0]["post_clip_mean"]), 3)
        _close(actual, expected, 1e-12, f"Post-clipping mean at sigma={sigma}")
