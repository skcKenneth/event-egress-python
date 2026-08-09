from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

from _bootstrap import DEFAULT_WORKERS, ROOT
from render_ieee_figures import render_all
from run_estimator_diagnostics import run as run_estimator_diagnostics
from run_information_regime_map import run as run_information_map
from run_information_value_validation import run as run_final_validation
from run_sensor_ablation import run as run_sensor_ablation
from run_time_step_sensitivity import run as run_time_step_sensitivity
from run_topology_generalization import run as run_topology_generalization

from egress_sim.figure_protocol import (
    REFERENCE_PROTOCOL,
    REQUIRED_TABLES,
    validate_protocol_structure,
    validate_reference_figure_inputs,
    write_reference_manifest,
)


def _effect_row(frame: pd.DataFrame, label: str) -> pd.Series:
    row = frame.loc[frame["comparison"] == label]
    if len(row) != 1:
        raise ValueError(f"expected one effect row for {label!r}, found {len(row)}")
    return row.iloc[0]


def _write_results_summary(artifact_root: Path) -> None:
    """Write a compact human-readable index of the frozen numerical results."""

    tables = artifact_root / "tables"
    validation = pd.read_csv(tables / "information_value_validation_summary.csv")
    effects = pd.read_csv(tables / "information_value_validation_effects.csv")
    topology = pd.read_csv(tables / "topology_generalization_effects.csv")
    sensor = pd.read_csv(tables / "sensor_condition_effects.csv")
    estimator = pd.read_csv(tables / "estimator_diagnostics_final.csv")

    primary = _effect_row(effects, "primary-sensing-value")
    contracted = _effect_row(effects, "spread-contraction-diagnostic")
    oracle = _effect_row(effects, "oracle-residual-information")
    policy_means = validation.set_index("policy")["clearance_mean_s"]
    resolved_topologies = int((topology["mean_ci95_high"] < 0.0).sum())

    sensor_lines: list[str] = []
    for regime in ("default", "high_information"):
        subset = sensor.loc[sensor["regime"] == regime]
        clean = subset.loc[subset["fault"] == "clean"].iloc[0]
        faults = subset.loc[subset["fault"] != "clean", "mean_difference"]
        sensor_lines.append(
            f"- `{regime}`: clean {clean['mean_difference']:+.2f} s; "
            f"fault range {faults.min():+.2f} to {faults.max():+.2f} s."
        )

    estimator_lines: list[str] = []
    for regime in ("default", "high_information"):
        subset = estimator.loc[estimator["regime"] == regime]
        estimator_lines.append(
            f"- `{regime}`: mean final capacity MAE "
            f"{subset['capacity_mae'].mean():.4f}; mean observed-link fraction "
            f"{100.0 * subset['capacity_observed_fraction'].mean():.1f}%."
        )

    text = f"""# Frozen results summary

This file is generated from `paper_artifacts/tables/`. Negative paired effects
favor the conditioned controller.

## Confirmatory held-out validation

- Static nearest mean clearance: {policy_means['static_nearest']:.2f} s.
- Reactive balance mean clearance: {policy_means['deterministic_load_balance']:.2f} s.
- Unconditioned mean clearance: {policy_means['unconditioned_mean_only']:.2f} s.
- Equal-spread conditioned mean clearance: {policy_means['conditioned_mean_equal_spread']:.2f} s.
- Oracle-capacity mean clearance: {policy_means['oracle_capacity_mean_only']:.2f} s.
- Primary center-only effect: {primary['mean_difference']:+.2f} s
  ({primary['mean_difference_percent']:+.2f}%; 95% paired-bootstrap interval
  [{primary['mean_ci95_low']:+.2f}, {primary['mean_ci95_high']:+.2f}] s).
- Additional contracted-spread effect relative to equal-spread conditioning:
  {contracted['mean_difference']:+.2f} s
  ([{contracted['mean_ci95_low']:+.2f}, {contracted['mean_ci95_high']:+.2f}] s).
- Oracle-minus-conditioned effect: {oracle['mean_difference']:+.2f} s
  ([{oracle['mean_ci95_low']:+.2f}, {oracle['mean_ci95_high']:+.2f}] s).

## Estimator diagnostics

{chr(10).join(estimator_lines)}

## Network-family transfer

{resolved_topologies} of {len(topology)} paired-bootstrap intervals lie entirely
below zero. Point estimates range from {topology['mean_difference'].min():+.2f}
to {topology['mean_difference'].max():+.2f} s.

## Controller-facing estimate faults

{chr(10).join(sensor_lines)}

See the CSV files for complete seed-level outputs, uncertainty-compliance cells,
time-step diagnostics, tail summaries, and confidence intervals.
"""
    (artifact_root / "RESULTS_SUMMARY.md").write_text(text, encoding="utf-8")


def _write_artifact_checksums(artifact_root: Path) -> None:
    """Hash the complete frozen bundle except the checksum file itself."""

    included: list[Path] = []
    for directory in (artifact_root / "tables", artifact_root / "figures"):
        included.extend(path for path in directory.rglob("*") if path.is_file())
    included.extend(
        path
        for path in (
            artifact_root / "reference_manifest.json",
            artifact_root / "RESULTS_SUMMARY.md",
        )
        if path.is_file()
    )
    lines = []
    for path in sorted(included, key=lambda item: item.relative_to(artifact_root).as_posix()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(artifact_root).as_posix()
        lines.append(f"{digest}  {relative}")
    (artifact_root / "ARTIFACTS.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _copy_artifacts(source_tables: Path, source_figures: Path) -> None:
    """Atomically refresh the frozen table and figure directories after validation."""

    source_tables = Path(source_tables)
    source_figures = Path(source_figures)
    validate_protocol_structure(source_tables)
    expected_figures = {
        f"fig{index}_{name}.{extension}"
        for index, name in (
            (1, "system_overview"),
            (2, "performance_landscape"),
            (3, "context_validation"),
        )
        for extension in ("pdf", "png")
    }
    missing_figures = sorted(
        name for name in expected_figures if not (source_figures / name).exists()
    )
    if missing_figures:
        raise FileNotFoundError(
            "Missing rendered paper figures: " + ", ".join(missing_figures)
        )

    artifact_root = ROOT / "paper_artifacts"
    artifact_tables = artifact_root / "tables"
    artifact_figures = artifact_root / "figures"
    if artifact_tables.exists():
        shutil.rmtree(artifact_tables)
    if artifact_figures.exists():
        shutil.rmtree(artifact_figures)
    artifact_tables.mkdir(parents=True, exist_ok=True)
    artifact_figures.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_TABLES:
        shutil.copy2(source_tables / name, artifact_tables / name)
    for name in sorted(expected_figures):
        shutil.copy2(source_figures / name, artifact_figures / name)
    write_reference_manifest(
        artifact_tables,
        artifact_root / "reference_manifest.json",
    )
    _write_results_summary(artifact_root)
    _write_artifact_checksums(artifact_root)
    validate_reference_figure_inputs(artifact_tables)


def run(*, workers: int, refresh_artifacts: bool) -> None:
    protocol = REFERENCE_PROTOCOL

    def stage(label: str) -> None:
        print(f"[paper pipeline] {label}", flush=True)

    stage("1/8 confirmatory held-out center-only validation")
    run_final_validation(
        seed_count=protocol.validation_seeds,
        workers=workers,
    )
    stage("2/8 uncertainty-compliance map")
    run_information_map(
        seed_count=protocol.map_seeds_per_cell,
        workers=workers,
    )
    stage("3/8 time-step robustness")
    run_time_step_sensitivity(
        seed_count=protocol.time_step_seeds,
        workers=workers,
    )
    stage("4/8 estimator diagnostics")
    run_estimator_diagnostics(
        seed_count=protocol.estimator_seeds_per_regime,
        workers=workers,
    )
    stage("5/8 network-family transfer")
    run_topology_generalization(
        seed_count=protocol.topology_seeds_per_family,
        workers=workers,
    )
    stage("6/8 observation-fault diagnostics")
    run_sensor_ablation(
        seed_count=protocol.sensor_seeds_per_regime,
        workers=workers,
    )

    tables = ROOT / "outputs" / "tables"
    figures = ROOT / "outputs" / "figures" / "ieee"
    stage("7/8 protocol validation and figure rendering")
    validate_protocol_structure(tables)
    render_all(tables, figures, strict=False)
    if refresh_artifacts:
        stage("8/8 freezing validated paper artifacts")
        _copy_artifacts(tables, figures)
        strict_check = ROOT / "outputs" / "figures" / "strict-check"
        if strict_check.exists():
            shutil.rmtree(strict_check)
        render_all(
            ROOT / "paper_artifacts" / "tables",
            strict_check,
            strict=True,
        )
        print(f"Refreshed frozen paper artifacts in {ROOT / 'paper_artifacts'}")
    else:
        print(f"Wrote generated tables to {tables}")
        print(f"Wrote generated figures to {figures}")
        print("Pass --refresh-artifacts to replace the tracked frozen artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate the frozen paper experiment and figure suite."
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--refresh-artifacts",
        action="store_true",
        help="Replace paper_artifacts after successful protocol validation.",
    )
    args = parser.parse_args()
    run(workers=args.workers, refresh_artifacts=args.refresh_artifacts)
