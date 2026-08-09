from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


class FigureProtocolError(ValueError):
    """Raised when paper artifacts do not match the frozen protocol."""


@dataclass(frozen=True)
class FigureProtocol:
    validation_seed_start: int = 100_000
    validation_seeds: int = 200
    map_seeds_per_cell: int = 50
    time_step_seeds: int = 30
    estimator_seeds_per_regime: int = 50
    topology_seeds_per_family: int = 50
    sensor_seeds_per_regime: int = 30


REFERENCE_PROTOCOL = FigureProtocol()

REQUIRED_TABLES = (
    "information_value_validation_runs.csv",
    "information_value_validation_summary.csv",
    "information_value_validation_effects.csv",
    "information_regime_map_runs.csv",
    "information_regime_map_summary.csv",
    "time_step_sensitivity_runs.csv",
    "time_step_sensitivity_summary.csv",
    "time_step_sensitivity_effects.csv",
    "estimator_diagnostics_runs.csv",
    "estimator_diagnostics_final.csv",
    "estimator_diagnostics_summary.csv",
    "topology_generalization_runs.csv",
    "topology_generalization_summary.csv",
    "topology_generalization_effects.csv",
    "sensor_ablation_runs.csv",
    "sensor_ablation_summary.csv",
    "sensor_condition_effects.csv",
)

FINAL_POLICIES = {
    "static_nearest",
    "deterministic_load_balance",
    "unconditioned_mean_only",
    "conditioned_mean_equal_spread",
    "conditioned_mean_shrunk_spread",
    "oracle_capacity_mean_only",
}
MAP_POLICIES = {
    "unconditioned_mean_only",
    "conditioned_mean_equal_spread",
}
TIME_STEP_POLICIES = {
    "deterministic_load_balance",
    "unconditioned_mean_only",
    "conditioned_mean_equal_spread",
}
TOPOLOGY_POLICIES = MAP_POLICIES
SENSOR_POLICIES = {
    "unconditioned_mean_only",
    "conditioned_clean",
    "conditioned_noise_sigma_015",
    "conditioned_delay_60s",
    "conditioned_dropout_20pct",
    "conditioned_bias_plus_10pct",
}
TOPOLOGIES = {
    "base_asymmetric",
    "symmetric",
    "path_bottleneck",
    "exit_bottleneck",
    "expanded_venue",
}
SENSOR_REGIMES = {"default", "high_information"}
SENSOR_FAULTS = {
    "clean",
    "noise_sigma_015",
    "delay_60s",
    "dropout_20pct",
    "bias_plus_10pct",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureProtocolError(message)


def _read(tables: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(tables / name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_reference_manifest(tables: Path, manifest_path: Path) -> None:
    tables = Path(tables)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": asdict(REFERENCE_PROTOCOL),
        "sha256": {name: _sha256(tables / name) for name in REQUIRED_TABLES},
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_hashes(tables: Path) -> None:
    manifest_path = tables.parent / "reference_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing frozen artifact manifest: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        payload.get("protocol") == asdict(REFERENCE_PROTOCOL),
        "Reference manifest protocol does not match the code protocol.",
    )
    expected = payload.get("sha256", {})
    _require(
        set(expected) == set(REQUIRED_TABLES),
        "Reference manifest table set is incomplete or stale.",
    )
    for name in REQUIRED_TABLES:
        actual = _sha256(tables / name)
        _require(
            actual == expected[name],
            f"Frozen table hash mismatch for {name}.",
        )



def _validate_run_table(
    frame: pd.DataFrame,
    *,
    label: str,
    grouping: list[str],
) -> None:
    """Check generic run-level invariants before study-specific validation."""

    required = {
        "scenario_seed",
        "policy_seed",
        "policy",
        "completed",
        "mass_balance_error",
        "clearance_time_seconds",
    }
    missing = required - set(frame.columns)
    _require(not missing, f"{label} is missing columns: {sorted(missing)}")
    _require(bool(frame["completed"].all()), f"{label} contains incomplete runs.")
    _require(
        float(frame["mass_balance_error"].max()) <= 1e-7,
        f"{label} contains a mass-balance error above 1e-7.",
    )
    _require(
        bool((frame["policy_seed"] == frame["scenario_seed"] + 1_000_003).all()),
        f"{label} does not preserve the frozen policy-seed offset.",
    )
    duplicate_columns = [*grouping, "scenario_seed", "policy"]
    _require(
        not bool(frame.duplicated(duplicate_columns).any()),
        f"{label} contains duplicate run rows for {duplicate_columns}.",
    )
    _require(
        bool((frame["clearance_time_seconds"] > 0).all()),
        f"{label} contains non-positive clearance times.",
    )

def validate_protocol_structure(tables: Path) -> None:
    """Validate seed counts, policy sets, regimes, and frozen seed ranges."""
    tables = Path(tables)
    missing = [name for name in REQUIRED_TABLES if not (tables / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing paper figure-data tables:\n  - " + "\n  - ".join(missing)
        )
    protocol = REFERENCE_PROTOCOL

    final_runs = _read(tables, "information_value_validation_runs.csv")
    _validate_run_table(final_runs, label="Final validation", grouping=[])
    final_counts = final_runs.groupby("policy")["scenario_seed"].nunique()
    _require(
        set(final_counts.index) == FINAL_POLICIES,
        "Final-validation policy set does not match the frozen protocol.",
    )
    _require(
        bool((final_counts == protocol.validation_seeds).all()),
        f"Final validation requires {protocol.validation_seeds} seeds per policy.",
    )
    expected_validation_seeds = set(
        range(
            protocol.validation_seed_start,
            protocol.validation_seed_start + protocol.validation_seeds,
        )
    )
    _require(
        set(final_runs["scenario_seed"].unique()) == expected_validation_seeds,
        "Final-validation seed range does not match the frozen confirmatory protocol.",
    )

    map_runs = _read(tables, "information_regime_map_runs.csv")
    _validate_run_table(
        map_runs,
        label="Information map",
        grouping=["capacity_log_sigma", "compliance_mean"],
    )
    map_counts = map_runs.groupby(
        ["capacity_log_sigma", "compliance_mean", "policy"]
    )["scenario_seed"].nunique()
    _require(
        bool((map_counts == protocol.map_seeds_per_cell).all()),
        f"Information map requires {protocol.map_seeds_per_cell} seeds per cell and policy.",
    )
    _require(
        set(map_runs["policy"].unique()) == MAP_POLICIES,
        "Information-map policy set does not match the frozen protocol.",
    )
    _require(len(map_counts) == 32, "Information map must contain 16 paired cells.")
    _require(
        set(map_runs["scenario_seed"].unique()) == set(range(96_000, 96_050)),
        "Information-map seed range must be 96000--96049.",
    )

    time_runs = _read(tables, "time_step_sensitivity_runs.csv")
    _validate_run_table(
        time_runs, label="Time-step study", grouping=["time_step_seconds"]
    )
    time_counts = time_runs.groupby(
        ["time_step_seconds", "policy"]
    )["scenario_seed"].nunique()
    _require(
        bool((time_counts == protocol.time_step_seeds).all()),
        f"Time-step study requires {protocol.time_step_seeds} seeds per step and policy.",
    )
    _require(
        set(time_runs["time_step_seconds"].unique()) == {10.0, 5.0, 2.5},
        "Time-step study must contain 10, 5, and 2.5 second steps.",
    )
    _require(
        set(time_runs["policy"].unique()) == TIME_STEP_POLICIES,
        "Time-step policy set does not match the frozen protocol.",
    )
    _require(
        set(time_runs["scenario_seed"].unique())
        == set(range(92_000, 92_000 + protocol.time_step_seeds)),
        "Time-step seed range does not match the frozen protocol.",
    )

    estimator = _read(tables, "estimator_diagnostics_final.csv")
    _validate_run_table(estimator, label="Estimator diagnostics", grouping=["regime"])
    estimator_counts = estimator.groupby("regime")["scenario_seed"].nunique()
    _require(
        set(estimator_counts.index) == SENSOR_REGIMES,
        "Estimator diagnostics must contain default and high-information regimes.",
    )
    _require(
        bool((estimator_counts == protocol.estimator_seeds_per_regime).all()),
        f"Estimator diagnostics require {protocol.estimator_seeds_per_regime} seeds per regime.",
    )
    _require(
        set(estimator["scenario_seed"].unique())
        == set(range(97_000, 97_000 + protocol.estimator_seeds_per_regime)),
        "Estimator-diagnostic seed range does not match the frozen protocol.",
    )

    topology = _read(tables, "topology_generalization_runs.csv")
    _validate_run_table(topology, label="Topology study", grouping=["topology"])
    topology_counts = topology.groupby(
        ["topology", "policy"]
    )["scenario_seed"].nunique()
    _require(
        set(topology["topology"].unique()) == TOPOLOGIES,
        "Topology study does not contain all frozen network families.",
    )
    _require(
        bool((topology_counts == protocol.topology_seeds_per_family).all()),
        f"Topology study requires {protocol.topology_seeds_per_family} seeds per family and policy.",
    )
    _require(
        set(topology["policy"].unique()) == TOPOLOGY_POLICIES,
        "Topology-study policy set does not match the frozen protocol.",
    )
    _require(
        set(topology["scenario_seed"].unique())
        == set(range(98_000, 98_000 + protocol.topology_seeds_per_family)),
        "Topology-study seed range does not match the frozen protocol.",
    )

    sensor = _read(tables, "sensor_ablation_runs.csv")
    _validate_run_table(sensor, label="Sensor study", grouping=["regime"])
    _require(
        set(sensor["regime"].unique()) == SENSOR_REGIMES,
        "Sensor study must contain both frozen regimes.",
    )
    conditioned = sensor[sensor["policy"] != "unconditioned_mean_only"]
    _require(
        set(conditioned["fault"].unique()) == SENSOR_FAULTS,
        "Sensor study does not contain all frozen fault conditions.",
    )
    sensor_counts = sensor.groupby(
        ["regime", "policy"]
    )["scenario_seed"].nunique()
    _require(
        bool((sensor_counts == protocol.sensor_seeds_per_regime).all()),
        f"Sensor study requires {protocol.sensor_seeds_per_regime} seeds per regime and policy.",
    )
    _require(
        set(sensor["policy"].unique()) == SENSOR_POLICIES,
        "Sensor-study policy set does not match the frozen protocol.",
    )
    _require(
        set(sensor["scenario_seed"].unique())
        == set(range(99_000, 99_000 + protocol.sensor_seeds_per_regime)),
        "Sensor-study seed range does not match the frozen protocol.",
    )


def validate_reference_figure_inputs(tables: Path) -> None:
    """Validate the bundled frozen tables structurally and byte-for-byte."""
    tables = Path(tables)
    validate_protocol_structure(tables)
    _validate_hashes(tables)


def validate_frozen_artifact_bundle(artifact_root: Path) -> None:
    """Validate the table manifest and the complete frozen-bundle checksums."""

    artifact_root = Path(artifact_root)
    validate_reference_figure_inputs(artifact_root / "tables")
    checksum_path = artifact_root / "ARTIFACTS.sha256"
    if not checksum_path.exists():
        raise FileNotFoundError(f"Missing frozen artifact checksum file: {checksum_path}")

    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise FigureProtocolError(
                f"Malformed checksum line {line_number} in {checksum_path}."
            ) from error
        _require(
            len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
            f"Invalid SHA-256 digest on line {line_number} in {checksum_path}.",
        )
        _require(relative not in entries, f"Duplicate checksum entry for {relative}.")
        entries[relative] = digest

    required_bundle_entries = {
        *(f"tables/{name}" for name in REQUIRED_TABLES),
        "figures/fig1_system_overview.pdf",
        "figures/fig1_system_overview.png",
        "figures/fig2_performance_landscape.pdf",
        "figures/fig2_performance_landscape.png",
        "figures/fig3_context_validation.pdf",
        "figures/fig3_context_validation.png",
        "reference_manifest.json",
        "RESULTS_SUMMARY.md",
    }
    _require(
        set(entries) == required_bundle_entries,
        "Frozen artifact checksum entry set is incomplete or stale.",
    )
    for relative, expected in entries.items():
        path = artifact_root / relative
        _require(path.is_file(), f"Missing frozen artifact listed in checksums: {relative}.")
        _require(
            _sha256(path) == expected,
            f"Frozen artifact hash mismatch for {relative}.",
        )
