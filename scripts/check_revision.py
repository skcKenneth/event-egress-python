"""Check immutable provenance and replay two original primary seeds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from egress_sim.config import load_spec
from egress_sim.figure_protocol import validate_frozen_artifact_bundle
from egress_sim.policies import build_policy
from egress_sim.simulation import run_simulation


def main() -> None:
    validate_frozen_artifact_bundle(ROOT / "paper_artifacts")
    for path in (ROOT / "paper" / "frozen_tables").glob("*.csv"):
        reference = ROOT / "paper_artifacts" / "tables" / path.name
        if path.read_bytes() != reference.read_bytes():
            raise AssertionError(f"Imported source table differs: {path.name}")
    raw = pd.read_csv(ROOT / "paper_artifacts/tables/information_value_validation_runs.csv")
    spec = load_spec(ROOT / "config/default.json")
    replay = []
    for seed in (100000, 100001):
        for kind in ("deterministic_load_balance", "unconditioned_mean_only",
                     "conditioned_mean_equal_spread"):
            result = run_simulation(spec, build_policy(kind), seed=seed).summary()
            expected = raw[(raw.scenario_seed == seed) & (raw.policy == kind)].iloc[0]
            for key in ("clearance_time_seconds", "mean_clearance_time_seconds",
                        "peak_density_per_m2", "capacity_mae", "path_capacity_mae",
                        "exit_capacity_mae", "capacity_observed_fraction"):
                if not np.isclose(result[key], expected[key], atol=1e-8, rtol=1e-10):
                    raise AssertionError((seed, kind, key, result[key], expected[key]))
            if not result["completed"]:
                raise AssertionError(f"Original replay incomplete: {seed} {kind}")
            replay.append({"seed": seed, "policy": kind,
                           "clearance_time_seconds": result["clearance_time_seconds"]})
    revision = ROOT / "revision_artifacts/2026-11"
    manifest = revision / "ARTIFACTS.sha256"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            if hashlib.sha256((revision / name).read_bytes()).hexdigest() != digest:
                raise AssertionError(f"Revision artifact changed: {name}")
    report = {"frozen_bundle": "passed", "imported_source_tables": "passed",
              "primary_seed_replays": replay, "revision_manifest": manifest.exists()}
    output = ROOT / "outputs/qa"
    output.mkdir(parents=True, exist_ok=True)
    (output / "revision_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
