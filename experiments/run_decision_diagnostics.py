"""Run the prospectively fixed PIC2026 same-state action diagnostics."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT
from egress_sim.config import load_spec
from egress_sim.model import EgressState
from egress_sim.revision_diagnostics import (
    POLICY_KINDS, SNAPSHOT_SECONDS, collect_snapshots, evaluate_snapshot,
    summarize_decisions,
)


def _run_seed(args: tuple) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    seed, output = args
    spec = load_spec(ROOT / "config" / "default.json")
    eligibility, decisions, candidates, branches = [], [], [], []
    snapshot_dir = Path(output) / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for kind in POLICY_KINDS:
        for snapshot in collect_snapshots(spec, seed, kind):
            eligibility.append({
                "seed": seed, "policy": kind,
                "snapshot_seconds": snapshot.snapshot_seconds,
                "eligible": snapshot.eligible, "skip_reason": snapshot.skip_reason,
            })
            if not snapshot.eligible:
                continue
            decision, candidate, branch = evaluate_snapshot(snapshot, spec)
            decisions.append(decision)
            candidates.extend(candidate)
            branches.extend(branch)
            # Portable arrays preserve the complete physical/estimator snapshot
            # without a pickle dependency. Clean U/C policies carry no additional
            # observation memory; their pre-decision kind and RNG are recorded.
            arrays = {f"state_{f.name}": getattr(snapshot.state, f.name)
                      for f in fields(EgressState)}
            arrays.update({
                "scenario_path_capacity_multiplier": snapshot.scenario.path_capacity_multiplier,
                "scenario_exit_capacity_multiplier": snapshot.scenario.exit_capacity_multiplier,
                "scenario_compliance": snapshot.scenario.compliance,
                "policy_kind": kind,
                "rng_state_json": json.dumps(snapshot.rng_state),
                "candidates": snapshot.trace.candidates,
                "selected_index": snapshot.trace.selected_index,
                "prediction_path_capacity_multiplier": np.asarray([s.path_capacity_multiplier for s in snapshot.trace.scenarios]),
                "prediction_exit_capacity_multiplier": np.asarray([s.exit_capacity_multiplier for s in snapshot.trace.scenarios]),
                "prediction_compliance": np.asarray([s.compliance for s in snapshot.trace.scenarios]),
            })
            np.savez_compressed(snapshot_dir / f"{seed}_{kind}_{snapshot.snapshot_seconds:g}.npz", **arrays)
    return eligibility, decisions, candidates, branches


def run(seed_count: int = 30, workers: int = 1, *, seed_start: int = 110000,
        output: str | Path | None = None) -> None:
    if seed_count <= 0 or workers <= 0:
        raise ValueError("seeds and workers must be positive")
    output = Path(output) if output is not None else ROOT / "outputs" / "pic2026_revision"
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(seed, str(output)) for seed in range(seed_start, seed_start + seed_count)]
    accumulated: list[list[dict]] = [[], [], [], []]
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = pool.map(_run_seed, tasks) if pool is not None else map(_run_seed, tasks)
        for index, result in enumerate(results, 1):
            for target, rows in zip(accumulated, result):
                target.extend(rows)
            print(f"diagnostic seed {index}/{seed_count} complete", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    frames = [pd.DataFrame(rows) for rows in accumulated]
    names = ("snapshot_eligibility", "decision_diagnostics", "candidate_scores", "counterfactual_branches")
    for name, frame in zip(names, frames):
        frame.to_csv(output / f"{name}.csv", index=False)
    per_seed, summary = summarize_decisions(frames[1])
    per_seed.to_csv(output / "decision_diagnostics_per_seed.csv", index=False)
    summary.to_csv(output / "decision_diagnostics_summary.csv", index=False)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unavailable"
    metadata = {
        "seed_start": seed_start, "seed_end": seed_start + seed_count - 1,
        "policy_seed_offset": 1000003, "policy_kinds": POLICY_KINDS,
        "snapshot_seconds": SNAPSHOT_SECONDS, "bootstrap_samples": 5000,
        "bootstrap_seed": 20261108, "workers": workers,
        "source_head": sha, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "eligible_snapshots": int(frames[0].eligible.sum()),
        "ineligible_snapshots": int((~frames[0].eligible).sum()),
        "incomplete_branches": int((~frames[3].completed).sum()) if len(frames[3]) else 0,
        "interpretation": "same-state action diagnostics; not full-policy comparison",
    }
    (output / "decision_diagnostics_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=110000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.seeds, args.workers, seed_start=args.seed_start, output=args.output)
