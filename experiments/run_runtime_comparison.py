"""Compare matched reactive and mean-controller decision latency on shared states."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns
from typing import Any, Callable, Iterable

# This runner must configure numerical-library threads before importing NumPy.
# The main CLI launches experiment runners in a fresh interpreter.
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
NUMPY_PRELOADED = "numpy" in sys.modules
for _variable in THREAD_VARIABLES:
    os.environ[_variable] = "1"

from _bootstrap import ROOT

import numpy as np
import pandas as pd

from egress_sim.config import EgressSpec, load_spec
from egress_sim.policies import build_policy


POLICY_KINDS = (
    "deterministic_load_balance",
    "unconditioned_mean_only",
    "conditioned_mean_equal_spread",
)
COLLECTOR_KINDS = POLICY_KINDS[1:]
SNAPSHOT_TIMES = (120.0, 360.0, 600.0)
DEFAULT_OUTPUT = ROOT / "outputs" / "pic2026_revision"
RUN_COLUMNS = (
    "snapshot_id", "scenario_seed", "policy_seed", "origin_policy",
    "snapshot_seconds", "phase", "policy", "repeat", "order_position",
    "elapsed_ns", "elapsed_ms",
)


def _snapshot_id(snapshot: Any) -> str:
    return (
        f"{snapshot.seed}-{snapshot.policy_kind}-"
        f"{snapshot.snapshot_seconds:g}"
    )


def _phase(seconds: float) -> str:
    return {120.0: "early", 360.0: "middle", 600.0: "late"}.get(
        float(seconds), f"t{seconds:g}"
    )


def _prepare_policy(snapshot: Any, policy_kind: str) -> Any:
    """Restore pre-decision policy state without enabling diagnostic tracing.

    The three benchmark policies are clean mean/reactive policies. Estimator
    values and history belong to the shared EgressState. For the origin policy
    we restore the complete object; for the other mean policy we also preserve
    its compatible observation cache (unused in these clean configurations).
    """
    if policy_kind == snapshot.policy_kind:
        policy = deepcopy(snapshot.policy)
    else:
        policy = build_policy(policy_kind)
        for attribute in ("_last_path_observation", "_last_exit_observation"):
            if hasattr(policy, attribute) and hasattr(snapshot.policy, attribute):
                setattr(policy, attribute, deepcopy(getattr(snapshot.policy, attribute)))
    if hasattr(policy, "capture_trace"):
        policy.capture_trace = False
    return policy


def _prepare_decision(snapshot: Any, policy_kind: str) -> tuple[Any, Any, Any]:
    state = snapshot.state.clone()
    policy = _prepare_policy(snapshot, policy_kind)
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(snapshot.rng_state)
    return state, policy, rng


def benchmark_snapshots(
    spec: EgressSpec,
    snapshots: Iterable[Any],
    *,
    warmups: int = 5,
    repetitions: int = 30,
    timer_ns: Callable[[], int] = perf_counter_ns,
) -> pd.DataFrame:
    """Time only decide(), restoring identical inputs before each invocation."""
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions positive")
    rows: list[dict[str, object]] = []
    for snapshot_index, snapshot in enumerate(snapshots):
        if not snapshot.eligible:
            continue
        if snapshot.state is None or snapshot.policy is None or snapshot.rng_state is None:
            raise ValueError("eligible snapshot lacks pre-decision state/policy/RNG")
        for iteration in range(warmups + repetitions):
            # Every policy is invoked once per round. Rotate the first policy
            # across rounds and snapshots to reduce systematic order effects.
            rotation = (snapshot_index + iteration) % len(POLICY_KINDS)
            order = POLICY_KINDS[rotation:] + POLICY_KINDS[:rotation]
            for order_position, policy_kind in enumerate(order):
                state, policy, rng = _prepare_decision(snapshot, policy_kind)
                if iteration < warmups:
                    policy.decide(state, spec, rng)
                    continue
                start_ns = timer_ns()
                policy.decide(state, spec, rng)
                elapsed_ns = timer_ns() - start_ns
                if elapsed_ns < 0:
                    raise RuntimeError("monotonic timer returned a negative duration")
                rows.append(
                    {
                        "snapshot_id": _snapshot_id(snapshot),
                        "scenario_seed": snapshot.seed,
                        "policy_seed": snapshot.seed + 1_000_003,
                        "origin_policy": snapshot.policy_kind,
                        "snapshot_seconds": snapshot.snapshot_seconds,
                        "phase": _phase(snapshot.snapshot_seconds),
                        "policy": policy_kind,
                        "repeat": iteration - warmups,
                        "order_position": order_position,
                        "elapsed_ns": elapsed_ns,
                        "elapsed_ms": elapsed_ns / 1_000_000.0,
                    }
                )
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def summarize_timings(frame: pd.DataFrame, control_interval_seconds: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, policy_frame in frame.groupby("policy", sort=True):
        groups = [("overall", policy_frame)]
        groups.extend(policy_frame.groupby("phase", sort=False))
        for phase, subset in groups:
            values = subset["elapsed_ms"].to_numpy(dtype=float)
            p95 = float(np.quantile(values, 0.95))
            rows.append(
                {
                    "policy": policy,
                    "phase": phase,
                    "snapshot_count": int(subset["snapshot_id"].nunique()),
                    "timed_decisions": len(values),
                    "median_ms": float(np.median(values)),
                    "p95_ms": p95,
                    "min_ms": float(np.min(values)),
                    "max_ms": float(np.max(values)),
                    "p95_control_interval_fraction": p95 / (control_interval_seconds * 1000.0),
                }
            )
    return pd.DataFrame(rows, columns=(
        "policy", "phase", "snapshot_count", "timed_decisions", "median_ms",
        "p95_ms", "min_ms", "max_ms", "p95_control_interval_fraction",
    ))


def _git_value(arguments: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(
    spec: EgressSpec, *, seed_start: int, seeds: int, warmups: int,
    repetitions: int, eligible_count: int, skipped_count: int,
) -> dict[str, object]:
    versions = {}
    for package in ("numpy", "pandas", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "study": "PIC2026 matched three-policy decision runtime",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_value(["rev-parse", "HEAD"]),
        "working_tree_dirty": bool(_git_value(["status", "--porcelain"])),
        "script_sha256": _sha256(Path(__file__)),
        "config_sha256": _sha256(ROOT / "config" / "default.json"),
        "environment_seeds": list(range(seed_start, seed_start + seeds)),
        "policy_seeds": list(range(seed_start + 1_000_003, seed_start + seeds + 1_000_003)),
        "collector_policies": list(COLLECTOR_KINDS),
        "benchmark_policies": list(POLICY_KINDS),
        "snapshot_times_seconds": list(SNAPSHOT_TIMES),
        "eligible_snapshots": eligible_count,
        "skipped_snapshots": skipped_count,
        "warmups_per_snapshot_policy": warmups,
        "timed_repetitions_per_snapshot_policy": repetitions,
        "timer": "time.perf_counter_ns",
        "timed_region": "policy.decide only; excludes clones, construction and file I/O",
        "reset": "full physical/estimator state, policy object/cache and pre-decision RNG on every call",
        "execution": "serial single process, cyclic policy order",
        "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
        "numpy_preloaded_before_thread_configuration": NUMPY_PRELOADED,
        "python": sys.version,
        "packages": versions,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "control_interval_seconds": spec.control_interval_steps * spec.time_step_seconds,
        "interpretation": "sampled decision latency on this machine; not a real-time deployment guarantee",
    }


def run(
    *, seed_start: int = 110_000, seeds: int = 5, warmups: int = 5,
    repetitions: int = 30, output: str | Path = DEFAULT_OUTPUT,
) -> Path:
    if seed_start < 0 or seeds < 1:
        raise ValueError("seed_start must be non-negative and seeds positive")
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions positive")
    if NUMPY_PRELOADED:
        raise RuntimeError(
            "Launch this runner in a fresh interpreter so thread limits precede NumPy imports."
        )
    from egress_sim.revision_diagnostics import collect_snapshots

    spec = load_spec(ROOT / "config" / "default.json")
    snapshots = []
    for seed in range(seed_start, seed_start + seeds):
        for policy_kind in COLLECTOR_KINDS:
            snapshots.extend(collect_snapshots(spec, seed, policy_kind, SNAPSHOT_TIMES))
    eligibility = pd.DataFrame([
        {
            "snapshot_id": _snapshot_id(snapshot),
            "scenario_seed": snapshot.seed,
            "policy_seed": snapshot.seed + 1_000_003,
            "origin_policy": snapshot.policy_kind,
            "snapshot_seconds": snapshot.snapshot_seconds,
            "phase": _phase(snapshot.snapshot_seconds),
            "eligible": snapshot.eligible,
            "skip_reason": snapshot.skip_reason,
        }
        for snapshot in snapshots
    ])
    frame = benchmark_snapshots(spec, snapshots, warmups=warmups, repetitions=repetitions)
    summary = summarize_timings(frame, spec.control_interval_steps * spec.time_step_seconds)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "runtime_comparison_runs.csv", index=False)
    summary.to_csv(output / "runtime_comparison_summary.csv", index=False)
    eligibility.to_csv(output / "runtime_comparison_snapshots.csv", index=False)
    eligible_count = sum(bool(snapshot.eligible) for snapshot in snapshots)
    metadata = _metadata(
        spec, seed_start=seed_start, seeds=seeds, warmups=warmups,
        repetitions=repetitions, eligible_count=eligible_count,
        skipped_count=len(snapshots) - eligible_count,
    )
    (output / "runtime_comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(frame)} timed decisions from {eligible_count} shared snapshots to {output}")
    print(summary.loc[summary["phase"] == "overall"].to_string(index=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=110_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(**vars(parser.parse_args()))
