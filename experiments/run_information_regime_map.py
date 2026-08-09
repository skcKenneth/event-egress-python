from __future__ import annotations

import argparse
from collections.abc import Iterable

import pandas as pd

from _bootstrap import DEFAULT_WORKERS, ROOT
from _stats import paired_effects

from egress_sim.analysis import (
    PolicyVariant,
    ensure_output_directories,
    execute_policy_variant_tasks,
)
from egress_sim.config import load_spec


SEED_START = 96_000
CAPACITY_LEVELS = (0.0, 0.15, 0.30, 0.45)
COMPLIANCE_LEVELS = (0.35, 0.50, 0.68, 0.85)
POLICY_SEED_OFFSET = 1_000_003


def _sigma_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _paths(tables, suffix: str) -> tuple:
    return (
        tables / f"information_regime_map_runs{suffix}.csv",
        tables / f"information_regime_map_summary{suffix}.csv",
    )


def _summarize(raw: pd.DataFrame, seed_count: int) -> pd.DataFrame:
    rows: list[dict] = []
    for (sigma, compliance), subset in raw.groupby(
        ["capacity_log_sigma", "compliance_mean"]
    ):
        rows.append(
            {
                "capacity_log_sigma": sigma,
                "compliance_mean": compliance,
                "runs": seed_count,
                **paired_effects(
                    subset,
                    "conditioned_mean_equal_spread",
                    "unconditioned_mean_only",
                    label=f"info-regime-{sigma}-{compliance}",
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["capacity_log_sigma", "compliance_mean"],
        ignore_index=True,
    )


def run(
    seed_count: int = 50,
    workers: int = DEFAULT_WORKERS,
    *,
    capacity_levels: Iterable[float] = CAPACITY_LEVELS,
    suffix: str = "",
) -> None:
    """Run selected cells of the frozen uncertainty-compliance map.

    ``suffix`` supports checkpointed builds on constrained machines.  The
    default call still runs all 16 cells and writes the canonical paper tables.
    """

    selected = tuple(float(value) for value in capacity_levels)
    invalid = set(selected) - set(CAPACITY_LEVELS)
    if invalid:
        raise ValueError(f"unsupported capacity sigma values: {sorted(invalid)}")

    spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    seeds = range(SEED_START, SEED_START + seed_count)
    variants = (
        PolicyVariant("unconditioned_mean_only", "unconditioned_mean_only", {}),
        PolicyVariant(
            "conditioned_mean_equal_spread",
            "conditioned_mean_equal_spread",
            {},
        ),
    )
    tasks = [
        (
            spec,
            int(seed),
            int(seed) + POLICY_SEED_OFFSET,
            variant,
            sigma,
            compliance,
            {},
            {},
        )
        for sigma in selected
        for compliance in COMPLIANCE_LEVELS
        for seed in seeds
        for variant in variants
    ]
    raw = execute_policy_variant_tasks(tasks, workers=workers)
    runs_path, summary_path = _paths(tables, suffix)
    raw.to_csv(runs_path, index=False)
    _summarize(raw, seed_count).to_csv(summary_path, index=False)
    print(f"Wrote {runs_path.name} and {summary_path.name}")


def merge_checkpoints(seed_count: int = 50) -> None:
    """Merge the four canonical sigma checkpoints into the frozen map tables."""

    tables, _ = ensure_output_directories(ROOT)
    frames: list[pd.DataFrame] = []
    for sigma in CAPACITY_LEVELS:
        suffix = f"_sigma_{_sigma_tag(sigma)}"
        runs_path, _ = _paths(tables, suffix)
        if not runs_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {runs_path}")
        frame = pd.read_csv(runs_path)
        observed = set(frame["capacity_log_sigma"].unique())
        if observed != {sigma}:
            raise ValueError(
                f"checkpoint {runs_path.name} contains sigma values {observed}"
            )
        counts = frame.groupby(["compliance_mean", "policy"])[
            "scenario_seed"
        ].nunique()
        if not bool((counts == seed_count).all()) or len(counts) != 8:
            raise ValueError(
                f"checkpoint {runs_path.name} does not contain {seed_count} "
                "seeds for four paired compliance cells"
            )
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(
        ["capacity_log_sigma", "compliance_mean", "scenario_seed", "policy"],
        ignore_index=True,
    )
    runs_path, summary_path = _paths(tables, "")
    raw.to_csv(runs_path, index=False)
    _summarize(raw, seed_count).to_csv(summary_path, index=False)
    print(f"Merged checkpoints into {runs_path.name} and {summary_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the uncertainty-compliance map. Use --sigma once per "
            "checkpoint and --merge-checkpoints on machines with short job limits."
        )
    )
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--sigma",
        type=float,
        action="append",
        help="Run only this capacity log-sigma; may be repeated.",
    )
    parser.add_argument(
        "--merge-checkpoints",
        action="store_true",
        help="Merge the four canonical sigma checkpoint files.",
    )
    args = parser.parse_args()
    if args.merge_checkpoints:
        merge_checkpoints(args.seeds)
    else:
        levels = CAPACITY_LEVELS if args.sigma is None else tuple(args.sigma)
        suffix = ""
        if args.sigma is not None:
            if len(levels) != 1:
                raise SystemExit(
                    "Checkpoint mode accepts exactly one --sigma per invocation."
                )
            suffix = f"_sigma_{_sigma_tag(levels[0])}"
        run(
            args.seeds,
            args.workers,
            capacity_levels=levels,
            suffix=suffix,
        )
