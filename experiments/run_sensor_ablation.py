from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import DEFAULT_WORKERS, ROOT
from _stats import paired_effects

from egress_sim.analysis import (
    PolicyVariant,
    aggregate_results,
    ensure_output_directories,
    run_policy_variants,
)
from egress_sim.config import load_spec


SEED_START = 99_000
REGIMES = (
    ("default", 0.22, 0.68),
    ("high_information", 0.45, 0.85),
)
FAULTS = (
    ("clean", {}),
    ("noise_sigma_015", {"sensor_noise_sigma": 0.15}),
    ("delay_60s", {"sensor_delay_steps": 6}),
    ("dropout_20pct", {"sensor_dropout": 0.20}),
    ("bias_plus_10pct", {"sensor_bias": 1.10}),
)


def run(seed_count: int = 30, workers: int = DEFAULT_WORKERS) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    frames: list[pd.DataFrame] = []
    effect_rows: list[dict[str, float | str]] = []

    for regime, sigma, compliance in REGIMES:
        variants = [
            PolicyVariant(
                "unconditioned_mean_only",
                "unconditioned_mean_only",
                {},
            )
        ]
        variants.extend(
            PolicyVariant(
                "conditioned_mean_equal_spread",
                f"conditioned_{fault}",
                kwargs,
            )
            for fault, kwargs in FAULTS
        )
        frame = run_policy_variants(
            spec,
            range(SEED_START, SEED_START + seed_count),
            tuple(variants),
            capacity_log_sigma=sigma,
            compliance_mean=compliance,
            workers=workers,
        )
        frame["regime"] = regime
        policy_to_fault = {
            "unconditioned_mean_only": "baseline",
            **{f"conditioned_{fault}": fault for fault, _ in FAULTS},
        }
        frame["fault"] = frame["policy"].map(policy_to_fault)
        frames.append(frame)

        for fault, _ in FAULTS:
            effect_rows.append(
                {
                    "regime": regime,
                    "fault": fault,
                    "capacity_log_sigma": sigma,
                    "compliance_mean": compliance,
                    **paired_effects(
                        frame,
                        f"conditioned_{fault}",
                        "unconditioned_mean_only",
                        label=f"sensor-{regime}-{fault}",
                    ),
                }
            )

    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(tables / "sensor_ablation_runs.csv", index=False)
    aggregate_results(raw, ["regime", "fault", "policy"]).to_csv(
        tables / "sensor_ablation_summary.csv", index=False
    )
    pd.DataFrame(effect_rows).to_csv(
        tables / "sensor_condition_effects.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    run(args.seeds, args.workers)
