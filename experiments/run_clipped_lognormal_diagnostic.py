from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _bootstrap import ROOT

from egress_sim.analysis import ensure_output_directories
from egress_sim.config import load_spec


SIGMAS = (0.0, 0.15, 0.22, 0.30, 0.45)


def run(samples: int = 1_000_000) -> None:
    spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    rng = np.random.default_rng(20260623)
    rows = []
    for sigma in SIGMAS:
        if sigma <= 0.0:
            values = np.ones(samples)
        else:
            raw = np.exp(rng.normal(-0.5 * sigma**2, sigma, samples))
            values = np.clip(
                raw,
                spec.uncertainty.capacity_multiplier_min,
                spec.uncertainty.capacity_multiplier_max,
            )
        rows.append(
            {
                "capacity_log_sigma": sigma,
                "samples": samples,
                "clip_min": spec.uncertainty.capacity_multiplier_min,
                "clip_max": spec.uncertainty.capacity_multiplier_max,
                "post_clip_mean": float(np.mean(values)),
                "post_clip_std": float(np.std(values, ddof=1)),
                "post_clip_q05": float(np.quantile(values, 0.05)),
                "post_clip_q50": float(np.quantile(values, 0.50)),
                "post_clip_q95": float(np.quantile(values, 0.95)),
                "mass_at_lower_clip_percent": float(
                    100.0 * np.mean(values <= spec.uncertainty.capacity_multiplier_min)
                ),
                "mass_at_upper_clip_percent": float(
                    100.0 * np.mean(values >= spec.uncertainty.capacity_multiplier_max)
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        tables / "clipped_lognormal_diagnostic.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_000)
    args = parser.parse_args()
    run(args.samples)
