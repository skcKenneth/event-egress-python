from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from _bootstrap import ROOT

from egress_sim.config import load_spec
from egress_sim.model import initial_state
from egress_sim.policies import RobustRollingHorizonPolicy


def run(repeats: int = 25, seed: int = 20_260_622) -> Path:
    if repeats < 2:
        raise ValueError("repeats must be at least 2")

    spec = load_spec(ROOT / "config" / "default.json")
    output = ROOT / "outputs" / "tables" / "runtime_benchmark.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        state = initial_state(spec)
        policy = RobustRollingHorizonPolicy()
        rng = np.random.default_rng(seed + repeat)
        start = perf_counter()
        policy.decide(state, spec, rng)
        elapsed_seconds = perf_counter() - start
        rows.append(
            {
                "repeat": repeat,
                "seed": seed + repeat,
                "elapsed_seconds": elapsed_seconds,
                "candidate_count": spec.robust_control.candidate_count,
                "scenario_count": spec.robust_control.prediction_scenarios,
                "rollout_steps": spec.robust_control.rollout_steps,
                "scenario_rollouts": (
                    spec.robust_control.candidate_count
                    * spec.robust_control.prediction_scenarios
                ),
                "queue_advances": (
                    spec.robust_control.candidate_count
                    * spec.robust_control.prediction_scenarios
                    * spec.robust_control.rollout_steps
                ),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "platform": platform.platform(),
                "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    elapsed = frame["elapsed_seconds"]
    print(f"Wrote {output}")
    print(f"median_seconds={elapsed.median():.4f}")
    print(f"p90_seconds={elapsed.quantile(0.9):.4f}")
    print(f"max_seconds={elapsed.max():.4f}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20_260_622)
    args = parser.parse_args()
    run(args.repeats, args.seed)
