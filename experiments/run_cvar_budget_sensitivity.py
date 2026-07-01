from __future__ import annotations

import argparse
from dataclasses import replace
from time import perf_counter

import numpy as np
import pandas as pd

from _bootstrap import ROOT
from _stats import paired_effects

from egress_sim.analysis import PolicyVariant, ensure_output_directories, run_policy_variants
from egress_sim.config import load_spec
from egress_sim.model import initial_state
from egress_sim.policies import RobustRollingHorizonPolicy, ScenarioMeanPolicy


BUDGETS = (8, 16, 32, 64)
REGIMES = (
    ("default", 0.22, 0.68),
    ("high_compliance_strong_uncertainty", 0.45, 0.85),
)


def _runtime_rows(spec, budget: int, repeats: int = 8) -> list[dict[str, float | int | str]]:
    spec = replace(spec, robust_control=replace(spec.robust_control, prediction_scenarios=budget))
    rows = []
    for policy_class in (ScenarioMeanPolicy, RobustRollingHorizonPolicy):
        for repeat in range(repeats):
            policy = policy_class()
            state = initial_state(spec)
            rng = np.random.default_rng(80_000 + budget * 100 + repeat)
            start = perf_counter()
            recommendation = policy.decide(state, spec, rng)
            elapsed = perf_counter() - start
            rows.append(
                {
                    "prediction_scenarios": budget,
                    "policy": policy.name,
                    "repeat": repeat,
                    "elapsed_seconds": elapsed,
                    "candidate_count": spec.robust_control.candidate_count,
                    "rollout_steps": spec.robust_control.rollout_steps,
                    "recommendation_hash": float(np.sum(recommendation * np.arange(1, recommendation.size + 1).reshape(recommendation.shape))),
                }
            )
    return rows


def run(seed_count: int = 20, workers: int = 4) -> None:
    base_spec = load_spec(ROOT / "config" / "default.json")
    tables, _ = ensure_output_directories(ROOT)
    frames = []
    effects = []
    runtime_rows = []
    decision_rows = []
    for budget in BUDGETS:
        runtime_rows.extend(_runtime_rows(base_spec, budget))
        spec = replace(
            base_spec,
            robust_control=replace(base_spec.robust_control, prediction_scenarios=budget),
        )
        mean_name = f"conditioned_mean_S{budget}"
        cvar_name = f"conditioned_cvar_S{budget}"
        variants = (
            PolicyVariant("scenario_mean_only", mean_name, {}),
            PolicyVariant("robust_rolling_horizon", cvar_name, {}),
        )
        for regime, sigma, compliance in REGIMES:
            frame = run_policy_variants(
                spec,
                range(80_000, 80_000 + seed_count),
                variants,
                capacity_log_sigma=sigma,
                compliance_mean=compliance,
                workers=workers,
            )
            frame["regime"] = regime
            frame["prediction_scenarios"] = budget
            frames.append(frame)
            effects.append(
                {
                    "regime": regime,
                    "prediction_scenarios": budget,
                    "proposed_policy": cvar_name,
                    "baseline_policy": mean_name,
                    **paired_effects(
                        frame,
                        cvar_name,
                        mean_name,
                        label=f"cvar-budget-{regime}-S{budget}",
                    ),
                }
            )
        for seed in range(80_000, 80_000 + min(seed_count, 20)):
            state = initial_state(spec)
            rng_mean = np.random.default_rng(seed + 1_000_003)
            rng_cvar = np.random.default_rng(seed + 1_000_003)
            mean_rec = ScenarioMeanPolicy().decide(state, spec, rng_mean)
            cvar_rec = RobustRollingHorizonPolicy().decide(state, spec, rng_cvar)
            decision_rows.append(
                {
                    "seed": seed,
                    "prediction_scenarios": budget,
                    "same_initial_recommendation": bool(np.allclose(mean_rec, cvar_rec)),
                    "initial_recommendation_l1": float(np.sum(np.abs(mean_rec - cvar_rec))),
                }
            )
    pd.concat(frames, ignore_index=True).to_csv(
        tables / "cvar_budget_runs.csv", index=False
    )
    pd.DataFrame(effects).to_csv(tables / "cvar_budget_effects.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(tables / "cvar_budget_runtime.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(tables / "cvar_budget_decisions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.seeds, args.workers)
