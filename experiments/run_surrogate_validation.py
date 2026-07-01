from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT

from egress_sim.config import load_spec
from egress_sim.model import EgressState, Scenario, advance_state, current_peak_density, initial_state
from egress_sim.policies import RobustRollingHorizonPolicy
from egress_sim.uncertainty import sample_scenario


STATE_SECONDS = (240.0, 480.0)


def _surrogate_clearance_minutes(
    state: EgressState,
    spec,
    policy: RobustRollingHorizonPolicy,
    recommendation: np.ndarray,
    scenario: Scenario,
) -> float:
    rollout = state.clone()
    rollout.peak_density = current_peak_density(state, spec)
    controller_spec = replace(spec, free_flow_time=policy._controller_travel_time(spec))
    for _ in range(spec.robust_control.rollout_steps):
        if rollout.active_population <= 1e-8:
            break
        advance_state(rollout, controller_spec, recommendation, scenario)
    simulated_seconds = rollout.elapsed_seconds - state.elapsed_seconds
    actual_choice = (
        scenario.compliance[:, None] * recommendation
        + (1.0 - scenario.compliance[:, None]) * spec.habitual_matrix
    )
    future_path_work = rollout.path_queue + rollout.remaining[:, None] * actual_choice
    path_capacity = controller_spec.path_capacity * scenario.path_capacity_multiplier
    path_stage_seconds = float(np.max(future_path_work / np.maximum(path_capacity, 1e-9)))
    exit_work = (
        rollout.exit_queue
        + np.sum(rollout.transit, axis=(0, 2))
        + np.sum(future_path_work, axis=0)
    )
    exit_capacity = controller_spec.exit_capacity * scenario.exit_capacity_multiplier
    exit_stage_seconds = float(np.max(exit_work / np.maximum(exit_capacity, 1e-9)))
    occupied_paths = future_path_work > 1e-8
    free_flow_seconds = float(
        np.max(controller_spec.free_flow_time[occupied_paths])
        if np.any(occupied_paths)
        else 0.0
    )
    return (simulated_seconds + path_stage_seconds + exit_stage_seconds + free_flow_seconds) / 60.0


def _direct_clearance_minutes(
    state: EgressState,
    spec,
    policy: RobustRollingHorizonPolicy,
    recommendation: np.ndarray,
    scenario: Scenario,
    max_seconds: float,
) -> float:
    rollout = state.clone()
    controller_spec = replace(spec, free_flow_time=policy._controller_travel_time(spec))
    max_steps = int(np.ceil(max_seconds / spec.time_step_seconds))
    for _ in range(max_steps):
        if rollout.active_population <= 1e-8:
            break
        advance_state(rollout, controller_spec, recommendation, scenario)
    return (rollout.elapsed_seconds - state.elapsed_seconds) / 60.0


def _collect_state(
    spec,
    *,
    seed: int,
    capacity_log_sigma: float,
    compliance_mean: float,
    state_time_seconds: float,
    nonstationary: bool,
) -> tuple[EgressState, Scenario]:
    scenario_rng = np.random.default_rng(seed)
    policy_rng = np.random.default_rng(seed + 1_000_003)
    scenario = sample_scenario(
        spec,
        scenario_rng,
        capacity_log_sigma=capacity_log_sigma,
        compliance_mean=compliance_mean,
    )
    policy = RobustRollingHorizonPolicy()
    state = initial_state(spec)
    recommendation = state.last_recommendation.copy()
    max_steps = int(np.ceil(state_time_seconds / spec.time_step_seconds))
    for step in range(max_steps):
        if step % spec.control_interval_steps == 0:
            recommendation = policy.decide(state, spec, policy_rng)
        step_scenario = scenario
        if nonstationary and 240.0 <= state.elapsed_seconds < 600.0:
            path = scenario.path_capacity_multiplier.copy()
            exit_values = scenario.exit_capacity_multiplier.copy()
            path[:, 0] *= 0.55
            exit_values[0] *= 0.55
            step_scenario = Scenario(path, exit_values, scenario.compliance)
        advance_state(state, spec, recommendation, step_scenario)
    return state, scenario


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy()


def _summarize(group: pd.DataFrame) -> dict[str, float]:
    surrogate = group["surrogate_minutes"].to_numpy()
    direct = group["direct_minutes"].to_numpy()
    over = surrogate - direct
    pearson = float(np.corrcoef(surrogate, direct)[0, 1]) if len(group) > 1 else np.nan
    spearman = (
        float(np.corrcoef(_rank(surrogate), _rank(direct))[0, 1])
        if len(group) > 1
        else np.nan
    )
    by_scenario = []
    for _, subset in group.groupby(["seed", "state_seconds", "scenario_index"]):
        surrogate_values = subset.sort_values("candidate_index")["surrogate_minutes"].to_numpy()
        direct_values = subset.sort_values("candidate_index")["direct_minutes"].to_numpy()
        s_order = np.argsort(surrogate_values)
        d_order = np.argsort(direct_values)
        by_scenario.append(
            {
                "top1": float(s_order[0] == d_order[0]),
                "top3_overlap": len(set(s_order[:3]) & set(d_order[:3])) / 3.0,
                "regret": float(direct_values[s_order[0]] - direct_values[d_order[0]]),
            }
        )
    scenario_frame = pd.DataFrame(by_scenario)
    return {
        "rows": float(len(group)),
        "pearson": pearson,
        "spearman": spearman,
        "mean_overestimation_minutes": float(np.mean(over)),
        "median_overestimation_minutes": float(np.median(over)),
        "p90_overestimation_minutes": float(np.quantile(over, 0.90)),
        "max_overestimation_minutes": float(np.max(over)),
        "top1_agreement_percent": float(100.0 * scenario_frame["top1"].mean()),
        "mean_top3_overlap": float(scenario_frame["top3_overlap"].mean()),
        "mean_regret_minutes": float(scenario_frame["regret"].mean()),
        "max_regret_minutes": float(scenario_frame["regret"].max()),
    }


def run(seed_count: int = 8, workers: int = 1) -> None:
    del workers
    spec = load_spec(ROOT / "config" / "default.json")
    tables = ROOT / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    conditions = (
        ("default", 0.22, 0.68, False),
        ("high_compliance_strong_uncertainty", 0.45, 0.85, False),
        ("low_compliance", 0.22, 0.35, False),
        ("temporary_primary_exit_capacity_55pct", 0.45, 0.68, True),
    )
    rows = []
    for condition, sigma, compliance, nonstationary in conditions:
        for seed in range(70_000, 70_000 + seed_count):
            for state_seconds in STATE_SECONDS:
                state, _ = _collect_state(
                    spec,
                    seed=seed,
                    capacity_log_sigma=sigma,
                    compliance_mean=compliance,
                    state_time_seconds=state_seconds,
                    nonstationary=nonstationary,
                )
                policy = RobustRollingHorizonPolicy()
                rng = np.random.default_rng(seed + 2_000_003 + int(state_seconds))
                candidates = policy._candidate_recommendations(state, spec, rng)
                scenarios = policy._prediction_scenarios(state, spec, rng)
                for scenario_index, scenario in enumerate(scenarios):
                    for candidate_index, candidate in enumerate(candidates):
                        rows.append(
                            {
                                "condition": condition,
                                "seed": seed,
                                "state_seconds": state_seconds,
                                "scenario_index": scenario_index,
                                "candidate_index": candidate_index,
                                "surrogate_minutes": _surrogate_clearance_minutes(
                                    state, spec, policy, candidate, scenario
                                ),
                                "direct_minutes": _direct_clearance_minutes(
                                    state, spec, policy, candidate, scenario, 3600.0
                                ),
                            }
                        )
    raw = pd.DataFrame(rows)
    raw["overestimation_minutes"] = raw["surrogate_minutes"] - raw["direct_minutes"]
    raw.to_csv(tables / "surrogate_validation_runs.csv", index=False)
    summary = []
    for condition, subset in raw.groupby("condition"):
        summary.append({"condition": condition, **_summarize(subset)})
    pd.DataFrame(summary).to_csv(tables / "surrogate_validation_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args.seeds, args.workers)
