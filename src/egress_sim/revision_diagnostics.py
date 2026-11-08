"""Paired same-state diagnostics for the PIC2026 manuscript revision.

These are action diagnostics, not a second estimate of complete-policy value.
The collector uses the same decisions as the simulator and traces their actual
candidate/scenario evaluation without consuming extra random numbers.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EgressSpec
from .model import EgressState, Scenario, advance_state, initial_state
from .policies import (
    DecisionTrace, DeterministicLoadBalancePolicy, RobustRollingHorizonPolicy,
    build_policy,
)
from .uncertainty import sample_scenario


POLICY_KINDS = ("unconditioned_mean_only", "conditioned_mean_equal_spread")
SNAPSHOT_SECONDS = (120.0, 360.0, 600.0)
EQUIVALENCE_ATOL = 1e-12


@dataclass(frozen=True)
class DecisionSnapshot:
    seed: int
    policy_kind: str
    snapshot_seconds: float
    eligible: bool
    skip_reason: str = ""
    state: EgressState | None = None
    scenario: Scenario | None = None
    policy: RobustRollingHorizonPolicy | None = None
    rng_state: dict | None = None
    selected_recommendation: np.ndarray | None = None
    trace: DecisionTrace | None = None


@dataclass(frozen=True)
class BranchOutcome:
    completed: bool
    residual_clearance_seconds: float
    elapsed_observed_seconds: float
    remaining_population: float
    reactive_update_seconds: tuple[float, ...]


def equivalent_recommendations(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, rtol=0.0, atol=EQUIVALENCE_ATOL))


def collect_snapshots(
    spec: EgressSpec, seed: int, policy_kind: str,
    times: tuple[float, ...] = SNAPSHOT_SECONDS,
) -> list[DecisionSnapshot]:
    """Collect full pre-decision snapshots on one actual U/C trajectory.

    ``policy`` and ``rng_state`` refer to immediately before the captured
    decision. The saved policy has tracing disabled, ready for runtime replay.
    Cleared or duration-limited trajectories yield explicit ineligible rows.
    """
    if policy_kind not in POLICY_KINDS:
        raise ValueError(f"diagnostic policy must be one of {POLICY_KINDS}")
    interval = spec.time_step_seconds * spec.control_interval_steps
    targets = sorted(set(float(t) for t in times))
    if not targets:
        return []
    if any(t < 0 or not np.isclose(t / interval, round(t / interval)) for t in targets):
        raise ValueError("snapshot times must be nonnegative control-update times")
    scenario = sample_scenario(spec, np.random.default_rng(seed))
    policy = build_policy(policy_kind)
    rng = np.random.default_rng(seed + 1_000_003)
    state = initial_state(spec)
    recommendation = state.last_recommendation.copy()
    found: dict[float, DecisionSnapshot] = {}
    step = 0
    while (state.active_population > 1e-8
           and state.elapsed_seconds < spec.maximum_duration_seconds - 1e-9
           and state.elapsed_seconds <= targets[-1] + 1e-9):
        if step % spec.control_interval_steps == 0:
            target = next((t for t in targets
                           if np.isclose(t, state.elapsed_seconds, atol=1e-9)), None)
            if target is not None:
                saved_state = state.clone()
                saved_policy = copy.deepcopy(policy)
                saved_policy.capture_trace = False
                saved_policy.last_trace = None
                saved_policy._trace_components = None
                saved_rng = copy.deepcopy(rng.bit_generator.state)
                policy.capture_trace = True
            else:
                policy.capture_trace = False
            recommendation = policy.decide(state, spec, rng)
            if target is not None:
                found[target] = DecisionSnapshot(
                    seed, policy_kind, target, True,
                    state=saved_state,
                    scenario=Scenario(scenario.path_capacity_multiplier.copy(),
                                      scenario.exit_capacity_multiplier.copy(),
                                      scenario.compliance.copy()),
                    policy=saved_policy, rng_state=saved_rng,
                    selected_recommendation=recommendation.copy(),
                    trace=copy.deepcopy(policy.last_trace),
                )
            if state.elapsed_seconds >= targets[-1] - 1e-9:
                break
        advance_state(state, spec, recommendation, scenario)
        step += 1
    reason = ("cleared_before_snapshot" if state.active_population <= 1e-8
              else "maximum_duration_before_snapshot")
    return [found.get(t, DecisionSnapshot(seed, policy_kind, t, False, reason))
            for t in targets]


def continue_branch(
    state: EgressState, spec: EgressSpec, scenario: Scenario,
    recommendation: np.ndarray, mode: str,
) -> BranchOutcome:
    """Evaluate one cloned action branch under the realized environment.

    The absolute simulation ceiling is retained. Censored elapsed time is
    separate from clearance time, which is NaN when the branch is incomplete.
    """
    if mode not in ("fixed", "reactive_continuation"):
        raise ValueError("unknown continuation mode")
    rollout = state.clone()
    start = rollout.elapsed_seconds
    interval = spec.control_interval_steps * spec.time_step_seconds
    next_update = start + interval
    action = recommendation.copy()
    reactive = DeterministicLoadBalancePolicy()
    rng = np.random.default_rng(0)  # reactive does not consume it
    updates: list[float] = []
    while (rollout.active_population > 1e-8
           and rollout.elapsed_seconds + spec.time_step_seconds
           <= spec.maximum_duration_seconds + 1e-9):
        if mode == "reactive_continuation" and rollout.elapsed_seconds >= next_update - 1e-9:
            action = reactive.decide(rollout, spec, rng)
            updates.append(float(rollout.elapsed_seconds))
            next_update += interval
        advance_state(rollout, spec, action, scenario)
    completed = rollout.active_population <= 1e-8
    elapsed = float(rollout.elapsed_seconds - start)
    return BranchOutcome(
        completed, elapsed if completed else float("nan"), elapsed,
        rollout.active_population, tuple(updates),
    )


def evaluate_snapshot(
    snapshot: DecisionSnapshot, spec: EgressSpec,
) -> tuple[dict, list[dict], list[dict]]:
    """Return one decision row plus candidate and counterfactual rows."""
    if not snapshot.eligible:
        raise ValueError("cannot evaluate ineligible snapshot")
    assert snapshot.state is not None and snapshot.scenario is not None
    assert snapshot.trace is not None and snapshot.selected_recommendation is not None
    trace = snapshot.trace
    selected = trace.selected_index
    reactive = 1  # construction rule: habitual, reactive, then other candidates
    base = {"seed": snapshot.seed, "policy": snapshot.policy_kind,
            "snapshot_seconds": snapshot.snapshot_seconds}
    is_reactive = equivalent_recommendations(trace.candidates[selected], trace.candidates[reactive])
    full_gap = float(trace.objective[selected] - trace.objective[reactive])
    clearance = trace.clearance_minutes.mean(axis=1) * 60.0
    clearance_gap = float(clearance[selected] - clearance[reactive])
    candidates = []
    for index in range(len(trace.candidates)):
        candidates.append({
            **base, "candidate_index": index, "selected": index == selected,
            "equivalent_to_reactive": equivalent_recommendations(trace.candidates[index], trace.candidates[reactive]),
            "objective_minutes": float(trace.objective[index]),
            "mean_scenario_cost_minutes": float(trace.mean_cost[index]),
            "cvar_cost_minutes": float(trace.cvar_cost[index]),
            "switch_cost_minutes": float(trace.switch_cost[index]),
            "clearance_component_seconds": float(clearance[index]),
            "mean_delay_minutes": float(trace.mean_delay_minutes[index].mean()),
            "density_penalty_minutes": float(trace.density_penalty[index].mean()),
            "recommendation_json": json.dumps(trace.candidates[index].tolist(), separators=(",", ":")),
        })
    outcomes: dict[tuple[str, str], BranchOutcome] = {}
    branches = []
    for mode in ("fixed", "reactive_continuation"):
        for name, action in (("selected", trace.candidates[selected]),
                             ("reactive", trace.candidates[reactive])):
            outcome = continue_branch(snapshot.state, spec, snapshot.scenario, action, mode)
            outcomes[mode, name] = outcome
            branches.append({
                **base, "mode": mode, "action": name,
                "completed": outcome.completed,
                "residual_clearance_seconds": outcome.residual_clearance_seconds,
                "elapsed_observed_seconds": outcome.elapsed_observed_seconds,
                "remaining_population": outcome.remaining_population,
                "reactive_update_seconds": ";".join(f"{t:g}" for t in outcome.reactive_update_seconds),
            })
    row = {**base, "selected_index": selected, "reactive_index": reactive,
           "reactive_selected": is_reactive,
           "objective_gap_minutes": full_gap,
           "clearance_component_gap_seconds": clearance_gap}
    for mode in ("fixed", "reactive_continuation"):
        left, right = outcomes[mode, "selected"], outcomes[mode, "reactive"]
        complete = left.completed and right.completed
        difference = left.residual_clearance_seconds - right.residual_clearance_seconds
        row[f"{mode}_pair_completed"] = complete
        row[f"{mode}_difference_seconds"] = difference
        row[f"{mode}_selected_worse"] = float(difference > 1e-9) if complete else float("nan")
    fixed_difference = row["fixed_difference_seconds"]
    predicted_sign = 0 if abs(clearance_gap) <= 1e-9 else int(np.sign(clearance_gap))
    actual_sign = (0 if abs(fixed_difference) <= 1e-9 else int(np.sign(fixed_difference))) if np.isfinite(fixed_difference) else None
    # Protocol: strict opposite signs are reversals; ties are separate.
    row["clearance_rank_disagreement"] = (
        float(predicted_sign * actual_sign == -1)
        if row["fixed_pair_completed"] else float("nan")
    )
    row["predicted_clearance_tie"] = float(predicted_sign == 0)
    row["actual_fixed_clearance_tie"] = float(actual_sign == 0) if actual_sign is not None else float("nan")
    row["predicted_clearance_better_fixed_worse"] = (
        float(clearance_gap < -1e-9 and fixed_difference > 1e-9)
        if row["fixed_pair_completed"] else float("nan")
    )
    for mode in ("fixed", "reactive_continuation"):
        row[f"objective_favored_{mode}_worse"] = (
            float(full_gap < -1e-9 and row[f"{mode}_difference_seconds"] > 1e-9)
            if row[f"{mode}_pair_completed"] else float("nan")
        )
    return row, candidates, branches


SUMMARY_METRICS = (
    "reactive_selected", "objective_gap_minutes", "clearance_component_gap_seconds",
    "fixed_difference_seconds", "reactive_continuation_difference_seconds",
    "fixed_selected_worse", "reactive_continuation_selected_worse",
    "clearance_rank_disagreement", "predicted_clearance_better_fixed_worse",
    "predicted_clearance_tie", "actual_fixed_clearance_tie",
    "objective_favored_fixed_worse", "objective_favored_reactive_continuation_worse",
)


def summarize_decisions(
    decisions: pd.DataFrame, *, bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20261108,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seed-weighted estimates and seed-cluster percentile intervals.

    Identical bootstrap seed draws are used for all groups and all metrics.
    Decisions within a seed are averaged first, including across U/C origins
    for the combined group. Missing/censored metric values are never imputed.
    """
    if decisions.empty:
        return pd.DataFrame(), pd.DataFrame()
    seeds = np.sort(decisions.seed.unique())
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, len(seeds), (bootstrap_samples, len(seeds)))
    groups: list[tuple[str, float, pd.DataFrame]] = [("all", -1.0, decisions)]
    for policy, frame in decisions.groupby("policy", sort=True):
        groups.append((str(policy), -1.0, frame))
        for phase, subset in frame.groupby("snapshot_seconds", sort=True):
            groups.append((str(policy), float(phase), subset))
    per_seed_rows, summary_rows = [], []
    for policy, phase, frame in groups:
        for metric in SUMMARY_METRICS:
            numeric = frame[["seed", metric]].copy()
            numeric[metric] = pd.to_numeric(numeric[metric], errors="raise")
            values = numeric.groupby("seed")[metric].mean().reindex(seeds).to_numpy(dtype=float)
            for seed, value in zip(seeds, values):
                per_seed_rows.append({"policy": policy, "snapshot_seconds": phase,
                                      "seed": int(seed), "metric": metric, "value": value})
            samples = values[draws]
            counts = np.isfinite(samples).sum(axis=1)
            boot = np.divide(np.nansum(samples, axis=1), counts,
                             out=np.full(bootstrap_samples, np.nan), where=counts > 0)
            finite = values[np.isfinite(values)]
            valid_boot = boot[np.isfinite(boot)]
            summary_rows.append({
                "policy": policy, "snapshot_seconds": phase, "metric": metric,
                "eligible_decisions": len(frame), "valid_decisions": int(numeric[metric].notna().sum()),
                "valid_seeds": len(finite), "estimate": float(finite.mean()) if len(finite) else np.nan,
                "ci95_low": float(np.quantile(valid_boot, .025)) if len(valid_boot) else np.nan,
                "ci95_high": float(np.quantile(valid_boot, .975)) if len(valid_boot) else np.nan,
                "bootstrap_samples": bootstrap_samples,
            })
    return pd.DataFrame(per_seed_rows), pd.DataFrame(summary_rows)
