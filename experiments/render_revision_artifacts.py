"""Generate revised paper numbers/figures without modifying the frozen bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from _bootstrap import ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from render_ieee_figures import figure1, figure2, figure3
from egress_sim.figure_protocol import validate_frozen_artifact_bundle
from egress_sim.ieee_style import BLUE, ORANGE, INK, GRID, use_ieee_style, save_ieee

TABLES = ROOT / "paper_artifacts/tables"
REVISION = ROOT / "revision_artifacts/2026-11"
PAPER = ROOT / "paper"
U = "unconditioned_mean_only"
C = "conditioned_mean_equal_spread"
R = "deterministic_load_balance"


def _write_tex(path: Path, values: dict[str, str]) -> None:
    text = "% Generated from authoritative CSVs by render_revision_artifacts.py.\n"
    text += "\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in values.items())
    path.write_text(text + "\n", encoding="utf-8")


def original_macros() -> None:
    summary = pd.read_csv(TABLES / "information_value_validation_summary.csv").set_index("policy")
    effects = pd.read_csv(TABLES / "information_value_validation_effects.csv").set_index("comparison")
    values: dict[str, str] = {}
    for prefix, policy in (("Static", "static_nearest"), ("Reactive", R),
                           ("Uncond", U), ("Cond", C), ("Oracle", "oracle_capacity_mean_only")):
        row = summary.loc[policy]
        for suffix, col, precision in (("Mean", "clearance_mean_s", 2),
                                       ("Pninety", "clearance_p90_s", 1),
                                       ("Density", "peak_density_mean", 3)):
            values[prefix + suffix] = f"{row[col]:.{precision}f}"
    for prefix, key in (("Dynamic", "dynamic-vs-static"), ("Planning", "planning-vs-reactive"),
                        ("Primary", "primary-sensing-value"),
                        ("Spread", "spread-contraction-diagnostic"),
                        ("OracleGap", "oracle-residual-information")):
        row = effects.loc[key]
        for suffix, col in (("Diff", "mean_difference"), ("CILow", "mean_ci95_low"),
                            ("CIHigh", "mean_ci95_high")):
            values[prefix + suffix] = f"{row[col]:+.2f}"
    primary = effects.loc["primary-sensing-value"]
    values.update({
        "PrimaryPct": f"{primary.mean_difference_percent:.2f}",
        "PrimaryReduction": f"{-primary.mean_difference:.2f}",
        "PrimaryReductionPct": f"{-primary.mean_difference_percent:.2f}",
        "PrimaryPninetyDiff": f"{primary.p90_difference:+.2f}",
        "PrimaryPninetyCILow": f"{primary.p90_ci95_low:+.2f}",
        "PrimaryPninetyCIHigh": f"{primary.p90_ci95_high:+.2f}",
        "PrimaryWin": f"{primary.win_rate_percent:.1f}",
        "PrimaryTie": f"{primary.tie_rate_percent:.1f}",
        "PrimaryLoss": f"{primary.loss_rate_percent:.1f}",
        "WorstHarm": f"{primary.worst_observed_difference:.1f}",
        "OracleGapReduction": f"{-effects.loc['oracle-residual-information'].mean_difference:.2f}",
    })
    est = pd.read_csv(TABLES / "estimator_diagnostics_summary.csv")
    est = est[est.regime == "default"]
    early = est.sort_values("time_seconds").iloc[0]
    t240 = est[np.isclose(est.time_seconds, 240)].iloc[0]
    final = pd.read_csv(TABLES / "estimator_diagnostics_final.csv")
    final = final[final.regime == "default"]
    values.update({"EstimatorInitialMAE": f"{early.capacity_mae_mean:.3f}",
                   "EstimatorMAETwoForty": f"{t240.capacity_mae_mean:.3f}",
                   "EstimatorObservedTwoForty": f"{100*t240.capacity_observed_fraction_mean:.1f}",
                   "EstimatorFinalMAE": f"{final.capacity_mae.mean():.3f}",
                   "EstimatorFinalObserved": f"{100*final.capacity_observed_fraction.mean():.1f}"})
    grid = pd.read_csv(TABLES / "information_regime_map_summary.csv")
    for suffix, rho in (("ThreeFive", .35), ("FiveZero", .5), ("SixEight", .68), ("EightFive", .85)):
        row = grid[np.isclose(grid.capacity_log_sigma, .45) & np.isclose(grid.compliance_mean, rho)].iloc[0]
        values["MapHighC" + suffix] = f"{row.mean_difference:.1f}"
    dt = pd.read_csv(TABLES / "time_step_sensitivity_effects.csv")
    for suffix, step in (("Ten", 10), ("Five", 5), ("TwoFive", 2.5)):
        row = dt[(dt.comparison == "sensing-vs-unconditioned") & np.isclose(dt.time_step_seconds, step)].iloc[0]
        values["Time" + suffix] = f"{row.mean_difference:.1f}"
    topology = pd.read_csv(TABLES / "topology_generalization_effects.csv")
    values.update({"TopologyCount": str(len(topology)),
                   "TopologyResolvedCount": str(int((topology.mean_ci95_high < 0).sum())),
                   "TopologyBest": f"{topology.mean_difference.min():.1f}",
                   "TopologyWorst": f"{topology.mean_difference.max():.1f}"})
    faults = pd.read_csv(TABLES / "sensor_condition_effects.csv")
    for prefix, regime in (("Default", "default"), ("HighInfo", "high_information")):
        subset = faults[faults.regime == regime]
        values[prefix + "CleanSensor"] = f"{subset[subset.fault == 'clean'].iloc[0].mean_difference:.1f}"
        faults_only = subset[subset.fault != "clean"]
        values[prefix + "FaultLow"] = f"{faults_only.mean_difference.min():.1f}"
        values[prefix + "FaultHigh"] = f"{faults_only.mean_difference.max():.1f}"
    _write_tex(PAPER / "results.tex", values)


def diagnostic_row(summary: pd.DataFrame, policy: str, metric: str) -> pd.Series:
    return summary[(summary.policy == policy) & (summary.snapshot_seconds == -1)
                   & (summary.metric == metric)].iloc[0]


def revision_macros(summary: pd.DataFrame, timing: pd.DataFrame) -> None:
    effect = pd.read_csv(TABLES / "information_value_validation_effects.csv").set_index("comparison").loc["secondary-system-comparison"]
    final = pd.read_csv(TABLES / "estimator_diagnostics_final.csv")
    final = final[final.regime == "default"]
    values = {"SystemDiff": f"{effect.mean_difference:+.2f}",
              "SystemCILow": f"{effect.mean_ci95_low:+.2f}",
              "SystemCIHigh": f"{effect.mean_ci95_high:+.2f}",
              "PathFinalMAE": f"{final.path_capacity_mae.mean():.4f}",
              "ExitFinalMAE": f"{final.exit_capacity_mae.mean():.4f}",
              "PathObservedPct": f"{100*final.path_observed_fraction.mean():.2f}",
              "ExitObservedPct": f"{100*final.exit_observed_fraction.mean():.2f}"}
    eligible = pd.read_csv(REVISION / "snapshot_eligibility.csv")
    branches = pd.read_csv(REVISION / "counterfactual_branches.csv")
    completed_count = int(branches.completed.sum())
    text = (f"Of {len(eligible)} planned snapshots, {int(eligible.eligible.sum())} were eligible; "
            f"{completed_count} of {len(branches)} counterfactual branches completed. ")
    continuation_intervals = []
    for policy, label in ((U, "Unconditioned"), (C, "Conditioned")):
        selected = diagnostic_row(summary, policy, "reactive_selected")
        reversal = diagnostic_row(summary, policy, "clearance_rank_disagreement")
        continuation = diagnostic_row(summary, policy, "reactive_continuation_difference_seconds")
        continuation_intervals.append((continuation.ci95_low, continuation.ci95_high))
        text += (f"{label} control selected a recommendation equivalent to reactive in "
                 f"{100*selected.estimate:.1f}\\% of snapshots. Its fixed-action clearance "
                 f"ranking reversed in {100*reversal.estimate:.1f}\\% "
                 f"(95\\% seed-bootstrap interval [{100*reversal.ci95_low:.1f}, "
                 f"{100*reversal.ci95_high:.1f}]\\%). With the common reactive continuation, "
                 f"the selected-minus-reactive action effect was {continuation.estimate:+.2f}~s "
                 f"([{continuation.ci95_low:+.2f}, {continuation.ci95_high:+.2f}]~s). ")
    if all(low <= 0 <= high for low, high in continuation_intervals):
        text += "Both action-effect intervals include zero. "
    text += ("Strict reversals exclude predicted "
             "or actual ties, which remain separately tabulated. The diagnostics demonstrate "
             "that forecast ranking need not persist under realized capacity; they do not "
             "apportion the contributions of capacity mismatch, terminal approximation, "
             "finite scenarios and objective choice. The two reversal rates describe different "
             "on-policy state sets; their difference is not an isolated effect of conditioning.")
    # Reversal numerator excludes ties; its reported frequency denominator is
    # all completed decisions. Keep that distinction explicit in text/caption.
    text = text.replace("Strict reversals exclude predicted or actual ties, which remain separately tabulated.",
                        "Reversal frequencies use all completed decisions as denominator; predicted or actual ties are not reversals and are separately tabulated.")
    values["DiagnosticResultsText"] = text
    meta = json.loads((REVISION / "runtime_comparison_metadata.json").read_text())
    processor = str(meta["processor"]).replace("_", r"\_").replace("&", r"\&")
    p95max = timing[timing.phase == "overall"].p95_ms.max()
    runtime_states = int(meta["eligible_snapshots"])
    values["RuntimeResultsText"] = (
        f"On {processor}, Python {meta['python'].split()[0]} and NumPy {meta['packages']['numpy']}, "
        f"the benchmark recorded {int(timing[timing.phase == 'overall'].timed_decisions.sum())} "
        f"timed decisions over {runtime_states} shared snapshots. Both predictive controllers had similar "
        f"latency, with overall p95 at most {p95max:.2f}~ms; reactive control was substantially "
        "cheaper. These measurements exclude state copying and "
        "I/O and provide no hard real-time or deployment guarantee.")
    _write_tex(PAPER / "revision_results.tex", values)
    rows = [r"\begin{tabular}{lrr}", r"\toprule", r"Controller & Median (ms) & p95 (ms) \\", r"\midrule"]
    overall = timing[timing.phase == "overall"].set_index("policy")
    for policy, label in ((R, "Reactive"), (U, "Unconditioned mean"), (C, "Conditioned mean")):
        row = overall.loc[policy]
        rows.append(f"{label} & {row.median_ms:.3f} & {row.p95_ms:.3f} " + r"\\")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    (PAPER / "runtime_table.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def figure4(summary: pd.DataFrame, output: Path) -> None:
    use_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))
    colors = {U: ORANGE, C: BLUE}
    labels = {U: "Unconditioned", C: "Conditioned"}
    for index, policy in enumerate((U, C)):
        row = diagnostic_row(summary, policy, "clearance_rank_disagreement")
        axes[0].errorbar(index, row.estimate*100,
                        yerr=[[(row.estimate-row.ci95_low)*100], [(row.ci95_high-row.estimate)*100]],
                        color=colors[policy], fmt="o", capsize=3)
        axes[0].text(index, row.ci95_high*100+2, f"{row.estimate*100:.1f}%", ha="center", fontsize=7)
        for offset, metric, marker in ((-.09, "fixed_difference_seconds", "s"),
                                        (.09, "reactive_continuation_difference_seconds", "o")):
            row = diagnostic_row(summary, policy, metric)
            axes[1].errorbar(index+offset, row.estimate,
                            yerr=[[row.estimate-row.ci95_low], [row.ci95_high-row.estimate]],
                            color=colors[policy], fmt=marker, capsize=3,
                            markerfacecolor="white" if marker == "s" else colors[policy])
    axes[0].set_title("(a) Fixed-action clearance rank reversals", loc="left", fontweight="bold")
    axes[0].set_ylabel("Completed decisions (%)")
    axes[0].set_ylim(0, 45)
    axes[1].set_title("(b) Selected minus reactive action", loc="left", fontweight="bold")
    axes[1].set_ylabel("Residual clearance difference (s)")
    axes[1].axhline(0, color=INK, linestyle=":", linewidth=.8)
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], color=INK, marker="s", markerfacecolor="white", linestyle="", label="Fixed to clearance"),
                            Line2D([], [], color=INK, marker="o", linestyle="", label="120 s, then reactive")],
                   frameon=False, fontsize=6, loc="lower left")
    for ax in axes:
        ax.set_xticks([0, 1], [labels[U], labels[C]])
        ax.set_xlim(-.5, 1.5)
        ax.grid(axis="y", color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.075, right=.985, bottom=.17, top=.86, wspace=.38)
    save_ieee(fig, output / "fig4_decision_diagnostics")


def main() -> None:
    validate_frozen_artifact_bundle(ROOT / "paper_artifacts")
    REVISION.mkdir(parents=True, exist_ok=True)
    figures = REVISION / "figures"
    figures.mkdir(exist_ok=True)
    summary = pd.read_csv(REVISION / "decision_diagnostics_summary.csv")
    timing = pd.read_csv(REVISION / "runtime_comparison_summary.csv")
    original_macros()
    revision_macros(summary, timing)
    figure1(TABLES, figures)
    figure2(TABLES, figures)
    figure3(TABLES, figures)
    figure4(summary, figures)
    for path in figures.glob("*.pdf"):
        shutil.copyfile(path, PAPER / path.name)
    for name in ("results.tex", "revision_results.tex", "runtime_table.tex"):
        shutil.copyfile(PAPER / name, REVISION / name)
    lines = []
    for path in sorted(REVISION.rglob("*")):
        if path.is_file() and path.name != "ARTIFACTS.sha256":
            lines.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(REVISION).as_posix())
    (REVISION / "ARTIFACTS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated revised figures, CSV-derived TeX and manifest in {REVISION}")


if __name__ == "__main__":
    main()
