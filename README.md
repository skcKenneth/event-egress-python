# Event Egress: Matched Value-of-Information Benchmark

This repository contains the mesoscopic queue-network simulator and frozen experiment suite for:

> **When Does Online Estimation of Persistent Capacity Improve Dynamic Crowd Guidance? A Mesoscopic Queue-Network Study**

The research question is deliberately narrow: does online estimation of persistent but initially unknown path and exit capacities improve rolling-horizon guidance **beyond** reactivity and predictive lookahead?

The revised primary experiment is a center-only controller ablation. The conditioned and unconditioned mean controllers use the same state representation, candidate-construction rule, prediction horizon, scenario count, compliance model, objective, clipping rule, and policy-seed schedule. At each control update, independent observation, candidate, and scenario RNG substreams are derived from the same policy seed. The scenario substreams therefore remain aligned even after the policies induce different states or consume observation noise. Within the capacity-scenario model, the intervention is the prediction center: unit nominal capacity for the unconditioned controller versus the current throughput estimate for the conditioned controller. Both use the same pre-clipping capacity log-spread.

## Frozen headline result

<!-- HEADLINE_RESULTS_START -->
On the post-revision confirmatory range **100000-100199**, the equal-spread
conditioned mean controller averaged **705.80 s** versus **721.45 s** for the
unconditioned mean controller. The paired effect was **-15.65 s** (**-2.17%**;
95% paired-bootstrap interval **[-20.8, -10.8] s**). Exact policy means,
run-level results, tail metrics, and diagnostics are in
`paper_artifacts/RESULTS_SUMMARY.md` and `paper_artifacts/tables/`.
`CONFIRMATORY_PROTOCOL.md` retrospectively describes the tracked protocol;
no original preregistration record was supplied in this repository.
<!-- HEADLINE_RESULTS_END -->

## November 2026 targeted revision

The revision preserves the original frozen study and separately adds matched-state
action diagnostics and a three-policy runtime comparison. See
[the new diagnostic protocol](docs/pic2026_revision/DIAGNOSTIC_PROTOCOL.md) and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

```bash
python main.py decision-diagnostics
python main.py runtime-comparison
python main.py revision-artifacts
python main.py check-revision
```

Run the diagnostic and timing jobs sequentially. New reviewed data are under
`revision_artifacts/2026-11/`; manuscript and response sources are under `paper/`.
The direct frozen conditioned-minus-reactive effect is **-3.00 s**
(**95% CI [-7.50, +1.35] s**), so the default system-level advantage is unresolved.

## Scope and safety boundary

This is a synthetic computational benchmark, not an operational evacuation system. It does not model continuous pedestrian interaction, collision avoidance, pressure, injury risk, fire or smoke propagation, venue-specific geometry, or regulatory decision making. The queue-density output is a model proxy and must not be interpreted as a safety certification.

## Main capabilities

- discrete-time conservation-based queue network;
- source release, path queues, travel-delay cells, and exit queues;
- persistent stochastic path and exit capacities;
- partial compliance with route recommendations;
- static, reactive, unconditioned, throughput-conditioned, oracle, CVaR, and fallback policies;
- paired-seed experiments and paired bootstrap intervals;
- vectorized candidate-scenario rollouts, tested numerically against the scalar reference implementation;
- direct capacity-estimator convergence and saturation-observability diagnostics;
- uncertainty-compliance, time-step, topology-family, controller-facing capacity-estimate fault, and objective-sensitivity studies;
- frozen CSV tables, vector figures, a SHA-256 artifact manifest, and unit tests.

## Repository layout

```text
config/                       Base and topology-family configurations
src/egress_sim/               Simulator, policies, uncertainty, and analysis
experiments/                  Reproducible experiment runners
scripts/                      Verification and frozen-artifact checks
tests/                        Model, statistics, and protocol tests
paper_artifacts/tables/       Frozen CSV inputs used by the revised paper
paper_artifacts/figures/      Frozen PDF/PNG figures used by the revised paper
paper_artifacts/reference_manifest.json
outputs/                      Regenerated or exploratory outputs (git-ignored)
main.py                       Command-line entry point
REPRODUCIBILITY.md            Exact protocol and seed documentation
```

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the exact frozen-build dependency versions, use:

```bash
python -m pip install -r requirements-lock.txt
```

## Verify the implementation

```bash
python main.py verify
```

This runs the unit tests and a deterministic smoke simulation. The tests cover mass conservation, non-negativity, recommendation normalization, matched candidate construction, center-only scenario coupling, RNG-substream alignment, separate environment and policy seeds, estimator updates, topology configuration validity, and output statistics.

## Use the frozen paper artifacts

Check the table protocol, CSV manifest, and complete frozen-bundle checksums:

```bash
python main.py check-figures
```

Render the three IEEE-style figures from the frozen CSV files:

```bash
python main.py figures
```

The tracked artifacts are under `paper_artifacts/`. Exploratory runs write to `outputs/` and do not silently replace the frozen files.

## Regenerate the complete paper protocol

```bash
python main.py figure-pipeline --workers 4 --refresh-artifacts
```

The frozen protocol uses:

| Study | Paired seeds |
|---|---:|
| Confirmatory held-out primary validation | 200 per policy |
| Uncertainty-compliance map | 50 per cell and policy |
| Time-step robustness | 30 per step and policy |
| Estimator diagnostics | 50 per regime |
| Topology-family transfer | 50 per family and policy |
| Capacity-estimate fault diagnostics | 30 per regime and policy |

The pipeline regenerates all required tables, validates the seed structure, renders vector PDF and 600-dpi PNG figures, copies the successful results into `paper_artifacts/`, writes both the CSV manifest and complete-bundle checksums, and validates the copied artifacts byte-for-byte.

## Individual studies

```bash
python main.py final-validation --seeds 200 --workers 4
python main.py estimator --seeds 50 --workers 4
python main.py topology --seeds 50 --workers 4
python main.py sensor --seeds 30 --workers 4
```

The uncertainty-compliance map can also be checkpointed one environmental sigma at a time:

```bash
python experiments/run_information_regime_map.py --sigma 0.00 --seeds 50 --workers 4
python experiments/run_information_regime_map.py --sigma 0.15 --seeds 50 --workers 4
python experiments/run_information_regime_map.py --sigma 0.30 --seeds 50 --workers 4
python experiments/run_information_regime_map.py --sigma 0.45 --seeds 50 --workers 4
python experiments/run_information_regime_map.py --merge-checkpoints
```

Additional runners in `experiments/` cover residual-spread ablation, objective-weight sensitivity, model mismatch, nonstationary capacity stress, terminal-surrogate validation, and runtime measurement.

## Primary design correction

Earlier versions centered conditioned scenarios on online estimates while also narrowing their log-standard deviation. That conflated the value of the information center with a change in prediction dispersion. In this revision:

```text
unconditioned: center = 1,       pre-clipping log-spread = sigma
conditioned:   center = m_hat_t, pre-clipping log-spread = sigma
```

Both apply the same clipping interval and consume aligned scenario variates. Because clipping is nonlinear, changing the center can also change the post-clipping moments; the paper therefore states the intervention precisely as a change in the **pre-clipping scenario center** under a common spread and clipping rule. The legacy contracted spread, `max(0.05, 0.45 sigma)`, is retained only as the explicitly labeled secondary policy kind `conditioned_mean_contracted_spread`.

## Reproducibility notes

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for seed ranges, RNG coupling, file provenance, environment details, and interpretation of the estimator diagnostic. The frozen paper tables are the authoritative numerical source for the revised manuscript.

## Citation and licence status

Citation metadata are provided in `CITATION.cff`. No open-source license has been selected for this package; see `LICENSE_STATUS.md`. Scientific use should cite the accompanying paper and preserve the stated non-operational scope.

Before a public or camera-ready release, confirm the complete author list in `CITATION.cff`, fill the manuscript author/affiliation block, and select an explicit software license if redistribution is intended.
