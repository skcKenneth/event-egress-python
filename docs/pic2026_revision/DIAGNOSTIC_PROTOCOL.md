# PIC2026 targeted diagnostic protocol

Recorded on 2026-11-08 before executing the new diagnostic seed block.
Baseline: `3a8271f482e68268560d3234b0790d3ba40c97cf`.
This is a protocol for the new revision diagnostics, not a retrospective
preregistration of the original experiments. No favorable result is required.

## Fixed design

- Use `config/default.json` without tuning. Collector policies are
  `unconditioned_mean` and `conditioned_mean` (equal pre-clipping spread,
  mean objective, no CVaR weight).
- Environmental seeds: 110000–110029 inclusive; policy seeds are environmental
  seed + 1000003, i.e. 1110003–1110032. This block is disjoint from the tracked
  experiment defaults and frozen ranges inspected at the baseline commit.
- Save each collector's state immediately before actual decisions at absolute
  times 120, 360, and 600 seconds, at most 180 snapshots. Already-cleared
  snapshots are marked ineligible and are not replaced.
- Save the complete state, including transit, accumulated person-time,
  recommendation, estimates, histories and saturation counts; preserve the
  pre-decision policy and RNG state and the realized capacity/compliance.
- Trace the actual decision evaluation without extra RNG draws. Retain every
  candidate recommendation, objective and constituent costs, selected index,
  and reactive index. Recommendations equivalent at rtol=0, atol=1e-12 count
  as reactive selections even when their candidate indices differ.

## Same-state counterfactuals

For selected and reactive recommendations from each identical saved state:

1. Hold each recommendation fixed until clearance.
2. Hold each recommendation for 120 seconds, then apply the same reactive
   policy, recomputing every 120 seconds thereafter. Recommendations can
   differ across branches because their resulting states differ.

Both branches use the identical realized persistent capacity and compliance.
Measure elapsed seconds from the snapshot. Preserve the original absolute
3600-second simulation ceiling (do not restart the budget at the snapshot).
Record completion and remaining population. A censored duration is not an
observed clearance time; effect summaries use complete pairs and disclose
incomplete counts. These are action diagnostics, not full RH-versus-reactive
policy comparisons.

## Outcomes and inference

- Report reactive-selection rate, full-objective gap (selected minus reactive),
  clearance-component gap, and actual residual-clearance differences for both
  continuation modes. Keep multiobjective choices separate from failures of
  the clearance surrogate.
- For fixed-action ranking, a strict opposite sign of predicted clearance
  difference and actual clearance difference is a reversal. Report predicted
  and actual ties separately; use 1e-9 numerical tolerance. Also report the
  frequency of full-objective-favored actions that increase realized clearance.
- Provide raw decision/candidate/branch rows, snapshot eligibility, per-seed
  summaries, and aggregate summaries by collector, phase and overall.
- Aggregate within each environmental seed before computing across-seed means;
  bootstrap environmental seeds 5000 times for percentile 95% intervals.
  Never treat repeated decisions or both trajectories as independent seeds.
  Use a fixed recorded analysis RNG seed. Disclose eligible seed/snapshot counts.

## Runtime comparison

- First five environmental seeds, 110000–110004, both collectors, all eligible
  phases (at most 30 shared snapshots).
- On every shared snapshot benchmark Reactive, Unconditioned mean and
  Conditioned equal-spread mean. Clone the complete state, policy state and
  pre-decision RNG before every call; disable optional tracing for timing.
- Five warmup rounds then 30 timed rounds per policy/snapshot. Rotate order
  Reactive/U/C, U/C/Reactive, C/Reactive/U. Use a serial process with BLAS
  thread settings fixed to one before importing NumPy. Run while no diagnostic
  or other task-controlled numerical job is active.
- Time only `decide` using a monotonic high-resolution clock, excluding policy
  construction, state/RNG cloning and I/O. Record raw rows, phase and overall
  median/p95, hardware, Python/package versions, thread settings and source SHA.
- This is a machine-specific computational-cost observation, not a hard
  real-time guarantee or an operational evacuation validation.

## Preservation and checks

Original `paper_artifacts` bytes and checksums remain unchanged. New diagnostic
outputs and revised figures are kept separately. Test instrumentation parity,
RNG parity, snapshot independence, identical-action parity, continuation timing,
censoring, and scalar/batched cost components. Replay primary seeds 100000 and
100001 against frozen records. Report negative and null diagnostics unchanged.

Author/affiliation completion and current conference instructions remain
submission gates. Reviewer PDFs and acceptance correspondence remain local.
