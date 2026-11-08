# Figure refinement verification — 2026-11-08

Baseline: `dc1d9f4b901751af99f760e3c2b0e64add413a24`.
This is a presentation-only change with a hard six-page, four-main-figure limit.

## Scientific preservation

- All 231 files in the original and prior-revision artifact bundles retained
  their bytes and checksums. No experiment runner or simulator source changed.
- All existing CSVs, estimates, confidence bounds and bootstrap rules remain
  unchanged. No research experiment was rerun.
- Outside the four figure environments and typesetting commands, manuscript
  body, equations, numerical macros and author information match the baseline.
- Tests compare every displayed effect/interval with the existing CSVs, retain
  the original marginal bootstrap seeds, and compare histogram bin counts,
  estimator percentile-band vertices, heatmap symbols and diagnostic units.

## Statistical interpretation

- Policy categories have no connecting line. Their marginal mean intervals are
  distinguished from paired-effect intervals; simulations remain paired by seed.
- Main contrasts remain -15.65 s [-20.75, -10.80] and -3.00 s [-7.50, +1.35].
- The MAE band is the 10th–90th percentile range across 50 runs at each plotted
  time, not a confidence band. Optional histogram percentile shading was removed.
- Heatmap stars include zero, including [0,0]. Time-step ticks are exactly
  10, 5 and 2.5 seconds at their numerical positions.
- Effect markers are hollow when their 95% interval contains zero. Figure 4
  introduces no U-minus-C test; the two collector state sets remain distinct.

## Readability and compilation

- The fixed figure width matches the measured IEEEtran text width: 516 TeX
  points. Major labels and legends are at least 7.2 pt at final placement;
  principal annotations are 8 pt. Mathematical scripts are smaller by design.
- Automated checks verify text remains inside each fixed canvas. A regression
  test verifies heatmap foreground/background contrast is at least 4.5:1.
- All 53 unit tests passed, including eight figure-preservation/readability
  tests. Original frozen-bundle validation also passed.
- The manuscript compiled to six pages with four figures, with no unresolved
  citations/references or overfull boxes. A first-page underfull-vbox warning
  was inspected visually and has no visible clipping or oversized paragraph gap.
- All six manuscript pages were reviewed at final size; grayscale checks retain
  CI visibility and marker-shape/fill distinctions. The four-page before/after
  companion is separate from the manuscript.

## Delivery verification

Final PDF and Overleaf ZIP are exported from committed raw Git blobs. Each ZIP
member is compared with that commit, then the ZIP is compiled in an independent
directory and its PDF text is compared with the delivered PDF. Previous delivery
files are archived before replacing the latest filenames. Merge status and final
commit/file hashes are recorded separately in `DELIVERY_MANIFEST.json` and
`FINAL_VERIFICATION.json` beside the local deliverables.

The figure-only PR is to be merged only after these checks pass. This change
does not submit the manuscript to CMT or rerun scientific experiments.
