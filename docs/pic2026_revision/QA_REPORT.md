# PIC2026 revision verification — 2026-11-08

## Verified local outcomes

- All 45 unit tests passed, including 11 new diagnostic/runtime tests. The
  deterministic smoke simulation completed with mass error 9.095e-12.
  Tests used a workspace TEMP/TMP directory because the sandbox denied the
  default temporary directory during one preliminary run.
- Original frozen table/figure/summary manifests passed. Original
  `paper_artifacts/` has no Git diff against baseline `3a8271f`.
- The eight imported LaTeX CSVs match the original frozen CSVs byte-for-byte.
- Replayed environmental seeds 100000 and 100001 for Reactive, Unconditioned
  and Conditioned mean policies: all six simulations matched their frozen
  clearance, person-time, density and estimator diagnostics within numerical
  tolerance. This is a two-seed replay, not a rerun of all 200 original seeds.
- The diagnostic protocol was recorded in commit `1fd58cd` before executing
  new seeds; implementation commit `39720fc` preceded both production jobs.
- All 180 planned snapshots were eligible, and all 720 counterfactual branches
  completed. All 5000-draw seed-cluster summaries were independently reproduced
  within numerical precision. One portable NPZ snapshot independently
  reproduced exact candidates, selected action and prediction scenarios;
  all four of its counterfactual branch outcomes matched the CSVs exactly.
- Runtime produced 2700 timed decisions on 30 shared snapshots, 900 per policy,
  with five warmups and 30 timed repetitions per snapshot/policy. Metadata
  confirms thread environment values of one before NumPy import.
- Primary numerical macros, new diagnostic summaries and raw timing summaries
  were independently checked against their CSV sources. No favorable result
  was used as an acceptance criterion.

## New evidence and limits

Fixed-action clearance rank reversals were 26/90 (28.9%) on unconditioned
trajectories and 13/90 (14.4%) on conditioned trajectories. These frequencies
use all completed decisions as denominator; ties are separately recorded.
Because the state sets differ, the rate difference is descriptive and does
not isolate a causal conditioning effect. All four mean counterfactual effect
intervals include zero. The original direct conditioned-minus-reactive
comparison also remains unresolved: -3.00 s, 95% CI [-7.50, +1.35].

Runtime median/p95 in milliseconds: Reactive 0.0214/0.0387;
Unconditioned 4.0501/4.59036; Conditioned 4.0424/4.51525. The actual revision
environment was Python 3.14.6, NumPy 2.5.3, pandas 3.0.6, matplotlib 3.11.2.
These differ from the historical package pins; both environments are recorded
separately. Timing excludes preparation and I/O, and is machine-specific.

## Document QA

- Manuscript: six pages, within the supplied CFP's 4–10-page allowance.
- English response: three pages, covering all four review topics.
- Both compiled with latexmk/pdflatex; manuscript bibliography uses BibTeX.
  Final logs contain no unresolved references/citations, overfull or underfull
  boxes. All 18 cited bibliography keys resolve.
- All six manuscript pages and three response pages were visually inspected.
  Revised Figures 2 and 4 were also inspected at figure scale. Final-page
  columns were balanced and an excessive paragraph gap was removed.
- Abstract, tables, figures, English response and Traditional Chinese notes
  retain the same effect definitions and uncertainty statements.

## Remaining submission requirements

The manuscript is a review copy with an incomplete author/affiliation block.
Authors must provide the final details and review scientific content and the
AI-assistance disclosure. Current camera-ready instructions must be checked;
the supplied acceptance correspondence described November 20 as tentative.
No CMT submission, registration, payment, merge, publication or field
validation was performed as part of this revision.

Original reviewer PDFs and acceptance correspondence remain outside the public
repository. Historical protocol descriptions written now are retrospective;
they are not presented as recovered preregistration records.
