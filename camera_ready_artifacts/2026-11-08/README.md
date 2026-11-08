# Camera-ready figure refinement — 2026-11-08

Presentation baseline: `dc1d9f4b901751af99f760e3c2b0e64add413a24`.
The manuscript remains six pages with the same four main figures. No research
experiments, observations, estimates, bootstrap rules or intervals were changed.

Run `python main.py camera-ready-figures` from the repository root. The renderer
checks original and prior-revision manifests, records every input hash and all
plotted numbers, and uses a fixed canvas matching IEEEtran's measured text width
(516 TeX points). Major labels/legends are at least 7.2 pt, principal annotations
8 pt, and figure titles 8.5 pt. Mathematical script glyphs are smaller by design.
`figure_metadata.json` records the sizes and canvas-boundary checks.

- Figure 1 preserves the diagram and heatmap data. Stars mean paired-bootstrap
  intervals include zero, including degenerate [0,0] intervals. Cell borders
  mark intervals wholly below zero. Heatmap text passes a 4.5:1 contrast check.
- Figure 2 removes policy connectors and optional histogram percentile shading,
  enlarges the two primary paired contrasts, and defines the estimator band as
  the run-level MAE 10th–90th percentiles across 50 seeds, not a confidence band.
- Figure 3 retains numerical time-step positions and uses only tested ticks;
  near-zero topology intervals and the two fault-regime series remain visible.
- Figure 4 preserves sampling units and different collector state sets; it adds
  no U-minus-C inference or connectors. Effect marker fill consistently indicates
  whether a 95% interval includes zero; shape identifies continuation.

`figures/` contains fixed-width PDF and 600-dpi preview PNG files. `before/`
contains baseline figure PDFs from Git. `figure_comparison.pdf` and its portable
LaTeX source show before/after views on four separate companion pages; these
pages are not added to the six-page manuscript. `figure_data.json` contains the
exact plotting inputs. `ARTIFACTS.sha256` covers this directory except itself.

Compile the comparison from this directory with latexmk and a separate output
directory; compile the manuscript using the commands in `paper/README.md`.
