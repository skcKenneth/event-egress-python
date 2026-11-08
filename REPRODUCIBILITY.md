# Reproducing the PIC2026 paper and revision

This November 2026 document is a retrospective description of the original
bundle plus instructions for the new diagnostics. Original frozen files under
`paper_artifacts/` remain the numerical authority for the 200-seed study.
`.gitattributes` protects their bytes from Windows newline conversion.

## Environment and original checks

The original dependency pins remain in `requirements-lock.txt`. The separately
recorded revision environment is in `requirements-revision-lock.txt` and the
new result metadata. Use a clean virtual environment and compatible Python.
Version differences must be reported; the two-seed replay checks agreement on
this machine, not universal bitwise reproducibility across all environments.

```powershell
python main.py verify
python main.py check-figures
python main.py check-revision
```

The original full-suite commands are documented in README. They write to
`outputs/` unless explicit artifact refresh is requested. Do not run
`--refresh-artifacts` when reproducing this revision.

## New, separately recorded diagnostics

Read `docs/pic2026_revision/DIAGNOSTIC_PROTOCOL.md` first. Run sequentially so
the runtime job is not competing with the numerical diagnostic job:

```powershell
python main.py decision-diagnostics
python main.py runtime-comparison
```

Default diagnostic seeds are 110000–110029; runtime uses the first five. Both
save only pre-decision states at 120/360/600 seconds, including eligibility
records. The benchmark sets BLAS thread environment variables to one before
NumPy imports. The existing `runtime` command is the legacy initial-state
benchmark and does not reproduce the three-policy comparison.

Raw regeneration outputs go to `outputs/pic2026_revision/`. The reviewed copy
is retained separately under `revision_artifacts/2026-11/`, with its own hashes.
Use the artifact renderer after copying validated new tables there:

```powershell
python main.py revision-artifacts
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/response_to_reviewers.tex
```

The renderer regenerates numerical TeX inputs from frozen/new CSVs and writes
revised figures to the paper and revision directories. It never writes into
the original `paper_artifacts/`. Build auxiliaries are ignored by Git.

## Interpretation and release boundary

Repeated decisions are clustered by environmental seed for inference. Censored
branches are marked incomplete, not assigned the simulation cap as clearance.
The snapshot counterfactuals measure action consequences under specified
continuations, not full RH-versus-reactive policy effects. Runtime observations
are hardware-specific and do not certify operational real-time performance.

The submitted manuscript version in CMT was not inspected. Manuscript author
order, affiliation and emails were filled from the author's November 8, 2026
information. Final author review and current conference instructions remain
submission requirements. Private review PDFs and correspondence are excluded
from this repo.
