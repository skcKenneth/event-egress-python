# PIC2026 revised manuscript (review copy)

This directory imports the user-supplied equal-spread LaTeX source and revises it against the four review topics. The source ZIP SHA-256 is `13d97cecb3092a2102df382d76a3473525b2224c00ea419240588104c68a777f`; the code baseline is `3a8271f482e68268560d3234b0790d3ba40c97cf`. The separate older 2.73% PDF and private review/acceptance files are retained locally outside this repository.

The manuscript uses the archived 200-run primary data, not a retuned primary experiment. Original `paper_artifacts/` files remain immutable. The additional protocol is recorded in `docs/pic2026_revision/DIAGNOSTIC_PROTOCOL.md`; new data and figures are separately versioned. Documentation reconstructed during the revision is retrospective, not evidence of historical preregistration.

## Sources and generated inputs

- `main.tex`: English IEEE-conference manuscript; `references.bib`: bibliography.
- `results.tex`: generated numerical macros from the original frozen tables.
- `revision_results.tex`, `runtime_table.tex`, and the four figure PDFs: generated from frozen tables and new diagnostic/runtime results.
- `response_to_reviewers.tex`: English topic-by-topic response.
- `revision_notes_zh.md`: Traditional Chinese revision explanation and claim boundaries.

Regenerate numerical inputs and figures using the revision build instructions in the repository reproduction documentation before compilation. Do not manually change generated numbers or overwrite the original frozen artifact bundle.

## Compile

The manuscript uses `IEEEtran`, no `fontspec`, and BibTeX. From this directory:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error response_to_reviewers.tex
```

Inspect the final logs for unresolved citations/references and the PDFs for layout defects. The CFP supplied with the source allows 4–10 pages including references; the revision targets approximately seven. Latest final-manuscript instructions take precedence when available.

## Submission status

These are review copies. `main.tex` has no completed author/affiliation block; no author identities or institutions have been inferred. The authors must supply those details, review all scientific content and the AI-assistance disclosure, and verify current conference instructions before submission. This repository revision does not submit to CMT or establish publication, field validation, or safety certification.
