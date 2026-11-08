# November 2026 revision diagnostics

This directory contains new action diagnostics and machine-specific runtime
observations. Original 200-seed paper artifacts remain unchanged elsewhere.

The protocol was committed at `1fd58cde1ab5563018e595b012d640bcca49f5c2`
before production execution. The implementation was committed at
`39720fca10f0fc2b059d75c4a9d39c5c130e3bf9` before either production job.

Thirty seeds yielded 180 eligible pre-decision snapshots and 720 completed
counterfactual branches; no snapshot was skipped and no branch was censored.
Two controller trajectories and three phases per seed are dependent samples.
Summary inference averages within seed and bootstraps 30 environmental seeds.
Rank reversal frequencies count strict sign reversals over all completed
decisions; predicted/actual ties are recorded separately. Objective-favored
but slower frequencies are joint frequencies, not conditional risk estimates.

Runtime uses 30 shared snapshots from the first five seeds, five warmups and
30 timed repetitions per controller per snapshot (2700 timed calls total).
Raw timings, phase summaries and actual hardware/software metadata are included.
Measured decision time excludes copying, construction and I/O.

`ARTIFACTS.sha256` covers this directory except itself. Portable NPZ snapshots
include full state, realized scenario and pre-decision RNG for inspection.
Figures and numerical TeX inputs are regenerated from the CSVs by
`python main.py revision-artifacts`; this never overwrites `paper_artifacts/`.
