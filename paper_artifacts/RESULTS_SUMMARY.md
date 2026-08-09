# Frozen results summary

This file is generated from `paper_artifacts/tables/`. Negative paired effects
favor the conditioned controller.

## Confirmatory held-out validation

- Static nearest mean clearance: 981.30 s.
- Reactive balance mean clearance: 708.80 s.
- Unconditioned mean clearance: 721.45 s.
- Equal-spread conditioned mean clearance: 705.80 s.
- Oracle-capacity mean clearance: 696.90 s.
- Primary center-only effect: -15.65 s
  (-2.17%; 95% paired-bootstrap interval
  [-20.75, -10.80] s).
- Additional contracted-spread effect relative to equal-spread conditioning:
  -5.00 s
  ([-7.50, -2.75] s).
- Oracle-minus-conditioned effect: -8.90 s
  ([-11.85, -6.25] s).

## Estimator diagnostics

- `default`: mean final capacity MAE 0.1062; mean observed-link fraction 37.2%.
- `high_information`: mean final capacity MAE 0.1381; mean observed-link fraction 47.6%.

## Network-family transfer

5 of 5 paired-bootstrap intervals lie entirely
below zero. Point estimates range from -29.80
to -2.60 s.

## Controller-facing estimate faults

- `default`: clean -14.67 s; fault range -20.00 to -13.33 s.
- `high_information`: clean -63.67 s; fault range -64.00 to -46.33 s.

See the CSV files for complete seed-level outputs, uncertainty-compliance cells,
time-step diagnostics, tail summaries, and confidence intervals.
