# Original validation: retrospective protocol description

Written for the November 2026 revision from the tracked code and frozen tables.
This document was not present in the baseline repository. It is not a recovered
preregistration and does not establish when the original outcomes were first
inspected. The original source described a held-out post-revision validation;
the verifiable record supplied here is its implementation and frozen results.

The primary environmental range is 100000–100199, paired across six policies.
The primary comparison is `conditioned_mean_equal_spread` minus
`unconditioned_mean_only`. Each simulation samples persistent capacity and
compliance once. Policy seeds equal environmental seed + 1000003. At each
update, observation, candidate and scenario RNG substreams are separated.

The primary controllers share current queue observations, candidate-generation
rules, 16 candidates, eight scenarios, a 300-second fixed-action horizon,
120-second update interval, mean objective, capacity clipping and pre-clipping
log-spread. The capacity-scenario center is nominal versus the online estimate.
Clipping can change post-clipping moments when the center changes.

Frozen mean clearances are 721.45 and 705.80 seconds. The paired difference is
-15.65 seconds (-2.17%), with percentile paired-bootstrap interval
[-20.75, -10.80] seconds. Bootstrap resampling uses whole paired seeds (5000
draws). The frozen direct conditioned-minus-reactive comparison is -3.00
seconds, interval [-7.50, +1.35]; this does not establish superiority over reactive.

Other policies are static nearest, reactive balancing, contracted-spread
conditioning (secondary) and oracle-capacity mean (diagnostic). The oracle
also removes capacity-scenario spread and is not an optimality bound.

See `experiments/run_information_value_validation.py`,
`paper_artifacts/reference_manifest.json` and the frozen effects/run-level CSVs.
The new diagnostic protocol is separately recorded before its execution in
`docs/pic2026_revision/DIAGNOSTIC_PROTOCOL.md`.
