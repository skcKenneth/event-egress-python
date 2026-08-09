from __future__ import annotations

from _bootstrap import DEFAULT_WORKERS
from run_baselines import run as run_baselines
from run_capacity_stress import run as run_capacity_stress
from run_compliance_sensitivity import run as run_compliance_sensitivity
from run_estimator_diagnostics import run as run_estimator_diagnostics
from run_model_mismatch import run as run_model_mismatch
from run_sensor_ablation import run as run_sensor_ablation
from run_surrogate_validation import run as run_surrogate_validation
from run_topology_generalization import run as run_topology_generalization
from run_weight_sensitivity import run as run_weight_sensitivity


if __name__ == "__main__":
    # This is an exploratory suite, not the frozen paper protocol.  Use
    # run_ieee_figure_pipeline.py for the exact paper seed counts.
    run_baselines(seed_count=30, workers=DEFAULT_WORKERS)
    run_capacity_stress(seed_count=20, workers=DEFAULT_WORKERS)
    run_compliance_sensitivity(seed_count=20, workers=DEFAULT_WORKERS)
    run_estimator_diagnostics(seed_count=20, workers=DEFAULT_WORKERS)
    run_topology_generalization(seed_count=20, workers=DEFAULT_WORKERS)
    run_sensor_ablation(seed_count=20, workers=DEFAULT_WORKERS)
    run_weight_sensitivity(seed_count=15, workers=DEFAULT_WORKERS)
    run_model_mismatch(seed_count=15, workers=DEFAULT_WORKERS)
    run_surrogate_validation(seed_count=4, workers=1)
