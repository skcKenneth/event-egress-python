from __future__ import annotations

from run_baselines import run as run_baselines
from run_capacity_stress import run as run_capacity_stress
from run_compliance_sensitivity import run as run_compliance_sensitivity
from run_sensor_ablation import run as run_sensor_ablation
from run_regime_map import run as run_regime_map
from run_model_mismatch import run as run_model_mismatch
from run_weight_sensitivity import run as run_weight_sensitivity
from run_incident_fallback import run as run_incident_fallback
from run_fallback_sensitivity import run as run_fallback_sensitivity
from run_travel_time_context import run as run_travel_time_context
from run_travel_time_context_variants import run as run_travel_time_context_variants
from run_nonstationary_grid import run as run_nonstationary_grid
from run_cvar_budget_sensitivity import run as run_cvar_budget_sensitivity
from run_surrogate_validation import run as run_surrogate_validation
from run_tail_validation import run as run_tail_validation
from _bootstrap import DEFAULT_WORKERS


from _bootstrap import DEFAULT_WORKERS

if __name__ == "__main__":
    run_baselines(seed_count=50, workers=DEFAULT_WORKERS)
    run_compliance_sensitivity(seed_count=30, workers=DEFAULT_WORKERS)
    run_capacity_stress(seed_count=30, workers=DEFAULT_WORKERS)
    run_sensor_ablation(seed_count=30, workers=DEFAULT_WORKERS)
    run_regime_map(seed_count=20, workers=DEFAULT_WORKERS)
    run_model_mismatch(seed_count=20, workers=DEFAULT_WORKERS)
    run_weight_sensitivity(seed_count=20, workers=DEFAULT_WORKERS)
    run_incident_fallback(seed_count=50, workers=DEFAULT_WORKERS)
    run_fallback_sensitivity(seed_count=20, workers=DEFAULT_WORKERS)
    run_travel_time_context(seed_count=50, workers=DEFAULT_WORKERS)
    run_travel_time_context_variants(seed_count=50, workers=DEFAULT_WORKERS)
    run_nonstationary_grid(seed_count=30, workers=DEFAULT_WORKERS)
    run_cvar_budget_sensitivity(seed_count=20, workers=DEFAULT_WORKERS)
    run_surrogate_validation(seed_count=8, workers=1)
    run_tail_validation(seed_count=200, workers=DEFAULT_WORKERS)
