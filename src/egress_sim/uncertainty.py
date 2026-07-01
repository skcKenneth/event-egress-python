from __future__ import annotations

import numpy as np

from .config import EgressSpec
from .model import Scenario


def sample_scenario(
    spec: EgressSpec,
    rng: np.random.Generator,
    *,
    capacity_log_sigma: float | None = None,
    compliance_mean: float | None = None,
) -> Scenario:
    uncertainty = spec.uncertainty
    sigma = uncertainty.capacity_log_sigma if capacity_log_sigma is None else capacity_log_sigma
    mean = uncertainty.compliance_mean if compliance_mean is None else compliance_mean
    lower = uncertainty.capacity_multiplier_min
    upper = uncertainty.capacity_multiplier_max

    if sigma <= 0.0:
        path_multiplier = np.ones_like(spec.path_capacity)
        exit_multiplier = np.ones_like(spec.exit_capacity)
    else:
        path_multiplier = np.exp(rng.normal(-0.5 * sigma**2, sigma, spec.path_capacity.shape))
        exit_multiplier = np.exp(rng.normal(-0.5 * sigma**2, sigma, spec.exit_capacity.shape))
        path_multiplier = np.clip(path_multiplier, lower, upper)
        exit_multiplier = np.clip(exit_multiplier, lower, upper)

    concentration = uncertainty.compliance_concentration
    alpha = max(mean * concentration, 1e-6)
    beta = max((1.0 - mean) * concentration, 1e-6)
    compliance = rng.beta(alpha, beta, spec.zone_count)
    compliance = np.clip(compliance, 0.02, 0.98)
    return Scenario(path_multiplier, exit_multiplier, compliance)


def nominal_scenario(spec: EgressSpec) -> Scenario:
    return Scenario(
        path_capacity_multiplier=np.ones_like(spec.path_capacity),
        exit_capacity_multiplier=np.ones_like(spec.exit_capacity),
        compliance=np.full(spec.zone_count, spec.uncertainty.compliance_mean),
    )


def sample_conditioned_scenario(
    spec: EgressSpec,
    rng: np.random.Generator,
    path_center: np.ndarray,
    exit_center: np.ndarray,
    *,
    residual_sigma: float | None = None,
    compliance_mean: float | None = None,
) -> Scenario:
    uncertainty = spec.uncertainty
    residual_sigma = (
        max(0.05, 0.45 * uncertainty.capacity_log_sigma)
        if residual_sigma is None
        else residual_sigma
    )
    path_noise = np.exp(
        rng.normal(-0.5 * residual_sigma**2, residual_sigma, spec.path_capacity.shape)
    )
    exit_noise = np.exp(
        rng.normal(-0.5 * residual_sigma**2, residual_sigma, spec.exit_capacity.shape)
    )
    path_multiplier = np.clip(
        path_center * path_noise,
        uncertainty.capacity_multiplier_min,
        uncertainty.capacity_multiplier_max,
    )
    exit_multiplier = np.clip(
        exit_center * exit_noise,
        uncertainty.capacity_multiplier_min,
        uncertainty.capacity_multiplier_max,
    )
    concentration = uncertainty.compliance_concentration
    mean = uncertainty.compliance_mean if compliance_mean is None else compliance_mean
    compliance = rng.beta(
        max(mean * concentration, 1e-6),
        max((1.0 - mean) * concentration, 1e-6),
        spec.zone_count,
    )
    return Scenario(path_multiplier, exit_multiplier, np.clip(compliance, 0.02, 0.98))
