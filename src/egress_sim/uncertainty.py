from __future__ import annotations

import numpy as np

from .config import EgressSpec
from .model import Scenario


def _sample_compliance(
    spec: EgressSpec,
    rng: np.random.Generator,
    *,
    compliance_mean: float | None = None,
) -> np.ndarray:
    uncertainty = spec.uncertainty
    mean = uncertainty.compliance_mean if compliance_mean is None else float(compliance_mean)
    concentration = uncertainty.compliance_concentration
    alpha = max(mean * concentration, 1e-6)
    beta = max((1.0 - mean) * concentration, 1e-6)
    compliance = rng.beta(alpha, beta, spec.zone_count)
    return np.clip(compliance, 0.02, 0.98)


def sample_centered_scenario(
    spec: EgressSpec,
    rng: np.random.Generator,
    path_center: np.ndarray,
    exit_center: np.ndarray,
    *,
    capacity_log_sigma: float | None = None,
    compliance_mean: float | None = None,
) -> Scenario:
    """Sample a capacity scenario around supplied centers.

    The function always consumes the same number and order of normal and beta
    draws, including when ``capacity_log_sigma`` is zero. Therefore two calls
    starting from the same RNG state use aligned standardized perturbations
    while changing only ``path_center`` and ``exit_center``.
    """

    uncertainty = spec.uncertainty
    sigma = (
        uncertainty.capacity_log_sigma
        if capacity_log_sigma is None
        else float(capacity_log_sigma)
    )
    if sigma < 0.0:
        raise ValueError("capacity_log_sigma must be non-negative")

    path_center = np.asarray(path_center, dtype=float)
    exit_center = np.asarray(exit_center, dtype=float)
    if path_center.shape != spec.path_capacity.shape:
        raise ValueError("path_center has incorrect shape")
    if exit_center.shape != spec.exit_capacity.shape:
        raise ValueError("exit_center has incorrect shape")

    # Drawing at sigma=0 is intentional: it preserves common-random-number
    # alignment between conditioned and unconditioned controllers.
    path_noise = np.exp(
        rng.normal(-0.5 * sigma**2, sigma, spec.path_capacity.shape)
    )
    exit_noise = np.exp(
        rng.normal(-0.5 * sigma**2, sigma, spec.exit_capacity.shape)
    )
    lower = uncertainty.capacity_multiplier_min
    upper = uncertainty.capacity_multiplier_max
    path_multiplier = np.clip(path_center * path_noise, lower, upper)
    exit_multiplier = np.clip(exit_center * exit_noise, lower, upper)

    return Scenario(
        path_capacity_multiplier=path_multiplier,
        exit_capacity_multiplier=exit_multiplier,
        compliance=_sample_compliance(spec, rng, compliance_mean=compliance_mean),
    )


def sample_scenario(
    spec: EgressSpec,
    rng: np.random.Generator,
    *,
    capacity_log_sigma: float | None = None,
    compliance_mean: float | None = None,
) -> Scenario:
    """Sample the persistent environment or an unconditioned prediction."""

    return sample_centered_scenario(
        spec,
        rng,
        np.ones_like(spec.path_capacity),
        np.ones_like(spec.exit_capacity),
        capacity_log_sigma=capacity_log_sigma,
        compliance_mean=compliance_mean,
    )


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
    """Sample around online estimates using a matched pre-clipping log-spread.

    By default, ``residual_sigma`` equals the environmental capacity log-sigma.
    This makes the primary conditioned-versus-unconditioned comparison a
    center-only intervention.  Alternative residual spreads remain available
    as explicit sensitivity analyses.
    """

    return sample_centered_scenario(
        spec,
        rng,
        path_center,
        exit_center,
        capacity_log_sigma=residual_sigma,
        compliance_mean=compliance_mean,
    )
