"""Lognormal gas and fluctuating Gunn-Peterson approximation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .skewers import smooth_periodic


def lognormal_density(delta_gaussian: np.ndarray) -> np.ndarray:
    """Map Gaussian fluctuations to Delta_b > 0 with ensemble mean near one."""
    delta = np.asarray(delta_gaussian, dtype=float)
    variance = float(np.var(delta))
    return np.exp(delta - 0.5 * variance)


def fgpa_exponent(gamma: float) -> float:
    """Return beta = 2 - 0.7(gamma - 1)."""
    return 2.0 - 0.7 * (float(gamma) - 1.0)


def optical_depth_unit(density_ratio: np.ndarray, gamma: float) -> np.ndarray:
    """Return Delta_b^beta; the dimensional normalization A is fitted later."""
    density = np.asarray(density_ratio, dtype=float)
    if np.any(density <= 0):
        raise ValueError("density_ratio must be strictly positive")
    return density ** fgpa_exponent(gamma)


def remap_optical_depth_cic(
    tau: np.ndarray, displacement_pixels: np.ndarray
) -> np.ndarray:
    """Periodically remap optical depth with cloud-in-cell deposition.

    ``displacement_pixels`` is positive toward increasing pixel number.  This
    routine conserves the sum of optical depth along each periodic skewer.
    """
    values = np.asarray(tau, dtype=float)
    shifts = np.asarray(displacement_pixels, dtype=float)
    if values.shape != shifts.shape:
        raise ValueError("tau and displacement_pixels must have identical shapes")
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
        shifts = shifts[None, :]
    n_skewers, n_pixels = values.shape
    output = np.zeros_like(values)
    base = np.arange(n_pixels)
    for row in range(n_skewers):
        destination = (base + shifts[row]) % n_pixels
        left = np.floor(destination).astype(int)
        fraction = destination - left
        right = (left + 1) % n_pixels
        np.add.at(output[row], left, values[row] * (1.0 - fraction))
        np.add.at(output[row], right, values[row] * fraction)
    return output[0] if one_dimensional else output


def prepare_tau_unit(
    density_ratio: np.ndarray,
    *,
    gamma: float,
    displacement_pixels: np.ndarray | None = None,
    thermal_sigma_pixels: float = 0.0,
) -> np.ndarray:
    """Build unit-normalized redshift-space, thermally broadened optical depth."""
    tau_unit = optical_depth_unit(density_ratio, gamma)
    if displacement_pixels is not None:
        tau_unit = remap_optical_depth_cic(tau_unit, displacement_pixels)
    return smooth_periodic(tau_unit, thermal_sigma_pixels)


def calibrate_mean_flux(tau_unit: np.ndarray, target_mean_flux: float) -> float:
    """Solve <exp(-A tau_unit)> = target_mean_flux for A."""
    if not 0.0 < target_mean_flux < 1.0:
        raise ValueError("target_mean_flux must lie strictly between zero and one")
    tau = np.asarray(tau_unit, dtype=float)
    if np.any(tau < 0) or not np.any(tau > 0):
        raise ValueError("tau_unit must be non-negative and not identically zero")

    def residual(log_a: float) -> float:
        return float(np.mean(np.exp(-np.exp(log_a) * tau)) - target_mean_flux)

    return float(np.exp(brentq(residual, -30.0, 30.0)))


def transmitted_flux(tau_unit: np.ndarray, normalization: float) -> np.ndarray:
    """Calculate F = exp(-A tau_unit)."""
    return np.exp(-float(normalization) * np.asarray(tau_unit, dtype=float))


def flux_contrast(flux: np.ndarray) -> np.ndarray:
    """Calculate delta_F = F/<F> - 1 using the ensemble mean."""
    values = np.asarray(flux, dtype=float)
    mean_flux = float(np.mean(values))
    if mean_flux <= 0:
        raise ValueError("mean flux must be positive")
    return values / mean_flux - 1.0

