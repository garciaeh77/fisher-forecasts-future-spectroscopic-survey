"""Optical-depth FGPA on 1D lognormal skewers.

This sits beside the Kaiser formula in flux.py. The Kaiser path is the fast
P_1D(k, z). This path builds a density, sets tau = A * Delta^beta, fixes A
so the mean transmitted flux matches a target, and measures P_1D from the
skewers. Pressure and thermal smoothing default to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from p1d.flux import k_hmpc_to_skm, p1d_h_to_kms
from p1d.matter_table import TabulatedMatterPower
from p1d.projection import project_p1d
from p1d.skewers import generate_gaussian_skewers, measure_p1d, rfft_modes, smooth_periodic


def lognormal_density(delta_gaussian):
    """Map Gaussian fluctuations to Delta_b > 0 with ensemble mean near one."""
    delta = np.asarray(delta_gaussian, dtype=float)
    variance = float(np.var(delta))
    return np.exp(delta - 0.5 * variance)


def fgpa_exponent(gamma):
    """beta = 2 - 0.7 (gamma - 1), from the temperature-density relation."""
    return 2.0 - 0.7 * (float(gamma) - 1.0)


def optical_depth_unit(density_ratio, gamma):
    """Delta_b^beta. The amplitude A is fitted separately."""
    density = np.asarray(density_ratio, dtype=float)
    if np.any(density <= 0):
        raise ValueError("density_ratio must be strictly positive")
    return density ** fgpa_exponent(gamma)


def remap_optical_depth_cic(tau, displacement_pixels):
    """Periodic cloud-in-cell remap. displacement_pixels increases with pixel number."""
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


def prepare_tau_unit(density_ratio, gamma, displacement_pixels=None, thermal_sigma_pixels=0.0):
    """Unit optical depth, optionally displaced and thermally smoothed."""
    tau_unit = optical_depth_unit(density_ratio, gamma)
    if displacement_pixels is not None:
        tau_unit = remap_optical_depth_cic(tau_unit, displacement_pixels)
    return smooth_periodic(tau_unit, thermal_sigma_pixels)


def calibrate_mean_flux(tau_unit, target_mean_flux):
    """Solve <exp(-A tau_unit)> = target_mean_flux for A."""
    if not 0.0 < target_mean_flux < 1.0:
        raise ValueError("target_mean_flux must lie strictly between zero and one")
    tau = np.asarray(tau_unit, dtype=float)
    if np.any(tau < 0) or not np.any(tau > 0):
        raise ValueError("tau_unit must be non-negative and not identically zero")

    def residual(log_a):
        return float(np.mean(np.exp(-np.exp(log_a) * tau)) - target_mean_flux)

    return float(np.exp(brentq(residual, -30.0, 30.0)))


def transmitted_flux(tau_unit, normalization):
    """F = exp(-A tau_unit)."""
    return np.exp(-float(normalization) * np.asarray(tau_unit, dtype=float))


def flux_contrast(flux):
    """delta_F = F/<F> - 1."""
    values = np.asarray(flux, dtype=float)
    mean_flux = float(np.mean(values))
    if mean_flux <= 0:
        raise ValueError("mean flux must be positive")
    return values / mean_flux - 1.0


@dataclass(frozen=True)
class SkewerConfig:
    redshift: float = 3.0
    As: float = 2.1e-9
    ns: float = 0.965
    omega_m: float = 0.31
    omega_b: float = 0.049
    h: float = 0.67
    growth_ratio: float = 1.0
    length_mpch: float = 200.0
    n_pixels: int = 1024
    n_skewers: int = 256
    target_mean_flux: float = 0.80
    gamma: float = 1.5
    pressure_sigma_pixels: float = 0.0
    thermal_sigma_pixels: float = 0.0
    seed: int = 12345


def run_fgpa_skewers(matter_power: TabulatedMatterPower, config: SkewerConfig, displacement_pixels=None):
    """Project P_CDM, draw skewers, apply the optical-depth FGPA, measure P_1D.

    Returns comoving and forest-unit spectra. Forest k is in s/km and forest
    P_1D is in km/s, using H(z)/((1+z) h).
    """
    k_modes = rfft_modes(config.length_mpch, config.n_pixels)
    nonzero = k_modes > 0
    if np.max(k_modes[nonzero]) > matter_power.kmax:
        raise ValueError("skewer Nyquist mode exceeds the P_CDM table kmax")
    if np.min(k_modes[nonzero]) < matter_power.kmin:
        raise ValueError("fundamental skewer mode lies below the P_CDM table kmin")

    q_grid = np.geomspace(matter_power.kmin, matter_power.kmax, 8192)
    pk3d = matter_power(q_grid, As=config.As, ns=config.ns, growth_ratio=config.growth_ratio)
    p1d_target = np.zeros_like(k_modes)
    p1d_target[nonzero] = project_p1d(k_modes[nonzero], q_grid, pk3d)

    gaussian = generate_gaussian_skewers(
        p1d_target,
        length_mpch=config.length_mpch,
        n_pixels=config.n_pixels,
        n_skewers=config.n_skewers,
        seed=config.seed,
    )
    gaussian = smooth_periodic(gaussian, config.pressure_sigma_pixels)
    density = lognormal_density(gaussian)
    tau_unit = prepare_tau_unit(
        density,
        gamma=config.gamma,
        displacement_pixels=displacement_pixels,
        thermal_sigma_pixels=config.thermal_sigma_pixels,
    )
    normalization = calibrate_mean_flux(tau_unit, config.target_mean_flux)
    flux = transmitted_flux(tau_unit, normalization)
    delta_flux = flux_contrast(flux)
    p1d_flux = measure_p1d(delta_flux, config.length_mpch)
    z, om, h = config.redshift, config.omega_m, config.h
    return {
        "k_hmpc": k_modes,
        "k_skm": k_hmpc_to_skm(k_modes, z, om, h),
        "pk3d_mpc3h3": pk3d,
        "q_hmpc": q_grid,
        "p1d_matter_mpc_h": p1d_target,
        "p1d_matter_measured_mpc_h": measure_p1d(gaussian, config.length_mpch),
        "p1d_flux_mpc_h": p1d_flux,
        "p1d_flux_kms": p1d_h_to_kms(p1d_flux, z, om, h),
        "fgpa_A": normalization,
        "fgpa_beta": fgpa_exponent(config.gamma),
        "mean_flux": float(np.mean(flux)),
        "density_mean": float(np.mean(density)),
        "density_variance": float(np.var(density)),
        "sample_flux_skewer": flux[0],
        "pixel_mpch": config.length_mpch / config.n_pixels,
    }
