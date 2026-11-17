"""Optical-depth FGPA on 1D lognormal skewers (JAX).

A is solved with a fixed-iteration bisection so the calibration stays
JIT / differentiable-friendly (gradients flow through the final A).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from p1d_jax.flux import k_hmpc_to_skm, p1d_h_to_kms
from p1d_jax.matter_table import TabulatedMatterPower
from p1d_jax.projection import project_p1d
from p1d_jax.skewers import generate_gaussian_skewers, measure_p1d, rfft_modes, smooth_periodic


def lognormal_density(delta_gaussian):
    """Map Gaussian fluctuations to Delta_b > 0 with ensemble mean near one."""
    delta = jnp.asarray(delta_gaussian, dtype=jnp.float64)
    variance = jnp.var(delta)
    return jnp.exp(delta - 0.5 * variance)


def fgpa_exponent(gamma):
    """beta = 2 - 0.7 (gamma - 1), from the temperature-density relation."""
    return 2.0 - 0.7 * (jnp.asarray(gamma, dtype=jnp.float64) - 1.0)


def optical_depth_unit(density_ratio, gamma):
    """Delta_b^beta. The amplitude A is fitted separately."""
    density = jnp.asarray(density_ratio, dtype=jnp.float64)
    return density ** fgpa_exponent(gamma)


def remap_optical_depth_cic(tau, displacement_pixels):
    """Periodic cloud-in-cell remap. displacement_pixels increases with pixel number."""
    values = jnp.asarray(tau, dtype=jnp.float64)
    shifts = jnp.asarray(displacement_pixels, dtype=jnp.float64)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
        shifts = shifts[None, :]
    n_pixels = values.shape[-1]
    base = jnp.arange(n_pixels, dtype=jnp.float64)

    def remap_row(vals, shift):
        destination = (base + shift) % n_pixels
        left = jnp.floor(destination).astype(jnp.int32)
        fraction = destination - left
        right = (left + 1) % n_pixels
        out = jnp.zeros(n_pixels, dtype=jnp.float64)
        out = out.at[left].add(vals * (1.0 - fraction))
        out = out.at[right].add(vals * fraction)
        return out

    remapped = jax.vmap(remap_row)(values, shifts)
    return remapped[0] if one_dimensional else remapped


def prepare_tau_unit(density_ratio, gamma, displacement_pixels=None, thermal_sigma_pixels=0.0):
    """Unit optical depth, optionally displaced and thermally smoothed."""
    tau_unit = optical_depth_unit(density_ratio, gamma)
    if displacement_pixels is not None:
        tau_unit = remap_optical_depth_cic(tau_unit, displacement_pixels)
    return smooth_periodic(tau_unit, thermal_sigma_pixels)


def calibrate_mean_flux(tau_unit, target_mean_flux, n_iter=64):
    """Solve <exp(-A tau_unit)> = target_mean_flux for A via fixed bisection."""
    tau = jnp.asarray(tau_unit, dtype=jnp.float64)
    target = jnp.asarray(target_mean_flux, dtype=jnp.float64)

    def residual(log_a):
        return jnp.mean(jnp.exp(-jnp.exp(log_a) * tau)) - target

    def body(_i, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        # residual > 0 => mean flux too high => A too small => raise lo
        too_high = residual(mid) > 0.0
        return jnp.where(too_high, mid, lo), jnp.where(too_high, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (-30.0, 30.0))
    return jnp.exp(0.5 * (lo + hi))


def transmitted_flux(tau_unit, normalization):
    """F = exp(-A tau_unit)."""
    return jnp.exp(-jnp.asarray(normalization, dtype=jnp.float64) * jnp.asarray(tau_unit, dtype=jnp.float64))


def flux_contrast(flux):
    """delta_F = F/<F> - 1."""
    values = jnp.asarray(flux, dtype=jnp.float64)
    mean_flux = jnp.mean(values)
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
    """Project P_CDM, draw skewers, apply the optical-depth FGPA, measure P_1D."""
    k_modes = rfft_modes(config.length_mpch, config.n_pixels)
    q_grid = jnp.geomspace(matter_power.kmin, matter_power.kmax, 8192)
    pk3d = matter_power(q_grid, As=config.As, ns=config.ns, growth_ratio=config.growth_ratio)
    p1d_all = project_p1d(k_modes, q_grid, pk3d)
    p1d_target = jnp.where(k_modes > 0.0, p1d_all, 0.0)

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
        "mean_flux": jnp.mean(flux),
        "density_mean": jnp.mean(density),
        "density_variance": jnp.var(density),
        "sample_flux_skewer": flux[0],
        "pixel_mpch": config.length_mpch / config.n_pixels,
    }
