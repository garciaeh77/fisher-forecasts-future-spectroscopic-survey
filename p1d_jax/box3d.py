"""Periodic Gaussian box and binned P(k, μ) measurement (JAX)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from p1d_jax.eisenstein_hu import growth
from p1d_jax.fgpa_field import (
    calibrate_mean_flux,
    flux_contrast,
    lognormal_density,
    optical_depth_unit,
    transmitted_flux,
)
from p1d_jax.skewers import smooth_periodic


def growth_rate(z, omega_m, h, dz=1.0e-2):
    """f = d ln D / d ln a from the growth function used for P_CDM."""
    z0 = jnp.asarray(z, dtype=jnp.float64)
    z_hi = z0 + dz
    z_lo = jnp.maximum(z0 - dz, 0.0)
    d_hi = growth(z_hi, omega_m, h)
    d_lo = growth(z_lo, omega_m, h)
    ln_d_hi = jnp.log(d_hi if jnp.ndim(d_hi) == 0 else d_hi[0])
    ln_d_lo = jnp.log(d_lo if jnp.ndim(d_lo) == 0 else d_lo[0])
    ln_a_hi = jnp.log(1.0 / (1.0 + z_hi))
    ln_a_lo = jnp.log(1.0 / (1.0 + z_lo))
    return (ln_d_hi - ln_d_lo) / (ln_a_hi - ln_a_lo)


def _mode_grid(n, length):
    k1d = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=length / n)
    kx, ky, kz = jnp.meshgrid(k1d, k1d, k1d, indexing="ij")
    k = jnp.sqrt(kx**2 + ky**2 + kz**2)
    mu = jnp.where(k > 0.0, jnp.abs(kz) / k, 0.0)
    return k, mu, kz


def _independent_mode_mask(n):
    index = jnp.arange(n)
    linear = jnp.arange(n**3).reshape(n, n, n)
    neg = (-index) % n
    linear_neg = linear[jnp.ix_(neg, neg, neg)]
    independent = linear < linear_neg
    self_conjugate = linear == linear_neg
    return independent, self_conjugate, neg


def gaussian_box(power, length, n, seed, exact_amplitude=False):
    """Real Gaussian field whose modes have variance P(k)/L³.

    ``power(k)`` is P(k) in (Mpc/h)³ for k in h/Mpc. The zero mode is zero.
    ``exact_amplitude`` replaces the Rayleigh draw by the rms.
    ``seed`` may be an int or a PRNG key.
    """
    k, mu, kz = _mode_grid(n, length)
    # Avoid log(0) inside tabulated power: evaluate at a dummy k where k == 0.
    k_safe = jnp.where(k > 0.0, k, 1.0)
    pk = jnp.where(k > 0.0, jnp.asarray(power(k_safe), dtype=jnp.float64), 0.0)
    sigma = jnp.sqrt(0.5 * n**6 * pk / length**3)
    independent, self_conjugate, neg = _independent_mode_mask(n)

    key = seed if isinstance(seed, jax.Array) else jax.random.PRNGKey(int(seed))
    key_re, key_im, key_phase, key_sign = jax.random.split(key, 4)
    re = jax.random.normal(key_re, (n, n, n), dtype=jnp.float64)
    im = jax.random.normal(key_im, (n, n, n), dtype=jnp.float64)

    if exact_amplitude:
        phase = jnp.exp(1j * jax.random.uniform(key_phase, (n, n, n), minval=0.0, maxval=2.0 * jnp.pi))
        signs = jnp.where(
            jax.random.bernoulli(key_sign, 0.5, (n, n, n)),
            1.0,
            -1.0,
        )
        fk = jnp.where(independent, phase * sigma * jnp.sqrt(2.0), 0.0 + 0.0j)
        fk = jnp.where(self_conjugate, (signs * sigma * jnp.sqrt(2.0)).astype(jnp.complex128), fk)
    else:
        fk = jnp.where(independent, sigma * (re + 1j * im), 0.0 + 0.0j)
        fk = jnp.where(
            self_conjugate,
            (sigma * jnp.sqrt(2.0) * re).astype(jnp.complex128),
            fk,
        )

    partner = jnp.conj(fk[jnp.ix_(neg, neg, neg)])
    linear = jnp.arange(n**3).reshape(n, n, n)
    linear_neg = linear[jnp.ix_(neg, neg, neg)]
    fk = fk + jnp.where(linear > linear_neg, partner, 0.0)
    delta = jnp.fft.ifftn(fk).real
    return {
        "delta": delta,
        "k": k,
        "mu": mu,
        "kz": kz,
        "pk": pk,
        "fourier": fk,
    }


def measure_power(delta, length):
    """Mode-by-mode P(k) in (Mpc/h)³, with the same normalisation as gaussian_box."""
    n = delta.shape[0]
    fk = jnp.fft.fftn(delta)
    return (length**3 / n**6) * jnp.abs(fk) ** 2


def bin_power(k, mu, power, k_edges, mu_edges):
    """Mean power in rectangular (k, μ) bins. Empty bins are nan."""
    k = jnp.asarray(k, dtype=jnp.float64).ravel()
    mu = jnp.asarray(mu, dtype=jnp.float64).ravel()
    power = jnp.asarray(power, dtype=jnp.float64).ravel()
    k_edges = jnp.asarray(k_edges, dtype=jnp.float64)
    mu_edges = jnp.asarray(mu_edges, dtype=jnp.float64)
    n_k = k_edges.shape[0] - 1
    n_mu = mu_edges.shape[0] - 1

    use = (k > 0.0) & jnp.isfinite(power)
    k_index = jnp.digitize(k, k_edges) - 1
    mu_index = jnp.digitize(mu, mu_edges) - 1
    inside = use & (k_index >= 0) & (k_index < n_k) & (mu_index >= 0) & (mu_index < n_mu)
    flat = jnp.where(inside, mu_index * n_k + k_index, 0)
    weights = jnp.where(inside, power, 0.0)
    counts_w = jnp.where(inside, 1.0, 0.0)
    sums = jnp.bincount(flat, weights=weights, length=n_k * n_mu)
    counts = jnp.bincount(flat, weights=counts_w, length=n_k * n_mu)
    mean = jnp.where(counts > 0.0, sums / counts, jnp.nan)
    k_cent = jnp.sqrt(k_edges[:-1] * k_edges[1:])
    return k_cent, mean.reshape(n_mu, n_k), counts.reshape(n_mu, n_k)


def kaiser_flux_from_density(box, b_f, beta_f):
    """Multiply each matter mode by b_F (1 + β_F μ²) and return the real field."""
    fk = box["fourier"] * b_f * (1.0 + beta_f * box["mu"] ** 2)
    fk = fk.at[0, 0, 0].set(0.0)
    return jnp.fft.ifftn(fk).real


def los_displacement(box, f, length):
    """Comoving line-of-sight shift v_∥/(aH), from the linear continuity equation."""
    k = box["k"]
    factor = jnp.where(k > 0.0, box["kz"] / k**2, 0.0)
    disp_k = 1j * jnp.asarray(f, dtype=jnp.float64) * factor * box["fourier"]
    n = k.shape[0]
    if n % 2 == 0:
        disp_k = disp_k.at[:, :, n // 2].set(0.0)
    return jnp.fft.ifftn(disp_k).real


def cic_shift_los(field, displacement, length):
    """Periodic cloud-in-cell shift along the last axis. displacement is comoving."""
    values = jnp.asarray(field, dtype=jnp.float64)
    n = values.shape[-1]
    dx = length / n
    shifts = jnp.asarray(displacement, dtype=jnp.float64) / dx
    base = jnp.arange(n, dtype=jnp.float64)
    destination = (base + shifts) % n
    left = jnp.floor(destination).astype(jnp.int32)
    fraction = destination - left
    right = (left + 1) % n

    flat = values.reshape(-1, n)
    left_f = left.reshape(-1, n)
    right_f = right.reshape(-1, n)
    fraction_f = fraction.reshape(-1, n)

    def shift_row(vals, left_idx, right_idx, frac):
        out = jnp.zeros(n, dtype=jnp.float64)
        out = out.at[left_idx].add(vals * (1.0 - frac))
        out = out.at[right_idx].add(vals * frac)
        return out

    remapped = jax.vmap(shift_row)(flat, left_f, right_f, fraction_f)
    return remapped.reshape(values.shape)


def fgpa_flux_box(box, length, gamma, target_mean_flux, f=None, thermal_sigma_mpch=0.0):
    """Lognormal density, τ = A Δ^β, optional linear RSD and thermal broadening."""
    density = lognormal_density(box["delta"])
    tau_unit = optical_depth_unit(density, gamma)
    if f is not None:
        shift = los_displacement(box, f, length)
        tau_unit = cic_shift_los(tau_unit, shift, length)
    n = tau_unit.shape[-1]
    sigma_pixels = jnp.asarray(thermal_sigma_mpch, dtype=jnp.float64) / (length / n)
    tau_unit = smooth_periodic(tau_unit, sigma_pixels)
    amplitude = calibrate_mean_flux(tau_unit, target_mean_flux)
    flux = transmitted_flux(tau_unit, amplitude)
    return {
        "density": density,
        "flux": flux,
        "contrast": flux_contrast(flux),
        "A": amplitude,
        "mean_flux": jnp.mean(flux),
        "var_gaussian": jnp.var(box["delta"]),
        "var_lognormal": jnp.var(density),
        "thermal_sigma_pixels": sigma_pixels,
    }
