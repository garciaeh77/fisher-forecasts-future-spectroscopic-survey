"""Periodic Gaussian box and a binned measurement of P(k, μ).

The Fourier convention matches the rest of the package: the variance of the
field is the sum over modes of P(k)/L³. Part I functions are not modified.
"""

from __future__ import annotations

import numpy as np

from p1d.eisenstein_hu import growth
from p1d.fgpa_field import (
    calibrate_mean_flux,
    flux_contrast,
    lognormal_density,
    optical_depth_unit,
    transmitted_flux,
)
from p1d.skewers import smooth_periodic


def growth_rate(z, omega_m, h, dz=1.0e-2):
    """f = d ln D / d ln a from the growth function used for P_CDM."""
    z_hi = float(z) + dz
    z_lo = max(float(z) - dz, 0.0)
    ln_d_hi = np.log(float(growth(z_hi, omega_m, h)))
    ln_d_lo = np.log(float(growth(z_lo, omega_m, h)))
    ln_a_hi = np.log(1.0 / (1.0 + z_hi))
    ln_a_lo = np.log(1.0 / (1.0 + z_lo))
    return float((ln_d_hi - ln_d_lo) / (ln_a_hi - ln_a_lo))


def _mode_grid(n, length):
    k1d = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    mu = np.zeros_like(k)
    nonzero = k > 0.0
    mu[nonzero] = np.abs(kz[nonzero]) / k[nonzero]
    return k, mu, kz


def _independent_mode_mask(n):
    index = np.arange(n)
    linear = np.arange(n**3).reshape(n, n, n)
    neg = (-index) % n
    linear_neg = linear[np.ix_(neg, neg, neg)]
    independent = linear < linear_neg
    self_conjugate = linear == linear_neg
    return independent, self_conjugate, neg


def gaussian_box(power, length, n, seed, exact_amplitude=False):
    """Real Gaussian field whose modes have variance P(k)/L³.

    ``power(k)`` is P(k) in (Mpc/h)³ for k in h/Mpc. The zero mode is zero.
    ``exact_amplitude`` replaces the Rayleigh draw by the rms, which makes a
    one-box measurement reproduce P(k) mode by mode.
    """
    if n < 4 or length <= 0:
        raise ValueError("n must be at least 4 and length must be positive")
    k, mu, kz = _mode_grid(n, length)
    pk = np.zeros((n, n, n), dtype=float)
    nonzero = k > 0.0
    pk[nonzero] = np.asarray(power(k[nonzero]), dtype=float)
    if np.any(pk < 0.0):
        raise ValueError("power must be non-negative")
    # <|fftn(delta)|^2> = n^6 P / L^3. A complex mode splits that equally.
    sigma = np.sqrt(0.5 * n**6 * pk / length**3)
    independent, self_conjugate, neg = _independent_mode_mask(n)
    rng = np.random.default_rng(seed)
    fk = np.zeros((n, n, n), dtype=np.complex128)
    n_ind = int(np.count_nonzero(independent))
    if exact_amplitude:
        phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=n_ind))
        fk[independent] = phase * sigma[independent] * np.sqrt(2.0)
        signs = rng.choice(np.array([-1.0, 1.0]), size=int(np.count_nonzero(self_conjugate)))
        fk[self_conjugate] = signs * sigma[self_conjugate] * np.sqrt(2.0)
    else:
        fk[independent] = sigma[independent] * (
            rng.standard_normal(n_ind) + 1j * rng.standard_normal(n_ind)
        )
        fk[self_conjugate] = (
            sigma[self_conjugate] * np.sqrt(2.0) * rng.standard_normal(int(np.count_nonzero(self_conjugate)))
        )
    partner = np.conj(fk[np.ix_(neg, neg, neg)])
    linear = np.arange(n**3).reshape(n, n, n)
    linear_neg = linear[np.ix_(neg, neg, neg)]
    fk = fk + np.where(linear > linear_neg, partner, 0.0)
    delta = np.fft.ifftn(fk).real
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
    fk = np.fft.fftn(delta)
    return (length**3 / n**6) * np.abs(fk) ** 2


def bin_power(k, mu, power, k_edges, mu_edges):
    """Mean power in rectangular (k, μ) bins. Empty bins are nan."""
    k = np.asarray(k, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    power = np.asarray(power, dtype=float).ravel()
    use = (k > 0.0) & np.isfinite(power)
    k_index = np.digitize(k[use], k_edges) - 1
    mu_index = np.digitize(mu[use], mu_edges) - 1
    n_k = len(k_edges) - 1
    n_mu = len(mu_edges) - 1
    inside = (k_index >= 0) & (k_index < n_k) & (mu_index >= 0) & (mu_index < n_mu)
    flat = mu_index[inside] * n_k + k_index[inside]
    sums = np.bincount(flat, weights=power[use][inside], minlength=n_k * n_mu)
    counts = np.bincount(flat, minlength=n_k * n_mu)
    mean = np.full(n_k * n_mu, np.nan)
    filled = counts > 0
    mean[filled] = sums[filled] / counts[filled]
    k_cent = np.sqrt(k_edges[:-1] * k_edges[1:])
    return k_cent, mean.reshape(n_mu, n_k), counts.reshape(n_mu, n_k)


def kaiser_flux_from_density(box, b_f, beta_f):
    """Multiply each matter mode by b_F (1 + β_F μ²) and return the real field."""
    fk = box["fourier"] * b_f * (1.0 + beta_f * box["mu"] ** 2)
    fk[0, 0, 0] = 0.0
    return np.fft.ifftn(fk).real


def los_displacement(box, f, length):
    """Comoving line-of-sight shift v_∥/(aH), from the linear continuity equation.

    With numpy's FFT sign, ψ(k) = i f (k_∥/k²) δ(k) gives δ_s = δ (1 + f μ²)
    at first order.
    """
    k = box["k"]
    factor = np.zeros_like(k)
    nonzero = k > 0.0
    factor[nonzero] = box["kz"][nonzero] / k[nonzero] ** 2
    disp_k = 1j * float(f) * factor * box["fourier"]
    n = k.shape[0]
    if n % 2 == 0:
        # k and -k are the same bin at the Nyquist frequency, so an odd
        # multiplier in k_parallel cannot be represented on a real grid.
        disp_k[:, :, n // 2] = 0.0
    return np.fft.ifftn(disp_k).real


def cic_shift_los(field, displacement, length):
    """Periodic cloud-in-cell shift along the last axis. displacement is comoving."""
    values = np.asarray(field, dtype=float)
    n = values.shape[-1]
    dx = length / n
    shifts = np.asarray(displacement, dtype=float) / dx
    base = np.arange(n)
    destination = (base + shifts) % n
    left = np.floor(destination).astype(np.int64)
    fraction = destination - left
    right = (left + 1) % n
    flat = values.reshape(-1, n)
    left_f = left.reshape(-1, n)
    right_f = right.reshape(-1, n)
    fraction_f = fraction.reshape(-1, n)
    rows = np.arange(flat.shape[0])[:, None]
    output = np.zeros_like(flat)
    np.add.at(output, (rows, left_f), flat * (1.0 - fraction_f))
    np.add.at(output, (rows, right_f), flat * fraction_f)
    return output.reshape(values.shape)


def fgpa_flux_box(box, length, gamma, target_mean_flux, f=None, thermal_sigma_mpch=0.0):
    """Lognormal density, τ = A Δ^β, optional linear RSD and thermal broadening.

    Order: optical depth → peculiar-velocity CIC shift along the line of sight
    → Gaussian thermal smooth in that same coordinate → solve A so
    ⟨exp(-A τ)⟩ equals the target. ``thermal_sigma_mpch`` is the Gaussian
    width in comoving Mpc/h (convert from km/s with H(z)/((1+z)h)).
    """
    density = lognormal_density(box["delta"])
    tau_unit = optical_depth_unit(density, gamma)
    if f is not None:
        shift = los_displacement(box, f, length)
        tau_unit = cic_shift_los(tau_unit, shift, length)
    n = tau_unit.shape[-1]
    sigma_pixels = float(thermal_sigma_mpch) / (length / n) if thermal_sigma_mpch else 0.0
    tau_unit = smooth_periodic(tau_unit, sigma_pixels)
    amplitude = calibrate_mean_flux(tau_unit, target_mean_flux)
    flux = transmitted_flux(tau_unit, amplitude)
    return {
        "density": density,
        "flux": flux,
        "contrast": flux_contrast(flux),
        "A": amplitude,
        "mean_flux": float(np.mean(flux)),
        "var_gaussian": float(np.var(box["delta"])),
        "var_lognormal": float(np.var(density)),
        "thermal_sigma_pixels": sigma_pixels,
    }
