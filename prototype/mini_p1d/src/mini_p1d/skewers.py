"""Gaussian 1D skewer generation and power measurement."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def rfft_modes(length_mpch: float, n_pixels: int) -> np.ndarray:
    """Non-negative Fourier modes in h/Mpc for a periodic Mpc/h skewer."""
    dx = length_mpch / n_pixels
    return 2.0 * np.pi * np.fft.rfftfreq(n_pixels, d=dx)


def generate_gaussian_skewers(
    p1d_mpc_h: np.ndarray,
    *,
    length_mpch: float,
    n_pixels: int,
    n_skewers: int,
    seed: int = 12345,
) -> np.ndarray:
    """Draw periodic Gaussian skewers with the requested discrete P1D.

    ``p1d_mpc_h`` must be sampled on ``rfft_modes(length_mpch, n_pixels)``.
    The zero mode is set to zero.  NumPy's FFT normalization is accounted for
    such that measuring the ensemble with :func:`measure_p1d` recovers P1D.
    """
    p1d = np.asarray(p1d_mpc_h, dtype=float)
    n_modes = n_pixels // 2 + 1
    if p1d.shape != (n_modes,) or np.any(p1d < 0):
        raise ValueError(f"p1d must be non-negative with shape ({n_modes},)")
    if length_mpch <= 0 or n_pixels < 4 or n_skewers < 1:
        raise ValueError("invalid skewer geometry")

    rng = np.random.default_rng(seed)
    fourier = np.zeros((n_skewers, n_modes), dtype=np.complex128)
    variance_complex = n_pixels**2 * p1d / length_mpch
    sigma_component = np.sqrt(0.5 * variance_complex)
    fourier[:, 1:] = sigma_component[1:] * (
        rng.normal(size=(n_skewers, n_modes - 1))
        + 1j * rng.normal(size=(n_skewers, n_modes - 1))
    )
    fourier[:, 0] = 0.0
    if n_pixels % 2 == 0:
        fourier[:, -1] = np.sqrt(variance_complex[-1]) * rng.normal(
            size=n_skewers
        )
    return np.fft.irfft(fourier, n=n_pixels, axis=-1)


def smooth_periodic(field: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """Gaussian-smooth the final axis using periodic boundaries."""
    if sigma_pixels <= 0:
        return np.asarray(field, dtype=float).copy()
    return gaussian_filter1d(field, sigma=sigma_pixels, axis=-1, mode="wrap")


def measure_p1d(field: np.ndarray, length_mpch: float) -> np.ndarray:
    """Measure ensemble-mean P1D of dimensionless periodic skewers."""
    values = np.asarray(field, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    n_pixels = values.shape[-1]
    dx = length_mpch / n_pixels
    modes = np.fft.rfft(values, axis=-1)
    return (dx**2 / length_mpch) * np.mean(np.abs(modes) ** 2, axis=0)

