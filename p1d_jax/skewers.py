"""JAX Gaussian 1D skewers and measured P_1D."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rfft_modes(length_mpch, n_pixels):
    """Non-negative Fourier modes in h/Mpc for a periodic Mpc/h skewer."""
    length = jnp.asarray(length_mpch, dtype=jnp.float64)
    dx = length / n_pixels
    return 2.0 * jnp.pi * jnp.fft.rfftfreq(n_pixels, d=dx)


def generate_gaussian_skewers(p1d_mpc_h, length_mpch, n_pixels, n_skewers, seed=12345):
    """Draw periodic Gaussian skewers with the requested discrete P_1D.

    ``p1d_mpc_h`` is sampled on ``rfft_modes``. The zero mode is set to zero.
    ``seed`` may be an int or a ``jax.Array`` PRNG key.
    """
    p1d = jnp.asarray(p1d_mpc_h, dtype=jnp.float64)
    n_modes = n_pixels // 2 + 1
    key = seed if isinstance(seed, jax.Array) else jax.random.PRNGKey(int(seed))

    variance_complex = n_pixels**2 * p1d / length_mpch
    sigma_component = jnp.sqrt(0.5 * variance_complex)

    key_re, key_im, key_nyq = jax.random.split(key, 3)
    re = jax.random.normal(key_re, (n_skewers, n_modes - 1), dtype=jnp.float64)
    im = jax.random.normal(key_im, (n_skewers, n_modes - 1), dtype=jnp.float64)
    interior = sigma_component[1:] * (re + 1j * im)

    zero = jnp.zeros((n_skewers, 1), dtype=jnp.complex128)
    fourier = jnp.concatenate([zero, interior], axis=-1)

    if n_pixels % 2 == 0:
        nyq = jnp.sqrt(variance_complex[-1]) * jax.random.normal(
            key_nyq, (n_skewers,), dtype=jnp.float64
        )
        fourier = fourier.at[:, -1].set(nyq.astype(jnp.complex128))

    return jnp.fft.irfft(fourier, n=n_pixels, axis=-1)


def smooth_periodic(field, sigma_pixels):
    """Gaussian-smooth the last axis with periodic boundaries. sigma <= 0 is a no-op."""
    values = jnp.asarray(field, dtype=jnp.float64)
    sigma = jnp.asarray(sigma_pixels, dtype=jnp.float64)

    def _smooth(v):
        n = v.shape[-1]
        freqs = jnp.fft.rfftfreq(n)
        kernel = jnp.exp(-2.0 * (jnp.pi * freqs * sigma) ** 2)
        modes = jnp.fft.rfft(v, axis=-1)
        return jnp.fft.irfft(modes * kernel, n=n, axis=-1)

    return jnp.where(sigma > 0.0, _smooth(values), values)


def measure_p1d(field, length_mpch):
    """Ensemble-mean P_1D of dimensionless periodic skewers, in Mpc/h."""
    values = jnp.asarray(field, dtype=jnp.float64)
    if values.ndim == 1:
        values = values[None, :]
    n_pixels = values.shape[-1]
    dx = length_mpch / n_pixels
    modes = jnp.fft.rfft(values, axis=-1)
    return (dx**2 / length_mpch) * jnp.mean(jnp.abs(modes) ** 2, axis=0)
