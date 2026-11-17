"""JAX Kaiser flux power and line-of-sight projection."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from p1d_jax.eisenstein_hu import hubble_kms, linear_matter_power


def flux_power_3d(k_hmpc, mu, z, ns, As, omega_m, omega_b, h, b_f=-0.14, beta_f=1.4):
    p_m = linear_matter_power(k_hmpc, z, ns, As, omega_m, omega_b, h)
    return b_f**2 * (1.0 + beta_f * mu**2) ** 2 * p_m


def _integrate_one(kp, k, p_m, b_f, beta_f):
    """Match NumPy: trapezoid only on samples with k >= kp."""
    mu = kp / k
    p3 = b_f**2 * (1.0 + beta_f * mu**2) ** 2 * p_m
    integrand = k * p3
    dq = jnp.diff(k)
    pieces = 0.5 * (integrand[1:] + integrand[:-1]) * dq
    # tail[i] = int_{k[i]}^{kmax}; pad a trailing zero for kp above kmax
    tail = jnp.concatenate([jnp.cumsum(pieces[::-1])[::-1], jnp.array([0.0])])
    tail_pad = jnp.concatenate([tail, jnp.array([0.0])])
    idx = jnp.searchsorted(k, kp, side="left")
    return tail_pad[idx] / (2.0 * jnp.pi)


def flux_power_1d(
    kpar_hmpc,
    z,
    ns,
    As,
    omega_m,
    omega_b,
    h,
    b_f=-0.14,
    beta_f=1.4,
    k_max=30.0,
    n_k=800,
    matter_power=None,
):
    kpar = jnp.atleast_1d(jnp.asarray(kpar_hmpc, dtype=jnp.float64))
    log_k = jnp.linspace(jnp.log(jnp.min(kpar)), jnp.log(k_max), n_k)
    k = jnp.exp(log_k)
    if matter_power is None:
        p_m = linear_matter_power(k, z, ns, As, omega_m, omega_b, h)
    else:
        p_m = jnp.asarray(matter_power(k), dtype=jnp.float64)
    return jax.vmap(lambda kp: _integrate_one(kp, k, p_m, b_f, beta_f))(kpar)


def velocity_per_mpch(z, omega_m, h):
    return hubble_kms(z, omega_m, h) / (1.0 + z) / h


def k_skm_to_hmpc(k_skm, z, omega_m, h):
    return jnp.asarray(k_skm, dtype=jnp.float64) * velocity_per_mpch(z, omega_m, h)


def k_hmpc_to_skm(k_hmpc, z, omega_m, h):
    return jnp.asarray(k_hmpc, dtype=jnp.float64) / velocity_per_mpch(z, omega_m, h)


def p1d_h_to_kms(p_mpch, z, omega_m, h):
    return jnp.asarray(p_mpch, dtype=jnp.float64) * velocity_per_mpch(z, omega_m, h)


def p1d_skm(k_skm, z, ns, As, omega_m, omega_b, h, b_f=-0.14, beta_f=1.4, matter_power=None):
    k_h = k_skm_to_hmpc(k_skm, z, omega_m, h)
    p_h = flux_power_1d(
        k_h, z, ns, As, omega_m, omega_b, h, b_f=b_f, beta_f=beta_f, matter_power=matter_power
    )
    return p1d_h_to_kms(p_h, z, omega_m, h)
