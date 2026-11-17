"""JAX alpha-beta-gamma-delta shape on Eisenstein-Hu."""

from __future__ import annotations

import jax.numpy as jnp

from p1d_jax.eisenstein_hu import linear_matter_power
from p1d_jax.flux import p1d_skm

K_PIVOT_HMPC = 1.0


def shape_factor(k_hmpc, k_pivot=K_PIVOT_HMPC, alpha=0.0, beta=0.0, gamma=0.0, delta=0.0):
    k = jnp.asarray(k_hmpc, dtype=jnp.float64)
    ell = jnp.log(k / k_pivot)
    return jnp.exp(alpha * ell + beta * ell**2 + gamma * ell**3 + delta * ell**4)


def linear_matter_power_abgd(
    k_hmpc, z, ns, As, omega_m, omega_b, h, alpha=0.0, beta=0.0, gamma=0.0, delta=0.0, k_pivot=K_PIVOT_HMPC
):
    base = linear_matter_power(k_hmpc, z, ns, As, omega_m, omega_b, h)
    return base * shape_factor(k_hmpc, k_pivot=k_pivot, alpha=alpha, beta=beta, gamma=gamma, delta=delta)


def p1d_skm_abgd(
    k_skm, z, ns, As, omega_m, omega_b, h, alpha=0.0, beta=0.0, gamma=0.0, delta=0.0, b_f=-0.14, beta_f=1.4
):
    def matter_power(k):
        return linear_matter_power_abgd(
            k, z, ns, As, omega_m, omega_b, h, alpha=alpha, beta=beta, gamma=gamma, delta=delta
        )

    return p1d_skm(k_skm, z, ns, As, omega_m, omega_b, h, b_f=b_f, beta_f=beta_f, matter_power=matter_power)
