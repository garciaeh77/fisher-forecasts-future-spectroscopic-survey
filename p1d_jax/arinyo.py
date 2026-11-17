"""Arinyo-i-Prats nonlinear flux power (JAX), published fiducial parameters.

Equation (3.3) times D_1 in (3.6), with q_2 = 0. Numbers are the fiducial
simulation at z=3 in their Tables 2 and 3 — not refitted.
"""

from __future__ import annotations

import jax.numpy as jnp

Q1_Z3 = 0.648
K_P_Z3 = 13.1  # h/Mpc
K_V_TO_A_V_Z3 = 1.007  # (h/Mpc)^{a_v}
A_V_Z3 = 0.627
B_V_Z3 = 1.66
BETA_Z3 = 1.205
B_TAU_Z3 = 0.5546
MEAN_FLUX_Z3 = 0.6960


def jeans_wavenumber_scale():
    """k_v in h/Mpc, from the published product k_v^{a_v}."""
    return K_V_TO_A_V_Z3 ** (1.0 / A_V_Z3)


def flux_bias_from_tau_bias(b_tau=B_TAU_Z3, mean_flux=MEAN_FLUX_Z3):
    """b_Fδ = b_τδ ln(Fbar), their equation (2.4). Natural log."""
    return jnp.asarray(b_tau, dtype=jnp.float64) * jnp.log(jnp.asarray(mean_flux, dtype=jnp.float64))


def dimensionless_variance(k_hmpc, p_lin):
    """Δ²(k) = k³ P_L(k) / (2 π²), their equation (3.5)."""
    k = jnp.asarray(k_hmpc, dtype=jnp.float64)
    p = jnp.asarray(p_lin, dtype=jnp.float64)
    return k**3 * p / (2.0 * jnp.pi**2)


def nonlinear_factor(k_hmpc, mu, p_lin, q2=0.0):
    """D_1(k, μ) from their equation (3.6), q_2 defaulting to the published zero."""
    k = jnp.asarray(k_hmpc, dtype=jnp.float64)
    angle = jnp.asarray(mu, dtype=jnp.float64)
    delta2 = dimensionless_variance(k, p_lin)
    k_v = jeans_wavenumber_scale()
    enhancement = Q1_Z3 * delta2 + jnp.asarray(q2, dtype=jnp.float64) * delta2**2
    anisotropic = 1.0 - (k / k_v) ** A_V_Z3 * jnp.abs(angle) ** B_V_Z3
    pressure = (k / K_P_Z3) ** 2
    return jnp.exp(enhancement * anisotropic - pressure)


def flux_power(k_hmpc, mu, p_lin, q2=0.0):
    """P_F(k, μ) in (Mpc/h)³ using the published z=3 bias and D_1."""
    k = jnp.asarray(k_hmpc, dtype=jnp.float64)
    angle = jnp.asarray(mu, dtype=jnp.float64)
    p = jnp.asarray(p_lin, dtype=jnp.float64)
    b_f = flux_bias_from_tau_bias()
    kaiser = b_f**2 * (1.0 + BETA_Z3 * angle**2) ** 2 * p
    return kaiser * nonlinear_factor(k, angle, p, q2=q2)
