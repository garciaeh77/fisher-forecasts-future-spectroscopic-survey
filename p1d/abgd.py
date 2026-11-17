"""Smooth alpha-beta-gamma-delta shape on top of the Eisenstein-Hu spectrum.

With L = ln(k / k_pivot),

    P(k) = P_EH(k) * exp(alpha L + beta L^2 + gamma L^3 + delta L^4).

At k = k_pivot the exponential is 1, so the amplitude matches Eisenstein-Hu.
alpha, beta, gamma, and delta are dimensionless shape coefficients. They are
not the IGM temperature-density slope.
"""

from __future__ import annotations

import numpy as np

from p1d.eisenstein_hu import linear_matter_power
from p1d.flux import p1d_skm

# Comoving pivot near the forest scale, in h/Mpc.
K_PIVOT_HMPC = 1.0


def shape_factor(k_hmpc, k_pivot=K_PIVOT_HMPC, alpha=0.0, beta=0.0, gamma=0.0, delta=0.0):
    """Multiplicative bend. Equals 1 at ``k_pivot`` and when every coefficient is 0."""
    k = np.asarray(k_hmpc, dtype=float)
    if k_pivot <= 0 or np.any(k <= 0):
        raise ValueError("k and k_pivot must be positive")
    ell = np.log(k / k_pivot)
    return np.exp(alpha * ell + beta * ell**2 + gamma * ell**3 + delta * ell**4)


def linear_matter_power_abgd(
    k_hmpc,
    z,
    ns,
    As,
    omega_m,
    omega_b,
    h,
    alpha=0.0,
    beta=0.0,
    gamma=0.0,
    delta=0.0,
    k_pivot=K_PIVOT_HMPC,
):
    """Linear P_CDM in (Mpc/h)^3 with an alpha-beta-gamma-delta bend."""
    p_eh = linear_matter_power(k_hmpc, z, ns, As, omega_m, omega_b, h)
    return p_eh * shape_factor(
        k_hmpc, k_pivot=k_pivot, alpha=alpha, beta=beta, gamma=gamma, delta=delta
    )


def p1d_skm_abgd(
    k_skm,
    z,
    ns,
    As,
    omega_m,
    omega_b,
    h,
    alpha=0.0,
    beta=0.0,
    gamma=0.0,
    delta=0.0,
    k_pivot=K_PIVOT_HMPC,
    b_f=-0.14,
    beta_f=1.4,
):
    """P_1D in km/s using the bent cold-dark-matter spectrum."""

    def matter_power(k):
        return linear_matter_power_abgd(
            k,
            z,
            ns,
            As,
            omega_m,
            omega_b,
            h,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            k_pivot=k_pivot,
        )

    return p1d_skm(
        k_skm,
        z,
        ns,
        As,
        omega_m,
        omega_b,
        h,
        b_f=b_f,
        beta_f=beta_f,
        matter_power=matter_power,
    )
