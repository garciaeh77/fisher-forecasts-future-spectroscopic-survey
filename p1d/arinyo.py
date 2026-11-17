"""Arinyo-i-Prats nonlinear flux power, with the published fiducial parameters.

The formula is their equation (3.3) times D_1 in equation (3.6), with q_2 fixed
to zero, which is the fit they use for most of the tables. The numbers are the
fiducial simulation at z=3 in their Tables 2 and 3. They are not refitted.
"""

from __future__ import annotations

import numpy as np

# Fiducial SPH run, z=3, q_2 = 0. Table 3 of Arinyo-i-Prats et al. (2015).
# k_v is recovered from the tabulated product k_v^{a_v}.
Q1_Z3 = 0.648
K_P_Z3 = 13.1  # h/Mpc
K_V_TO_A_V_Z3 = 1.007  # (h/Mpc)^{a_v}
A_V_Z3 = 0.627
B_V_Z3 = 1.66
# Table 2: beta and b_tau_delta. Mean transmission at z=3 is their section 3.2.
BETA_Z3 = 1.205
B_TAU_Z3 = 0.5546
MEAN_FLUX_Z3 = 0.6960


def jeans_wavenumber_scale():
    """k_v in h/Mpc, from the published product k_v^{a_v}."""
    return float(K_V_TO_A_V_Z3 ** (1.0 / A_V_Z3))


def flux_bias_from_tau_bias(b_tau=B_TAU_Z3, mean_flux=MEAN_FLUX_Z3):
    """b_Fδ = b_τδ ln(Fbar), their equation (2.4). Natural log."""
    if not 0.0 < mean_flux < 1.0:
        raise ValueError("mean_flux must lie strictly between zero and one")
    return float(b_tau) * float(np.log(mean_flux))


def dimensionless_variance(k_hmpc, p_lin):
    """Δ²(k) = k³ P_L(k) / (2 π²), their equation (3.5)."""
    k = np.asarray(k_hmpc, dtype=float)
    p = np.asarray(p_lin, dtype=float)
    return k**3 * p / (2.0 * np.pi**2)


def nonlinear_factor(k_hmpc, mu, p_lin, q2=0.0):
    """D_1(k, μ) from their equation (3.6), q_2 defaulting to the published zero."""
    k = np.asarray(k_hmpc, dtype=float)
    angle = np.asarray(mu, dtype=float)
    delta2 = dimensionless_variance(k, p_lin)
    k_v = jeans_wavenumber_scale()
    enhancement = (Q1_Z3 * delta2 + float(q2) * delta2**2)
    anisotropic = 1.0 - (k / k_v) ** A_V_Z3 * np.abs(angle) ** B_V_Z3
    pressure = (k / K_P_Z3) ** 2
    return np.exp(enhancement * anisotropic - pressure)


def flux_power(k_hmpc, mu, p_lin, q2=0.0):
    """P_F(k, μ) in (Mpc/h)³ using the published z=3 bias and D_1."""
    k = np.asarray(k_hmpc, dtype=float)
    angle = np.asarray(mu, dtype=float)
    p = np.asarray(p_lin, dtype=float)
    b_f = flux_bias_from_tau_bias()
    kaiser = b_f**2 * (1.0 + BETA_Z3 * angle**2) ** 2 * p
    return kaiser * nonlinear_factor(k, angle, p, q2=q2)
