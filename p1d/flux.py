"""Linearised FGPA flux power and its line-of-sight projection."""

from __future__ import annotations

import numpy as np

from p1d.eisenstein_hu import hubble_kms, linear_matter_power


def flux_power_3d(k_hmpc, mu, z, ns, As, omega_m, omega_b, h, b_f=-0.14, beta_f=1.4):
    """Kaiser flux power, b_F^2 (1 + beta_F mu^2)^2 P_CDM.

    Units match ``linear_matter_power``: (Mpc/h)^3.
    """
    p_m = linear_matter_power(k_hmpc, z, ns, As, omega_m, omega_b, h)
    return b_f**2 * (1.0 + beta_f * mu**2) ** 2 * p_m


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
    """Project P_F(k, mu) to P_1D(k_parallel).

    ``kpar_hmpc`` is in h/Mpc. The return value is in Mpc/h.
    ``matter_power``, if given, is called as ``matter_power(k)`` with ``k``
    in h/Mpc and must return P_CDM in (Mpc/h)^3. The default is Eisenstein-Hu.
    """
    kpar = np.atleast_1d(np.asarray(kpar_hmpc, dtype=float))
    log_k = np.linspace(np.log(kpar.min()), np.log(k_max), n_k)
    k = np.exp(log_k)
    if matter_power is None:
        p_m = linear_matter_power(k, z, ns, As, omega_m, omega_b, h)
    else:
        p_m = np.asarray(matter_power(k), dtype=float)
    out = np.empty(kpar.size)
    for i, kp in enumerate(kpar):
        use = k >= kp
        kk = k[use]
        mu = kp / kk
        p3 = b_f**2 * (1.0 + beta_f * mu**2) ** 2 * p_m[use]
        # int_{k_par}^{k_max} k dk / (2 pi) P_3D
        out[i] = np.trapezoid(kk * p3, kk) / (2.0 * np.pi)
    return out


def velocity_per_mpch(z, omega_m, h):
    """km/s per comoving Mpc/h, equal to H(z) / ((1+z) h)."""
    return hubble_kms(z, omega_m, h) / (1.0 + z) / h


def k_skm_to_hmpc(k_skm, z, omega_m, h):
    """Observed forest wavenumber (s/km) to comoving h/Mpc."""
    return np.asarray(k_skm, dtype=float) * velocity_per_mpch(z, omega_m, h)


def k_hmpc_to_skm(k_hmpc, z, omega_m, h):
    """Comoving h/Mpc to forest wavenumber in s/km."""
    return np.asarray(k_hmpc, dtype=float) / velocity_per_mpch(z, omega_m, h)


def p1d_h_to_kms(p_mpch, z, omega_m, h):
    """P_1D in Mpc/h to P_1D in km/s."""
    return np.asarray(p_mpch, dtype=float) * velocity_per_mpch(z, omega_m, h)


def p1d_skm(k_skm, z, ns, As, omega_m, omega_b, h, b_f=-0.14, beta_f=1.4, matter_power=None):
    """P_1D in km/s on a forest k grid in s/km."""
    k_h = k_skm_to_hmpc(k_skm, z, omega_m, h)
    p_h = flux_power_1d(
        k_h, z, ns, As, omega_m, omega_b, h, b_f=b_f, beta_f=beta_f, matter_power=matter_power
    )
    return p1d_h_to_kms(p_h, z, omega_m, h)
