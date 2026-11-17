"""Eisenstein & Hu (1998) linear cold-dark-matter transfer function.

T(k) is normalised so that T(k -> 0) = 1. Wavenumbers passed to
``transfer`` are in Mpc^{-1} (no h).
"""

from __future__ import annotations

import numpy as np

C_KM_S = 299792.458
K_PIVOT_MPC = 0.05  # Planck pivot, Mpc^{-1}


def transfer(k_mpc, omega_m, omega_b, h, t_cmb=2.7255):
    """EH98 transfer function, including baryon acoustic oscillations."""
    k = np.asarray(k_mpc, dtype=float)
    scalar = k.ndim == 0
    k = np.atleast_1d(k)
    if np.any(k <= 0):
        raise ValueError("k must be positive")

    omh2 = omega_m * h**2
    obh2 = omega_b * h**2
    fb = omega_b / omega_m
    theta = t_cmb / 2.7

    z_eq = 2.50e4 * omh2 * theta ** (-4)
    k_eq = 7.46e-2 * omh2 * theta ** (-2)

    b1 = 0.313 * omh2 ** (-0.419) * (1.0 + 0.607 * omh2**0.674)
    b2 = 0.238 * omh2**0.223
    z_d = 1291.0 * omh2**0.251 / (1.0 + 0.659 * omh2**0.828) * (1.0 + b1 * obh2**b2)

    r_d = 31.5 * obh2 * theta ** (-4) * (z_d / 1.0e3) ** (-1.0)
    r_eq = 31.5 * obh2 * theta ** (-4) * (z_eq / 1.0e3) ** (-1.0)
    sound = (2.0 / (3.0 * k_eq)) * np.sqrt(6.0 / r_eq) * np.log(
        (np.sqrt(1.0 + r_d) + np.sqrt(r_d + r_eq)) / (1.0 + np.sqrt(r_eq))
    )

    k_silk = 1.6 * obh2**0.52 * omh2**0.73 * (1.0 + (10.4 * omh2) ** (-0.95))

    a1 = (46.9 * omh2) ** 0.670 * (1.0 + (32.1 * omh2) ** (-0.532))
    a2 = (12.0 * omh2) ** 0.424 * (1.0 + (45.0 * omh2) ** (-0.582))
    alpha_c = a1 ** (-fb) * a2 ** (-(fb**3))

    bb1 = 0.944 / (1.0 + (458.0 * omh2) ** (-0.708))
    bb2 = (0.395 * omh2) ** (-0.0266)
    beta_c = 1.0 / (1.0 + bb1 * ((1.0 - fb) ** bb2 - 1.0))

    y = (1.0 + z_eq) / (1.0 + z_d)
    gy = y * (
        -6.0 * np.sqrt(1.0 + y)
        + (2.0 + 3.0 * y) * np.log((np.sqrt(1.0 + y) + 1.0) / (np.sqrt(1.0 + y) - 1.0))
    )
    alpha_b = 2.07 * k_eq * sound * (1.0 + r_d) ** (-0.75) * gy
    beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * np.sqrt((17.2 * omh2) ** 2 + 1.0)
    beta_node = 8.41 * omh2**0.435

    q = k / (13.41 * k_eq)

    def t0(alpha, beta):
        coeff = 14.2 / alpha + 386.0 / (1.0 + 69.9 * q**1.08)
        arg = np.log(np.e + 1.8 * beta * q)
        return arg / (arg + coeff * q**2)

    f = 1.0 / (1.0 + (k * sound / 5.4) ** 4)
    t_c = f * t0(1.0, beta_c) + (1.0 - f) * t0(alpha_c, beta_c)

    s_tilde = sound / (1.0 + (beta_node / (k * sound)) ** 3) ** (1.0 / 3.0)
    baryon_env = t0(1.0, 1.0) / (1.0 + (k * sound / 5.2) ** 2) + (
        alpha_b / (1.0 + (beta_b / (k * sound)) ** 3) * np.exp(-((k / k_silk) ** 1.4))
    )
    t_b = baryon_env * np.sin(k * s_tilde) / (k * s_tilde)
    out = fb * t_b + (1.0 - fb) * t_c
    return out[0] if scalar else out


def hubble_kms(z, omega_m, h):
    """H(z) in km/s/Mpc for a flat universe."""
    return 100.0 * h * np.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))


def growth(z, omega_m, h, n_steps=2048):
    """Linear growth normalised so D(a) -> a deep in matter domination.

    Returns D(z) on the same shape as ``z``.
    """
    z = np.asarray(z, dtype=float)
    scalar = z.ndim == 0
    a_targets = 1.0 / (1.0 + np.atleast_1d(z))
    a_grid = np.logspace(-4, 0.0, n_steps)
    h_norm = np.sqrt(omega_m / a_grid**3 + (1.0 - omega_m))
    integrand = 1.0 / (a_grid * h_norm) ** 3
    # cumulative integral from a=0, starting the grid at 1e-4
    da = np.diff(a_grid)
    piece = 0.5 * (integrand[1:] + integrand[:-1]) * da
    cumulative = np.concatenate([[0.0], np.cumsum(piece)])
    # add the missing 0 -> a_grid[0] piece, where H ~ sqrt(Omega_m) a^{-3/2}
    # integrand ~ a^2 / Omega_m^{3/2}, integral = a^3 / (3 Omega_m^{3/2})
    cumulative = cumulative + (a_grid[0] ** 3) / (3.0 * omega_m**1.5)
    prefac = 2.5 * omega_m * h_norm * cumulative
    d_grid = np.interp(a_targets, a_grid, prefac)
    return d_grid[0] if scalar else d_grid


def linear_matter_power(k_hmpc, z, ns, As, omega_m, omega_b, h):
    """Linear P_CDM(k, z) in (Mpc/h)^3.

    ``k_hmpc`` is in h/Mpc. The amplitude uses the primordial A_s at
    k_pivot = 0.05 Mpc^{-1} and the EH98 transfer function.
    """
    k_h = np.asarray(k_hmpc, dtype=float)
    k_mpc = k_h * h
    t = transfer(k_mpc, omega_m, omega_b, h)
    h0 = 100.0 * h / C_KM_S  # Mpc^{-1}
    d = growth(z, omega_m, h)
    shape = (k_mpc / K_PIVOT_MPC) ** (ns - 1.0) * k_mpc * t**2
    # P in Mpc^3, then convert to (Mpc/h)^3
    p_mpc = (8.0 * np.pi**2 / 25.0) * As / (omega_m**2 * h0**4) * shape * d**2
    return p_mpc * h**3
