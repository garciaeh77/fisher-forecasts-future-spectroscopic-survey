"""Projection of isotropic 3D matter power to a line of sight."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def project_p1d(
    k_parallel_hmpc: np.ndarray,
    q_hmpc: np.ndarray,
    pk3d_mpc3h3: np.ndarray,
) -> np.ndarray:
    r"""Calculate P_1D(k_parallel) = (2 pi)^-1 int_k^qmax q P_3D(q)dq.

    The finite upper limit is the largest value in ``q_hmpc``.  Callers must
    test convergence against a larger q_max because the result is sensitive
    to unresolved small-scale power.
    """
    kp = np.atleast_1d(np.asarray(k_parallel_hmpc, dtype=float))
    q = np.asarray(q_hmpc, dtype=float)
    pk = np.asarray(pk3d_mpc3h3, dtype=float)
    if q.ndim != 1 or pk.shape != q.shape or np.any(np.diff(q) <= 0):
        raise ValueError("q and pk3d must be same-length 1D arrays with increasing q")
    if np.any(kp < q[0]) or np.any(kp > q[-1]):
        raise ValueError("all k_parallel values must lie within the q integration grid")
    integrand = q * pk
    reverse_integral = cumulative_trapezoid(
        integrand[::-1], q[::-1], initial=0.0
    )
    tail_integral = -reverse_integral[::-1]
    return np.interp(kp, q, tail_integral) / (2.0 * np.pi)

