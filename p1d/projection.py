"""Isotropic projection of P_3D(k) onto P_1D(k_parallel)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def project_p1d(k_parallel_hmpc, q_hmpc, pk3d_mpc3h3):
    """P_1D(k_parallel) = (2 pi)^-1 int_{k}^{qmax} q P_3D(q) dq.

    The upper limit is the last entry of ``q_hmpc``. Every ``k_parallel`` must
    lie inside that grid. The integral does not extrapolate.
    """
    kp = np.atleast_1d(np.asarray(k_parallel_hmpc, dtype=float))
    q = np.asarray(q_hmpc, dtype=float)
    pk = np.asarray(pk3d_mpc3h3, dtype=float)
    if q.ndim != 1 or pk.shape != q.shape or np.any(np.diff(q) <= 0):
        raise ValueError("q and pk3d must be same-length 1D arrays with increasing q")
    if np.any(kp <= 0):
        raise ValueError("k_parallel must be positive")
    if np.any(kp < q[0]) or np.any(kp > q[-1]):
        raise ValueError("all k_parallel values must lie within the q integration grid")
    integrand = q * pk
    reverse_integral = cumulative_trapezoid(integrand[::-1], q[::-1], initial=0.0)
    tail_integral = -reverse_integral[::-1]
    return np.interp(kp, q, tail_integral) / (2.0 * np.pi)
