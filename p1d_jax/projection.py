"""JAX isotropic projection of P_3D(k) onto P_1D(k_parallel)."""

from __future__ import annotations

import jax.numpy as jnp


def project_p1d(k_parallel_hmpc, q_hmpc, pk3d_mpc3h3):
    """P_1D(k_parallel) = (2 pi)^-1 int_k^{qmax} q P_3D(q) dq.

    Matches the NumPy path: reverse cumulative trapezoid, then interpolate
    the antiderivative onto ``k_parallel``.
    """
    kp = jnp.atleast_1d(jnp.asarray(k_parallel_hmpc, dtype=jnp.float64))
    q = jnp.asarray(q_hmpc, dtype=jnp.float64)
    pk = jnp.asarray(pk3d_mpc3h3, dtype=jnp.float64)
    integrand = q * pk
    dq = jnp.diff(q)
    pieces = 0.5 * (integrand[1:] + integrand[:-1]) * dq
    # tail[i] = int_{q[i]}^{qmax}
    tail = jnp.concatenate([jnp.cumsum(pieces[::-1])[::-1], jnp.array([0.0])])
    return jnp.interp(kp, q, tail) / (2.0 * jnp.pi)
