"""JAX Jeans cutoff on a tabulated linear matter spectrum."""

from __future__ import annotations

import jax.numpy as jnp

from p1d_jax.matter_table import TabulatedMatterPower


def jeans_suppression(k_hmpc, x_j_mpch):
    k = jnp.asarray(k_hmpc, dtype=jnp.float64)
    x_j = jnp.asarray(x_j_mpch, dtype=jnp.float64)
    return jnp.where(x_j == 0.0, jnp.ones_like(k), jnp.exp(-2.0 * x_j**2 * k**2))


def with_jeans(table: TabulatedMatterPower, x_j_mpch):
    factor = jeans_suppression(table.k_hmpc, x_j_mpch)
    return TabulatedMatterPower(
        table.k_hmpc,
        table.pk_mpc3h3 * factor,
        As_fid=table.As_fid,
        ns_fid=table.ns_fid,
        kpivot_hmpc=table.kpivot_hmpc,
    )
