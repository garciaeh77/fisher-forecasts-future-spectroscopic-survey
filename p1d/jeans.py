"""Jeans cutoff on a tabulated linear matter spectrum.

The existing Eisenstein-Hu, Kaiser, and lognormal-skewer paths are unchanged.
Multiply a table by this factor before passing it in. A cutoff of zero leaves
the table untouched.

The factor is the one used by Arya et al., exp(-2 x_J^2 k^2), with k in
h/Mpc and x_J in Mpc/h. 100 kpc/h is x_J = 0.1.
"""

from __future__ import annotations

import numpy as np

from p1d.matter_table import TabulatedMatterPower


def jeans_suppression(k_hmpc, x_j_mpch):
    """Multiplicative Jeans factor. Equals 1 everywhere if x_J is 0."""
    k = np.asarray(k_hmpc, dtype=float)
    x_j = float(x_j_mpch)
    if x_j < 0:
        raise ValueError("x_j_mpch must be non-negative")
    if x_j == 0.0:
        return np.ones_like(k, dtype=float)
    return np.exp(-2.0 * x_j**2 * k**2)


def with_jeans(table: TabulatedMatterPower, x_j_mpch):
    """Return a new table. The input table is not modified."""
    factor = jeans_suppression(table.k_hmpc, x_j_mpch)
    return TabulatedMatterPower(
        table.k_hmpc,
        np.asarray(table.pk_mpc3h3, dtype=float) * factor,
        As_fid=table.As_fid,
        ns_fid=table.ns_fid,
        kpivot_hmpc=table.kpivot_hmpc,
    )
