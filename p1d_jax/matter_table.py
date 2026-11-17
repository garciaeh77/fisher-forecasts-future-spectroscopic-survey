"""JAX tabulated linear P_CDM with A_s / n_s / growth rescaling.

Log-log interpolation uses ``jnp.interp`` (linear in ln k, ln P). For the dense
Eisenstein-Hu tables used here this matches the NumPy PCHIP path closely.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from p1d_jax.eisenstein_hu import growth, linear_matter_power

K_PIVOT_MPC = 0.05


@dataclass(frozen=True)
class TabulatedMatterPower:
    k_hmpc: jnp.ndarray
    pk_mpc3h3: jnp.ndarray
    As_fid: float = 2.1e-9
    ns_fid: float = 0.965
    kpivot_hmpc: float = K_PIVOT_MPC / 0.67

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_hmpc", jnp.asarray(self.k_hmpc, dtype=jnp.float64))
        object.__setattr__(self, "pk_mpc3h3", jnp.asarray(self.pk_mpc3h3, dtype=jnp.float64))

    @property
    def kmin(self) -> float:
        return float(self.k_hmpc[0])

    @property
    def kmax(self) -> float:
        return float(self.k_hmpc[-1])

    def __call__(self, k_hmpc, As=None, ns=None, growth_ratio=1.0):
        k = jnp.asarray(k_hmpc, dtype=jnp.float64)
        As_eval = self.As_fid if As is None else As
        ns_eval = self.ns_fid if ns is None else ns
        pk_fid = jnp.exp(
            jnp.interp(jnp.log(k), jnp.log(self.k_hmpc), jnp.log(self.pk_mpc3h3))
        )
        amplitude = As_eval / self.As_fid
        tilt = (k / self.kpivot_hmpc) ** (ns_eval - self.ns_fid)
        return pk_fid * amplitude * tilt * growth_ratio**2

    @classmethod
    def from_eisenstein_hu(
        cls, z, ns, As, omega_m, omega_b, h, kmin=1.0e-3, kmax=100.0, n_k=4000
    ):
        k = jnp.geomspace(kmin, kmax, n_k)
        pk = linear_matter_power(k, z, ns, As, omega_m, omega_b, h)
        return cls(k, pk, As_fid=float(As), ns_fid=float(ns), kpivot_hmpc=K_PIVOT_MPC / h)


def growth_ratio(z, z_fid, omega_m, h):
    d = growth(z, omega_m, h)
    d_fid = growth(z_fid, omega_m, h)
    d = d if jnp.ndim(d) == 0 else d[0]
    d_fid = d_fid if jnp.ndim(d_fid) == 0 else d_fid[0]
    return d / d_fid
