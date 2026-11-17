"""Tabulated linear P_CDM, with A_s, n_s, and growth rescaling.

Wavenumbers are h/Mpc. Power is (Mpc/h)^3. The default table is Eisenstein-Hu
evaluated on a fixed k grid. A two-column file can replace that table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from p1d.eisenstein_hu import growth, linear_matter_power

K_PIVOT_MPC = 0.05


@dataclass(frozen=True)
class TabulatedMatterPower:
    """Log-log interpolation of a fiducial linear spectrum."""

    k_hmpc: np.ndarray
    pk_mpc3h3: np.ndarray
    As_fid: float = 2.1e-9
    ns_fid: float = 0.965
    kpivot_hmpc: float = K_PIVOT_MPC / 0.67

    def __post_init__(self) -> None:
        k = np.asarray(self.k_hmpc, dtype=float)
        pk = np.asarray(self.pk_mpc3h3, dtype=float)
        if k.ndim != 1 or pk.ndim != 1 or k.size != pk.size:
            raise ValueError("k_hmpc and pk_mpc3h3 must be same-length 1D arrays")
        if k.size < 4 or np.any(k <= 0) or np.any(pk <= 0):
            raise ValueError("the power table needs at least four positive k and P values")
        if np.any(np.diff(k) <= 0):
            raise ValueError("k_hmpc must be strictly increasing")
        if self.As_fid <= 0 or self.kpivot_hmpc <= 0:
            raise ValueError("As_fid and kpivot_hmpc must be positive")
        object.__setattr__(self, "k_hmpc", k)
        object.__setattr__(self, "pk_mpc3h3", pk)
        object.__setattr__(
            self,
            "_log_interpolator",
            PchipInterpolator(np.log(k), np.log(pk), extrapolate=False),
        )

    @property
    def kmin(self) -> float:
        return float(self.k_hmpc[0])

    @property
    def kmax(self) -> float:
        return float(self.k_hmpc[-1])

    def __call__(
        self,
        k_hmpc,
        As=None,
        ns=None,
        growth_ratio=1.0,
    ):
        """P_m(k) rescaled from the fiducial table.

        ``growth_ratio`` is D(z)/D(z_fid). The table is assumed to already
        contain the fiducial primordial tilt, so only the difference
        ns - ns_fid is applied.
        """
        k = np.asarray(k_hmpc, dtype=float)
        if np.any(k < self.kmin) or np.any(k > self.kmax):
            raise ValueError(
                f"requested k lies outside [{self.kmin:g}, {self.kmax:g}] h/Mpc"
            )
        As_eval = self.As_fid if As is None else float(As)
        ns_eval = self.ns_fid if ns is None else float(ns)
        if As_eval <= 0 or growth_ratio < 0:
            raise ValueError("As must be positive and growth_ratio non-negative")
        pk_fid = np.exp(self._log_interpolator(np.log(k)))
        amplitude = As_eval / self.As_fid
        tilt = (k / self.kpivot_hmpc) ** (ns_eval - self.ns_fid)
        return pk_fid * amplitude * tilt * growth_ratio**2

    @classmethod
    def from_eisenstein_hu(
        cls,
        z,
        ns,
        As,
        omega_m,
        omega_b,
        h,
        kmin=1.0e-3,
        kmax=100.0,
        n_k=4000,
    ):
        """Build the fiducial table from the Eisenstein-Hu spectrum at redshift z."""
        k = np.geomspace(kmin, kmax, n_k)
        pk = linear_matter_power(k, z, ns, As, omega_m, omega_b, h)
        return cls(k, pk, As_fid=As, ns_fid=ns, kpivot_hmpc=K_PIVOT_MPC / h)


def load_power_table(path):
    """Load a whitespace-delimited file with columns k[h/Mpc], P[(Mpc/h)^3]."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("power file must contain at least two columns: k and P(k)")
    return data[:, 0], data[:, 1]


def growth_ratio(z, z_fid, omega_m, h):
    """D(z) / D(z_fid) from the same growth integral used by Eisenstein-Hu."""
    return float(growth(z, omega_m, h) / growth(z_fid, omega_m, h))
