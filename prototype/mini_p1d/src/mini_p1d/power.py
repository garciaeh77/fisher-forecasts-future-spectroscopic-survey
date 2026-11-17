"""Three-dimensional matter-power helpers.

All wavenumbers use h/Mpc and all 3D powers use (Mpc/h)^3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class TabulatedMatterPower:
    """Log-log interpolation of a fiducial linear CDM matter spectrum."""

    k_hmpc: np.ndarray
    pk_mpc3h3: np.ndarray
    As_fid: float = 2.1e-9
    ns_fid: float = 0.965
    kpivot_hmpc: float = 0.05 / 0.674

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
        k_hmpc: np.ndarray | float,
        *,
        As: float | None = None,
        ns: float | None = None,
        growth_ratio: float = 1.0,
    ) -> np.ndarray:
        """Evaluate P_m(k,z), rescaled from the fiducial table.

        ``growth_ratio`` is D(z)/D(z_fid).  The tilt convention assumes that
        the table already contains the fiducial primordial tilt.
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


def demo_cdm_power(
    k_hmpc: np.ndarray,
    *,
    amplitude: float = 2.0e4,
    turnover_hmpc: float = 0.02,
    ns: float = 0.965,
) -> np.ndarray:
    """Return a smooth CDM-like *demonstration* spectrum.

    This is intentionally not a precision transfer function.  It makes the
    package runnable before the user supplies a CLASS/CAMB power table.
    """
    k = np.asarray(k_hmpc, dtype=float)
    x = k / turnover_hmpc
    return amplitude * x**ns / (1.0 + x ** (ns + 3.0))


def load_power_table(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a whitespace-delimited two-column k, P(k) file."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("power file must contain at least two columns: k and P(k)")
    return data[:, 0], data[:, 1]

