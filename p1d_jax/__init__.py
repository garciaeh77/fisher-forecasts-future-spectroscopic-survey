"""JAX port of the full analytic + FGPA + box P_1D / P_F pipeline.

Enable 64-bit floats on import. The NumPy package ``p1d`` is unchanged.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from p1d_jax.abgd import linear_matter_power_abgd, p1d_skm_abgd, shape_factor
from p1d_jax.arinyo import flux_power as arinyo_flux_power
from p1d_jax.box3d import (
    bin_power,
    fgpa_flux_box,
    gaussian_box,
    growth_rate,
    kaiser_flux_from_density,
    los_displacement,
    measure_power,
)
from p1d_jax.eisenstein_hu import growth, hubble_kms, linear_matter_power, transfer
from p1d_jax.fgpa_field import SkewerConfig, run_fgpa_skewers
from p1d_jax.flux import flux_power_1d, flux_power_3d, p1d_skm
from p1d_jax.jeans import jeans_suppression, with_jeans
from p1d_jax.matter_table import TabulatedMatterPower
from p1d_jax.projection import project_p1d
from p1d_jax.skewers import generate_gaussian_skewers, measure_p1d, rfft_modes

__all__ = [
    "transfer",
    "growth",
    "hubble_kms",
    "linear_matter_power",
    "flux_power_3d",
    "flux_power_1d",
    "p1d_skm",
    "shape_factor",
    "linear_matter_power_abgd",
    "p1d_skm_abgd",
    "TabulatedMatterPower",
    "jeans_suppression",
    "with_jeans",
    "project_p1d",
    "rfft_modes",
    "generate_gaussian_skewers",
    "measure_p1d",
    "SkewerConfig",
    "run_fgpa_skewers",
    "arinyo_flux_power",
    "growth_rate",
    "gaussian_box",
    "measure_power",
    "bin_power",
    "kaiser_flux_from_density",
    "los_displacement",
    "fgpa_flux_box",
]
