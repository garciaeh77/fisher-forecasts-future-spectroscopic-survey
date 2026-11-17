"""End-to-end matter-to-FGPA-flux P1D prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fgpa import (
    calibrate_mean_flux,
    flux_contrast,
    lognormal_density,
    prepare_tau_unit,
    transmitted_flux,
)
from .power import TabulatedMatterPower
from .projection import project_p1d
from .skewers import (
    generate_gaussian_skewers,
    measure_p1d,
    rfft_modes,
    smooth_periodic,
)


@dataclass(frozen=True)
class PipelineConfig:
    redshift: float = 3.0
    As: float = 2.1e-9
    ns: float = 0.965
    growth_ratio: float = 1.0
    length_mpch: float = 200.0
    n_pixels: int = 1024
    n_skewers: int = 512
    target_mean_flux: float = 0.80
    gamma: float = 1.5
    pressure_sigma_pixels: float = 0.0
    thermal_sigma_pixels: float = 0.0
    seed: int = 12345


def run_pipeline(
    matter_power: TabulatedMatterPower,
    config: PipelineConfig,
    *,
    displacement_pixels: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Run the standalone approximate P1D pipeline.

    Independent 1D Gaussian skewers are appropriate for a P1D prototype, but
    not for cross-skewer or full 3D correlation predictions.  RSD is skipped
    unless externally supplied displacement fields are passed.
    """
    k_modes = rfft_modes(config.length_mpch, config.n_pixels)
    nonzero = k_modes > 0
    if np.max(k_modes[nonzero]) > matter_power.kmax:
        raise ValueError(
            "the skewer Nyquist mode exceeds the supplied P3D kmax; "
            "increase table kmax or reduce n_pixels/length resolution"
        )
    if np.min(k_modes[nonzero]) < matter_power.kmin:
        raise ValueError(
            "the fundamental skewer mode lies below the supplied P3D kmin; "
            "extend the table to lower k or use a shorter skewer"
        )

    # Evaluate P3D on a dense grid that includes every requested Fourier mode.
    q_grid = np.geomspace(matter_power.kmin, matter_power.kmax, 8192)
    pk3d = matter_power(
        q_grid,
        As=config.As,
        ns=config.ns,
        growth_ratio=config.growth_ratio,
    )
    p1d_target = np.zeros_like(k_modes)
    p1d_target[nonzero] = project_p1d(k_modes[nonzero], q_grid, pk3d)

    gaussian = generate_gaussian_skewers(
        p1d_target,
        length_mpch=config.length_mpch,
        n_pixels=config.n_pixels,
        n_skewers=config.n_skewers,
        seed=config.seed,
    )
    gaussian = smooth_periodic(gaussian, config.pressure_sigma_pixels)
    density = lognormal_density(gaussian)
    tau_unit = prepare_tau_unit(
        density,
        gamma=config.gamma,
        displacement_pixels=displacement_pixels,
        thermal_sigma_pixels=config.thermal_sigma_pixels,
    )
    normalization = calibrate_mean_flux(tau_unit, config.target_mean_flux)
    flux = transmitted_flux(tau_unit, normalization)
    delta_flux = flux_contrast(flux)

    return {
        "k_hmpc": k_modes,
        "q_hmpc": q_grid,
        "pk3d_mpc3h3": pk3d,
        "p1d_matter_target_mpc_h": p1d_target,
        "p1d_matter_measured_mpc_h": measure_p1d(
            gaussian, config.length_mpch
        ),
        "p1d_flux_mpc_h": measure_p1d(delta_flux, config.length_mpch),
        "fgpa_A": normalization,
        "mean_flux": float(np.mean(flux)),
        "density_mean": float(np.mean(density)),
        "density_variance": float(np.var(density)),
        "sample_density_skewer": density[0],
        "sample_flux_skewer": flux[0],
    }

