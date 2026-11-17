import numpy as np

from mini_p1d.fgpa import (
    calibrate_mean_flux,
    lognormal_density,
    optical_depth_unit,
    transmitted_flux,
)


def test_lognormal_density_mean():
    rng = np.random.default_rng(9)
    delta = rng.normal(scale=0.7, size=1_000_000)
    density = lognormal_density(delta)
    assert abs(np.mean(density) - 1.0) < 0.005


def test_mean_flux_calibration():
    density = np.linspace(0.1, 3.0, 50_000)
    tau_unit = optical_depth_unit(density, gamma=1.5)
    normalization = calibrate_mean_flux(tau_unit, 0.78)
    flux = transmitted_flux(tau_unit, normalization)
    assert abs(np.mean(flux) - 0.78) < 1.0e-10

