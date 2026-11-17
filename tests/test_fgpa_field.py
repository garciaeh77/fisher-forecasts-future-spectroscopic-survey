import numpy as np
import pytest

from p1d.fgpa_field import (
    calibrate_mean_flux,
    fgpa_exponent,
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


def test_beta_from_gamma():
    assert fgpa_exponent(1.5) == pytest.approx(2.0 - 0.7 * 0.5)


def test_mean_flux_must_lie_between_zero_and_one():
    tau = np.linspace(0.1, 1.0, 20)
    with pytest.raises(ValueError):
        calibrate_mean_flux(tau, 1.0)
    with pytest.raises(ValueError):
        calibrate_mean_flux(tau, 0.0)
