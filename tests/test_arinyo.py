import numpy as np

from p1d.arinyo import (
    BETA_Z3,
    flux_bias_from_tau_bias,
    flux_power,
    jeans_wavenumber_scale,
    nonlinear_factor,
)


def test_published_bias_is_negative_and_beta_is_the_table_value():
    b_f = flux_bias_from_tau_bias()
    assert b_f < 0.0
    assert abs(b_f - 0.5546 * np.log(0.6960)) < 1.0e-12
    assert BETA_Z3 == 1.205


def test_nonlinear_factor_is_unity_at_low_k_for_a_vanishing_spectrum():
    k = np.array([1.0e-3, 1.0e-2])
    # Δ² = 0 when P_L = 0, so only the pressure piece remains, and it is tiny.
    factor = nonlinear_factor(k, np.zeros_like(k), np.zeros_like(k))
    assert factor[0] > 0.999
    assert factor[-1] > 0.99


def test_pressure_term_suppresses_k_near_the_published_scale():
    k = np.array([1.0, 13.1])
    mu = np.zeros(2)
    # A flat low amplitude keeps the q1 piece small compared with (k/k_p)².
    p = np.full(2, 1.0e-4)
    factor = nonlinear_factor(k, mu, p)
    assert factor[1] < factor[0]
    assert factor[1] < 0.5


def test_kv_is_recovered_from_the_tabulated_product():
    k_v = jeans_wavenumber_scale()
    assert abs(k_v**0.627 - 1.007) < 1.0e-10


def test_flux_power_matches_kaiser_times_the_nonlinear_factor():
    k = np.array([0.2, 1.0])
    mu = np.array([0.0, 1.0])
    p = np.array([1.0e3, 10.0])
    b_f = flux_bias_from_tau_bias()
    expected = b_f**2 * (1.0 + BETA_Z3 * mu**2) ** 2 * p * nonlinear_factor(k, mu, p)
    np.testing.assert_allclose(flux_power(k, mu, p), expected)
