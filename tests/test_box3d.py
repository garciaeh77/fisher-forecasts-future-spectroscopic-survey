import numpy as np

from p1d.box3d import gaussian_box, kaiser_flux_from_density, los_displacement, measure_power


def test_exact_amplitude_reproduces_the_input_spectrum():
    length = 40.0
    n = 16
    k_pivot = 0.5

    def power(k):
        return 20.0 * (k / k_pivot) ** -1.5

    box = gaussian_box(power, length, n, seed=3, exact_amplitude=True)
    measured = measure_power(box["delta"], length)
    nonzero = box["k"] > 0.0
    np.testing.assert_allclose(measured[nonzero], box["pk"][nonzero], rtol=1.0e-8, atol=1.0e-8)


def test_kaiser_multiplier_is_recovered_in_mu_bins():
    length = 40.0
    n = 16
    b_f = -0.14
    beta = 1.4

    def power(k):
        return 50.0 * np.ones_like(k)

    box = gaussian_box(power, length, n, seed=5, exact_amplitude=True)
    flux = kaiser_flux_from_density(box, b_f, beta)
    measured = measure_power(flux, length)
    expected = b_f**2 * (1.0 + beta * box["mu"] ** 2) ** 2 * box["pk"]
    nonzero = box["k"] > 0.0
    np.testing.assert_allclose(measured[nonzero], expected[nonzero], rtol=1.0e-6, atol=1.0e-8)


def test_line_of_sight_shift_has_the_kaiser_sign():
    length = 40.0
    n = 16
    f = 0.5

    def power(k):
        return 10.0 * np.ones_like(k)

    box = gaussian_box(power, length, n, seed=9, exact_amplitude=True)
    disp_k = np.fft.fftn(los_displacement(box, f, length))
    derivative = 1j * box["kz"] * disp_k
    expected = -f * box["mu"] ** 2 * box["fourier"]
    k_nyquist = np.pi * n / length
    resolved = (box["k"] > 0.0) & (np.abs(np.abs(box["kz"]) - k_nyquist) > 1.0e-8)
    np.testing.assert_allclose(derivative[resolved], expected[resolved], atol=1.0e-6)
