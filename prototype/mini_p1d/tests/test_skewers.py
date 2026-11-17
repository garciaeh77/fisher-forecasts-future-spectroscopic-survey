import numpy as np

from mini_p1d.skewers import generate_gaussian_skewers, measure_p1d, rfft_modes


def test_gaussian_skewers_recover_input_power():
    length = 100.0
    n_pixels = 128
    k = rfft_modes(length, n_pixels)
    target = np.zeros_like(k)
    target[1:] = 0.5 / (1.0 + k[1:] ** 2)
    fields = generate_gaussian_skewers(
        target,
        length_mpch=length,
        n_pixels=n_pixels,
        n_skewers=5000,
        seed=7,
    )
    measured = measure_p1d(fields, length)
    np.testing.assert_allclose(measured[1:20], target[1:20], rtol=0.06)

