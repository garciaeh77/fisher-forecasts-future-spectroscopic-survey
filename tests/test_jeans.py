import numpy as np

from p1d.jeans import jeans_suppression, with_jeans
from p1d.matter_table import TabulatedMatterPower


def test_zero_jeans_length_is_unity():
    k = np.array([0.1, 1.0, 10.0])
    np.testing.assert_allclose(jeans_suppression(k, 0.0), np.ones(3))


def test_jeans_suppresses_small_scales_only():
    k = np.array([1.0e-3, 10.0])
    factor = jeans_suppression(k, 0.1)
    assert factor[0] > 0.999
    assert factor[1] < 0.2


def test_with_jeans_does_not_modify_the_original_table():
    k = np.geomspace(1.0e-2, 20.0, 40)
    pk = np.full(40, 100.0)
    table = TabulatedMatterPower(k, pk)
    screened = with_jeans(table, 0.1)
    np.testing.assert_allclose(table.pk_mpc3h3, pk)
    assert screened(k)[0] > screened(k)[-1]
