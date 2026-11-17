import numpy as np
import pytest

from p1d.projection import project_p1d


def test_power_law_projection():
    amplitude = 3.2
    slope = -3.2
    q = np.geomspace(1.0e-3, 1.0e5, 200_000)
    pk = amplitude * q**slope
    k = np.array([0.02, 0.1, 0.8])
    measured = project_p1d(k, q, pk)
    exact = -amplitude * k ** (slope + 2.0) / (2.0 * np.pi * (slope + 2.0))
    np.testing.assert_allclose(measured, exact, rtol=2.0e-4)


def test_projection_refuses_extrapolation():
    q = np.geomspace(0.01, 10.0, 1000)
    pk = q ** -2.5
    with pytest.raises(ValueError):
        project_p1d(np.array([0.001]), q, pk)
