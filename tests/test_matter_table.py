import numpy as np
import pytest

from p1d.eisenstein_hu import linear_matter_power
from p1d.matter_table import TabulatedMatterPower


def test_table_rejects_unsorted_or_nonpositive_k():
    with pytest.raises(ValueError):
        TabulatedMatterPower(np.array([0.1, 0.2, 0.05, 0.4]), np.ones(4))
    with pytest.raises(ValueError):
        TabulatedMatterPower(np.array([-1.0, 0.1, 0.2, 0.3]), np.ones(4))


def test_as_and_ns_rescaling_matches_eisenstein_hu():
    om, ob, h = 0.31, 0.049, 0.67
    ns_fid, as_fid, z = 0.965, 2.1e-9, 3.0
    table = TabulatedMatterPower.from_eisenstein_hu(z, ns_fid, as_fid, om, ob, h)
    k = np.logspace(-2, 1, 40)
    direct = linear_matter_power(k, z, ns_fid + 0.02, as_fid * 1.1, om, ob, h)
    via = table(k, As=as_fid * 1.1, ns=ns_fid + 0.02)
    np.testing.assert_allclose(via, direct, rtol=2.0e-4)


def test_outside_table_raises():
    table = TabulatedMatterPower(np.geomspace(0.01, 1.0, 20), np.geomspace(1.0e4, 10.0, 20))
    with pytest.raises(ValueError):
        table(np.array([0.001]))
