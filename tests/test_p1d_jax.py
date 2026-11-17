"""Compare the JAX pipeline to the NumPy reference."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from p1d.arinyo import flux_power as np_arinyo
from p1d.eisenstein_hu import linear_matter_power as np_pk
from p1d.flux import p1d_skm as np_p1d
from p1d.fgpa_field import calibrate_mean_flux as np_calibrate
from p1d.fgpa_field import lognormal_density as np_lognormal
from p1d.jeans import jeans_suppression as np_jeans
from p1d.matter_table import TabulatedMatterPower as NpTable
from p1d.projection import project_p1d as np_project
from p1d_jax.arinyo import flux_power as jax_arinyo
from p1d_jax.eisenstein_hu import linear_matter_power as jax_pk
from p1d_jax.fgpa_field import SkewerConfig, calibrate_mean_flux, run_fgpa_skewers
from p1d_jax.fgpa_field import lognormal_density as jax_lognormal
from p1d_jax.flux import p1d_skm as jax_p1d
from p1d_jax.jeans import jeans_suppression as jax_jeans
from p1d_jax.matter_table import TabulatedMatterPower as JaxTable
from p1d_jax.projection import project_p1d as jax_project
from p1d_jax.box3d import gaussian_box, measure_power

OM, OB, H = 0.31, 0.049, 0.67
NS, AS, Z = 0.965, 2.1e-9, 3.0


def test_linear_matter_power_matches_numpy():
    k = np.logspace(-3, 1, 200)
    p_np = np_pk(k, Z, NS, AS, OM, OB, H)
    p_jx = np.asarray(jax_pk(jnp.asarray(k), Z, NS, AS, OM, OB, H))
    assert np.max(np.abs(p_jx / p_np - 1.0)) < 1.0e-10


def test_p1d_skm_matches_numpy():
    k = np.logspace(-3, np.log10(0.02), 80)
    p_np = np_p1d(k, Z, NS, AS, OM, OB, H)
    p_jx = np.asarray(jax_p1d(jnp.asarray(k), Z, NS, AS, OM, OB, H))
    assert np.max(np.abs(p_jx / p_np - 1.0)) < 5.0e-4


def test_p1d_skm_grad_as():
    k = jnp.logspace(-3, -1.7, 40)

    def mean_power(As):
        return jnp.mean(jax_p1d(k, Z, NS, As, OM, OB, H))

    g = jax.grad(mean_power)(AS)
    eps = 1.0e-4 * AS
    fd = (mean_power(AS + eps) - mean_power(AS - eps)) / (2.0 * eps)
    assert abs(float(g / fd - 1.0)) < 1.0e-3


def test_p1d_skm_jit():
    k = jnp.logspace(-3, -1.7, 32)
    fn = jax.jit(lambda kk: jax_p1d(kk, Z, NS, AS, OM, OB, H))
    p1 = np.asarray(fn(k))
    p2 = np.asarray(fn(k))
    assert np.allclose(p1, p2)


def test_projection_matches_numpy():
    q = np.geomspace(1e-3, 30.0, 2000)
    pk = np_pk(q, Z, NS, AS, OM, OB, H)
    kp = np.logspace(-2, 0, 40)
    p_np = np_project(kp, q, pk)
    p_jx = np.asarray(jax_project(jnp.asarray(kp), jnp.asarray(q), jnp.asarray(pk)))
    assert np.max(np.abs(p_jx / p_np - 1.0)) < 5.0e-4


def test_jeans_matches_numpy():
    k = np.logspace(-2, 1, 100)
    x_j = 0.1
    j_np = np_jeans(k, x_j)
    j_jx = np.asarray(jax_jeans(jnp.asarray(k), x_j))
    assert np.allclose(j_jx, j_np)


def test_arinyo_matches_numpy():
    k = np.logspace(-2, 1, 80)
    mu = 0.5
    p_lin = np_pk(k, Z, NS, AS, OM, OB, H)
    a_np = np_arinyo(k, mu, p_lin)
    a_jx = np.asarray(jax_arinyo(jnp.asarray(k), mu, jnp.asarray(p_lin)))
    assert np.max(np.abs(a_jx / a_np - 1.0)) < 1.0e-10


def test_lognormal_and_calibrate():
    rng = np.random.default_rng(0)
    delta = rng.normal(size=(16, 128))
    dens_np = np_lognormal(delta)
    dens_jx = np.asarray(jax_lognormal(jnp.asarray(delta)))
    assert np.allclose(dens_jx, dens_np)
    tau = dens_np**1.65
    a_np = np_calibrate(tau, 0.8)
    a_jx = float(calibrate_mean_flux(jnp.asarray(tau), 0.8))
    assert abs(a_jx / a_np - 1.0) < 1.0e-6


def test_matter_table_rescaling():
    table = JaxTable.from_eisenstein_hu(Z, NS, AS, OM, OB, H, n_k=1000)
    k = jnp.logspace(-2, 1, 50)
    p0 = table(k)
    p_as = table(k, As=1.1 * AS)
    assert np.allclose(np.asarray(p_as / p0), 1.1, rtol=1e-10)


def test_fgpa_skewers_runs_and_mean_flux():
    table = JaxTable.from_eisenstein_hu(Z, NS, AS, OM, OB, H, n_k=2000)
    cfg = SkewerConfig(
        n_pixels=256,
        n_skewers=32,
        length_mpch=100.0,
        target_mean_flux=0.80,
        seed=7,
    )
    out = run_fgpa_skewers(table, cfg)
    assert abs(float(out["mean_flux"]) - 0.80) < 1.0e-4
    assert out["p1d_flux_kms"].shape == out["k_skm"].shape


def test_gaussian_box_exact_amplitude():
    def power(k):
        return jnp.exp(-((jnp.log(k) + 1.0) ** 2))

    box = gaussian_box(power, length=50.0, n=16, seed=1, exact_amplitude=True)
    measured = measure_power(box["delta"], 50.0)
    use = box["k"] > 0.0
    ratio = np.asarray(measured[use] / box["pk"][use])
    assert np.max(np.abs(ratio - 1.0)) < 1.0e-5
