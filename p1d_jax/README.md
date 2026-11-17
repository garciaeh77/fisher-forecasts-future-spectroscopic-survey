# `p1d_jax` — full JAX port of `p1d`

JIT- and autodiff-friendly port of the analytic Kaiser path, αβγδ shaping,
Jeans cutoff, FGPA skewers / 3D box, and Arinyo D₁. The NumPy package `p1d/`
is unchanged.

## Contents

| Module | Role |
| --- | --- |
| `eisenstein_hu` | EH98 transfer, growth, `P_CDM` |
| `flux` | Kaiser `P_F` and `p1d_skm` |
| `abgd` | α–β–γ–δ shape on `P_CDM` |
| `matter_table` | tabulated `P_CDM` + `A_s`/`n_s` rescaling |
| `jeans` | `exp(-2 x_J² k²)` cutoff |
| `projection` | isotropic `P_3D → P_1D` |
| `skewers` | Gaussian 1D draws + measured `P_1D` |
| `fgpa_field` | lognormal FGPA skewers (`A` via fixed bisection) |
| `arinyo` | published z=3 Arinyo-i-Prats `D_1` |
| `box3d` | Gaussian box, Kaiser / FGPA flux, `(k, μ)` bins |

64-bit floats are enabled on import (`jax_enable_x64`).

## Use

```python
import jax
import jax.numpy as jnp
from p1d_jax import p1d_skm, TabulatedMatterPower, SkewerConfig, run_fgpa_skewers

k = jnp.logspace(-3, -1.7, 64)
p = p1d_skm(k, 3.0, 0.965, 2.1e-9, 0.31, 0.049, 0.67)

def amp(As):
    return p1d_skm(k, 3.0, 0.965, As, 0.31, 0.049, 0.67).sum()

d_amp = jax.grad(amp)(2.1e-9)

table = TabulatedMatterPower.from_eisenstein_hu(3.0, 0.965, 2.1e-9, 0.31, 0.049, 0.67)
out = run_fgpa_skewers(table, SkewerConfig(n_skewers=32, n_pixels=256))
```

## Design notes vs NumPy

- Projection / `p1d` integrals use a reverse cumulative trapezoid (matches
  sliced `np.trapezoid` within ~5×10⁻⁴).
- `calibrate_mean_flux` uses fixed-iteration bisection instead of `brentq`.
- Periodic Gaussian smoothing uses an FFT kernel instead of `scipy.ndimage`.
- CIC remaps use `ndarray.at[].add` / `vmap`.
- Tabulated interpolation is linear in `(ln k, ln P)` (`jnp.interp`), not PCHIP.
- Random draws take an int seed or a JAX PRNG key.

## Tests

From the repository root (JAX installed, e.g. `lyaemu/.venv_jax`):

```bash
PYTHONPATH=. python -m pytest -q tests/test_p1d_jax.py
```
