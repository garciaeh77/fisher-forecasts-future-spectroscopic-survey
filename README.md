# Analytic Lyman-alpha P_1D

Stage 1 of the future-survey Fisher forecast. This computes a linear
cold-dark-matter spectrum with the Eisenstein & Hu (1998) transfer function,
applies a linearised Fluctuating Gunn-Peterson (Kaiser) flux model, and
projects it to the 1D flux power P_1D(k, z).

Two paths share the same P_CDM:

1. **Kaiser path** (`p1d_skm`): closed-form flux power and a line-of-sight integral. This is the fast P_1D(k, z) for a later Fisher forecast.
2. **Optical-depth path** (`run_fgpa_skewers`): lognormal skewers, tau = A * Delta^beta, with A set by a target mean flux. This checks where the Kaiser formula is only an amplitude.

P_CDM is either computed with Eisenstein-Hu or read from a two-column file and rescaled in A_s, n_s, and growth. An optional alpha-beta-gamma-delta bend multiplies the Eisenstein-Hu spectrum.

## Run

```bash
cd fisher-forecasts-future-spectroscopic-survey
python make_checks.py
python -m pytest -q
```

Plots are written to `plots/`. `SUMMARY.md` explains the pipeline, the FGPA, and every plot. `REPORT.md` is the meeting note on how this fulfills the task.

An earlier skewer prototype (command-line tool, examples, tests, and figures) is kept unchanged in `prototype/mini_p1d/`. The forecast pipeline uses the versions under `p1d/`.

Part II adds Jeans smoothing, a 3D FGPA box with linear peculiar velocities and thermal broadening, and Arinyo-i-Prats overlays (`make_extension_plots.py`, `make_meeting_plots.py`, `make_full_physics_plots.py`). A JAX port of the same pipeline is in `p1d_jax/` for later derivatives.

## Defaults

| Quantity | Value |
|---|---|
| Omega_m | 0.31 |
| Omega_b | 0.049 |
| h | 0.67 |
| n_s | 0.965 |
| A_s | 2.1e-9 |
| b_F | -0.14 |
| beta_F | 1.4 |

`b_F` and `beta_F` are constant. They set the amplitude of P_1D. The shape in
`n_s` and `A_s` comes from P_CDM.

## Alpha-beta-gamma-delta shape

`p1d/abgd.py` multiplies the Eisenstein-Hu spectrum by a smooth bend,

    P(k) = P_EH(k) * exp(alpha*L + beta*L^2 + gamma*L^3 + delta*L^4),

with `L = ln(k / k_pivot)` and `k_pivot = 1 h/Mpc`. All four coefficients
default to zero, which returns the Eisenstein-Hu result. Call
`p1d_skm_abgd` for P_1D in km/s. See `REPORT.md` for the pipeline and for
which external Lyα codes this calculator does not depend on.
