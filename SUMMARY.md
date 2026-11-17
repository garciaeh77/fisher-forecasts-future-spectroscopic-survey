# Pipeline, FGPA, and plots

The pipeline turns `n_s`, `A_s`, and a cold-dark-matter spectrum into the one-dimensional Lyman-alpha flux power `P_1D(k, z)`. The function to call is `p1d_skm`. You give it `k_parallel` in s/km and it returns `P_1D` in km/s.

It does this in three steps.

1. **Build `P_CDM(k, z)`.** The default is the Eisenstein & Hu (1998) linear spectrum: primordial `A_s (k/k_pivot)^(n_s-1)`, times `T(k)^2`, times the growth factor `D(z)^2`. You can instead pass a table of `k` and `P(k)` and only rescale `A_s`, `n_s`, and growth. You can also multiply the Eisenstein-Hu spectrum by a smooth alpha-beta-gamma-delta bend. All four coefficients set to zero leaves that spectrum unchanged.
2. **Turn matter into flux.** Two versions of the fluctuating Gunn-Peterson approximation sit side by side. The Kaiser version is the one inside `p1d_skm`.
3. **Project to one dimension and convert units.** The line-of-sight integral removes the modes perpendicular to the spectrum. `H(z)/((1+z) h)` converts between `h/Mpc` and s/km.

## How the FGPA is implemented

The full approximation says the optical depth is `tau = A (rho/rho_bar)^beta` and the transmitted flux is `F = exp(-tau)`. A denser patch absorbs more. `beta` comes from the temperature-density relation, `beta = 2 - 0.7 (gamma - 1)`.

**Kaiser path, used for `P_1D(k, z)`.** For small fluctuations that becomes a biased tracer of the dark-matter field, with peculiar velocities. In `p1d/flux.py`,

```text
P_F(k, mu, z) = b_F^2 (1 + beta_F mu^2)^2 P_CDM(k, z)
```

with `b_F = -0.14`, `beta_F = 1.4`, and `mu = k_parallel / k`. Both numbers are fixed. `b_F` is negative because more density means less transmitted flux. Then

```text
P_1D(k_parallel) = integral from k_parallel to k_max of (k dk / 2 pi) P_F(k, mu = k_parallel/k)
```

with `k_max = 30 h/Mpc`. This path never builds a density skewer and never exponentiates `tau`. It is smooth in `n_s` and `A_s`, which is what a Fisher forecast needs.

**Optical-depth path, used as a check.** `run_fgpa_skewers` draws Gaussian skewers from the matter `P_1D`, maps them to a lognormal density `Delta`, sets `tau = A Delta^beta`, and solves for `A` so the mean transmitted flux is 0.80. At `gamma = 1.5`, `beta = 1.65` and the fitted `A` is 0.485. It then measures `P_1D` from the flux contrast `F/<F> - 1`. Pressure smoothing and thermal broadening are in the code and are turned off. Peculiar velocities are applied only if a displacement is supplied. The Kaiser formula is the redshift-space model. The skewers are real-space optical depth plus the mean-flux calibration.

The two agree in shape at `z = 3`. They do not agree in absolute amplitude, because one amplitude is the fixed `b_F` and the other is the mean flux.

## The plots

All of them are in `plots/` as PNG and PDF.

An earlier prototype package is stored unchanged in `prototype/mini_p1d/` (`cli.py`, `pipeline.py`, `power.py`, `fgpa.py`, `projection.py`, `skewers.py`, its tests, and its two figures). The forecast pipeline in `p1d/` is the combined version.

| Plot | What it shows |
|---|---|
| `01_pcdm_ns_As` | Linear `P_CDM(k)` at `z = 0`. Raising `n_s` tilts the spectrum up at high `k`. Scaling `A_s` moves the whole curve up or down. The curve peaks near `0.02 h/Mpc` and then falls, with baryon wiggles. |
| `02_p1d_ns_As` | The same `n_s` and `A_s` changes after the Kaiser FGPA and the line-of-sight integral, as `P_1D` in km/s at `z = 3`, from `10^-3` to `0.02` s/km. |
| `03_p1d_fractional` | The strict test. `Delta A_s/A_s = +0.01` produces a flat `+1%` in `P_1D` at every `k_parallel`. `Delta n_s = +0.01` tilts it, from about `+2%` at `10^-3` s/km to about `+4%` at `0.02` s/km. |
| `04_p1d_redshifts` | `P_1D` at `z = 2.2`, `3.0`, and `4.0` with the cosmology held fixed. The shape stays similar. The amplitude changes through `D(z)` and through the s/km conversion. `b_F` and `beta_F` do not evolve. |
| `05_abgd_shape` | One alpha, beta, or gamma coefficient at a time. On `P_CDM` every curve crosses zero at the pivot `k_p = 1 h/Mpc`. The zero-coefficient case is exactly Eisenstein-Hu. |
| `06_table_rescaling` | A stored `P(k)` table, after changing `A_s` or `n_s`, matches a direct Eisenstein-Hu evaluation. The fractional residual is about `10^-6`. |
| `07_pipeline_stages` | The four stages on one figure: `P_CDM`, the isotropic matter `P_1D`, Kaiser versus skewer flux `P_1D` in km/s, and one transmitted-flux skewer whose mean is 0.80. |
| `08_kaiser_vs_fgpa_shape` | Both flux `P_1D` curves divided by their value at `0.005` s/km. The shapes follow each other. The skewer curve is noisy because it is measured from 256 finite skewers. |
