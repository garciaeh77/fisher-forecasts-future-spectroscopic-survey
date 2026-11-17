# Summary meeting — Kier, Vera, Simeon

**Purpose.** Written brief of Part I and Part II of the Lyman-α \(P_{1\mathrm{D}}\) calculator, with precise code notes, open questions, and a **10-minute spoken script**.

**Repository.** `fisher-forecasts-future-spectroscopic-survey`  
**Notes.** `docs/part1.tex`, `docs/part2.tex`, `docs/references.bib`  
**Package.** `p1d/`

**Fiducial cosmology (all numerical examples unless stated).**
\[
\Omega_m=0.31,\quad \Omega_b=0.049,\quad h=0.67,\quad
n_s=0.965,\quad A_s=2.1\times 10^{-9}.
\]
Default forest redshift for Part I comparisons: \(z=3\).  
Fixed Kaiser bias: \(b_F=-0.14\), \(\beta_F=1.4\).

---

# Part I — what was done

## Scientific goal

Build a **forward model** for the one-dimensional Lyman-α flux power \(P_{1\mathrm{D}}(k_\parallel,z)\), with:

- **inputs:** \(n_s\), \(A_s\), and a linear cold-dark-matter spectrum \(P_\mathrm{CDM}\);
- **output:** \(k_\parallel\) in \(\mathrm{s\,km}^{-1}\), \(P_{1\mathrm{D}}\) in \(\mathrm{km\,s}^{-1}\);
- **scope:** one redshift at a time; **not** yet the quasar/galaxy cross-power, covariance, or Fisher matrix.

Two flux paths share the **same** \(P_\mathrm{CDM}\):

1. **Kaiser (analytic)** — formula only; smooth in \(n_s\) and \(A_s\).
2. **FGPA sightlines** — Gaussian → lognormal density → optical depth → measured periodogram.

## Equations implemented

**Matter spectrum (Eisenstein–Hu).**  
`p1d/eisenstein_hu.py` → `linear_matter_power(k, z, ns, As, omega_m, omega_b, h)`  
Transfer function with BAO wiggles; growth \(D(z)\); pivot \(k=0.05\,\mathrm{Mpc}^{-1}\).

**Optional table.**  
`p1d/matter_table.py` → `TabulatedMatterPower`  
Stores \(P(k)\) and rescales in \(A_s\) / \(n_s\) about a pivot without rebuilding EH each time. Built with `from_eisenstein_hu(...)`.

**Smooth deformation (α, β, γ, δ).**  
`p1d/abgd.py` → `p1d_skm_abgd`  
Multiplicative shape change of \(P_\mathrm{CDM}\) about \(k_p=1\,h\,\mathrm{Mpc}^{-1}\); zero coefficients recover EH.

**Kaiser flux.**
\[
P_F(k,\mu)=b_F^{2}\,(1+\beta_F\mu^{2})^{2}\,P_\mathrm{CDM}(k),\qquad
\mu=k_\parallel/k.
\]
\[
P_{1\mathrm{D}}(k_\parallel)
=
\int_{k_\parallel}^{k_\mathrm{max}}
\frac{k\,\mathrm{d}k}{2\pi}\,
P_F\!\left(k,\mu=\tfrac{k_\parallel}{k}\right),
\]
with default \(k_\mathrm{max}=30\,h\,\mathrm{Mpc}^{-1}\), trapezoidal quadrature on ~800 log-\(k\) nodes.

**Code:** `p1d/flux.py`  
- `flux_power_3d` — \(P_F(k,\mu)\)  
- `flux_power_1d` — line-of-sight integral  
- `p1d_skm` — end-to-end in forest units (calls EH unless `matter_power=` is passed)  
- Unit conversion uses \(H(z)/[(1+z)h]\) via `k_hmpc_to_skm`, `p1d_h_to_kms`.

**Isotropic 1D matter projection.**  
`p1d/projection.py` → `project_p1d`  
\[
P_{1\mathrm{D,m}}(k_\parallel)=\int_{k_\parallel}^{k_\mathrm{max}}\frac{k\,\mathrm{d}k}{2\pi}\,P_\mathrm{CDM}(k).
\]

**Gaussian skewers.**  
`p1d/skewers.py`  
- `rfft_modes(L, N)` — \(k_j=2\pi j/L\)  
- `generate_gaussian_skewers` — complex Gaussian modes with variance set by \(P_{1\mathrm{D,m}}\); zero mode = 0; real field via irfft  
- `measure_p1d` — periodogram  
- `smooth_periodic` — optional pixel Gaussian (pressure/thermal); **default width = 0**

**FGPA / lognormal path.**  
`p1d/fgpa_field.py` → `run_fgpa_skewers(table, SkewerConfig)`  

Sequence inside `run_fgpa_skewers`:

1. Build \(k\) modes for length \(L\), \(N\) pixels.  
2. Evaluate tabulated \(P_\mathrm{CDM}\) on a fine \(q\)-grid; `project_p1d` → target 1D matter power.  
3. `generate_gaussian_skewers` → \(\delta_G\).  
4. Optional pixel pressure smooth (default off).  
5. `lognormal_density`: \(\Delta=\exp(\delta_G-\tfrac12\mathrm{Var}(\delta_G))\).  
6. `prepare_tau_unit`: \(\tau_\mathrm{unit}=\Delta^\beta\), \(\beta=2-0.7(\gamma-1)\); default \(\gamma=1.5\) ⇒ \(\beta=1.65\).  
7. Optional LOS CIC remapping + thermal smooth (default off).  
8. `calibrate_mean_flux`: solve \(\langle e^{-A\tau_\mathrm{unit}}\rangle=\langle F\rangle_\mathrm{target}\) (default \(0.80\)).  
9. \(F=e^{-A\tau}\), \(\delta_F=F/\langle F\rangle-1\), measure \(P_{1\mathrm{D}}\), convert to km/s.

**Entry points.**  
- Analytic: `p1d_skm(...)`  
- Sightlines: `run_fgpa_skewers(TabulatedMatterPower, SkewerConfig)`  

**Figures 01–08.** `python make_checks.py` → `plots/01_…`–`08_…`

## Part I result that mattered for the meeting

On figure **07**, Kaiser and the mean of 256 FGPA lines share \(P_\mathrm{CDM}\) but **do not overlie in absolute amplitude**.  
On figure **08**, dividing each curve by its value at \(0.005\,\mathrm{s\,km}^{-1}\) shows the **shapes** agree.  
Interpretation: different normalisations / physics (fixed \(b_F\) + RSD vs \(\langle F\rangle=0.80\) in real space), not a coding bug.

---

# Part II — what was implemented

Nothing in Part I was rewritten. Extensions are **new modules** or optional arguments in front of the old calls.

## Requests from 13 October (mapping)

| Request | Implementation |
|---|---|
| Pressure / Jeans filter | `p1d/jeans.py` |
| Lognormal vs Gaussian on small scales | fig 10 via existing `lognormal_density` |
| More sightlines / noise test | fig 12, up to \(10\,000\) lines |
| Spectra to \(k\sim 0.2\,\mathrm{s\,km}^{-1}\) at \(z\sim 5\) | fig 13 |
| 3D realisation → measured \(P_F(k,\mu)\) | `p1d/box3d.py`, fig 14 |
| Compare to Arinyo-i-Prats (2015) | `p1d/arinyo.py`, fig 14 |

## New code, precisely

### 1. Jeans cutoff — `p1d/jeans.py`

\[
P_\mathrm{CDM}(k)\;\to\;P_\mathrm{CDM}(k)\,\exp(-2 x_J^{2} k^{2}),
\]
with \(k\) in \(h\,\mathrm{Mpc}^{-1}\), \(x_J\) in \(\mathrm{Mpc}/h\).  
Arya et al. (2022) form. Examined lengths: \(x_J=0.10\) (\(100\,\mathrm{kpc}/h\)) and \(0.05\) (\(50\,\mathrm{kpc}/h\)).

- `jeans_suppression(k, x_j)` — multiplicative array; \(x_J=0\) → ones; negative rejected.  
- `with_jeans(table, x_j)` — **new** `TabulatedMatterPower`; copies `As_fid`, `ns_fid`, pivot; **does not mutate** the input.

Usage pattern:
```text
table = TabulatedMatterPower.from_eisenstein_hu(...)
table_j = with_jeans(table, 0.10)          # or pass table for x_J=0
p1d_skm(..., matter_power=table_j)         # Kaiser on screened P
run_fgpa_skewers(table_j, config)          # sightlines on screened P
```

Pixel-space Gaussians from Part I remain available and stay at **zero width** in Part II figures; the only small-scale cut is Jeans on \(P_\mathrm{CDM}\).

### 2. Extension figures — `make_extension_plots.py`

- **09** — \(P_\mathrm{CDM}\) and Kaiser \(P_F(k,\mu)\) with/without Jeans.  
- **10** — \(P_{1\mathrm{D}}\) of \(\delta_G\) vs \(\Delta-1\) for three \(x_J\) (\(L=40\,\mathrm{Mpc}/h\), \(N=1024\), 512 lines).  
- **11** — Kaiser vs FGPA flux \(P_{1\mathrm{D}}\) on the same screened spectra.  
- **12** — noise: one draw of \(10\,000\) lines; spectra of first 256 / 1024 / 4096 / all; Kaiser overlay.

### 3. Meeting figures — `make_meeting_plots.py`

- **13** — \(z=5\), out to \(0.2\,\mathrm{s\,km}^{-1}\); \(\langle F\rangle=e^{-\tau_\mathrm{eff}}\) from Faucher-Giguère \(\tau_\mathrm{eff}=0.0018(1+z)^{3.92}\) → \(\langle F\rangle\approx 0.132\) (**extrapolation** past \(z=4.2\)).  
- **14** — one \(40\,\mathrm{Mpc}/h\), \(128^{3}\) cube; measure \(P_F(k,\mu)\); overlay Kaiser + Arinyo.

### 4. Arinyo formula — `p1d/arinyo.py`

\[
P_F(k,\mu)=b_{F\delta}^{2}(1+\beta\mu^{2})^{2}\,P_L(k)\,D_1(k,\mu),
\]
with published \(z=3\) fiducial-simulation numbers (Tables 2–3), \(q_2=0\), **not refitted**.  
\(P_L\) is our EH spectrum **without** an extra Jeans factor (\(D_1\) already has \(k_p\)).

### 5. 3D box — `p1d/box3d.py`

- `gaussian_box(power, L, n, seed, exact_amplitude=...)` — periodic Gaussian field; for fig 14, `exact_amplitude=True` (rms amplitude, random phase only — same spirit as Arinyo ICs).  
- `growth_rate` — \(f=\mathrm{d}\ln D/\mathrm{d}\ln a\) from Part I growth.  
- `los_displacement` — linear continuity: \(\psi_\parallel(\mathbf{k})=\mathrm{i}\,f\,k_\parallel\delta/k^{2}\).  
- `fgpa_flux_box` — lognormal → \(\tau\) → optional CIC shift along one axis → solve \(A\) for target \(\langle F\rangle\) (cube uses \(0.696\), Arinyo’s \(z=3\) value).  
- `bin_power` — average \(|\delta_F(\mathbf{k})|^{2}\) in \((k,\mu)\) bins.  
- Estimator check: field \(b_F(1+\beta_F\mu^{2})\delta(\mathbf{k})\) recovers analytic Kaiser to \(\sim 10^{-13}\).

### Tests

```text
python -m pytest -q
```
Includes `tests/test_jeans.py`, `tests/test_arinyo.py`, `tests/test_box3d.py` (full suite ~21 tests).

## Part II findings (tied to figures)

1. **Jeans (09, 11).** Large scales unchanged; \(100\,\mathrm{kpc}/h\) cuts earlier/harder than \(50\). Sightline flux falls **less steeply** than Kaiser at high \(k\) (lognormal regenerates small-scale power).  
2. **Gaussian vs lognormal (10).** No cutoff ⇒ ~⅓ of Gaussian pixels have \(1+\delta\le 0\) (unusable for \(\Delta^\beta\)). At \(100\,\mathrm{kpc}/h\), \(\mathrm{Var}(\Delta)\sim 2\) (weakly nonlinear).  
3. **Noise (12).** Scatter falls roughly as \(1/\sqrt{N}\) from 256 → 10 000 lines. **Offset from Kaiser remains** at 10 000 lines → model difference, not sample variance.  
4. **\(z=5\) (13).** Reach \(0.2\,\mathrm{s\,km}^{-1}\). Sightline \(P_{1\mathrm{D}}\) ≫ Kaiser because \(\langle F\rangle=0.13\) vs fixed \(b_F=-0.14\).  
5. **Cube (14).** Measured \(P_F(k,\mu)\) has Kaiser-sense anisotropy; **shape** tracks Arinyo; **amplitude** lies above published formula. Limits: one small box, linear velocities only, parameters not refit.

### Extra sightline demo (after Part II notes)

FGPA transmitted-flux lines at varied \(A_s\)/\(n_s\) (same seed):  
`plots/15_…`, overlay+zoom `16_…`, \(\Delta F\) `17_…`, matched dramatic shifts `19_fgpa_matched_dramatic_As_ns` (\(A_s\times 0.5\) ↔ \(n_s\approx 0.806\) matched on \(\mathrm{Var}(\delta_G)\)).

---

# Potential questions for the meeting

## Decisions we need from Kier / Vera / Simeon

1. **Normalisation.** Keep Kaiser at fixed \(b_F=-0.14\), or retune \(b_F\) (and \(\beta_F\)) so large-scale flux matches FGPA / cube at a chosen \(\langle F\rangle\)?  
2. **Mean transmission at \(z=5\).** Keep FG08 extrapolation (\(\langle F\rangle\sim 0.13\)), hold \(0.80\) to isolate growth/units, or use another \(\langle F\rangle(z)\)?  
3. **Jeans length.** Fix \(x_J=100\,\mathrm{kpc}/h\), treat as nuisance, or drop explicit Jeans and rely on Arinyo \(k_p\) only?  
4. **Arinyo.** Stay with published numbers, refit on our cosmology, or move to hydro / emulator target?  
5. **3D fidelity.** Is one \(40\,\mathrm{Mpc}/h\), \(128^{3}\) box enough for shape checks, or larger volume / resolution before trusting \(k\sim k_p\)?  
6. **Velocities / thermal.** Add thermal broadening and/or nonlinear RSD next, or keep linear continuity until Fisher exists?  
7. **Forecast order.** Cross-power with quasars/galaxies first, then covariance, then Fisher — or \(P_{1\mathrm{D}}\) covariance alone first?  
8. **Which forward model enters Fisher?** Kaiser only, FGPA ensemble, Arinyo \(D_1\), or 3D lognormal box?  
9. **Parameters to vary.** Only \(n_s\), \(A_s\), or also IGM/bias (\(A\), \(\gamma\), \(x_J\), \(b_F\), \(\beta_F\))?

## Questions they may ask you

- Why don’t Kaiser and FGPA match in absolute power?  
  → Different normalisations: \(b_F^2\) + RSD vs \(\langle F\rangle\) in real space; shapes agree when divided out (fig 08); offset survives 10 000 lines (fig 12).  
- Is the lognormal “correct”?  
  → Positive density + weakly nonlinear at \(100\,\mathrm{kpc}/h\); not hydro; Arya papers document limitations (PDF, strong overdensities).  
- Why is the cube above Arinyo?  
  → Different model (lognormal + explicit Jeans vs hydro-calibrated \(D_1\)), different cosmology, no refit; slope ok, amplitude not.  
- What’s next?  
  → Ask them to pick from the decision list above; your default proposal: lock normalisation + \(\langle F\rangle(z)\), then cross-power → covariance → Fisher.

---

# 10-minute presentation script

**Audience:** Kier, Vera, Simeon  
**Length:** ~10 minutes speaking + discussion  
**Props:** Overleaf Part I / Part II (or PDFs); optional slides/figures 07, 09–14  
**Tone:** summary meeting — what exists, what we learned, what to decide next.

Suggested timing:

| Min | Block |
|-----|--------|
| 0:00–0:45 | Opening |
| 0:45–3:00 | Part I (two methods + fig 07) |
| 3:00–7:30 | Part II (six requests → findings) |
| 7:30–9:00 | Code map (short) |
| 9:00–10:00 | Open questions / ask for guidance |

---

## [0:00] Opening (~45 s)

Thank you Kier, Vera, Simeon. This is a short summary of the Lyman-α \(P_{1\mathrm{D}}\) forward model.

I will cover three things: what Part I already did; what I implemented after the 13 October requests; and a short list of decisions where I would like your guidance before cross-power, covariance, and Fisher.

I am not presenting the Fisher forecast today — only the forward model and the Part II extensions.

---

## [0:45] Part I — two ways to get flux power (~2 min)

The calculator lives in the `p1d` package. Inputs are \(n_s\), \(A_s\), and a linear \(P_\mathrm{CDM}\). Output is one-dimensional flux power in forest units, at one redshift.

**Matter.** Default \(P_\mathrm{CDM}\) is Eisenstein–Hu, in `eisenstein_hu.py`. I can also load a table and rescale \(A_s\) or \(n_s\), or apply a smooth four-parameter shape deformation.

**Method 1 — Kaiser.** In `flux.py`, function `p1d_skm`. The forest is a linearly biased tracer with coherent velocities:

\[
P_F=b_F^{2}(1+\beta_F\mu^{2})^{2}P_\mathrm{CDM},
\]

with \(b_F=-0.14\) and \(\beta_F=1.4\) fixed — a round form of early DESI. Then I integrate out transverse modes to get \(P_{1\mathrm{D}}(k_\parallel)\). No random field. Smooth in cosmology.

**Method 2 — FGPA sightlines.** In `fgpa_field.py`, function `run_fgpa_skewers`. Project \(P_\mathrm{CDM}\) to 1D, draw Gaussian lines, map to a **lognormal** density so the density stays positive, set \(\tau=A\Delta^{1.65}\), solve \(A\) so mean transmission is \(0.80\), measure the flux power from the lines.

**Figure 07** is the Part I summary: same matter spectrum into both paths. In absolute units the curves do not overlie. **Figure 08** shows that once each is divided by its large-scale amplitude, the shapes agree. So the October offset was a model-normalisation issue, not a coding error.

---

## [3:00] Part II — what you asked for and what I built (~4.5 min)

I did not rewrite Part I. Each extension is a new call in front of those functions, or a new module beside them.

**Pressure.** `jeans.py`: multiply a stored \(P_\mathrm{CDM}\) by \(\exp(-2 x_J^{2} k^{2})\). I used \(100\) and \(50\,\mathrm{kpc}/h\). `with_jeans` returns a new table; the original is untouched. That table can go into Kaiser or into the sightlines.

**Figure 09:** large scales unchanged; \(100\,\mathrm{kpc}/h\) falls first.  
**Figure 10:** without a cutoff, a Gaussian density goes negative on about a third of the pixels — you cannot raise it to a power. At \(100\,\mathrm{kpc}/h\) the lognormal variance is about two, so weakly nonlinear. The lognormal always keeps more small-scale power than the Gaussian after the cutoff.  
**Figure 11:** same cutoff in flux space; sightlines fall less steeply than Kaiser at high \(k\).

**Ensemble noise — Figure 12.** October’s fluctuating curve was 256 lines. I went to \(10\,000\). The scatter falls roughly as one over square root of \(N\). The vertical offset from Kaiser is **still there** at \(10\,000\) lines. So that offset is the flux model, not sample variance.

**Redshift 5 — Figure 13.** Spectra reach \(0.2\,\mathrm{s\,km}^{-1}\). Mean transmission \(0.13\) from the Faucher-Giguère fit extrapolated past \(z=4.2\). Sightline power sits far above Kaiser because those two amplitudes answer different conditions.

**Three-dimensional box — Figure 14.** One \(40\,\mathrm{Mpc}/h\), \(128^{3}\) cube, Jeans \(100\,\mathrm{kpc}/h\), mean transmission \(0.696\). Linear continuity velocities; measure \(P_F(k,\mu)\). Parallel power lies above real space; transverse below — Kaiser sense of anisotropy. The slope follows the published Arinyo-i-Prats formula at their \(z=3\) parameters; the amplitude lies above it. I have not refitted their numbers. The box is small, and thermal broadening is still off.

---

## [7:30] Code map in one minute (~1 min)

Remember three entry points:

1. `p1d_skm` — analytic Kaiser \(P_{1\mathrm{D}}\) (optional `matter_power=` for a Jeans table).  
2. `run_fgpa_skewers` — optical-depth lines from a table.  
3. `box3d.fgpa_flux_box` + `bin_power` — measured \(P_F(k,\mu)\); Arinyo overlay from `arinyo.flux_power`.

Reproduce figures:

```text
python make_checks.py              # 01–08
python make_extension_plots.py     # 09–12
python make_meeting_plots.py       # 13–14
python -m pytest -q
```

Written notes with equations, references, and figures are in the two Overleaf / `.tex` documents: Part I and Part II.

---

## [9:00] What I would like from you (~1 min)

Before I start cross-power and Fisher, three decisions would help most:

1. **How should we normalise** Kaiser versus mean transmission — keep \(b_F=-0.14\), or match large-scale amplitude to the FGPA / cube?  
2. **What \(\langle F\rangle\) at \(z=5\)** should we use — extrapolated FG08, or something else?  
3. **Which forward model** should enter the forecast first — analytic Kaiser, FGPA, Arinyo, or the 3D box?

I am happy to take questions, or to walk any figure in more detail.

---

## Backup lines (if asked)

**“Show me a sightline.”**  
Bottom-right of fig 07, or the later \(A_s\)/\(n_s\) overlays (`15`–`19`). Differences in \(F(x)\) are subtle when \(\langle F\rangle\) is held fixed; \(\Delta F\) panels make them clearer. Matched dramatic test: \(A_s\times 0.5\) vs \(n_s\approx 0.806\) (same density variance).

**“Is this ready for Fisher?”**  
Forward model for auto \(P_{1\mathrm{D}}\) exists in two forms. Cross-power, survey window, noise, and covariance are not implemented yet.

**“What did you not do?”**  
No thermal broadening in the Part II figures; velocities in the cube are linear only; one small box; Arinyo parameters not refit; FG08 at \(z=5\) is an extrapolation; no \(P_x\), no covariance, no Fisher.

---

## Checklist before the call

- [ ] Open Part I / Part II PDF or Overleaf  
- [ ] Have figures 07, 09–14 ready (talk deck or image viewer)  
- [ ] Know the three decision questions by heart  
- [ ] Repo path and `pytest` green if someone asks about tests  

**End of document.**
