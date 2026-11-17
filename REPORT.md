# Meeting summary: analytic \(P_{1\mathrm{D}}\) calculator

## The task

Build a mini analytic calculator that takes \(n_s\), \(A_s\), and \(P_\mathrm{CDM}\)
(an \(\alpha\)–\(\beta\)–\(\gamma\)–\(\delta\) shape to start with) and returns
\(P_{1\mathrm{D}}(k,z)\). The cross-power \(P_x\) is the second stage.

## What is in the repository

https://github.com/garciaeh77/fisher-forecasts-future-spectroscopic-survey

| Piece | File | What it does |
|---|---|---|
| Eisenstein–Hu \(P_\mathrm{CDM}\) | `p1d/eisenstein_hu.py` | Linear spectrum from \(n_s\), \(A_s\), \(\Omega_m\), \(\Omega_b\), \(h\), and \(z\). Includes baryon wiggles and growth. |
| Power table | `p1d/matter_table.py` | Reads \(k\), \(P(k)\) or builds that table from Eisenstein–Hu. Rescales \(A_s\), \(n_s\), and \(D(z)\). Refuses to extrapolate. |
| \(\alpha\beta\gamma\delta\) bend | `p1d/abgd.py` | Multiplies \(P_\mathrm{EH}\) by \(\exp(\alpha L+\beta L^2+\gamma L^3+\delta L^4)\), \(L=\ln(k/k_p)\), \(k_p=1\,h\,\mathrm{Mpc}^{-1}\). All four zero recovers Eisenstein–Hu. |
| Kaiser \(P_{1\mathrm{D}}\) | `p1d/flux.py` | \(P_F=b_F^2(1+\beta_F\mu^2)^2 P_\mathrm{CDM}\), then the line-of-sight integral. Output in km/s, \(k_\parallel\) in s/km. \(b_F=-0.14\), \(\beta_F=1.4\). |
| Isotropic projection | `p1d/projection.py` | \(\int q\,P_{3\mathrm{D}}(q)\,dq/(2\pi)\). Checked against a power-law tail. |
| Optical-depth FGPA | `p1d/fgpa_field.py`, `p1d/skewers.py` | Gaussian skewers from the matter \(P_{1\mathrm{D}}\), lognormal density, \(\beta=2-0.7(\gamma-1)\), \(A\) solved for a target mean flux, then a measured flux \(P_{1\mathrm{D}}\). |
| Checks | `make_checks.py`, `tests/` | Eight plot pairs (PNG and PDF) and ten tests. |

Defaults: \(\Omega_m=0.31\), \(\Omega_b=0.049\), \(h=0.67\), \(n_s=0.965\), \(A_s=2.1\times10^{-9}\), \(z=3\), mean flux \(0.80\), \(\gamma=1.5\) so \(\beta=1.65\).

## How this fulfills the task

- **Input \(n_s\), \(A_s\).** Both enter the Eisenstein–Hu spectrum and the rescaling of a stored \(P(k)\) table. A 1% change in \(A_s\) moves \(P_{1\mathrm{D}}\) by a flat 1%. A change in \(n_s\) tilts it. Plot `03`.
- **Input \(P_\mathrm{CDM}\).** Three ways. Eisenstein–Hu builds it. `TabulatedMatterPower` accepts an external table (CLASS, CAMB, or a sterile-neutrino spectrum) and only rescales \(A_s\), \(n_s\), and growth. The \(\alpha\beta\gamma\delta\) factor bends the Eisenstein–Hu shape. Plot `06` shows the rescaled table agrees with a direct Eisenstein–Hu call at the level of \(10^{-6}\).
- **Output \(P_{1\mathrm{D}}(k,z)\).** `p1d_skm(k, z, ns, As, ...)` is the function to quote in the meeting. Redshift enters through growth and through the conversion \(H(z)/((1+z)h)\) between s/km and \(h\,\mathrm{Mpc}^{-1}\). Plot `04`.
- **\(P_x\) is not in this repository.** It was left for the next stage.

## The two flux models, and why both are there

The Kaiser path is the linearized fluctuating Gunn-Peterson approximation. A denser patch absorbs more, so the flux contrast is \(b_F\) times the dark-matter contrast, with a Kaiser factor for peculiar velocities. It is smooth and fast, which is what a Fisher forecast needs.

The skewer path is the optical-depth formula \(\tau=A\Delta^\beta\), \(F=e^{-\tau}\). \(A\) is chosen so the mean flux is 0.80. At the fiducial run, \(A=0.485\) and \(\beta=1.65\). Plot `07` shows the four stages: \(P_\mathrm{CDM}\), matter \(P_{1\mathrm{D}}\), both flux \(P_{1\mathrm{D}}\) curves, and one transmitted-flux skewer. Plot `08` divides each flux \(P_{1\mathrm{D}}\) by its value at \(0.005\,\mathrm{s/km}\). The shapes follow each other. The absolute amplitude does not, because the Kaiser amplitude is the fixed \(b_F\) and the skewer amplitude is the mean flux. The skewer curve is noisy because it is measured from 256 finite skewers.

Pressure smoothing and thermal broadening are implemented and default to off. Peculiar velocities are applied only if a displacement field is passed in. The command-line path does not generate velocities. The Kaiser formula is the redshift-space model used for \(P_{1\mathrm{D}}(k,z)\).

## Plots to show

| Plot | Point to make |
|---|---|
| `plots/01` | \(n_s\) tilts \(P_\mathrm{CDM}\), \(A_s\) rescales it. |
| `plots/03` | The \(A_s\) response of \(P_{1\mathrm{D}}\) is flat and equal to \(\Delta A_s/A_s\). |
| `plots/05` | The \(\alpha\beta\gamma\delta\) bend is zero at the pivot and zero when every coefficient is zero. |
| `plots/06` | An external \(P(k)\) table, after rescaling, matches Eisenstein–Hu. |
| `plots/07` | The pipeline from \(P_\mathrm{CDM}\) to a flux skewer and to \(P_{1\mathrm{D}}\). |
| `plots/08` | Kaiser and optical-depth FGPA agree in shape at \(z=3\). |

## What is still outside this task

Survey noise, resolution, covariance, and the Fisher matrix are not here. \(P_x\) is not here. LaCE, LyaCoLoRe, cup1d, P1D_Mocks, and qsotools are not dependencies. LaCE is the emulator comparison to do next. LyaCoLoRe is the reference if the skewer FGPA is later replaced by a 3D density field.
