# Scientific references

The code implements a deliberately minimal combination of standard
approximations. It does not copy source code from the projects below.

## FGPA and IGM physics

- Kooistra, Lee & Horowitz, *Constraining the Fluctuating Gunn-Peterson
  Approximation Using Lyman-alpha Forest Tomography at z=2*,
  [arXiv:2201.10169](https://arxiv.org/abs/2201.10169). In particular,
  `tau proportional to Delta^beta` with
  `beta = 2 - 0.7 (gamma - 1)`.
- Arya et al., *Lognormal semi-numerical simulations of the Lyman-alpha
  forest: comparison with full hydrodynamic simulations*,
  [arXiv:2206.08013](https://arxiv.org/abs/2206.08013). This describes the
  Gaussian-to-lognormal density mapping, photoionization equilibrium,
  temperature-density relation, thermal broadening, and transmitted flux.
- Bi & Davidsen, *Evolution of Structure in the Intergalactic Medium and the
  Nature of the Lyman-alpha Forest*,
  [arXiv:astro-ph/9611062](https://arxiv.org/abs/astro-ph/9611062).

## Related public software

- [LyaCoLoRe](https://github.com/igmhub/LyaCoLoRe): survey-scale correlated
  Ly-alpha mocks based on CoLoRe outputs.
- [P1D_Mocks](https://github.com/m-herbold/P1D_Mocks): fits and generates
  one-dimensional lognormal flux mocks from a target flux P1D.
- [qsotools](https://github.com/p-slash/qsotools): Ly-alpha mock generation,
  spectrum operations, and P1D analysis utilities.
- [fake_spectra](https://github.com/sbird/fake_spectra): synthetic spectra
  from hydrodynamical simulation snapshots.

## Scope warning

The current FGPA normalization is calibrated to a requested mean transmitted
flux. It is not yet calculated from `T0`, the hydrogen photoionization rate,
atomic constants, and the cosmological expansion rate. Thermal smoothing is a
fixed Gaussian kernel, not a spatially varying Voigt-profile calculation.

