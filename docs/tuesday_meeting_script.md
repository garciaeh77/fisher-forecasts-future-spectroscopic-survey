# Speaker script — Tuesday 13 October 2026

About 15 minutes, then the questions. The slides are `docs/tuesday_meeting.pdf`. Say the sentences. Do not read the formulae off the slide unless someone asks.

Open the PDF in presentation mode. The plots are already on the slides.

---

## Slide 1 — Title

"This is the first piece of the forecast: an analytic one-dimensional Lyman-alpha power spectrum. I will show how it is built, where the fluctuating Gunn-Peterson approximation sits, what the plots are checking, and the four decisions I need before the cross-power, the covariance, and the Fisher matrix."

## Slide 2 — The task

"The task was a calculator. The inputs are n_s, A_s, and a cold-dark-matter spectrum. The output is P_1D of k and z. k is in seconds per kilometre, and P_1D is in kilometres per second, which is how the forest is measured. The cold-dark-matter spectrum can be Eisenstein-Hu, a table from a file, or a four-parameter bend on top of Eisenstein-Hu. The cross-power P_x is not in this repository. It is the next stage, together with the covariance and the Fisher matrix."

## Slide 3 — What I built

"There are three steps. First I build P_CDM. Then I turn it into flux. Then I integrate along the line of sight. There are two flux models on purpose. The Kaiser formula is smooth in n_s and A_s, so it is the one I would differentiate in a Fisher forecast. The skewer calculation is the optical-depth formula, and it is there to check the shape."

## Slide 4 — Plot 07, the four stages

Point at each panel as you name it.

"This is the whole pipeline on one figure, at redshift 3. Top left is the Eisenstein-Hu cold-dark-matter spectrum. Top right is that spectrum integrated along the line of sight, still as matter, not yet as flux. Bottom left is the flux P_1D in kilometres per second. The smooth blue curve is the Kaiser formula. The orange curve is measured from skewers. Bottom right is one transmitted-flux skewer. The dashed line is the mean flux, which I set to 0.80. The amplitude A that produces that mean flux is 0.485."

## Slide 5 — Eisenstein-Hu

"I started from the matter spectrum, not from the flux. Eisenstein and Hu 1998 give a closed form for the transfer function, including the baryon wiggles. n_s and A_s enter as the primordial power law, multiplied by T of k squared and by the growth factor. The pivot is 0.05 per megaparsec. I compared this with CAMB for the same cosmology. The median ratio from 10 to the minus 3 to 10 h per megaparsec is about 1.01."

## Slide 6 — Plot 01

"This is that spectrum at redshift zero. The solid orange curve is the fiducial n_s of 0.965. Lowering n_s to 0.90 puts more power on large scales. Raising it to 1.02 does the opposite. The grey curves are the same shape moved up and down by changing A_s by 20 percent. If this plot were flat, or if n_s only rescaled it, the transfer function would be wrong."

## Slide 7 — Plot 02

"This is the same set of changes after the flux model and the line-of-sight integral, now in forest units at redshift 3. The point of the figure is that those n_s and A_s changes survive into the quantity the forest actually measures."

## Slide 8 — Kaiser FGPA

"The full fluctuating Gunn-Peterson approximation sets the optical depth proportional to the density to the power beta, and the transmitted flux to e to the minus tau. Beta comes from the temperature-density relation. I do not build that density field for the forecast function. For small fluctuations it becomes a biased tracer of the dark matter, with a Kaiser factor for peculiar velocities. b_F is minus 0.14 and beta_F is 1.4. Both are fixed. b_F is negative because a denser patch absorbs more and transmits less. This expression never exponentiates tau."

If asked why the biases are constant: "They set the amplitude. They do not yet carry redshift evolution or a fit to the mean flux. That is what the skewer path is for."

## Slide 9 — The integral

"A forest spectrum is one-dimensional, so I integrate out the perpendicular modes, from k_parallel out to 30 h per megaparsec. Then I convert with H of z over (1+z) times h. The input k is in seconds per kilometre. The output is in kilometres per second."

## Slide 10 — Plot 03

"This is the plot I trust most. A 1 percent increase in A_s must raise P_1D by 1 percent at every k. That is the orange line, sitting on the dashed line at 0.01. Raising n_s by 0.01 must tilt the spectrum. The blue curve goes from about 2 percent at 10 to the minus 3 per kilometre per second up to about 4 percent at 0.02. If the orange line sloped, the integral or the units would be wrong."

## Slide 11 — Plot 04

"Here the cosmology is fixed and only the redshift changes: 2.2, 3, and 4. The shape stays similar. The amplitude changes because the growth factor changes and because the conversion from comoving wavenumber to seconds per kilometre depends on redshift. b_F and beta_F are still the same numbers at every redshift. This plot does not yet include evolution of the flux bias."

## Slide 12 — A table of P(k)

"The task also asked for P_CDM as an input, not only as something the code computes. So a two-column file, k and P of k, can be read in. Changing A_s multiplies the table. Changing n_s tilts it around the pivot. Changing redshift multiplies by the square of the growth ratio. If you ask for a k outside the file, the code stops instead of extrapolating. That is how a CLASS or CAMB spectrum, or a sterile-neutrino spectrum, would enter later."

## Slide 13 — Plot 06

"Left: the fiducial table, A_s scaled by 1.2, and n_s raised by 0.03. Right: the rescaled table divided by a direct Eisenstein-Hu evaluation. The vertical axis is that fractional difference times one million. They agree. The wiggles in the residual are interpolation noise at the baryon features, at the level of 10 to the minus 6."

## Slide 14 — Alpha beta gamma delta

"The four-parameter shape is a smooth bend on top of Eisenstein-Hu. L is the log of k over the pivot, and the pivot is 1 h per megaparsec. At that pivot the exponential is 1, so the amplitude is still the Eisenstein-Hu amplitude. Setting all four numbers to zero gives the Eisenstein-Hu spectrum back exactly. This bend does not add a sharp small-scale cutoff."

## Slide 15 — Plot 05

"Each curve turns on one coefficient. On the matter spectrum they all cross zero at the pivot, the vertical dotted line. The flat line is every coefficient set to zero. The right panel is the same bend after the projection to P_1D. The integral smooths it, so it does not cross zero at the same place."

## Slide 16 — Optical-depth skewers

"The second implementation does use tau equals A times Delta to the beta. I draw Gaussian skewers from the matter P_1D, map them to a lognormal density, and solve for A so the mean transmitted flux is 0.80. With gamma equal to 1.5, beta is 1.65 and A is 0.485. Pressure smoothing and thermal broadening are written and switched off. Peculiar velocities are applied only if some other code supplies a displacement. The Kaiser formula is the redshift-space model. The skewers are the real-space optical depth plus the mean-flux calibration."

## Slide 17 — Plot 08

"Both curves are divided by their own value at 0.005 seconds per kilometre, so this is a shape comparison, not an amplitude comparison. They follow each other. The orange curve is noisy because it is measured from 256 finite skewers. On the previous figure, in absolute units, the skewer P_1D sits above the Kaiser curve. That is expected. One amplitude is the fixed b_F. The other is the mean flux."

## Slide 18 — What is not built

"So the deliverable of this stage is P_1D of k and z from n_s, A_s, and P_CDM. What I have not built is the cross-power, the covariance, and the Fisher matrix. I would take derivatives of the Kaiser P_1D, not of the skewers, because the skewers depend on the random seed."

## Slide 19 — Questions

Read them, then stop.

"Four decisions, and then I can write the next piece.

First, for P_x: is the cross with quasars, Lyman-break galaxies, or both, and is it the three-dimensional P of k, mu, and z?

Second, for the covariance: is the first matrix the one-dimensional P_1D covariance, or the three-dimensional form C equals 2 P_T squared over N_k, with aliasing and noise?

Third, which survey knobs enter that covariance first: area, quasar density, galaxy density, resolution, and signal-to-noise?

Fourth, for the Fisher matrix: are the free parameters only n_s and A_s, with b_F and beta_F held fixed? And should the first number be compared with the DESI-II BAO forecast, or with the emulator Fisher we already have?"

## Backup slide — other prototype

Only if someone asks what the other code was.

"This is the other prototype, stored unchanged in the repository. Those two figures use a smooth demonstration spectrum, not Eisenstein-Hu. I kept that code as it was, and the forecast pipeline is the combined version: Eisenstein-Hu, the Kaiser P_1D, the table, and the optical-depth check."
