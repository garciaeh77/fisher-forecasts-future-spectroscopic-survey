# Follow-up talk after 13 October

Spoken script for `docs/followup_talk.pptx`. About 12–15 minutes.

Open the notes pane in PowerPoint (**View → Notes**) if you prefer to read from the slides. The figures are already embedded.

**File.** `/Users/helenagescu/fisher-forecasts-future-spectroscopic-survey/docs/followup_talk.pptx`

---

## Slide 1 — Title

Thank you. On 13 October I presented a calculator for the one-dimensional Lyman-alpha flux power. It takes the same matter spectrum and returns the flux power in two different ways.

You asked for six extensions. Today I will summarise what was already in place, what you asked for, how I implemented each request, and what the new figures show.

I am not covering the cross-power, the covariance, or the Fisher matrix here. Those come later.

---

## Slide 2 — What was ready on 13 October

The task was a forward model. The inputs are \(n_s\), \(A_s\), and the linear cold-dark-matter spectrum \(P_\mathrm{CDM}\). The output is \(P_{1\mathrm{D}}(k_\parallel,z)\), with \(k_\parallel\) in s/km and \(P_{1\mathrm{D}}\) in km/s. Every evaluation is at one redshift. Both methods use the same \(P_\mathrm{CDM}\).

### Method 1 — Kaiser, analytic

This method never builds a density field. On large scales I treat the forest as a linearly biased tracer of matter, with coherent peculiar velocities. The three-dimensional flux power is

\[
P_F(k,\mu)=b_F^{2}\,(1+\beta_F\mu^{2})^{2}\,P_\mathrm{CDM}(k).
\]

Here \(\mu=k_\parallel/k\) is the cosine of the angle between the wavevector and the line of sight. I use \(b_F=-0.14\) and \(\beta_F=1.4\), held fixed at every redshift. Those numbers are a round form of the early DESI large-scale bias.

A forest spectrum only measures the line-of-sight wavenumber, so I integrate out the transverse modes:

\[
P_{1\mathrm{D}}(k_\parallel)
=
\int_{k_\parallel}^{k_\mathrm{max}}
\frac{k\,\mathrm{d}k}{2\pi}\,
P_F\!\left(k,\mu=\frac{k_\parallel}{k}\right),
\]

with \(k_\mathrm{max}=30\,h\,\mathrm{Mpc}^{-1}\). There is no random draw. The result is a smooth function of \(n_s\) and \(A_s\).

### Method 2 — optical-depth sightlines

This method does build a transmission field, along independent periodic lines.

First I project \(P_\mathrm{CDM}\) onto one line of sight. That gives the one-dimensional matter power. Then I draw a Gaussian density contrast \(\delta_G\) whose Fourier modes carry exactly that power. The zero mode is set to zero, so each line has mean contrast zero.

### What “lognormal density” means

A Gaussian contrast can go below \(-1\). Then \(1+\delta_G\) would be negative. Gas density cannot be negative, so I cannot feed \(\delta_G\) directly into an optical-depth formula.

I replace it by the lognormal map

\[
\Delta=\exp\!\left(\delta_G-\tfrac12\mathrm{Var}(\delta_G)\right).
\]

Three properties matter:

1. The exponential makes \(\Delta\) positive everywhere.
2. Subtracting half the variance keeps the spatial mean of \(\Delta\) near one.
3. Underdense pixels give \(0<\Delta<1\); overdense pixels give \(\Delta>1\).

\(\ln\Delta\) is Gaussian, which is why the field is called lognormal. That step is the only reason method 2 has a usable gas density.

From \(\Delta\) I set the optical depth to \(\tau=A\Delta^{1.65}\). The exponent \(1.65\) is the recombination factor for a temperature–density index \(\gamma=1.5\). Transmission is \(F=e^{-\tau}\). I solve for \(A\) so that the average of \(F\) over every pixel of every line equals \(0.80\). The flux power is the periodogram of \(F/\langle F\rangle-1\).

So method 1 is a formula. Method 2 is a realisation. The lognormal map is the bridge from a Gaussian contrast to a positive density.

---

## Slide 3 — Figure 07

This is the summary figure from the last meeting. Please look at the four panels in this order.

Top left is the three-dimensional matter spectrum. Top right is that spectrum projected onto one line of sight. There is no flux bias in that projection.

Bottom left is the flux power in absolute units. The smooth curve is Kaiser. The fluctuating curve is the average of 256 sightlines. They do not overlie. Bottom right is one transmitted-flux line. For this ensemble the mean transmission is \(0.80\) and the fitted amplitude is \(A=0.485\).

The offset has a clear origin. Kaiser is normalised by \(b_F^{2}\). The sightlines are normalised by the mean transmission. Kaiser includes peculiar velocities through \(\beta_F\). These sightlines are in real space, and the density variance on them is about \(19\), so \(F=e^{-\tau}\) is far from linear.

If I divide each curve by its own value at \(0.005\,\mathrm{s\,km}^{-1}\), the shapes agree. That is the backup slide at the end. More lines quiet the fluctuations. They do not remove the offset.

---

## Slide 4 — The six requests

These are the six requests from 13 October:

1. Extend the spectra to smaller scales, out to \(k\sim 0.2\,\mathrm{s\,km}^{-1}\) at \(z\sim 5\).
2. Use more sightlines, and check whether the fluctuations in the October figure shrink.
3. Build a three-dimensional realisation, so that \(P_F(k,\mu)\) is measured rather than assumed.
4. Compare that measurement with the nonlinear model of Arinyo-i-Prats et al. (2015).
5. Apply a Gaussian pressure filter to the gas.
6. Use a lognormal density for weakly nonlinear dark matter.

I will take them in the order that matches the code. The pressure filter and the lognormal comparison come first, because both flux paths use them. Then the ensemble test, then redshift 5, then the cube and the Arinyo-i-Prats comparison.

One organising choice: I did not rewrite the October functions. Each extension is a new call placed in front of them, or a new module beside them.

---

## Slide 5 — How the extensions sit in the code

The two October entry points are unchanged. If I call them with no extra argument, I recover the October calculation: no Jeans cutoff, mean transmission \(0.80\), real-space sightlines.

The pressure filter is one function, `with_jeans`. It copies the wavenumber grid and the power, multiplies the power by \(\exp(-2 x_J^{2} k^{2})\), and returns a new table. The original table is left untouched. I then pass that table into the sightlines, or pass it as the matter spectrum inside the Kaiser integral. A length of \(100\,\mathrm{kpc}/h\) is \(x_J=0.10\,\mathrm{Mpc}/h\). Fifty kiloparsecs removes less power.

The cube is a separate module. It does not modify the one-dimensional lines. The velocity is linear theory from the continuity equation, and the optical depth is shifted along one axis by cloud-in-cell.

The Arinyo-i-Prats formula is also a separate module. The parameters are taken from their fiducial simulation at \(z=3\): \(q_1=0.648\), \(k_p=13.1\,h\,\mathrm{Mpc}^{-1}\), \(\beta=1.205\), and a transmission bias of \(-0.201\). I have not refitted them.

---

## Slide 6 — Figure 09: the Jeans factor

The pressure model is the Jeans factor of Arya and collaborators. \(P_\mathrm{CDM}\) is multiplied by \(\exp(-2 x_J^{2} k^{2})\). I use two comoving lengths, \(100\) and \(50\,\mathrm{kpc}/h\). Because the exponent scales as \(x_J^{2}\), \(100\,\mathrm{kpc}/h\) removes more power than \(50\).

On the left, all three matter spectra agree from the largest scales through about \(1\,h\,\mathrm{Mpc}^{-1}\). After that, \(100\,\mathrm{kpc}/h\) falls first, then \(50\).

On the right I evaluate Kaiser's formula at \(\mu=0\) and \(\mu=1\). The line-of-sight curve is higher by \((1+1.4)^{2}=5.76\). The dashed curve puts the \(100\,\mathrm{kpc}/h\) factor on the \(\mu=1\) line. It leaves the large-scale amplitude alone and cuts the small-scale end in the same place as the matter spectrum.

Please treat this right-hand panel as the formula. The measurement in a volume is the cube, later.

---

## Slide 7 — Figure 10: Gaussian versus lognormal

The lognormal map was already the density used for the optical depth. This figure is the comparison you asked for. The lines are short, \(L=40\,\mathrm{Mpc}/h\) with \(1024\) pixels and \(512\) lines, so the Nyquist mode sits past the cutoff.

The lower curve in each panel is the Gaussian contrast. The upper curve is \(\Delta-1\) from the lognormal map.

With no Jeans factor the Gaussian variance is \(6.1\), and \(34\%\) of the pixels have \(1+\delta\le 0\). That field cannot be raised to a power and used as a gas density. The lognormal field stays positive, and its variance is \(175\). That is strongly nonlinear.

At \(100\,\mathrm{kpc}/h\) the variances fall to \(1.28\) and \(2.42\). That is the weakly nonlinear regime. At \(50\,\mathrm{kpc}/h\) they rise again, to \(2.0\) and \(5.6\), because a shorter length removes less power.

In every panel the lognormal spectrum stays above the Gaussian one at high \(k\). The exponential puts small-scale power back after the cutoff has removed it.

---

## Slide 8 — Figure 11: flux with the same cutoff

Same short lines, now as flux power in km/s. The mean transmission is held at \(0.80\).

The left panel is the Kaiser integral of the screened matter spectrum. The right panel is the sightline measurement.

Below about \(0.01\,\mathrm{s\,km}^{-1}\), changing the Jeans length does nothing visible. Above that, \(100\,\mathrm{kpc}/h\) falls first in both panels, and \(50\,\mathrm{kpc}/h\) falls later. The sightlines fall less steeply than Kaiser. That is the same regeneration of small-scale power, passed through \(F=e^{-\tau}\).

The fitted amplitudes on these short lines are \(A=2.41\) with no cutoff, \(0.26\) at \(100\,\mathrm{kpc}/h\), and \(0.33\) at \(50\,\mathrm{kpc}/h\). The October figure had \(A=0.485\), on lines whose Nyquist mode was only about \(8\,h\,\mathrm{Mpc}^{-1}\). These lines resolve the power the cutoff is meant to remove, so the amplitude that restores \(\langle F\rangle=0.80\) changes.

---

## Slide 9 — Figure 12: more sightlines

The fluctuating curve on 13 October was the average of \(256\) lines. It was not \(500\).

I drew \(10000\) lines in that same geometry: \(200\,\mathrm{Mpc}/h\), \(512\) pixels, no Jeans cutoff, mean transmission \(0.80\). The pale curve is the first \(256\). The dashed curve is the first \(1024\). The dotted curve is the first \(4096\). The solid curve is all \(10000\). The smooth curve is Kaiser.

Over \(10^{-3}\) to \(0.02\,\mathrm{s\,km}^{-1}\), the root-mean-square fractional difference from the \(10000\)-line spectrum is \(0.067\) at \(256\) lines, \(0.030\) at \(1024\) lines, and \(0.012\) at \(4096\) lines. The ratio of the \(256\)-line and \(1024\)-line numbers is \(2.28\), against \(2\) for noise that scales as \(1/\sqrt{N}\). The ratio of the \(256\)-line and \(4096\)-line numbers is \(5.47\), against \(4\).

Thirty-nine separate blocks of \(256\) lines have a median fractional scatter of \(0.061\), the same size.

So the fluctuations shrink as expected when I add lines. The vertical offset from Kaiser is still present at \(10000\) lines. That offset is the difference between a mean transmission of \(0.80\), in real space, and a bias of \(-0.14\).

---

## Slide 10 — Figure 13: redshift 5

At redshift \(5\), in this cosmology, \(0.2\,\mathrm{s\,km}^{-1}\) is \(27.4\,h\,\mathrm{Mpc}^{-1}\). These lines are \(40\,\mathrm{Mpc}/h\) with \(512\) pixels, so the Nyquist mode is \(40\,h\,\mathrm{Mpc}^{-1}\). The plotted range stops at \(0.2\). The Kaiser integral is carried to \(80\,h\,\mathrm{Mpc}^{-1}\), past the old cutoff of \(30\), so the upper limit is not sitting on the scale we care about.

The mean transmission is \(0.132\). I took it from the Faucher-Giguère power law, \(\tau_{\mathrm{eff}}=0.0018(1+z)^{3.92}\), with metals left in. That fit was measured from redshift \(2\) to \(4.2\). At redshift \(5\) it is an extrapolation, and I am treating it as an input, not as a measurement.

With \(100\,\mathrm{kpc}/h\) the lognormal variance is \(0.78\), so the density is weakly nonlinear. With no cutoff the variance is \(6.7\) and the fitted amplitude is \(A=74\), because the target transmission is low. At \(100\,\mathrm{kpc}/h\), \(A=6.3\). At \(50\,\mathrm{kpc}/h\), \(A=10.7\).

The Kaiser panel has the same pattern as at redshift \(3\): the curves agree on large scales, and \(100\,\mathrm{kpc}/h\) falls first. The sightline panel is much higher at every \(k\). Its amplitude is fixed by a mean transmission of \(0.13\). Kaiser's amplitude is still fixed by a bias of \(-0.14\).

---

## Slide 11 — Figure 14: one cube

This is the three-dimensional measurement. One periodic cube, \(40\,\mathrm{Mpc}/h\), \(128^{3}\) cells, redshift \(3\), Jeans length \(100\,\mathrm{kpc}/h\). Each Fourier mode is given the rms of \(P(k)\), with a random phase. That is the same choice Arinyo-i-Prats made for their initial conditions, so one box is not limited by a Rayleigh draw.

The Gaussian variance is \(1.38\). The lognormal variance is \(2.85\). The mean transmission is \(0.696\), which is their value at redshift \(3\), rather than \(0.80\). The line-of-sight velocity is linear theory, with growth rate \(f=0.98\). The rms shift is about \(1\,\mathrm{Mpc}/h\), three cells.

Before trusting the optical depth, I built a field that is exactly \(b_F(1+\beta_F\mu^{2})\delta(\mathbf{k})\), using \(-0.14\) and \(1.4\). The estimator recovers that power with a maximum fractional residual of \(10^{-13}\). So the binning is doing what it should.

On the figure, the solid blue curve is our Kaiser formula on the screened spectrum. The green dashed curve is their published formula, not a refit, evaluated on our linear \(P_\mathrm{CDM}\). The dotted curve is the lognormal flux in real space. The red curve is the same flux after the velocity shift.

At \(k=1.16\,h\,\mathrm{Mpc}^{-1}\), the redshift-space measurement divided by their formula is \(2.0\) for \(\mu<0.25\), and \(1.3\) for \(\mu>0.75\). The parallel bin lies above the real-space curve, and the transverse bin lies below it. That is the sense of the Kaiser factor. The slope follows their formula. The amplitude does not. This cube is a lognormal field with an explicit \(100\,\mathrm{kpc}/h\) cutoff. Their formula was calibrated on a hydrodynamic simulation, with a different cosmology, and their pressure scale is inside \(D_1\) rather than applied to \(P_\mathrm{CDM}\) beforehand.

---

## Slide 12 — Conclusions

Five statements.

1. Nothing from 13 October was replaced. The new cutoff is a table I can pass in.
2. At \(100\,\mathrm{kpc}/h\) the lognormal density in these volumes is weakly nonlinear. At \(50\,\mathrm{kpc}/h\) the cutoff is weaker. Without a cutoff, the Gaussian field is not a usable gas density.
3. The fluctuations were sample variance. They fall as \(1/\sqrt{N}\). The offset from the blue curve is the flux model.
4. Redshift \(5\) now reaches the wavenumber you asked for. The price of a realistic mean transmission is that the sightline power sits well above the Kaiser curve with \(b_F=-0.14\).
5. The cube measures an anisotropic \(P(k,\mu)\). The anisotropy has the Kaiser sign. The shape is close to the published nonlinear formula. The amplitude is higher, and I have not adjusted their parameters to remove that.

I am happy to go to the backup figure, or to the limits of the cube.

---

## Slide 13 — Limits

These are the limits I want on the table before questions.

1. The cube is a single \(40\,\mathrm{Mpc}/h\) volume. The Nyquist mode is about \(10\,h\,\mathrm{Mpc}^{-1}\), and the pressure scale in their formula is \(13\,h\,\mathrm{Mpc}^{-1}\), so I am comparing shapes on the scales the grid actually resolves.
2. The velocities are the linear continuity field. I have not put in thermal broadening. The old Gaussian smoother in pixel units is still in the sightline code, and every figure here leaves it at zero. The smoothing you see is the Jeans factor on the matter spectrum.
3. The redshift-\(5\) transmission is an extrapolation past redshift \(4.2\). If we would rather hold the mean transmission at \(0.80\), so that redshift \(5\) isolates the growth and the unit conversion, I can remake that figure. I chose \(0.13\) because a transmission of \(0.80\) is not a \(z=5\) forest.
4. Their formula is used exactly as published for the fiducial simulation at redshift \(3\). I have not moved \(q_1\) or the biases to improve the match.
5. The forecast itself — cross-power, covariance, Fisher — is the next piece of work, not this one.

---

## Slide 14 — Backup: figure 08

I will show this only if someone asks whether the two October curves have the same shape.

Each flux spectrum is divided by its value at \(0.005\,\mathrm{s\,km}^{-1}\). Across the forest window the shapes agree. The residual wiggles on the sightline curve are the \(256\)-line scatter from the earlier slide.

Agreement of the amplitudes, on large scales and in real space, would also require the flux bias to equal \(-A\beta\), with \(\beta_F\) set to zero. Matching the angular factor \((1+\beta_F\mu^{2})^{2}\) needs velocities that know about modes perpendicular to the line of sight. Independent one-dimensional lines do not contain those modes. The cube does.
