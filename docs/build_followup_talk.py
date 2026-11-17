#!/usr/bin/env python3
"""Build the follow-up talk and a printable speaker script.

Writes docs/followup_talk.pptx and docs/followup_talk_script.md.
Equations are rendered with matplotlib mathtext (LaTeX syntax) to PNGs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
DOCS = Path(__file__).resolve().parent
EQ_DIR = DOCS / "_eq_cache"

NAVY = RGBColor(0x1B, 0x3A, 0x4B)
TEAL = RGBColor(0x1F, 0x6F, 0x8B)
INK = RGBColor(0x1C, 0x1C, 0x1C)
MUTED = RGBColor(0x3E, 0x4C, 0x57)
RULE = RGBColor(0xD5, 0xDE, 0xE6)
INK_MPL = "#1C1C1C"
TEAL_MPL = "#1F6F8B"

W, H = Inches(13.333), Inches(7.5)


def _set_run(run, size, color, bold=False, italic=False, font="Calibri"):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font


def _box(slide, left, top, width, height):
    shape = slide.shapes.add_textbox(left, top, width, height)
    frame = shape.text_frame
    frame.word_wrap = True
    frame.auto_size = None
    return frame


def _para(frame, text, size, color, bold=False, italic=False, align=PP_ALIGN.LEFT, space_before=0, space_after=0, first=False):
    p = frame.paragraphs[0] if first else frame.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    run = p.add_run()
    run.text = text
    _set_run(run, size, color, bold, italic)
    return p


def _bar(slide):
    shape = slide.shapes.add_shape(1, Inches(0), Inches(0), W, Inches(0.08))  # rectangle
    shape.fill.solid()
    shape.fill.fore_color.rgb = NAVY
    shape.line.fill.background()


def _footer(slide, number, total):
    frame = _box(slide, Inches(0.55), Inches(7.12), Inches(12.2), Inches(0.28))
    _para(frame, "Lyman-α  P1D    ·    follow-up to 13 October", 12, MUTED, first=True)
    frame2 = _box(slide, Inches(10.4), Inches(7.12), Inches(2.4), Inches(0.28))
    _para(frame2, f"{number}   /   {total}", 12, MUTED, align=PP_ALIGN.RIGHT, first=True)


def _notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def _title(slide, text):
    frame = _box(slide, Inches(0.55), Inches(0.22), Inches(12.2), Inches(0.62))
    _para(frame, text, 28, NAVY, bold=True, first=True)


def _kicker(slide, text):
    frame = _box(slide, Inches(0.55), Inches(0.82), Inches(12.2), Inches(0.42))
    _para(frame, text, 16, TEAL, italic=True, first=True)


def _render_eq(name: str, latex: str, fontsize: float = 18, color: str = INK_MPL, dpi: int = 220) -> Path:
    """Render a mathtext (LaTeX-syntax) expression to a transparent PNG."""
    EQ_DIR.mkdir(parents=True, exist_ok=True)
    path = EQ_DIR / f"{name}.png"
    expr = latex if latex.startswith("$") else f"${latex}$"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    text = ax.text(0.5, 0.5, expr, fontsize=fontsize, ha="center", va="center", color=color)
    fig.canvas.draw()
    bbox = text.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    pad = 0.04
    fig.set_size_inches(max(bbox.width + 2 * pad, 0.2), max(bbox.height + 2 * pad, 0.15))
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, expr, fontsize=fontsize, ha="center", va="center", color=color)
    fig.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def _eq(slide, name: str, latex: str, left, top, max_w, fontsize: float = 17, color: str = INK_MPL, align: str = "left"):
    """Place a rendered equation image inside a bounding box."""
    path = _render_eq(name, latex, fontsize=fontsize, color=color)
    with Image.open(path) as im:
        px_w, px_h = im.size
    # Convert pixel size at render dpi to inches, then cap by max_w.
    dpi = 220
    nat_w = Inches(px_w / dpi)
    nat_h = Inches(px_h / dpi)
    if nat_w > max_w:
        scale = max_w / nat_w
        nat_w = max_w
        nat_h = Emu(int(nat_h * scale))
    if align == "center":
        left = left + (max_w - nat_w) // 2
    elif align == "right":
        left = left + (max_w - nat_w)
    slide.shapes.add_picture(str(path), left, top, nat_w, nat_h)
    return nat_h


def _picture(slide, name, left, top, max_w, max_h):
    path = PLOTS / name
    # Fit inside the box, preserving aspect, using the file's pixel size.
    with Image.open(path) as im:
        aspect = im.size[0] / im.size[1]
    if max_w / aspect <= max_h:
        width, height = max_w, Emu(int(max_w / aspect))
    else:
        height, width = max_h, Emu(int(max_h * aspect))
    left = left + (max_w - width) // 2
    top = top + (max_h - height) // 2
    slide.shapes.add_picture(str(path), left, top, width, height)


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    prs.core_properties.title = "Lyman-alpha P1D after 13 October"
    prs.core_properties.author = "Helena"
    prs.core_properties.subject = "Follow-up to the 13 October meeting"
    blank = prs.slide_layouts[6]

    slides = []

    def new():
        slide = prs.slides.add_slide(blank)
        _bar(slide)
        slides.append(slide)
        return slide

    # 1. Title
    s = new()
    frame = _box(s, Inches(0.7), Inches(1.9), Inches(11.8), Inches(1.6))
    _para(frame, "Lyman-α forest  P1D", 40, NAVY, bold=True, first=True)
    _para(frame, "What was ready on 13 October, and what I have done with the requests", 22, INK, space_before=10)
    frame = _box(s, Inches(0.7), Inches(4.3), Inches(11), Inches(1.2))
    _para(frame, "Helena", 20, TEAL, first=True)
    _para(frame, "Analytic Kaiser projection  ·  optical-depth sightlines  ·  one three-dimensional box", 16, MUTED, space_before=8)
    _notes(s, """
Thank you. On 13 October I presented a calculator for the one-dimensional Lyman-alpha flux power. It takes the same matter spectrum and returns the flux power in two different ways.

You asked for six extensions. Today I will summarise what was already in place, what you asked for, how I implemented each request, and what the new figures show.

I am not covering the cross-power, the covariance, or the Fisher matrix here. Those come later.
""")

    # 2. What was ready
    s = new()
    _title(s, "What was ready on 13 October")
    _kicker(s, "Same P_CDM. Method 1 is an analytic formula. Method 2 builds transmission along lines.")
    left = _box(s, Inches(0.5), Inches(1.35), Inches(6.0), Inches(5.4))
    _para(left, "Method 1 — Kaiser, analytic", 18, TEAL, bold=True, first=True)
    for line in [
        "Treat the forest as a linearly biased tracer of the matter field, with coherent peculiar velocities.",
        "P_F(k, μ) = b_F² (1 + β_F μ²)² P_CDM(k).",
        "b_F = −0.14 and β_F = 1.4, fixed. Round form of early DESI.",
        "A forest spectrum sees only k_∥. Integrate transverse modes:",
        "P_1D(k_∥) = ∫_{k_∥}^{k_max} (k dk / 2π) P_F(k, μ = k_∥/k).",
        "No random draw. Smooth in n_s and A_s.",
    ]:
        _para(left, "•  " + line, 15, INK, space_before=7)
    right = _box(s, Inches(6.7), Inches(1.35), Inches(6.1), Inches(5.4))
    _para(right, "Method 2 — optical-depth sightlines", 18, TEAL, bold=True, first=True)
    for line in [
        "Project P_CDM onto one line: P_1D,matter(k_∥).",
        "Draw Gaussian contrast δ_G with that 1D power.",
        "Lognormal density: Δ = exp(δ_G − ½ Var(δ_G)).",
        "Δ is always positive. Mean of Δ is near one.",
        "Optical depth τ = A Δ^1.65, then F = e^{−τ}.",
        "Solve A so ⟨F⟩ = 0.80. Measure P_1D of F/⟨F⟩ − 1.",
    ]:
        _para(right, "•  " + line, 15, INK, space_before=7)
    _notes(s, """
The task was a forward model. The inputs are n_s, A_s, and the linear cold-dark-matter spectrum. The output is the one-dimensional flux power, with wavenumber in s/km and power in km/s. Every evaluation is at one redshift. Both methods use the same matter spectrum.

Method 1 is analytic. It never builds a density field. On large scales I treat the forest as a linearly biased tracer of matter, with coherent peculiar velocities. Kaiser's formula multiplies the matter spectrum by b_F squared times (1 + beta_F mu squared) squared. Mu is the cosine of the angle between the wavevector and the line of sight. I use b_F equals minus 0.14 and beta_F equals 1.4, held fixed. Those are a round form of the early DESI large-scale bias. A forest spectrum only measures the line-of-sight wavenumber, so I integrate out the transverse modes, from k_parallel up to 30 h per megaparsec. There is no random draw. The result is a smooth function of n_s and A_s.

Method 2 builds a transmission field along independent periodic lines. First I project the matter spectrum onto one line of sight. That gives the one-dimensional matter power. Then I draw a Gaussian density contrast whose Fourier modes carry exactly that power. The zero mode is set to zero, so each line has mean contrast zero.

What I mean by lognormal density is the next map. A Gaussian contrast can go below minus one. Then one plus delta would be negative. Gas density cannot be negative, so I cannot feed that contrast directly into an optical-depth formula. I replace it by Delta equals exp of (delta_G minus half the variance of delta_G). That is the lognormal map. Three properties matter. The exponential makes Delta positive everywhere. Subtracting half the variance keeps the spatial mean of Delta near one. Underdense pixels give values between zero and one; overdense pixels give values greater than one. The logarithm of Delta is Gaussian, which is why the field is called lognormal. That step is the only reason method 2 has a usable gas density.

From Delta I set the optical depth to A times Delta to the power 1.65. The exponent 1.65 is the recombination factor for a temperature-density index of 1.5. Transmission is e to the minus tau. I solve for A so that the average of F over every pixel of every line equals 0.80. The flux power is the periodogram of F over mean F, minus one.

So method 1 is a formula. Method 2 is a realisation. The lognormal map is the bridge from a Gaussian contrast to a positive density.
""")

    # 3. Figure 07
    s = new()
    _title(s, "The figure from 13 October")
    _kicker(s, "Kaiser and the sightlines share P_CDM. Their flux amplitudes answer different conditions.")
    _picture(s, "07_pipeline_stages.png", Inches(1.6), Inches(1.4), Inches(10.1), Inches(5.5))
    _notes(s, """
This is the summary figure from the last meeting. Please look at the four panels in this order.

Top left is the three-dimensional matter spectrum. Top right is that spectrum projected onto one line of sight. There is no flux bias in that projection.

Bottom left is the flux power in absolute units. The smooth curve is Kaiser. The fluctuating curve is the average of 256 sightlines. They do not overlie. Bottom right is one transmitted-flux line. For this ensemble the mean transmission is 0.80 and the fitted amplitude A is 0.485.

The offset has a clear origin. Kaiser is normalised by b_F squared. The sightlines are normalised by the mean transmission. Kaiser includes peculiar velocities through beta_F. These sightlines are in real space, and the density variance on them is about 19, so F equals e to the minus tau is far from linear.

If I divide each curve by its own value at 0.005 s/km, the shapes agree. That is the backup slide at the end. More lines quiet the fluctuations. They do not remove the offset.
""")

    # 4. Requests
    s = new()
    _title(s, "What you asked for on 13 October")
    frame = _box(s, Inches(0.7), Inches(1.35), Inches(12.0), Inches(5.4))
    requests = [
        "Extend the spectra to smaller scales, out to k ∼ 0.2 s/km at z ∼ 5.",
        "Use more sightlines, and test whether the fluctuations in the October figure shrink.",
        "Build a three-dimensional realisation, so that P_F(k, μ) is measured rather than assumed.",
        "Compare that measurement with the nonlinear model of Arinyo-i-Prats et al. (2015).",
        "Apply a Gaussian pressure filter to the gas.",
        "Use a lognormal density for weakly nonlinear dark matter.",
    ]
    for i, text in enumerate(requests, start=1):
        _para(frame, f"{i}     {text}", 22, INK, space_before=0 if i == 1 else 14, first=(i == 1))
    _notes(s, """
These are the six requests from 13 October.

One: extend the spectra to smaller scales, out to 0.2 s/km at redshift about 5.
Two: use more sightlines, and check whether the fluctuations in the October figure shrink.
Three: build a three-dimensional realisation, so that P_F of k and mu is measured rather than assumed.
Four: compare that measurement with the nonlinear model of Arinyo-i-Prats and collaborators from 2015.
Five: apply a Gaussian pressure filter to the gas.
Six: use a lognormal density for weakly nonlinear dark matter.

I will take them in the order that matches the code. The pressure filter and the lognormal comparison come first, because both flux paths use them. Then the ensemble test, then redshift 5, then the cube and the Arinyo-i-Prats comparison.

One organising choice: I did not rewrite the October functions. Each extension is a new call placed in front of them, or a new module beside them.
""")

    # 5. Code
    s = new()
    _title(s, "How the extensions sit in the code")
    _kicker(s, "The October functions still run. The new physics is passed in, or measured beside them.")
    frame = _box(s, Inches(0.55), Inches(1.5), Inches(12.2), Inches(5.2))
    rows = [
        ("Unchanged", "p1d_skm is still the Kaiser integral. run_fgpa_skewers is still the optical-depth lines. Both default to an unsmoothed spectrum."),
        ("Pressure", "with_jeans multiplies a stored P_CDM by exp(−2 x_J² k²) and returns a new table. The original table is untouched. 100 kpc/h is x_J = 0.10 Mpc/h. 50 kpc/h removes less power."),
        ("Cube", "gaussian_box draws one periodic volume. fgpa_flux_box applies the lognormal map, the optical depth, and a linear line-of-sight velocity. bin_power measures P(k, μ)."),
        ("Published formula", "The Arinyo-i-Prats factor D_1(k, μ) is coded with the fiducial-simulation numbers at z = 3 from their Tables 2 and 3. I have not refitted them."),
    ]
    first = True
    for head, body in rows:
        _para(frame, head, 18, TEAL, bold=True, space_before=0 if first else 12, first=first)
        _para(frame, body, 18, INK, space_before=2)
        first = False
    _notes(s, """
The two October entry points are unchanged. If I call them with no extra argument, I recover the October calculation: no Jeans cutoff, mean transmission 0.80, real-space sightlines.

The pressure filter is one function, with_jeans. It copies the wavenumber grid and the power, multiplies the power by the exponential of minus 2 x_J squared k squared, and returns a new table. The original table is left untouched. I then pass that table into the sightlines, or pass it as the matter spectrum inside the Kaiser integral. A length of 100 kpc per h is x_J equals 0.10 Mpc per h. Fifty kiloparsecs removes less power.

The cube is a separate module. It does not modify the one-dimensional lines. The velocity is linear theory from the continuity equation, and the optical depth is shifted along one axis by cloud-in-cell.

The Arinyo-i-Prats formula is also a separate module. The parameters are taken from their fiducial simulation at redshift 3: q_1 equals 0.648, k_p equals 13.1 h per megaparsec, beta equals 1.205, and a transmission bias of minus 0.201. I have not refitted them.
""")

    # 6. Figure 09
    s = new()
    _title(s, "Pressure: a Jeans factor on P_CDM")
    _kicker(s, "100 kpc/h falls first. The right-hand panel is the Kaiser formula, evaluated on a grid.")
    _picture(s, "09_3d_power_spectrum.png", Inches(0.45), Inches(1.4), Inches(12.4), Inches(5.5))
    _notes(s, """
The pressure model is the Jeans factor of Arya and collaborators. The matter spectrum is multiplied by the exponential of minus 2 x_J squared k squared. I use two comoving lengths, 100 and 50 kpc per h. Because the exponent scales as x_J squared, 100 kpc removes more power than 50.

On the left, all three matter spectra agree from the largest scales through about 1 h per megaparsec. After that, 100 kpc falls first, then 50.

On the right I evaluate Kaiser's formula at mu equals 0 and mu equals 1. The line-of-sight curve is higher by (1 + 1.4) squared, which is 5.76. The dashed curve puts the 100 kpc factor on the mu equals 1 line. It leaves the large-scale amplitude alone and cuts the small-scale end in the same place as the matter spectrum.

Please treat this right-hand panel as the formula. The measurement in a volume is the cube, later.
""")

    # 7. Figure 10
    s = new()
    _title(s, "Lognormal density, against the Gaussian field")
    _kicker(s, "At 100 kpc/h the variance is of order one. Without a cutoff, a third of the Gaussian pixels are negative.")
    _picture(s, "10_gaussian_vs_lognormal.png", Inches(0.35), Inches(1.4), Inches(12.6), Inches(5.5))
    _notes(s, """
The lognormal map was already the density used for the optical depth. This figure is the comparison you asked for. The lines are short: 40 Mpc per h, 1024 pixels, 512 lines, so the Nyquist mode sits past the cutoff.

The lower curve in each panel is the Gaussian contrast. The upper curve is Delta minus 1 from the lognormal map.

With no Jeans factor the Gaussian variance is 6.1, and 34 percent of the pixels have 1 plus delta less than or equal to zero. That field cannot be raised to a power and used as a gas density. The lognormal field stays positive, and its variance is 175. That is strongly nonlinear.

At 100 kpc per h the variances fall to 1.28 and 2.42. That is the weakly nonlinear regime. At 50 kpc they rise again, to 2.0 and 5.6, because a shorter length removes less power.

In every panel the lognormal spectrum stays above the Gaussian one at high k. The exponential puts small-scale power back after the cutoff has removed it.
""")

    # 8. Figure 11
    s = new()
    _title(s, "The same cutoff, in the flux power at z = 3")
    _kicker(s, "The two calculations agree below 0.01 s/km. Above that, the sightlines fall less steeply than Kaiser.")
    _picture(s, "11_flux_jeans_scales.png", Inches(0.45), Inches(1.4), Inches(12.4), Inches(5.5))
    _notes(s, """
Same short lines, now as flux power in kilometres per second. The mean transmission is held at 0.80.

The left panel is the Kaiser integral of the screened matter spectrum. The right panel is the sightline measurement.

Below about 0.01 s/km, changing the Jeans length does nothing visible. Above that, 100 kpc falls first in both panels, and 50 kpc falls later. The sightlines fall less steeply than Kaiser. That is the same regeneration of small-scale power, passed through F equals e to the minus tau.

The fitted amplitudes on these short lines are A equals 2.41 with no cutoff, 0.26 at 100 kpc, and 0.33 at 50. The October figure had A equals 0.485, on lines whose Nyquist mode was only about 8 h per megaparsec. These lines resolve the power the cutoff is meant to remove, so the amplitude that restores a mean transmission of 0.80 changes.
""")

    # 9. Figure 12
    s = new()
    _title(s, "The fluctuations were the size of the ensemble")
    _kicker(s, "From 256 to 4096 lines the scatter falls by about 5. The offset from Kaiser remains at 10000 lines.")
    _picture(s, "12_skewer_noise.png", Inches(2.5), Inches(1.4), Inches(8.3), Inches(5.5))
    _notes(s, """
The fluctuating curve on 13 October was the average of 256 lines. It was not 500.

I drew 10000 lines in that same geometry: 200 Mpc per h, 512 pixels, no Jeans cutoff, mean transmission 0.80. The pale curve is the first 256. The dashed curve is the first 1024. The dotted curve is the first 4096. The solid curve is all 10000. The smooth curve is Kaiser.

Over 10 to the minus 3 to 0.02 s/km, the root-mean-square fractional difference from the 10000-line spectrum is 0.067 at 256 lines, 0.030 at 1024 lines, and 0.012 at 4096 lines. The ratio of the 256-line and 1024-line numbers is 2.28, against 2 for noise that scales as one over square root of N. The ratio of the 256-line and 4096-line numbers is 5.47, against 4.

Thirty-nine separate blocks of 256 lines have a median fractional scatter of 0.061, the same size.

So the fluctuations shrink as expected when I add lines. The vertical offset from Kaiser is still present at 10000 lines. That offset is the difference between a mean transmission of 0.80, in real space, and a bias of minus 0.14.
""")

    # 10. Figure 13
    s = new()
    _title(s, "Redshift 5, out to 0.2 s/km")
    _kicker(s, "Nyquist is past the request. The mean transmission, 0.13, is an extrapolation of the z ≤ 4.2 fit.")
    _picture(s, "13_z5_smallscale_flux.png", Inches(0.4), Inches(1.4), Inches(12.5), Inches(5.5))
    _notes(s, """
At redshift 5, in this cosmology, 0.2 s/km is 27.4 h per megaparsec. These lines are 40 Mpc per h with 512 pixels, so the Nyquist mode is 40 h per megaparsec. The plotted range stops at 0.2. The Kaiser integral is carried to 80 h per megaparsec, past the old cutoff of 30, so the upper limit is not sitting on the scale we care about.

The mean transmission is 0.132. I took it from the Faucher-Giguere power law, tau_eff equals 0.0018 times (1+z) to the 3.92, with metals left in. That fit was measured from redshift 2 to 4.2. At redshift 5 it is an extrapolation, and I am treating it as an input, not as a measurement.

With 100 kpc per h the lognormal variance is 0.78, so the density is weakly nonlinear. With no cutoff the variance is 6.7 and the fitted amplitude is A equals 74, because the target transmission is low. At 100 kpc, A is 6.3. At 50, A is 10.7.

The Kaiser panel has the same pattern as at redshift 3: the curves agree on large scales, and 100 kpc falls first. The sightline panel is much higher at every k. Its amplitude is fixed by a mean transmission of 0.13. Kaiser's amplitude is still fixed by a bias of minus 0.14.
""")

    # 11. Figure 14
    s = new()
    _title(s, "One cube: measured P_F(k, μ)")
    _kicker(s, "Parallel power lies above real space. The slope follows Arinyo-i-Prats; the amplitude lies above it.")
    _picture(s, "14_p3d_kmu.png", Inches(0.4), Inches(1.4), Inches(12.5), Inches(5.5))
    _notes(s, """
This is the three-dimensional measurement. One periodic cube, 40 Mpc per h, 128 cubed, redshift 3, Jeans length 100 kpc per h. Each Fourier mode is given the rms of P of k, with a random phase. That is the same choice Arinyo-i-Prats made for their initial conditions, so one box is not limited by a Rayleigh draw.

The Gaussian variance is 1.38. The lognormal variance is 2.85. The mean transmission is 0.696, which is their value at redshift 3, rather than 0.80. The line-of-sight velocity is linear theory, with growth rate f equals 0.98. The rms shift is about 1 Mpc per h, three cells.

Before trusting the optical depth, I built a field that is exactly b_F times (1 + beta_F mu squared) times delta of k, using minus 0.14 and 1.4. The estimator recovers that power with a maximum fractional residual of 10 to the minus 13. So the binning is doing what it should.

On the figure, the solid blue curve is our Kaiser formula on the screened spectrum. The green dashed curve is their published formula, not a refit, evaluated on our linear matter spectrum. The dotted curve is the lognormal flux in real space. The red curve is the same flux after the velocity shift.

At k equals 1.16 h per megaparsec, the redshift-space measurement divided by their formula is 2.0 for mu below 0.25, and 1.3 for mu above 0.75. The parallel bin lies above the real-space curve, and the transverse bin lies below it. That is the sense of the Kaiser factor. The slope follows their formula. The amplitude does not. This cube is a lognormal field with an explicit 100 kpc cutoff. Their formula was calibrated on a hydrodynamic simulation, with a different cosmology, and their pressure scale is inside D_1 rather than applied to the matter spectrum beforehand.
""")

    # 12. Conclusions
    s = new()
    _title(s, "Conclusions")
    frame = _box(s, Inches(0.6), Inches(1.4), Inches(12.1), Inches(5.4))
    conclusions = [
        "The two October calculations are unchanged. A Jeans table can be passed in front of either of them.",
        "At 100 kpc/h the lognormal density is weakly nonlinear (variance ∼ 2). At 50 kpc/h the cutoff is weaker. With no cutoff, a Gaussian density takes negative values.",
        "The fluctuations in the October sightline curve scale as 1/√N. The offset from Kaiser is the flux model.",
        "At z = 5 the spectra reach 0.2 s/km. The sightline amplitude is set by a mean transmission of 0.13.",
        "In one cube, P_F(k, μ) has the Kaiser anisotropy and follows the shape of the published Arinyo-i-Prats formula. The amplitude lies above that formula.",
    ]
    for i, text in enumerate(conclusions, start=1):
        _para(frame, f"{i}     {text}", 20, INK, space_before=0 if i == 1 else 14, first=(i == 1))
    _notes(s, """
Five statements.

First, nothing from 13 October was replaced. The new cutoff is a table I can pass in.

Second, at 100 kpc per h the lognormal density in these volumes is weakly nonlinear. At 50 kpc the cutoff is weaker. Without a cutoff, the Gaussian field is not a usable gas density.

Third, the fluctuations were sample variance. They fall as one over square root of N. The offset from the blue curve is the flux model.

Fourth, redshift 5 now reaches the wavenumber you asked for. The price of a realistic mean transmission is that the sightline power sits well above the Kaiser curve with b_F equals minus 0.14.

Fifth, the cube measures an anisotropic P of k and mu. The anisotropy has the Kaiser sign. The shape is close to the published nonlinear formula. The amplitude is higher, and I have not adjusted their parameters to remove that.

I am happy to go to the backup figure, or to the limits of the cube.
""")

    # 13. Limits
    s = new()
    _title(s, "What this calculation does not yet include")
    frame = _box(s, Inches(0.6), Inches(1.45), Inches(12.1), Inches(5.3))
    limits = [
        "The cube is one volume, 40 Mpc/h on a side. Modes above about 10 h/Mpc are unresolved, and their k_p is 13 h/Mpc.",
        "Peculiar velocities are linear theory. Thermal broadening is off. The pixel-space Gaussian smoother from October is still available, and it is set to zero width.",
        "The z = 5 mean transmission extrapolates a fit that stops at z = 4.2.",
        "The Arinyo-i-Prats numbers are their fiducial hydrodynamic run at z = 3. Their cosmology is not ours, and the parameters are not refitted.",
        "The cross-power, the covariance, and the Fisher matrix are still outside this calculation.",
    ]
    for i, text in enumerate(limits, start=1):
        _para(frame, f"{i}     {text}", 20, INK, space_before=0 if i == 1 else 14, first=(i == 1))
    _notes(s, """
These are the limits I want on the table before questions.

The cube is a single 40 Mpc per h volume. The Nyquist mode is about 10 h per megaparsec, and the pressure scale in their formula is 13, so I am comparing shapes on the scales the grid actually resolves.

The velocities are the linear continuity field. I have not put in thermal broadening. The old Gaussian smoother in pixel units is still in the sightline code, and every figure here leaves it at zero. The smoothing you see is the Jeans factor on the matter spectrum.

The redshift-5 transmission is an extrapolation past redshift 4.2. If we would rather hold the mean transmission at 0.80, so that redshift 5 isolates the growth and the unit conversion, I can remake that figure. I chose 0.13 because a transmission of 0.80 is not a redshift-5 forest.

Their formula is used exactly as published for the fiducial simulation at redshift 3. I have not moved q_1 or the biases to improve the match.

And the forecast itself — cross-power, covariance, Fisher — is the next piece of work, not this one.
""")

    # 14. Backup
    s = new()
    _title(s, "Backup — shapes, once the amplitude is divided out")
    _kicker(s, "Each curve divided by its value at 0.005 s/km. From the 13 October comparison.")
    _picture(s, "08_kaiser_vs_fgpa_shape.png", Inches(2.3), Inches(1.4), Inches(8.7), Inches(5.5))
    _notes(s, """
I will show this only if someone asks whether the two October curves have the same shape.

Each flux spectrum is divided by its value at 0.005 s/km. Across the forest window the shapes agree. The residual wiggles on the sightline curve are the 256-line scatter from the earlier slide.

Agreement of the amplitudes, on large scales and in real space, would also require the flux bias to equal minus A times beta, with beta_F set to zero. Matching the angular factor (1 + beta_F mu squared) squared needs velocities that know about modes perpendicular to the line of sight. Independent one-dimensional lines do not contain those modes. The cube does.
""")

    total = len(slides)
    for i, slide in enumerate(slides, start=1):
        if i == 1:
            continue
        _footer(slide, i, total)

    out = DOCS / "followup_talk.pptx"
    prs.save(out)
    print(f"wrote {out} ({total} slides)")


if __name__ == "__main__":
    build()
