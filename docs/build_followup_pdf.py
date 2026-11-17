#!/usr/bin/env python3
"""Render the follow-up talk as a multi-page PDF matching the PowerPoint layout."""

from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
OUT = Path(__file__).resolve().parent / "followup_talk.pdf"

W, H = 13.333 * inch, 7.5 * inch
NAVY = HexColor("#1B3A4B")
TEAL = HexColor("#1F6F8B")
INK = HexColor("#1C1C1C")
MUTED = HexColor("#3E4C57")
RULE = HexColor("#D5DEE6")


def _fonts():
    candidates = [
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
        ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Helvetica.ttc"),
    ]
    for regular, bold in candidates:
        if Path(regular).exists() and Path(bold).exists():
            try:
                pdfmetrics.registerFont(TTFont("Talk", regular))
                pdfmetrics.registerFont(TTFont("Talk-Bold", bold))
                return "Talk", "Talk-Bold"
            except Exception:
                continue
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_B = _fonts()


def _bar(c):
    c.setFillColor(NAVY)
    c.rect(0, H - 0.08 * inch, W, 0.08 * inch, fill=1, stroke=0)


def _footer(c, number, total):
    c.setFillColor(MUTED)
    c.setFont(FONT, 10)
    c.drawString(0.55 * inch, 0.28 * inch, "Lyman-\u03b1  P1D    \u00b7    follow-up to 13 October")
    c.drawRightString(W - 0.55 * inch, 0.28 * inch, f"{number}   /   {total}")


def _title(c, text):
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 24)
    c.drawString(0.55 * inch, H - 0.55 * inch, text)


def _kicker(c, text):
    c.setFillColor(TEAL)
    c.setFont(FONT, 13)
    c.drawString(0.55 * inch, H - 0.9 * inch, text)


def _wrap(c, text, x, y, max_width, size, color, leading=None, font=None):
    font = font or FONT
    leading = leading or size * 1.28
    c.setFont(font, size)
    c.setFillColor(color)
    words = text.split()
    line = ""
    for word in words:
        trial = word if not line else f"{line} {word}"
        if c.stringWidth(trial, font, size) <= max_width:
            line = trial
        else:
            c.drawString(x, y, line)
            y -= leading
            line = word
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def _bullets(c, items, x, y, max_width, size=12, gap=8):
    for item in items:
        c.setFillColor(INK)
        c.setFont(FONT, size)
        c.drawString(x, y, "\u2022")
        y = _wrap(c, item, x + 14, y, max_width - 14, size, INK) - gap
    return y


def _picture(c, name, left, bottom, max_w, max_h):
    path = PLOTS / name
    with PILImage.open(path) as im:
        aspect = im.size[0] / im.size[1]
    if max_w / aspect <= max_h:
        width, height = max_w, max_w / aspect
    else:
        height, width = max_h, max_h * aspect
    x = left + (max_w - width) / 2
    y = bottom + (max_h - height) / 2
    c.drawImage(str(path), x, y, width=width, height=height, preserveAspectRatio=True, mask="auto")


def build():
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    total = 14

    def page_end(n):
        if n > 1:
            _footer(c, n, total)
        c.showPage()

    # 1 Title
    _bar(c)
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 36)
    c.drawString(0.7 * inch, H - 2.6 * inch, "Lyman-\u03b1 forest  P1D")
    c.setFillColor(INK)
    c.setFont(FONT, 18)
    _wrap(
        c,
        "What was ready on 13 October, and what I have done with the requests",
        0.7 * inch,
        H - 3.15 * inch,
        11.5 * inch,
        18,
        INK,
    )
    c.setFillColor(TEAL)
    c.setFont(FONT, 16)
    c.drawString(0.7 * inch, H - 4.6 * inch, "Helena")
    c.setFillColor(MUTED)
    c.setFont(FONT, 13)
    c.drawString(
        0.7 * inch,
        H - 5.0 * inch,
        "Analytic Kaiser projection  \u00b7  optical-depth sightlines  \u00b7  one three-dimensional box",
    )
    page_end(1)

    # 2 Two methods
    _bar(c)
    _title(c, "What was ready on 13 October")
    _kicker(c, "Same P_CDM. Method 1 is an analytic formula. Method 2 builds transmission along lines.")
    c.setFillColor(TEAL)
    c.setFont(FONT_B, 14)
    c.drawString(0.55 * inch, H - 1.4 * inch, "Method 1 \u2014 Kaiser, analytic")
    left_items = [
        "Treat the forest as a linearly biased tracer of the matter field, with coherent peculiar velocities.",
        "P_F(k, \u03bc) = b_F\u00b2 (1 + \u03b2_F \u03bc\u00b2)\u00b2 P_CDM(k).",
        "b_F = \u22120.14 and \u03b2_F = 1.4, fixed. Round form of early DESI.",
        "A forest spectrum sees only k\u2225. Integrate transverse modes.",
        "P_1D(k\u2225) = \u222b (k dk / 2\u03c0) P_F(k, \u03bc = k\u2225/k) from k\u2225 to k_max.",
        "No random draw. Smooth in n_s and A_s.",
    ]
    _bullets(c, left_items, 0.55 * inch, H - 1.7 * inch, 5.7 * inch, size=11, gap=4)
    c.setFillColor(TEAL)
    c.setFont(FONT_B, 14)
    c.drawString(6.85 * inch, H - 1.4 * inch, "Method 2 \u2014 optical-depth sightlines")
    right_items = [
        "Project P_CDM onto one line: P_1D,matter(k\u2225).",
        "Draw Gaussian contrast \u03b4_G with that 1D power.",
        "Lognormal density: \u0394 = exp(\u03b4_G \u2212 \u00bd Var(\u03b4_G)).",
        "\u0394 is always positive. Mean of \u0394 is near one.",
        "Optical depth \u03c4 = A \u0394^1.65, then F = e^\u2212\u03c4.",
        "Solve A so \u27e8F\u27e9 = 0.80. Measure P_1D of F/\u27e8F\u27e9 \u2212 1.",
    ]
    _bullets(c, right_items, 6.85 * inch, H - 1.7 * inch, 5.9 * inch, size=11, gap=4)
    page_end(2)

    # Figure slides
    figure_slides = [
        (3, "The figure from 13 October",
         "Kaiser and the sightlines share P_CDM. Their flux amplitudes answer different conditions.",
         "07_pipeline_stages.png"),
        (6, "Pressure: a Jeans factor on P_CDM",
         "100 kpc/h falls first. The right-hand panel is the Kaiser formula, evaluated on a grid.",
         "09_3d_power_spectrum.png"),
        (7, "Lognormal density, against the Gaussian field",
         "At 100 kpc/h the variance is of order one. Without a cutoff, a third of the Gaussian pixels are negative.",
         "10_gaussian_vs_lognormal.png"),
        (8, "The same cutoff, in the flux power at z = 3",
         "The two calculations agree below 0.01 s/km. Above that, the sightlines fall less steeply than Kaiser.",
         "11_flux_jeans_scales.png"),
        (9, "The fluctuations were the size of the ensemble",
         "From 256 to 4096 lines the scatter falls by about 5. The offset from Kaiser remains at 10000 lines.",
         "12_skewer_noise.png"),
        (10, "Redshift 5, out to 0.2 s/km",
         "Nyquist is past the request. The mean transmission, 0.13, is an extrapolation of the z \u2264 4.2 fit.",
         "13_z5_smallscale_flux.png"),
        (11, "One cube: measured P_F(k, \u03bc)",
         "Parallel power lies above real space. The slope follows Arinyo-i-Prats; the amplitude lies above it.",
         "14_p3d_kmu.png"),
        (14, "Backup \u2014 shapes, once the amplitude is divided out",
         "Each curve divided by its value at 0.005 s/km. From the 13 October comparison.",
         "08_kaiser_vs_fgpa_shape.png"),
    ]

    # 3, then 4, 5, then figure slides interleaved with text slides in order
    # Build remaining pages in presentation order.

    # Slide 3 figure 07
    _bar(c)
    _title(c, "The figure from 13 October")
    _kicker(c, "Kaiser and the sightlines share P_CDM. Their flux amplitudes answer different conditions.")
    _picture(c, "07_pipeline_stages.png", 1.4 * inch, 0.7 * inch, 10.5 * inch, 5.5 * inch)
    page_end(3)

    # Slide 4 requests
    _bar(c)
    _title(c, "What you asked for on 13 October")
    requests = [
        "Extend the spectra to smaller scales, out to k \u223c 0.2 s/km at z \u223c 5.",
        "Use more sightlines, and test whether the fluctuations in the October figure shrink.",
        "Build a three-dimensional realisation, so that P_F(k, \u03bc) is measured rather than assumed.",
        "Compare that measurement with the nonlinear model of Arinyo-i-Prats et al. (2015).",
        "Apply a Gaussian pressure filter to the gas.",
        "Use a lognormal density for weakly nonlinear dark matter.",
    ]
    y = H - 1.5 * inch
    for i, text in enumerate(requests, start=1):
        c.setFillColor(INK)
        c.setFont(FONT, 16)
        c.drawString(0.7 * inch, y, f"{i}")
        y = _wrap(c, text, 1.15 * inch, y, 11.2 * inch, 16, INK) - 10
    page_end(4)

    # Slide 5 code
    _bar(c)
    _title(c, "How the extensions sit in the code")
    _kicker(c, "The October functions still run. The new physics is passed in, or measured beside them.")
    rows = [
        ("Unchanged", "p1d_skm is still the Kaiser integral. run_fgpa_skewers is still the optical-depth lines. Both default to an unsmoothed spectrum."),
        ("Pressure", "with_jeans multiplies a stored P_CDM by exp(\u22122 x_J\u00b2 k\u00b2) and returns a new table. The original table is untouched. 100 kpc/h is x_J = 0.10 Mpc/h."),
        ("Cube", "gaussian_box draws one periodic volume. fgpa_flux_box applies the lognormal map, the optical depth, and a linear line-of-sight velocity."),
        ("Published formula", "The Arinyo-i-Prats factor D_1(k, \u03bc) is coded with the fiducial-simulation numbers at z = 3. They are not refitted."),
    ]
    y = H - 1.5 * inch
    for head, body in rows:
        c.setFillColor(TEAL)
        c.setFont(FONT_B, 14)
        c.drawString(0.55 * inch, y, head)
        y -= 18
        y = _wrap(c, body, 0.55 * inch, y, 12.2 * inch, 13, INK) - 12
    page_end(5)

    for num, title, kicker, fig in [
        (6, "Pressure: a Jeans factor on P_CDM",
         "100 kpc/h falls first. The right-hand panel is the Kaiser formula, evaluated on a grid.",
         "09_3d_power_spectrum.png"),
        (7, "Lognormal density, against the Gaussian field",
         "At 100 kpc/h the variance is of order one. Without a cutoff, a third of the Gaussian pixels are negative.",
         "10_gaussian_vs_lognormal.png"),
        (8, "The same cutoff, in the flux power at z = 3",
         "The two calculations agree below 0.01 s/km. Above that, the sightlines fall less steeply than Kaiser.",
         "11_flux_jeans_scales.png"),
        (9, "The fluctuations were the size of the ensemble",
         "From 256 to 4096 lines the scatter falls by about 5. The offset from Kaiser remains at 10000 lines.",
         "12_skewer_noise.png"),
        (10, "Redshift 5, out to 0.2 s/km",
         "Nyquist is past the request. The mean transmission, 0.13, is an extrapolation of the z \u2264 4.2 fit.",
         "13_z5_smallscale_flux.png"),
        (11, "One cube: measured P_F(k, \u03bc)",
         "Parallel power lies above real space. The slope follows Arinyo-i-Prats; the amplitude lies above it.",
         "14_p3d_kmu.png"),
    ]:
        _bar(c)
        _title(c, title)
        _kicker(c, kicker)
        if num == 9:
            _picture(c, fig, 2.3 * inch, 0.7 * inch, 8.7 * inch, 5.5 * inch)
        else:
            _picture(c, fig, 0.4 * inch, 0.65 * inch, 12.5 * inch, 5.55 * inch)
        page_end(num)

    # 12 conclusions
    _bar(c)
    _title(c, "Conclusions")
    conclusions = [
        "The two October calculations are unchanged. A Jeans table can be passed in front of either of them.",
        "At 100 kpc/h the lognormal density is weakly nonlinear (variance \u223c 2). At 50 kpc/h the cutoff is weaker. With no cutoff, a Gaussian density takes negative values.",
        "The fluctuations in the October sightline curve scale as 1/\u221aN. The offset from Kaiser is the flux model.",
        "At z = 5 the spectra reach 0.2 s/km. The sightline amplitude is set by a mean transmission of 0.13.",
        "In one cube, P_F(k, \u03bc) has the Kaiser anisotropy and follows the shape of the published Arinyo-i-Prats formula. The amplitude lies above that formula.",
    ]
    y = H - 1.5 * inch
    for i, text in enumerate(conclusions, start=1):
        c.setFillColor(INK)
        c.setFont(FONT, 15)
        c.drawString(0.6 * inch, y, f"{i}")
        y = _wrap(c, text, 1.05 * inch, y, 11.5 * inch, 15, INK) - 12
    page_end(12)

    # 13 limits
    _bar(c)
    _title(c, "What this calculation does not yet include")
    limits = [
        "The cube is one volume, 40 Mpc/h on a side. Modes above about 10 h/Mpc are unresolved, and their k_p is 13 h/Mpc.",
        "Peculiar velocities are linear theory. Thermal broadening is off. The pixel-space Gaussian smoother from October is still available, and it is set to zero width.",
        "The z = 5 mean transmission extrapolates a fit that stops at z = 4.2.",
        "The Arinyo-i-Prats numbers are their fiducial hydrodynamic run at z = 3. Their cosmology is not ours, and the parameters are not refitted.",
        "The cross-power, the covariance, and the Fisher matrix are still outside this calculation.",
    ]
    y = H - 1.5 * inch
    for i, text in enumerate(limits, start=1):
        c.setFillColor(INK)
        c.setFont(FONT, 15)
        c.drawString(0.6 * inch, y, f"{i}")
        y = _wrap(c, text, 1.05 * inch, y, 11.5 * inch, 15, INK) - 12
    page_end(13)

    # 14 backup
    _bar(c)
    _title(c, "Backup \u2014 shapes, once the amplitude is divided out")
    _kicker(c, "Each curve divided by its value at 0.005 s/km. From the 13 October comparison.")
    _picture(c, "08_kaiser_vs_fgpa_shape.png", 2.2 * inch, 0.7 * inch, 8.9 * inch, 5.5 * inch)
    page_end(14)

    c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
