#!/usr/bin/env python3
"""Overlay three FGPA sightlines: fiducial, lower A_s, lower n_s, with zoom."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p1d.fgpa_field import SkewerConfig, run_fgpa_skewers
from p1d.matter_table import TabulatedMatterPower

OUT = Path(__file__).resolve().parent / "plots"
OM, OB, H = 0.31, 0.049, 0.67
NS, AS, Z = 0.965, 2.1e-9, 3.0
SEED = 7


def one_skewer(ns: float, As: float):
    table = TabulatedMatterPower.from_eisenstein_hu(Z, NS, AS, OM, OB, H, kmax=80.0, n_k=3000)
    config = SkewerConfig(
        redshift=Z,
        As=As,
        ns=ns,
        omega_m=OM,
        omega_b=OB,
        h=H,
        length_mpch=200.0,
        n_pixels=512,
        n_skewers=1,
        target_mean_flux=0.80,
        gamma=1.5,
        seed=SEED,
    )
    result = run_fgpa_skewers(table, config)
    x = np.arange(config.n_pixels) * result["pixel_mpch"]
    return x, result["sample_flux_skewer"], result["fgpa_A"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [
        (NS, AS, r"fiducial ($n_s=0.965$, $A_s=2.1\times10^{-9}$)", "C0", "-"),
        (NS, AS * 0.8, r"lower $A_s$ ($\times 0.8$)", "C1", "--"),
        (0.90, AS, r"lower $n_s$ ($=0.90$)", "C3", "-."),
    ]
    series = []
    for ns, As, label, color, ls in cases:
        x, flux, A = one_skewer(ns, As)
        series.append((label, color, ls, x, flux, A))

    x = series[0][3]
    f0, fA, fn = series[0][4], series[1][4], series[2][4]
    diff = np.abs(f0 - fA) + np.abs(f0 - fn)
    win = 100
    smooth = np.convolve(diff, np.ones(win) / win, mode="same")
    i0 = int(np.argmax(smooth[win:-win]) + win)
    half = 40
    lo = max(0, i0 - half)
    hi = min(len(x) - 1, i0 + half)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), gridspec_kw={"height_ratios": [1.15, 1]})
    for label, color, ls, x, flux, A in series:
        axes[0].plot(x, flux, color=color, ls=ls, lw=1.1, label=label)
        axes[1].plot(x, flux, color=color, ls=ls, lw=1.3)

    for ax in axes:
        ax.axhline(0.80, color="0.4", ls=":", lw=0.9)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel(r"$F$")
        ax.grid(alpha=0.25)

    axes[0].set_xlim(0, 200)
    axes[0].axvspan(x[lo], x[hi], color="0.75", alpha=0.35, zorder=0)
    axes[0].legend(fontsize=9, loc="lower left")
    axes[0].set_title(
        r"FGPA sightlines from Eisenstein–Hu $P_{\mathrm{CDM}}$ "
        r"($z=3$, same seed, $\langle F\rangle=0.80$)"
    )
    axes[1].set_xlim(x[lo], x[hi])
    axes[1].set_xlabel(r"$x$ [Mpc/$h$]")
    axes[1].set_title(rf"Zoom: ${x[lo]:.0f}$–${x[hi]:.0f}\,\mathrm{{Mpc}}/h$")

    fig.tight_layout()
    out = OUT / "16_fgpa_sightlines_overlay_zoom"
    fig.savefig(out.with_suffix(".png"), dpi=170)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}.png / .pdf  (zoom {x[lo]:.1f}–{x[hi]:.1f} Mpc/h)")
    for label, _, _, _, _, A in series:
        print(f"  {label}: A={A:.3f}")


if __name__ == "__main__":
    main()
