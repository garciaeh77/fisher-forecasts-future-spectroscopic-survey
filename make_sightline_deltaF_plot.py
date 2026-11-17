#!/usr/bin/env python3
"""FGPA sightlines: fiducial / lower A_s / lower n_s, with zoomed Delta F."""

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
    return x, result["sample_flux_skewer"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, f_fid = one_skewer(NS, AS)
    _, f_As = one_skewer(NS, AS * 0.8)
    _, f_ns = one_skewer(0.90, AS)
    d_As = f_As - f_fid
    d_ns = f_ns - f_fid

    diff = np.abs(d_As) + np.abs(d_ns)
    win = 100
    smooth = np.convolve(diff, np.ones(win) / win, mode="same")
    i0 = int(np.argmax(smooth[win:-win]) + win)
    half = 40
    lo = max(0, i0 - half)
    hi = min(len(x) - 1, i0 + half)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 8.0), gridspec_kw={"height_ratios": [1.05, 1.0, 1.05]}
    )

    axes[0].plot(x, f_fid, "C0", lw=1.0, label="fiducial")
    axes[0].plot(x, f_As, "C1", ls="--", lw=1.0, label=r"lower $A_s$ ($\times 0.8$)")
    axes[0].plot(x, f_ns, "C3", ls="-.", lw=1.0, label=r"lower $n_s$ ($=0.90$)")
    axes[0].axhline(0.80, color="0.4", ls=":", lw=0.9)
    axes[0].axvspan(x[lo], x[hi], color="0.75", alpha=0.35, zorder=0)
    axes[0].set_xlim(0, 200)
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_ylabel(r"$F$")
    axes[0].legend(fontsize=9, loc="lower left")
    axes[0].set_title(r"FGPA sightlines ($z=3$, same seed, $\langle F\rangle=0.80$)")
    axes[0].grid(alpha=0.25)

    for y, c, ls in [(f_fid, "C0", "-"), (f_As, "C1", "--"), (f_ns, "C3", "-.")]:
        axes[1].plot(x, y, color=c, ls=ls, lw=1.25)
    axes[1].axhline(0.80, color="0.4", ls=":", lw=0.9)
    axes[1].set_xlim(x[lo], x[hi])
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_ylabel(r"$F$")
    axes[1].set_title(rf"Zoom of $F$: ${x[lo]:.0f}$–${x[hi]:.0f}\,\mathrm{{Mpc}}/h$")
    axes[1].grid(alpha=0.25)

    axes[2].axhline(0.0, color="0.35", lw=0.9)
    axes[2].plot(x, d_As, "C1", ls="--", lw=1.4, label=r"$F(A_s\times0.8)-F_{\mathrm{fid}}$")
    axes[2].plot(x, d_ns, "C3", ls="-.", lw=1.4, label=r"$F(n_s=0.90)-F_{\mathrm{fid}}$")
    axes[2].set_xlim(x[lo], x[hi])
    pad = 1.15 * max(np.max(np.abs(d_As[lo : hi + 1])), np.max(np.abs(d_ns[lo : hi + 1])))
    axes[2].set_ylim(-pad, pad)
    axes[2].set_xlabel(r"$x$ [Mpc/$h$]")
    axes[2].set_ylabel(r"$\Delta F$")
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].set_title(r"Zoom of $\Delta F$ relative to fiducial")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    out = OUT / "17_fgpa_sightlines_deltaF"
    fig.savefig(out.with_suffix(".png"), dpi=170)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")
    print(f"rms ΔF(As)={np.sqrt(np.mean(d_As**2)):.4f}, rms ΔF(ns)={np.sqrt(np.mean(d_ns**2)):.4f}")


if __name__ == "__main__":
    main()
