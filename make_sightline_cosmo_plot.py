#!/usr/bin/env python3
"""Five FGPA sightlines from Eisenstein-Hu P_CDM, varying A_s and n_s.

Fiducial cosmology, then A_s x 0.8 / 1.2 at fixed n_s, then n_s = 0.90 / 1.02
at fixed A_s. Same random seed so phase structure is shared and only the
matter spectrum changes.
"""

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


def _one_skewer(ns: float, As: float):
    table = TabulatedMatterPower.from_eisenstein_hu(
        Z, NS, AS, OM, OB, H, kmax=80.0, n_k=3000
    )
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
    return x, result["sample_flux_skewer"], result["fgpa_A"], result["mean_flux"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    cases = [
        (NS, AS, rf"fiducial: $n_s={NS}$, $A_s={AS:.1e}$", "C0"),
        (NS, AS * 0.8, rf"lower $A_s$: $A_s\times 0.8$, $n_s={NS}$", "C1"),
        (NS, AS * 1.2, rf"higher $A_s$: $A_s\times 1.2$, $n_s={NS}$", "C2"),
        (0.90, AS, rf"lower $n_s$: $n_s=0.90$, $A_s$ fiducial", "C3"),
        (1.02, AS, rf"higher $n_s$: $n_s=1.02$, $A_s$ fiducial", "C4"),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    for ax, (ns, As, label, color) in zip(axes, cases):
        x, flux, A, mean_f = _one_skewer(ns, As)
        ax.plot(x, flux, color=color, lw=0.9)
        ax.axhline(0.80, color="0.35", ls="--", lw=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(r"$F$")
        ax.set_title(rf"{label}  ($A={A:.3f}$, $\langle F\rangle={mean_f:.3f}$)", fontsize=11)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel(r"$x$ [Mpc/$h$]")
    fig.suptitle(
        rf"FGPA sightlines from Eisenstein–Hu $P_{{\mathrm{{CDM}}}}$ "
        rf"($z={Z:.0f}$, same seed, $\langle F\rangle=0.80$)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout()
    out = OUT / "15_fgpa_sightlines_As_ns"
    fig.savefig(out.with_suffix(".png"), dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}.png and {out}.pdf")


if __name__ == "__main__":
    main()
