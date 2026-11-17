#!/usr/bin/env python3
"""Full-physics P_1D and P_3D(k, μ) from one cube (Kier follow-up).

Physics in the generated field:
  Jeans on P_CDM → Gaussian box → lognormal Δ → FGPA τ = A Δ^β
  → linear peculiar-velocity CIC shift → thermal Gaussian along the LOS
  → calibrate ⟨F⟩ → measure P_F(k, μ) and P_1D from the same skewers.

    python make_full_physics_plots.py
    python make_full_physics_plots.py --As 2.3e-9 --ns 0.96 --lambda-filter 0.1 --thermal-kms 10

Writes plots/15_full_physics_p1d_p3d and plots/16_full_physics_param_vary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p1d.arinyo import flux_power as arinyo_power
from p1d.box3d import bin_power, fgpa_flux_box, gaussian_box, growth_rate, measure_power
from p1d.flux import flux_power_1d, p1d_h_to_kms, velocity_per_mpch
from p1d.jeans import with_jeans
from p1d.matter_table import TabulatedMatterPower
from p1d.skewers import measure_p1d, rfft_modes

OUT = Path(__file__).resolve().parent / "plots"
OM, OB, H = 0.31, 0.049, 0.67
B_F, BETA_F = -0.14, 1.4


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=150)
    fig.savefig(OUT / f"{name}.pdf")
    print(f"wrote {name}")


def thermal_sigma_mpch(thermal_kms, z, omega_m, h):
    """Convert a thermal Gaussian width in km/s to comoving Mpc/h."""
    if thermal_kms <= 0.0:
        return 0.0
    return float(thermal_kms) / float(velocity_per_mpch(z, omega_m, h))


def run_cube(
    *,
    z=3.0,
    As=2.1e-9,
    ns=0.965,
    lambda_filter=0.10,
    thermal_kms=10.0,
    mean_flux=0.696,
    length=40.0,
    n=128,
    seed=7,
    exact_amplitude=True,
    apply_rsd=True,
):
    """One full-physics cube → P_3D bins and LOS P_1D."""
    table = TabulatedMatterPower.from_eisenstein_hu(
        z, ns, As, OM, OB, H, kmin=1.0e-3, kmax=120.0, n_k=4000
    )
    screened = table if lambda_filter <= 0.0 else with_jeans(table, lambda_filter)

    def power(k):
        return screened(k, As=As, ns=ns)

    box = gaussian_box(power, length, n, seed=seed, exact_amplitude=exact_amplitude)
    f = growth_rate(z, OM, H) if apply_rsd else None
    sigma_x = thermal_sigma_mpch(thermal_kms, z, OM, H)
    flux = fgpa_flux_box(
        box,
        length,
        gamma=1.5,
        target_mean_flux=mean_flux,
        f=f,
        thermal_sigma_mpch=sigma_x,
    )
    contrast = flux["contrast"]
    p_mode = measure_power(contrast, length)
    k_modes = rfft_modes(length, n)
    # Every transverse cell is one LOS skewer (last axis).
    p1d_h = measure_p1d(contrast.reshape(-1, n), length)

    k_nyq = np.pi * n / length
    k_edges = np.geomspace(2.0 * np.pi / length, 0.85 * k_nyq, 10)
    mu_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    k_cent, b_flux, counts = bin_power(box["k"], box["mu"], p_mode, k_edges, mu_edges)

    # Analytic overlays on the same Jeans-smoothed P_CDM (no thermal in Kaiser).
    formula = B_F**2 * (1.0 + BETA_F * box["mu"] ** 2) ** 2 * box["pk"]
    _, b_kaiser, _ = bin_power(box["k"], box["mu"], formula, k_edges, mu_edges)
    nonzero = box["k"] > 0
    p_lin = np.zeros_like(box["k"])
    p_lin[nonzero] = table(box["k"][nonzero], As=As, ns=ns)  # Arinyo: EH without extra Jeans
    p_arinyo = np.zeros_like(p_lin)
    p_arinyo[nonzero] = arinyo_power(box["k"][nonzero], box["mu"][nonzero], p_lin[nonzero])
    _, b_arinyo, _ = bin_power(box["k"], box["mu"], p_arinyo, k_edges, mu_edges)

    p1d_kaiser_h = np.zeros_like(k_modes)
    use = k_modes > 0
    p1d_kaiser_h[use] = flux_power_1d(
        k_modes[use],
        z,
        ns,
        As,
        OM,
        OB,
        H,
        b_f=B_F,
        beta_f=BETA_F,
        matter_power=power,
    )

    return {
        "z": z,
        "As": As,
        "ns": ns,
        "lambda_filter": lambda_filter,
        "thermal_kms": thermal_kms,
        "thermal_sigma_mpch": sigma_x,
        "mean_flux": flux["mean_flux"],
        "A": flux["A"],
        "f": float(growth_rate(z, OM, H)),
        "length": length,
        "n": n,
        "k_hmpc": k_modes,
        "k_skm": k_modes / velocity_per_mpch(z, OM, H),
        "p1d_mpc_h": p1d_h,
        "p1d_kms": p1d_h_to_kms(p1d_h, z, OM, H),
        "p1d_kaiser_mpc_h": p1d_kaiser_h,
        "p1d_kaiser_kms": p1d_h_to_kms(p1d_kaiser_h, z, OM, H),
        "k_cent": k_cent,
        "p3d_flux": b_flux,
        "p3d_kaiser": b_kaiser,
        "p3d_arinyo": b_arinyo,
        "counts": counts,
        "box_pk": box["pk"],
    }


def plot_full_physics(result, tag="15_full_physics_p1d_p3d"):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    k1 = result["k_skm"]
    ok1 = (k1 > 0) & (k1 < 0.85 * np.pi * result["n"] / result["length"] / velocity_per_mpch(result["z"], OM, H))
    axes[0].loglog(k1[ok1], result["p1d_kaiser_kms"][ok1], color="C0", label="Kaiser analytic")
    axes[0].loglog(
        k1[ok1],
        result["p1d_kms"][ok1],
        color="C3",
        label="box skewers (full physics)",
    )
    axes[0].set_xlabel(r"$k_\parallel$ [s\,km$^{-1}$]")
    axes[0].set_ylabel(r"$P_{1\mathrm{D}}$ [km\,s$^{-1}$]")
    axes[0].set_title(r"$P_{1\mathrm{D}}(k_\parallel)$ from LOS skewers")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    panels = [(0, r"$P_F$, $\mu<0.25$"), (-1, r"$P_F$, $\mu>0.75$")]
    for ax, (row, title) in zip(axes[1:], panels):
        kc = result["k_cent"]
        ok = np.isfinite(result["p3d_flux"][row])
        ax.loglog(kc[ok], result["p3d_kaiser"][row][ok], color="C0", label="Kaiser")
        ax.loglog(kc[ok], result["p3d_arinyo"][row][ok], color="C2", ls="--", label="Arinyo $z=3$")
        ax.loglog(kc[ok], result["p3d_flux"][row][ok], color="C3", label="box (full physics)")
        ax.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[1].set_ylabel(r"$P_F(k,\mu)$ [$({\mathrm{Mpc}}/h)^3$]")

    fig.suptitle(
        rf"Full physics: lognormal + FGPA + Jeans $x_J={1e3*result['lambda_filter']:.0f}$ kpc/$h$ "
        rf"+ linear RSD + thermal $\sigma_v={result['thermal_kms']:.0f}$ km/s, "
        rf"$\langle F\rangle={result['mean_flux']:.3f}$, $z={result['z']:.0f}$"
    )
    fig.tight_layout()
    _save(fig, tag)
    plt.close(fig)


def plot_param_variations(fiducial_kwargs, tag="16_full_physics_param_vary"):
    """Vary As, ns, Jeans filter, and thermal width; show P_1D response."""
    fid = run_cube(**fiducial_kwargs)
    variations = [
        ("$A_s$", "As", [0.9 * fiducial_kwargs["As"], fiducial_kwargs["As"], 1.1 * fiducial_kwargs["As"]],
         [r"$0.9A_s$", r"$A_s$", r"$1.1A_s$"]),
        ("$n_s$", "ns", [fiducial_kwargs["ns"] - 0.02, fiducial_kwargs["ns"], fiducial_kwargs["ns"] + 0.02],
         [r"$n_s-0.02$", r"$n_s$", r"$n_s+0.02$"]),
        (r"$\lambda_\mathrm{filter}=x_J$", "lambda_filter", [0.05, 0.10, 0.15],
         [r"$50$", r"$100$", r"$150$ kpc/$h$"]),
        (r"$\sigma_{v,\mathrm{th}}$", "thermal_kms", [0.0, 10.0, 20.0],
         [r"$0$", r"$10$", r"$20$ km/s"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, (title, key, values, labels) in zip(axes.ravel(), variations):
        for val, lab, ls in zip(values, labels, ["--", "-", ":"]):
            kwargs = dict(fiducial_kwargs)
            kwargs[key] = val
            # Reuse seed so phase structure is comparable across param shifts.
            out = run_cube(**kwargs)
            k = out["k_skm"]
            ok = (k > 1.0e-3) & (k < 0.02)
            ax.loglog(k[ok], out["p1d_kms"][ok], ls=ls, label=lab)
        ax.set_title(title)
        ax.set_xlabel(r"$k_\parallel$ [s\,km$^{-1}$]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(r"$P_{1\mathrm{D}}$ [km\,s$^{-1}$]")
    axes[1, 0].set_ylabel(r"$P_{1\mathrm{D}}$ [km\,s$^{-1}$]")
    fig.suptitle(
        r"Full-physics box skewers while varying one parameter "
        rf"($x_J={1e3*fid['lambda_filter']:.0f}$ kpc/$h$, "
        rf"$\sigma_v={fid['thermal_kms']:.0f}$ km/s fiducial)"
    )
    fig.tight_layout()
    _save(fig, tag)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--As", type=float, default=2.1e-9)
    p.add_argument("--ns", type=float, default=0.965)
    p.add_argument(
        "--lambda-filter",
        type=float,
        default=0.10,
        help="Jeans length x_J in Mpc/h (Kier: lambda_filter)",
    )
    p.add_argument("--thermal-kms", type=float, default=10.0, help="thermal Gaussian σ in km/s")
    p.add_argument("--mean-flux", type=float, default=0.696)
    p.add_argument("--z", type=float, default=3.0)
    p.add_argument("--length", type=float, default=40.0)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip-variations", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    kwargs = dict(
        z=args.z,
        As=args.As,
        ns=args.ns,
        lambda_filter=args.lambda_filter,
        thermal_kms=args.thermal_kms,
        mean_flux=args.mean_flux,
        length=args.length,
        n=args.n,
        seed=args.seed,
    )
    print(
        f"full physics: As={args.As:.3e}, ns={args.ns}, "
        f"x_J={args.lambda_filter} Mpc/h, thermal={args.thermal_kms} km/s, "
        f"<F>={args.mean_flux}, box={args.length} Mpc/h ^ {args.n}^3"
    )
    result = run_cube(**kwargs)
    print(
        f"A={result['A']:.4g}, measured <F>={result['mean_flux']:.4f}, "
        f"f={result['f']:.4f}, sigma_x={result['thermal_sigma_mpch']:.4f} Mpc/h"
    )
    plot_full_physics(result)
    if not args.skip_variations:
        plot_param_variations(kwargs)
    return result


if __name__ == "__main__":
    main()
