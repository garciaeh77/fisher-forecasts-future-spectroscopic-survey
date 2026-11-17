#!/usr/bin/env python3
"""Meeting runs that were still open: z=5 small-scale flux, and a 3D P(k, mu).

Does not change plots 01-12. Writes plots/13 and plots/14.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p1d.arinyo import flux_power as arinyo_power
from p1d.box3d import (
    bin_power,
    fgpa_flux_box,
    gaussian_box,
    growth_rate,
    kaiser_flux_from_density,
    measure_power,
)
from p1d.eisenstein_hu import linear_matter_power
from p1d.fgpa_field import (
    calibrate_mean_flux,
    fgpa_exponent,
    flux_contrast,
    lognormal_density,
    prepare_tau_unit,
    transmitted_flux,
)
from p1d.flux import flux_power_1d, k_hmpc_to_skm, k_skm_to_hmpc, p1d_h_to_kms, velocity_per_mpch
from p1d.jeans import jeans_suppression, with_jeans
from p1d.matter_table import TabulatedMatterPower
from p1d.projection import project_p1d
from p1d.skewers import generate_gaussian_skewers, measure_p1d, rfft_modes

OUT = Path(__file__).resolve().parent / "plots"
OM, OB, H = 0.31, 0.049, 0.67
NS, AS = 0.965, 2.1e-9
B_F, BETA_F = -0.14, 1.4
XJ_100, XJ_50 = 0.10, 0.05


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150)
    fig.savefig(OUT / f"{name}.pdf")
    print(f"wrote {name}")


def mean_flux_fg08(z):
    """Power-law fit of Faucher-Giguere et al. (2008), metals left in.

    The fit was measured on 2 <= z <= 4.2. At z=5 it is an extrapolation.
    """
    tau_eff = 0.0018 * (1.0 + z) ** 3.92
    return float(np.exp(-tau_eff)), float(tau_eff)


def _draw(table, length, n_pixels, n_skewers, seed, ns, as_):
    k_modes = rfft_modes(length, n_pixels)
    nonzero = k_modes > 0
    q = np.geomspace(table.kmin, table.kmax, 8192)
    pk = table(q, As=as_, ns=ns)
    target = np.zeros_like(k_modes)
    target[nonzero] = project_p1d(k_modes[nonzero], q, pk)
    gaussian = generate_gaussian_skewers(
        target, length_mpch=length, n_pixels=n_pixels, n_skewers=n_skewers, seed=seed
    )
    return k_modes, gaussian


def plot_z5():
    z = 5.0
    mean_flux, tau_eff = mean_flux_fg08(z)
    dv = float(velocity_per_mpch(z, OM, H))
    print(f"z=5: dv/dchi={dv:.4f} km/s per Mpc/h, tau_eff={tau_eff:.4f}, mean flux={mean_flux:.4f}")
    print(f"z=5: 0.2 s/km = {0.2 * dv:.3f} h/Mpc")
    length, n_pixels, n_skewers = 40.0, 512, 256
    table = TabulatedMatterPower.from_eisenstein_hu(
        z, NS, AS, OM, OB, H, kmin=1.0e-3, kmax=80.0, n_k=4000
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for xj, label, ls in [
        (0.0, "no Jeans", "-"),
        (XJ_100, r"$x_J=100\,\mathrm{kpc}/h$", "--"),
        (XJ_50, r"$x_J=50\,\mathrm{kpc}/h$", ":"),
    ]:
        screened = table if xj == 0.0 else with_jeans(table, xj)
        k_modes, gaussian = _draw(screened, length, n_pixels, n_skewers, seed=11, ns=NS, as_=AS)
        density = lognormal_density(gaussian)
        tau_unit = prepare_tau_unit(density, gamma=1.5)
        amplitude = calibrate_mean_flux(tau_unit, mean_flux)
        flux = transmitted_flux(tau_unit, amplitude)
        contrast = flux_contrast(flux)
        k_skm = k_hmpc_to_skm(k_modes, z, OM, H)
        use = (k_skm > 1.0e-3) & (k_skm < 0.2) & (k_modes > 0)
        kaiser_h = flux_power_1d(
            k_skm_to_hmpc(k_skm[use], z, OM, H),
            z,
            NS,
            AS,
            OM,
            OB,
            H,
            k_max=80.0,
            matter_power=lambda q, s=screened: s(q),
        )
        axes[0].loglog(k_skm[use], p1d_h_to_kms(kaiser_h, z, OM, H), ls=ls, label=f"Kaiser, {label}")
        measured = p1d_h_to_kms(measure_p1d(contrast, length), z, OM, H)
        axes[1].loglog(k_skm[use], measured[use], ls=ls, label=label)
        print(
            f"z5 xJ={xj}: A={amplitude:.4g}, mean={float(np.mean(flux)):.4f}, "
            f"var lognormal={float(np.var(density)):.4g}, var gaussian={float(np.var(gaussian)):.4g}"
        )
    for ax, title in [
        (axes[0], r"Kaiser, integral to $80\,h\,\mathrm{Mpc}^{-1}$"),
        (axes[1], r"lognormal sightlines"),
    ]:
        ax.set_xlabel(r"$k_\parallel$ [s/km]")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(1.0e-3, 0.2)
    axes[0].set_ylabel(r"$P_{1\mathrm{D}}$ [km/s]")
    fig.suptitle(
        rf"$z=5$, $\langle F\rangle={mean_flux:.3f}$ from $\tau_{{\rm eff}}=0.0018(1+z)^{{3.92}}$, "
        rf"$L=40$ Mpc/$h$, {n_skewers} lines"
    )
    fig.tight_layout()
    _save(fig, "13_z5_smallscale_flux")
    plt.close(fig)
    print("fgpa beta", fgpa_exponent(1.5))


def _screened_power(z, xj):
    def power(k):
        p = linear_matter_power(k, z, NS, AS, OM, OB, H)
        return p * jeans_suppression(k, xj)

    return power


def _linear_power(z):
    def power(k):
        k = np.asarray(k, dtype=float)
        out = np.zeros_like(k)
        nonzero = k > 0.0
        out[nonzero] = linear_matter_power(k[nonzero], z, NS, AS, OM, OB, H)
        return out

    return power


def plot_box():
    z = 3.0
    length, n = 40.0, 128
    xj = XJ_100
    f = growth_rate(z, OM, H)
    print(f"z=3 growth rate f={f:.4f}")
    box = gaussian_box(_screened_power(z, xj), length, n, seed=7, exact_amplitude=True)
    kaiser_field = kaiser_flux_from_density(box, B_F, BETA_F)
    measured_kaiser = measure_power(kaiser_field, length)
    formula = B_F**2 * (1.0 + BETA_F * box["mu"] ** 2) ** 2 * box["pk"]
    nonzero = box["k"] > 0
    frac = np.max(np.abs(measured_kaiser[nonzero] / formula[nonzero] - 1.0))
    print(f"3D Kaiser recovery: max fractional residual {frac:.3e}")

    real = fgpa_flux_box(box, length, gamma=1.5, target_mean_flux=0.696)
    red = fgpa_flux_box(box, length, gamma=1.5, target_mean_flux=0.696, f=f)
    print(
        f"3D real space: A={real['A']:.4g}, mean={real['mean_flux']:.4f}, "
        f"var G={real['var_gaussian']:.4g}, var LN={real['var_lognormal']:.4g}"
    )
    print(
        f"3D redshift space: A={red['A']:.4g}, mean={red['mean_flux']:.4f}"
    )
    p_real = measure_power(real["contrast"], length)
    p_red = measure_power(red["contrast"], length)
    p_lin = _linear_power(z)(box["k"])
    p_arinyo = np.zeros_like(p_lin)
    p_arinyo[nonzero] = arinyo_power(box["k"][nonzero], box["mu"][nonzero], p_lin[nonzero])

    k_nyq = np.pi * n / length
    k_edges = np.geomspace(2.0 * np.pi / length, 0.85 * k_nyq, 10)
    mu_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    k_cent, b_real, c_real = bin_power(box["k"], box["mu"], p_real, k_edges, mu_edges)
    _, b_red, _ = bin_power(box["k"], box["mu"], p_red, k_edges, mu_edges)
    _, b_kaiser, _ = bin_power(box["k"], box["mu"], formula, k_edges, mu_edges)
    _, b_arinyo, _ = bin_power(box["k"], box["mu"], p_arinyo, k_edges, mu_edges)
    print("3D bin counts, mu<0.25", c_real[0].astype(int).tolist())
    print("3D bin counts, mu>0.75", c_real[-1].astype(int).tolist())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    panels = [(0, r"$\mu<0.25$"), (-1, r"$\mu>0.75$")]
    for ax, (row, title) in zip(axes, panels):
        ok = np.isfinite(b_red[row])
        ax.loglog(k_cent[ok], b_kaiser[row][ok], color="C0", label=r"Kaiser, $b_F=-0.14$, $\beta_F=1.4$")
        ax.loglog(k_cent[ok], b_arinyo[row][ok], color="C2", ls="--", label="Arinyo-i-Prats $z=3$")
        ax.loglog(k_cent[ok], b_real[row][ok], color="C1", ls=":", label="lognormal, real space")
        ax.loglog(k_cent[ok], b_red[row][ok], color="C3", label="lognormal, linear RSD")
        ax.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$P_F(k,\mu)$ [$({\mathrm{Mpc}}/h)^3$]")
    fig.suptitle(
        rf"One $40\,{{\mathrm{{Mpc}}}}/h$ box, $128^3$, $x_J=100$ kpc/$h$, "
        rf"$\langle F\rangle=0.696$, $z=3$"
    )
    fig.tight_layout()
    _save(fig, "14_p3d_kmu")
    plt.close(fig)

    # Ratios at the middle resolved bin, for the report.
    mid = len(k_cent) // 2
    for name, row in [("mu<0.25", 0), ("mu>0.75", -1)]:
        print(
            f"{name} at k={k_cent[mid]:.3f}: "
            f"real/kaiser={b_real[row][mid] / b_kaiser[row][mid]:.3f}, "
            f"rsd/kaiser={b_red[row][mid] / b_kaiser[row][mid]:.3f}, "
            f"arinyo/kaiser={b_arinyo[row][mid] / b_kaiser[row][mid]:.3f}, "
            f"rsd/arinyo={b_red[row][mid] / b_arinyo[row][mid]:.3f}"
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_z5()
    plot_box()


if __name__ == "__main__":
    main()
