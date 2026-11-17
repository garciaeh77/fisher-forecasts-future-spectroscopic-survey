#!/usr/bin/env python3
"""Small-scale extension plots. Does not change plots 01-08.

    python make_extension_plots.py

Writes plots/09 through plots/12.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p1d.eisenstein_hu import linear_matter_power
from p1d.fgpa_field import (
    SkewerConfig,
    calibrate_mean_flux,
    fgpa_exponent,
    flux_contrast,
    lognormal_density,
    prepare_tau_unit,
    transmitted_flux,
)
from p1d.flux import flux_power_3d, p1d_skm
from p1d.jeans import with_jeans
from p1d.matter_table import TabulatedMatterPower
from p1d.projection import project_p1d
from p1d.skewers import generate_gaussian_skewers, measure_p1d, rfft_modes

OUT = Path(__file__).resolve().parent / "plots"
OM, OB, H = 0.31, 0.049, 0.67
NS, AS, Z = 0.965, 2.1e-9, 3.0
XJ_100 = 0.10  # Mpc/h, 100 kpc/h comoving
XJ_50 = 0.05


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150)
    fig.savefig(OUT / f"{name}.pdf")
    print(f"wrote {name}")


def _base_table(kmax=120.0):
    return TabulatedMatterPower.from_eisenstein_hu(
        Z, NS, AS, OM, OB, H, kmin=1.0e-3, kmax=kmax, n_k=4000
    )


def _draw(table, length, n_pixels, n_skewers, seed):
    """Gaussian lines from the isotropic projection of ``table``. Returns modes too."""
    k_modes = rfft_modes(length, n_pixels)
    nonzero = k_modes > 0
    q = np.geomspace(table.kmin, table.kmax, 8192)
    pk = table(q, As=AS, ns=NS)
    target = np.zeros_like(k_modes)
    target[nonzero] = project_p1d(k_modes[nonzero], q, pk)
    gaussian = generate_gaussian_skewers(
        target, length_mpch=length, n_pixels=n_pixels, n_skewers=n_skewers, seed=seed
    )
    return k_modes, gaussian


def _flux_from_density(density, length, gamma=1.5, mean_flux=0.80):
    tau_unit = prepare_tau_unit(density, gamma=gamma)
    amplitude = calibrate_mean_flux(tau_unit, mean_flux)
    flux = transmitted_flux(tau_unit, amplitude)
    contrast = flux_contrast(flux)
    return {
        "A": amplitude,
        "mean_flux": float(np.mean(flux)),
        "p1d_mpc_h": measure_p1d(contrast, length),
        "flux": flux,
        "clipped": float(np.mean(density <= 1.0e-6)),
    }


def plot_3d(table):
    k = np.logspace(-3, np.log10(30.0), 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cases = [
        (0.0, "no Jeans", "-"),
        (XJ_100, r"$x_J=100\,\mathrm{kpc}/h$", "--"),
        (XJ_50, r"$x_J=50\,\mathrm{kpc}/h$", ":"),
    ]
    for xj, label, ls in cases:
        screened = table if xj == 0.0 else with_jeans(table, xj)
        axes[0].loglog(k, screened(k), ls=ls, label=label)
    axes[0].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    axes[0].set_ylabel(r"$P_{\mathrm{CDM}}$ [$({\mathrm{Mpc}}/h)^3$]")
    axes[0].set_title(rf"3D matter spectrum at $z={Z:.0f}$")
    axes[0].legend(fontsize=8)

    mu_values = [(0.0, r"$\mu=0$"), (1.0, r"$\mu=1$")]
    for mu, label in mu_values:
        axes[1].loglog(
            k,
            flux_power_3d(k, mu, Z, NS, AS, OM, OB, H),
            label=rf"Kaiser {label}",
        )
    screened = with_jeans(table, XJ_100)
    axes[1].loglog(
        k,
        (-0.14) ** 2 * (1.0 + 1.4 * 1.0**2) ** 2 * screened(k),
        ls="--",
        label=r"Kaiser $\mu=1$, $x_J=100\,\mathrm{kpc}/h$",
    )
    axes[1].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    axes[1].set_ylabel(r"$P_F(k,\mu)$ [$({\mathrm{Mpc}}/h)^3$]")
    axes[1].set_title(r"3D flux spectrum, $b_F=-0.14$, $\beta_F=1.4$")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "09_3d_power_spectrum")
    plt.close(fig)


def plot_gaussian_vs_lognormal(table):
    """Density power on short lines, so the Nyquist mode is past the Jeans scales."""
    length, n_pixels, n_skewers = 40.0, 1024, 512
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    summaries = []
    for ax, xj, title in [
        (axes[0], 0.0, "no Jeans"),
        (axes[1], XJ_100, r"$x_J=100\,\mathrm{kpc}/h$"),
        (axes[2], XJ_50, r"$x_J=50\,\mathrm{kpc}/h$"),
    ]:
        screened = table if xj == 0.0 else with_jeans(table, xj)
        k_modes, gaussian = _draw(screened, length, n_pixels, n_skewers, seed=7)
        lognormal = lognormal_density(gaussian)
        use = k_modes > 0
        p_g = measure_p1d(gaussian, length)
        p_l = measure_p1d(lognormal - 1.0, length)
        ax.loglog(k_modes[use], p_g[use], label=r"Gaussian $\delta$")
        ax.loglog(k_modes[use], p_l[use], label=r"lognormal $\Delta-1$")
        ax.set_title(title)
        ax.set_xlabel(r"$k_\parallel$ [$h\,\mathrm{Mpc}^{-1}$]")
        ax.grid(alpha=0.3)
        summaries.append(
            {
                "xj": xj,
                "var_g": float(np.var(gaussian)),
                "var_l": float(np.var(lognormal)),
                "neg_frac": float(np.mean(1.0 + gaussian <= 0.0)),
            }
        )
    axes[0].set_ylabel(r"density $P_{1\mathrm{D}}$ [Mpc/$h$]")
    axes[0].legend(fontsize=8)
    fig.suptitle(rf"Gaussian versus lognormal density, $L={length:.0f}$ Mpc/$h$, {n_skewers} lines")
    fig.tight_layout()
    _save(fig, "10_gaussian_vs_lognormal")
    plt.close(fig)
    for row in summaries:
        print(
            f"density xJ={row['xj']}: var Gaussian {row['var_g']:.4g}, "
            f"var lognormal {row['var_l']:.4g}, "
            f"fraction with 1+delta<=0 {row['neg_frac']:.3f}"
        )
    return summaries


def plot_flux_jeans(table):
    length, n_pixels, n_skewers = 40.0, 1024, 512
    config_kwargs = dict(
        redshift=Z,
        As=AS,
        ns=NS,
        omega_m=OM,
        omega_b=OB,
        h=H,
        length_mpch=length,
        n_pixels=n_pixels,
        n_skewers=n_skewers,
        target_mean_flux=0.80,
        gamma=1.5,
        seed=7,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    print("flux comparison, beta", fgpa_exponent(1.5))
    for xj, label, ls in [
        (0.0, "no Jeans", "-"),
        (XJ_100, r"$x_J=100\,\mathrm{kpc}/h$", "--"),
        (XJ_50, r"$x_J=50\,\mathrm{kpc}/h$", ":"),
    ]:
        screened = table if xj == 0.0 else with_jeans(table, xj)
        k_modes, gaussian = _draw(screened, length, n_pixels, n_skewers, seed=11)
        lognormal = lognormal_density(gaussian)
        flux = _flux_from_density(lognormal, length)
        # Kaiser on the same matter spectrum, same forest k.
        from p1d.flux import k_hmpc_to_skm, p1d_h_to_kms

        k_skm = k_hmpc_to_skm(k_modes, Z, OM, H)
        use = (k_skm > 1.0e-3) & (k_modes > 0) & (k_modes < 20.0)
        kaiser = p1d_skm(
            k_skm[use],
            Z,
            NS,
            AS,
            OM,
            OB,
            H,
            matter_power=lambda q, s=screened: s(q),
        )
        axes[0].loglog(k_skm[use], kaiser, ls=ls, label=f"Kaiser, {label}")
        axes[1].loglog(k_skm[use], p1d_h_to_kms(flux["p1d_mpc_h"], Z, OM, H)[use], ls=ls, label=label)
        print(
            f"flux xJ={xj}: A={flux['A']:.4g}, mean={flux['mean_flux']:.4f}, "
            f"lognormal var={float(np.var(lognormal)):.4g}"
        )
    for ax, title in [
        (axes[0], "Kaiser on the same $P_\\mathrm{CDM}$"),
        (axes[1], r"lognormal FGPA, $\langle F\rangle=0.80$"),
    ]:
        ax.set_xlabel(r"$k_\parallel$ [s/km]")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$P_{1\mathrm{D}}$ [km/s]")
    fig.suptitle("Small-scale flux power with a Jeans cutoff")
    fig.tight_layout()
    _save(fig, "11_flux_jeans_scales")
    plt.close(fig)
    del config_kwargs


def plot_noise(table):
    """Same geometry as plot 07. One draw of 10000 lines, averaged in subsets."""
    length, n_pixels, n_total = 200.0, 512, 10000
    k_modes, gaussian = _draw(table, length, n_pixels, n_total, seed=7)
    density = lognormal_density(gaussian)
    flux = _flux_from_density(density, length)
    from p1d.flux import k_hmpc_to_skm, p1d_h_to_kms

    k_skm = k_hmpc_to_skm(k_modes, Z, OM, H)
    use = (k_skm > 1.0e-3) & (k_skm < 0.02)
    full = p1d_h_to_kms(flux["p1d_mpc_h"], Z, OM, H)

    def subset_p1d(n):
        contrast = flux_contrast(flux["flux"][:n])
        return p1d_h_to_kms(measure_p1d(contrast, length), Z, OM, H)

    p256 = subset_p1d(256)
    p1024 = subset_p1d(1024)
    p4096 = subset_p1d(4096)
    p10000 = full
    kaiser = p1d_skm(k_skm[use], Z, NS, AS, OM, OB, H)

    # Scatter across independent blocks of 256 lines, relative to each block's own spectrum.
    block = 256
    n_blocks = n_total // block
    blocks = np.stack(
        [
            p1d_h_to_kms(
                measure_p1d(flux_contrast(flux["flux"][i * block : (i + 1) * block]), length),
                Z,
                OM,
                H,
            )
            for i in range(n_blocks)
        ]
    )
    frac_std_256 = np.std(blocks[:, use] / np.mean(blocks[:, use], axis=0), axis=0)

    def frac_rms(p):
        return float(np.sqrt(np.mean((p[use] / p10000[use] - 1.0) ** 2)))

    rms256 = frac_rms(p256)
    rms1024 = frac_rms(p1024)
    rms4096 = frac_rms(p4096)
    median_scatter = float(np.median(frac_std_256))
    print(
        f"noise: rms vs 10000-line mean is {rms256:.3f} at 256 lines, "
        f"{rms1024:.3f} at 1024 lines, {rms4096:.3f} at 4096 lines"
    )
    print(
        f"noise: median fractional scatter among {n_blocks} independent "
        f"256-line averages is {median_scatter:.3f}"
    )
    print(f"noise: A={flux['A']:.4g}, mean flux={flux['mean_flux']:.4f}, beta={fgpa_exponent(1.5):.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k_skm[use], kaiser, color="C0", label="Kaiser analytic")
    ax.loglog(k_skm[use], p256[use], color="C1", alpha=0.35, label="256 lines (figure 07)")
    ax.loglog(k_skm[use], p1024[use], color="C1", alpha=0.55, ls="--", label="1024 lines")
    ax.loglog(k_skm[use], p4096[use], color="C1", alpha=0.8, ls=":", label="4096 lines")
    ax.loglog(k_skm[use], p10000[use], color="C1", label="10000 lines")
    ax.set_xlabel(r"$k_\parallel$ [s/km]")
    ax.set_ylabel(r"$P_{1\mathrm{D}}$ [km/s]")
    ax.set_title("Figure 07 lower left, more lines")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "12_skewer_noise")
    plt.close(fig)
    return {
        "rms256": rms256,
        "rms1024": rms1024,
        "rms4096": rms4096,
        "median_scatter_256": median_scatter,
        "ratio_256_1024": rms256 / rms1024 if rms1024 else np.nan,
        "ratio_256_4096": rms256 / rms4096 if rms4096 else np.nan,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = _base_table(kmax=120.0)
    # Direct check that the unsmoothed 3D curve is the Eisenstein-Hu evaluation.
    k_check = np.array([0.1, 1.0])
    direct = linear_matter_power(k_check, Z, NS, AS, OM, OB, H)
    np.testing.assert_allclose(table(k_check), direct, rtol=2.0e-3)
    plot_3d(table)
    plot_gaussian_vs_lognormal(table)
    plot_flux_jeans(table)
    stats = plot_noise(table)
    print("noise ratio 256/1024", stats["ratio_256_1024"])
    print("noise ratio 256/4096", stats["ratio_256_4096"])


if __name__ == "__main__":
    main()
