#!/usr/bin/env python3
"""Check plots for the analytic P_1D calculator.

Run from the repository root:

    python make_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p1d.abgd import linear_matter_power_abgd, p1d_skm_abgd
from p1d.eisenstein_hu import linear_matter_power
from p1d.fgpa_field import SkewerConfig, run_fgpa_skewers
from p1d.flux import p1d_skm
from p1d.matter_table import TabulatedMatterPower
from p1d.projection import project_p1d

OUT = Path(__file__).resolve().parent / "plots"
OM = 0.31
OB = 0.049
H = 0.67
NS = 0.965
AS = 2.1e-9
Z = 3.0


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150)
    fig.savefig(OUT / f"{name}.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_pcdm()
    plot_p1d_parameters()
    plot_fractional()
    plot_redshifts()
    plot_abgd()
    plot_table_rescaling()
    plot_pipeline()
    print(f"Wrote plots to {OUT}")


def plot_pcdm():
    k = np.logspace(-3, 1, 400)
    fig, ax = plt.subplots(figsize=(7, 5))
    for ns, ls in [(0.90, "--"), (NS, "-"), (1.02, ":")]:
        p = linear_matter_power(k, 0.0, ns, AS, OM, OB, H)
        ax.loglog(k, p, ls=ls, label=rf"$n_s={ns:.3f}$")
    for fac, ls in [(0.8, "-."), (1.2, (0, (3, 1, 1, 1)))]:
        p = linear_matter_power(k, 0.0, NS, AS * fac, OM, OB, H)
        ax.loglog(k, p, ls=ls, color="0.35", label=rf"$A_s\times{fac}$")
    ax.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    ax.set_ylabel(r"$P_{\mathrm{CDM}}(k, z=0)$ [$({\mathrm{Mpc}}/h)^3$]")
    ax.set_title("Eisenstein-Hu linear spectrum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "01_pcdm_ns_As")
    plt.close(fig)


def plot_p1d_parameters():
    k = np.logspace(-3, np.log10(0.02), 80)
    fig, ax = plt.subplots(figsize=(7, 5))
    for ns, ls in [(0.90, "--"), (NS, "-"), (1.02, ":")]:
        p = p1d_skm(k, Z, ns, AS, OM, OB, H)
        ax.loglog(k, p, ls=ls, label=rf"$n_s={ns:.3f}$")
    for fac, ls in [(0.8, "-."), (1.2, (0, (3, 1, 1, 1)))]:
        p = p1d_skm(k, Z, NS, AS * fac, OM, OB, H)
        ax.loglog(k, p, ls=ls, color="0.35", label=rf"$A_s\times{fac}$")
    ax.set_xlabel(r"$k_\parallel$ [s/km]")
    ax.set_ylabel(r"$P_{1\mathrm{D}}$ [km/s]")
    ax.set_title(rf"Analytic $P_{{1D}}$ at $z={Z:.1f}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "02_p1d_ns_As")
    plt.close(fig)


def plot_fractional():
    k = np.logspace(-3, np.log10(0.02), 80)
    p0 = p1d_skm(k, Z, NS, AS, OM, OB, H)
    dns = 0.01
    p_ns = p1d_skm(k, Z, NS + dns, AS, OM, OB, H)
    das = 0.01
    p_as = p1d_skm(k, Z, NS, AS * (1.0 + das), OM, OB, H)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(k, (p_ns - p0) / p0, label=rf"$\Delta n_s=+{dns}$")
    ax.semilogx(k, (p_as - p0) / p0, label=rf"$\Delta A_s/A_s=+{das}$")
    ax.axhline(das, color="0.4", ls="--", lw=1, label=r"$\Delta A_s/A_s$")
    ax.set_xlabel(r"$k_\parallel$ [s/km]")
    ax.set_ylabel(r"$\Delta P_{1\mathrm{D}} / P_{1\mathrm{D}}$")
    ax.set_title(rf"Response at $z={Z:.1f}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "03_p1d_fractional")
    plt.close(fig)


def plot_redshifts():
    k = np.logspace(-3, np.log10(0.02), 80)
    fig, ax = plt.subplots(figsize=(7, 5))
    for z in (2.2, 3.0, 4.0):
        p = p1d_skm(k, z, NS, AS, OM, OB, H)
        ax.loglog(k, p, label=rf"$z={z:.1f}$")
    ax.set_xlabel(r"$k_\parallel$ [s/km]")
    ax.set_ylabel(r"$P_{1\mathrm{D}}$ [km/s]")
    ax.set_title("Redshift dependence at fixed cosmology")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "04_p1d_redshifts")
    plt.close(fig)


def plot_abgd():
    k_cdm = np.logspace(-2, 1, 300)
    k = np.logspace(-3, np.log10(0.02), 60)
    cases = [
        (0.0, 0.0, 0.0, 0.0, "0"),
        (0.15, 0.0, 0.0, 0.0, r"$\alpha=0.15$"),
        (0.0, 0.05, 0.0, 0.0, r"$\beta=0.05$"),
        (0.0, 0.0, 0.02, 0.0, r"$\gamma=0.02$"),
    ]
    p0 = linear_matter_power(k_cdm, 0.0, NS, AS, OM, OB, H)
    p1d_0 = p1d_skm(k, Z, NS, AS, OM, OB, H)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for alpha, beta, gamma, delta, label in cases:
        p = linear_matter_power_abgd(
            k_cdm, 0.0, NS, AS, OM, OB, H, alpha=alpha, beta=beta, gamma=gamma, delta=delta
        )
        p1 = p1d_skm_abgd(
            k, Z, NS, AS, OM, OB, H, alpha=alpha, beta=beta, gamma=gamma, delta=delta
        )
        axes[0].semilogx(k_cdm, p / p0 - 1.0, label=label)
        axes[1].semilogx(k, p1 / p1d_0 - 1.0, label=label)
    axes[0].axvline(1.0, color="0.5", ls=":", lw=1)
    axes[0].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    axes[0].set_ylabel(r"$P_{\mathrm{CDM}}/P_{\mathrm{EH}} - 1$")
    axes[0].set_title(r"Shape at $z=0$, pivot $k_p=1$")
    axes[1].set_xlabel(r"$k_\parallel$ [s/km]")
    axes[1].set_ylabel(r"$P_{1\mathrm{D}}/P_{1\mathrm{D,EH}} - 1$")
    axes[1].set_title(rf"$P_{{1D}}$ response at $z={Z:.1f}$")
    for ax in axes:
        ax.axhline(0.0, color="0.4", lw=1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "05_abgd_shape")
    plt.close(fig)
    zero = p1d_skm_abgd(k, Z, NS, AS, OM, OB, H)
    err = np.max(np.abs(zero / p1d_0 - 1.0))
    print(f"alpha=beta=gamma=delta=0 matches Eisenstein-Hu to {err:.2e}")


def plot_table_rescaling():
    table = TabulatedMatterPower.from_eisenstein_hu(Z, NS, AS, OM, OB, H)
    k = np.logspace(-2, 1, 300)
    p_fid = table(k)
    p_as = table(k, As=AS * 1.2)
    p_ns = table(k, ns=NS + 0.03)
    direct_as = linear_matter_power(k, Z, NS, AS * 1.2, OM, OB, H)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].loglog(k, p_fid, label="fiducial table")
    axes[0].loglog(k, p_as, label=r"$A_s\times 1.2$")
    axes[0].loglog(k, p_ns, label=r"$n_s+0.03$")
    axes[0].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    axes[0].set_ylabel(r"$P_{\mathrm{CDM}}$ [$({\mathrm{Mpc}}/h)^3$]")
    axes[0].set_title(rf"Rescaled table at $z={Z:.1f}$")
    axes[1].semilogx(k, 1.0e6 * (p_as / direct_as - 1.0))
    axes[1].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    axes[1].set_ylabel(r"$10^{6}\,(P_{\mathrm{table}}/P_{\mathrm{EH}}-1)$")
    axes[1].set_title(r"Table vs direct Eisenstein-Hu, $A_s\times 1.2$")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "06_table_rescaling")
    plt.close(fig)


def plot_pipeline():
    table = TabulatedMatterPower.from_eisenstein_hu(Z, NS, AS, OM, OB, H, kmax=80.0, n_k=3000)
    config = SkewerConfig(
        redshift=Z,
        As=AS,
        ns=NS,
        omega_m=OM,
        omega_b=OB,
        h=H,
        length_mpch=200.0,
        n_pixels=512,
        n_skewers=256,
        target_mean_flux=0.80,
        gamma=1.5,
        seed=7,
    )
    result = run_fgpa_skewers(table, config)
    q = np.geomspace(1.0e-2, 20.0, 600)
    pk = table(q)
    k_par = np.geomspace(0.05, 5.0, 80)
    matter_p1d = project_p1d(k_par, q, pk)

    k_skm = result["k_skm"]
    use = (k_skm > 1.0e-3) & (k_skm < 0.02)
    kaiser = p1d_skm(k_skm[use], Z, NS, AS, OM, OB, H)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    k3 = np.logspace(-2, 1, 300)
    axes[0, 0].loglog(k3, linear_matter_power(k3, Z, NS, AS, OM, OB, H))
    axes[0, 0].set_title(r"1. Eisenstein-Hu $P_{\mathrm{CDM}}(k,z)$")
    axes[0, 0].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    axes[0, 0].set_ylabel(r"$P_{\mathrm{CDM}}$ [$({\mathrm{Mpc}}/h)^3$]")

    axes[0, 1].loglog(k_par, matter_p1d)
    axes[0, 1].set_title(r"2. Isotropic matter $P_{1\mathrm{D}}$")
    axes[0, 1].set_xlabel(r"$k_\parallel$ [$h\,\mathrm{Mpc}^{-1}$]")
    axes[0, 1].set_ylabel(r"$P_{1\mathrm{D}}$ [Mpc/$h$]")

    axes[1, 0].loglog(k_skm[use], kaiser, label="Kaiser analytic")
    axes[1, 0].loglog(k_skm[use], result["p1d_flux_kms"][use], label="FGPA skewers")
    axes[1, 0].set_title(r"3. Flux $P_{1\mathrm{D}}$ at $z=3$")
    axes[1, 0].set_xlabel(r"$k_\parallel$ [s/km]")
    axes[1, 0].set_ylabel(r"$P_{1\mathrm{D}}$ [km/s]")
    axes[1, 0].legend(fontsize=8)

    x = np.arange(config.n_pixels) * result["pixel_mpch"]
    axes[1, 1].plot(x, result["sample_flux_skewer"], lw=0.8)
    axes[1, 1].axhline(result["mean_flux"], color="black", ls="--", lw=1, label="mean flux")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].set_title(
        rf"4. One skewer, $\langle F\rangle={result['mean_flux']:.3f}$, $A={result['fgpa_A']:.3g}$"
    )
    axes[1, 1].set_xlabel(r"$x$ [Mpc/$h$]")
    axes[1, 1].set_ylabel("transmitted flux")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.suptitle("Analytic Kaiser path and optical-depth FGPA path")
    fig.tight_layout()
    _save(fig, "07_pipeline_stages")
    plt.close(fig)

    shape_k = k_skm[use]
    pivot = np.argmin(np.abs(shape_k - 0.005))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(shape_k, kaiser / kaiser[pivot], label="Kaiser / Kaiser($k_p$)")
    flux = result["p1d_flux_kms"][use]
    ax.semilogx(shape_k, flux / flux[pivot], label="FGPA skewers / FGPA($k_p$)")
    ax.set_xlabel(r"$k_\parallel$ [s/km]")
    ax.set_ylabel(r"$P_{1\mathrm{D}}(k)/P_{1\mathrm{D}}(k_p)$, $k_p=0.005$ s/km")
    ax.set_title(r"Shape comparison at $z=3$, mean flux $0.80$, $\gamma=1.5$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "08_kaiser_vs_fgpa_shape")
    plt.close(fig)
    print(
        f"FGPA skewers: mean flux {result['mean_flux']:.6f}, "
        f"A {result['fgpa_A']:.6g}, beta {result['fgpa_beta']:.3f}, "
        f"density variance {result['density_variance']:.4g}"
    )


if __name__ == "__main__":
    main()
