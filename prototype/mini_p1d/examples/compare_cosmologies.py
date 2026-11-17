"""Make parameter-comparison plots for the mini-P1D FGPA prototype.

The internal smooth CDM-like spectrum is deliberately only a demonstration.
For scientific work, replace it with CLASS or CAMB power-spectrum tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mini_p1d.pipeline import PipelineConfig, run_pipeline
from mini_p1d.power import TabulatedMatterPower, demo_cdm_power


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="comparison_plots", help="output directory")
    parser.add_argument("--nskewers", type=int, default=512)
    parser.add_argument("--npix", type=int, default=1024)
    parser.add_argument("--length", type=float, default=200.0, help="Mpc/h")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def build_model() -> TabulatedMatterPower:
    k = np.geomspace(1.0e-3, 100.0, 5000)
    return TabulatedMatterPower(k, demo_cdm_power(k))


def run_case(
    model: TabulatedMatterPower,
    *,
    As: float,
    ns: float,
    args: argparse.Namespace,
) -> dict[str, np.ndarray | float]:
    config = PipelineConfig(
        redshift=3.0,
        As=As,
        ns=ns,
        length_mpch=args.length,
        n_pixels=args.npix,
        n_skewers=args.nskewers,
        target_mean_flux=0.80,
        gamma=1.5,
        thermal_sigma_pixels=1.0,
        seed=args.seed,
    )
    return run_pipeline(model, config)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = build_model()

    cases = [
        (1.8e-9, 0.965, r"$A_s=1.8\times10^{-9}$"),
        (2.1e-9, 0.965, r"$A_s=2.1\times10^{-9}$"),
        (2.4e-9, 0.965, r"$A_s=2.4\times10^{-9}$"),
        (2.1e-9, 0.925, r"$n_s=0.925$"),
        (2.1e-9, 1.005, r"$n_s=1.005$"),
    ]
    results = [
        (label, run_case(model, As=As, ns=ns, args=args))
        for As, ns, label in cases
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for label, result in results:
        q = result["q_hmpc"]
        k = result["k_hmpc"]
        positive = k > 0
        axes[0].loglog(q, result["pk3d_mpc3h3"], label=label)
        axes[1].loglog(
            k[positive], result["p1d_matter_target_mpc_h"][positive], label=label
        )
    axes[0].set(
        xlabel=r"$k\ [h\,\mathrm{Mpc}^{-1}]$",
        ylabel=r"$P_{\rm m}^{3D}(k)\ [({\rm Mpc}/h)^3]$",
    )
    axes[1].set(
        xlabel=r"$k_\parallel\ [h\,\mathrm{Mpc}^{-1}]$",
        ylabel=r"$P_{\rm m}^{1D}(k_\parallel)\ [{\rm Mpc}/h]$",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "matter_power_comparison.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.8, 4.8))
    for label, result in results:
        k = result["k_hmpc"]
        positive = k > 0
        axis.loglog(k[positive], result["p1d_flux_mpc_h"][positive], label=label)
    axis.set(
        xlabel=r"$k_\parallel\ [h\,\mathrm{Mpc}^{-1}]$",
        ylabel=r"$P_F^{1D}(k_\parallel)\ [{\rm Mpc}/h]$",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "flux_p1d_comparison.png", dpi=200)
    plt.close(fig)

    print(f"saved: {output / 'matter_power_comparison.png'}")
    print(f"saved: {output / 'flux_p1d_comparison.png'}")


if __name__ == "__main__":
    main()

