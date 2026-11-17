"""Command-line interface for the mini P1D pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .pipeline import PipelineConfig, run_pipeline
from .power import TabulatedMatterPower, demo_cdm_power, load_power_table


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project a tabulated CDM P3D and generate approximate FGPA flux P1D"
    )
    parser.add_argument("--pk-file", help="two-column k[h/Mpc], P[(Mpc/h)^3] file")
    parser.add_argument("--output", default="output", help="output directory")
    parser.add_argument("--As", type=float, default=2.1e-9)
    parser.add_argument("--ns", type=float, default=0.965)
    parser.add_argument("--As-fid", type=float, default=2.1e-9)
    parser.add_argument("--ns-fid", type=float, default=0.965)
    parser.add_argument("--kpivot", type=float, default=0.05 / 0.674)
    parser.add_argument("--growth-ratio", type=float, default=1.0)
    parser.add_argument("--redshift", type=float, default=3.0)
    parser.add_argument("--length", type=float, default=200.0, help="Mpc/h")
    parser.add_argument("--npix", type=int, default=1024)
    parser.add_argument("--nskewers", type=int, default=512)
    parser.add_argument("--mean-flux", type=float, default=0.80)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--pressure-sigma", type=float, default=0.0, help="pixels")
    parser.add_argument("--thermal-sigma", type=float, default=0.0, help="pixels")
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def _save_plot(output: Path, result: dict, config: PipelineConfig) -> None:
    k = result["k_hmpc"]
    select = k > 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].loglog(
        k[select], result["p1d_matter_target_mpc_h"][select], label="Target matter P1D"
    )
    axes[0].loglog(
        k[select], result["p1d_matter_measured_mpc_h"][select], "--", label="Measured skewers"
    )
    axes[0].loglog(
        k[select], result["p1d_flux_mpc_h"][select], label="FGPA flux P1D"
    )
    axes[0].set(xlabel=r"$k_\parallel\ [h\,\mathrm{Mpc}^{-1}]$", ylabel=r"$P_{1D}\ [\mathrm{Mpc}/h]$")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    x = np.arange(config.n_pixels) * config.length_mpch / config.n_pixels
    axes[1].plot(x, result["sample_flux_skewer"], lw=1)
    axes[1].axhline(result["mean_flux"], color="black", ls="--", lw=1, label="Mean flux")
    axes[1].set(xlabel=r"$x\ [\mathrm{Mpc}/h]$", ylabel="Transmitted flux")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.suptitle(f"Mini FGPA P1D prototype, z={config.redshift:g}")
    fig.tight_layout()
    fig.savefig(output / "diagnostic.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parser().parse_args()
    if args.pk_file:
        k, pk = load_power_table(args.pk_file)
        source = str(Path(args.pk_file).resolve())
    else:
        k = np.geomspace(1.0e-3, 100.0, 5000)
        pk = demo_cdm_power(k)
        source = "internal smooth demonstration spectrum (not precision cosmology)"

    matter_power = TabulatedMatterPower(
        k,
        pk,
        As_fid=args.As_fid,
        ns_fid=args.ns_fid,
        kpivot_hmpc=args.kpivot,
    )
    config = PipelineConfig(
        redshift=args.redshift,
        As=args.As,
        ns=args.ns,
        growth_ratio=args.growth_ratio,
        length_mpch=args.length,
        n_pixels=args.npix,
        n_skewers=args.nskewers,
        target_mean_flux=args.mean_flux,
        gamma=args.gamma,
        pressure_sigma_pixels=args.pressure_sigma,
        thermal_sigma_pixels=args.thermal_sigma,
        seed=args.seed,
    )
    result = run_pipeline(matter_power, config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    np.savez(output / "mini_p1d_results.npz", **arrays)
    _save_plot(output, result, config)
    summary = (
        f"power_source: {source}\n"
        f"redshift: {config.redshift}\n"
        f"As: {config.As:.10g}\n"
        f"ns: {config.ns:.10g}\n"
        f"growth_ratio: {config.growth_ratio:.10g}\n"
        f"target_mean_flux: {config.target_mean_flux:.10g}\n"
        f"realized_mean_flux: {result['mean_flux']:.10g}\n"
        f"gamma: {config.gamma:.10g}\n"
        f"fgpa_A: {result['fgpa_A']:.10g}\n"
        f"density_mean: {result['density_mean']:.10g}\n"
        f"density_variance: {result['density_variance']:.10g}\n"
        "rsd: false (no velocity-displacement input supplied by CLI)\n"
    )
    (output / "summary.txt").write_text(summary)
    print(summary, end="")
    print(f"saved: {output / 'mini_p1d_results.npz'}")
    print(f"saved: {output / 'diagnostic.png'}")


if __name__ == "__main__":
    main()

