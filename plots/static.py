"""Plotting for the static benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.static import StaticBenchmarkResult



def plot_static_benchmark(result: StaticBenchmarkResult, outfile: Path, title: str) -> None:
    flux = np.asarray(result.flux_values, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    axE, axErr, axJ, axZeta = axes.ravel()

    # Plot lower full-spectrum levels for context (faint lines), then emphasize
    # the tracked computational manifold on top.
    n_full = int(result.circuit_full_relative_energies.shape[1])
    if n_full > 4:
        for i in range(1, n_full):
            axE.plot(
                flux,
                result.circuit_full_relative_energies[:, i],
                color="0.55",
                linewidth=0.8,
                alpha=0.25,
            )
            axE.plot(
                flux,
                result.duffing_full_relative_energies[:, i],
                color="C0",
                linestyle="--",
                linewidth=0.7,
                alpha=0.14,
            )
        axE.plot([], [], color="0.55", linewidth=1.2, alpha=0.35, label="circuit lower levels")
        axE.plot([], [], color="C0", linestyle="--", linewidth=1.1, alpha=0.35, label="duffing lower levels")

    for i in (1, 2, 3):
        color = f"C{i - 1}"
        axE.plot(flux, result.circuit_relative_energies[:, i], color=color, linewidth=1.8, label=rf"circuit $E_{{{i}}}$")
        axE.plot(flux, result.duffing_relative_energies[:, i], color=color, linestyle="--", linewidth=1.4, label=rf"duffing $E_{{{i}}}$")
        axE.plot(flux, result.effective_relative_energies[:, i], color=color, linestyle=":", linewidth=1.4, label=rf"effective $E_{{{i}}}$")
    axE.set_ylabel("Energy (GHz, rel. ground)")
    axE.set_title("Dressed computational energies")
    axE.grid(True, alpha=0.3)
    axE.legend(loc="best", fontsize="small", ncol=3)

    axErr.plot(flux, result.effective_error_rmse, color="C3", linewidth=1.8, label="effective vs circuit")
    axErr.plot(flux, result.duffing_error_rmse, color="C0", linewidth=1.8, label="duffing vs circuit")
    y_max = float(max(np.max(result.effective_error_rmse), np.max(result.duffing_error_rmse)))
    if np.any(result.near_mask):
        axErr.fill_between(flux, 0.0, y_max * 1.05, where=result.near_mask, color="C3", alpha=0.08)
    if np.any(result.idle_mask):
        axErr.fill_between(flux, 0.0, y_max * 1.05, where=result.idle_mask, color="C0", alpha=0.05)
    axErr.set_ylabel("Per-flux RMSE (GHz)")
    axErr.set_title("Error to circuit reference")
    axErr.grid(True, alpha=0.3)
    axErr.legend(loc="best", fontsize="small")

    axJ.plot(flux, result.circuit_parameters["J"], color="C1", linewidth=1.8, label=r"circuit $J$")
    axJ.plot(flux, result.duffing_parameters["J"], color="C1", linestyle="--", linewidth=1.4, label=r"duffing $J$")
    axJ.plot(flux, result.effective_parameters["J"], color="C1", linestyle=":", linewidth=1.4, label=r"effective $J$")
    axJ.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    axJ.set_ylabel(r"Exchange $J$ (GHz)")
    axJ.set_title(r"Extracted $J$")
    axJ.grid(True, alpha=0.3)
    axJ.legend(loc="best", fontsize="small")

    axZeta.plot(flux, result.circuit_parameters["zeta"], color="C2", linewidth=1.8, label=r"circuit $\zeta$")
    axZeta.plot(flux, result.duffing_parameters["zeta"], color="C2", linestyle="--", linewidth=1.4, label=r"duffing $\zeta$")
    axZeta.plot(flux, result.effective_parameters["zeta"], color="C2", linestyle=":", linewidth=1.4, label=r"effective $\zeta$")
    axZeta.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    axZeta.set_ylabel(r"Residual ZZ $\zeta$ (GHz)")
    axZeta.set_title(r"Extracted $\zeta$")
    axZeta.grid(True, alpha=0.3)
    axZeta.legend(loc="best", fontsize="small")

    axes[1, 0].set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
    axes[1, 1].set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
    fig.suptitle(title)
    fig.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, format="pdf")
    plt.close(fig)
