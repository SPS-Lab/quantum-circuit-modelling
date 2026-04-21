"""Plotting for the static benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from comparison.static import StaticBenchmarkResult
from plots.style import (
    DEFAULT_PLOT_FONT_SIZE,
    MODEL_ALPHA_CIRCUIT,
    MODEL_ALPHA_DUFFING,
    MODEL_ALPHA_EFFECTIVE,
    benchmark_plot_style,
    model_legend_handles,
)



def plot_static_benchmark(
    result: StaticBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    flux = np.asarray(result.flux_values, dtype=float)

    with benchmark_plot_style(font_size):
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
                    alpha=MODEL_ALPHA_CIRCUIT * 0.28,
                )
                axE.plot(
                    flux,
                    result.duffing_full_relative_energies[:, i],
                    color="0.55",
                    linewidth=0.8,
                    alpha=MODEL_ALPHA_DUFFING * 0.28,
                )

        for i in (1, 2, 3):
            color = f"C{i - 1}"
            axE.plot(flux, result.circuit_relative_energies[:, i], color=color, linewidth=1.8, alpha=MODEL_ALPHA_CIRCUIT)
            axE.plot(flux, result.duffing_relative_energies[:, i], color=color, linewidth=1.8, alpha=MODEL_ALPHA_DUFFING)
            axE.plot(flux, result.effective_relative_energies[:, i], color=color, linewidth=1.8, alpha=MODEL_ALPHA_EFFECTIVE)
        axE.set_ylabel("Energy (GHz, rel. ground)")
        axE.grid(True, alpha=0.3)
        axE.legend(
            handles=[
                Line2D([0], [0], color="C0", linewidth=1.8, label=r"$E_{1}$"),
                Line2D([0], [0], color="C1", linewidth=1.8, label=r"$E_{2}$"),
                Line2D([0], [0], color="C2", linewidth=1.8, label=r"$E_{3}$"),
                Line2D([0], [0], color="0.55", linewidth=1.1, alpha=0.4, label="lower levels"),
            ],
            loc="best",
            ncol=2,
        )

        axErr.plot(flux, result.effective_error_rmse, color="k", linewidth=1.8, alpha=MODEL_ALPHA_EFFECTIVE)
        axErr.plot(flux, result.duffing_error_rmse, color="k", linewidth=1.8, alpha=MODEL_ALPHA_DUFFING)
        y_max = float(max(np.max(result.effective_error_rmse), np.max(result.duffing_error_rmse)))
        if np.any(result.near_mask):
            axErr.fill_between(flux, 0.0, y_max * 1.05, where=result.near_mask, color="C3", alpha=0.08)
        if np.any(result.idle_mask):
            axErr.fill_between(flux, 0.0, y_max * 1.05, where=result.idle_mask, color="C0", alpha=0.05)
        axErr.set_ylabel("Per-flux RMSE (GHz)")
        axErr.grid(True, alpha=0.3)

        axJ.plot(flux, result.circuit_parameters["J"], color="C1", linewidth=1.8, alpha=MODEL_ALPHA_CIRCUIT)
        axJ.plot(flux, result.duffing_parameters["J"], color="C1", linewidth=1.8, alpha=MODEL_ALPHA_DUFFING)
        axJ.plot(flux, result.effective_parameters["J"], color="C1", linewidth=1.8, alpha=MODEL_ALPHA_EFFECTIVE)
        axJ.axhline(0.0, color="0.35", linewidth=1.0)
        axJ.set_ylabel(r"Exchange $J$ (GHz)")
        axJ.grid(True, alpha=0.3)

        axZeta.plot(flux, result.circuit_parameters["zeta"], color="C2", linewidth=1.8, alpha=MODEL_ALPHA_CIRCUIT)
        axZeta.plot(flux, result.duffing_parameters["zeta"], color="C2", linewidth=1.8, alpha=MODEL_ALPHA_DUFFING)
        axZeta.plot(flux, result.effective_parameters["zeta"], color="C2", linewidth=1.8, alpha=MODEL_ALPHA_EFFECTIVE)
        axZeta.axhline(0.0, color="0.35", linewidth=1.0)
        axZeta.set_ylabel(r"Residual ZZ $\zeta$ (GHz)")
        axZeta.grid(True, alpha=0.3)

        axes[1, 0].set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
        axes[1, 1].set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
        fig.legend(handles=model_legend_handles(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
