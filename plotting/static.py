"""Plotting for the static benchmark. All numerics in GHz."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from comparison.static import StaticBenchmarkResult
from plotting.style import (
    BENCHMARK_TIGHT_LAYOUT_H_PAD,
    BENCHMARK_TIGHT_LAYOUT_RECT,
    BENCHMARK_TIGHT_LAYOUT_W_PAD,
    DEFAULT_PLOT_FONT_SIZE,
    MODEL_LEGEND_BBOX_TO_ANCHOR,
    MODEL_ALPHA_CIRCUIT,
    MODEL_ALPHA_DUFFING,
    energy_level_alpha,
    STATIC_LEVEL_LEGEND_BBOX_TO_ANCHOR,
    STATIC_LEVEL_LEGEND_FONT_SCALE,
    STATIC_LEVEL_LEGEND_LOC,
    STATIC_LEVEL_LEGEND_NCOL,
    benchmark_plot_style,
    model_color,
    model_legend_handles,
    model_plot_kwargs,
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
                level_alpha = energy_level_alpha(i - 1)
                axE.plot(
                    flux,
                    result.circuit_full_relative_energies[:, i],
                    color=model_color("circuit"),
                    linewidth=0.8,
                    alpha=MODEL_ALPHA_CIRCUIT * level_alpha * 0.45,
                )
                axE.plot(
                    flux,
                    result.duffing_full_relative_energies[:, i],
                    color=model_color("duffing"),
                    linewidth=0.8,
                    alpha=MODEL_ALPHA_DUFFING * level_alpha * 0.45,
                )

        for i in (1, 2, 3):
            level_alpha = energy_level_alpha(i - 1)
            axE.plot(flux, result.circuit_relative_energies[:, i], linewidth=1.8, color=model_color("circuit"), alpha=MODEL_ALPHA_CIRCUIT * level_alpha)
            axE.plot(flux, result.duffing_relative_energies[:, i], linewidth=1.8, color=model_color("duffing"), alpha=MODEL_ALPHA_DUFFING * level_alpha)
            axE.plot(flux, result.effective_relative_energies[:, i], linewidth=1.8, color=model_color("effective"), alpha=model_plot_kwargs("effective")["alpha"] * level_alpha)
        axE.set_ylabel("Energy rel. ground")
        axE.grid(True, alpha=0.3)
        axE.legend(
            handles=[
                Line2D([0], [0], color="0.15", linewidth=1.8, alpha=energy_level_alpha(0), label=r"$E_{1}$"),
                Line2D([0], [0], color="0.15", linewidth=1.8, alpha=energy_level_alpha(1), label=r"$E_{2}$"),
                Line2D([0], [0], color="0.15", linewidth=1.8, alpha=energy_level_alpha(2), label=r"$E_{3}$"),
                Line2D([0], [0], color="0.15", linewidth=1.1, alpha=energy_level_alpha(3) * 0.7, label="other levels"),
            ],
            loc=STATIC_LEVEL_LEGEND_LOC,
            bbox_to_anchor=STATIC_LEVEL_LEGEND_BBOX_TO_ANCHOR,
            ncol=STATIC_LEVEL_LEGEND_NCOL,
            fontsize=font_size * STATIC_LEVEL_LEGEND_FONT_SCALE,
            framealpha=0.9,
            borderpad=0.25,
            labelspacing=0.25,
            handlelength=1.4,
            columnspacing=0.9,
            title="Levels (alpha)",
        )

        axErr.plot(flux, result.effective_error_rmse, linewidth=1.8, **model_plot_kwargs("effective"))
        axErr.plot(flux, result.duffing_error_rmse, linewidth=1.8, **model_plot_kwargs("duffing"))
        y_max = float(max(np.max(result.effective_error_rmse), np.max(result.duffing_error_rmse)))
        if np.any(result.near_mask):
            axErr.fill_between(flux, 0.0, y_max * 1.05, where=result.near_mask, color="C3", alpha=0.08)
        if np.any(result.idle_mask):
            axErr.fill_between(flux, 0.0, y_max * 1.05, where=result.idle_mask, color="C0", alpha=0.05)
        axErr.set_ylabel("Per-flux RMSE")
        axErr.grid(True, alpha=0.3)

        axJ.plot(flux, result.circuit_parameters["J"], linewidth=1.8, **model_plot_kwargs("circuit"))
        axJ.plot(flux, result.duffing_parameters["J"], linewidth=1.8, **model_plot_kwargs("duffing"))
        axJ.plot(flux, result.effective_parameters["J"], linewidth=1.8, **model_plot_kwargs("effective"))
        axJ.axhline(0.0, color="0.35", linewidth=1.0)
        axJ.set_ylabel(r"Exchange $J$")
        axJ.grid(True, alpha=0.3)

        axZeta.plot(flux, result.circuit_parameters["zeta"], linewidth=1.8, **model_plot_kwargs("circuit"))
        axZeta.plot(flux, result.duffing_parameters["zeta"], linewidth=1.8, **model_plot_kwargs("duffing"))
        axZeta.plot(flux, result.effective_parameters["zeta"], linewidth=1.8, **model_plot_kwargs("effective"))
        axZeta.axhline(0.0, color="0.35", linewidth=1.0)
        axZeta.set_ylabel(r"Residual ZZ $\zeta$")
        axZeta.grid(True, alpha=0.3)

        axes[1, 0].set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
        axes[1, 1].set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
        fig.legend(handles=model_legend_handles(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=MODEL_LEGEND_BBOX_TO_ANCHOR)
        fig.tight_layout(
            rect=BENCHMARK_TIGHT_LAYOUT_RECT,
            h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD,
            w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD,
        )

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
