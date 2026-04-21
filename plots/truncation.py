"""Plotting for fixed-flux truncation benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.truncation import TruncationBenchmarkResult
from plots.style import (
    DEFAULT_PLOT_FONT_SIZE,
    MODEL_ALPHA_CIRCUIT,
    MODEL_ALPHA_DUFFING,
    benchmark_plot_style,
    model_legend_handles,
)


def plot_truncation_benchmark(
    result: TruncationBenchmarkResult,
    outfile: Path,
    title: str,
    *,
    lowest_excited_levels_to_plot: int = 6,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    x = np.asarray(result.duffing_ncut_values, dtype=float)
    with benchmark_plot_style(font_size):
        fig = plt.figure(figsize=(11.0, 8.0))
        gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
        ax_j = fig.add_subplot(gs[0, 0])
        ax_zeta = fig.add_subplot(gs[0, 1], sharex=ax_j)
        ax_levels = fig.add_subplot(gs[1, 0], sharex=ax_j)
        ax_diff = fig.add_subplot(gs[1, 1], sharex=ax_j)

        ax_j.plot(x, result.duffing_j, color="C0", marker="o", linewidth=1.8, alpha=MODEL_ALPHA_DUFFING)
        ax_j.axhline(
            result.circuit_j,
            color="C0",
            linewidth=1.4,
            alpha=MODEL_ALPHA_CIRCUIT,
        )
        ax_j.set_ylabel(r"Exchange $J$ (GHz)")
        ax_j.grid(True, alpha=0.3)

        ax_zeta.plot(x, result.duffing_zeta, color="C2", marker="o", linewidth=1.8, alpha=MODEL_ALPHA_DUFFING)
        ax_zeta.axhline(
            result.circuit_zeta,
            color="C2",
            linewidth=1.4,
            alpha=MODEL_ALPHA_CIRCUIT,
        )
        ax_zeta.set_ylabel(r"Residual ZZ $\zeta$ (GHz)")
        ax_zeta.grid(True, alpha=0.3)

        ax_j.set_xlabel("Duffing transmon ncut")
        ax_zeta.set_xlabel("Duffing transmon ncut")

        rel_duf = np.asarray(result.duffing_lowest_relative_energies, dtype=float)
        rel_cir = np.asarray(result.circuit_lowest_relative_energies, dtype=float).ravel()
        n_levels = int(min(rel_duf.shape[1], rel_cir.shape[0]))
        n_excited_to_show = int(min(max(1, int(lowest_excited_levels_to_plot)), max(0, n_levels - 1)))
        if n_levels > 1:
            for i in range(1, 1 + n_excited_to_show):
                color = f"C{(i - 1) % 10}"
                label = rf"$E_{{{i}}}$"
                ax_levels.plot(x, rel_duf[:, i], color=color, linewidth=1.6, alpha=MODEL_ALPHA_DUFFING, label=label)
                ax_levels.axhline(rel_cir[i], color=color, linewidth=1.2, alpha=MODEL_ALPHA_CIRCUIT)

                ax_diff.plot(x, rel_duf[:, i] - rel_cir[i], color=color, linewidth=1.6, alpha=0.95, label=label)

            ax_levels.legend(loc="best", ncol=2, title="Levels")

            ax_diff.axhline(0.0, color="0.35", linewidth=1.0)
            ax_diff.legend(loc="best", ncol=2, title="Levels")
        else:
            ax_levels.text(0.5, 0.5, "Not enough levels to display", transform=ax_levels.transAxes, ha="center", va="center")
            ax_diff.text(0.5, 0.5, "Not enough levels to display", transform=ax_diff.transAxes, ha="center", va="center")
        ax_levels.set_ylabel("Energy (GHz, rel. ground)")
        ax_levels.set_xlabel("Duffing transmon ncut")
        ax_levels.grid(True, alpha=0.3)
        ax_diff.set_ylabel("Energy difference (GHz)")
        ax_diff.set_xlabel("Duffing transmon ncut")
        ax_diff.grid(True, alpha=0.3)

        fig.legend(handles=model_legend_handles(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
