"""Plotting for static truncation-convergence benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.truncation import TruncationBenchmarkResult
from plotting.style import (
    BENCHMARK_TIGHT_LAYOUT_H_PAD,
    BENCHMARK_TIGHT_LAYOUT_W_PAD,
    DEFAULT_PLOT_FONT_SIZE,
    benchmark_plot_style,
)


def _plot_scalar_sweep(
    ax,
    *,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    title: str,
    xticklabels: list[str] | None = None,
) -> None:
    ax.plot(x, y, marker="o", linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Static RMSE (GHz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if xticklabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=25, ha="right")


def plot_truncation_benchmark(
    result: TruncationBenchmarkResult,
    outfile: Path,
    *,
    lowest_excited_levels_to_plot: int,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    del lowest_excited_levels_to_plot  # Kept for API compatibility.

    with benchmark_plot_style(font_size):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
        ax_c_ncut, ax_c_trunc, ax_d_ncut, ax_d_trunc = axes.ravel()

        circuit_ncut_x = np.asarray(result.circuit_ncut_values, dtype=float)
        _plot_scalar_sweep(
            ax_c_ncut,
            x=circuit_ncut_x,
            y=np.asarray(result.circuit_ncut_total_rmse, dtype=float),
            xlabel="Circuit ncut",
            title="Circuit ncut Convergence",
        )

        circuit_trunc_x = np.arange(result.circuit_truncation_qubit_values.size, dtype=float)
        circuit_trunc_labels = [
            f"{int(q)}/{int(c)}"
            for q, c in zip(result.circuit_truncation_qubit_values, result.circuit_truncation_coupler_values)
        ]
        _plot_scalar_sweep(
            ax_c_trunc,
            x=circuit_trunc_x,
            y=np.asarray(result.circuit_truncation_total_rmse, dtype=float),
            xlabel="Circuit q/c truncated dims",
            title="Circuit Truncated-Dim Convergence",
            xticklabels=circuit_trunc_labels,
        )

        duffing_ncut_x = np.asarray(result.duffing_ncut_values, dtype=float)
        _plot_scalar_sweep(
            ax_d_ncut,
            x=duffing_ncut_x,
            y=np.asarray(result.duffing_ncut_total_rmse, dtype=float),
            xlabel="Duffing extraction ncut",
            title="Duffing ncut Convergence",
        )

        duffing_trunc_x = np.arange(result.duffing_hilbert_qubit_values.size, dtype=float)
        duffing_trunc_labels = [
            f"{int(q)}/{int(c)}"
            for q, c in zip(result.duffing_hilbert_qubit_values, result.duffing_hilbert_coupler_values)
        ]
        _plot_scalar_sweep(
            ax_d_trunc,
            x=duffing_trunc_x,
            y=np.asarray(result.duffing_hilbert_total_rmse, dtype=float),
            xlabel="Duffing q/c Hilbert dims",
            title="Duffing Truncated-Dim Convergence",
            xticklabels=duffing_trunc_labels,
        )

        fig.suptitle(
            "Static truncation convergence at "
            f"flux={float(result.flux):.6f}, compared over {int(result.lowest_excited_levels_compared)} excited levels"
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96), h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD, w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
