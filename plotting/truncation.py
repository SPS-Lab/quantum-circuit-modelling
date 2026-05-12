"""Plotting for circuit and Duffing static truncation-convergence benchmarks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.truncation import CircuitTruncationBenchmarkResult, DuffingTruncationBenchmarkResult
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


def plot_circuit_truncation_benchmark(
    result: CircuitTruncationBenchmarkResult,
    outfile: Path,
    *,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    with benchmark_plot_style(font_size):
        fig, (ax_ncut, ax_trunc) = plt.subplots(1, 2, figsize=(9.8, 4.4))
        _plot_scalar_sweep(
            ax_ncut,
            x=np.asarray(result.circuit_ncut_values, dtype=float),
            y=np.asarray(result.circuit_ncut_total_rmse, dtype=float),
            xlabel="Circuit ncut",
            title="Circuit ncut Convergence",
        )
        trunc_x = np.arange(result.circuit_truncation_qubit_values.size, dtype=float)
        trunc_labels = [
            f"{int(q)}/{int(c)}"
            for q, c in zip(result.circuit_truncation_qubit_values, result.circuit_truncation_coupler_values)
        ]
        _plot_scalar_sweep(
            ax_trunc,
            x=trunc_x,
            y=np.asarray(result.circuit_truncation_total_rmse, dtype=float),
            xlabel="Circuit q/c truncated dims",
            title="Circuit Truncated-Dim Convergence",
            xticklabels=trunc_labels,
        )
        fig.suptitle(
            "Circuit static truncation convergence over "
            f"{int(result.flux_values.size)} flux points, compared over {int(result.lowest_excited_levels_compared)} excited levels"
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD, w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)


def plot_duffing_truncation_benchmark(
    result: DuffingTruncationBenchmarkResult,
    outfile: Path,
    *,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    with benchmark_plot_style(font_size):
        fig, (ax_ncut, ax_trunc) = plt.subplots(1, 2, figsize=(9.8, 4.4))
        _plot_scalar_sweep(
            ax_ncut,
            x=np.asarray(result.duffing_ncut_values, dtype=float),
            y=np.asarray(result.duffing_ncut_total_rmse, dtype=float),
            xlabel="Duffing extraction ncut",
            title="Duffing ncut Convergence",
        )
        trunc_x = np.arange(result.duffing_hilbert_qubit_values.size, dtype=float)
        trunc_labels = [
            f"{int(q)}/{int(c)}"
            for q, c in zip(result.duffing_hilbert_qubit_values, result.duffing_hilbert_coupler_values)
        ]
        _plot_scalar_sweep(
            ax_trunc,
            x=trunc_x,
            y=np.asarray(result.duffing_hilbert_total_rmse, dtype=float),
            xlabel="Duffing q/c Hilbert dims",
            title="Duffing Truncated-Dim Convergence",
            xticklabels=trunc_labels,
        )
        fig.suptitle(
            "Duffing static truncation convergence over "
            f"{int(result.flux_values.size)} flux points, compared over {int(result.lowest_excited_levels_compared)} excited levels"
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD, w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
