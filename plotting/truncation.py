"""Plotting for circuit and Duffing static truncation-convergence benchmarks."""
#Circuit static truncation convergence over 5 flux points, compared over {int(result.lowest_excited_levels_compared)} excited levels
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


def _plot_metric_sweeps(
    ax,
    *,
    x: np.ndarray,
    total_rmse: np.ndarray,
    energy_rmse: np.ndarray,
    j_abs_error: np.ndarray,
    zeta_abs_error: np.ndarray,
    xlabel: str,
    title: str,
    xticklabels: list[str] | None = None,
) -> None:
    ax.plot(x, total_rmse, marker="o", linewidth=1.8, label="total_rmse")
    ax.plot(x, energy_rmse, marker="s", linewidth=1.6, label="energy_rmse")
    ax.plot(x, j_abs_error, marker="^", linewidth=1.6, label="|dJ|")
    ax.plot(x, zeta_abs_error, marker="d", linewidth=1.6, label="|dzeta|")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Error (GHz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
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
        fig, (ax_ncut, ax_trunc) = plt.subplots(1, 2, figsize=(11.8, 4.8))
        _plot_metric_sweeps(
            ax_ncut,
            x=np.asarray(result.circuit_ncut_values, dtype=float),
            total_rmse=np.asarray(result.circuit_ncut_total_rmse, dtype=float),
            energy_rmse=np.asarray(result.circuit_ncut_energy_rmse, dtype=float),
            j_abs_error=np.asarray(result.circuit_ncut_j_abs_error, dtype=float),
            zeta_abs_error=np.asarray(result.circuit_ncut_zeta_abs_error, dtype=float),
            xlabel="Circuit ncut",
            title="Circuit ncut",
        )
        trunc_x = np.arange(result.circuit_truncation_qubit_values.size, dtype=float)
        trunc_labels = [
            f"{int(q)}/{int(c)}"
            for q, c in zip(result.circuit_truncation_qubit_values, result.circuit_truncation_coupler_values)
        ]
        _plot_metric_sweeps(
            ax_trunc,
            x=trunc_x,
            total_rmse=np.asarray(result.circuit_truncation_total_rmse, dtype=float),
            energy_rmse=np.asarray(result.circuit_truncation_energy_rmse, dtype=float),
            j_abs_error=np.asarray(result.circuit_truncation_j_abs_error, dtype=float),
            zeta_abs_error=np.asarray(result.circuit_truncation_zeta_abs_error, dtype=float),
            xlabel="Circuit q/c truncated dims",
            title="Circuit Truncated-Dim",
            xticklabels=trunc_labels,
        )
        fig.suptitle(
            "Circuit static truncation convergence"
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
        fig, (ax_ncut, ax_trunc) = plt.subplots(1, 2, figsize=(11.8, 4.8))
        _plot_metric_sweeps(
            ax_ncut,
            x=np.asarray(result.duffing_ncut_values, dtype=float),
            total_rmse=np.asarray(result.duffing_ncut_total_rmse, dtype=float),
            energy_rmse=np.asarray(result.duffing_ncut_energy_rmse, dtype=float),
            j_abs_error=np.asarray(result.duffing_ncut_j_abs_error, dtype=float),
            zeta_abs_error=np.asarray(result.duffing_ncut_zeta_abs_error, dtype=float),
            xlabel="Duffing extraction ncut",
            title="Duffing ncut",
        )
        trunc_x = np.arange(result.duffing_hilbert_qubit_values.size, dtype=float)
        trunc_labels = [
            f"{int(q)}/{int(c)}"
            for q, c in zip(result.duffing_hilbert_qubit_values, result.duffing_hilbert_coupler_values)
        ]
        _plot_metric_sweeps(
            ax_trunc,
            x=trunc_x,
            total_rmse=np.asarray(result.duffing_hilbert_total_rmse, dtype=float),
            energy_rmse=np.asarray(result.duffing_hilbert_energy_rmse, dtype=float),
            j_abs_error=np.asarray(result.duffing_hilbert_j_abs_error, dtype=float),
            zeta_abs_error=np.asarray(result.duffing_hilbert_zeta_abs_error, dtype=float),
            xlabel="Duffing q/c Hilbert dims",
            title="Duffing Truncated-Dim",
            xticklabels=trunc_labels,
        )
        fig.suptitle(
            "Duffing static truncation convergence"
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD, w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
