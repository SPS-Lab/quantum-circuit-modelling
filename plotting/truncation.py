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


def _plot_metric_sweeps(
    ax,
    *,
    x: np.ndarray,
    energy_rmse: np.ndarray,
    j_abs_error: np.ndarray,
    zeta_abs_error: np.ndarray,
    xlabel: str,
    title: str,
    xticklabels: list[str] | None = None,
) -> None:
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
        fig, (ax_ncut, ax_q, ax_c) = plt.subplots(3, 1, figsize=(7.2, 12.8))
        _plot_metric_sweeps(
            ax_ncut,
            x=np.asarray(result.circuit_ncut_values, dtype=float),
            energy_rmse=np.asarray(result.circuit_ncut_energy_rmse, dtype=float),
            j_abs_error=np.asarray(result.circuit_ncut_j_abs_error, dtype=float),
            zeta_abs_error=np.asarray(result.circuit_ncut_zeta_abs_error, dtype=float),
            xlabel="ncut",
            title="ncut",
        )
        _plot_metric_sweeps(
            ax_q,
            x=np.asarray(result.circuit_qubit_truncated_dim_values, dtype=float),
            energy_rmse=np.asarray(result.circuit_qubit_truncation_energy_rmse, dtype=float),
            j_abs_error=np.asarray(result.circuit_qubit_truncation_j_abs_error, dtype=float),
            zeta_abs_error=np.asarray(result.circuit_qubit_truncation_zeta_abs_error, dtype=float),
            xlabel="qubit truncated dim",
            title="Q-Dim",
        )
        _plot_metric_sweeps(
            ax_c,
            x=np.asarray(result.circuit_coupler_truncated_dim_values, dtype=float),
            energy_rmse=np.asarray(result.circuit_coupler_truncation_energy_rmse, dtype=float),
            j_abs_error=np.asarray(result.circuit_coupler_truncation_j_abs_error, dtype=float),
            zeta_abs_error=np.asarray(result.circuit_coupler_truncation_zeta_abs_error, dtype=float),
            xlabel="coupler truncated dim",
            title="C-Dim",
        )
        fig.suptitle("Circuit static truncation convergence")
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
        fig, (ax_ncut, ax_q, ax_c) = plt.subplots(3, 1, figsize=(7.2, 12.8))
        _plot_metric_sweeps(
            ax_ncut,
            x=np.asarray(result.duffing_ncut_values, dtype=float),
            energy_rmse=np.asarray(result.duffing_ncut_energy_rmse, dtype=float),
            j_abs_error=np.asarray(result.duffing_ncut_j_abs_error, dtype=float),
            zeta_abs_error=np.asarray(result.duffing_ncut_zeta_abs_error, dtype=float),
            xlabel="extraction ncut",
            title="ncut",
        )
        _plot_metric_sweeps(
            ax_q,
            x=np.asarray(result.duffing_hilbert_qubit_dim_values, dtype=float),
            energy_rmse=np.asarray(result.duffing_hilbert_qubit_energy_rmse, dtype=float),
            j_abs_error=np.asarray(result.duffing_hilbert_qubit_j_abs_error, dtype=float),
            zeta_abs_error=np.asarray(result.duffing_hilbert_qubit_zeta_abs_error, dtype=float),
            xlabel="qubit Hilbert dim",
            title="Q-Dim",
        )
        _plot_metric_sweeps(
            ax_c,
            x=np.asarray(result.duffing_hilbert_coupler_dim_values, dtype=float),
            energy_rmse=np.asarray(result.duffing_hilbert_coupler_energy_rmse, dtype=float),
            j_abs_error=np.asarray(result.duffing_hilbert_coupler_j_abs_error, dtype=float),
            zeta_abs_error=np.asarray(result.duffing_hilbert_coupler_zeta_abs_error, dtype=float),
            xlabel="coupler Hilbert dim",
            title="C-Dim",
        )
        fig.suptitle("Duffing static truncation convergence")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD, w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
