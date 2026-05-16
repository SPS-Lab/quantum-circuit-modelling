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


def _charge_basis_dim_from_ncut(ncut_values: np.ndarray) -> np.ndarray:
    ncut = np.asarray(ncut_values, dtype=float)
    return np.asarray(2.0 * ncut + 1.0, dtype=float)


def _integer_ticklabels(values: np.ndarray) -> list[str]:
    arr = np.asarray(values, dtype=float).ravel()
    return [str(int(round(value))) for value in arr]


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
    ax.plot(x, energy_rmse, marker="s", linewidth=1.6, label="$RMSE_E$")
    ax.plot(x, j_abs_error, marker="^", linewidth=1.6, label=r"$|\Delta J|$")
    ax.plot(x, zeta_abs_error, marker="d", linewidth=1.6, label=r"$|\Delta \zeta|$")
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
    subplot_specs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, list[str] | None]] = []
    if np.asarray(result.circuit_ncut_values).size > 0:
        n_q_values = _charge_basis_dim_from_ncut(result.circuit_ncut_values)
        subplot_specs.append(
            (
                n_q_values,
                np.asarray(result.circuit_ncut_energy_rmse, dtype=float),
                np.asarray(result.circuit_ncut_j_abs_error, dtype=float),
                np.asarray(result.circuit_ncut_zeta_abs_error, dtype=float),
                r"$N_Q$",
                "Qubit charge basis",
                _integer_ticklabels(n_q_values),
            )
        )
    if np.asarray(result.circuit_qubit_truncated_dim_values).size > 0:
        subplot_specs.append(
            (
                np.asarray(result.circuit_qubit_truncated_dim_values, dtype=float),
                np.asarray(result.circuit_qubit_truncation_energy_rmse, dtype=float),
                np.asarray(result.circuit_qubit_truncation_j_abs_error, dtype=float),
                np.asarray(result.circuit_qubit_truncation_zeta_abs_error, dtype=float),
                "$N_{E,q}$",
                "Qubit energy basis",
                None,
            )
        )
    if np.asarray(result.circuit_coupler_truncated_dim_values).size > 0:
        subplot_specs.append(
            (
                np.asarray(result.circuit_coupler_truncated_dim_values, dtype=float),
                np.asarray(result.circuit_coupler_truncation_energy_rmse, dtype=float),
                np.asarray(result.circuit_coupler_truncation_j_abs_error, dtype=float),
                np.asarray(result.circuit_coupler_truncation_zeta_abs_error, dtype=float),
                "$N_{E,c}$",
                "Coupler energy basis",
                None,
            )
        )
    if not subplot_specs:
        raise ValueError("Circuit truncation plot requires at least one populated sweep")

    with benchmark_plot_style(font_size):
        fig_height = max(4.4, 4.2 * len(subplot_specs))
        fig, axes = plt.subplots(len(subplot_specs), 1, figsize=(7.2, fig_height))
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes], dtype=object)
        for ax, (x, energy_rmse, j_abs_error, zeta_abs_error, xlabel, title, xticklabels) in zip(axes, subplot_specs):
            _plot_metric_sweeps(
                ax,
                x=x,
                energy_rmse=energy_rmse,
                j_abs_error=j_abs_error,
                zeta_abs_error=zeta_abs_error,
                xlabel=xlabel,
                title=title,
                xticklabels=xticklabels,
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
    subplot_specs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, list[str] | None]] = []
    if np.asarray(result.duffing_ncut_values).size > 0:
        n_q_values = _charge_basis_dim_from_ncut(result.duffing_ncut_values)
        subplot_specs.append(
            (
                n_q_values,
                np.asarray(result.duffing_ncut_energy_rmse, dtype=float),
                np.asarray(result.duffing_ncut_j_abs_error, dtype=float),
                np.asarray(result.duffing_ncut_zeta_abs_error, dtype=float),
                r"$N_Q$",
                r"Extraction $N_Q$",
                _integer_ticklabels(n_q_values),
            )
        )
    if np.asarray(result.duffing_hilbert_qubit_dim_values).size > 0:
        subplot_specs.append(
            (
                np.asarray(result.duffing_hilbert_qubit_dim_values, dtype=float),
                np.asarray(result.duffing_hilbert_qubit_energy_rmse, dtype=float),
                np.asarray(result.duffing_hilbert_qubit_j_abs_error, dtype=float),
                np.asarray(result.duffing_hilbert_qubit_zeta_abs_error, dtype=float),
                "qubit Hilbert dim",
                "Q-Dim",
                None,
            )
        )
    if np.asarray(result.duffing_hilbert_coupler_dim_values).size > 0:
        subplot_specs.append(
            (
                np.asarray(result.duffing_hilbert_coupler_dim_values, dtype=float),
                np.asarray(result.duffing_hilbert_coupler_energy_rmse, dtype=float),
                np.asarray(result.duffing_hilbert_coupler_j_abs_error, dtype=float),
                np.asarray(result.duffing_hilbert_coupler_zeta_abs_error, dtype=float),
                "coupler Hilbert dim",
                "C-Dim",
                None,
            )
        )
    if not subplot_specs:
        raise ValueError("Duffing truncation plot requires at least one populated sweep")

    with benchmark_plot_style(font_size):
        fig_height = max(4.4, 4.2 * len(subplot_specs))
        fig, axes = plt.subplots(len(subplot_specs), 1, figsize=(7.2, fig_height))
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes], dtype=object)
        for ax, (x, energy_rmse, j_abs_error, zeta_abs_error, xlabel, title, xticklabels) in zip(axes, subplot_specs):
            _plot_metric_sweeps(
                ax,
                x=x,
                energy_rmse=energy_rmse,
                j_abs_error=j_abs_error,
                zeta_abs_error=zeta_abs_error,
                xlabel=xlabel,
                title=title,
                xticklabels=xticklabels,
            )
        fig.suptitle("Duffing static truncation convergence")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD, w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
