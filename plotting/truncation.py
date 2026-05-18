"""Plotting for circuit and Duffing static truncation-convergence benchmarks."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.truncation import (
    CircuitTruncationBenchmarkResult,
    DuffingTruncationBenchmarkResult,
    TruncationBenchmarkResult,
)
from plotting.style import (
    BENCHMARK_TIGHT_LAYOUT_H_PAD,
    BENCHMARK_TIGHT_LAYOUT_W_PAD,
    DEFAULT_PLOT_FONT_SIZE,
    benchmark_plot_style,
    TRUNCATION_METRIC_LEGEND_BBOX_TO_ANCHOR,
    TRUNCATION_METRIC_LEGEND_NCOL,
    truncation_metric_legend_handles,
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
    ax.plot(x, energy_rmse, marker="s", linewidth=1.6)
    ax.plot(x, j_abs_error, marker="^", linewidth=1.6)
    ax.plot(x, zeta_abs_error, marker="d", linewidth=1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Error (GHz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if xticklabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=25, ha="right")


def _circuit_subplot_specs(
    result: CircuitTruncationBenchmarkResult,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, list[str] | None]]:
    specs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, list[str] | None]] = {}
    if np.asarray(result.circuit_ncut_values).size > 0:
        n_q_values = _charge_basis_dim_from_ncut(result.circuit_ncut_values)
        specs["ncut"] = (
            n_q_values,
            np.asarray(result.circuit_ncut_energy_rmse, dtype=float),
            np.asarray(result.circuit_ncut_j_abs_error, dtype=float),
            np.asarray(result.circuit_ncut_zeta_abs_error, dtype=float),
            r"$N_Q$",
            r"Circuit: $N_Q$ sweep",
            _integer_ticklabels(n_q_values),
        )
    if np.asarray(result.circuit_qubit_truncated_dim_values).size > 0:
        specs["qubit"] = (
            np.asarray(result.circuit_qubit_truncated_dim_values, dtype=float),
            np.asarray(result.circuit_qubit_truncation_energy_rmse, dtype=float),
            np.asarray(result.circuit_qubit_truncation_j_abs_error, dtype=float),
            np.asarray(result.circuit_qubit_truncation_zeta_abs_error, dtype=float),
            r"$N_{E,q}$",
            r"Circuit: $N_{E,q}$ sweep",
            None,
        )
    if np.asarray(result.circuit_coupler_truncated_dim_values).size > 0:
        specs["coupler"] = (
            np.asarray(result.circuit_coupler_truncated_dim_values, dtype=float),
            np.asarray(result.circuit_coupler_truncation_energy_rmse, dtype=float),
            np.asarray(result.circuit_coupler_truncation_j_abs_error, dtype=float),
            np.asarray(result.circuit_coupler_truncation_zeta_abs_error, dtype=float),
            r"$N_{E,c}$",
            r"Circuit: $N_{E,c}$ sweep",
            None,
        )
    return specs


def _duffing_subplot_specs(
    result: DuffingTruncationBenchmarkResult,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, list[str] | None]]:
    specs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, list[str] | None]] = {}
    if np.asarray(result.duffing_ncut_values).size > 0:
        n_q_values = _charge_basis_dim_from_ncut(result.duffing_ncut_values)
        specs["ncut"] = (
            n_q_values,
            np.asarray(result.duffing_ncut_energy_rmse, dtype=float),
            np.asarray(result.duffing_ncut_j_abs_error, dtype=float),
            np.asarray(result.duffing_ncut_zeta_abs_error, dtype=float),
            r"$N_Q$",
            r"Duffing: $N_Q$ sweep",
            _integer_ticklabels(n_q_values),
        )
    if np.asarray(result.duffing_hilbert_qubit_dim_values).size > 0:
        specs["qubit"] = (
            np.asarray(result.duffing_hilbert_qubit_dim_values, dtype=float),
            np.asarray(result.duffing_hilbert_qubit_energy_rmse, dtype=float),
            np.asarray(result.duffing_hilbert_qubit_j_abs_error, dtype=float),
            np.asarray(result.duffing_hilbert_qubit_zeta_abs_error, dtype=float),
            r"$N_{E,q}$",
            r"Duffing: $N_{E,q}$ sweep",
            None,
        )
    if np.asarray(result.duffing_hilbert_coupler_dim_values).size > 0:
        specs["coupler"] = (
            np.asarray(result.duffing_hilbert_coupler_dim_values, dtype=float),
            np.asarray(result.duffing_hilbert_coupler_energy_rmse, dtype=float),
            np.asarray(result.duffing_hilbert_coupler_j_abs_error, dtype=float),
            np.asarray(result.duffing_hilbert_coupler_zeta_abs_error, dtype=float),
            r"$N_{E,c}$",
            r"Duffing: $N_{E,c}$ sweep",
            None,
        )
    return specs


def plot_circuit_truncation_benchmark(
    result: CircuitTruncationBenchmarkResult,
    outfile: Path,
    *,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    subplot_specs = list(_circuit_subplot_specs(result).values())
    if not subplot_specs:
        raise ValueError("Circuit truncation plot requires at least one populated sweep")

    with benchmark_plot_style(font_size):
        fig_height = max(4.4, 4.2 * len(subplot_specs))
        fig, axes = plt.subplots(len(subplot_specs), 1, figsize=(6.6, fig_height))
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
        fig.suptitle("Circuit static truncation convergence", y=0.982)
        fig.legend(
            handles=truncation_metric_legend_handles(),
            loc="upper center",
            bbox_to_anchor=TRUNCATION_METRIC_LEGEND_BBOX_TO_ANCHOR,
            ncol=TRUNCATION_METRIC_LEGEND_NCOL,
            frameon=True,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD, w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)


def plot_duffing_truncation_benchmark(
    result: DuffingTruncationBenchmarkResult,
    outfile: Path,
    *,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    subplot_specs = list(_duffing_subplot_specs(result).values())
    if not subplot_specs:
        raise ValueError("Duffing truncation plot requires at least one populated sweep")

    with benchmark_plot_style(font_size):
        fig_height = max(4.4, 4.2 * len(subplot_specs))
        fig, axes = plt.subplots(len(subplot_specs), 1, figsize=(6.6, fig_height))
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
        fig.suptitle("Duffing static truncation convergence", y=0.982)
        fig.legend(
            handles=truncation_metric_legend_handles(),
            loc="upper center",
            bbox_to_anchor=TRUNCATION_METRIC_LEGEND_BBOX_TO_ANCHOR,
            ncol=TRUNCATION_METRIC_LEGEND_NCOL,
            frameon=True,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD, w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)


def plot_truncation_benchmark(
    result: TruncationBenchmarkResult,
    outfile: Path,
    *,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    circuit_result = CircuitTruncationBenchmarkResult(**result.circuit)
    duffing_result = DuffingTruncationBenchmarkResult(**result.duffing)
    circuit_specs = _circuit_subplot_specs(circuit_result)
    duffing_specs = _duffing_subplot_specs(duffing_result)
    row_order = [name for name in ("ncut", "qubit", "coupler") if name in circuit_specs or name in duffing_specs]
    if not row_order:
        raise ValueError("Combined truncation plot requires at least one populated sweep")

    with benchmark_plot_style(font_size):
        fig_height = max(4.8, 3.9 * len(row_order))
        fig, axes = plt.subplots(len(row_order), 2, figsize=(12.2, fig_height), squeeze=False)
        for row_index, sweep_name in enumerate(row_order):
            left_ax = axes[row_index, 0]
            right_ax = axes[row_index, 1]
            if sweep_name in circuit_specs:
                x, energy_rmse, j_abs_error, zeta_abs_error, xlabel, title, xticklabels = circuit_specs[sweep_name]
                _plot_metric_sweeps(
                    left_ax,
                    x=x,
                    energy_rmse=energy_rmse,
                    j_abs_error=j_abs_error,
                    zeta_abs_error=zeta_abs_error,
                    xlabel=xlabel,
                    title=title,
                    xticklabels=xticklabels,
                )
            else:
                left_ax.axis("off")
            if sweep_name in duffing_specs:
                x, energy_rmse, j_abs_error, zeta_abs_error, xlabel, title, xticklabels = duffing_specs[sweep_name]
                _plot_metric_sweeps(
                    right_ax,
                    x=x,
                    energy_rmse=energy_rmse,
                    j_abs_error=j_abs_error,
                    zeta_abs_error=zeta_abs_error,
                    xlabel=xlabel,
                    title=title,
                    xticklabels=xticklabels,
                )
            else:
                right_ax.axis("off")
        fig.suptitle("Static truncation convergence", y=0.982)
        fig.legend(
            handles=truncation_metric_legend_handles(),
            loc="upper center",
            bbox_to_anchor=TRUNCATION_METRIC_LEGEND_BBOX_TO_ANCHOR,
            ncol=TRUNCATION_METRIC_LEGEND_NCOL,
            frameon=True,
        )
        fig.tight_layout(
            rect=(0.0, 0.0, 1.0, 0.91),
            h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD,
            w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD,
        )
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
