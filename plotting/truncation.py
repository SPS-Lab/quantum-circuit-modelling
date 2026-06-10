"""Plotting for circuit and Duffing static truncation-convergence benchmarks."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import numpy as np

from comparison.truncation import (
    CircuitTruncationBenchmarkResult,
    DuffingTruncationBenchmarkResult,
    TruncationBenchmarkResult,
)
from plotting.style import (
    add_column_title,
    benchmark_tight_layout,
    figure_legend_bbox_to_anchor,
    render_benchmark_figures,
    stacked_figure_size,
    truncation_metric_plot_kwargs,
)

def _truncation_metric_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], label=r"$RMSE_{E,\mathrm{comp}}$", **truncation_metric_plot_kwargs("energy_rmse")),
        Line2D([0], [0], label=r"$|\Delta J|$", **truncation_metric_plot_kwargs("j_abs_error")),
        Line2D([0], [0], label=r"$|\Delta \zeta|$", **truncation_metric_plot_kwargs("zeta_abs_error")),
    ]


def _add_truncation_metric_figure_legend(fig: plt.Figure):
    return fig.legend(
        handles=_truncation_metric_legend_handles(),
        loc="upper center",
        bbox_to_anchor=figure_legend_bbox_to_anchor(fig),
        ncol=3
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
    title: str | None,
    ylabel: str = "Error",
    xticklabels: list[str] | None = None,
) -> None:
    y_series = (
        np.asarray(energy_rmse, dtype=float),
        np.asarray(j_abs_error, dtype=float),
        np.asarray(zeta_abs_error, dtype=float),
    )
    y_max = max(float(np.nanmax(np.abs(values))) for values in y_series if values.size > 0)
    ax.plot(x, energy_rmse, **truncation_metric_plot_kwargs("energy_rmse"))
    ax.plot(x, j_abs_error, **truncation_metric_plot_kwargs("j_abs_error"))
    ax.plot(x, zeta_abs_error, **truncation_metric_plot_kwargs("zeta_abs_error"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if y_max > 0.0:
        ax.yaxis.set_major_locator(MultipleLocator(3.0 * 10.0 ** np.floor(np.log10(y_max))))
    ax.grid()
    if xticklabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)


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
            r"Charge dimension ($N_Q$)",
            r"Circuit: $N_Q$ sweep",
            _integer_ticklabels(n_q_values),
        )
    if np.asarray(result.circuit_qubit_truncated_dim_values).size > 0:
        qubit_dims = np.asarray(result.circuit_qubit_truncated_dim_values, dtype=float)
        specs["qubit"] = (
            qubit_dims,
            np.asarray(result.circuit_qubit_truncation_energy_rmse, dtype=float),
            np.asarray(result.circuit_qubit_truncation_j_abs_error, dtype=float),
            np.asarray(result.circuit_qubit_truncation_zeta_abs_error, dtype=float),
            r"Qubit truncation ($N_{E,q}$)",
            r"Circuit: $N_{E,q}$ sweep",
            _integer_ticklabels(qubit_dims),
        )
    if np.asarray(result.circuit_coupler_truncated_dim_values).size > 0:
        coupler_dims = np.asarray(result.circuit_coupler_truncated_dim_values, dtype=float)
        specs["coupler"] = (
            coupler_dims,
            np.asarray(result.circuit_coupler_truncation_energy_rmse, dtype=float),
            np.asarray(result.circuit_coupler_truncation_j_abs_error, dtype=float),
            np.asarray(result.circuit_coupler_truncation_zeta_abs_error, dtype=float),
            r"Coupler truncation ($N_{E,c}$)",
            r"Circuit: $N_{E,c}$ sweep",
            _integer_ticklabels(coupler_dims),
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
            r"Charge dimension ($N_Q$)",
            r"Duffing: $N_Q$ sweep",
            _integer_ticklabels(n_q_values),
        )
    if np.asarray(result.duffing_hilbert_qubit_dim_values).size > 0:
        qubit_dims = np.asarray(result.duffing_hilbert_qubit_dim_values, dtype=float)
        specs["qubit"] = (
            qubit_dims,
            np.asarray(result.duffing_hilbert_qubit_energy_rmse, dtype=float),
            np.asarray(result.duffing_hilbert_qubit_j_abs_error, dtype=float),
            np.asarray(result.duffing_hilbert_qubit_zeta_abs_error, dtype=float),
            r"Qubit truncation ($N_{E,q}$)",
            r"Duffing: $N_{E,q}$ sweep",
            _integer_ticklabels(qubit_dims),
        )
    if np.asarray(result.duffing_hilbert_coupler_dim_values).size > 0:
        coupler_dims = np.asarray(result.duffing_hilbert_coupler_dim_values, dtype=float)
        specs["coupler"] = (
            coupler_dims,
            np.asarray(result.duffing_hilbert_coupler_energy_rmse, dtype=float),
            np.asarray(result.duffing_hilbert_coupler_j_abs_error, dtype=float),
            np.asarray(result.duffing_hilbert_coupler_zeta_abs_error, dtype=float),
            r"Coupler truncation ($N_{E,c}$)",
            r"Duffing: $N_{E,c}$ sweep",
            _integer_ticklabels(coupler_dims),
        )
    return specs


def plot_circuit_truncation_benchmark(
    result: CircuitTruncationBenchmarkResult,
    outfile: Path,
) -> None:
    subplot_specs = list(_circuit_subplot_specs(result).values())
    if not subplot_specs:
        raise ValueError("Circuit truncation plot requires at least one populated sweep")

    def _build_figure() -> plt.Figure:
        fig, axes = plt.subplots(
            len(subplot_specs),
            1,
            figsize=stacked_figure_size("truncation_single_model", len(subplot_specs)),
        )
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
        legend = _add_truncation_metric_figure_legend(fig)
        benchmark_tight_layout(fig, reserve_artists=[legend])
        return fig

    render_benchmark_figures(outfile, _build_figure)


def plot_duffing_truncation_benchmark(
    result: DuffingTruncationBenchmarkResult,
    outfile: Path,
) -> None:
    subplot_specs = list(_duffing_subplot_specs(result).values())
    if not subplot_specs:
        raise ValueError("Duffing truncation plot requires at least one populated sweep")

    def _build_figure() -> plt.Figure:
        fig, axes = plt.subplots(
            len(subplot_specs),
            1,
            figsize=stacked_figure_size("truncation_single_model", len(subplot_specs)),
        )
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
        legend = _add_truncation_metric_figure_legend(fig)
        benchmark_tight_layout(fig, reserve_artists=[legend])
        return fig

    render_benchmark_figures(outfile, _build_figure)


def plot_truncation_benchmark(
    result: TruncationBenchmarkResult,
    outfile: Path,
) -> None:
    circuit_result = CircuitTruncationBenchmarkResult(**result.circuit)
    duffing_result = DuffingTruncationBenchmarkResult(**result.duffing)
    circuit_specs = _circuit_subplot_specs(circuit_result)
    duffing_specs = _duffing_subplot_specs(duffing_result)
    row_order = ("ncut", "qubit", "coupler")
    if not any(name in circuit_specs or name in duffing_specs for name in row_order):
        raise ValueError("Combined truncation plot requires at least one populated sweep")

    def _build_figure() -> plt.Figure:
        fig, axes = plt.subplots(
            3,
            2,
            figsize=stacked_figure_size("truncation_combined", 3),
            squeeze=False,
        )
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
                    title=None,
                    ylabel="Error",
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
                    title=None,
                    ylabel="",
                    xticklabels=xticklabels,
                )
            else:
                right_ax.axis("off")
        if axes[0, 0].axison:
            add_column_title(axes[0, 0], "Circuit")
        if axes[0, 1].axison:
            add_column_title(axes[0, 1], "Duffing")
        legend = _add_truncation_metric_figure_legend(fig)
        benchmark_tight_layout(fig, reserve_artists=[legend])
        return fig

    render_benchmark_figures(outfile, _build_figure)
