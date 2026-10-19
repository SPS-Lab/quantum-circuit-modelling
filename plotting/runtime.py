"""Plotting for CZ runtime benchmark versus propagated qubit truncation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from comparison.runtime import RuntimeBenchmarkResult
from plotting.style import (
    add_model_figure_legend,
    figure_size,
    model_legend_handles,
    model_plot_kwargs,
    render_benchmark_figures,
)


def plot_runtime_benchmark(
    result: RuntimeBenchmarkResult,
    outfile: Path,
) -> None:
    x = np.asarray(result.qubit_truncation_values, dtype=int)
    duffing_build = np.asarray(result.duffing_build_runtime_s, dtype=float)
    duffing_build_std = np.asarray(result.duffing_build_runtime_std_s, dtype=float)
    circuit_build = np.asarray(result.circuit_build_runtime_s, dtype=float)
    circuit_build_std = np.asarray(result.circuit_build_runtime_std_s, dtype=float)
    duffing_prop = np.asarray(result.duffing_propagation_runtime_s, dtype=float)
    duffing_prop_std = np.asarray(result.duffing_propagation_runtime_std_s, dtype=float)
    circuit_prop = np.asarray(result.circuit_propagation_runtime_s, dtype=float)
    circuit_prop_std = np.asarray(result.circuit_propagation_runtime_std_s, dtype=float)
    has_duffing = bool(
        np.any(np.isfinite(duffing_build)) or np.any(np.isfinite(duffing_prop))
    )

    def _build_figure() -> plt.Figure:
        width, height = figure_size("runtime")
        fig, (ax_build, ax_prop) = plt.subplots(1, 2, figsize=(1.12 * width, height))

        ax_build.errorbar(
            x,
            circuit_build,
            yerr=circuit_build_std,
            marker="o",
            capsize=3.0,
            label="circuit",
            **model_plot_kwargs("circuit"),
        )
        if np.any(np.isfinite(duffing_build)):
            ax_build.errorbar(
                x,
                duffing_build,
                yerr=duffing_build_std,
                marker="o",
                capsize=3.0,
                label="duffing",
                **model_plot_kwargs("duffing"),
            )
        ax_build.set_xlabel(r"Qubit truncation ($N_{E,q}$)")
        ax_build.set_ylabel(r"Runtime ($s$)")
        ax_build.set_title("Build")
        ax_build.grid()
        ax_build.set_xticks(x)

        ax_prop.errorbar(
            x,
            circuit_prop,
            yerr=circuit_prop_std,
            marker="o",
            capsize=3.0,
            label="circuit",
            **model_plot_kwargs("circuit"),
        )
        if np.any(np.isfinite(duffing_prop)):
            ax_prop.errorbar(
                x,
                duffing_prop,
                yerr=duffing_prop_std,
                marker="o",
                capsize=3.0,
                label="duffing",
                **model_plot_kwargs("duffing"),
            )
        fit_mask = (x > 0) & np.isfinite(circuit_prop) & (circuit_prop > 0)
        if np.unique(x[fit_mask]).size >= 2:
            # Fit log(T) = log(a) + p log(N), weighting relative errors equally.
            exponent, log_prefactor = np.polyfit(
                np.log(x[fit_mask]), np.log(circuit_prop[fit_mask]), deg=1
            )
            fit_x = np.geomspace(x[fit_mask].min(), x[fit_mask].max(), 100)
            (fit_line,) = ax_prop.plot(
                fit_x,
                np.exp(log_prefactor + exponent * np.log(fit_x)),
                color="0.4",
                linestyle="--",
                label=(
                    ("circuit" if has_duffing else "")
                    + rf": $T \propto N_{{E,q}}^{{{exponent:.2f}}}$"
                ),
            )
            ax_prop.legend(handles=[fit_line], loc="upper left")
        ax_prop.set_xlabel(r"Qubit truncation ($N_{E,q}$)")
        ax_prop.set_title("Propagation (log–log)")
        ax_prop.set_xscale("log")
        ax_prop.set_yscale("log")
        ax_prop.grid(which="both")
        ax_prop.set_xticks(x)
        ax_prop.xaxis.set_major_formatter(ScalarFormatter())

        handles = model_legend_handles()[:2 if has_duffing else 1]
        if not has_duffing:
            handles[0].set_label("Runtime circuit model")
        legend = add_model_figure_legend(
            fig,
            handles=handles,
            ncol=2 if has_duffing else 1,
        )
        legend.set_in_layout(False)
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.28, top=0.75, wspace=0.27)
        return fig

    render_benchmark_figures(outfile, _build_figure)
