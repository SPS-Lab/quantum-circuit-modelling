"""Plotting for CZ runtime benchmark versus propagated qubit truncation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.runtime import RuntimeBenchmarkResult
from plotting.style import (
    add_model_figure_legend,
    benchmark_tight_layout,
    benchmark_plot_style,
    figure_size,
    model_legend_handles,
    model_plot_kwargs,
    save_benchmark_figure,
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

    with benchmark_plot_style():
        fig, (ax_build, ax_prop) = plt.subplots(1, 2, figsize=figure_size("runtime"), sharex=True)

        ax_build.errorbar(
            x,
            circuit_build,
            yerr=circuit_build_std,
            marker="o",
            capsize=3.0,
            label="circuit",
            **model_plot_kwargs("circuit"),
        )
        ax_build.errorbar(
            x,
            duffing_build,
            yerr=duffing_build_std,
            marker="o",
            capsize=3.0,
            label="duffing",
            **model_plot_kwargs("duffing"),
        )
        ax_build.set_xlabel(r"$N_{E,q}$")
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
        ax_prop.errorbar(
            x,
            duffing_prop,
            yerr=duffing_prop_std,
            marker="o",
            capsize=3.0,
            label="duffing",
            **model_plot_kwargs("duffing"),
        )
        ax_prop.set_xlabel(r"$N_{E,q}$")
        ax_prop.set_ylabel(r"Runtime ($s$)")
        ax_prop.set_title("Propagation")
        ax_prop.grid()
        ax_prop.set_xticks(x)

        legend = add_model_figure_legend(
            fig,
            handles=model_legend_handles()[:2],
            ncol=2,
        )

        benchmark_tight_layout(fig, reserve_artists=[legend])
        save_benchmark_figure(fig, outfile)
