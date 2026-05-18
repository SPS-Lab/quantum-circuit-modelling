"""Plotting for CZ runtime benchmark versus propagated qubit truncation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.runtime import RuntimeBenchmarkResult
from plotting.style import (
    BENCHMARK_TIGHT_LAYOUT_H_PAD,
    BENCHMARK_TIGHT_LAYOUT_W_PAD,
    DEFAULT_PLOT_FONT_SIZE,
    benchmark_plot_style,
    model_legend_handles,
    model_plot_kwargs,
)


def plot_runtime_benchmark(
    result: RuntimeBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
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

    with benchmark_plot_style(font_size):
        fig, (ax_build, ax_prop) = plt.subplots(1, 2, figsize=(8.8, 4.4), sharex=True)

        ax_build.errorbar(
            x,
            circuit_build,
            yerr=circuit_build_std,
            linewidth=2.0,
            marker="o",
            capsize=3.0,
            label="circuit",
            **model_plot_kwargs("circuit"),
        )
        ax_build.errorbar(
            x,
            duffing_build,
            yerr=duffing_build_std,
            linewidth=2.0,
            marker="o",
            capsize=3.0,
            label="duffing",
            **model_plot_kwargs("duffing"),
        )
        ax_build.set_xlabel(r"$N_{E,q}$")
        ax_build.set_ylabel(r"Runtime ($s$)")
        ax_build.set_title("Build")
        ax_build.grid(True, alpha=0.3)
        ax_build.set_xticks(x)

        ax_prop.errorbar(
            x,
            circuit_prop,
            yerr=circuit_prop_std,
            linewidth=2.0,
            marker="o",
            capsize=3.0,
            label="circuit",
            **model_plot_kwargs("circuit"),
        )
        ax_prop.errorbar(
            x,
            duffing_prop,
            yerr=duffing_prop_std,
            linewidth=2.0,
            marker="o",
            capsize=3.0,
            label="duffing",
            **model_plot_kwargs("duffing"),
        )
        ax_prop.set_xlabel(r"$N_{E,q}$")
        ax_prop.set_ylabel(r"Runtime ($s$)")
        ax_prop.set_title("Propagation")
        ax_prop.grid(True, alpha=0.3)
        ax_prop.set_xticks(x)

        fig.legend(
            handles=model_legend_handles()[:2],
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.suptitle(title, y=0.985)

        fig.tight_layout(
            rect=(0.0, 0.0, 1.0, 0.90),
            h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD,
            w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD,
        )
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
