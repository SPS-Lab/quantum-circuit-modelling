"""Strong-scaling timing and speedup figures for a fixed circuit workload."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.dask_scaling import DaskScalingResult
from plotting.style import figure_size, render_benchmark_figures


def plot_dask_scaling_benchmark(result: DaskScalingResult, outfile: Path) -> None:
    def build_figure():
        width, height = figure_size("runtime")
        fig, axes = plt.subplots(2, 2, figsize=(1.3 * width, 2.1 * height))
        x = result.parallelism
        stages = (
            (
                result.preparation_samples_s,
                result.serial_preparation_samples_s,
                result.preparation_speedup,
                "Unitary preparation",
            ),
            (
                result.propagation_samples_s,
                result.serial_preparation_samples_s
                + result.serial_state_propagation_samples_s,
                result.propagation_speedup,
                "Preparation + propagation",
            ),
        )
        for column, (samples, serial, speedup, title) in enumerate(stages):
            runtime_ax, speedup_ax = axes[:, column]
            for mode_index, mode in enumerate(result.modes):
                median = np.median(samples[mode_index], axis=1)
                low, high = np.min(samples[mode_index], axis=1), np.max(
                    samples[mode_index], axis=1
                )
                color = f"C{mode_index}"
                runtime_ax.errorbar(
                    x,
                    median,
                    yerr=[median - low, high - median],
                    marker="o",
                    capsize=3,
                    label=f"Dask {mode}",
                    color=color,
                )
                speedup_ax.plot(x, speedup[mode_index], marker="o", color=color)
            runtime_ax.axhline(
                np.median(serial), color="0.4", linestyle=":", label="Serial"
            )
            runtime_ax.set_title(title)
            runtime_ax.set_ylabel("Runtime (s)")
            runtime_ax.set_ylim(bottom=0)
            speedup_ax.plot(x, x, color="0.4", linestyle="--", label="Ideal")
            speedup_ax.set_ylabel(r"Speedup $T_1/T_p$")
            speedup_ax.legend(loc="upper left")
            for ax in (runtime_ax, speedup_ax):
                ax.set_xlabel(r"Parallelism $p$")
                ax.set_xticks(x)
                ax.grid(alpha=0.3)
        axes[0, 0].legend()
        fig.tight_layout()
        return fig

    render_benchmark_figures(outfile, build_figure)
