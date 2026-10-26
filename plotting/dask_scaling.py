"""Strong-scaling timing and speedup figures for a fixed circuit workload."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.dask_scaling import DaskScalingResult
from plotting.style import figure_size, render_benchmark_figures


def plot_dask_scaling_benchmark(result: DaskScalingResult, outfile: Path) -> None:
    x = result.parallelism.tolist()

    def sample_line(samples):
        median = np.median(samples, axis=1)
        low, high = np.min(samples, axis=1), np.max(samples, axis=1)
        return {
            "x": x,
            "y": median.tolist(),
            "mean": np.mean(samples, axis=1).tolist(),
            "stddev": np.std(samples, axis=1, ddof=0).tolist(),
            "median": median.tolist(),
            "min": low.tolist(),
            "max": high.tolist(),
            "error_lower": (median - low).tolist(),
            "error_upper": (high - median).tolist(),
        }

    panels = {}
    for key, title, samples, serial, speedup in (
        (
            "preparation",
            "Unitary preparation",
            result.preparation_samples_s,
            result.serial_preparation_samples_s,
            result.preparation_speedup,
        ),
        (
            "propagation",
            "Preparation + propagation",
            result.propagation_samples_s,
            result.serial_preparation_samples_s
            + result.serial_state_propagation_samples_s,
            result.propagation_speedup,
        ),
    ):
        panels[key] = {
            "title": title,
            "runtime_lines": {
                str(mode): sample_line(samples[i])
                for i, mode in enumerate(result.modes)
            },
            "speedup_lines": {
                str(mode): {
                    "x": x,
                    "y": speedup[i].tolist(),
                    "baseline_median_s": float(np.median(samples[i, 0])),
                    "error_lower": None,
                    "error_upper": None,
                }
                for i, mode in enumerate(result.modes)
            },
        }
        panels[key]["runtime_lines"]["serial"] = sample_line(
            np.broadcast_to(serial, (len(x), len(serial)))
        )
        panels[key]["speedup_lines"]["ideal"] = {"x": x, "y": x}

    def build_figure():
        width, height = figure_size("runtime")
        fig, axes = plt.subplots(2, 2, figsize=(1.3 * width, 2.1 * height))
        for column, panel in enumerate(panels.values()):
            runtime_ax, speedup_ax = axes[:, column]
            for mode_index, mode in enumerate(result.modes):
                line = panel["runtime_lines"][str(mode)]
                color = f"C{mode_index}"
                runtime_ax.errorbar(
                    x,
                    line["y"],
                    yerr=[line["error_lower"], line["error_upper"]],
                    marker="o",
                    capsize=3,
                    label=f"Dask {mode}",
                    color=color,
                )
                speedup_ax.plot(
                    x, panel["speedup_lines"][str(mode)]["y"], marker="o", color=color
                )
            runtime_ax.axhline(
                panel["runtime_lines"]["serial"]["y"][0],
                color="0.4",
                linestyle=":",
                label="Serial",
            )
            runtime_ax.set_title(panel["title"])
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
    with outfile.with_name(f"{outfile.stem}_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "stage",
                "mode",
                "parallelism",
                "mean_s",
                "stddev_s",
                "median_s",
                "min_s",
                "max_s",
                "speedup",
                "efficiency",
            )
        )
        for stage, panel in panels.items():
            for mode, line in panel["runtime_lines"].items():
                # The serial horizontal reference has one measurement group.
                indices = range(1) if mode == "serial" else range(len(x))
                for i in indices:
                    speedup = (
                        panel["speedup_lines"][mode]["y"][i]
                        if mode != "serial"
                        else None
                    )
                    writer.writerow(
                        (
                            stage,
                            mode,
                            x[i],
                            line["mean"][i],
                            line["stddev"][i],
                            line["median"][i],
                            line["min"][i],
                            line["max"][i],
                            speedup,
                            speedup / x[i] if speedup is not None else None,
                        )
                    )
