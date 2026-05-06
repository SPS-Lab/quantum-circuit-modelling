"""Plotting for the driven single-qubit RX benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.rx import RxBenchmarkResult
from plotting.style import (
    BENCHMARK_TIGHT_LAYOUT_H_PAD,
    BENCHMARK_TIGHT_LAYOUT_RECT,
    BENCHMARK_TIGHT_LAYOUT_W_PAD,
    DEFAULT_PLOT_FONT_SIZE,
    benchmark_plot_style,
    model_legend_handles,
    model_plot_kwargs,
)


def plot_rx_benchmark(
    result: RxBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    t = np.asarray(result.times_ns, dtype=float)

    with benchmark_plot_style(font_size):
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
        ax_env = axes[0, 0]
        ax_00 = axes[0, 1]
        ax_10 = axes[1, 0]
        ax_leak = axes[1, 1]

        ax_env.plot(t, result.pulse_envelope * result.drive_amplitude, color="C4", linewidth=2.0)
        ax_env.set_title("Drive Envelope")
        ax_env.set_ylabel("Amplitude (GHz)")
        ax_env.grid(True, alpha=0.3)

        for model, y in (
            ("circuit", result.circuit_pop_00_to_01),
            ("duffing", result.duffing_pop_00_to_01),
            ("effective", result.effective_pop_00_to_01),
        ):
            ax_00.plot(t, y, color="k", linewidth=2.0, **model_plot_kwargs(model))
        ax_00.set_title(r"Population $|00\rangle \rightarrow |01\rangle$")
        ax_00.set_ylabel("Population")
        ax_00.set_ylim(-0.02, 1.02)
        ax_00.grid(True, alpha=0.3)

        for model, y in (
            ("circuit", result.circuit_pop_10_to_11),
            ("duffing", result.duffing_pop_10_to_11),
            ("effective", result.effective_pop_10_to_11),
        ):
            ax_10.plot(t, y, color="k", linewidth=2.0, **model_plot_kwargs(model))
        ax_10.set_title(r"Population $|10\rangle \rightarrow |11\rangle$")
        ax_10.set_xlabel("Time (ns)")
        ax_10.set_ylabel("Population")
        ax_10.set_ylim(-0.02, 1.02)
        ax_10.grid(True, alpha=0.3)

        ax_leak.plot(
            t,
            result.circuit_leakage_from_00,
            color="k",
            linewidth=2.0,
            label=r"circuit $|00\rangle$",
            **model_plot_kwargs("circuit"),
        )
        ax_leak.plot(
            t,
            result.duffing_leakage_from_00,
            color="k",
            linewidth=2.0,
            label=r"duffing $|00\rangle$",
            **model_plot_kwargs("duffing"),
        )
        ax_leak.plot(
            t,
            result.effective_leakage_from_00,
            color="k",
            linewidth=2.0,
            label=r"effective $|00\rangle$",
            **model_plot_kwargs("effective"),
        )
        ax_leak.plot(
            t,
            result.circuit_spectator_population_delta,
            color="C3",
            linewidth=1.8,
            linestyle="-",
            label=r"circuit spectator $\Delta P$",
        )
        ax_leak.plot(
            t,
            result.duffing_spectator_population_delta,
            color="C1",
            linewidth=1.8,
            linestyle="--",
            label=r"duffing spectator $\Delta P$",
        )
        ax_leak.plot(
            t,
            result.effective_spectator_population_delta,
            color="C0",
            linewidth=1.8,
            linestyle="-.",
            label=r"effective spectator $\Delta P$",
        )
        ax_leak.set_title(r"Leakage From $|00\rangle$ And Spectator Mismatch")
        ax_leak.set_xlabel("Time (ns)")
        ax_leak.set_ylabel("Magnitude")
        ax_leak.grid(True, alpha=0.3)
        ax_leak.legend(loc="best", frameon=False, fontsize=font_size * 0.58)

        fig.legend(
            handles=model_legend_handles(),
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
        fig.suptitle(title)
        fig.tight_layout(
            rect=BENCHMARK_TIGHT_LAYOUT_RECT,
            h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD,
            w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD,
        )

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
