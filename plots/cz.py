"""Plotting for the CZ-relevant dynamics benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.cz import CzBenchmarkResult
from plots.style import (
    DEFAULT_PLOT_FONT_SIZE,
    MODEL_ALPHA_CIRCUIT,
    MODEL_ALPHA_DUFFING,
    MODEL_ALPHA_EFFECTIVE,
    benchmark_plot_style,
    model_legend_handles,
)


def plot_cz_benchmark(
    result: CzBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    t = np.asarray(result.times_ns, dtype=float)

    with benchmark_plot_style(font_size):
        fig = plt.figure(figsize=(11.0, 8.0))
        gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.1))
        ax_flux = fig.add_subplot(gs[0, 0])
        ax_phase = fig.add_subplot(gs[0, 1], sharex=ax_flux)
        ax_p01 = fig.add_subplot(gs[1, :], sharex=ax_flux)

        ax_flux.plot(t, result.pulse_flux_values, color="C4", linewidth=2.0)
        ax_flux.axhline(result.idle_flux, color="0.4", linewidth=1.0)
        ax_flux.axhline(result.target_flux, color="0.3", linewidth=1.0)
        ax_flux.set_ylabel(r"Flux bias ($\Phi/\Phi_0$)")
        ax_flux.grid(True, alpha=0.3)

        ax_phase.plot(t, result.circuit_conditional_phase, color="k", linewidth=2.0, alpha=MODEL_ALPHA_CIRCUIT)
        ax_phase.plot(t, result.duffing_conditional_phase, color="k", linewidth=2.0, alpha=MODEL_ALPHA_DUFFING)
        ax_phase.plot(t, result.effective_conditional_phase, color="k", linewidth=2.0, alpha=MODEL_ALPHA_EFFECTIVE)
        ax_phase.set_ylabel("Conditional phase (rad)")
        ax_phase.grid(True, alpha=0.3)

        ax_p01.plot(t, result.circuit_populations_plus_plus[:, 1], color="k", linewidth=2.0, alpha=MODEL_ALPHA_CIRCUIT)
        ax_p01.plot(t, result.duffing_populations_plus_plus[:, 1], color="k", linewidth=2.0, alpha=MODEL_ALPHA_DUFFING)
        ax_p01.plot(t, result.effective_populations_plus_plus[:, 1], color="k", linewidth=2.0, alpha=MODEL_ALPHA_EFFECTIVE)
        ax_p01.set_ylabel(r"$P_{01}(t)$")
        ax_p01.grid(True, alpha=0.3)

        ax_p01.set_xlabel("Time (ns)")
        fig.legend(handles=model_legend_handles(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
