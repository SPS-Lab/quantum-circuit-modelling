"""Plotting for the leakage benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.leakage import LeakageBenchmarkResult
from plots.style import DEFAULT_PLOT_FONT_SIZE, benchmark_plot_style


def plot_leakage_benchmark(
    result: LeakageBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    t = np.asarray(result.times_ns, dtype=float)

    with benchmark_plot_style(font_size):
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
        ax_flux, ax_leak, ax_inter, ax_comp = axes.ravel()

        ax_flux.plot(t, result.pulse_flux_values, color="C4", linewidth=2.0)
        ax_flux.axhline(result.idle_flux, color="0.4", linestyle=":", linewidth=1.0, label="idle")
        ax_flux.axhline(result.target_flux, color="0.3", linestyle="--", linewidth=1.0, label="target")
        ax_flux.set_ylabel(r"Flux bias ($\Phi/\Phi_0$)")
        ax_flux.set_title(f"Leakage pulse ({result.sweep_target} sweep)")
        ax_flux.grid(True, alpha=0.3)
        ax_flux.legend(loc="best")

        ax_leak.plot(t, result.circuit_leakage_11, color="k", linewidth=2.0, label="circuit")
        ax_leak.plot(t, result.duffing_leakage_11, color="C0", linestyle="--", linewidth=1.8, label="duffing")
        ax_leak.plot(t, result.effective_leakage_11, color="C3", linestyle=":", linewidth=1.8, label="effective")
        ax_leak.set_ylabel(r"Leakage from $|11\rangle$")
        ax_leak.set_title(r"$L(t)$ for $|11\rangle$ input")
        ax_leak.grid(True, alpha=0.3)
        ax_leak.legend(loc="best")

        ax_inter.plot(
            t,
            result.circuit_state_011_11,
            color="k",
            linewidth=2.0,
            label=r"circuit $P_{|0,1,1\rangle}$",
        )
        ax_inter.plot(
            t,
            result.duffing_state_011_11,
            color="C0",
            linestyle="--",
            linewidth=1.8,
            label=r"duffing $P_{|0,1,1\rangle}$",
        )
        ax_inter.set_ylabel("Population")
        ax_inter.set_title(r"Leakage-channel comparison from $|11\rangle$")
        ax_inter.grid(True, alpha=0.3)
        ax_inter.legend(loc="best")

        ax_comp.plot(t, result.circuit_populations_11[:, 0], color="C1", linewidth=1.5, label=r"circuit $P_{00}$")
        ax_comp.plot(t, result.circuit_populations_11[:, 1], color="C2", linewidth=1.5, label=r"circuit $P_{01}$")
        ax_comp.plot(t, result.circuit_populations_11[:, 2], color="C3", linewidth=1.5, label=r"circuit $P_{10}$")
        ax_comp.plot(t, result.circuit_populations_11[:, 3], color="k", linewidth=2.0, label=r"circuit $P_{11}$")
        ax_comp.set_ylabel("Computational populations")
        ax_comp.set_title(r"Circuit computational manifold from $|11\rangle$")
        ax_comp.grid(True, alpha=0.3)
        ax_comp.legend(loc="best", ncol=2)

        axes[1, 0].set_xlabel("Time (ns)")
        axes[1, 1].set_xlabel("Time (ns)")
        fig.suptitle(title)
        fig.tight_layout()

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
