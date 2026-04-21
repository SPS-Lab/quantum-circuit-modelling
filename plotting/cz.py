"""Plotting for the CZ-relevant dynamics benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.cz import CzBenchmarkResult
from plotting.style import (
    BENCHMARK_TIGHT_LAYOUT_H_PAD,
    BENCHMARK_TIGHT_LAYOUT_RECT,
    BENCHMARK_TIGHT_LAYOUT_W_PAD,
    DEFAULT_PLOT_FONT_SIZE,
    MODEL_LEGEND_BBOX_TO_ANCHOR,
    benchmark_plot_style,
    model_legend_handles,
    model_plot_kwargs,
)


def _pi_over_two_tick_label(k: int) -> str:
    """Return a compact mathtext label for k*(pi/2)."""
    if k == 0:
        return "0"
    if k == 1:
        return r"$\pi/2$"
    if k == -1:
        return r"$-\pi/2$"
    if k % 2 == 0:
        half = k // 2
        if half == 1:
            return r"$\pi$"
        if half == -1:
            return r"$-\pi$"
        return rf"${half}\pi$"
    return rf"${k}\pi/2$"


def _set_phase_axis_pi_ticks(ax: plt.Axes, phase_arrays: list[np.ndarray]) -> None:
    """Set y-ticks on the phase axis at multiples of pi/2 over the data range."""
    phase_values = np.concatenate([np.asarray(arr, dtype=float).ravel() for arr in phase_arrays])
    phase_values = phase_values[np.isfinite(phase_values)]
    if phase_values.size == 0:
        return

    step = np.pi / 2.0
    k_min = int(np.floor(np.min(phase_values) / step))
    k_max = int(np.ceil(np.max(phase_values) / step))
    ticks = step * np.arange(k_min, k_max + 1, dtype=float)
    labels = [_pi_over_two_tick_label(k) for k in range(k_min, k_max + 1)]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


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

        ax_phase.plot(t, result.circuit_conditional_phase, color="k", linewidth=2.0, **model_plot_kwargs("circuit"))
        ax_phase.plot(t, result.duffing_conditional_phase, color="k", linewidth=2.0, **model_plot_kwargs("duffing"))
        ax_phase.plot(t, result.effective_conditional_phase, color="k", linewidth=2.0, **model_plot_kwargs("effective"))
        _set_phase_axis_pi_ticks(
            ax_phase,
            [
                result.circuit_conditional_phase,
                result.duffing_conditional_phase,
                result.effective_conditional_phase,
            ],
        )
        ax_phase.set_ylabel("CPhase (rad)")
        ax_phase.grid(True, alpha=0.3)

        ax_p01.plot(t, result.circuit_populations_plus_plus[:, 1], color="k", linewidth=2.0, **model_plot_kwargs("circuit"))
        ax_p01.plot(t, result.duffing_populations_plus_plus[:, 1], color="k", linewidth=2.0, **model_plot_kwargs("duffing"))
        ax_p01.plot(t, result.effective_populations_plus_plus[:, 1], color="k", linewidth=2.0, **model_plot_kwargs("effective"))
        ax_p01.set_ylabel(r"$P_{01}(t)$")
        ax_p01.grid(True, alpha=0.3)

        ax_p01.set_xlabel("Time (ns)")
        fig.legend(handles=model_legend_handles(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=MODEL_LEGEND_BBOX_TO_ANCHOR)
        fig.tight_layout(
            rect=BENCHMARK_TIGHT_LAYOUT_RECT,
            h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD,
            w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD,
        )

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
