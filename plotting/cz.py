"""Plotting for the CZ-relevant dynamics benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.cz import CzBenchmarkResult
from plotting.style import (
    add_model_figure_legend,
    benchmark_tight_layout,
    benchmark_plot_style,
    figure_size,
    model_plot_kwargs,
    pulse_schedule_plot_kwargs,
    save_benchmark_figure,
)


def _pi_over_two_tick_label(k: int) -> str:
    """Return a compact mathtext label for k*(pi/2)."""
    if k == 0:
        return "0"
    if k == 1:
        return r"$\frac{\pi}{2}$"
    if k == -1:
        return r"$-\frac{\pi}{2}$"
    if k % 2 == 0:
        half = k // 2
        if half == 1:
            return r"$\pi$"
        if half == -1:
            return r"$-\pi$"
        return rf"${half}\pi$"
    return rf"$\frac{{{k}\pi}}{{2}}$"


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
) -> None:
    t = np.asarray(result.times_ns, dtype=float)

    with benchmark_plot_style():
        fig = plt.figure(figsize=figure_size("cz"))
        ax_phase = fig.add_subplot(1, 1, 1)
        ax_flux = ax_phase.twinx()

        ax_phase.plot(t, result.circuit_conditional_phase, **model_plot_kwargs("circuit"))
        ax_phase.plot(t, result.duffing_conditional_phase, **model_plot_kwargs("duffing"))
        ax_phase.plot(t, result.effective_conditional_phase, **model_plot_kwargs("effective"))
        _set_phase_axis_pi_ticks(
            ax_phase,
            [
                result.circuit_conditional_phase,
                result.duffing_conditional_phase,
                result.effective_conditional_phase,
            ],
        )
        ax_phase.set_ylabel("CPhase (rad)")
        ax_phase.set_xlabel("Time (ns)")
        ax_phase.grid()

        flux_line = ax_flux.plot(
            t,
            result.pulse_flux_values,
            label="pulse flux",
            **pulse_schedule_plot_kwargs(),
        )[0]
        ax_flux.set_ylabel(r"Flux bias ($\phi$)")
        ax_flux.grid(False)
        flux_legend = ax_flux.legend(
            handles=[flux_line],
            loc="lower right",
        )
        flux_legend.set_in_layout(False)

        legend = add_model_figure_legend(fig)
        benchmark_tight_layout(fig, reserve_artists=[legend])
        save_benchmark_figure(fig, outfile)
