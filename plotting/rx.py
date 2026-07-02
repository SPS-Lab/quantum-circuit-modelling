"""Plotting for the driven single-qubit RX benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.rx import RxBenchmarkResult
from plotting.errors import sweep_normalized_absolute_error_percent
from plotting.style import (
    PULSE_BACKGROUND_ALPHA,
    PULSE_BACKGROUND_FILL_ALPHA,
    add_model_figure_legend,
    benchmark_tight_layout,
    model_plot_kwargs,
    pulse_schedule_plot_kwargs,
    render_benchmark_figures,
    stacked_figure_size,
)


def _add_drive_background(ax: plt.Axes, times_ns: np.ndarray, envelope: np.ndarray, amplitude: float) -> None:
    bg = ax.twinx()
    bg.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_alpha(0.0)
    bg.plot(
        np.asarray(times_ns, dtype=float),
        float(amplitude) * np.asarray(envelope, dtype=float),
        **pulse_schedule_plot_kwargs(alpha=PULSE_BACKGROUND_ALPHA),
    )
    bg.fill_between(
        np.asarray(times_ns, dtype=float),
        0.0,
        float(amplitude) * np.asarray(envelope, dtype=float),
        color=pulse_schedule_plot_kwargs()["color"],
        alpha=PULSE_BACKGROUND_FILL_ALPHA,
    )
    bg.set_ylim(0.0, max(1e-12, 1.05 * float(amplitude)))
    bg.set_yticks([])
    for spine in bg.spines.values():
        spine.set_visible(False)


def plot_rx_populations_benchmark(
    result: RxBenchmarkResult,
    outfile: Path,
) -> None:
    t = np.asarray(result.times_ns, dtype=float)

    def _build_figure() -> plt.Figure:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=stacked_figure_size("rx_populations", 2),
            sharex=True,
            squeeze=False,
        )
        ax_00, ax_00_error, ax_10, ax_10_error = axes.ravel()

        for ax in axes.ravel():
            _add_drive_background(ax, t, result.pulse_envelope, result.drive_amplitude)

        for model, y in (
            ("circuit", result.circuit_pop_00_to_01),
            ("duffing", result.duffing_pop_00_to_01),
            ("effective", result.effective_pop_00_to_01),
        ):
            ax_00.plot(t, y, **model_plot_kwargs(model))
        ax_00.set_title(r"Population $|00\rangle \rightarrow |01\rangle$")
        ax_00.set_ylabel("Population")
        ax_00.set_ylim(-0.02, 1.02)
        ax_00.grid()

        ax_00_error.plot(
            t,
            sweep_normalized_absolute_error_percent(
                result.duffing_pop_00_to_01,
                result.circuit_pop_00_to_01,
            ),
            **model_plot_kwargs("duffing"),
        )
        ax_00_error.plot(
            t,
            sweep_normalized_absolute_error_percent(
                result.effective_pop_00_to_01,
                result.circuit_pop_00_to_01,
            ),
            **model_plot_kwargs("effective"),
        )
        ax_00_error.set_title(r"Population error $|00\rangle \rightarrow |01\rangle$")
        ax_00_error.set_ylabel("Error (%)")
        ax_00_error.grid()

        for model, y in (
            ("circuit", result.circuit_pop_10_to_11),
            ("duffing", result.duffing_pop_10_to_11),
            ("effective", result.effective_pop_10_to_11),
        ):
            ax_10.plot(t, y, **model_plot_kwargs(model))
        ax_10.set_title(r"Population $|10\rangle \rightarrow |11\rangle$")
        ax_10.set_xlabel("Time (ns)")
        ax_10.set_ylabel("Population")
        ax_10.set_ylim(-0.02, 1.02)
        ax_10.grid()

        ax_10_error.plot(
            t,
            sweep_normalized_absolute_error_percent(
                result.duffing_pop_10_to_11,
                result.circuit_pop_10_to_11,
            ),
            **model_plot_kwargs("duffing"),
        )
        ax_10_error.plot(
            t,
            sweep_normalized_absolute_error_percent(
                result.effective_pop_10_to_11,
                result.circuit_pop_10_to_11,
            ),
            **model_plot_kwargs("effective"),
        )
        ax_10_error.set_title(r"Population error $|10\rangle \rightarrow |11\rangle$")
        ax_10_error.set_xlabel("Time (ns)")
        ax_10_error.set_ylabel("Error (%)")
        ax_10_error.grid()

        legend = add_model_figure_legend(fig)
        benchmark_tight_layout(fig, reserve_artists=[legend])
        return fig

    render_benchmark_figures(outfile, _build_figure)


def plot_rx_diagnostics_benchmark(
    result: RxBenchmarkResult,
    outfile: Path,
) -> None:
    t = np.asarray(result.times_ns, dtype=float)

    def _build_figure() -> plt.Figure:
        fig, axes = plt.subplots(
            3,
            2,
            figsize=stacked_figure_size("rx_diagnostics", 3),
            sharex=True,
            squeeze=False,
        )
        (
            ax_leak_00,
            ax_leak_00_error,
            ax_leak_10,
            ax_leak_10_error,
            ax_delta,
            ax_delta_error,
        ) = axes.ravel()

        for ax in axes.ravel():
            _add_drive_background(ax, t, result.pulse_envelope, result.drive_amplitude)

        for model, y in (
            ("circuit", result.circuit_leakage_from_00),
            ("duffing", result.duffing_leakage_from_00),
            ("effective", result.effective_leakage_from_00),
        ):
            ax_leak_00.plot(t, y, **model_plot_kwargs(model))
        ax_leak_00.set_title(r"Leakage From $|00\rangle$")
        ax_leak_00.set_ylabel("Leakage")
        ax_leak_00.grid()

        ax_leak_00_error.plot(
            t,
            sweep_normalized_absolute_error_percent(
                result.duffing_leakage_from_00,
                result.circuit_leakage_from_00,
            ),
            **model_plot_kwargs("duffing"),
        )
        ax_leak_00_error.plot(
            t,
            sweep_normalized_absolute_error_percent(
                result.effective_leakage_from_00,
                result.circuit_leakage_from_00,
            ),
            **model_plot_kwargs("effective"),
        )
        ax_leak_00_error.set_title(r"Leakage Error From $|00\rangle$")
        ax_leak_00_error.set_ylabel("Error (%)")
        ax_leak_00_error.grid()

        for model, y in (
            ("circuit", result.circuit_leakage_from_10),
            ("duffing", result.duffing_leakage_from_10),
            ("effective", result.effective_leakage_from_10),
        ):
            ax_leak_10.plot(t, y, **model_plot_kwargs(model))
        ax_leak_10.set_title(r"Leakage From $|10\rangle$")
        ax_leak_10.set_ylabel("Leakage")
        ax_leak_10.grid()

        ax_leak_10_error.plot(
            t,
            sweep_normalized_absolute_error_percent(
                result.duffing_leakage_from_10,
                result.circuit_leakage_from_10,
            ),
            **model_plot_kwargs("duffing"),
        )
        ax_leak_10_error.plot(
            t,
            sweep_normalized_absolute_error_percent(
                result.effective_leakage_from_10,
                result.circuit_leakage_from_10,
            ),
            **model_plot_kwargs("effective"),
        )
        ax_leak_10_error.set_title(r"Leakage Error From $|10\rangle$")
        ax_leak_10_error.set_ylabel("Error (%)")
        ax_leak_10_error.grid()

        for model, y in (
            ("circuit", result.circuit_spectator_population_delta),
            ("duffing", result.duffing_spectator_population_delta),
            ("effective", result.effective_spectator_population_delta),
        ):
            ax_delta.plot(t, y, **model_plot_kwargs(model))
        ax_delta.set_title("Spectator Mismatch")
        ax_delta.set_xlabel("Time (ns)")
        ax_delta.set_ylabel("Magnitude")
        ax_delta.grid()

        ax_delta_error.plot(
            t,
            sweep_normalized_absolute_error_percent(
                result.duffing_spectator_population_delta,
                result.circuit_spectator_population_delta,
            ),
            **model_plot_kwargs("duffing"),
        )
        ax_delta_error.plot(
            t,
            sweep_normalized_absolute_error_percent(
                result.effective_spectator_population_delta,
                result.circuit_spectator_population_delta,
            ),
            **model_plot_kwargs("effective"),
        )
        ax_delta_error.set_title("Spectator-Mismatch Error")
        ax_delta_error.set_xlabel("Time (ns)")
        ax_delta_error.set_ylabel("Error (%)")
        ax_delta_error.grid()

        legend = add_model_figure_legend(fig)
        benchmark_tight_layout(fig, reserve_artists=[legend])
        return fig

    render_benchmark_figures(outfile, _build_figure)
