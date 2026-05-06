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
    pulse_schedule_plot_kwargs,
)


def _add_drive_background(ax: plt.Axes, times_ns: np.ndarray, envelope: np.ndarray, amplitude: float) -> None:
    bg = ax.twinx()
    bg.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_alpha(0.0)
    bg.plot(
        np.asarray(times_ns, dtype=float),
        float(amplitude) * np.asarray(envelope, dtype=float),
        **pulse_schedule_plot_kwargs(alpha=0.35),
    )
    bg.fill_between(
        np.asarray(times_ns, dtype=float),
        0.0,
        float(amplitude) * np.asarray(envelope, dtype=float),
        color=pulse_schedule_plot_kwargs()["color"],
        alpha=0.08,
    )
    bg.set_ylim(0.0, max(1e-12, 1.05 * float(amplitude)))
    bg.set_yticks([])
    for spine in bg.spines.values():
        spine.set_visible(False)


def plot_rx_populations_benchmark(
    result: RxBenchmarkResult,
    outfile: Path,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    t = np.asarray(result.times_ns, dtype=float)

    with benchmark_plot_style(font_size):
        fig, axes = plt.subplots(2, 1, figsize=(9.8, 7.2), sharex=True)
        ax_00, ax_10 = axes

        for ax in axes:
            _add_drive_background(ax, t, result.pulse_envelope, result.drive_amplitude)

        for model, y in (
            ("circuit", result.circuit_pop_00_to_01),
            ("duffing", result.duffing_pop_00_to_01),
            ("effective", result.effective_pop_00_to_01),
        ):
            ax_00.plot(t, y, linewidth=2.2, **model_plot_kwargs(model))
        ax_00.set_title(r"Population $|00\rangle \rightarrow |01\rangle$")
        ax_00.set_ylabel("Population")
        ax_00.set_ylim(-0.02, 1.02)
        ax_00.grid(True, alpha=0.3)

        for model, y in (
            ("circuit", result.circuit_pop_10_to_11),
            ("duffing", result.duffing_pop_10_to_11),
            ("effective", result.effective_pop_10_to_11),
        ):
            ax_10.plot(t, y, linewidth=2.2, **model_plot_kwargs(model))
        ax_10.set_title(r"Population $|10\rangle \rightarrow |11\rangle$")
        ax_10.set_xlabel("Time (ns)")
        ax_10.set_ylabel("Population")
        ax_10.set_ylim(-0.02, 1.02)
        ax_10.grid(True, alpha=0.3)

        fig.legend(
            handles=model_legend_handles(),
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.985),
        )
        fig.tight_layout(
            rect=BENCHMARK_TIGHT_LAYOUT_RECT,
            h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD,
            w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD,
        )

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)


def plot_rx_diagnostics_benchmark(
    result: RxBenchmarkResult,
    outfile: Path,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    t = np.asarray(result.times_ns, dtype=float)

    with benchmark_plot_style(font_size):
        fig, axes = plt.subplots(3, 1, figsize=(9.8, 9.2), sharex=True)
        ax_leak_00, ax_leak_10, ax_delta = axes

        for ax in axes:
            _add_drive_background(ax, t, result.pulse_envelope, result.drive_amplitude)

        for model, y in (
            ("circuit", result.circuit_leakage_from_00),
            ("duffing", result.duffing_leakage_from_00),
            ("effective", result.effective_leakage_from_00),
        ):
            ax_leak_00.plot(t, y, linewidth=2.2, **model_plot_kwargs(model))
        ax_leak_00.set_title(r"Leakage From $|00\rangle$")
        ax_leak_00.set_ylabel("Leakage")
        ax_leak_00.grid(True, alpha=0.3)

        for model, y in (
            ("circuit", result.circuit_leakage_from_10),
            ("duffing", result.duffing_leakage_from_10),
            ("effective", result.effective_leakage_from_10),
        ):
            ax_leak_10.plot(t, y, linewidth=2.2, **model_plot_kwargs(model))
        ax_leak_10.set_title(r"Leakage From $|10\rangle$")
        ax_leak_10.set_ylabel("Leakage")
        ax_leak_10.grid(True, alpha=0.3)

        for model, y in (
            ("circuit", result.circuit_spectator_population_delta),
            ("duffing", result.duffing_spectator_population_delta),
            ("effective", result.effective_spectator_population_delta),
        ):
            ax_delta.plot(t, y, linewidth=2.2, **model_plot_kwargs(model))
        ax_delta.set_title(r"Spectator Mismatch $|P_{00\rightarrow01} - P_{10\rightarrow11}|$")
        ax_delta.set_xlabel("Time (ns)")
        ax_delta.set_ylabel("Magnitude")
        ax_delta.grid(True, alpha=0.3)

        fig.legend(
            handles=model_legend_handles(),
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.985),
        )
        fig.tight_layout(
            rect=BENCHMARK_TIGHT_LAYOUT_RECT,
            h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD,
            w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD,
        )

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
