"""Plotting for isolated single-qubit idle evolution across models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D

from comparison.idle_single_qubit import IdleSingleQubitBenchmarkResult
from plotting.style import (
    BENCHMARK_TIGHT_LAYOUT_H_PAD,
    BENCHMARK_TIGHT_LAYOUT_RECT,
    BENCHMARK_TIGHT_LAYOUT_W_PAD,
    benchmark_plot_style,
    model_plot_kwargs,
)

MODEL_COLORS = {
    "circuit": "C0",
    "duffing": "C1",
    "effective": "C2",
}

MODEL_MARKERS = {
    "circuit": "o",
    "duffing": "s",
    "effective": "^",
}


def _plot_energy_ladder(ax: plt.Axes, result: IdleSingleQubitBenchmarkResult) -> None:
    energies = {
        "circuit": np.asarray(result.circuit_relative_energies, dtype=float),
        "duffing": np.asarray(result.duffing_relative_energies, dtype=float),
        "effective": np.asarray(result.effective_relative_energies, dtype=float),
    }
    n_show = min(max(vals.size for vals in energies.values()), 8)
    x = np.arange(n_show, dtype=float)
    x_offsets = {
        "circuit": -0.06,
        "duffing": 0.0,
        "effective": 0.06,
    }

    for name in ("circuit", "duffing", "effective"):
        vals = energies[name][:n_show]
        x_local = x[: vals.size] + float(x_offsets[name])
        ax.plot(
            x_local,
            vals,
            color=MODEL_COLORS[name],
            marker=MODEL_MARKERS[name],
            markersize=5.0,
            linewidth=2.0,
            **model_plot_kwargs(name),
        )

    ax.set_xlabel("Level index")
    ax.set_ylabel("Energy rel. ground (GHz)")
    ax.set_title("Native single-qubit spectrum")
    ax.set_xticks(x, [str(int(v)) for v in x])
    ax.grid(True, alpha=0.3)


def _phase_ticks_from_data(values: list[np.ndarray]) -> tuple[np.ndarray, list[str]]:
    combined = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return np.array([], dtype=float), []
    vmin = float(np.min(combined))
    vmax = float(np.max(combined))
    span = max(vmax - vmin, 1e-12)
    step = 2.0 if span > 12.0 else 1.0
    k_min = int(np.floor(vmin / step))
    k_max = int(np.ceil(vmax / step))
    ticks = step * np.arange(k_min, k_max + 1, dtype=float)
    labels = [f"{tick:.0f}" if abs(tick - round(tick)) < 1e-12 else f"{tick:.1f}" for tick in ticks]
    return ticks, labels


def _model_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[name],
            marker=MODEL_MARKERS[name],
            markersize=6.0,
            linewidth=2.2,
            label=name,
            **model_plot_kwargs(name),
        )
        for name in ("circuit", "duffing", "effective")
    ]


def _phase_population_rgb(amplitudes: np.ndarray) -> np.ndarray:
    amp = np.asarray(amplitudes, dtype=complex)
    if amp.ndim != 2:
        raise ValueError(f"amplitudes must be 2D (n_time, n_state), got {amp.shape}")

    n_time, n_state = amp.shape
    if n_state == 0:
        return np.zeros((0, n_time, 3), dtype=float)

    pop = np.clip(np.abs(amp) ** 2, 0.0, 1.0)
    phase = np.angle(amp)
    hue = (phase + np.pi) / (2.0 * np.pi)
    sat = np.full_like(hue, 0.85, dtype=float)
    val = np.full_like(hue, 0.95, dtype=float)
    hsv_hue = np.stack((hue, sat, val), axis=-1)
    rgb_hue = hsv_to_rgb(hsv_hue)
    weight = np.sqrt(np.clip(pop, 0.0, 1.0))[..., np.newaxis]
    bg = np.full_like(rgb_hue, 0.92, dtype=float)
    rgb = (1.0 - weight) * bg + weight * rgb_hue
    return np.transpose(rgb, (1, 0, 2))


def plot_idle_single_qubit_benchmark(
    result: IdleSingleQubitBenchmarkResult,
    figure_path: Path,
    title: str,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(result.times_ns, dtype=float)

    with benchmark_plot_style():
        fig = plt.figure(figsize=(13.0, 16.0))
        gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.15])
        ax_ladder = fig.add_subplot(gs[0, 0])
        ax_x = fig.add_subplot(gs[0, 1])
        ax_y = fig.add_subplot(gs[1, 0])
        ax_z = fig.add_subplot(gs[1, 1])
        ax_pop = fig.add_subplot(gs[2, 0])
        ax_phase = fig.add_subplot(gs[2, 1])
        ax_heat = fig.add_subplot(gs[3, :])

        _plot_energy_ladder(ax_ladder, result)

        ax_x.plot(t, result.circuit_bloch_x, color=MODEL_COLORS["circuit"], linewidth=2.0, **model_plot_kwargs("circuit"))
        ax_x.plot(t, result.duffing_bloch_x, color=MODEL_COLORS["duffing"], linewidth=2.0, **model_plot_kwargs("duffing"))
        ax_x.plot(t, result.effective_bloch_x, color=MODEL_COLORS["effective"], linewidth=2.0, **model_plot_kwargs("effective"))
        ax_x.set_ylabel(r"$\langle \sigma_x \rangle$")
        ax_x.set_title("Idle evolution from $(|0\\rangle + |1\\rangle)/\\sqrt{2}$")
        ax_x.grid(True, alpha=0.3)

        ax_y.plot(t, result.circuit_bloch_y, color=MODEL_COLORS["circuit"], linewidth=2.0, **model_plot_kwargs("circuit"))
        ax_y.plot(t, result.duffing_bloch_y, color=MODEL_COLORS["duffing"], linewidth=2.0, **model_plot_kwargs("duffing"))
        ax_y.plot(t, result.effective_bloch_y, color=MODEL_COLORS["effective"], linewidth=2.0, **model_plot_kwargs("effective"))
        ax_y.set_ylabel(r"$\langle \sigma_y \rangle$")
        ax_y.grid(True, alpha=0.3)

        ax_z.plot(t, result.circuit_bloch_z, color=MODEL_COLORS["circuit"], linewidth=2.0, **model_plot_kwargs("circuit"))
        ax_z.plot(t, result.duffing_bloch_z, color=MODEL_COLORS["duffing"], linewidth=2.0, **model_plot_kwargs("duffing"))
        ax_z.plot(t, result.effective_bloch_z, color=MODEL_COLORS["effective"], linewidth=2.0, **model_plot_kwargs("effective"))
        ax_z.set_ylabel(r"$\langle \sigma_z \rangle$")
        ax_z.set_title("Logical inversion")
        ax_z.grid(True, alpha=0.3)

        ax_pop.plot(
            t,
            result.circuit_population_1,
            color=MODEL_COLORS["circuit"],
            linewidth=2.0,
            label=r"$P_1$",
            **model_plot_kwargs("circuit"),
        )
        ax_pop.plot(
            t,
            result.duffing_population_1,
            color=MODEL_COLORS["duffing"],
            linewidth=2.0,
            **model_plot_kwargs("duffing"),
        )
        ax_pop.plot(
            t,
            result.effective_population_1,
            color=MODEL_COLORS["effective"],
            linewidth=2.0,
            **model_plot_kwargs("effective"),
        )
        ax_pop.plot(t, result.circuit_population_0, color=MODEL_COLORS["circuit"], linewidth=1.4, alpha=0.35, linestyle=":")
        ax_pop.plot(t, result.duffing_population_0, color=MODEL_COLORS["duffing"], linewidth=1.4, alpha=0.35, linestyle=":")
        ax_pop.plot(t, result.effective_population_0, color=MODEL_COLORS["effective"], linewidth=1.4, alpha=0.35, linestyle=":")
        ax_pop.set_xlabel("Time (ns)")
        ax_pop.set_ylabel("Logical population")
        ax_pop.set_title(r"Solid: $P_1$, dotted: $P_0$")
        ax_pop.set_ylim(-0.02, 1.02)
        ax_pop.grid(True, alpha=0.3)

        phase_arrays = [
            np.asarray(result.circuit_relative_phase_cycles, dtype=float),
            np.asarray(result.duffing_relative_phase_cycles, dtype=float),
            np.asarray(result.effective_relative_phase_cycles, dtype=float),
        ]
        ax_phase.plot(t, phase_arrays[0], color=MODEL_COLORS["circuit"], linewidth=2.0, **model_plot_kwargs("circuit"))
        ax_phase.plot(t, phase_arrays[1], color=MODEL_COLORS["duffing"], linewidth=2.0, **model_plot_kwargs("duffing"))
        ax_phase.plot(t, phase_arrays[2], color=MODEL_COLORS["effective"], linewidth=2.0, **model_plot_kwargs("effective"))
        ax_phase.set_xlabel("Time (ns)")
        ax_phase.set_ylabel(r"Relative phase / $2\pi$")
        ax_phase.set_title("Logical phase accumulation (cycles)")
        ticks, labels = _phase_ticks_from_data(phase_arrays)
        if ticks.size > 0:
            ax_phase.set_yticks(ticks, labels)
        ax_phase.grid(True, alpha=0.3)

        leak_text = "\n".join(
            [
                f"max leak circuit: {float(np.max(result.circuit_logical_leakage)):.2e}",
                f"max leak duffing: {float(np.max(result.duffing_logical_leakage)):.2e}",
                f"max leak effective: {float(np.max(result.effective_logical_leakage)):.2e}",
            ]
        )
        ax_phase.text(
            0.03,
            0.97,
            leak_text,
            transform=ax_phase.transAxes,
            va="top",
            ha="left",
            fontsize="small",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )

        logical_amp_all = np.vstack(
            [
                np.asarray(result.circuit_logical_amplitudes, dtype=complex).T,
                np.asarray(result.duffing_logical_amplitudes, dtype=complex).T,
                np.asarray(result.effective_logical_amplitudes, dtype=complex).T,
            ]
        ).T
        heat_rgb = _phase_population_rgb(logical_amp_all)
        ax_heat.imshow(
            heat_rgb,
            aspect="auto",
            origin="lower",
            extent=(float(t[0]), float(t[-1]), -0.5, 5.5),
            interpolation="nearest",
        )
        ax_heat.set_xlabel("Time (ns)")
        ax_heat.set_ylabel("Logical state")
        ax_heat.set_title("Logical amplitudes: hue = raw phase, color strength = sqrt(population)")
        ax_heat.set_yticks(
            np.arange(6, dtype=float),
            [
                "circuit |0>",
                "circuit |1>",
                "duffing |0>",
                "duffing |1>",
                "effective |0>",
                "effective |1>",
            ],
        )
        for y in (1.5, 3.5):
            ax_heat.axhline(y, color="white", linewidth=0.9, alpha=0.85)

        phase_mappable = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
        phase_mappable.set_array([])
        cbar = fig.colorbar(phase_mappable, ax=ax_heat, pad=0.015)
        cbar.set_ticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
        cbar.set_ticklabels(["$-\\pi$", "$-\\pi/2$", "$0$", "$\\pi/2$", "$\\pi$"])
        cbar.set_label("Raw phase hue (rad)")

        fig.suptitle(title)
        fig.legend(
            handles=_model_legend_handles(),
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )
        fig.tight_layout(rect=BENCHMARK_TIGHT_LAYOUT_RECT, h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD, w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD)
        fig.savefig(figure_path, format="pdf")
        plt.close(fig)
