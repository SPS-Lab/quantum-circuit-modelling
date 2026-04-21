"""Plotting for the CZ-relevant dynamics benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
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


def _computational_hsv_rgb(
    statevector: np.ndarray,
    *,
    reference_index: int = 0,
    reference_floor: float = 1e-10,
) -> np.ndarray:
    psi = np.asarray(statevector, dtype=complex)
    if psi.ndim != 2 or psi.shape[1] != 4:
        raise ValueError(f"statevector must be (n_time, 4), got {psi.shape}")

    pop = np.clip(np.abs(psi) ** 2, 0.0, 1.0)
    phase_ref = np.zeros(psi.shape[0], dtype=float)

    idx_ref = int(reference_index)
    for m in range(psi.shape[0]):
        if pop[m, idx_ref] >= float(reference_floor):
            use_idx = idx_ref
        else:
            use_idx = int(np.argmax(pop[m, :]))
        phase_ref[m] = float(np.angle(psi[m, use_idx]))

    psi_rel = psi * np.exp(-1.0j * phase_ref)[:, np.newaxis]
    phase = np.angle(psi_rel)
    hue = (phase + np.pi) / (2.0 * np.pi)
    saturation = np.ones_like(hue)
    value = pop
    hsv = np.stack((hue, saturation, value), axis=-1)
    rgb = hsv_to_rgb(hsv)
    return np.transpose(rgb, (1, 0, 2))


def plot_cz_benchmark(
    result: CzBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    t = np.asarray(result.times_ns, dtype=float)
    basis_labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]

    rgb_eff = _computational_hsv_rgb(result.effective_statevector_plus_plus)
    rgb_duf = _computational_hsv_rgb(result.duffing_statevector_plus_plus)
    rgb_cir = _computational_hsv_rgb(result.circuit_statevector_plus_plus)

    with benchmark_plot_style(font_size):
        fig = plt.figure(figsize=(14.0, 9.0))
        gs = fig.add_gridspec(
            2,
            4,
            width_ratios=(1.0, 1.0, 1.0, 0.08),
            height_ratios=(0.95, 1.2),
        )

        top = gs[0, 0:3].subgridspec(1, 2, wspace=0.28)
        ax_flux = fig.add_subplot(top[0, 0])
        ax_phase = fig.add_subplot(top[0, 1], sharex=ax_flux)

        ax_eff = fig.add_subplot(gs[1, 0], sharex=ax_flux)
        ax_duf = fig.add_subplot(gs[1, 1], sharex=ax_flux)
        ax_cir = fig.add_subplot(gs[1, 2], sharex=ax_flux)

        cbar_grid = gs[:, 3].subgridspec(2, 1, hspace=0.45, height_ratios=(1.0, 1.0))
        ax_cbar_phase = fig.add_subplot(cbar_grid[0, 0])
        ax_cbar_pop = fig.add_subplot(cbar_grid[1, 0])

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

        for ax, rgb, panel_title in (
            (ax_eff, rgb_eff, "Effective computational state"),
            (ax_duf, rgb_duf, "Duffing computational state"),
            (ax_cir, rgb_cir, "Circuit computational state"),
        ):
            ax.imshow(
                rgb,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=(float(t[0]), float(t[-1]), -0.5, 3.5),
            )
            ax.set_yticks(np.arange(4, dtype=int))
            ax.set_yticklabels(basis_labels)
            ax.set_title(panel_title)
            ax.set_xlabel("Time (ns)")

        ax_eff.set_ylabel("Basis state")

        phase_mappable = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
        phase_mappable.set_array([])
        cbar_phase = fig.colorbar(phase_mappable, cax=ax_cbar_phase)
        cbar_phase.set_ticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
        cbar_phase.set_ticklabels(["$-\\pi$", "$-\\pi/2$", "$0$", "$\\pi/2$", "$\\pi$"])
        cbar_phase.set_label("Relative phase hue")

        pop_mappable = plt.cm.ScalarMappable(cmap="gray", norm=plt.Normalize(vmin=0.0, vmax=1.0))
        pop_mappable.set_array([])
        cbar_pop = fig.colorbar(pop_mappable, cax=ax_cbar_pop)
        cbar_pop.set_label("Population brightness")

        fig.legend(
            handles=model_legend_handles(),
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=MODEL_LEGEND_BBOX_TO_ANCHOR,
        )
        fig.tight_layout(
            rect=BENCHMARK_TIGHT_LAYOUT_RECT,
            h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD,
            w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD,
        )

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
