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
) -> np.ndarray:
    psi = np.asarray(statevector, dtype=complex)
    if psi.ndim != 2 or psi.shape[1] != 4:
        raise ValueError(f"statevector must be (n_time, 4), got {psi.shape}")

    pop = np.clip(np.abs(psi) ** 2, 0.0, 1.0)
    psi_rel = np.array(psi, copy=True)
    phase_at_time = np.zeros(psi.shape[0], dtype=float)
    min_overlap = 1e-12
    # Parallel-transport gauge: fix global phase by maximizing overlap with
    # previous time-step state so phase evolution is as smooth as possible.
    for m in range(1, psi.shape[0]):
        overlap = np.vdot(psi_rel[m - 1], psi[m])
        if abs(overlap) >= min_overlap:
            phase_at_time[m] = float(np.angle(overlap))
        else:
            phase_at_time[m] = phase_at_time[m - 1]
        psi_rel[m] = psi[m] * np.exp(-1.0j * phase_at_time[m])

    phase = np.angle(psi_rel)
    hue = (phase + np.pi) / (2.0 * np.pi)
    saturation = np.ones_like(hue)
    value = pop
    hsv = np.stack((hue, saturation, value), axis=-1)
    rgb = hsv_to_rgb(hsv)
    return np.transpose(rgb, (1, 0, 2))


def _local_z_removed_hsv_from_comp_amplitudes(
    computational_amplitudes: np.ndarray,
    conditional_phase: np.ndarray,
) -> np.ndarray:
    comp = np.asarray(computational_amplitudes, dtype=complex)
    cphase = np.asarray(conditional_phase, dtype=float).ravel()
    if comp.ndim != 3 or comp.shape[1:] != (4, 4):
        raise ValueError(f"computational_amplitudes must be (n_time, 4, 4), got {comp.shape}")
    if comp.shape[0] != cphase.size:
        raise ValueError(
            "computational_amplitudes time axis must match conditional_phase length, "
            f"got {comp.shape[0]} and {cphase.size}"
        )

    diag = np.diagonal(comp, axis1=1, axis2=2)
    value = np.clip(np.abs(diag) ** 2, 0.0, 1.0)

    # Remove global + local Z phases in the computational subspace:
    # rows |00>,|01>,|10> are fixed to zero phase, and |11> carries CPhase.
    phase = np.zeros_like(value, dtype=float)
    phase[:, 3] = np.angle(np.exp(1.0j * cphase))

    hue = (phase + np.pi) / (2.0 * np.pi)
    saturation = np.ones_like(hue)
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

    if np.asarray(result.effective_computational_amplitudes).size > 0:
        rgb_eff = _local_z_removed_hsv_from_comp_amplitudes(
            result.effective_computational_amplitudes,
            result.effective_conditional_phase,
        )
        rgb_duf = _local_z_removed_hsv_from_comp_amplitudes(
            result.duffing_computational_amplitudes,
            result.duffing_conditional_phase,
        )
        rgb_cir = _local_z_removed_hsv_from_comp_amplitudes(
            result.circuit_computational_amplitudes,
            result.circuit_conditional_phase,
        )
    else:
        # Backward-compatible fallback for older result files.
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
            (ax_eff, rgb_eff, "Effective computational propagator (local-Z removed)"),
            (ax_duf, rgb_duf, "Duffing computational propagator (local-Z removed)"),
            (ax_cir, rgb_cir, "Circuit computational propagator (local-Z removed)"),
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
        cbar_phase.set_label("Phase hue after global/local-Z removal")

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
