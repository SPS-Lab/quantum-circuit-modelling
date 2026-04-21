"""
Plotting for the combined leakage/population + transition-flow benchmark.
Leakage/flow benchmark from |1,0,1>:
    population+phase states and canonical signed transitions
Color strength ~ sqrt(population) for States.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import hsv_to_rgb
import numpy as np

from comparison.leakage_flow import LeakageFlowBenchmarkResult
from plotting.style import DEFAULT_PLOT_FONT_SIZE, benchmark_plot_style


def _decode_labels(labels: np.ndarray) -> list[str]:
    arr = np.asarray(labels).ravel()
    out: list[str] = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(bytes(x).decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _state_label_math(label: str) -> str:
    text = str(label).strip()
    if text.startswith("|") and text.endswith(">"):
        inner = text[1:-1]
        return rf"$\left|{inner}\right\rangle$"
    return text


def _transition_label_math(label: str) -> str:
    text = str(label).strip()
    if "->" in text:
        src, dst = text.split("->", 1)
        src = src.strip()
        dst = dst.strip()
        if src.startswith("|") and src.endswith(">") and dst.startswith("|") and dst.endswith(">"):
            src_inner = src[1:-1]
            dst_inner = dst[1:-1]
            return rf"$\left|{src_inner}\right\rangle \to \left|{dst_inner}\right\rangle$"
    return text


def _phase_population_rgb(
    amplitudes: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    amp = np.asarray(amplitudes, dtype=complex)
    if amp.ndim != 2:
        raise ValueError(f"amplitudes must be 2D (n_time, n_state), got {amp.shape}")

    n_time, n_state = amp.shape
    if n_state == 0:
        return np.zeros((0, n_time, 3), dtype=float)

    pop = np.clip(np.abs(amp) ** 2, 0.0, 1.0)
    amp_rel = np.array(amp, copy=True)
    phase_at_time = np.zeros(n_time, dtype=float)
    min_overlap = 1e-12
    # Parallel-transport gauge: set global phase from step-to-step overlap.
    for m in range(1, n_time):
        overlap = np.vdot(amp_rel[m - 1], amp[m])
        if abs(overlap) >= min_overlap:
            phase_at_time[m] = float(np.angle(overlap))
        else:
            phase_at_time[m] = phase_at_time[m - 1]
        amp_rel[m] = amp[m] * np.exp(-1.0j * phase_at_time[m])

    phase = np.angle(amp_rel)

    # Phase as hue, with population controlling color-strength over a light-gray
    # background so near-zero population stays visible as light rows (not black).
    hue = (phase + np.pi) / (2.0 * np.pi)
    sat = np.full_like(hue, 0.85, dtype=float)
    val = np.full_like(hue, 0.95, dtype=float)
    hsv_hue = np.stack((hue, sat, val), axis=-1)
    rgb_hue = hsv_to_rgb(hsv_hue)

    # Nonlinear contrast boost for low-but-nonzero populations.
    weight = np.sqrt(np.clip(pop, 0.0, 1.0))[..., np.newaxis]
    bg = np.full_like(rgb_hue, 0.92, dtype=float)
    rgb = (1.0 - weight) * bg + weight * rgb_hue
    return np.transpose(rgb, (1, 0, 2))


def _set_y_ticks(ax: plt.Axes, labels: list[str], *, transition: bool, tick_font_size: float) -> None:
    n = len(labels)
    if n == 0:
        ax.set_yticks([])
        return
    ax.set_yticks(np.arange(n, dtype=int))
    if transition:
        ax.set_yticklabels([_transition_label_math(lbl) for lbl in labels])
    else:
        ax.set_yticklabels([_state_label_math(lbl) for lbl in labels])
    ax.tick_params(axis="y", labelsize=tick_font_size, pad=2.5)


def _overlay_flux_track(
    ax: plt.Axes,
    *,
    times_ns: np.ndarray,
    pulse_flux_values: np.ndarray,
    idle_flux: float,
    target_flux: float,
    n_rows: int,
) -> None:
    t = np.asarray(times_ns, dtype=float).ravel()
    flux = np.asarray(pulse_flux_values, dtype=float).ravel()
    n_rows_eff = max(1, int(n_rows))
    if t.size == 0 or flux.size == 0 or t.size != flux.size:
        return
    flux_lo = float(min(np.min(flux), float(idle_flux), float(target_flux)))
    flux_hi = float(max(np.max(flux), float(idle_flux), float(target_flux)))
    flux_span = float(max(1e-15, flux_hi - flux_lo))
    flux_norm = np.clip((flux - flux_lo) / flux_span, 0.0, 1.0)
    y_flux = -0.5 + flux_norm * float(n_rows_eff - 1)
    ax.plot(t, y_flux, color="black", linewidth=1.0, alpha=0.35, zorder=4)


def plot_leakage_flow_benchmark(
    result: LeakageFlowBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    t = np.asarray(result.times_ns, dtype=float).ravel()

    pop_labels = _decode_labels(result.population_state_labels_11)
    tr_labels = _decode_labels(result.transition_labels_11)

    pop_amp_duf = np.asarray(result.duffing_population_state_amplitudes_11, dtype=complex)
    pop_amp_cir = np.asarray(result.circuit_population_state_amplitudes_11, dtype=complex)

    tr_duf = np.asarray(result.duffing_transition_signed_currents_11, dtype=float)
    tr_cir = np.asarray(result.circuit_transition_signed_currents_11, dtype=float)

    pop_rgb_duf = _phase_population_rgb(pop_amp_duf, pop_labels)
    pop_rgb_cir = _phase_population_rgb(pop_amp_cir, pop_labels)

    vabs = float(
        max(
            np.max(np.abs(tr_duf), initial=0.0),
            np.max(np.abs(tr_cir), initial=0.0),
            1e-12,
        )
    )
    tick_font_size = max(8.0, 0.42 * float(font_size))

    with benchmark_plot_style(font_size):
        transition_cmap = mcolors.LinearSegmentedColormap.from_list(
            "transition_blue_gray_red",
            [
                (0.0, "#2b6cb0"),
                (0.5, (0.92, 0.92, 0.92)),
                (1.0, "#c53030"),
            ],
            N=256,
        )

        fig = plt.figure(figsize=(13.5, 10.2))
        gs = fig.add_gridspec(
            2,
            3,
            width_ratios=(1.0, 1.0, 0.08),
            height_ratios=(1.0, 1.0),
            hspace=0.28,
            wspace=0.30,
        )

        ax_pop_duf = fig.add_subplot(gs[0, 0])
        ax_pop_cir = fig.add_subplot(gs[0, 1], sharex=ax_pop_duf)
        ax_tr_duf = fig.add_subplot(gs[1, 0], sharex=ax_pop_duf)
        ax_tr_cir = fig.add_subplot(gs[1, 1], sharex=ax_pop_duf)

        cbar_grid = gs[:, 2].subgridspec(2, 1, hspace=0.45, height_ratios=(1.0, 1.0))
        ax_cbar_phase = fig.add_subplot(cbar_grid[0, 0])
        ax_cbar_tr = fig.add_subplot(cbar_grid[1, 0])

        if pop_rgb_duf.size > 0:
            ax_pop_duf.imshow(
                pop_rgb_duf,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=(float(t[0]), float(t[-1]), -0.5, pop_rgb_duf.shape[0] - 0.5),
                zorder=2,
            )
            _set_y_ticks(ax_pop_duf, pop_labels, transition=False, tick_font_size=tick_font_size)
        else:
            ax_pop_duf.text(0.5, 0.5, "No states selected", ha="center", va="center", transform=ax_pop_duf.transAxes)
            ax_pop_duf.set_yticks([])

        if pop_rgb_cir.size > 0:
            ax_pop_cir.imshow(
                pop_rgb_cir,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=(float(t[0]), float(t[-1]), -0.5, pop_rgb_cir.shape[0] - 0.5),
                zorder=2,
            )
            _set_y_ticks(ax_pop_cir, pop_labels, transition=False, tick_font_size=tick_font_size)
        else:
            ax_pop_cir.text(0.5, 0.5, "No states selected", ha="center", va="center", transform=ax_pop_cir.transAxes)
            ax_pop_cir.set_yticks([])

        im_tr = ax_tr_duf.imshow(
            tr_duf.T if tr_duf.size > 0 else np.zeros((1, t.size), dtype=float),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(float(t[0]), float(t[-1]), -0.5, max(0, tr_duf.shape[1] - 1) + 0.5),
            vmin=-vabs,
            vmax=vabs,
            cmap=transition_cmap,
            zorder=2,
        )
        _set_y_ticks(ax_tr_duf, tr_labels, transition=True, tick_font_size=tick_font_size)

        ax_tr_cir.imshow(
            tr_cir.T if tr_cir.size > 0 else np.zeros((1, t.size), dtype=float),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(float(t[0]), float(t[-1]), -0.5, max(0, tr_cir.shape[1] - 1) + 0.5),
            vmin=-vabs,
            vmax=vabs,
            cmap=transition_cmap,
            zorder=2,
        )
        _set_y_ticks(ax_tr_cir, tr_labels, transition=True, tick_font_size=tick_font_size)

        _overlay_flux_track(
            ax_pop_duf,
            times_ns=t,
            pulse_flux_values=result.pulse_flux_values,
            idle_flux=float(result.idle_flux),
            target_flux=float(result.target_flux),
            n_rows=max(1, pop_rgb_duf.shape[0]),
        )
        _overlay_flux_track(
            ax_pop_cir,
            times_ns=t,
            pulse_flux_values=result.pulse_flux_values,
            idle_flux=float(result.idle_flux),
            target_flux=float(result.target_flux),
            n_rows=max(1, pop_rgb_cir.shape[0]),
        )
        _overlay_flux_track(
            ax_tr_duf,
            times_ns=t,
            pulse_flux_values=result.pulse_flux_values,
            idle_flux=float(result.idle_flux),
            target_flux=float(result.target_flux),
            n_rows=max(1, tr_duf.shape[1]),
        )
        _overlay_flux_track(
            ax_tr_cir,
            times_ns=t,
            pulse_flux_values=result.pulse_flux_values,
            idle_flux=float(result.idle_flux),
            target_flux=float(result.target_flux),
            n_rows=max(1, tr_cir.shape[1]),
        )

        ax_pop_duf.set_title("Duffing population+phase")
        ax_pop_cir.set_title("Circuit population+phase")
        ax_tr_duf.set_title("Duffing transitions")
        ax_tr_cir.set_title("Circuit transitions")

        ax_pop_duf.set_ylabel("States")
        ax_tr_duf.set_ylabel("Transitions")

        ax_tr_duf.set_xlabel("Time (ns)")
        ax_tr_cir.set_xlabel("Time (ns)")

        phase_mappable = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
        phase_mappable.set_array([])
        cbar_phase = fig.colorbar(phase_mappable, cax=ax_cbar_phase)
        cbar_phase.set_ticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
        cbar_phase.set_ticklabels(["$-\\pi$", "$-\\pi/2$", "$0$", "$\\pi/2$", "$\\pi$"])
        cbar_phase.set_label("Phase hue (rad)")
        #ax_cbar_phase.set_title("Color strength ~ sqrt(population)", fontsize=max(9.0, 0.62 * float(font_size)))

        cbar_tr = fig.colorbar(im_tr, cax=ax_cbar_tr)
        cbar_tr.set_label("Signed current (1/ns)")

        fig.subplots_adjust(left=0.20, right=0.95, bottom=0.08, top=0.92)

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
