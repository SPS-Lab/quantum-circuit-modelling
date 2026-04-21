"""Plotting for the combined leakage/population + transition-flow benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
    *,
    preferred_reference_label: str = "|1,0,1>",
    reference_floor: float = 1e-10,
) -> np.ndarray:
    amp = np.asarray(amplitudes, dtype=complex)
    if amp.ndim != 2:
        raise ValueError(f"amplitudes must be 2D (n_time, n_state), got {amp.shape}")

    n_time, n_state = amp.shape
    if n_state == 0:
        return np.zeros((0, n_time, 3), dtype=float)

    pop = np.clip(np.abs(amp) ** 2, 0.0, 1.0)

    ref_idx = 0
    if preferred_reference_label in labels:
        ref_idx = int(labels.index(preferred_reference_label))

    phase_ref = np.zeros(n_time, dtype=float)
    for m in range(n_time):
        if pop[m, ref_idx] >= float(reference_floor):
            use_idx = ref_idx
        else:
            use_idx = int(np.argmax(pop[m, :]))
        phase_ref[m] = float(np.angle(amp[m, use_idx]))

    phase_rot = np.exp(-1.0j * phase_ref)[:, np.newaxis]
    amp_rel = amp * phase_rot
    phase = np.angle(amp_rel)

    hue = (phase + np.pi) / (2.0 * np.pi)
    saturation = np.ones_like(hue)
    value = pop
    hsv = np.stack((hue, saturation, value), axis=-1)
    rgb = hsv_to_rgb(hsv)
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


def plot_leakage_flow_benchmark(
    result: LeakageFlowBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    t = np.asarray(result.times_ns, dtype=float).ravel()

    pop_labels_eff = _decode_labels(result.effective_population_state_labels_11)
    pop_labels_duf = _decode_labels(result.duffing_population_state_labels_11)
    pop_labels_cir = _decode_labels(result.circuit_population_state_labels_11)

    tr_labels_eff = _decode_labels(result.effective_transition_labels_11)
    tr_labels_duf = _decode_labels(result.duffing_transition_labels_11)
    tr_labels_cir = _decode_labels(result.circuit_transition_labels_11)

    pop_amp_eff = np.asarray(result.effective_population_state_amplitudes_11, dtype=complex)
    pop_amp_duf = np.asarray(result.duffing_population_state_amplitudes_11, dtype=complex)
    pop_amp_cir = np.asarray(result.circuit_population_state_amplitudes_11, dtype=complex)

    tr_eff = np.asarray(result.effective_transition_signed_currents_11, dtype=float)
    tr_duf = np.asarray(result.duffing_transition_signed_currents_11, dtype=float)
    tr_cir = np.asarray(result.circuit_transition_signed_currents_11, dtype=float)

    pop_rgb_eff = _phase_population_rgb(pop_amp_eff, pop_labels_eff)
    pop_rgb_duf = _phase_population_rgb(pop_amp_duf, pop_labels_duf)
    pop_rgb_cir = _phase_population_rgb(pop_amp_cir, pop_labels_cir)

    vabs = float(
        max(
            np.max(np.abs(tr_eff), initial=0.0),
            np.max(np.abs(tr_duf), initial=0.0),
            np.max(np.abs(tr_cir), initial=0.0),
            1e-12,
        )
    )
    tick_font_size = max(8.0, 0.42 * float(font_size))

    with benchmark_plot_style(font_size):
        fig = plt.figure(figsize=(16.0, 10.5))
        gs = fig.add_gridspec(
            2,
            4,
            width_ratios=(1.0, 1.0, 1.0, 0.08),
            height_ratios=(1.0, 1.0),
            hspace=0.28,
            wspace=0.30,
        )

        ax_pop_eff = fig.add_subplot(gs[0, 0])
        ax_pop_duf = fig.add_subplot(gs[0, 1], sharex=ax_pop_eff)
        ax_pop_cir = fig.add_subplot(gs[0, 2], sharex=ax_pop_eff)
        ax_tr_eff = fig.add_subplot(gs[1, 0], sharex=ax_pop_eff)
        ax_tr_duf = fig.add_subplot(gs[1, 1], sharex=ax_pop_eff)
        ax_tr_cir = fig.add_subplot(gs[1, 2], sharex=ax_pop_eff)

        cbar_grid = gs[:, 3].subgridspec(2, 1, hspace=0.45, height_ratios=(1.0, 1.0))
        ax_cbar_phase = fig.add_subplot(cbar_grid[0, 0])
        ax_cbar_tr = fig.add_subplot(cbar_grid[1, 0])

        if pop_rgb_eff.size > 0:
            ax_pop_eff.imshow(
                pop_rgb_eff,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=(float(t[0]), float(t[-1]), -0.5, pop_rgb_eff.shape[0] - 0.5),
            )
            _set_y_ticks(ax_pop_eff, pop_labels_eff, transition=False, tick_font_size=tick_font_size)
        else:
            ax_pop_eff.text(0.5, 0.5, "No states selected", ha="center", va="center", transform=ax_pop_eff.transAxes)
            ax_pop_eff.set_yticks([])

        if pop_rgb_duf.size > 0:
            ax_pop_duf.imshow(
                pop_rgb_duf,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=(float(t[0]), float(t[-1]), -0.5, pop_rgb_duf.shape[0] - 0.5),
            )
            _set_y_ticks(ax_pop_duf, pop_labels_duf, transition=False, tick_font_size=tick_font_size)
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
            )
            _set_y_ticks(ax_pop_cir, pop_labels_cir, transition=False, tick_font_size=tick_font_size)
        else:
            ax_pop_cir.text(0.5, 0.5, "No states selected", ha="center", va="center", transform=ax_pop_cir.transAxes)
            ax_pop_cir.set_yticks([])

        im_tr_eff = ax_tr_eff.imshow(
            tr_eff.T if tr_eff.size > 0 else np.zeros((1, t.size), dtype=float),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(float(t[0]), float(t[-1]), -0.5, max(0, tr_eff.shape[1] - 1) + 0.5),
            vmin=-vabs,
            vmax=vabs,
            cmap="coolwarm",
        )
        _set_y_ticks(ax_tr_eff, tr_labels_eff, transition=True, tick_font_size=tick_font_size)

        ax_tr_duf.imshow(
            tr_duf.T if tr_duf.size > 0 else np.zeros((1, t.size), dtype=float),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(float(t[0]), float(t[-1]), -0.5, max(0, tr_duf.shape[1] - 1) + 0.5),
            vmin=-vabs,
            vmax=vabs,
            cmap="coolwarm",
        )
        _set_y_ticks(ax_tr_duf, tr_labels_duf, transition=True, tick_font_size=tick_font_size)

        ax_tr_cir.imshow(
            tr_cir.T if tr_cir.size > 0 else np.zeros((1, t.size), dtype=float),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(float(t[0]), float(t[-1]), -0.5, max(0, tr_cir.shape[1] - 1) + 0.5),
            vmin=-vabs,
            vmax=vabs,
            cmap="coolwarm",
        )
        _set_y_ticks(ax_tr_cir, tr_labels_cir, transition=True, tick_font_size=tick_font_size)

        ax_pop_eff.set_title("Effective population+phase")
        ax_pop_duf.set_title("Duffing population+phase")
        ax_pop_cir.set_title("Circuit population+phase")
        ax_tr_eff.set_title("Effective transitions")
        ax_tr_duf.set_title("Duffing transitions")
        ax_tr_cir.set_title("Circuit transitions")

        ax_pop_eff.set_ylabel("States")
        ax_tr_eff.set_ylabel("Transitions")

        ax_tr_eff.set_xlabel("Time (ns)")
        ax_tr_duf.set_xlabel("Time (ns)")
        ax_tr_cir.set_xlabel("Time (ns)")

        phase_mappable = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
        phase_mappable.set_array([])
        cbar_phase = fig.colorbar(phase_mappable, cax=ax_cbar_phase)
        cbar_phase.set_ticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
        cbar_phase.set_ticklabels(["$-\\pi$", "$-\\pi/2$", "$0$", "$\\pi/2$", "$\\pi$"])
        cbar_phase.set_label("Phase hue (rad)")
        ax_cbar_phase.set_title("Brightness = population", fontsize=max(9.0, 0.62 * float(font_size)))

        cbar_tr = fig.colorbar(im_tr_eff, cax=ax_cbar_tr)
        cbar_tr.set_label("Signed current (1/ns)")

        fig.suptitle(title)
        fig.subplots_adjust(left=0.20, right=0.95, bottom=0.08, top=0.92)

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
