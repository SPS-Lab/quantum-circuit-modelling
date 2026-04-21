"""Plotting for state-to-state leakage-current benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.state_to_state_leakage import StateToStateLeakageBenchmarkResult
from plotting.style import (
    DEFAULT_PLOT_FONT_SIZE,
    MODEL_LEGEND_BBOX_TO_ANCHOR,
    benchmark_plot_style,
    model_legend_handles,
    model_plot_kwargs,
)


def _time_integral(values: np.ndarray, times_ns: np.ndarray) -> float:
    y = np.asarray(values, dtype=float).ravel()
    t = np.asarray(times_ns, dtype=float).ravel()
    try:
        return float(np.trapezoid(y, x=t))
    except AttributeError:  # pragma: no cover - compatibility fallback
        return float(np.trapz(y, x=t))


def _transitions_to_matrix(
    transitions: dict[str, np.ndarray],
    times_ns: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    t = np.asarray(times_ns, dtype=float).ravel()
    if not transitions:
        return [], np.zeros((0, t.size), dtype=float)

    labels = [str(k) for k in transitions.keys()]
    traces: dict[str, np.ndarray] = {}
    for label in labels:
        row = np.asarray(transitions[label], dtype=float).ravel()
        if row.shape != t.shape:
            raise ValueError(f"Transition trace {label!r} shape {row.shape} does not match times {t.shape}")
        traces[label] = row

    labels = sorted(labels, key=lambda lbl: (-_time_integral(np.abs(traces[lbl]), t), lbl))
    rows = [traces[label] for label in labels]
    return labels, np.vstack(rows)


def _top_n_with_other(
    labels: list[str],
    matrix: np.ndarray,
    *,
    top_n: int,
) -> tuple[list[str], np.ndarray]:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"Transition matrix must be 2D, got shape {m.shape}")
    n_rows = int(m.shape[0])
    n = int(max(1, top_n))
    if n_rows <= n:
        return labels, m
    head_labels = list(labels[:n])
    head = m[:n, :]
    other = np.sum(m[n:, :], axis=0, keepdims=True)
    return head_labels + ["other transitions"], np.vstack([head, other])


def _set_ticks(ax: plt.Axes, labels: list[str], *, tick_font_size: float) -> None:
    def _to_math_label(label: str) -> str:
        text = str(label).strip()
        if text == "other transitions":
            return text
        if "->" in text:
            src, dst = text.split("->", 1)
            src = src.strip()
            dst = dst.strip()
            if src.startswith("|") and src.endswith(">") and dst.startswith("|") and dst.endswith(">"):
                src_inner = src[1:-1]
                dst_inner = dst[1:-1]
                return rf"$\left|{src_inner}\right\rangle \to \left|{dst_inner}\right\rangle$"
        if text.startswith("|") and text.endswith(">"):
            inner = text[1:-1]
            return rf"$\left|{inner}\right\rangle$"
        return text

    n = len(labels)
    if n == 0:
        return
    idx = np.arange(n, dtype=int)
    ax.set_yticks(idx)
    ax.set_yticklabels([_to_math_label(labels[i]) for i in idx])
    ax.tick_params(axis="y", labelsize=tick_font_size, pad=2.5)


def _net_current_from_transitions(transitions: dict[str, np.ndarray], times_ns: np.ndarray) -> np.ndarray:
    t = np.asarray(times_ns, dtype=float).ravel()
    if not transitions:
        return np.zeros(t.size, dtype=float)
    stacked = []
    for value in transitions.values():
        row = np.asarray(value, dtype=float).ravel()
        if row.shape != t.shape:
            raise ValueError(f"Transition trace shape {row.shape} does not match times {t.shape}")
        stacked.append(row)
    return np.sum(np.vstack(stacked), axis=0)


def plot_state_to_state_leakage_benchmark(
    result: StateToStateLeakageBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
    top_transition_rows: int = 12,
) -> None:
    t = np.asarray(result.times_ns, dtype=float)

    transitions_circuit = (
        result.circuit_comp_to_leak_signed_currents_11
        if result.circuit_comp_to_leak_signed_currents_11
        else result.circuit_comp_to_leak_currents_11
    )
    transitions_duffing = (
        result.duffing_comp_to_leak_signed_currents_11
        if result.duffing_comp_to_leak_signed_currents_11
        else result.duffing_comp_to_leak_currents_11
    )

    labels_circuit, matrix_circuit = _transitions_to_matrix(transitions_circuit, t)
    labels_duffing, matrix_duffing = _transitions_to_matrix(transitions_duffing, t)
    labels_circuit, matrix_circuit = _top_n_with_other(labels_circuit, matrix_circuit, top_n=top_transition_rows)
    labels_duffing, matrix_duffing = _top_n_with_other(labels_duffing, matrix_duffing, top_n=top_transition_rows)

    vabs = float(max(np.max(np.abs(matrix_circuit), initial=0.0), np.max(np.abs(matrix_duffing), initial=0.0), 1e-12))
    tick_font_size = max(8.0, 0.44 * float(font_size))
    net_circuit = (
        np.asarray(result.circuit_net_comp_to_leak_current_11, dtype=float).ravel()
        if np.asarray(result.circuit_net_comp_to_leak_current_11, dtype=float).size == t.size
        else _net_current_from_transitions(transitions_circuit, t)
    )
    net_duffing = (
        np.asarray(result.duffing_net_comp_to_leak_current_11, dtype=float).ravel()
        if np.asarray(result.duffing_net_comp_to_leak_current_11, dtype=float).size == t.size
        else _net_current_from_transitions(transitions_duffing, t)
    )

    with benchmark_plot_style(font_size):
        fig = plt.figure(figsize=(11.0, 10.0))
        gs = fig.add_gridspec(3, 3, width_ratios=(1.0, 1.0, 0.07), height_ratios=(1.0, 1.25, 1.25))
        top = gs[0, 0:2].subgridspec(1, 2, wspace=0.42)
        ax_flux = fig.add_subplot(top[0, 0])
        ax_leak = fig.add_subplot(top[0, 1], sharex=ax_flux)
        ax_circuit = fig.add_subplot(gs[1, 0:2], sharex=ax_flux)
        ax_duffing = fig.add_subplot(gs[2, 0:2], sharex=ax_flux)
        ax_cbar = fig.add_subplot(gs[1:, 2])

        ax_flux.plot(t, result.pulse_flux_values, color="C4", linewidth=2.0)
        ax_flux.axhline(result.idle_flux, color="0.4", linewidth=1.0)
        ax_flux.axhline(result.target_flux, color="0.3", linewidth=1.0)
        ax_flux.set_ylabel(r"Flux bias ($\Phi/\Phi_0$)")
        ax_flux.grid(True, alpha=0.3)

        ax_leak.plot(t, result.circuit_leakage_11, color="k", linewidth=2.0, **model_plot_kwargs("circuit"))
        ax_leak.plot(t, result.duffing_leakage_11, color="k", linewidth=2.0, **model_plot_kwargs("duffing"))
        ax_leak.plot(t, result.effective_leakage_11, color="k", linewidth=2.0, **model_plot_kwargs("effective"))
        ax_leak.set_ylabel("Leakage")
        ax_leak.grid(True, alpha=0.3)
        ax_net = ax_leak.twinx()
        ax_net.plot(t, net_circuit, color="C0", linewidth=1.7, linestyle=":")
        ax_net.plot(t, net_duffing, color="C1", linewidth=1.7, linestyle=":")
        ax_net.axhline(0.0, color="0.5", linewidth=1.0, alpha=0.7)
        ax_net.set_ylabel("Net comp->leak current (1/ns)")

        if matrix_circuit.size > 0:
            im_circuit = ax_circuit.imshow(
                matrix_circuit,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=(float(t[0]), float(t[-1]), -0.5, matrix_circuit.shape[0] - 0.5),
                vmin=-vabs,
                vmax=vabs,
                cmap="coolwarm",
            )
            _set_ticks(ax_circuit, labels_circuit, tick_font_size=tick_font_size)
        else:
            im_circuit = None
            ax_circuit.text(
                0.5,
                0.5,
                "No state-to-state current traces in this result file.",
                ha="center",
                va="center",
                transform=ax_circuit.transAxes,
            )
            ax_circuit.set_yticks([])
        ax_circuit.set_ylabel("Circuit transitions")

        if matrix_duffing.size > 0:
            im_duffing = ax_duffing.imshow(
                matrix_duffing,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=(float(t[0]), float(t[-1]), -0.5, matrix_duffing.shape[0] - 0.5),
                vmin=-vabs,
                vmax=vabs,
                cmap="coolwarm",
            )
            _set_ticks(ax_duffing, labels_duffing, tick_font_size=tick_font_size)
        else:
            im_duffing = None
            ax_duffing.text(
                0.5,
                0.5,
                "No state-to-state current traces in this result file.",
                ha="center",
                va="center",
                transform=ax_duffing.transAxes,
            )
            ax_duffing.set_yticks([])
        ax_duffing.set_ylabel("Duffing transitions")
        ax_duffing.set_xlabel("Time (ns)")

        im_for_colorbar = im_circuit if im_circuit is not None else im_duffing
        if im_for_colorbar is not None:
            cbar = fig.colorbar(im_for_colorbar, cax=ax_cbar)
            cbar.set_label("Signed current (1/ns)")
        else:
            ax_cbar.axis("off")

        fig.legend(
            handles=model_legend_handles(),
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=MODEL_LEGEND_BBOX_TO_ANCHOR,
        )
        fig.subplots_adjust(left=0.20, right=0.94, bottom=0.07, top=0.92, hspace=0.36, wspace=0.30)

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
