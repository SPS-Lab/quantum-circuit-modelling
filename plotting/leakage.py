"""Plotting for the leakage benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.leakage import LeakageBenchmarkResult
from plotting.style import (
    DEFAULT_PLOT_FONT_SIZE,
    MODEL_LEGEND_BBOX_TO_ANCHOR,
    benchmark_plot_style,
    model_legend_handles,
    model_plot_kwargs,
)


def _state_label_key(label: str) -> tuple[int, int, int] | tuple[str]:
    text = str(label).strip()
    if text.startswith("|") and text.endswith(">"):
        parts = text[1:-1].split(",")
        if len(parts) == 3:
            try:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                return (text,)
    return (text,)


def _state_excitation_sum(label: str) -> int | None:
    key = _state_label_key(label)
    if len(key) == 3 and all(isinstance(v, int) for v in key):
        return int(key[0]) + int(key[1]) + int(key[2])
    return None


def _time_integral(values: np.ndarray, times_ns: np.ndarray) -> float:
    y = np.asarray(values, dtype=float).ravel()
    t = np.asarray(times_ns, dtype=float).ravel()
    try:
        return float(np.trapezoid(y, x=t))
    except AttributeError:  # pragma: no cover - compatibility fallback
        return float(np.trapz(y, x=t))


def _destinations_to_matrix(
    destinations: dict[str, np.ndarray],
    times_ns: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    t = np.asarray(times_ns, dtype=float).ravel()
    if not destinations:
        return [], np.zeros((0, t.size), dtype=float)

    raw_labels = [str(k) for k in destinations.keys()]
    rows_map: dict[str, np.ndarray] = {}
    for label in raw_labels:
        row = np.asarray(destinations[label], dtype=float).ravel()
        if row.shape != t.shape:
            raise ValueError(f"Destination trace {label!r} shape {row.shape} does not match times {t.shape}")
        rows_map[label] = row

    # Show all destinations, but rank rows by integrated population so the important
    # channels are grouped at the top of each heatmap.
    labels = sorted(
        raw_labels,
        key=lambda lbl: (-_time_integral(rows_map[lbl], t), _state_label_key(lbl)),
    )
    rows = [rows_map[label] for label in labels]
    return labels, np.vstack(rows)


def _top_n_with_other(
    labels: list[str],
    matrix: np.ndarray,
    *,
    top_n: int,
) -> tuple[list[str], np.ndarray]:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"Destination matrix must be 2D, got shape {m.shape}")
    n_rows = m.shape[0]
    if n_rows == 0:
        return labels, m

    n = int(max(1, top_n))
    if n_rows <= n:
        return labels, m

    head_labels = list(labels[:n])
    head = m[:n, :]

    # Reorder displayed ket rows by excitation sum so lower-sum states are lower
    # in the heatmap (ties are left to lexical state ordering).
    ranked = list(zip(head_labels, head))
    ranked.sort(
        key=lambda x: (
            _state_excitation_sum(x[0]) if _state_excitation_sum(x[0]) is not None else 10**9,
            _state_label_key(x[0]),
        )
    )
    head_labels = [label for label, _ in ranked]
    head = np.vstack([row for _, row in ranked])
    other = np.sum(m[n:, :], axis=0, keepdims=True)
    return head_labels + ["other"], np.vstack([head, other])


def _set_destination_ticks(ax: plt.Axes, labels: list[str], *, tick_font_size: float) -> None:
    n = len(labels)
    if n == 0:
        return
    idx = np.arange(n, dtype=int)
    ax.set_yticks(idx)
    ax.set_yticklabels([labels[i] for i in idx])
    ax.tick_params(axis="y", labelsize=tick_font_size, pad=2.5)


def plot_leakage_benchmark(
    result: LeakageBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
    top_destination_rows: int = 10,
) -> None:
    t = np.asarray(result.times_ns, dtype=float)
    labels_circuit, matrix_circuit = _destinations_to_matrix(
        result.circuit_leakage_destination_populations_11,
        t,
    )
    labels_duffing, matrix_duffing = _destinations_to_matrix(
        result.duffing_leakage_destination_populations_11,
        t,
    )
    labels_circuit, matrix_circuit = _top_n_with_other(
        labels_circuit,
        matrix_circuit,
        top_n=top_destination_rows,
    )
    labels_duffing, matrix_duffing = _top_n_with_other(
        labels_duffing,
        matrix_duffing,
        top_n=top_destination_rows,
    )
    vmax = float(max(np.max(matrix_circuit, initial=0.0), np.max(matrix_duffing, initial=0.0), 1e-12))
    heatmap_tick_font_size = max(10.0, 0.50 * float(font_size))

    with benchmark_plot_style(font_size):
        fig = plt.figure(figsize=(11.0, 10.0))
        gs = fig.add_gridspec(3, 3, width_ratios=(1.0, 1.0, 0.06), height_ratios=(1.0, 1.2, 1.2))
        top = gs[0, 0:2].subgridspec(1, 2, wspace=0.42)
        ax_flux = fig.add_subplot(top[0, 0])
        ax_leak = fig.add_subplot(top[0, 1], sharex=ax_flux)
        ax_circuit_dest = fig.add_subplot(gs[1, 0:2], sharex=ax_flux)
        ax_duffing_dest = fig.add_subplot(gs[2, 0:2], sharex=ax_flux)
        ax_cbar = fig.add_subplot(gs[1:, 2])

        ax_flux.plot(t, result.pulse_flux_values, color="C4", linewidth=2.0)
        ax_flux.axhline(result.idle_flux, color="0.4", linewidth=1.0)
        ax_flux.axhline(result.target_flux, color="0.3", linewidth=1.0)
        ax_flux.set_ylabel(r"Flux bias ($\Phi/\Phi_0$)")
        ax_flux.grid(True, alpha=0.3)

        ax_leak.plot(t, result.circuit_leakage_11, color="k", linewidth=2.0, **model_plot_kwargs("circuit"))
        ax_leak.plot(t, result.duffing_leakage_11, color="k", linewidth=2.0, **model_plot_kwargs("duffing"))
        ax_leak.plot(t, result.effective_leakage_11, color="k", linewidth=2.0, **model_plot_kwargs("effective"))
        ax_leak.set_ylabel(r"Leakage")
        ax_leak.grid(True, alpha=0.3)

        if matrix_circuit.size > 0:
            im_circuit = ax_circuit_dest.imshow(
                matrix_circuit,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=(float(t[0]), float(t[-1]), -0.5, matrix_circuit.shape[0] - 0.5),
                vmin=0.0,
                vmax=vmax,
                cmap="magma",
            )
            _set_destination_ticks(ax_circuit_dest, labels_circuit, tick_font_size=heatmap_tick_font_size)
        else:
            im_circuit = None
            ax_circuit_dest.text(
                0.5,
                0.5,
                "No per-destination leakage traces in this result file.",
                ha="center",
                va="center",
                transform=ax_circuit_dest.transAxes,
            )
            ax_circuit_dest.set_yticks([])
        ax_circuit_dest.set_ylabel("Circuit destination")

        if matrix_duffing.size > 0:
            im_duffing = ax_duffing_dest.imshow(
                matrix_duffing,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=(float(t[0]), float(t[-1]), -0.5, matrix_duffing.shape[0] - 0.5),
                vmin=0.0,
                vmax=vmax,
                cmap="magma",
            )
            _set_destination_ticks(ax_duffing_dest, labels_duffing, tick_font_size=heatmap_tick_font_size)
        else:
            im_duffing = None
            ax_duffing_dest.text(
                0.5,
                0.5,
                "No per-destination leakage traces in this result file.",
                ha="center",
                va="center",
                transform=ax_duffing_dest.transAxes,
            )
            ax_duffing_dest.set_yticks([])
        ax_duffing_dest.set_ylabel("Duffing destination")
        ax_duffing_dest.set_xlabel("Time (ns)")

        im_for_colorbar = im_circuit if im_circuit is not None else im_duffing
        if im_for_colorbar is not None:
            cbar = fig.colorbar(im_for_colorbar, cax=ax_cbar)
            cbar.set_label("Population")
        else:
            ax_cbar.axis("off")

        fig.legend(handles=model_legend_handles(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=MODEL_LEGEND_BBOX_TO_ANCHOR)
        fig.subplots_adjust(left=0.18, right=0.94, bottom=0.07, top=0.92, hspace=0.35, wspace=0.30)

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
