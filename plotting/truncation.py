"""Plotting for fixed-flux truncation benchmark. All in GHz units."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from comparison.truncation import TruncationBenchmarkResult
from plotting.style import (
    BENCHMARK_TIGHT_LAYOUT_H_PAD,
    BENCHMARK_TIGHT_LAYOUT_RECT,
    BENCHMARK_TIGHT_LAYOUT_W_PAD,
    DEFAULT_PLOT_FONT_SIZE,
    MODEL_LEGEND_BBOX_TO_ANCHOR,
    TRUNCATION_LEVEL_LEGEND_BBOX_TO_ANCHOR,
    TRUNCATION_LEVEL_LEGEND_FONT_SCALE,
    TRUNCATION_LEVEL_LEGEND_MAX_ITEMS,
    TRUNCATION_LEVEL_LEGEND_NCOL,
    TRUNCATION_LEVEL_LEGEND_SHOW_ON_DIFF,
    TRUNCATION_LEVEL_LEGEND_TITLE_FONT_SCALE,
    benchmark_plot_style,
    model_legend_handles,
    model_plot_kwargs,
)


def _compact_level_legend(
    handles: list[Line2D],
    labels: list[str],
    *,
    max_items: int,
) -> tuple[list[Line2D], list[str]]:
    """Keep level legends compact by collapsing middle entries to an ellipsis."""
    if len(handles) <= max_items:
        return handles, labels

    max_items = max(3, int(max_items))
    n_head = max(1, (max_items - 1) // 2)
    n_tail = max(1, max_items - n_head - 1)
    head_h = handles[:n_head]
    head_l = labels[:n_head]
    tail_h = handles[-n_tail:]
    tail_l = labels[-n_tail:]
    ellipsis = Line2D([], [], linestyle="none")
    return head_h + [ellipsis] + tail_h, head_l + [r"$\cdots$"] + tail_l


def plot_truncation_benchmark(
    result: TruncationBenchmarkResult,
    outfile: Path,
    title: str,
    *,
    lowest_excited_levels_to_plot: int = 6,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    x = np.asarray(result.duffing_ncut_values, dtype=float)
    with benchmark_plot_style(font_size):
        fig = plt.figure(figsize=(11.0, 8.0))
        gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
        ax_j = fig.add_subplot(gs[0, 0])
        ax_zeta = fig.add_subplot(gs[0, 1], sharex=ax_j)
        ax_levels = fig.add_subplot(gs[1, 0], sharex=ax_j)
        ax_diff = fig.add_subplot(gs[1, 1], sharex=ax_j)

        ax_j.plot(x, result.duffing_j, color="C0", marker="o", linewidth=1.8, **model_plot_kwargs("duffing"))
        ax_j.axhline(
            result.circuit_j,
            color="C0",
            linewidth=1.4,
            **model_plot_kwargs("circuit"),
        )
        ax_j.set_ylabel(r"Exchange $J$")
        ax_j.grid(True, alpha=0.3)

        ax_zeta.plot(x, result.duffing_zeta, color="C2", marker="o", linewidth=1.8, **model_plot_kwargs("duffing"))
        ax_zeta.axhline(
            result.circuit_zeta,
            color="C2",
            linewidth=1.4,
            **model_plot_kwargs("circuit"),
        )
        ax_zeta.set_ylabel(r"Residual ZZ $\zeta$")
        ax_zeta.grid(True, alpha=0.3)

        ax_j.set_xlabel("Duffing transmon ncut")
        ax_zeta.set_xlabel("Duffing transmon ncut")

        rel_duf = np.asarray(result.duffing_lowest_relative_energies, dtype=float)
        rel_cir = np.asarray(result.circuit_lowest_relative_energies, dtype=float).ravel()
        n_levels = int(min(rel_duf.shape[1], rel_cir.shape[0]))
        n_excited_to_show = int(min(max(1, int(lowest_excited_levels_to_plot)), max(0, n_levels - 1)))
        if n_levels > 1:
            level_handles: list[Line2D] = []
            level_labels: list[str] = []
            for i in range(1, 1 + n_excited_to_show):
                color = f"C{(i - 1) % 10}"
                label = rf"$E_{{{i}}}$"
                ax_levels.plot(x, rel_duf[:, i], color=color, linewidth=1.6, **model_plot_kwargs("duffing"))
                ax_levels.axhline(rel_cir[i], color=color, linewidth=1.2, **model_plot_kwargs("circuit"))
                ax_diff.plot(x, rel_duf[:, i] - rel_cir[i], color=color, linewidth=1.6, alpha=0.95)
                level_handles.append(Line2D([0], [0], color=color, linewidth=1.6, **model_plot_kwargs("duffing")))
                level_labels.append(label)

            compact_handles, compact_labels = _compact_level_legend(
                level_handles,
                level_labels,
                max_items=TRUNCATION_LEVEL_LEGEND_MAX_ITEMS,
            )
            legend_cols = min(int(TRUNCATION_LEVEL_LEGEND_NCOL), len(compact_handles))
            legend_kwargs = dict(
                handles=compact_handles,
                labels=compact_labels,
                loc="upper center",
                bbox_to_anchor=TRUNCATION_LEVEL_LEGEND_BBOX_TO_ANCHOR,
                ncol=legend_cols,
                title="Levels",
                framealpha=0.9,
                borderpad=0.25,
                labelspacing=0.25,
                handlelength=1.4,
                columnspacing=0.9,
                fontsize=font_size * TRUNCATION_LEVEL_LEGEND_FONT_SCALE,
                title_fontsize=font_size * TRUNCATION_LEVEL_LEGEND_TITLE_FONT_SCALE,
            )
            ax_levels.legend(**legend_kwargs)

            ax_diff.axhline(0.0, color="0.35", linewidth=1.0)
            if TRUNCATION_LEVEL_LEGEND_SHOW_ON_DIFF:
                ax_diff.legend(**legend_kwargs)
        else:
            ax_levels.text(0.5, 0.5, "Not enough levels to display", transform=ax_levels.transAxes, ha="center", va="center")
            ax_diff.text(0.5, 0.5, "Not enough levels to display", transform=ax_diff.transAxes, ha="center", va="center")
        ax_levels.set_ylabel("Energy rel. ground")
        ax_levels.set_xlabel("Duffing transmon ncut")
        ax_levels.grid(True, alpha=0.3)
        ax_diff.set_ylabel("Energy difference")
        ax_diff.set_xlabel("Duffing transmon ncut")
        ax_diff.grid(True, alpha=0.3)

        fig.legend(handles=model_legend_handles(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=MODEL_LEGEND_BBOX_TO_ANCHOR)
        fig.tight_layout(
            rect=BENCHMARK_TIGHT_LAYOUT_RECT,
            h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD,
            w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD,
        )

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
