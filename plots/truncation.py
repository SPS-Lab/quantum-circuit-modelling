"""Plotting for fixed-flux truncation benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.truncation import TruncationBenchmarkResult


def plot_truncation_benchmark(result: TruncationBenchmarkResult, outfile: Path, title: str) -> None:
    x = np.asarray(result.duffing_ncut_values, dtype=float)
    fig = plt.figure(figsize=(11.0, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
    ax_j = fig.add_subplot(gs[0, 0])
    ax_zeta = fig.add_subplot(gs[0, 1], sharex=ax_j)
    ax_levels = fig.add_subplot(gs[1, :], sharex=ax_j)

    ax_j.plot(x, result.duffing_j, color="C0", marker="o", linewidth=1.8, label="duffing")
    ax_j.axhline(
        result.circuit_j,
        color="k",
        linestyle="--",
        linewidth=1.4,
        label=f"circuit (ncut={result.circuit_reference_ncut})",
    )
    ax_j.set_ylabel(r"Exchange $J$ (GHz)")
    ax_j.set_title(r"$J$ vs Duffing transmon ncut")
    ax_j.grid(True, alpha=0.3)
    ax_j.legend(loc="best", fontsize="small")

    ax_zeta.plot(x, result.duffing_zeta, color="C2", marker="o", linewidth=1.8, label="duffing")
    ax_zeta.axhline(
        result.circuit_zeta,
        color="k",
        linestyle="--",
        linewidth=1.4,
        label=f"circuit (ncut={result.circuit_reference_ncut})",
    )
    ax_zeta.set_ylabel(r"Residual ZZ $\zeta$ (GHz)")
    ax_zeta.set_title(r"$\zeta$ vs Duffing transmon ncut")
    ax_zeta.grid(True, alpha=0.3)
    ax_zeta.legend(loc="best", fontsize="small")

    ax_j.set_xlabel("Duffing transmon ncut")
    ax_zeta.set_xlabel("Duffing transmon ncut")

    rel_duf = np.asarray(result.duffing_lowest_relative_energies, dtype=float)
    rel_cir = np.asarray(result.circuit_lowest_relative_energies, dtype=float).ravel()
    n_levels = int(min(rel_duf.shape[1], rel_cir.shape[0]))
    if n_levels > 1:
        for i in range(1, n_levels):
            color = f"C{(i - 1) % 10}"
            ax_levels.plot(x, rel_duf[:, i], color=color, linewidth=1.6, alpha=0.95)
            ax_levels.axhline(rel_cir[i], color=color, linestyle="--", linewidth=1.2, alpha=0.9)
        ax_levels.plot([], [], color="0.2", linewidth=1.8, label="duffing levels")
        ax_levels.plot(
            [],
            [],
            color="0.2",
            linestyle="--",
            linewidth=1.3,
            label=f"circuit levels (ncut={result.circuit_reference_ncut})",
        )
        ax_levels.set_title("Lowest full-spectrum levels (relative to ground)")
        ax_levels.legend(loc="best", fontsize="small")
    else:
        ax_levels.text(0.5, 0.5, "Not enough levels to display", transform=ax_levels.transAxes, ha="center", va="center")
        ax_levels.set_title("Lowest full-spectrum levels")
    ax_levels.set_ylabel("Energy (GHz, rel. ground)")
    ax_levels.set_xlabel("Duffing transmon ncut")
    ax_levels.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, format="pdf")
    plt.close(fig)
