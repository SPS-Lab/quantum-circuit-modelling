"""Plotting for the static benchmark. All numerics in GHz."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from comparison.static import StaticBenchmarkResult
from plotting.leakage_flow import _phase_population_rgb
from plotting.style import (
    BENCHMARK_TIGHT_LAYOUT_H_PAD,
    BENCHMARK_TIGHT_LAYOUT_RECT,
    BENCHMARK_TIGHT_LAYOUT_W_PAD,
    DEFAULT_PLOT_FONT_SIZE,
    MODEL_LEGEND_BBOX_TO_ANCHOR,
    MODEL_ALPHA_CIRCUIT,
    MODEL_ALPHA_DUFFING,
    energy_level_alpha,
    STATIC_LEVEL_LEGEND_BBOX_TO_ANCHOR,
    STATIC_LEVEL_LEGEND_FONT_SCALE,
    STATIC_LEVEL_LEGEND_LOC,
    STATIC_LEVEL_LEGEND_NCOL,
    benchmark_plot_style,
    model_color,
    model_legend_handles,
    model_plot_kwargs,
)


def _plot_static_energy_panel(
    ax: plt.Axes,
    flux: np.ndarray,
    *,
    circuit_relative: np.ndarray,
    duffing_relative: np.ndarray,
    effective_relative: np.ndarray | None = None,
    circuit_full_relative: np.ndarray | None = None,
    duffing_full_relative: np.ndarray | None = None,
) -> None:
    if circuit_full_relative is not None and duffing_full_relative is not None:
        n_full = int(circuit_full_relative.shape[1])
        if n_full > 4:
            for i in range(1, n_full):
                level_alpha = energy_level_alpha(i - 1)
                ax.plot(
                    flux,
                    circuit_full_relative[:, i],
                    color=model_color("circuit"),
                    linewidth=0.8,
                    alpha=MODEL_ALPHA_CIRCUIT * level_alpha * 0.45,
                )
                ax.plot(
                    flux,
                    duffing_full_relative[:, i],
                    color=model_color("duffing"),
                    linewidth=0.8,
                    alpha=MODEL_ALPHA_DUFFING * level_alpha * 0.45,
                )

    for i in (1, 2, 3):
        level_alpha = energy_level_alpha(i - 1)
        ax.plot(
            flux,
            circuit_relative[:, i],
            linewidth=1.8,
            color=model_color("circuit"),
            alpha=MODEL_ALPHA_CIRCUIT * level_alpha,
        )
        ax.plot(
            flux,
            duffing_relative[:, i],
            linewidth=1.8,
            color=model_color("duffing"),
            alpha=MODEL_ALPHA_DUFFING * level_alpha,
        )
        if effective_relative is not None:
            ax.plot(
                flux,
                effective_relative[:, i],
                linewidth=1.8,
                color=model_color("effective"),
                alpha=model_plot_kwargs("effective")["alpha"] * level_alpha,
            )


def _static_level_legend(font_size: float) -> list[Line2D]:
    return [
        Line2D([0], [0], color="0.15", linewidth=1.8, alpha=energy_level_alpha(0), label=r"$E_{1}$"),
        Line2D([0], [0], color="0.15", linewidth=1.8, alpha=energy_level_alpha(1), label=r"$E_{2}$"),
        Line2D([0], [0], color="0.15", linewidth=1.8, alpha=energy_level_alpha(2), label=r"$E_{3}$"),
        Line2D([0], [0], color="0.15", linewidth=1.1, alpha=energy_level_alpha(3) * 0.7, label="other levels"),
    ]


def plot_static_benchmark(
    result: StaticBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    flux = np.asarray(result.flux_values, dtype=float)

    with benchmark_plot_style(font_size):
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
        axE, axErr, axJ, axZeta = axes.ravel()

        _plot_static_energy_panel(
            axE,
            flux,
            circuit_relative=result.circuit_relative_energies,
            duffing_relative=result.duffing_relative_energies,
            effective_relative=result.effective_relative_energies,
            circuit_full_relative=result.circuit_full_relative_energies,
            duffing_full_relative=result.duffing_full_relative_energies,
        )
        axE.set_ylabel("Energy rel. ground")
        axE.grid(True, alpha=0.3)
        axE.legend(
            handles=_static_level_legend(font_size),
            loc=STATIC_LEVEL_LEGEND_LOC,
            bbox_to_anchor=STATIC_LEVEL_LEGEND_BBOX_TO_ANCHOR,
            ncol=STATIC_LEVEL_LEGEND_NCOL,
            fontsize=font_size * STATIC_LEVEL_LEGEND_FONT_SCALE,
            framealpha=0.9,
            borderpad=0.25,
            labelspacing=0.25,
            handlelength=1.4,
            columnspacing=0.9,
            title="Levels (alpha)",
        )

        axErr.plot(flux, result.effective_error_rmse, linewidth=1.8, **model_plot_kwargs("effective"))
        axErr.plot(flux, result.duffing_error_rmse, linewidth=1.8, **model_plot_kwargs("duffing"))
        y_max = float(max(np.max(result.effective_error_rmse), np.max(result.duffing_error_rmse)))
        if np.any(result.near_mask):
            axErr.fill_between(flux, 0.0, y_max * 1.05, where=result.near_mask, color="C3", alpha=0.08)
        if np.any(result.idle_mask):
            axErr.fill_between(flux, 0.0, y_max * 1.05, where=result.idle_mask, color="C0", alpha=0.05)
        axErr.set_ylabel("Per-flux RMSE")
        axErr.grid(True, alpha=0.3)

        axJ.plot(flux, result.circuit_parameters["J"], linewidth=1.8, **model_plot_kwargs("circuit"))
        axJ.plot(flux, result.duffing_parameters["J"], linewidth=1.8, **model_plot_kwargs("duffing"))
        axJ.plot(flux, result.effective_parameters["J"], linewidth=1.8, **model_plot_kwargs("effective"))
        axJ.axhline(0.0, color="0.35", linewidth=1.0)
        axJ.set_ylabel(r"Exchange $J$")
        axJ.grid(True, alpha=0.3)

        axZeta.plot(flux, result.circuit_parameters["zeta"], linewidth=1.8, **model_plot_kwargs("circuit"))
        axZeta.plot(flux, result.duffing_parameters["zeta"], linewidth=1.8, **model_plot_kwargs("duffing"))
        axZeta.plot(flux, result.effective_parameters["zeta"], linewidth=1.8, **model_plot_kwargs("effective"))
        axZeta.axhline(0.0, color="0.35", linewidth=1.0)
        axZeta.set_ylabel(r"Residual ZZ $\zeta$")
        axZeta.grid(True, alpha=0.3)

        axes[1, 0].set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
        axes[1, 1].set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
        fig.legend(handles=model_legend_handles(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=MODEL_LEGEND_BBOX_TO_ANCHOR)
        fig.subplots_adjust(left=0.1, right=0.93, bottom=0.06, top=0.96)

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)


def plot_static_raw_energies(
    result: StaticBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    flux = np.asarray(result.flux_values, dtype=float)

    with benchmark_plot_style(font_size):
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.6), sharex=True)
        _plot_static_energy_panel(
            ax,
            flux,
            circuit_relative=result.circuit_raw_energies,
            duffing_relative=result.duffing_raw_energies,
            effective_relative=result.effective_raw_energies,
            circuit_full_relative=result.circuit_full_raw_energies,
            duffing_full_relative=result.duffing_full_raw_energies,
        )
        ax.set_ylabel("Raw energy")
        ax.set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
        ax.grid(True, alpha=0.3)
        ax.legend(
            handles=_static_level_legend(font_size),
            loc=STATIC_LEVEL_LEGEND_LOC,
            bbox_to_anchor=STATIC_LEVEL_LEGEND_BBOX_TO_ANCHOR,
            ncol=STATIC_LEVEL_LEGEND_NCOL,
            fontsize=font_size * STATIC_LEVEL_LEGEND_FONT_SCALE,
            framealpha=0.9,
            borderpad=0.25,
            labelspacing=0.25,
            handlelength=1.4,
            columnspacing=0.9,
            title="Levels (alpha)",
        )
        fig.legend(
            handles=model_legend_handles(),
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=MODEL_LEGEND_BBOX_TO_ANCHOR,
        )
        fig.subplots_adjust(left=0.1, right=0.93, bottom=0.06, top=0.96)

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)


def plot_static_single_excitation_overlaps(
    result: StaticBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    flux = np.asarray(result.flux_values, dtype=float)

    with benchmark_plot_style(font_size):
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True, sharey=True)
        panels = (
            ("circuit", axes[0], np.asarray(result.circuit_computational_bare_overlaps, dtype=float)),
            ("duffing", axes[1], np.asarray(result.duffing_computational_bare_overlaps, dtype=float)),
        )
        bare_labels = (r"$|01\rangle$", r"$|10\rangle$")
        branch_labels = (r"$E_{1}$ branch", r"$E_{2}$ branch")
        colors = ("C0", "C1")
        linestyles = ("-", "--")

        for model_name, ax, overlaps in panels:
            for bare_idx, bare_label in enumerate(bare_labels, start=1):
                for branch_offset, branch_label in enumerate(branch_labels, start=1):
                    ax.plot(
                        flux,
                        overlaps[:, bare_idx, branch_offset],
                        color=colors[bare_idx - 1],
                        linestyle=linestyles[branch_offset - 1],
                        linewidth=1.8,
                        label=f"{branch_label} vs {bare_label}",
                    )
            ax.set_title(model_name)
            ax.set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(r"Bare overlap $|\langle \mathrm{bare} | \mathrm{dressed} \rangle|^2$")
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, framealpha=0.9)
        fig.subplots_adjust(left=0.1, right=0.93, bottom=0.06, top=0.96)

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)


def plot_static_computational_basis_amplitudes(
    result: StaticBenchmarkResult,
    outfile: Path,
    title: str,
    font_size: float = DEFAULT_PLOT_FONT_SIZE,
) -> None:
    flux = np.asarray(result.flux_values, dtype=float)
    basis_labels = (
        r"$|00\rangle$ component",
        r"$|01\rangle$ component",
        r"$|10\rangle$ component",
        r"$|11\rangle$ component",
    )
    branch_ticklabels = ("0", "1", "2", "3")
    panels = (
        ("Circuit", np.asarray(result.circuit_computational_bare_amplitudes, dtype=complex)),
        ("Duffing", np.asarray(result.duffing_computational_bare_amplitudes, dtype=complex)),
    )

    with benchmark_plot_style(font_size):
        fig = plt.figure(figsize=(12.6, 10.2))
        gs = fig.add_gridspec(
            4,
            3,
            width_ratios=(1.0, 1.0, 0.06),
            hspace=0.22,
            wspace=0.18,
        )
        axes = np.empty((4, 2), dtype=object)
        for row in range(4):
            for col in range(2):
                sharex = axes[0, col] if row > 0 else None
                sharey = axes[row, 0] if col > 0 else None
                axes[row, col] = fig.add_subplot(gs[row, col], sharex=sharex, sharey=sharey)

        cax = fig.add_subplot(gs[:, 2])

        for col, (model_name, amplitudes) in enumerate(panels):
            for row, basis_label in enumerate(basis_labels):
                ax = axes[row, col]
                rgb = _phase_population_rgb(amplitudes[:, row, :], [f"branch_{k}" for k in range(4)])
                ax.imshow(
                    rgb,
                    aspect="auto",
                    origin="lower",
                    interpolation="nearest",
                    extent=(float(flux[0]), float(flux[-1]), -0.5, 3.5),
                    zorder=2,
                )
                ax.grid(True, alpha=0.12)
                ax.set_yticks(np.arange(4, dtype=int))
                if col == 0:
                    ax.set_yticklabels(branch_ticklabels)
                    ax.set_ylabel(f"{basis_label}\nTracked branch")
                else:
                    ax.tick_params(axis="y", labelleft=False)
                if row == 0:
                    ax.set_title(f"{model_name} population+phase")
                if row == 3:
                    ax.set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")

        phase_mappable = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
        phase_mappable.set_array([])
        cbar = fig.colorbar(phase_mappable, cax=cax)
        cbar.set_ticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
        cbar.set_ticklabels(["$-\\pi$", "$-\\pi/2$", "$0$", "$\\pi/2$", "$\\pi$"])
        cbar.set_label("Phase hue (rad)\nStrength ~ sqrt(population)")

        fig.subplots_adjust(left=0.1, right=0.93, bottom=0.06, top=0.96)

        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, format="pdf", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
