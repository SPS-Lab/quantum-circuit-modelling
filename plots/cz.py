"""Plotting for the CZ-relevant dynamics benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparison.cz import CzBenchmarkResult


def plot_cz_benchmark(result: CzBenchmarkResult, outfile: Path, title: str) -> None:
    t = np.asarray(result.times_ns, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    ax_flux, ax_phase, ax_prob, ax_p01 = axes.ravel()

    ax_flux.plot(t, result.pulse_flux_values, color="C4", linewidth=2.0)
    ax_flux.axhline(result.idle_flux, color="0.4", linestyle=":", linewidth=1.0, label="idle")
    ax_flux.axhline(result.target_flux, color="0.3", linestyle="--", linewidth=1.0, label="target")
    ax_flux.set_ylabel(r"Flux bias ($\Phi/\Phi_0$)")
    ax_flux.set_title(f"Shared CZ pulse ({result.sweep_target} sweep)")
    ax_flux.grid(True, alpha=0.3)
    ax_flux.legend(loc="best", fontsize="small")

    ax_phase.plot(t, result.circuit_conditional_phase, color="k", linewidth=2.0, label="circuit")
    ax_phase.plot(t, result.duffing_conditional_phase, color="C0", linestyle="--", linewidth=1.8, label="duffing")
    ax_phase.plot(t, result.effective_conditional_phase, color="C3", linestyle=":", linewidth=1.8, label="effective")
    ax_phase.set_ylabel("Conditional phase (rad)")
    ax_phase.set_title("Accumulated conditional phase")
    ax_phase.grid(True, alpha=0.3)
    ax_phase.legend(loc="best", fontsize="small")

    psi = np.asarray(result.circuit_statevector_plus_plus, dtype=complex)  # (n_time, 4)
    prob = np.abs(psi) ** 2
    ax_prob.plot(t, prob[:, 0], color="C1", linewidth=1.5, label=r"$P_{00}$")
    ax_prob.plot(t, prob[:, 1], color="C2", linewidth=1.5, label=r"$P_{01}$")
    ax_prob.plot(t, prob[:, 2], color="C3", linewidth=1.5, label=r"$P_{10}$")
    ax_prob.plot(t, prob[:, 3], color="k", linewidth=1.8, label=r"$P_{11}$")
    ax_prob.set_ylabel("Population")
    ax_prob.set_title(r"Circuit populations from $|++\rangle$")
    ax_prob.grid(True, alpha=0.3)
    ax_prob.legend(loc="best", fontsize="small", ncol=2)

    ax_p01.plot(t, result.circuit_populations_plus_plus[:, 1], color="k", linewidth=2.0, label="circuit")
    ax_p01.plot(t, result.duffing_populations_plus_plus[:, 1], color="C0", linestyle="--", linewidth=1.8, label="duffing")
    ax_p01.plot(t, result.effective_populations_plus_plus[:, 1], color="C3", linestyle=":", linewidth=1.8, label="effective")
    ax_p01.set_ylabel(r"$P_{01}(t)$")
    ax_p01.set_title(r"Population $P_{01}$ from $|++\rangle$")
    ax_p01.grid(True, alpha=0.3)
    ax_p01.legend(loc="best", fontsize="small")

    axes[1, 0].set_xlabel("Time (ns)")
    axes[1, 1].set_xlabel("Time (ns)")
    fig.suptitle(title)
    fig.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, format="pdf")
    plt.close(fig)
