"""Plotting for the CZ-relevant dynamics benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from comparison.cz import CzBenchmarkResult


def plot_cz_benchmark(result: CzBenchmarkResult, outfile: Path, title: str) -> None:
    t = np.asarray(result.times_ns, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    ax_flux, ax_phase, ax_leak, ax_p11 = axes.ravel()

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

    ax_leak.plot(t, result.circuit_leakage_11, color="k", linewidth=2.0, label="circuit")
    ax_leak.plot(t, result.duffing_leakage_11, color="C0", linestyle="--", linewidth=1.8, label="duffing")
    ax_leak.plot(t, result.effective_leakage_11, color="C3", linestyle=":", linewidth=1.8, label="effective")
    ax_leak.set_ylabel(r"Leakage from $|11\rangle$")
    ax_leak.set_title(r"$L(t)$ for $|11\rangle$ input")
    ax_leak.grid(True, alpha=0.3)
    ax_leak.legend(loc="best", fontsize="small")

    psi = np.asarray(result.circuit_statevector_11, dtype=complex)  # (n_time, 4)
    prob = np.abs(psi.T) ** 2
    phase = np.angle(psi.T)
    hue = (phase + np.pi) / (2.0 * np.pi)
    sat = np.ones_like(hue)
    val = np.clip(prob, 0.0, 1.0)
    rgb = hsv_to_rgb(np.stack((hue, sat, val), axis=-1))
    ax_p11.imshow(
        rgb,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[t[0], t[-1], -0.5, 3.5],
    )
    ax_p11.set_ylabel("Circuit basis state")
    ax_p11.set_title(r"Circuit $|11\rangle$ statevector (hue=phase, value=$|c_k|^2$)")
    ax_p11.set_yticks([0, 1, 2, 3], [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"])

    axes[1, 0].set_xlabel("Time (ns)")
    axes[1, 1].set_xlabel("Time (ns)")
    fig.suptitle(title)
    fig.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, format="pdf")
    plt.close(fig)
