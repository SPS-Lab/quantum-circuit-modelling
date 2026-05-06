"""Plot stationary vs oscillating behavior for the toy 5 GHz ideal LC oscillator."""

from __future__ import annotations

from math import factorial
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toolkit.helpers import destroy

OMEGA_GHZ = 5.0
TRUNCATED_DIM = 24
TOTAL_TIME_NS = 2.0
DT_NS = 0.001
COHERENT_ALPHA = 1.5


def _time_grid(total_time_ns: float, dt_ns: float) -> np.ndarray:
    total = float(total_time_ns)
    dt = float(dt_ns)
    grid = np.arange(0.0, total, dt, dtype=float)
    if grid.size == 0 or abs(grid[-1] - total) > 1e-12:
        grid = np.append(grid, total)
    return grid


def _coherent_state(alpha: complex, dim: int) -> np.ndarray:
    coeff = np.array(
        [
            np.exp(-0.5 * abs(alpha) ** 2) * (alpha ** n) / np.sqrt(float(factorial(n)))
            for n in range(int(dim))
        ],
        dtype=complex,
    )
    coeff /= np.linalg.norm(coeff)
    return coeff


def _evolve_state(H: np.ndarray, psi0: np.ndarray, times_ns: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(np.asarray(H, dtype=complex))
    coeff0 = evecs.conj().T @ np.asarray(psi0, dtype=complex)
    phase = np.exp(-1.0j * 2.0 * np.pi * evals[:, np.newaxis] * np.asarray(times_ns, dtype=float)[np.newaxis, :])
    return (evecs @ (coeff0[:, np.newaxis] * phase)).T


def _expectation_traces(psi_t: np.ndarray, op: np.ndarray) -> np.ndarray:
    op_psi = psi_t @ np.asarray(op, dtype=complex).T
    return np.einsum("ti,ti->t", np.conjugate(psi_t), op_psi)


def _build_toy_lc_operators(dim: int, omega_ghz: float) -> dict[str, np.ndarray]:
    a = destroy(int(dim))
    adag = a.conj().T
    eye = np.eye(int(dim), dtype=complex)
    x = (a + adag) / np.sqrt(2.0)
    p = -1.0j * (a - adag) / np.sqrt(2.0)
    h = float(omega_ghz) * (adag @ a + 0.5 * eye)
    e_flux = 0.5 * float(omega_ghz) * (x @ x)
    e_charge = 0.5 * float(omega_ghz) * (p @ p)
    return {
        "a": a,
        "adag": adag,
        "x": x,
        "p": p,
        "h": h,
        "e_flux": e_flux,
        "e_charge": e_charge,
    }


def main() -> None:
    times_ns = _time_grid(TOTAL_TIME_NS, DT_NS)
    ops = _build_toy_lc_operators(TRUNCATED_DIM, OMEGA_GHZ)

    ground = np.zeros(TRUNCATED_DIM, dtype=complex)
    ground[0] = 1.0
    coherent = _coherent_state(COHERENT_ALPHA, TRUNCATED_DIM)

    cases = [
        ("Ground state |0>", ground, "An energy eigenstate: raw ket phase winds, but all LC observables are stationary."),
        (rf"Coherent state $|\alpha={COHERENT_ALPHA:.1f}\rangle$", coherent, "A non-eigenstate: quadratures oscillate at 5 GHz and electric/magnetic energies slosh."),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.8))

    for row, (title, psi0, subtitle) in enumerate(cases):
        psi_t = _evolve_state(ops["h"], psi0, times_ns)
        x_t = np.real(_expectation_traces(psi_t, ops["x"]))
        p_t = np.real(_expectation_traces(psi_t, ops["p"]))
        e_flux_t = np.real(_expectation_traces(psi_t, ops["e_flux"]))
        e_charge_t = np.real(_expectation_traces(psi_t, ops["e_charge"]))
        e_total_t = np.real(_expectation_traces(psi_t, ops["h"]))

        coeff_ground = psi_t[:, 0]
        raw_ground_phase = np.unwrap(np.angle(coeff_ground))

        ax_q = axes[row, 0]
        ax_q.plot(times_ns, x_t, color="C0", linewidth=2.0, label=r"$\langle X \rangle$ (flux-like)")
        ax_q.plot(times_ns, p_t, color="C1", linewidth=2.0, label=r"$\langle P \rangle$ (charge-like)")
        ax_q.set_ylabel("Dimensionless quadrature")
        ax_q.set_title(title)
        ax_q.grid(True, alpha=0.3)
        if row == 1:
            ax_q.set_xlabel("Time (ns)")
        ax_q.legend(loc="upper right", fontsize="small", frameon=False)
        ax_q.text(
            0.02,
            0.98,
            subtitle,
            transform=ax_q.transAxes,
            va="top",
            ha="left",
            fontsize="small",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.85"},
        )
        ax_q_phase = ax_q.twinx()
        ax_q_phase.plot(times_ns, raw_ground_phase, color="0.35", linewidth=1.2, linestyle="--", alpha=0.7)
        ax_q_phase.set_ylabel(r"arg$\,c_0$ (rad)", color="0.35")
        ax_q_phase.tick_params(axis="y", labelcolor="0.35")

        ax_e = axes[row, 1]
        ax_e.plot(times_ns, e_charge_t, color="C3", linewidth=2.0, label=r"$\langle E_C \rangle$")
        ax_e.plot(times_ns, e_flux_t, color="C2", linewidth=2.0, label=r"$\langle E_L \rangle$")
        ax_e.plot(times_ns, e_total_t, color="k", linewidth=1.8, linestyle=":", label=r"$\langle H \rangle$")
        ax_e.set_ylabel("Energy (GHz)")
        ax_e.grid(True, alpha=0.3)
        if row == 1:
            ax_e.set_xlabel("Time (ns)")
        ax_e.legend(loc="upper right", fontsize="small", frameon=False)

        ax_phase = axes[row, 2]
        ax_phase.plot(x_t, p_t, color="C4", linewidth=2.0)
        ax_phase.scatter([x_t[0]], [p_t[0]], color="C4", s=28)
        ax_phase.set_xlabel(r"$\langle X \rangle$")
        ax_phase.set_ylabel(r"$\langle P \rangle$")
        ax_phase.set_title("Phase-space trajectory")
        ax_phase.grid(True, alpha=0.3)
        max_abs = max(1.0, float(np.max(np.abs(np.concatenate([x_t, p_t])))) * 1.15)
        ax_phase.set_xlim(-max_abs, max_abs)
        ax_phase.set_ylim(-max_abs, max_abs)
        ax_phase.set_aspect("equal", adjustable="box")

    fig.suptitle("Toy ideal LC oscillator at 5 GHz: stationary eigenstate vs LC-like oscillation")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    out = _REPO_ROOT / "results" / "single-5ghz" / "toy_lc_stationary_vs_oscillating.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)

    print(f"Wrote figure: {out}")


if __name__ == "__main__":
    main()
