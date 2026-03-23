"""
Reusable state-evolution plots for any finite-dimensional Hamiltonian H.

Pass H as a square ``ndarray`` or as a nullary callable ``() -> H`` (e.g. ``lambda:
build_H(...)``). Model-specific parameters stay in the model module.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.linalg import expm

HamiltonianLike = Union[np.ndarray, Callable[[], np.ndarray]]


def as_hamiltonian(hamiltonian: HamiltonianLike) -> np.ndarray:
    """Return a square Hermitian matrix; accept array or ``() -> array``."""
    H = hamiltonian() if callable(hamiltonian) else hamiltonian
    H = np.asarray(H, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(
            "hamiltonian must be a square 2D array or a callable returning one, "
            f"got shape {getattr(H, 'shape', None)}"
        )
    return H


def evolve_state(psi0: np.ndarray, H: np.ndarray, t_values: np.ndarray) -> list[np.ndarray]:
    """Schrödinger evolution |psi(t)> = exp(-i H t) |psi0> for fixed H."""
    H = np.asarray(H, dtype=complex)
    psi0 = np.asarray(psi0, dtype=complex).ravel()
    if H.shape[0] != psi0.shape[0]:
        raise ValueError(
            f"H shape {H.shape} incompatible with psi0 length {psi0.shape[0]}"
        )
    out: list[np.ndarray] = []
    for t in t_values:
        U = expm(-1j * H * t)
        out.append(U @ psi0)
    return out


def plot_evolve_state(
    hamiltonian: HamiltonianLike,
    psi0: Optional[np.ndarray] = None,
    t: float = 4.0,
    dt: float = 0.05,
    outfile: str = "statevector_evolution.pdf",
    style: str = "panels",
    phase_amp_floor: float = 1e-12,
    basis_labels: Optional[Sequence[str]] = None,
    ylabel_basis: str = "Basis index",
    time_unit: str = "ns",
    xlabel: Optional[str] = None,
    suptitle: Optional[str] = None,
) -> None:
    """Plot |ψ(t)⟩ in the computational basis used by ``H`` (axis order 0 … dim-1).

    Parameters
    ----------
    hamiltonian
        Static Hamiltonian matrix ``(dim, dim)``, or a callable with no arguments
        that returns such a matrix (so any parameterization can live outside this module).
    psi0
        Initial state, length ``dim``. If ``None``, start in the first basis state ``|0…0⟩``
        (coefficient 1 on index 0).
    basis_labels
        Length-``dim`` strings for legends / y-axis (e.g. ``[r'$|00\\rangle$', ...]``).
        Default: ``$|k\\rangle$`` for ``k = 0 … dim-1``.
    ylabel_basis
        Heatmap y-axis label when ``style="heatmap"``.
    time_unit
        Used in the default time axis label if ``xlabel`` is not set.
    xlabel
        Override for the time axis label (default ``f"Time ({time_unit})"``).
    suptitle
        Figure suptitle; sensible defaults per ``style`` if omitted.

    style : {"heatmap", "line", "panels"}
        ``heatmap``: HSV image (hue=phase, brightness=|amplitude|).
        ``line``: |c_k| and phase mod 2π vs time, all levels on two axes.
        ``panels``: one row per basis state (|0⟩ at bottom); twin y: |c_k| and phase.
    phase_amp_floor
        For ``line`` and ``panels``, mask phase where |c_k| is below this value.
    """
    if style not in ("heatmap", "line", "panels"):
        raise ValueError('style must be "heatmap", "line", or "panels"')

    H = as_hamiltonian(hamiltonian)
    dim = H.shape[0]

    if basis_labels is not None and len(basis_labels) != dim:
        raise ValueError(f"basis_labels length {len(basis_labels)} != dim {dim}")

    if psi0 is None:
        psi0_arr = np.zeros(dim, dtype=complex)
        psi0_arr[0] = 1.0
    else:
        psi0_arr = np.asarray(psi0, dtype=complex).ravel()
        if psi0_arr.shape[0] != dim:
            raise ValueError(f"psi0 length {psi0_arr.shape[0]} must match H dim {dim}")

    n_steps = max(1, int(t / dt))
    t_values = np.linspace(0, t, n_steps)
    psi_t = evolve_state(psi0_arr, H, t_values)

    # (dim, n_times)
    psi_arr = np.array(psi_t).T
    amp = np.abs(psi_arr)
    phase = np.angle(psi_arr)

    labels = (
        list(basis_labels)
        if basis_labels is not None
        else [rf"$|{k}\rangle$" for k in range(dim)]
    )

    if dim <= 10:
        level_colors = plt.cm.tab10(np.linspace(0, 1, dim, endpoint=False))
    else:
        level_colors = plt.cm.tab20(np.linspace(0, 1, min(dim, 20), endpoint=False))

    xlab = xlabel if xlabel is not None else f"Time ({time_unit})"

    if suptitle is None:
        suptitle = {
            "heatmap": "Statevector evolution (hue=phase, brightness=|amplitude|)",
            "line": "Statevector evolution",
            "panels": "Statevector evolution (one panel per basis state)",
        }[style]

    if style == "panels":
        fig_h = max(4.0, 1.6 * dim)
        fig, axes = plt.subplots(
            dim,
            1,
            figsize=(10, fig_h),
            sharex=True,
        )
        if dim == 1:
            axes = np.array([axes])
        else:
            axes = axes.ravel()

        # |0> at bottom, highest index at top (same as imshow origin="lower").
        for k in range(dim):
            ax = axes[dim - 1 - k]
            c = level_colors[k % len(level_colors)]
            lbl = labels[k]
            ax.plot(t_values, amp[k], color=c, linestyle="-", label=r"$|c_k|$")
            ax.set_ylabel(r"$|c_k|$", color=c)
            ax.tick_params(axis="y", labelcolor=c)
            ax.set_title(lbl, loc="left", fontsize="medium")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0.0)

            valid = amp[k] >= phase_amp_floor
            ph_line = np.where(valid, np.mod(phase[k], 2 * np.pi), np.nan)
            ax2 = ax.twinx()
            ax2.plot(
                t_values,
                ph_line,
                color=c,
                linestyle="--",
                alpha=0.85,
                label=r"phase (mod $2\pi$)",
            )
            ax2.set_ylabel(r"phase (mod $2\pi$)", color=c)
            ax2.tick_params(axis="y", labelcolor=c)
            ax2.set_ylim(0.0, 2 * np.pi)
            ax2.set_yticks(
                [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
            )
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize="x-small")

        axes[-1].set_xlabel(xlab)
        fig.suptitle(suptitle, fontsize="medium")
        fig.tight_layout()
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
        return

    if style == "line":
        fig, (ax_amp, ax_ph) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
        )
        for k in range(dim):
            c = level_colors[k % len(level_colors)]
            lbl = labels[k]
            ax_amp.plot(t_values, amp[k], label=lbl, color=c)
            valid = amp[k] >= phase_amp_floor
            ph_line = np.where(valid, np.mod(phase[k], 2 * np.pi), np.nan)
            ax_ph.plot(t_values, ph_line, label=lbl, color=c)

        ax_amp.set_ylabel(r"$|c_k(t)|$")
        ax_amp.set_title("Amplitude by basis component")
        ax_amp.legend(loc="upper right", ncol=2, fontsize="small")
        ax_amp.grid(True, alpha=0.3)

        ax_ph.set_xlabel(xlab)
        ax_ph.set_ylabel(r"Phase (rad, mod $2\pi$)")
        ax_ph.set_title(
            r"Phase by basis component in $[0,\,2\pi)$ (omitted where $|c_k|$ below floor)"
        )
        ax_ph.set_ylim(0.0, 2 * np.pi)
        ax_ph.set_yticks(
            [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
            [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
        )
        ax_ph.legend(loc="upper right", ncol=2, fontsize="small")
        ax_ph.grid(True, alpha=0.3)

        fig.suptitle(suptitle, y=1.02)
        fig.tight_layout()
        plt.savefig(outfile, format="pdf")
        plt.close(fig)
        return

    # --- heatmap ---
    amplitude = np.clip(amp, 0.0, 1.0)
    hue = (phase + np.pi) / (2 * np.pi)
    saturation = np.ones_like(hue)
    value = amplitude
    hsv = np.stack((hue, saturation, value), axis=-1)
    rgb = hsv_to_rgb(hsv)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(
        rgb,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        resample=False,
        extent=[t_values[0], t_values[-1], -0.5, psi_arr.shape[0] - 0.5],
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylabel_basis)
    ax.set_title(suptitle)
    ax.set_yticks(np.arange(dim))
    ax.set_yticklabels(labels)

    phase_mappable = plt.cm.ScalarMappable(
        cmap="hsv", norm=plt.Normalize(vmin=-np.pi, vmax=np.pi)
    )
    phase_mappable.set_array([])
    cbar = fig.colorbar(phase_mappable, ax=ax)
    cbar.set_ticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
    cbar.set_ticklabels(["$-\pi$", "$-\pi/2$", "$0$", "$\pi/2$", "$\pi$"])
    cbar.set_label("Phase")

    fig.tight_layout()
    plt.savefig(outfile, format="pdf")
    plt.close(fig)
