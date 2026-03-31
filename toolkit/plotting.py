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
from toolkit.helpers import destroy

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


def plot_energy_levels(
    hamiltonian: HamiltonianLike,
    *,
    n_show: Optional[int] = None,
    subtract_ground: bool = True,
    outfile: str = "energy_levels.pdf",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    energy_unit: str = "",
    figsize: tuple[float, float] = (5.5, 7.0),
    annotate_n: int = 0,
) -> np.ndarray:
    """Plot low-lying eigenenergies of a Hermitian Hamiltonian (horizontal ladder).

    Eigenvalues are computed with ``numpy.linalg.eigvalsh``. By default energies are
    shifted so the ground level is at zero.

    Parameters
    ----------
    hamiltonian
        Square Hermitian matrix or ``() -> ndarray`` (same convention as
        :func:`plot_evolve_state`).
    n_show
        Plot only the lowest ``n_show`` levels. ``None`` means all (use caution for
        large Hilbert spaces).
    subtract_ground
        If True, subtract ``E_0`` from all plotted energies.
    outfile
        Output PDF path.
    title
        Figure title; a default is used if omitted.
    ylabel
        Energy axis label. If ``None``, a default is built from ``energy_unit`` and
        ``subtract_ground`` (e.g. ``Energy (GHz, rel. ground)``).
    energy_unit
        Appended to the default y-label, e.g. ``"GHz"`` -> ``Energy (GHz, rel. ground)``.
    figsize
        Matplotlib figure size.
    annotate_n
        Label the lowest ``annotate_n`` levels with their zero-based index (0, 1, …).

    Returns
    -------
    evals_plot : ndarray
        The energies actually plotted (same shape as the number of drawn levels),
        after optional ground subtraction.
    """
    H = as_hamiltonian(hamiltonian)
    dim = H.shape[0]
    evals = np.linalg.eigvalsh(H)
    evals = np.asarray(evals, dtype=float)

    if n_show is not None:
        evals = evals[: int(n_show)]
    n = evals.shape[0]

    if subtract_ground:
        e0 = evals[0]
        evals_plot = evals - e0
    else:
        evals_plot = evals.copy()

    fig, ax = plt.subplots(figsize=figsize)
    x0, x1 = 0.0, 1.0
    for E in evals_plot:
        ax.hlines(E, x0, x1, colors="C0", linewidth=1.2, alpha=0.9)

    ax.set_xlim(x0, x1)
    ax.set_xticks([])
    ax.set_xmargin(0.02)

    if ylabel is None:
        parts: list[str] = []
        if energy_unit:
            parts.append(energy_unit)
        if subtract_ground:
            parts.append("rel. ground")
        ax.set_ylabel(
            "Energy (" + ", ".join(parts) + ")" if parts else "Energy"
        )
    else:
        ax.set_ylabel(ylabel)

    if title is None:
        title = f"Energy levels (lowest {n} of {dim})"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    if annotate_n > 0:
        k_max = min(int(annotate_n), n)
        for k in range(k_max):
            ax.annotate(
                rf"${k}$",
                xy=(x1, evals_plot[k]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize="small",
                color="C0",
            )

    fig.tight_layout()
    plt.savefig(outfile, format="pdf")
    plt.close(fig)
    return evals_plot


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
    phase_prob_floor: float = 1e-12,
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
        ``heatmap``: HSV image (hue=phase, brightness=|c_k|^2).
        ``line``: |c_k|^2 and phase mod 2pi vs time, all levels on two axes.
        ``panels``: one row per basis state (|0⟩ at bottom); twin y: |c_k|^2 and phase.
    phase_prob_floor
        For ``line`` and ``panels``, mask phase where |c_k|^2 is below this value.

    Probability axes (``line`` and ``panels``) use a fixed y-range ``[0, 1]`` for |c_k|^2 when
    the state is normalized. The heatmap encodes |c_k|^2 in ``[0, 1]`` via clipping.
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
    prob = np.abs(psi_arr)**2  # probability |c_k|^2
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
            "heatmap": "Statevector evolution (hue=phase, brightness=|c_k|^2)",
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
            ax.plot(t_values, prob[k], color=c, linestyle="-", label=r"$|c_k|^2$")
            ax.set_ylabel(r"$|c_k|^2$", color=c)
            ax.tick_params(axis="y", labelcolor=c)
            ax.set_title(lbl, loc="left", fontsize="medium")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.0, 1.0)

            valid = prob[k] >= phase_prob_floor
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
        fig, (ax_prob, ax_ph) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
        )
        for k in range(dim):
            c = level_colors[k % len(level_colors)]
            lbl = labels[k]
            ax_prob.plot(t_values, prob[k], label=lbl, color=c)
            valid = prob[k] >= phase_prob_floor
            ph_line = np.where(valid, np.mod(phase[k], 2 * np.pi), np.nan)
            ax_ph.plot(t_values, ph_line, label=lbl, color=c)

        ax_prob.set_ylabel(r"$|c_k(t)|^2$")
        ax_prob.set_title("Probability by basis component")
        ax_prob.set_ylim(0.0, 1.0)
        ax_prob.legend(loc="upper right", ncol=2, fontsize="small")
        ax_prob.grid(True, alpha=0.3)

        ax_ph.set_xlabel(xlab)
        ax_ph.set_ylabel(r"Phase (rad, mod $2\pi$)")
        ax_ph.set_title(
            r"Phase by basis component in $[0,\,2\pi)$ (omitted where $|c_k|^2$ below floor)"
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
    brightness = np.clip(prob, 0.0, 1.0)
    hue = (phase + np.pi) / (2 * np.pi)
    saturation = np.ones_like(hue)
    value = brightness
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
