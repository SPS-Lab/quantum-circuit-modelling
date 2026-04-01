"""Plot helpers for the three-mode model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toolkit.plotting import plot_energy_levels, plot_energy_levels_vs_flux

from model2.analysis import dressed_computational_energies
from model2.core import (
    computational_subspace_block,
    coupler_frequency,
    three_mode_hamiltonian,
    three_mode_hamiltonian_from_kwargs,
    three_mode_hamiltonian_stack_vs_flux,
)


def plot_three_mode_zz_exchange_vs_flux(
    flux_values: np.ndarray,
    *,
    wc0: float,
    A: float,
    outfile: str = "three_mode_ZZ_exchange_vs_flux.pdf",
    title: str | None = None,
    dress_kw: dict | None = None,
    **ham_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot residual ZZ (dressed) and bare ``|01>-|10>`` splitting vs flux."""
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    dress_kw = dress_kw or {}

    zetas = np.empty_like(flux_values, dtype=float)
    exchanges = np.empty_like(flux_values, dtype=float)

    nq = int(ham_kwargs["nlevels_qubit"])
    nc = int(ham_kwargs["nlevels_coupler"])
    H_stack = three_mode_hamiltonian_stack_vs_flux(
        flux_values,
        wc0=wc0,
        A=A,
        ham_kwargs=ham_kwargs,
    )

    for k, H in enumerate(H_stack):
        E = dressed_computational_energies(H, nq, nc, **dress_kw)
        zetas[k] = E[3] - E[2] - E[1] + E[0]

        H_comp = computational_subspace_block(H, nq, nc, hermitianize=True)
        h11 = H_comp[1, 1].real
        h22 = H_comp[2, 2].real
        h12 = H_comp[1, 2]
        tr = h11 + h22
        det = h11 * h22 - h12 * np.conj(h12)
        disc = np.sqrt(max(0.0, 0.25 * tr**2 - det.real))
        exchanges[k] = 2.0 * disc

    fig, (ax_z, ax_j) = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True)
    ax_z.plot(flux_values, zetas, color="C0")
    ax_z.set_ylabel(r"Residual ZZ (GHz)")
    ax_z.set_title(title or r"Three-mode: $\zeta = E_{11}-E_{10}-E_{01}+E_{00}$ (dressed)")
    ax_z.grid(True, alpha=0.3)

    ax_j.plot(flux_values, exchanges, color="C1")
    ax_j.set_ylabel(r"Exchange bare $|01\rangle\!-\!|10\rangle$ splitting (GHz)")
    ax_j.set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
    ax_j.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(outfile, format="pdf")
    plt.close(fig)
    return zetas, exchanges


def plot_three_mode_energy_levels(
    outfile: str = "three_mode_energy_levels.pdf",
    n_show: int = 48,
    annotate_n: int = 10,
    title: str | None = "Three-mode spectrum (qubit-coupler-qubit)",
    **ham_kwargs,
) -> np.ndarray:
    """Diagonalize ``three_mode_hamiltonian`` and plot the lowest ``n_show`` energies."""
    H = three_mode_hamiltonian(**ham_kwargs)
    return plot_energy_levels(
        lambda: H,
        n_show=n_show,
        outfile=outfile,
        title=title,
        energy_unit="GHz",
        annotate_n=annotate_n,
    )


def plot_three_mode_energy_levels_vs_flux(
    flux_values: np.ndarray,
    *,
    wc0: float,
    A: float,
    outfile: str = "three_mode_energy_levels_vs_flux.pdf",
    n_show: int = 24,
    track_by_overlap: bool = True,
    subtract_ground: bool = False,
    title: str | None = None,
    **ham_kwargs,
) -> np.ndarray:
    """Plot the three-mode spectrum vs flux with ``w_c = wc0 + A cos(2π phi)``."""
    flux_values = np.asarray(flux_values, dtype=float)

    def hamiltonian_at_flux(phi: np.ndarray | float) -> np.ndarray:
        phi_arr = np.asarray(phi, dtype=float)

        if phi_arr.ndim == 0:
            wc = float(coupler_frequency(wc0, A, phi_arr))
            return three_mode_hamiltonian_from_kwargs(
                ham_kwargs,
                w_c=wc,
            )

        return three_mode_hamiltonian_stack_vs_flux(
            phi_arr,
            wc0=wc0,
            A=A,
            ham_kwargs=ham_kwargs,
        )

    if title is None:
        title = "Three-mode spectrum vs flux (coupler modulation)"

    return plot_energy_levels_vs_flux(
        flux_values,
        hamiltonian_at_flux,
        n_show=n_show,
        track_by_overlap=track_by_overlap,
        subtract_ground=subtract_ground,
        outfile=outfile,
        title=title,
        energy_unit="GHz",
    )
