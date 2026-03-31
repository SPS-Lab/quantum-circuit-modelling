import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

# Repo root (parent of model2/) so `toolkit` resolves when run from model2/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from toolkit.helpers import destroy
from toolkit.plotting import plot_energy_levels, plot_energy_levels_vs_flux

def three_mode_hamiltonian(w_1, w_c, w_2, alpha_1, alpha_c, alpha_2, g_1c, g_2c, nlevels_qubit, nlevels_coupler):
    """
    Constructs the three-mode Hamiltonian:

        H = sum_{j in {1,c,2}} [w_j a_j† a_j + (alpha_j/2) a_j† a_j† a_j a_j]
            + g_1c (a_1† a_c + a_c† a_1)
            + g_2c (a_2† a_c + a_c† a_2)

    Parameters
    ----------
    w_1, w_c, w_2 : float
        Mode frequencies.
    alpha_1, alpha_c, alpha_2 : float
        Mode anharmonicities.
    g_1c, g_2c : float
        Coupling strengths between modes.
    nlevels_qubit, nlevels_coupler : int
        Number of levels per mode.
    Returns
    -------
    H : ndarray
        Hamiltonian matrix, shape
        ``(nlevels_qubit * nlevels_coupler * nlevels_qubit,)`` squared.
    """
    from numpy import kron, eye

    # Single-mode operators (in number basis): subsystems are qubit_1, coupler, qubit_2
    a = [destroy(nlevels_qubit),
         destroy(nlevels_coupler),
         destroy(nlevels_qubit)]
    adag = [op.conj().T for op in a]
    n = [adag[j] @ a[j] for j in range(3)]

    id_q = eye(nlevels_qubit, dtype=complex)
    id_c = eye(nlevels_coupler, dtype=complex)

    # Tensor product operators
    def kron3(o1, o2, o3):
        return kron(kron(o1, o2), o3)

    # Annihilation/creation/number operators for each mode on the full Hilbert space
    a1 = kron3(a[0], id_c, id_q)
    ac = kron3(id_q, a[1], id_q)
    a2 = kron3(id_q, id_c, a[2])
    adag1 = kron3(adag[0], id_c, id_q)
    adagc = kron3(id_q, adag[1], id_q)
    adag2 = kron3(id_q, id_c, adag[2])

    n1 = kron3(n[0], id_c, id_q)
    nc = kron3(id_q, n[1], id_q)
    n2 = kron3(id_q, id_c, n[2])

    # Hamiltonian terms
    H = (
        w_1 * n1
        + w_c * nc
        + w_2 * n2
        + (alpha_1 / 2) * (adag1 @ adag1 @ a1 @ a1)
        + (alpha_c / 2) * (adagc @ adagc @ ac @ ac)
        + (alpha_2 / 2) * (adag2 @ adag2 @ a2 @ a2)
        + g_1c * (adag1 @ ac + adagc @ a1)
        + g_2c * (adag2 @ ac + adagc @ a2)
    )

    return H


def plot_three_mode_energy_levels(
    outfile: str = "three_mode_energy_levels.pdf",
    n_show: int = 48,
    annotate_n: int = 10,
    title: str | None = "Three-mode spectrum (qubit-coupler-qubit)",
    **ham_kwargs,
) -> np.ndarray:
    """Diagonalize :func:`three_mode_hamiltonian` and plot the lowest ``n_show`` energies."""
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
    """Spectrum vs flux with coupler ``w_c = wc0 + A cos(2π φ)``, same as :func:`propagate_piecewise`.

    Uses overlap tracking in :func:`toolkit.plotting.plot_energy_levels_vs_flux` so levels stay
    visually continuous through avoided crossings.

    ``ham_kwargs`` are passed to :func:`three_mode_hamiltonian` except ``w_c``, which is set from
    ``wc0``, ``A``, and ``φ`` (flux in units of Φ/Φ₀).
    """
    flux_values = np.asarray(flux_values, dtype=float)

    def hamiltonian_at_flux(phi: np.ndarray | float) -> np.ndarray:
        phi_arr = np.asarray(phi, dtype=float)
        if phi_arr.ndim == 0:
            wc = float(wc0 + A * np.cos(2 * np.pi * phi_arr))
            return three_mode_hamiltonian(
                ham_kwargs["w_1"],
                wc,
                ham_kwargs["w_2"],
                ham_kwargs["alpha_1"],
                ham_kwargs["alpha_c"],
                ham_kwargs["alpha_2"],
                ham_kwargs["g_1c"],
                ham_kwargs["g_2c"],
                ham_kwargs["nlevels_qubit"],
                ham_kwargs["nlevels_coupler"],
            )
        phi_arr = phi_arr.ravel()
        wc_arr = wc0 + A * np.cos(2 * np.pi * phi_arr)
        mats = [
            three_mode_hamiltonian(
                ham_kwargs["w_1"],
                float(wc_arr[i]),
                ham_kwargs["w_2"],
                ham_kwargs["alpha_1"],
                ham_kwargs["alpha_c"],
                ham_kwargs["alpha_2"],
                ham_kwargs["g_1c"],
                ham_kwargs["g_2c"],
                ham_kwargs["nlevels_qubit"],
                ham_kwargs["nlevels_coupler"],
            )
            for i in range(phi_arr.shape[0])
        ]
        return np.stack(mats, axis=0)

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


def propagate_piecewise(psi0, tlist, flux_values, params):
    psi = psi0.copy()
    states = [psi.copy()]
    for k in range(len(tlist) - 1):
        dt = tlist[k+1] - tlist[k]
        wc = params["wc0"] + params["A"] * np.cos(2*np.pi*flux_values[k])
        H = three_mode_hamiltonian(
            params["w1"], wc, params["w2"],
            params["a1"], params["ac"], params["a2"],
            params["g1c"], params["g2c"],
            nlevels_qubit=params["n1"],
            nlevels_coupler=params["nc"],
        )
        U = expm(-1j * H * dt)
        psi = U @ psi
        states.append(psi.copy())
    return np.array(states)


if __name__ == "__main__":
    _dir = Path(__file__).resolve().parent
    _common = dict(
        w_1=5.0,
        w_2=5.0,
        alpha_1=-0.2,
        alpha_c=-0.15,
        alpha_2=-0.2,
        g_1c=0.08,
        g_2c=0.08,
        nlevels_qubit=4,
        nlevels_coupler=5,
    )
    plot_three_mode_energy_levels(
        outfile=str(_dir / "three_mode_energy_levels.pdf"),
        w_c=5.2,
        **_common,
    )
    flux = np.linspace(0.0, 1.0, 80)
    plot_three_mode_energy_levels_vs_flux(
        flux,
        wc0=5.2,
        A=0.25,
        outfile=str(_dir / "three_mode_energy_levels_vs_flux.pdf"),
        n_show=16,
        **_common,
    )