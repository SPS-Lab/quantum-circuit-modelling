import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

# Repo root (parent of model2/) so `toolkit` resolves when run from model2/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from toolkit.helpers import destroy
from toolkit.plotting import plot_energy_levels

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


def reorder_by_overlap(prev_vecs, new_vecs): # Use this in model1 also?
    overlap = np.abs(prev_vecs.conj().T @ new_vecs)**2
    row_ind, col_ind = linear_sum_assignment(-overlap)
    return new_vecs[:, col_ind], col_ind


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
    _out = Path(__file__).resolve().parent / "three_mode_energy_levels.pdf"
    plot_three_mode_energy_levels(
        outfile=str(_out),
        w_1=5.0,
        w_c=5.2,
        w_2=5.0,
        alpha_1=-0.2,
        alpha_c=-0.15,
        alpha_2=-0.2,
        g_1c=0.08,
        g_2c=0.08,
        nlevels_qubit=4,
        nlevels_coupler=5,
    )