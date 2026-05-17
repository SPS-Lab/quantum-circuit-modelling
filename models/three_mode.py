"""Three-mode (qubit-coupler-qubit) Hamiltonian construction utilities."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from toolkit.helpers import destroy


class ThreeModeHamiltonianCommonKwargs(TypedDict):
    """Common fixed keyword arguments for three-mode Hamiltonians.

    These include all parameters except explicit coupler frequency ``w_c``.
    """

    w_0: float
    w_1: float
    alpha_0: float
    alpha_c: float
    alpha_1: float
    g_0c: float
    g_1c: float
    nlevels_qubit: int
    nlevels_coupler: int


class ThreeModeHamiltonianKwargs(ThreeModeHamiltonianCommonKwargs):
    """Keyword arguments required by ``three_mode_hamiltonian``."""

    w_c: float



def coupler_frequency(
    *,
    wc0: float,
    A: float,
    flux: np.ndarray | float
) -> np.ndarray:
    """Return coupler frequency ``w_c = wc0 + A cos(2pi flux)``."""
    return float(wc0) + float(A) * np.cos(2.0 * np.pi * flux)



def three_mode_hamiltonian(
    *,
    w_0: float,
    w_c: float,
    w_1: float,
    alpha_0: float,
    alpha_c: float,
    alpha_1: float,
    g_0c: float,
    g_1c: float,
    nlevels_qubit: int,
    nlevels_coupler: int,
) -> np.ndarray:
    """Construct the three-mode Hamiltonian (qubit-coupler-qubit)."""
    from numpy import eye, kron

    a_local = [
        destroy(int(nlevels_qubit)),
        destroy(int(nlevels_coupler)),
        destroy(int(nlevels_qubit)),
    ]
    adag_local = [op.conj().T for op in a_local]
    n_local = [adag_local[j] @ a_local[j] for j in range(3)]

    id_q = eye(int(nlevels_qubit), dtype=complex)
    id_c = eye(int(nlevels_coupler), dtype=complex)

    def kron3(o3: np.ndarray, o2: np.ndarray, o1: np.ndarray) -> np.ndarray:
        return kron(kron(o3, o2), o1)

    a1 = kron3(id_q, id_c, a_local[0])
    ac = kron3(id_q, a_local[1], id_q)
    a2 = kron3(a_local[2], id_c, id_q)
    adag1 = kron3(id_q, id_c, adag_local[0])
    adagc = kron3(id_q, adag_local[1], id_q)
    adag2 = kron3(adag_local[2], id_c, id_q)

    n1 = kron3(id_q, id_c, n_local[0])
    nc = kron3(id_q, n_local[1], id_q)
    n2 = kron3(n_local[2], id_c, id_q)

    return (
        float(w_0) * n1
        + float(w_c) * nc
        + float(w_1) * n2
        + (float(alpha_0) / 2.0) * (adag1 @ adag1 @ a1 @ a1)
        + (float(alpha_c) / 2.0) * (adagc @ adagc @ ac @ ac)
        + (float(alpha_1) / 2.0) * (adag2 @ adag2 @ a2 @ a2)
        + float(g_0c) * (adag1 @ ac + adagc @ a1)
        + float(g_1c) * (adag2 @ ac + adagc @ a2)
    )



def computational_state_indices(
    *,
    nlevels_qubit: int,
    nlevels_coupler: int
) -> np.ndarray:
    """Flat indices for computational states in ``|q1,0_c,q0>`` order with ``q0`` as LSB."""
    q1_significance = int(nlevels_coupler) * int(nlevels_qubit)
    return np.array([0, 1, q1_significance + 0, q1_significance + 1], dtype=int)


def canonical_state_order_qcq(
    *,
    nlevels_q0: int,
    nlevels_coupler: int,
    nlevels_q1: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat indices and labels in canonical ``|q1,c,q0>`` display order."""
    nq0 = int(nlevels_q0)
    nc = int(nlevels_coupler)
    nq1 = nq0 if nlevels_q1 is None else int(nlevels_q1)
    triples: list[tuple[int, int, int]] = []
    for q0 in range(nq0):
        for c in range(nc):
            for q1 in range(nq1):
                triples.append((q1, c, q0))

    triples_sorted = sorted(triples, key=lambda t: (t[0] + t[1] + t[2], t[0], t[1], t[2]))
    flat_idx = np.array([(q1 * nc + c) * nq0 + q0 for (q1, c, q0) in triples_sorted], dtype=int)
    labels = np.array([f"|{q1},{c},{q0}>" for (q1, c, q0) in triples_sorted], dtype=str)
    return flat_idx, labels



def three_mode_hamiltonian_from_kwargs(
    ham_kwargs: ThreeModeHamiltonianCommonKwargs,
    *,
    w_c: float,
) -> np.ndarray:
    """Build ``three_mode_hamiltonian`` from kwargs + coupler frequency."""
    return three_mode_hamiltonian(
        w_0=float(ham_kwargs["w_0"]),
        w_c=float(w_c),
        w_1=float(ham_kwargs["w_1"]),
        alpha_0=float(ham_kwargs["alpha_0"]),
        alpha_c=float(ham_kwargs["alpha_c"]),
        alpha_1=float(ham_kwargs["alpha_1"]),
        g_0c=float(ham_kwargs["g_0c"]),
        g_1c=float(ham_kwargs["g_1c"]),
        nlevels_qubit=int(ham_kwargs["nlevels_qubit"]),
        nlevels_coupler=int(ham_kwargs["nlevels_coupler"]),
    )



def three_mode_hamiltonian_stack_vs_flux(
    flux_values: np.ndarray,
    *,
    wc0: float,
    A: float,
    ham_kwargs: ThreeModeHamiltonianCommonKwargs,
) -> np.ndarray:
    """Return ``(n_flux, d, d)`` three-mode Hamiltonian stack for a flux sweep."""
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    wc_arr = np.asarray(coupler_frequency(wc0=wc0, A=A, flux=flux_values), dtype=float).ravel()
    mats = [
        three_mode_hamiltonian_from_kwargs(ham_kwargs, w_c=float(wc_arr[k]))
        for k in range(flux_values.shape[0])
    ]
    return np.stack(mats, axis=0)



def computational_subspace_block(
    H: np.ndarray,
    *,
    nlevels_qubit: int,
    nlevels_coupler: int,
    hermitianize: bool = False,
) -> np.ndarray:
    """Extract computational ``4x4`` block from one Hamiltonian or a stack."""
    H = np.asarray(H, dtype=complex)
    idx = computational_state_indices(
        nlevels_qubit=int(nlevels_qubit),
        nlevels_coupler=int(nlevels_coupler)
    )

    if H.ndim == 2:
        block = H[np.ix_(idx, idx)]
        if hermitianize:
            block = 0.5 * (block + block.conj().T)
        return block

    if H.ndim == 3:
        block = H[:, idx][:, :, idx]
        if hermitianize:
            block = 0.5 * (block + np.conjugate(np.swapaxes(block, 1, 2)))
        return block

    raise ValueError(f"H must be (d,d) or (n,d,d), got {H.shape}")
