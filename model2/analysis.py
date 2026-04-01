"""Energy-derived observables for the three-mode and model-1 Hamiltonians."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from model2.core import computational_state_indices


def dressed_computational_energies(
    H: np.ndarray,
    nlevels_qubit: int,
    nlevels_coupler: int,
    *,
    n_candidate_states: int = 16,
) -> np.ndarray:
    """Dressed energies ``(E_00, E_01, E_10, E_11)`` in the same units as ``H``."""
    H = np.asarray(H, dtype=complex)
    d = H.shape[0]
    idx = computational_state_indices(nlevels_qubit, nlevels_coupler)
    n_cand = max(4, min(int(n_candidate_states), d))

    evals, evecs = np.linalg.eigh(H)
    evecs_c = evecs[:, :n_cand]
    overlap = np.abs(evecs_c[idx, :]) ** 2
    row_ind, col_ind = linear_sum_assignment(-overlap)

    energies = np.empty(4, dtype=float)
    for k in range(4):
        energies[int(row_ind[k])] = float(evals[int(col_ind[k])])
    return energies


def exchange_splitting_bare_01_10(
    H: np.ndarray,
    nlevels_qubit: int,
    nlevels_coupler: int,
) -> float:
    """Return ``|lambda_+ - lambda_-|`` in the bare ``|01>,|10>`` subspace."""
    H = np.asarray(H, dtype=complex)
    idx = computational_state_indices(nlevels_qubit, nlevels_coupler)
    i01, i10 = int(idx[1]), int(idx[2])
    h11 = H[i01, i01].real
    h22 = H[i10, i10].real
    h12 = H[i01, i10]
    tr = h11 + h22
    det = h11 * h22 - h12 * np.conj(h12)
    disc = np.sqrt(max(0.0, 0.25 * tr**2 - det.real))
    return float(2.0 * disc)


def residual_zz_and_exchange(
    H: np.ndarray,
    nlevels_qubit: int,
    nlevels_coupler: int,
    **dress_kw,
) -> tuple[float, float]:
    """Return ``(zeta_zz, delta_ex)`` from dressed energies and bare ``|01>-|10|`` splitting."""
    E = dressed_computational_energies(
        H,
        nlevels_qubit,
        nlevels_coupler,
        **dress_kw,
    )
    zeta = float(E[3] - E[2] - E[1] + E[0])
    delta_ex = exchange_splitting_bare_01_10(H, nlevels_qubit, nlevels_coupler)
    return zeta, delta_ex


def model1_exchange_and_zz_from_eigenvalues(
    H_stack: np.ndarray,
    w1: np.ndarray | float,
    w2: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``|J|`` and ``zeta`` from model-1 eigenvalues."""
    H_stack = np.asarray(H_stack, dtype=complex)
    if H_stack.ndim == 2:
        H_stack = H_stack[np.newaxis, ...]
    if H_stack.ndim != 3 or H_stack.shape[1:] != (4, 4):
        raise ValueError(f"model1 H_stack must be (n,4,4) or (4,4), got {H_stack.shape}")

    w1 = np.asarray(w1, dtype=float).reshape(-1)
    w2 = np.asarray(w2, dtype=float).reshape(-1)
    n_flux = H_stack.shape[0]
    if w1.size == 1:
        w1 = np.full(n_flux, float(w1[0]))
    if w2.size == 1:
        w2 = np.full(n_flux, float(w2[0]))
    if w1.size != n_flux or w2.size != n_flux:
        raise ValueError(f"w1/w2 lengths must match n_flux={n_flux}, got {w1.size}/{w2.size}")

    evals = np.linalg.eigvalsh(H_stack)
    zeta = evals[:, 3] - evals[:, 2] - evals[:, 1] + evals[:, 0]
    delta = evals[:, 2] - evals[:, 1]
    detuning = w1 - w2
    rad = np.maximum(delta * delta - detuning * detuning, 0.0)
    j_abs = 0.25 * np.sqrt(rad)
    return j_abs.astype(float), zeta.astype(float)
