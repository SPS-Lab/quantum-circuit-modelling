"""Energy-derived observables for the three-mode and model-1 Hamiltonians."""

from __future__ import annotations

import numpy as np
from toolkit.spectrum import overlap_row_to_col_assignment

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
    col_ind = overlap_row_to_col_assignment(overlap)
    energies = np.asarray(evals[col_ind], dtype=float)
    return energies


def exchange_and_zz_from_4x4_eigenvalues(
    H_stack: np.ndarray,
    w1: np.ndarray | float,
    w2: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract effective ``|J|`` and ``zeta`` from a ``4x4`` eigenvalue stack."""
    H_stack = np.asarray(H_stack, dtype=complex)
    if H_stack.ndim == 2:
        H_stack = H_stack[np.newaxis, ...]
    if H_stack.ndim != 3 or H_stack.shape[1:] != (4, 4):
        raise ValueError(f"H_stack must be (n,4,4) or (4,4), got {H_stack.shape}")

    w1_arr = np.asarray(w1, dtype=float).reshape(-1)
    w2_arr = np.asarray(w2, dtype=float).reshape(-1)
    n_flux = H_stack.shape[0]
    if w1_arr.size == 1:
        w1_arr = np.full(n_flux, float(w1_arr[0]))
    if w2_arr.size == 1:
        w2_arr = np.full(n_flux, float(w2_arr[0]))
    if w1_arr.size != n_flux or w2_arr.size != n_flux:
        raise ValueError(
            f"w1/w2 lengths must match n_flux={n_flux}, got {w1_arr.size}/{w2_arr.size}"
        )

    evals = np.linalg.eigvalsh(H_stack)
    zeta = evals[:, 3] - evals[:, 2] - evals[:, 1] + evals[:, 0]
    delta = evals[:, 2] - evals[:, 1]
    detuning = w1_arr - w2_arr
    rad = np.maximum(delta * delta - detuning * detuning, 0.0)
    j_abs = 0.25 * np.sqrt(rad)
    return j_abs.astype(float), zeta.astype(float)
