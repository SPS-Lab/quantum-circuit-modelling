"""Eigenvalue / eigenvector tracking along parameter sweeps (max-overlap matching)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def reorder_by_overlap(
    prev_vecs: np.ndarray,
    new_vecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute columns of ``new_vecs`` to best continue ``prev_vecs`` (Hungarian on |⟨·|·⟩|²).

    Parameters
    ----------
    prev_vecs
        Previous orthonormal columns, shape ``(d, m)``.
    new_vecs
        New orthonormal columns (typically all eigenvectors), shape ``(d, n)`` with ``n ≥ m``.

    Returns
    -------
    matched : ndarray, shape ``(d, m)``
        Column ``i`` is the new eigenvector assigned to follow ``prev_vecs[:, i]``.
    col_indices : ndarray, shape ``(m,)``, int
        ``matched[:, i] == new_vecs[:, col_indices[i]]`` and eigenvalues align as ``w_new[col_indices[i]]``.
    """
    prev_vecs = np.asarray(prev_vecs, dtype=complex)
    new_vecs = np.asarray(new_vecs, dtype=complex)
    d, m = prev_vecs.shape
    d2, n = new_vecs.shape
    if d != d2:
        raise ValueError(f"Hilbert space dim mismatch: prev {d}, new {d2}")
    if m > n:
        raise ValueError(f"need at least as many new vectors as prev columns: m={m}, n={n}")

    overlap = np.abs(prev_vecs.conj().T @ new_vecs) ** 2
    row_ind, col_ind = linear_sum_assignment(-overlap)

    matched = np.zeros((d, m), dtype=complex)
    col_indices = np.empty(m, dtype=int)
    for k in range(len(row_ind)):
        r, c = int(row_ind[k]), int(col_ind[k])
        matched[:, r] = new_vecs[:, c]
        col_indices[r] = c
    return matched, col_indices


def track_energy_levels_stack(H_stack: np.ndarray, n_track: int) -> np.ndarray:
    """Lowest ``n_track`` energies at each step, matched for continuity along axis 0.

    Parameters
    ----------
    H_stack
        ``(n_param, d, d)`` Hermitian matrices (e.g. flux slices).
    n_track
        Number of levels to follow (≤ ``d``).

    Returns
    -------
    evals : ndarray, shape ``(n_param, n_track)``
        ``evals[k, i]`` is the energy of the adiabatically continued ``i``-th level at step ``k``.
    """
    H_stack = np.asarray(H_stack, dtype=complex)
    if H_stack.ndim != 3 or H_stack.shape[1] != H_stack.shape[2]:
        raise ValueError(f"H_stack must be (n, d, d), got {H_stack.shape}")
    n_flux, d, _ = H_stack.shape
    n_track = int(n_track)
    if n_track < 1 or n_track > d:
        raise ValueError(f"n_track must be in [1, d], d={d}, got {n_track}")

    # If every slice is the same (no parameter variation), overlap tracking is ill-defined:
    # `eigh` can rotate degenerate subspaces and the Hungarian step permutes labels vs index.
    if n_flux > 1 and np.allclose(H_stack, H_stack[0], rtol=0.0, atol=1e-13):
        w = np.linalg.eigvalsh(H_stack[0])
        return np.broadcast_to(w[:n_track], (n_flux, n_track)).copy()

    w, v = np.linalg.eigh(H_stack[0])
    evecs_prev = v[:, :n_track].copy()
    evals_out = np.zeros((n_flux, n_track), dtype=float)
    evals_out[0] = w[:n_track]

    for k in range(1, n_flux):
        w, v = np.linalg.eigh(H_stack[k])
        evecs_prev, col_idx = reorder_by_overlap(evecs_prev, v)
        evals_out[k] = w[col_idx]

    return evals_out
