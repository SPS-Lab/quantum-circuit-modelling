"""Eigenvalue / eigenvector tracking along parameter sweeps (max-overlap matching)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def overlap_row_to_col_assignment(overlap: np.ndarray) -> np.ndarray:
    """Return column indices per row maximizing total overlap (Hungarian on ``-overlap``).

    Parameters
    ----------
    overlap
        Score matrix of shape ``(n_rows, n_cols)`` with ``n_cols >= n_rows``.

    Returns
    -------
    row_to_col : ndarray, shape ``(n_rows,)``, int
        ``row_to_col[r]`` gives the assigned column for row ``r``.
    """
    overlap = np.asarray(overlap, dtype=float)
    if overlap.ndim != 2:
        raise ValueError(f"overlap must be 2D, got shape {overlap.shape}")
    n_rows, n_cols = overlap.shape
    if n_rows > n_cols:
        raise ValueError(
            f"need at least as many columns as rows for assignment: {n_rows}>{n_cols}"
        )

    row_ind, col_ind = linear_sum_assignment(-overlap)
    row_to_col = np.empty(n_rows, dtype=int)
    for k in range(len(row_ind)):
        row_to_col[int(row_ind[k])] = int(col_ind[k])
    return row_to_col


def reorder_by_overlap(
    prev_vecs: np.ndarray,
    new_vecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute columns of ``new_vecs`` to best continue ``prev_vecs`` (Hungarian on |⟨·|·⟩|²).

    Parameters
    ----------
    prev_vecs
        Previous orthonormal columns, shape ``(d, m)``, where typically ``m`` is the number of eigenvectors being tracked.
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
    col_indices = overlap_row_to_col_assignment(overlap)
    matched = new_vecs[:, col_indices]
    return matched, col_indices


def _validate_projector_blocks(
    projector_blocks: tuple[tuple[int, ...], ...] | None,
    n_track: int,
) -> tuple[tuple[int, ...], ...]:
    if projector_blocks is None:
        return tuple()
    seen: set[int] = set()
    normalized: list[tuple[int, ...]] = []
    for block in projector_blocks:
        rows = tuple(int(r) for r in block)
        if len(rows) < 2:
            continue
        for r in rows:
            if r < 0 or r >= n_track:
                raise ValueError(f"projector block row {r} out of bounds for n_track={n_track}")
            if r in seen:
                raise ValueError(f"projector blocks must be disjoint; repeated row {r}")
            seen.add(r)
        normalized.append(rows)
    return tuple(normalized)


def _assignment_with_projector_blocks(
    overlap: np.ndarray,
    *,
    projector_blocks: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    overlap = np.asarray(overlap, dtype=float)
    n_rows, n_cols = overlap.shape
    row_to_col = -np.ones(n_rows, dtype=int)
    used_cols = np.zeros(n_cols, dtype=bool)

    for block in projector_blocks:
        block_rows = np.asarray(block, dtype=int)
        block_size = int(block_rows.size)
        available_cols = np.flatnonzero(~used_cols)
        if available_cols.size < block_size:
            raise ValueError("Not enough columns left for projector-block assignment")

        block_overlap = overlap[np.ix_(block_rows, available_cols)]
        projector_score = np.sum(block_overlap, axis=0)
        top_local = np.argpartition(projector_score, -block_size)[-block_size:]
        chosen_cols = available_cols[top_local]

        within_overlap = overlap[np.ix_(block_rows, chosen_cols)]
        within_map = overlap_row_to_col_assignment(within_overlap)
        for i, row in enumerate(block_rows):
            col = int(chosen_cols[int(within_map[i])])
            row_to_col[int(row)] = col
            used_cols[col] = True

    remaining_rows = np.flatnonzero(row_to_col < 0)
    if remaining_rows.size > 0:
        remaining_cols = np.flatnonzero(~used_cols)
        rem_overlap = overlap[np.ix_(remaining_rows, remaining_cols)]
        rem_map = overlap_row_to_col_assignment(rem_overlap)
        for i, row in enumerate(remaining_rows):
            col = int(remaining_cols[int(rem_map[i])])
            row_to_col[int(row)] = col
            used_cols[col] = True

    if np.any(row_to_col < 0):
        raise RuntimeError("Failed to assign all tracked rows")
    return row_to_col


def track_energy_levels_stack(
    H_stack: np.ndarray,
    n_track: int,
    *,
    projector_blocks: tuple[tuple[int, ...], ...] | None = None,
) -> np.ndarray:
    """Lowest ``n_track`` energies at each step, matched for continuity along axis 0.

    Parameters
    ----------
    H_stack
        ``(n_param, d, d)`` Hermitian matrices (e.g. flux slices).
    n_track
        Number of levels to follow (≤ ``d``).

    projector_blocks
        Optional row-index blocks (within ``0..n_track-1``) tracked as transported
        subspaces rather than strict one-by-one eigenstate labels.

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
    projector_blocks = _validate_projector_blocks(projector_blocks, n_track)

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
        overlap = np.abs(evecs_prev.conj().T @ v) ** 2
        if projector_blocks:
            col_idx = _assignment_with_projector_blocks(overlap, projector_blocks=projector_blocks)
        else:
            col_idx = overlap_row_to_col_assignment(overlap)

        matched = v[:, col_idx]
        evals_step = np.asarray(w[col_idx], dtype=float)

        # Within each projector block, parallel-transport the basis and report
        # block-diagonal expectation energies in that smooth transported basis.
        for block in projector_blocks:
            block_rows = np.asarray(block, dtype=int)
            prev_block = evecs_prev[:, block_rows]
            new_block = matched[:, block_rows]
            ov_block = prev_block.conj().T @ new_block
            u, _, vh = np.linalg.svd(ov_block, full_matrices=False)
            q = vh.conj().T @ u.conj().T

            matched[:, block_rows] = new_block @ q

            block_evals = evals_step[block_rows]
            h_block = q.conj().T @ np.diag(block_evals) @ q
            evals_step[block_rows] = np.real(np.diag(h_block))

        evecs_prev = matched
        evals_out[k] = evals_step

    return evals_out
