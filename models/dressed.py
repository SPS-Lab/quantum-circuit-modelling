"""Dressed-subspace extraction and effective-parameter utilities."""

from __future__ import annotations

import numpy as np

from toolkit.spectrum import overlap_row_to_col_assignment

from models.three_mode import computational_state_indices



def lowdin_orthonormalize_columns(
    vectors: np.ndarray,
    *,
    min_eigval: float = 1e-15,
) -> np.ndarray:
    """Return Lowdin-orthonormalized columns for a column-stack ``vectors``."""
    vectors = np.asarray(vectors, dtype=complex)
    gram = vectors.conj().T @ vectors
    gram_evals, gram_vecs = np.linalg.eigh(gram)
    gram_evals = np.clip(gram_evals, float(min_eigval), None)
    gram_inv_sqrt = gram_vecs @ np.diag(1.0 / np.sqrt(gram_evals)) @ gram_vecs.conj().T
    return vectors @ gram_inv_sqrt


def _validate_projector_blocks(
    projector_blocks: tuple[tuple[int, ...], ...] | None,
    *,
    m: int,
) -> tuple[tuple[int, ...], ...]:
    if projector_blocks is None:
        return tuple()

    normalized: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for block in projector_blocks:
        rows = tuple(int(r) for r in block)
        if len(rows) < 2:
            continue
        for r in rows:
            if r < 0 or r >= m:
                raise ValueError(f"projector block row {r} out of bounds for m={m}")
            if r in seen:
                raise ValueError(f"projector blocks must be disjoint; repeated row {r}")
            seen.add(r)
        normalized.append(rows)
    return tuple(normalized)


def _assignment_with_projector_blocks(
    overlap: np.ndarray,
    *,
    prev_selected_full: np.ndarray,
    evecs_cand: np.ndarray,
    projector_blocks: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    """Assign rows to candidate columns using subspace-first tracking for blocks."""
    overlap = np.asarray(overlap, dtype=float)
    m, n_cand = overlap.shape
    row_to_col = -np.ones(m, dtype=int)
    used_cols = np.zeros(n_cand, dtype=bool)

    for block in projector_blocks:
        block_rows = np.asarray(block, dtype=int)
        block_size = int(block_rows.size)

        available_cols = np.flatnonzero(~used_cols)
        if available_cols.size < block_size:
            raise ValueError("Not enough candidate columns left for projector-block assignment")

        prev_block = np.asarray(prev_selected_full[:, block_rows], dtype=complex)
        cand_available = np.asarray(evecs_cand[:, available_cols], dtype=complex)
        overlap_block = np.abs(prev_block.conj().T @ cand_available) ** 2
        projector_score = np.sum(overlap_block, axis=0)

        # Pick subset maximizing overlap with previous tracked subspace projector.
        top_local = np.argpartition(projector_score, -block_size)[-block_size:]
        chosen_cols = available_cols[top_local]

        # Then preserve per-row continuity within the chosen subspace.
        within_overlap = np.abs(prev_block.conj().T @ evecs_cand[:, chosen_cols]) ** 2
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


def build_dressed_effective_stack(
    H_stack: np.ndarray,
    *,
    subspace_indices: np.ndarray,
    selection_mode: str,
    n_candidate_states: int,
    projector_blocks: tuple[tuple[int, ...], ...] | None = None,
) -> np.ndarray:
    """Return dressed effective subspace Hamiltonians from a full stack."""
    H_stack = np.asarray(H_stack, dtype=complex)
    if H_stack.ndim != 3 or H_stack.shape[1] != H_stack.shape[2]:
        raise ValueError(f"H_stack must be (n, d, d), got {H_stack.shape}")

    subspace_indices = np.asarray(subspace_indices, dtype=int).ravel()
    n_flux, d, _ = H_stack.shape
    m = subspace_indices.size
    if m < 1:
        raise ValueError("subspace_indices must be non-empty")
    if np.any(subspace_indices < 0) or np.any(subspace_indices >= d):
        raise ValueError(f"subspace_indices out of bounds for d={d}: {subspace_indices}")

    mode = str(selection_mode).strip().lower()
    if mode not in {"continuous", "bare"}:
        raise ValueError(
            "selection_mode must be one of {'continuous', 'bare'}, "
            f"got {selection_mode!r}"
        )

    n_cand = max(m, min(int(n_candidate_states), d))
    projector_blocks = _validate_projector_blocks(projector_blocks, m=m)
    H_eff = np.empty((n_flux, m, m), dtype=complex)
    prev_selected_full: np.ndarray | None = None

    for k in range(n_flux):
        evals_full, evecs_full = np.linalg.eigh(H_stack[k])
        evecs_cand = evecs_full[:, :n_cand]

        if mode == "bare" or prev_selected_full is None:
            overlap = np.abs(evecs_cand[subspace_indices, :]) ** 2
            col_ind = overlap_row_to_col_assignment(overlap)
        else:
            overlap = np.abs(prev_selected_full.conj().T @ evecs_cand) ** 2
            if projector_blocks:
                col_ind = _assignment_with_projector_blocks(
                    overlap,
                    prev_selected_full=prev_selected_full,
                    evecs_cand=evecs_cand,
                    projector_blocks=projector_blocks,
                )
            else:
                col_ind = overlap_row_to_col_assignment(overlap)
        evals_sel = np.asarray(evals_full[col_ind], dtype=float)
        selected_full = np.asarray(evecs_cand[:, col_ind], dtype=complex)
        if mode == "continuous":
            prev_selected_full = selected_full

        sub_components = np.asarray(selected_full[subspace_indices, :], dtype=complex)
        dressed_basis = lowdin_orthonormalize_columns(sub_components)
        heff = dressed_basis @ np.diag(evals_sel) @ dressed_basis.conj().T
        H_eff[k] = 0.5 * (heff + heff.conj().T)

    return H_eff



def build_dressed_effective_computational_stack(
    H_stack: np.ndarray,
    *,
    nlevels_qubit: int,
    nlevels_coupler: int,
    selection_mode: str,
    n_candidate_states: int,
    projector_track_single_excitation: bool = True,
) -> np.ndarray:
    """Return dressed effective computational ``4x4`` Hamiltonians from full stacks."""
    idx = computational_state_indices(nlevels_qubit=int(nlevels_qubit), nlevels_coupler=int(nlevels_coupler))
    projector_blocks: tuple[tuple[int, ...], ...] | None = ((1, 2),) if projector_track_single_excitation else None
    return build_dressed_effective_stack(
        H_stack,
        subspace_indices=idx,
        n_candidate_states=n_candidate_states,
        selection_mode=selection_mode,
        projector_blocks=projector_blocks,
    )



def extract_effective_model_parameters_from_4x4_stack(
    H_eff: np.ndarray,
    *,
    zeta_tol: float = 1e-12,
    gauge_fix_exchange: bool = True,
) -> dict[str, np.ndarray]:
    """Extract ``w0, w1, J, zeta`` from dressed effective computational ``4x4`` stack."""
    H_eff = np.asarray(H_eff, dtype=complex)
    if H_eff.ndim == 2:
        H_eff = H_eff[np.newaxis, ...]
    if H_eff.ndim != 3 or H_eff.shape[1:] != (4, 4):
        raise ValueError(f"H_eff must be (n,4,4) or (4,4), got {H_eff.shape}")

    H_param = np.array(H_eff, copy=True)
    if gauge_fix_exchange:
        # Local phase gauge on |10> so H[1,2] is real-positive at each flux point.
        # This removes basis-phase sign flips in extracted J while preserving
        # gauge-invariant spectral quantities.
        for k in range(H_param.shape[0]):
            h12 = H_param[k, 1, 2]
            mag = abs(h12)
            if mag < 1e-15:
                continue
            phase = np.conjugate(h12) / mag
            d = np.ones(4, dtype=complex)
            d[2] = phase
            H_param[k] = (np.conjugate(d)[:, np.newaxis] * H_param[k]) * d[np.newaxis, :]

    d00 = np.real(H_param[:, 0, 0])
    d01 = np.real(H_param[:, 1, 1])
    d10 = np.real(H_param[:, 2, 2])
    d11 = np.real(H_param[:, 3, 3])

    zeta = d11 - d10 - d01 + d00
    zeta = np.where(np.abs(zeta) < float(zeta_tol), 0.0, zeta)
    # Computational ordering follows |q1,q0>, so q0 excitations are at index 1.
    w0 = d01 - d00 + 0.5 * zeta
    w1 = d10 - d00 + 0.5 * zeta
    j = 0.5 * np.real(H_param[:, 1, 2])

    return {"w0": w0, "w1": w1, "J": j, "zeta": zeta}
