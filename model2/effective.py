"""Shared dressed-subspace extraction utilities for effective-model analysis."""

from __future__ import annotations

import numpy as np

from toolkit.spectrum import overlap_row_to_col_assignment

from model2.core import computational_state_indices


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


def build_dressed_effective_stack(
    H_stack: np.ndarray,
    *,
    subspace_indices: np.ndarray,
    n_candidate_states: int = 16,
    selection_mode: str = "continuous",
) -> np.ndarray:
    """Return dressed effective subspace Hamiltonians from a full stack.

    Parameters
    ----------
    H_stack
        Full Hamiltonian stack with shape ``(n_param, d, d)``.
    subspace_indices
        Bare basis indices defining the tracked subspace (length ``m``).
    n_candidate_states
        Number of low-energy full eigenstates to consider when assigning dressed states.
    selection_mode
        ``"continuous"`` for overlap continuation vs previous step, or ``"bare"`` for
        independent matching to bare states at each parameter point.
    """
    H_stack = np.asarray(H_stack, dtype=complex)
    if H_stack.ndim != 3 or H_stack.shape[1] != H_stack.shape[2]:
        raise ValueError(f"H_stack must be (n, d, d), got {H_stack.shape}")

    subspace_indices = np.asarray(subspace_indices, dtype=int).ravel()
    n_flux, d, _ = H_stack.shape
    m = subspace_indices.size
    if m < 1:
        raise ValueError("subspace_indices must be non-empty")
    if np.any(subspace_indices < 0) or np.any(subspace_indices >= d):
        raise ValueError(
            f"subspace_indices out of bounds for d={d}: {subspace_indices}"
        )

    mode = str(selection_mode).strip().lower()
    if mode not in {"continuous", "bare"}:
        raise ValueError(
            "selection_mode must be one of {'continuous', 'bare'}, "
            f"got {selection_mode!r}"
        )

    n_cand = max(m, min(int(n_candidate_states), d))
    H_eff = np.empty((n_flux, m, m), dtype=complex)
    prev_selected_full: np.ndarray | None = None

    for k in range(n_flux):
        evals_full, evecs_full = np.linalg.eigh(H_stack[k])
        evecs_cand = evecs_full[:, :n_cand]

        if mode == "bare" or prev_selected_full is None:
            overlap = np.abs(evecs_cand[subspace_indices, :]) ** 2
        else:
            overlap = np.abs(prev_selected_full.conj().T @ evecs_cand) ** 2

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
    nlevels_qubit: int,
    nlevels_coupler: int,
    *,
    n_candidate_states: int = 16,
    selection_mode: str = "continuous",
) -> np.ndarray:
    """Return dressed effective computational ``4x4`` Hamiltonians from full stacks."""
    idx = computational_state_indices(
        int(nlevels_qubit),
        int(nlevels_coupler),
    )
    return build_dressed_effective_stack(
        H_stack,
        subspace_indices=idx,
        n_candidate_states=n_candidate_states,
        selection_mode=selection_mode,
    )


def extract_model1_parameters_from_4x4_stack(
    H_eff: np.ndarray,
    *,
    zeta_tol: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Extract ``w1, w2, J, zeta`` from a dressed effective computational ``4x4`` stack."""
    H_eff = np.asarray(H_eff, dtype=complex)
    if H_eff.ndim == 2:
        H_eff = H_eff[np.newaxis, ...]
    if H_eff.ndim != 3 or H_eff.shape[1:] != (4, 4):
        raise ValueError(f"H_eff must be (n,4,4) or (4,4), got {H_eff.shape}")

    d00 = np.real(H_eff[:, 0, 0])
    d01 = np.real(H_eff[:, 1, 1])
    d10 = np.real(H_eff[:, 2, 2])
    d11 = np.real(H_eff[:, 3, 3])

    zeta = d11 - d10 - d01 + d00
    zeta = np.where(np.abs(zeta) < float(zeta_tol), 0.0, zeta)
    w1 = d10 - d00 + 0.5 * zeta
    w2 = d01 - d00 + 0.5 * zeta
    j = 0.5 * np.real(H_eff[:, 1, 2])

    return {"w1": w1, "w2": w2, "J": j, "zeta": zeta}
