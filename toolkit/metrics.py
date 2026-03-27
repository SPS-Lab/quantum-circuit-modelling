"""
Single-mode Hamiltonian spectral metrics.

Diagonal ``H``: energies ``diag(H)`` in index order (Fock level ``k`` at row ``k``); no ``eigh``.
Nondiagonal ``H``: ``eigh``; ground state ``evals[0]``. See :func:`_eigensystem_single_mode`.

Outputs: SingleModeStaticMetrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _is_diagonal(H: np.ndarray, rtol: float = 1e-10, atol: float = 1e-10) -> bool:
    H = np.asarray(H)
    return bool(np.allclose(H, np.diag(np.diag(H)), rtol=rtol, atol=atol))


def eigensystem(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hermitian eigendecomposition with ascending eigenvalues.

    Returns
    -------
    evals : (n,) float
        Eigenvalues in increasing order (same units as ``H``, e.g. GHz with ħ=1).
    evecs : (n, n) complex
        Columns ``evecs[:, k]`` are orthonormal eigenvectors in the same basis as ``H``.
    """
    H = np.asarray(H)
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs


def _eigensystem_single_mode(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenpairs for single-mode metrics: ordered basis for diagonal ``H``, else ``eigh``.

    If ``H`` is diagonal (e.g. Duffing in the number basis), eigenvalues are read from
    ``np.diag(H)`` **in index order** so that index ``k`` labels the ``k``-th Fock level.
    ``np.linalg.eigh`` would sort by energy and scramble that labeling when the diagonal
    is not monotonic (e.g. truncated Duffing with ``alpha < 0``).

    If ``H`` has off-diagonal entries (e.g. CPB in the charge basis), use ``eigh`` with
    ascending eigenvalues (ground state = ``evals[0]``).
    """
    H = np.asarray(H)
    if _is_diagonal(H):
        evals = np.diag(H).real.astype(float)
        n = evals.size
        evecs = np.eye(n, dtype=complex)
        return evals, evecs
    return eigensystem(H)


def transition_frequencies(evals: np.ndarray) -> np.ndarray:
    """Transition angular frequencies omega_{mn} = (E_n - E_m) / hbar with hbar = 1.

    Returns ``omega`` with ``omega[m, n] = evals[n] - evals[m]``.
    """
    e = np.asarray(evals, dtype=float).reshape(-1)
    return e[np.newaxis, :] - e[:, np.newaxis]


@dataclass(frozen=True)
class SingleModeStaticMetrics:
    """Spectral data for one effective mode (Duffing / CPB truncation)."""

    energies: np.ndarray
    eigenvectors: np.ndarray
    omega_01: float
    omega_12: Optional[float]
    alpha: Optional[float]
    omega_matrix: np.ndarray
    overlap_bare_eigen: np.ndarray


def single_mode_static_metrics(H: np.ndarray) -> SingleModeStaticMetrics:
    """Eigendecompose ``H`` and extract static metrics for a single-mode model.

    **Diagonal ``H`` (Duffing in number basis).** Eigenvalues are taken as
    ``diag(H)[0], diag(H)[1], ...`` in **basis order**, not sorted by magnitude, so
    index ``k`` remains the ``k``-th excitation Fock level.

    **General ``H`` (e.g. CPB).** Uses ``eigh``; ``evals[0]`` is the physical ground
    state in the truncated Hilbert space.

    **Qubit frequency.** Lowest transition ``omega_01 = E_1 - E_0`` (ket0 to ket1) in the ordered spectrum above.

    **Transition frequencies.** ``omega_matrix[m, n] = E_n - E_m``.

    **Anharmonicity.** ``alpha = omega_12 - omega_01`` when at least three levels exist.

    **Localization / overlap.** ``overlap_bare_eigen[i, k] = |<i|psi_k>|^2``.

    Parameters
    ----------
    H : (n, n) array
        Hermitian Hamiltonian from ``duffing_single_mode`` or
        ``cooper_pair_box_hamiltonian``.
    """
    evals, evecs = _eigensystem_single_mode(H)
    n = evals.size
    omega_mn = transition_frequencies(evals)

    if n < 2:
        raise ValueError("Need at least two levels for omega_01.")

    omega_01 = float(evals[1] - evals[0])
    omega_12: Optional[float] = None
    alpha: Optional[float] = None
    if n >= 3:
        omega_12 = float(evals[2] - evals[1])
        alpha = float(omega_12 - omega_01)

    overlap = np.abs(evecs) ** 2

    return SingleModeStaticMetrics(
        energies=evals,
        eigenvectors=evecs,
        omega_01=omega_01,
        omega_12=omega_12,
        alpha=alpha,
        omega_matrix=omega_mn,
        overlap_bare_eigen=overlap,
    )
