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

from toolkit.solver import eigensystem_single_mode

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
    evals, evecs = eigensystem_single_mode(H)
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
