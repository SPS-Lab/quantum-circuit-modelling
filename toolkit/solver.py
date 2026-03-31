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


def eigensystem_single_mode(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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