from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from scipy.linalg import expm

from runtime_utils import format_elapsed_compact, log_progress


def propagate_piecewise_constant(
    hamiltonians_ghz: np.ndarray,
    times_ns: np.ndarray,
    initial_states: np.ndarray,
    *,
    output_indices: np.ndarray | None = None,
    progress_label: str | None = None,
    progress_interval_s: float = 30.0,
) -> np.ndarray:
    """Propagate statevector columns with dense matrix exponentials.

    ``hamiltonians_ghz`` has shape ``(n_time, dimension, dimension)`` and
    ``times_ns`` is a nonempty, strictly increasing 1D grid. Hamiltonian ``k``
    acts on ``[times_ns[k], times_ns[k + 1])``; the final Hamiltonian is unused.
    Frequencies are in GHz, so each step is ``exp(-1j * 2*pi * H * dt_ns)``.

    ``initial_states`` has shape ``(dimension, n_states)``. The returned complex
    amplitudes have shape ``(n_time, n_outputs, n_states)``, including the
    initial state. ``output_indices`` selects basis rows to retain (in supplied
    order); omit it to retain all rows. Full states are always propagated.
    Inputs are not modified. No model-specific observables are computed.
    """
    hamiltonians = np.asarray(hamiltonians_ghz, dtype=complex)
    times = np.asarray(times_ns, dtype=float)
    states = np.asarray(initial_states, dtype=complex)
    if hamiltonians.ndim != 3 or hamiltonians.shape[1] != hamiltonians.shape[2]:
        raise ValueError(
            "hamiltonians_ghz must have shape (n_time, dimension, dimension)"
        )
    n_time, dimension, _ = hamiltonians.shape
    if (
        times.ndim != 1
        or times.size == 0
        or times.size != n_time
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "times_ns must be a finite, nonempty, strictly increasing grid matching n_time"
        )
    if states.ndim != 2 or states.shape[0] != dimension or states.shape[1] == 0:
        raise ValueError(
            "initial_states must have shape (dimension, n_states) with n_states > 0"
        )
    if output_indices is None:
        rows = np.arange(dimension)
    else:
        rows = np.asarray(output_indices)
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("output_indices must be a 1D integer array")
        if np.any(rows < 0) or np.any(rows >= dimension):
            raise ValueError("output_indices are outside the Hamiltonian basis")

    amplitudes = np.empty((n_time, rows.size, states.shape[1]), dtype=complex)
    amplitudes[0] = states[rows, :]
    interval = max(float(progress_interval_s), 1.0)
    started = time.perf_counter()
    last_progress = started
    for k in range(n_time - 1):
        dt = float(times[k + 1] - times[k])
        unitary = expm((-1.0j * 2.0 * np.pi * dt) * hamiltonians[k])
        states = unitary @ states
        amplitudes[k + 1] = states[rows, :]
        now = time.perf_counter()
        if progress_label is not None and (now - last_progress) >= interval:
            completed = k + 1
            total = n_time - 1
            percent = 100.0 * completed / total
            elapsed = format_elapsed_compact(now - started)
            log_progress(
                f"{progress_label} progress: step {completed}/{total} ({percent:.1f}%) after {elapsed}"
            )
            last_progress = now
    return amplitudes


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
