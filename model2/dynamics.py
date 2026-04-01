"""Time-propagation routines for the three-mode model."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from model2.core import coupler_frequency, three_mode_hamiltonian_from_kwargs
from model2.hamiltonian_types import ThreeModeHamiltonianCommonKwargs


def propagate_piecewise(
    psi0: np.ndarray,
    tlist: np.ndarray,
    flux_values: np.ndarray,
    *,
    wc0: float,
    A: float,
    ham_kwargs: ThreeModeHamiltonianCommonKwargs,
) -> np.ndarray:
    """Piecewise-constant time evolution using per-step flux values."""
    wc_arr = np.asarray(
        coupler_frequency(wc0, A, flux_values[:-1]),
        dtype=float,
    ).ravel()

    psi = psi0.copy()
    states = [psi.copy()]
    for k in range(len(tlist) - 1):
        dt = tlist[k + 1] - tlist[k]
        H = three_mode_hamiltonian_from_kwargs(
            ham_kwargs,
            w_c=float(wc_arr[k]),
        )
        U = expm(-1j * H * dt)
        psi = U @ psi
        states.append(psi.copy())
    return np.array(states)
