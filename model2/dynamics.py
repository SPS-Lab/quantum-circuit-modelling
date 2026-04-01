"""Time-propagation routines for the three-mode model."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from model2.core import coupler_frequency, three_mode_hamiltonian


def propagate_piecewise(
    psi0: np.ndarray,
    tlist: np.ndarray,
    flux_values: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Piecewise-constant time evolution using per-step flux values."""
    psi = psi0.copy()
    states = [psi.copy()]
    for k in range(len(tlist) - 1):
        dt = tlist[k + 1] - tlist[k]
        wc = coupler_frequency(params["wc0"], params["A"], flux_values[k])
        H = three_mode_hamiltonian(
            params["w1"],
            float(wc),
            params["w2"],
            params["a1"],
            params["ac"],
            params["a2"],
            params["g1c"],
            params["g2c"],
            nlevels_qubit=params["n1"],
            nlevels_coupler=params["nc"],
        )
        U = expm(-1j * H * dt)
        psi = U @ psi
        states.append(psi.copy())
    return np.array(states)
