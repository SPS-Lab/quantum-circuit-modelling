"""
Harmonic oscillator Hamiltonian
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Repo root (parent of this folder) so `toolkit.plotting` resolves when run from this folder
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from toolkit.plotting import plot_evolve_state as plot_evolve_state_under_hamiltonian
from toolkit.helpers import destroy

"""
w: angular frequency in GHz
nlevels: number of levels
"""
def harmonic_oscillator_hamiltonian(w, nlevels):
    a = destroy(nlevels)
    n = a.conj().T @ a + 1/2 * np.eye(nlevels)
    return w * n




H = harmonic_oscillator_hamiltonian(w=5.0, nlevels=4)
print(H)

plot_evolve_state_under_hamiltonian(H, psi0=np.array([0.5, 0.5, 0.5, 0.5]), t=10.0)
