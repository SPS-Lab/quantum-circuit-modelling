"""
Duffing single mode Hamiltonian
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Repo root (parent of model0/) so `toolkit.plotting` resolves when run from model0/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from toolkit.plotting import plot_evolve_state as plot_evolve_state_under_hamiltonian
from toolkit.helpers import destroy

"""
w: angular frequency in GHz
alpha: anharmonicity
nlevels: number of levels
"""
def duffing_single_mode(w, alpha, nlevels):
    a = destroy(nlevels)
    n = a.conj().T @ a
    I = np.eye(nlevels)
    return w * n + 0.5 * alpha * (n @ (n - I))

def energy_levels(w, alpha, nlevels):
    H = duffing_single_mode(w, alpha, nlevels)
    evals, evecs = np.linalg.eigh(H)
    return evals

def plot_energy_levels(w=5.0, alpha=-1, nlevels=6):
    evals = energy_levels(w, alpha, nlevels)

    for i, E in enumerate(evals):
        plt.axhline(y=E, color='r', linestyle='--', label=f'Level {i}')

    # No meaningful x-data is plotted (we only draw horizontal lines),
    # so Matplotlib would otherwise show its default x-range of 0..1.
    plt.gca().set_xticks([])
    plt.xlabel('')
    plt.ylabel('Energy (GHz)')
    plt.title('Energy Levels')
    plt.savefig("energy_levels_duffling.pdf", format="pdf")



def duf_plot_evolve_state(psi0=np.array([np.sqrt(1), np.sqrt(1), np.sqrt(1), np.sqrt(1), np.sqrt(1), np.sqrt(1)], dtype=complex),
                          w=5.0,
                          alpha=-1.0,
                          nlevels=6,
                          t=4.0,
                          dt=0.01,
                          outfile="statevector_evolution_duffling.pdf",
                          style="panels"):
    H = duffing_single_mode(w, alpha, nlevels)
    print(f"H: {H}")
    plot_evolve_state_under_hamiltonian(H, psi0, t, dt, outfile, style)


if __name__ == "__main__":
    duf_plot_evolve_state()