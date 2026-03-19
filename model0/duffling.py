"""
Duffing single mode Hamiltonian
"""

import numpy as np
import matplotlib.pyplot as plt
from helpers import destroy

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

    plt.xlabel('Level index')
    plt.ylabel('Energy (GHz)')
    plt.title('Energy Levels vs Level Index')
    plt.savefig("energy_levels_duffling.pdf", format="pdf")
