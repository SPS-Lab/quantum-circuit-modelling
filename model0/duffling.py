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
    print(f"a:\n{a}")
    n = a.conj().T @ a
    print("n: \n", n)
    I = np.eye(nlevels)
    return w * n + 0.5 * alpha * (n @ (n - I))

def plot_energy_levels(w, alpha, nlevels):
    H = duffing_single_mode(w, alpha, nlevels)
    evals, evecs = np.linalg.eigh(H)
    plt.plot(evals, label=f'w={w}, alpha={alpha}, nlevels={nlevels}')
    plt.legend()
    plt.xlabel('Level index')
    plt.ylabel('Energy (GHz)')
    plt.title('Energy Levels')
    plt.show()

plot_energy_levels(w=5.0, alpha=-1.25, nlevels=6)