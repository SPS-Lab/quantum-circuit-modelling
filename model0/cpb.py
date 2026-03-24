"""
Cooper pair box (CPB) Hamiltonian
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

"""
Flux-dependent Josephson energy
EJ(phi) = EJ_max * sqrt( cos^2(pi * phi / phi0) + d^2 * sin^2(pi * phi / phi0) )
EJ_max: maximum Josephson energy
flux_bias: applied flux phi / phi0
d: asymmetry parameter
"""
def flux_dependent_EJ(EJ_max, flux_bias, d):
    x = np.pi * np.asarray(flux_bias)
    #print(f"x:\n{x}")
    cosx = np.cos(x)
    #print(f"cosx:\n{cosx}")
    sinx = np.sin(x)
    #print(f"sinx:\n{sinx}")
    return EJ_max * np.sqrt(cosx**2 + (d**2) * sinx**2)

"""
EC: charging energy
EJ: Josephson energy
ng: dimensionless offset charge
nlevels: number of charge basis states |n>
"""
def cooper_pair_box_hamiltonian(EC, EJ, ng, nlevels):
    # Consecutive Cooper-pair numbers centered near ng (symmetric around ng=0).
    # Truncating only n >= 0 imposes a hard boundary at n = -1 and badly distorts
    # the transmon / Duffing limit when E_J >> E_C.
    n_center = int(round(ng))
    n_low = n_center - (nlevels // 2)
    n_vals = np.arange(n_low, n_low + nlevels, dtype=float)
    n_op = np.diag(n_vals)

    # (n_op - n_g I)^2 term
    Id = np.eye(nlevels, dtype=complex)
    n_shift = n_op - ng * Id
    H_charge = 4.0 * EC * (n_shift @ n_shift)

    # Josephson term: -EJ/2 * sum_n (|n> <n+1| + |n+1> <n|)
    off = np.zeros((nlevels, nlevels), dtype=complex)
    for k in range(nlevels - 1):
        off[k, k + 1] = 1.0
        off[k + 1, k] = 1.0
    H_josephson = -(EJ / 2.0) * off

    return H_charge + H_josephson


def plot_EJ_vs_flux(EJ_max=20.0, d=0.1, flux_bias=np.linspace(0, 1, 100)):
    EJ = flux_dependent_EJ(EJ_max, flux_bias, d)
    print(f"EJ:\n{EJ}")
    plt.plot(flux_bias, EJ)
    plt.xlabel('Flux bias ($\Phi / \Phi_0$)')
    plt.ylabel('Josephson Energy (GHz)')
    plt.title('Flux-Dependent Josephson Energy')
    plt.savefig("EJ_vs_flux_cpb.pdf", format="pdf")
    plt.close()

def energy_levels_vs_flux(EC, EJ_max, flux_bias, d, ng, nlevels):
    EJ_array = flux_dependent_EJ(EJ_max, flux_bias, d)

    energies = np.zeros((len(flux_bias), nlevels))

    for i, EJ in enumerate(EJ_array):
        H_cooper_pair_box = cooper_pair_box_hamiltonian(EC, EJ, ng, nlevels)
        print(f"H_cooper_pair_box:\n{H_cooper_pair_box}")
        evals, evacs = np.linalg.eigh(H_cooper_pair_box)
        print(f"evals:\n{evals}")
        print(f"evacs:\n{evacs}")
        energies[i, :] = evals

    return energies

def plot_energy_levels_vs_flux(EC=0.0, EJ_max=1.0, flux_bias=np.linspace(0, 1, 2), d=0.1, ng=0.5, nlevels=6):
    plot_EJ_vs_flux(EJ_max, d)
    
    energies = energy_levels_vs_flux(EC, EJ_max, flux_bias, d, ng, nlevels)

    # Subtract the lowest energy at each flux to plot energies relative to the ground state
    energies_relative = energies - energies[:, [0]]

    for level in range(nlevels):
        plt.plot(flux_bias, energies_relative[:, level])

    plt.xlabel('Flux bias ($\\Phi / \\Phi_0$)')
    plt.ylabel('Energy relative to ground (GHz)')
    plt.title('Energy Levels vs Flux Bias (relative to ground)')
    plt.savefig("energy_levels_vs_flux_cpb.pdf", format="pdf")


def cpb_plot_evolve_state(psi0=np.array([np.sqrt(1), np.sqrt(0), np.sqrt(0), np.sqrt(0), np.sqrt(0), np.sqrt(0)], dtype=complex),
                          EC=1.0, EJ_max=20.0, ng=1.0, nlevels=6,
                          t=4.0, dt=0.01, outfile="statevector_evolution_cpb.pdf", style="panels"):
    H = cooper_pair_box_hamiltonian(EC, EJ_max, ng, nlevels)
    print(f"H:\n{H}")
    plot_evolve_state_under_hamiltonian(H, psi0, t, dt, outfile, style)


if __name__ == "__main__":
    #plot_EJ_vs_flux(d = 0.0, flux_bias=np.linspace(0, 1.0, 100))
    plot_energy_levels_vs_flux()
    #cpb_plot_evolve_state()