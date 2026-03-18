"""
Cooper pair box (CPB) Hamiltonian
"""

import numpy as np
import matplotlib.pyplot as plt
from helpers import destroy

"""
Flux-dependent Josephson energy
EJ(phi) = EJ_max * sqrt( cos^2(pi * phi / phi0) + d^2 * sin^2(pi * phi / phi0) )
EJ_max: maximum Josephson energy
flux_bias: applied flux phi / phi0
d: asymmetry parameter
"""
def flux_dependent_EJ(EJ_max, flux_bias, d):
    x = np.pi * np.asarray(flux_bias)
    cosx = np.cos(x)
    sinx = np.sin(x)
    return EJ_max * np.sqrt(cosx**2 + (d**2) * sinx**2)

"""
EC: charging energy
EJ: Josephson energy
ng: dimensionless offset charge
nlevels: number of charge basis states |n>
"""
def cooper_pair_box_hamiltonian(EC, EJ, ng, nlevels):
    # Charge number operator n_op with eigenvalues 0, 1, ..., nlevels-1
    n_vals = np.arange(nlevels, dtype=float)
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


def plot_EJ_vs_flux(EJ_max, d):
    flux_bias = np.linspace(0, 1, 100)
    EJ = flux_dependent_EJ(EJ_max, flux_bias, d)
    plt.plot(flux_bias, EJ)
    plt.xlabel('Flux bias ($\Phi / \Phi_0$)')
    plt.ylabel('Josephson Energy (GHz)')
    plt.title('Flux-Dependent Josephson Energy')
    plt.show()

def energy_levels_vs_flux(EC, EJ_max, flux_bias, d, ng, nlevels):
    EJ_array = flux_dependent_EJ(EJ_max, flux_bias, d)

    energies = np.zeros((len(flux_bias), nlevels))

    for i, EJ in enumerate(EJ_array):
        H_cooper_pair_box = cooper_pair_box_hamiltonian(EC, EJ, ng, nlevels)
        evals, _ = np.linalg.eigh(H_cooper_pair_box)
        energies[i, :] = evals

    return energies

def plot_energy_levels_vs_flux(flux_bias):
    nlevels = 6
    energies = energy_levels_vs_flux(EC=1.0, EJ_max=20.0, flux_bias=flux_bias, d=0.1, ng=0.0, nlevels=nlevels)

    for level in range(nlevels):
        plt.plot(flux_bias, energies[:, level])

    plt.xlabel('Flux bias ($\\Phi / \\Phi_0$)')
    plt.ylabel('Energy (GHz)')
    plt.title('Energy Levels vs Flux Bias')
    plt.show()

plot_EJ_vs_flux(EJ_max=1.0, d=0.1)

plot_energy_levels_vs_flux(flux_bias=np.linspace(0, 1, 100)) #Dependence on flux bias not visible in plot
