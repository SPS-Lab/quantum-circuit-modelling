import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from helpers import px, py, pz, I2

def heff(w1, w2, J, zeta):
    return (
        0.5 * w1 * np.kron(pz, I2)
        + 0.5 * w2 * np.kron(I2, pz)
        + J * (np.kron(px, px) + np.kron(py, py))
        + 0.25 * zeta * np.kron(pz, pz)
    )


def energy_levels(w1, w2, J, zeta):
    H = heff(w1, w2, J, zeta)
    evals, evecs = np.linalg.eigh(H)
    return evals

def plot_energy_levels(w1=5.0, w2=5.2, J=0.01, zeta=0.002):
    evals = energy_levels(w1, w2, J, zeta)
    plt.plot(evals)
    plt.xlabel('Level index')
    plt.ylabel('Energy (GHz)')
    plt.title('Energy Levels vs Level Index')
    plt.show()

def evolve_state(psi0, w1, w2, J, zeta, t_values):
    H = heff(w1, w2, J, zeta)
    psi_values = []
    for t in t_values:
        U = expm(-1j * H * t)
        psi_values.append(U @ psi0)
    return psi_values

def plot_evolve_state(psi0=np.array([1, 0, 0, 0], dtype=complex), w1=5.0, w2=5.2, J=0.01, zeta=0.002, dt=0.5, t=10000):
    t_values = np.linspace(0, t, int(t/dt))
    psi_t = evolve_state(psi0, w1, w2, J, zeta, t_values)
    plt.plot(t_values, [np.abs(psi[0])**2 for psi in psi_t]) # Probability of |00>
    plt.plot(t_values, [np.abs(psi[1])**2 for psi in psi_t]) # Probability of |01>
    plt.plot(t_values, [np.abs(psi[2])**2 for psi in psi_t]) # Probability of |10>
    plt.plot(t_values, [np.abs(psi[3])**2 for psi in psi_t]) # Probability of |11>
    plt.xlabel('Time (ns)')
    plt.ylabel('Probability of state')
    plt.title('Probability of states vs Time')
    plt.legend(['|00>', '|01>', '|10>', '|11>'])
    plt.show()

plot_energy_levels()



plot_evolve_state()