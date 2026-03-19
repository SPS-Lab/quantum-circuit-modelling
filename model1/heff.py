import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
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
    
    print(f"U matrix element at end of time evolution:\n{expm(-1j * H * t_values[-1])}")

    return psi_values

def plot_evolve_state(psi0=np.array([1, 0, 0, 0], dtype=complex), w1=5.0, w2=5.2, J=0.01, zeta=0.002, dt=0.05, t=10):
    t_values = np.linspace(0, t, int(t/dt))
    psi_t = evolve_state(psi0, w1, w2, J, zeta, t_values)

    # Shape of psi_arr: (n_levels, n_times)
    psi_arr = np.array(psi_t).T

    # Numerical drift can push values infinitesimally above 1.0.
    amplitude = np.clip(np.abs(psi_arr), 0.0, 1.0)
    phase = np.angle(psi_arr)

    # Map phase -> hue in [0, 1], magnitude -> value/brightness in [0, 1]
    hue = (phase + np.pi) / (2 * np.pi)
    saturation = np.ones_like(hue)
    value = amplitude
    hsv = np.stack((hue, saturation, value), axis=-1)
    rgb = hsv_to_rgb(hsv)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(
        rgb,
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        resample=False,
        extent=[t_values[0], t_values[-1], -0.5, psi_arr.shape[0] - 0.5],
    )
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('State level')
    ax.set_title('Statevector evolution (hue=phase, brightness=|amplitude|)')
    ax.set_yticks([0, 1, 2, 3], ['|00>', '|01>', '|10>', '|11>'])

    # Phase-only colorbar for hue interpretation.
    phase_mappable = plt.cm.ScalarMappable(cmap='hsv', norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
    phase_mappable.set_array([])
    cbar = fig.colorbar(phase_mappable, ax=ax)
    cbar.set_ticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
    cbar.set_ticklabels(['-pi', '-0.5 pi', '0', '0.5 pi', 'pi'])
    cbar.set_label('Phase')

    fig.tight_layout()
    plt.show()

plot_energy_levels()



plot_evolve_state()