"""
Duffing single mode Hamiltonian
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.linalg import expm

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

    # No meaningful x-data is plotted (we only draw horizontal lines),
    # so Matplotlib would otherwise show its default x-range of 0..1.
    plt.gca().set_xticks([])
    plt.xlabel('')
    plt.ylabel('Energy (GHz)')
    plt.title('Energy Levels')
    plt.savefig("energy_levels_duffling.pdf", format="pdf")


def evolve_state(psi0, w, alpha, nlevels, t_values):
    """Schrödinger evolution |psi(t)> = exp(-i H t) |psi0> with static Duffing H."""
    H = duffing_single_mode(w, alpha, nlevels)
    psi_values = []
    for t in t_values:
        U = expm(-1j * H * t)
        psi_values.append(U @ psi0)
    return psi_values


def plot_evolve_state(
    psi0=np.array([np.sqrt(1/6), np.sqrt(1/6), np.sqrt(1/6), np.sqrt(1/6), np.sqrt(1/6), np.sqrt(1/6)], dtype=complex),
    w=5.0,
    alpha=-1.0,
    nlevels=6,
    dt=0.05,
    t=10.0,
    outfile="statevector_evolution_duffling.pdf",
):
    """Hue = phase, brightness = |amplitude| for each Fock level vs time (cf. model1/heff.py)."""
    if psi0 is None:
        psi0 = np.zeros(nlevels, dtype=complex)
        psi0[0] = 1.0
    psi0 = np.asarray(psi0, dtype=complex).ravel()
    if psi0.shape[0] != nlevels:
        raise ValueError(f"psi0 length {psi0.shape[0]} must match nlevels={nlevels}")

    t_values = np.linspace(0, t, int(t / dt))
    psi_t = evolve_state(psi0, w, alpha, nlevels, t_values)

    # Shape of psi_arr: (n_levels, n_times)
    psi_arr = np.array(psi_t).T

    amplitude = np.clip(np.abs(psi_arr), 0.0, 1.0)
    phase = np.angle(psi_arr)
    hue = (phase + np.pi) / (2 * np.pi)
    saturation = np.ones_like(hue)
    value = amplitude
    hsv = np.stack((hue, saturation, value), axis=-1)
    rgb = hsv_to_rgb(hsv)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(
        rgb,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        resample=False,
        extent=[t_values[0], t_values[-1], -0.5, psi_arr.shape[0] - 0.5],
    )
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Fock level")
    ax.set_title("Statevector evolution (hue=phase, brightness=|amplitude|)")
    y_labels = [rf"$|{k}\rangle$" for k in range(nlevels)]
    ax.set_yticks(np.arange(nlevels))
    ax.set_yticklabels(y_labels)

    phase_mappable = plt.cm.ScalarMappable(
        cmap="hsv", norm=plt.Normalize(vmin=-np.pi, vmax=np.pi)
    )
    phase_mappable.set_array([])
    cbar = fig.colorbar(phase_mappable, ax=ax)
    cbar.set_ticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
    cbar.set_ticklabels(["$-\pi$", "$-\pi/2$", "$0$", "$\pi/2$", "$\pi$"])
    cbar.set_label("Phase")

    fig.tight_layout()
    plt.savefig(outfile, format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    plot_evolve_state()
