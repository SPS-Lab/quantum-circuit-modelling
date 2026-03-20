import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from helpers import px, py, pz, I2


def w_vs_flux(w_O, w_A, flux_bias):
    w = w_O + w_A * np.cos(2 * np.pi * flux_bias)
    return w

def J_vs_flux(J_O, J_A, flux_bias):
    J = J_O + J_A * np.cos(2 * np.pi * flux_bias)
    return J

def zeta_vs_flux(zeta_O, zeta_A, flux_bias):
    zeta = zeta_O + zeta_A * np.cos(2 * np.pi * flux_bias)
    return zeta

def _coeff_for_ham(c):
    """Scalar * (4,4) stays (4,4); 1D (n,) -> (n,1,1) for batched eigh."""
    c = np.asarray(c)
    if c.ndim == 0:
        return c
    if c.ndim == 1:
        return c[:, np.newaxis, np.newaxis]
    raise ValueError("w1, w2, J, zeta must be scalars or 1D arrays")


def heff(w1, w2, J, zeta):
    """Build H_eff. Coefficients may be scalars or 1D arrays (e.g. vs flux)."""
    w1 = _coeff_for_ham(w1)
    w2 = _coeff_for_ham(w2)
    J = _coeff_for_ham(J)
    zeta = _coeff_for_ham(zeta)
    return (
        0.5 * w1 * np.kron(pz, I2)
        + 0.5 * w2 * np.kron(I2, pz)
        + J * (np.kron(px, px) + np.kron(py, py))
        + 0.25 * zeta * np.kron(pz, pz)
    )

def energy_levels_vs_flux(w_O, w_A, J_O, J_A, zeta_O, zeta_A, flux_bias):
    w1 = w_vs_flux(w_O, w_A, flux_bias)
    w2 = w_vs_flux(w_O, w_A, flux_bias)
    J = J_vs_flux(J_O, J_A, flux_bias)
    zeta = zeta_vs_flux(zeta_O, zeta_A, flux_bias)
    H = heff(w1, w2, J, zeta)  # (n_flux, 4, 4) when flux_bias is 1D
    evals, evecs = np.linalg.eigh(H)  # (n_flux, 4)
    return evals, evecs

def plot_energy_levels_vs_flux(w_O=5.0, w_A=0.2, J_O=0.01, J_A=0.002, zeta_O=0.002, zeta_A=0.0002, flux_bias=np.linspace(0, 1, 100)):
    evals, evecs = energy_levels_vs_flux(w_O, w_A, J_O, J_A, zeta_O, zeta_A, flux_bias)
    from pprint import pprint
    print("First set of state vectors (eigenvectors) at initial flux value:")
    pprint(evecs[0])
    for i in range(4):
        plt.plot(flux_bias, evals[:, i], label=rf"$E_{i}$ (GHz)")
    plt.xlabel('Flux bias ($\Phi / \Phi_0$)')
    plt.ylabel('Energy (GHz)')
    plt.title('Energy Levels vs Flux Bias')
    plt.legend()
    #plt.savefig("energy_levels_vs_flux_model1.pdf", format="pdf")
    plt.show()

def evolve_state(psi0, w1, w2, J, zeta, t_values):
    H = heff(w1, w2, J, zeta)
    psi_values = []
    for t in t_values:
        U = expm(-1j * H * t)
        psi_values.append(U @ psi0)
    
    print(f"U matrix element at end of time evolution:\n{expm(-1j * H * t_values[-1])}")

    return psi_values

def plot_evolve_state(psi0=np.array([np.sqrt(1/3), np.sqrt(0), np.sqrt(1/3), np.sqrt(1/3)], dtype=complex), w1=5.0, w2=5.2, J=1, zeta=0.002, dt=0.05, t=10):
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
    plt.savefig("statevector_evolution_model1.pdf", format="pdf")

plot_energy_levels_vs_flux()



plot_evolve_state()