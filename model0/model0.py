import numpy as np
import matplotlib.pyplot as plt
import cpb as cpb
import duffing as duf

def plot_energy_levels_vs_flux():
    #Both models
    nlevels = 6

    #CPB parameters
    EC = 1.0
    EJ_max = 10.0
    d = 1.1
    ng = 0.0
    flux_bias=np.linspace(0, 1, 100)

    #Duffing parameters
    w = 20.0
    alpha = -1.00

    cpb.plot_EJ_vs_flux(EJ_max, d)

    cpb_energies = cpb.energy_levels_vs_flux(EC, EJ_max, flux_bias, d, ng, nlevels)

    duf_energies = duf.energy_levels(w, alpha, nlevels)
    
    plt.plot(flux_bias, cpb_energies, label='CPB')

    # duf_energies is a 1D array (one value per level), independent of flux bias.
    # Draw a horizontal line for each Duffing level.
    for i, E in enumerate(duf_energies):
        label = 'Duffing' if i == 0 else None
        plt.axhline(y=E, color='r', linestyle='--', label=label)

    plt.xlabel('Flux bias ($\\Phi / \\Phi_0$) (for CPB)')
    plt.ylabel('Energy (GHz)')
    plt.title('Energy Levels vs Flux Bias')
    plt.savefig("energy_levels_vs_flux_model0.pdf", format="pdf")

plot_energy_levels_vs_flux()