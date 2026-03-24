import numpy as np
import matplotlib.pyplot as plt
import cpb as cpb
import duffing as duf

def plot_energy_levels_vs_flux():
    # CPB needs enough charge states for E_J >> E_C; otherwise the low spectrum
    # does not converge to the transmon / Duffing oscillator limit.
    nlevels = 32
    n_plot = 8  # low-lying levels only (Hilbert space must be larger for accurate CPB)

    # Duffing parameters (transmon leading order: w ~= sqrt(8 E_J E_C), alpha ~= -E_C)
    w = 10.0
    EC = 0.5
    alpha = -EC

    # CPB parameters (match Duffing: EJ_max from w^2 = 8 E_C E_J)
    EJ_max = w**2 / (8*EC)
    d = 0.0
    ng = 0.0

    flux_bias = np.linspace(0, 0.01, 2) #Low regime. 2 points is enough.

    # Duffing H = w n + (alpha/2) n(n−1) is a truncated oscillator; matching the CPB
    # ladder at moderate E_J/E_C needs the transmon 0->1 correction w ~= w_p - E_C
    # (same order as alpha ~= -E_C). Using w = w_p alone drifts badly when E_C is large.
    w_duffing = w - EC

    print(f"EJ_max: {EJ_max}")
    print(f"EC: {EC}")
    print(f"d: {d}")
    print(f"ng: {ng}")
    print(f"flux_bias: {flux_bias}")
    print(f"nlevels: {nlevels}")
    print(f"w (sqrt(8 EJ EC)): {w}")
    print(f"w_duffing (for Duffing compare): {w_duffing}")
    print(f"alpha: {alpha}")

    #cpb.plot_EJ_vs_flux(EJ_max, d)

    cpb_energies = cpb.energy_levels_vs_flux(EC, EJ_max, flux_bias, d, ng, nlevels)

    # Subtract the lowest energy at each flux (relative energies)
    cpb_energies_relative = cpb_energies - cpb_energies[:, [0]]

    duf_energies = duf.energy_levels(w_duffing, alpha, nlevels)
    duf_energies_relative = duf_energies - duf_energies[0]

    # Plot CPB energies relative to ground
    for level in range(n_plot):
        label = "CPB" if level == 0 else None
        plt.plot(flux_bias, cpb_energies_relative[:, level], label=label)

    # Draw a horizontal line for each Duffing level, also relative to ground
    for i in range(n_plot):
        E = duf_energies_relative[i]
        label = 'Duffing' if i == 0 else None
        plt.axhline(y=E, color='r', linestyle='--', label=label)

    plt.xlabel('Flux bias ($\\Phi / \\Phi_0$) (for CPB)')
    plt.ylabel('Energy relative to ground (GHz)')
    plt.title('Energy Levels vs Flux Bias (relative to ground)')
    plt.legend()
    plt.savefig(
        f"energy_levels_vs_flux_model0_wp={w}_wduff={w_duffing}_alpha={alpha}_EC={EC}_EJ_max={EJ_max}_d={d}_ng={ng}_nlevels={nlevels}_n_plot={n_plot}.pdf",
        format="pdf",
    )

plot_energy_levels_vs_flux()