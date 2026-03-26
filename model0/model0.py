import numpy as np
import matplotlib.pyplot as plt
import cpb as cpb


def duffing_relative_energies(w, alpha, n_plot):
    """Physical Fock ladder E_n = w n + (alpha/2) n(n-1), relative to ground.

    Do not use np.linalg.eigh on the truncated Duffing matrix for this: for alpha < 0
    the diagonal entries turn very negative at large n, so the lowest eigenvalues are
    high-occupation states, not the transmon-like |0>, |1>, ... levels.
    """
    n = np.arange(n_plot)
    E = w * n + 0.5 * alpha * n * (n - 1)
    return E - E[0]


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

    flux_bias = np.linspace(0, 1, 101) #Low regime. 2 points is enough.

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

    duf_energies_relative = duffing_relative_energies(w_duffing, alpha, n_plot)

    # Same color per excitation index: solid CPB vs flux, dashed Duffing reference
    for level in range(n_plot):
        color = f"C{level % 10}"
        plt.plot(
            flux_bias,
            cpb_energies_relative[:, level],
            color=color,
            label="CPB" if level == 0 else None,
        )
        plt.axhline(
            y=duf_energies_relative[level],
            color=color,
            linestyle="--",
            label="Duffing" if level == 0 else None,
        )

    plt.xlabel('Flux bias ($\\Phi / \\Phi_0$) (for CPB)')
    plt.ylabel('Energy relative to ground (GHz)')
    plt.title('Energy Levels vs Flux Bias (relative to ground)')
    plt.legend()
    plt.savefig(
        f"energy_levels_vs_flux_model0_wp={w}_wduff={w_duffing}_alpha={alpha}_EC={EC}_EJ_max={EJ_max}_d={d}_ng={ng}_nlevels={nlevels}_n_plot={n_plot}.pdf",
        format="pdf",
    )


def plot_energy_levels_vs_nlevels():
    """First n_plot CPB levels vs charge-basis size nlevels at fixed flux (convergence check)."""
    n_plot = 8
    flux_fixed = 0.0  # Phi / Phi0

    w = 10.0
    EC = 0.5
    alpha = -EC
    EJ_max = w**2 / (8 * EC)
    d = 0.0
    ng = 0.0
    w_duffing = w - EC

    nlevels_min = n_plot
    nlevels_max = 64
    nlevels_vals = np.arange(nlevels_min, nlevels_max + 1)

    EJ = cpb.flux_dependent_EJ(EJ_max, flux_fixed, d)
    energies_rel = np.zeros((len(nlevels_vals), n_plot))
    for i, nl in enumerate(nlevels_vals):
        H = cpb.cooper_pair_box_hamiltonian(EC, EJ, ng, nl)
        evals = np.linalg.eigh(H)[0]
        energies_rel[i, :] = evals[:n_plot] - evals[0]

    duf_rel = duffing_relative_energies(w_duffing, alpha, n_plot)

    plt.figure()
    for level in range(n_plot):
        color = f"C{level % 10}"
        plt.plot(
            nlevels_vals,
            energies_rel[:, level],
            color=color,
            label="CPB" if level == 0 else None,
        )
        plt.axhline(
            y=duf_rel[level],
            color=color,
            linestyle="--",
            label="Duffing (ref.)" if level == 0 else None,
        )

    plt.xlabel("nlevels (charge basis truncation)")
    plt.ylabel("Energy relative to ground (GHz)")
    plt.title(f"Energy levels vs nlevels at fixed flux $\\Phi/\\Phi_0 = {flux_fixed}$")
    plt.legend()
    plt.savefig(
        f"energy_levels_vs_nlevels_model0_flux={flux_fixed}_wp={w}_wduff={w_duffing}_"
        f"alpha={alpha}_EC={EC}_EJ_max={EJ_max}_d={d}_ng={ng}_nlevels_max={nlevels_max}_n_plot={n_plot}.pdf",
        format="pdf",
    )


plot_energy_levels_vs_nlevels()