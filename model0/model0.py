import numpy as np
import matplotlib.pyplot as plt
import cpb as cpb
from toolkit.metrics import single_mode_static_metrics
from cpb import cooper_pair_box_hamiltonian
from duffing import duffing_single_mode


def _cpb_charge_basis_index(n_phys: int, ng: float, nlevels: int) -> int:
    """Row/column index for integer Cooper-pair number ``n_phys`` in ``cooper_pair_box_hamiltonian``."""
    n_center = int(round(ng))
    n_low = n_center - (nlevels // 2)
    return int(n_phys - n_low)


def duffing_relative_energies(w, alpha, n_plot):
    """Physical Fock ladder E_n = w n + (alpha/2) n(n-1), relative to ground.

    Same as ``np.diag(duffing_single_mode(...))`` in index order. Metrics from ``H``
    should use :func:`toolkit.metrics.single_mode_static_metrics` (ordered diagonal),
    not raw ``eigh`` when ``alpha < 0`` (sorted eigenvalues mislabel Fock levels).
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

    # Duffing H = w n + (alpha/2) n(n-1) is a truncated oscillator; matching the CPB
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


def plot_static_metrics_vs_nlevels():
    """CPB static metrics vs charge-basis size at fixed flux; Duffing ref. from same metrics API.

    Duffing reference uses :func:`toolkit.metrics.single_mode_static_metrics` on
    ``duffing_single_mode`` (diagonal Hamiltonian: energies in Fock index order).

    Overlaps follow the **diagonal Fock** analogy ``|<k|psi_k>|^2`` (ideal Duffing: 1
    for each k): charges ``n_ref + k`` vs. ``psi_k`` with
    ``n_ref = round(n_g)`` (e.g. ``|<2|psi_2>|^2`` when ``n_g = 0``).
    """
    flux_fixed = 0.0
    w = 10.0
    EC = 0.5
    alpha = -EC
    EJ_max = w**2 / (8 * EC)
    d = 0.0
    ng = 0.0
    w_duffing = w - EC

    nlevels_min = 3
    nlevels_max = 64
    nlevels_vals = np.arange(nlevels_min, nlevels_max + 1)

    EJ = cpb.flux_dependent_EJ(EJ_max, flux_fixed, d)

    omega_01 = np.zeros(len(nlevels_vals))
    alpha_cpb = np.zeros(len(nlevels_vals))
    # Diagonal Fock-style: |<n_ref+k|psi_k>|^2 for k = 0, 1, 2.
    n_ref = int(round(ng))
    n_exc = n_ref + 1
    n2 = n_ref + 2
    overlap_ng_g = np.zeros(len(nlevels_vals))
    overlap_ne1_e1 = np.zeros(len(nlevels_vals))
    overlap_ne2_e2 = np.zeros(len(nlevels_vals))

    m_32 = None  # Will hold metrics for nlevels = 32

    for i, nl in enumerate(nlevels_vals):
        H = cooper_pair_box_hamiltonian(EC, EJ, ng, nl)
        m = single_mode_static_metrics(H)
        omega_01[i] = m.omega_01
        alpha_cpb[i] = m.alpha if m.alpha is not None else np.nan
        j0 = _cpb_charge_basis_index(n_ref, ng, nl)
        j1 = _cpb_charge_basis_index(n_exc, ng, nl)
        j2 = _cpb_charge_basis_index(n2, ng, nl)
        overlap_ng_g[i] = m.overlap_bare_eigen[j0, 0]
        if 0 <= j1 < nl:
            overlap_ne1_e1[i] = m.overlap_bare_eigen[j1, 1]
        else:
            overlap_ne1_e1[i] = np.nan
        if nl >= 3 and 0 <= j2 < nl:
            overlap_ne2_e2[i] = m.overlap_bare_eigen[j2, 2]
        else:
            overlap_ne2_e2[i] = np.nan

        if nl == 32:
            m_32 = m  # Save metrics for nlevels=32

    H_duff_ref = duffing_single_mode(w_duffing, alpha, max(nlevels_max, 64))
    m_duff = single_mode_static_metrics(H_duff_ref)
    ref_omega_01 = m_duff.omega_01
    ref_alpha = m_duff.alpha if m_duff.alpha is not None else np.nan
    ref_ol_g = float(m_duff.overlap_bare_eigen[n_ref, 0])
    ref_ol_e1 = float(m_duff.overlap_bare_eigen[n_exc, 1])
    ref_ol_e2 = float(m_duff.overlap_bare_eigen[n2, 2])

    # Print metrics for nlevels=32 and their differences from duffing reference
    if m_32 is not None:
        print("=== CPB metrics for nlevels=32 ===")
        print(f"omega_01 (CPB): {m_32.omega_01:.8f} GHz")
        print(f"omega_01 (Duffing): {ref_omega_01:.8f} GHz")
        print(f"Delta omega_01 (CPB - Duffing): {m_32.omega_01 - ref_omega_01:+.8f} GHz")
        alpha_32 = m_32.alpha if m_32.alpha is not None else np.nan
        print(f"alpha (CPB): {alpha_32:.8f} GHz")
        print(f"alpha (Duffing): {ref_alpha:.8f} GHz")
        print(f"Delta alpha (CPB - Duffing): {alpha_32 - ref_alpha:+.8f} GHz")
        print(f"Overlap |<n={n_ref}|psi_0>|^2: {m_32.overlap_bare_eigen[_cpb_charge_basis_index(n_ref, ng, 32), 0]:.8f}")
        print(f"Overlap |<n={n_exc}|psi_1>|^2: {m_32.overlap_bare_eigen[_cpb_charge_basis_index(n_exc, ng, 32), 1]:.8f}")
        print(f"Overlap |<n={n2}|psi_2>|^2: {m_32.overlap_bare_eigen[_cpb_charge_basis_index(n2, ng, 32), 2]:.8f}")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.0, 9.0))

    axes[0].plot(nlevels_vals, omega_01, color="C0", label="CPB")
    axes[0].axhline(ref_omega_01, color="C0", linestyle="--", alpha=0.8, label="Duffing (ref.)")
    axes[0].set_ylabel(r"$\omega_{01}$ (GHz)")
    axes[0].legend(loc="best")
    axes[0].set_title("Qubit frequency (0->1 transition)")

    axes[1].plot(nlevels_vals, alpha_cpb, color="C1", label="CPB")
    axes[1].axhline(ref_alpha, color="C1", linestyle="--", alpha=0.8, label="Duffing (ref.)")
    axes[1].set_ylabel(r"$\alpha \approx \omega_{12}-\omega_{01}$ (GHz)")
    axes[1].legend(loc="best")
    axes[1].set_title("Anharmonicity")

    axes[2].plot(
        nlevels_vals,
        overlap_ng_g,
        color="C2",
        label=rf"$|\langle n={n_ref}|\psi_0\rangle|^2$",
    )
    axes[2].plot(
        nlevels_vals,
        overlap_ne1_e1,
        color="C3",
        label=rf"$|\langle n={n_exc}|\psi_1\rangle|^2$",
    )
    axes[2].plot(
        nlevels_vals,
        overlap_ne2_e2,
        color="C4",
        label=rf"$|\langle n={n2}|\psi_2\rangle|^2$",
    )
    axes[2].axhline(
        ref_ol_g,
        color="C2",
        linestyle="--",
        alpha=0.8,
        label="Duffing (ref.)",
    )
    axes[2].axhline(ref_ol_e1, color="C3", linestyle="--", alpha=0.8)
    axes[2].axhline(ref_ol_e2, color="C4", linestyle="--", alpha=0.8)
    axes[2].axhline(1.0, color="gray", linestyle=":", alpha=0.6)
    axes[2].set_ylabel("overlap")
    axes[2].set_xlabel("nlevels (charge basis truncation)")
    axes[2].legend(loc="best")
    axes[2].set_title(
        "Diagonal Fock-style overlap: "
        rf"$n={n_ref}, {n_exc}, {n2}$ vs. $\psi_0, \psi_1, \psi_2$"
    )
    axes[2].set_ylim(0.0, 1.05)

    fig.suptitle(
        f"Static metrics vs nlevels at fixed flux $\\Phi/\\Phi_0 = {flux_fixed}$",
        y=1.01,
    )
    fig.tight_layout()
    plt.savefig(
        f"static_metrics_vs_nlevels_model0_flux={flux_fixed}_wp={w}_wduff={w_duffing}_"
        f"alpha={alpha}_EC={EC}_EJ_max={EJ_max}_d={d}_ng={ng}_nlevels_max={nlevels_max}.pdf",
        format="pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    # plot_energy_levels_vs_flux()
    # plot_energy_levels_vs_nlevels()
    plot_static_metrics_vs_nlevels()