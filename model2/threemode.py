import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

# Repo root (parent of model2/) so `toolkit` resolves when run from model2/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from toolkit.helpers import I2, destroy, pz
from toolkit.plotting import plot_energy_levels, plot_energy_levels_vs_flux
from toolkit.spectrum import track_energy_levels_stack


def _import_model1_heff():
    """Load :mod:`model1.heff` by path (``model1`` is not necessarily a package)."""
    path = _ROOT / "model1" / "heff.py"
    spec = importlib.util.spec_from_file_location("model1_heff", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def heff_spin_to_lab_hamiltonian(H_eff: np.ndarray, w1, w2) -> np.ndarray:
    """Convert :func:`model1.heff.heff` from ``(ω_j/2) σ_z`` to lab ``ω_j n_j`` with ``n = (I-σ_z)/2``.

    Interaction terms ``J (σ_xσ_x + σ_yσ_y)`` and ``(ζ/4) σ_zσ_z`` are unchanged. Equivalently::

        H' = H + (w_1+w_2)/2 · I_4 - w_1 (σ_z⊗I) - w_2 (I⊗σ_z).

    ``w1``, ``w2`` may be scalars or 1D arrays (length ``n_flux``), matching the batching
    used by :func:`model1.heff.heff`.
    """
    H_eff = np.asarray(H_eff, dtype=complex)
    w1_b = np.asarray(w1, dtype=complex)
    w2_b = np.asarray(w2, dtype=complex)
    if w1_b.ndim == 1:
        w1_b = w1_b[:, np.newaxis, np.newaxis]
    if w2_b.ndim == 1:
        w2_b = w2_b[:, np.newaxis, np.newaxis]

    pz1 = np.kron(pz, I2)
    pz2 = np.kron(I2, pz)
    eye4 = np.eye(4, dtype=complex)
    return H_eff + 0.5 * (w1_b + w2_b) * eye4 - w1_b * pz1 - w2_b * pz2


def three_mode_hamiltonian(w_1, w_c, w_2, alpha_1, alpha_c, alpha_2, g_1c, g_2c, nlevels_qubit, nlevels_coupler):
    """
    Constructs the three-mode Hamiltonian:

        H = sum_{j in {1,c,2}} [w_j a_j† a_j + (alpha_j/2) a_j† a_j† a_j a_j]
            + g_1c (a_1† a_c + a_c† a_1)
            + g_2c (a_2† a_c + a_c† a_2)

    Parameters
    ----------
    w_1, w_c, w_2 : float
        Mode frequencies.
    alpha_1, alpha_c, alpha_2 : float
        Mode anharmonicities.
    g_1c, g_2c : float
        Coupling strengths between modes.
    nlevels_qubit, nlevels_coupler : int
        Number of levels per mode.
    Returns
    -------
    H : ndarray
        Hamiltonian matrix, shape
        ``(nlevels_qubit * nlevels_coupler * nlevels_qubit,)`` squared.
    """
    from numpy import kron, eye

    # Single-mode operators (in number basis): subsystems are qubit_1, coupler, qubit_2
    a = [destroy(nlevels_qubit),
         destroy(nlevels_coupler),
         destroy(nlevels_qubit)]
    adag = [op.conj().T for op in a]
    n = [adag[j] @ a[j] for j in range(3)]

    id_q = eye(nlevels_qubit, dtype=complex)
    id_c = eye(nlevels_coupler, dtype=complex)

    # Tensor product operators
    def kron3(o1, o2, o3):
        return kron(kron(o1, o2), o3)

    # Annihilation/creation/number operators for each mode on the full Hilbert space
    a1 = kron3(a[0], id_c, id_q)
    ac = kron3(id_q, a[1], id_q)
    a2 = kron3(id_q, id_c, a[2])
    adag1 = kron3(adag[0], id_c, id_q)
    adagc = kron3(id_q, adag[1], id_q)
    adag2 = kron3(id_q, id_c, adag[2])

    n1 = kron3(n[0], id_c, id_q)
    nc = kron3(id_q, n[1], id_q)
    n2 = kron3(id_q, id_c, n[2])

    # Hamiltonian terms
    H = (
        w_1 * n1
        + w_c * nc
        + w_2 * n2
        + (alpha_1 / 2) * (adag1 @ adag1 @ a1 @ a1)
        + (alpha_c / 2) * (adagc @ adagc @ ac @ ac)
        + (alpha_2 / 2) * (adag2 @ adag2 @ a2 @ a2)
        + g_1c * (adag1 @ ac + adagc @ a1)
        + g_2c * (adag2 @ ac + adagc @ a2)
    )

    return H


def computational_state_indices(
    nlevels_qubit: int, nlevels_coupler: int
) -> np.ndarray:
    """Flat indices for bare ``|n_1, n_c, n_2⟩`` with ``n_c=0`` and ``n_1,n_2 ∈ {0,1}``.

    Tensor layout matches :func:`three_mode_hamiltonian`: index
    ``n_1 * (n_c n_q) + n_c * n_q + n_2``.
    """
    nq, nc = nlevels_qubit, nlevels_coupler
    return np.array(
        [
            0 * (nc * nq) + 0 * nq + 0,
            0 * (nc * nq) + 0 * nq + 1,
            1 * (nc * nq) + 0 * nq + 0,
            1 * (nc * nq) + 0 * nq + 1,
        ],
        dtype=int,
    )


def dressed_computational_energies(
    H: np.ndarray,
    nlevels_qubit: int,
    nlevels_coupler: int,
    *,
    n_candidate_states: int = 16,
) -> np.ndarray:
    """Dressed energies ``(E_00, E_01, E_10, E_11)`` in GHz (same units as ``H``).

    Lowest eigenstates are matched to the four bare computational kets (coupler in ``|0⟩``)
    by maximum overlap (Hungarian assignment).
    """
    H = np.asarray(H, dtype=complex)
    d = H.shape[0]
    idx = computational_state_indices(nlevels_qubit, nlevels_coupler)
    n_cand = max(4, min(int(n_candidate_states), d))

    w, v = np.linalg.eigh(H)
    v_c = v[:, :n_cand]
    overlap = np.abs(v_c[idx, :]) ** 2
    row_ind, col_ind = linear_sum_assignment(-overlap)
    E = np.empty(4, dtype=float)
    for k in range(4):
        E[int(row_ind[k])] = float(w[int(col_ind[k])])
    return E


def exchange_splitting_bare_01_10(
    H: np.ndarray, nlevels_qubit: int, nlevels_coupler: int
) -> float:
    """Splitting ``|λ_+ - λ_-|`` of the ``2×2`` block of ``H`` in the bare ``|01⟩,|10⟩`` subspace.

    Includes both direct coupling and qubit detuning within that subspace (GHz units).
    """
    H = np.asarray(H, dtype=complex)
    idx = computational_state_indices(nlevels_qubit, nlevels_coupler)
    i01, i10 = int(idx[1]), int(idx[2])
    h11 = H[i01, i01].real
    h22 = H[i10, i10].real
    h12 = H[i01, i10]
    tr = h11 + h22
    det = h11 * h22 - h12 * np.conj(h12)
    disc = np.sqrt(max(0.0, 0.25 * tr**2 - det.real))
    return float(2.0 * disc)


def residual_zz_and_exchange(
    H: np.ndarray, nlevels_qubit: int, nlevels_coupler: int, **dress_kw
) -> tuple[float, float]:
    """Return ``(ζ_ZZ, Δ_ex)`` with ``ζ_ZZ = E_11 - E_10 - E_01 + E_00`` (dressed) and bare-subspace splitting."""
    E = dressed_computational_energies(
        H, nlevels_qubit, nlevels_coupler, **dress_kw
    )
    zeta = float(E[3] - E[2] - E[1] + E[0])
    dex = exchange_splitting_bare_01_10(H, nlevels_qubit, nlevels_coupler)
    return zeta, dex


def plot_three_mode_zz_exchange_vs_flux(
    flux_values: np.ndarray,
    *,
    wc0: float,
    A: float,
    outfile: str = "three_mode_ZZ_exchange_vs_flux.pdf",
    title: str | None = None,
    dress_kw: dict | None = None,
    **ham_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot residual ZZ (dressed) and ``|01>-|10>`` bare-subspace splitting vs flux."""
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    dress_kw = dress_kw or {}

    zetas = np.empty_like(flux_values, dtype=float)
    exchanges = np.empty_like(flux_values, dtype=float)

    nq = int(ham_kwargs["nlevels_qubit"])
    nc = int(ham_kwargs["nlevels_coupler"])

    for k, phi in enumerate(flux_values):
        wc = float(wc0 + A * np.cos(2 * np.pi * phi))
        H = three_mode_hamiltonian(
            ham_kwargs["w_1"],
            wc,
            ham_kwargs["w_2"],
            ham_kwargs["alpha_1"],
            ham_kwargs["alpha_c"],
            ham_kwargs["alpha_2"],
            ham_kwargs["g_1c"],
            ham_kwargs["g_2c"],
            nq,
            nc,
        )
        zz, ex = residual_zz_and_exchange(H, nq, nc, **dress_kw)
        zetas[k] = zz
        exchanges[k] = ex

    fig, (ax_z, ax_j) = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True)
    ax_z.plot(flux_values, zetas, color="C0")
    ax_z.set_ylabel(r"Residual ZZ (GHz)")
    ax_z.set_title(title or r"Three-mode: $\zeta = E_{11}-E_{10}-E_{01}+E_{00}$ (dressed)")
    ax_z.grid(True, alpha=0.3)

    ax_j.plot(flux_values, exchanges, color="C1")
    ax_j.set_ylabel(r"Exchange bare $|01\rangle\!-\!|10\rangle$ splitting (GHz)")
    ax_j.set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
    ax_j.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(outfile, format="pdf")
    plt.close(fig)
    return zetas, exchanges


def plot_three_mode_energy_levels(
    outfile: str = "three_mode_energy_levels.pdf",
    n_show: int = 48,
    annotate_n: int = 10,
    title: str | None = "Three-mode spectrum (qubit-coupler-qubit)",
    **ham_kwargs,
) -> np.ndarray:
    """Diagonalize :func:`three_mode_hamiltonian` and plot the lowest ``n_show`` energies."""
    H = three_mode_hamiltonian(**ham_kwargs)
    return plot_energy_levels(
        lambda: H,
        n_show=n_show,
        outfile=outfile,
        title=title,
        energy_unit="GHz",
        annotate_n=annotate_n,
    )


def plot_three_mode_energy_levels_vs_flux(
    flux_values: np.ndarray,
    *,
    wc0: float,
    A: float,
    outfile: str = "three_mode_energy_levels_vs_flux.pdf",
    n_show: int = 24,
    track_by_overlap: bool = True,
    subtract_ground: bool = False,
    title: str | None = None,
    **ham_kwargs,
) -> np.ndarray:
    """Spectrum vs flux with coupler ``w_c = wc0 + A cos(2π φ)``, same as :func:`propagate_piecewise`.

    Uses overlap tracking in :func:`toolkit.plotting.plot_energy_levels_vs_flux` so levels stay
    visually continuous through avoided crossings.

    ``ham_kwargs`` are passed to :func:`three_mode_hamiltonian` except ``w_c``, which is set from
    ``wc0``, ``A``, and ``φ`` (flux in units of Φ/Φ₀).
    """
    flux_values = np.asarray(flux_values, dtype=float)

    def hamiltonian_at_flux(phi: np.ndarray | float) -> np.ndarray:
        phi_arr = np.asarray(phi, dtype=float)
        if phi_arr.ndim == 0:
            wc = float(wc0 + A * np.cos(2 * np.pi * phi_arr))
            return three_mode_hamiltonian(
                ham_kwargs["w_1"],
                wc,
                ham_kwargs["w_2"],
                ham_kwargs["alpha_1"],
                ham_kwargs["alpha_c"],
                ham_kwargs["alpha_2"],
                ham_kwargs["g_1c"],
                ham_kwargs["g_2c"],
                ham_kwargs["nlevels_qubit"],
                ham_kwargs["nlevels_coupler"],
            )
        phi_arr = phi_arr.ravel()
        wc_arr = wc0 + A * np.cos(2 * np.pi * phi_arr)
        mats = [
            three_mode_hamiltonian(
                ham_kwargs["w_1"],
                float(wc_arr[i]),
                ham_kwargs["w_2"],
                ham_kwargs["alpha_1"],
                ham_kwargs["alpha_c"],
                ham_kwargs["alpha_2"],
                ham_kwargs["g_1c"],
                ham_kwargs["g_2c"],
                ham_kwargs["nlevels_qubit"],
                ham_kwargs["nlevels_coupler"],
            )
            for i in range(phi_arr.shape[0])
        ]
        return np.stack(mats, axis=0)

    if title is None:
        title = "Three-mode spectrum vs flux (coupler modulation)"

    return plot_energy_levels_vs_flux(
        flux_values,
        hamiltonian_at_flux,
        n_show=n_show,
        track_by_overlap=track_by_overlap,
        subtract_ground=subtract_ground,
        outfile=outfile,
        title=title,
        energy_unit="GHz",
    )


def debug_model1_vs_model2(
    *,
    flux: float = 0.0,
    w_O: float = 5.0,
    w_A: float = 0.0,
    J_O: float = 0.0,
    J_A: float = 0.0,
    zeta_O: float = 0.0,
    zeta_A: float = 0.0,
    wc0: float = 5.0,
    A: float = 0.0,
    **ham_kwargs,
) -> None:
    """Print why spectra differ and compare ``H_eff`` to the three-mode computational ``4×4`` block.

    Model 1 and model 2 use **different Hamiltonians** and **different parameter sets**. Nothing in
    this repo derives ``(w_O, J_O, zeta_O, …)`` from ``(w_1, w_c, g_1c, \\alpha, …)``, so the
    matrices generally do not coincide and eigenvalues need not match—even at the same flux.
    """
    m1 = _import_model1_heff()
    phi = float(flux)

    w1_heff = m1.w_vs_flux(w_O, w_A, phi)
    w2_heff = m1.w_vs_flux(w_O, w_A, phi)
    Jv = m1.J_vs_flux(J_O, J_A, phi)
    zv = m1.zeta_vs_flux(zeta_O, zeta_A, phi)
    H1 = np.asarray(m1.heff(w1_heff, w2_heff, Jv, zv), dtype=complex)
    while H1.ndim > 2:
        H1 = H1[0]
    H1 = heff_spin_to_lab_hamiltonian(H1, w1_heff, w2_heff)

    nq = int(ham_kwargs["nlevels_qubit"])
    nc = int(ham_kwargs["nlevels_coupler"])
    wc = float(wc0 + A * np.cos(2 * np.pi * phi))
    H2 = three_mode_hamiltonian(
        ham_kwargs["w_1"],
        wc,
        ham_kwargs["w_2"],
        ham_kwargs["alpha_1"],
        ham_kwargs["alpha_c"],
        ham_kwargs["alpha_2"],
        ham_kwargs["g_1c"],
        ham_kwargs["g_2c"],
        nq,
        nc,
    )
    idx = computational_state_indices(nq, nc)
    H_comp = H2[np.ix_(idx, idx)]
    H_comp_h = 0.5 * (H_comp + H_comp.conj().T)

    e1 = np.linalg.eigvalsh(H1.real)
    e_comp = np.linalg.eigvalsh(H_comp_h.real)
    e2_full = np.linalg.eigvalsh(np.real(H2))

    fro = np.linalg.norm(H_comp - H1, ord="fro")

    print("\n=== model1 vs model2 mismatch (debug) ===\n")
    print(
        "1) Independent parameters: heff uses (w_O,w_A,J_O,J_A,zeta_O,zeta_A); "
        "three-mode uses (w_1,w_2,w_c,alpha_*,g_1c,g_2c,nlevels_*). "
        "They are not converted into each other here.\n"
    )
    print(
        f"2) Hilbert space: H_eff is 4×4; three-mode is {H2.shape[0]}×{H2.shape[0]} "
        f"({nq}×{nc}×{nq}). Comparing lowest four eigenvalues of the **full** three-mode H "
        "mixes coupler excitations with the qubit manifold; the 4×4 block is only the "
        "bare |n_1,0_c,n_2⟩ corner.\n"
    )
    print(f"3) At flux φ={phi}: w_c(three-mode)={wc:.6g} GHz (wc0={wc0}, A={A}).")
    print(
        f"   heff scalar freqs: w1=w2={float(np.asarray(w1_heff).reshape(())):.6g} GHz "
        f"(w_O={w_O}, w_A={w_A}); J={float(np.asarray(Jv).reshape(())):.6g}, "
        f"ζ={float(np.asarray(zv).reshape(())):.6g}.\n"
    )
    print("4) H_eff → lab ω·n convention (real part, GHz):")
    print(np.array2string(np.round(H1.real, 6), prefix="   "))
    print("\n5) Three-mode H on computational {|00⟩,|01⟩,|10⟩,|11⟩} subspace (n_c=0), GHz:")
    print(np.array2string(np.round(H_comp_h.real, 6), prefix="   "))
    print(f"\n6) Frobenius ‖H_comp − H_eff(lab)‖_F = {fro:.6g} GHz (same basis |q1 q2⟩ order).\n")
    print(f"7) Energy levels at φ={phi} (GHz, ascending):")
    print("   model 1 — H_eff lab frame (all 4):")
    print(f"      E = {np.array2string(np.round(e1, 6), separator=', ')}")
    print(f"      rel. ground = {np.array2string(np.round(e1 - e1[0], 6), separator=', ')}")
    print("   model 2 — full three-mode spectrum:")
    print(f"      E = {np.array2string(np.round(e2_full, 6), separator=', ')}")
    print(f"      rel. ground = {np.array2string(np.round(e2_full - e2_full[0], 6), separator=', ')}")
    print("   model 2 — {|00⟩,|01⟩,|10⟩,|11⟩} block only (n_c=0):")
    print(f"      E = {np.array2string(np.round(e_comp, 6), separator=', ')}")
    print(f"      rel. ground = {np.array2string(np.round(e_comp - e_comp[0], 6), separator=', ')}")
    print("   lowest 4 eigenvalues of full H2 (same as first four in list above):")
    print(f"      E = {np.array2string(np.round(e2_full[:4], 6), separator=', ')}")
    print("   (Comparison plot uses tracked lowest-4 of full H2 vs all H_eff levels.)\n")


def plot_compare_model1_model2_vs_flux(
    flux_values: np.ndarray,
    *,
    outfile: str = "model1_vs_model2_energy_vs_flux.pdf",
    subtract_ground: bool = True,
    title: str | None = None,
    n_levels: int = 4,
    verbose: bool = False,
    w_O: float = 5.0,
    w_A: float = 0.0,
    J_O: float = 0.0,
    J_A: float = 0.0,
    zeta_O: float = 0.0,
    zeta_A: float = 0.0,
    wc0: float = 5.0,
    A: float = 0.0,
    **ham_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay lowest tracked levels: model 1 ``H_eff`` vs model 2 three-mode.

    Model 1 uses :func:`model1.heff.heff` with flux-modulated ``w``, ``J``, ``zeta``, then
    :func:`heff_spin_to_lab_hamiltonian` so single-qubit diagonals match ``ω n`` (not ``(ω/2)σ_z``).
    Model 2 uses :func:`three_mode_hamiltonian` with ``w_c(\\phi) = wc0 + A\\cos(2\\pi\\phi)``.

    **Why the count is often 4:** ``H_eff`` is a fixed **4×4** two-qubit Hamiltonian, so there
    are only **four** eigenvalues. Model 2 can have ``n_q^2 n_c`` levels; we plot
    ``n_track = min(n_levels, 4, dim_2)`` so both curves use the same number of lines.
    Use ``verbose=True`` to print this cap to stderr.

    Flux dependence in model 2 **only** comes from ``A`` (and from model 1 from ``w_A``,
    ``J_A``, ``zeta_A``). Parameters are **not** fitted to each other.

    Returns ``(evals_m1, evals_m2)`` with shape ``(n_flux, n_track)``, after optional ground subtraction.
    """
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    m1 = _import_model1_heff()

    dim_heff = 4
    nq = int(ham_kwargs["nlevels_qubit"])
    nc = int(ham_kwargs["nlevels_coupler"])
    dim_three = nq * nc * nq
    n_track = min(int(n_levels), dim_heff, dim_three)
    if verbose:
        print(
            "[plot_compare_model1_model2_vs_flux] "
            f"n_levels requested={n_levels} → n_track={n_track} "
            f"(model1 dim={dim_heff} [H_eff is 4×4], "
            f"model2 dim={dim_three} [= {nq}×{nc}×{nq}])",
            flush=True,
        )

    w1f = m1.w_vs_flux(w_O, w_A, flux_values)
    w2f = m1.w_vs_flux(w_O, w_A, flux_values)
    jf = m1.J_vs_flux(J_O, J_A, flux_values)
    zf = m1.zeta_vs_flux(zeta_O, zeta_A, flux_values)
    H1_raw = np.asarray(m1.heff(w1f, w2f, jf, zf), dtype=complex)
    H1 = heff_spin_to_lab_hamiltonian(H1_raw, w1f, w2f)
    evals1 = track_energy_levels_stack(H1, n_track)

    wc_arr = wc0 + A * np.cos(2 * np.pi * flux_values)
    mats = [
        three_mode_hamiltonian(
            ham_kwargs["w_1"],
            float(wc_arr[i]),
            ham_kwargs["w_2"],
            ham_kwargs["alpha_1"],
            ham_kwargs["alpha_c"],
            ham_kwargs["alpha_2"],
            ham_kwargs["g_1c"],
            ham_kwargs["g_2c"],
            nq,
            nc,
        )
        for i in range(flux_values.shape[0])
    ]
    H2 = np.stack(mats, axis=0)
    evals2 = track_energy_levels_stack(H2, n_track)

    if subtract_ground:
        evals1 = evals1 - evals1[:, :1]
        evals2 = evals2 - evals2[:, :1]

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    if n_track <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_track, endpoint=False))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_track, 20), endpoint=False))
    for i in range(n_track):
        c = colors[i % len(colors)]
        ax.plot(
            flux_values,
            evals1[:, i],
            color=c,
            linestyle="-",
            linewidth=1.8,
            label=rf"model 1 $E_{{{i}}}$",
        )
        ax.plot(
            flux_values,
            evals2[:, i],
            color=c,
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
            label=rf"model 2 $E_{{{i}}}$",
        )

    ax.set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
    if subtract_ground:
        ax.set_ylabel(r"Energy (GHz, rel. ground)")
    else:
        ax.set_ylabel("Energy (GHz)")
    ax.set_title(
        title
        or rf"Lowest {n_track} levels: $H_\mathrm{{eff}}$ (solid) vs three-mode (dashed)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize="small")
    fig.tight_layout()
    plt.savefig(outfile, format="pdf")
    plt.close(fig)
    return evals1, evals2


def propagate_piecewise(psi0, tlist, flux_values, params):
    psi = psi0.copy()
    states = [psi.copy()]
    for k in range(len(tlist) - 1):
        dt = tlist[k+1] - tlist[k]
        wc = params["wc0"] + params["A"] * np.cos(2*np.pi*flux_values[k])
        H = three_mode_hamiltonian(
            params["w1"], wc, params["w2"],
            params["a1"], params["ac"], params["a2"],
            params["g1c"], params["g2c"],
            nlevels_qubit=params["n1"],
            nlevels_coupler=params["nc"],
        )
        U = expm(-1j * H * dt)
        psi = U @ psi
        states.append(psi.copy())
    return np.array(states)


if __name__ == "__main__":
    _dir = Path(__file__).resolve().parent
    _common = dict(
        w_1=5.0,
        w_2=5.0,
        alpha_1=-0.0,
        alpha_c=-0.0,
        alpha_2=-0.0,
        g_1c=0.00,
        g_2c=0.00,
        nlevels_qubit=2,
        nlevels_coupler=1,
    )
    plot_three_mode_energy_levels(
        outfile=str(_dir / "three_mode_energy_levels.pdf"),
        w_c=5.2,
        **_common,
    )
    flux = np.linspace(0.0, 1.0, 80)
    plot_three_mode_energy_levels_vs_flux(
        flux,
        wc0=5.2,
        A=0.25,
        outfile=str(_dir / "three_mode_energy_levels_vs_flux.pdf"),
        n_show=16,
        **_common,
    )
    plot_three_mode_zz_exchange_vs_flux(
        flux,
        wc0=5.2,
        A=0.25,
        outfile=str(_dir / "three_mode_ZZ_exchange_vs_flux.pdf"),
        **_common,
    )
    # Match `w_O` / `wc0` and set ``A`` / ``w_A``… to zero for a flux-static check; increase ``A``
    # when you want coupler modulation like the other plots below.
    debug_model1_vs_model2(
        flux=0.0,
        w_O=5.0,
        wc0=5.0,
        A=0.0,
        **_common,
    )
    plot_compare_model1_model2_vs_flux(
        flux,
        outfile=str(_dir / "model1_vs_model2_energy_vs_flux.pdf"),
        subtract_ground=True,
        verbose=True,
        w_O=5.0,
        wc0=5.0,
        A=0.0,
        **_common,
    )