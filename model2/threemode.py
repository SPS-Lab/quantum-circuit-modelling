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

from toolkit.helpers import destroy
from toolkit.plotting import plot_energy_levels, plot_energy_levels_vs_flux

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
        alpha_1=-0.2,
        alpha_c=-0.15,
        alpha_2=-0.2,
        g_1c=0.08,
        g_2c=0.08,
        nlevels_qubit=4,
        nlevels_coupler=5,
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