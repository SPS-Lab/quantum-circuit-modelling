"""Model-1 bridge utilities and model-1 vs model-2 comparisons."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from toolkit.helpers import I2, pz
from toolkit.spectrum import track_energy_levels_stack

from model2.core import (
    computational_state_indices,
    computational_subspace_block,
    coupler_frequency,
    three_mode_hamiltonian_stack_vs_flux,
)

# Repo root (parent of model2/) so `toolkit` / `model1` resolve when run from model2/.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _import_model1_heff():
    """Load ``model1.heff`` by path (``model1`` is not necessarily a package)."""
    path = _ROOT / "model1" / "heff.py"
    spec = importlib.util.spec_from_file_location("model1_heff", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def heff_spin_to_lab_hamiltonian(H_eff: np.ndarray, w1, w2) -> np.ndarray:
    """Convert model-1 ``(w/2) sigma_z`` convention to lab-frame ``w n`` convention."""
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


def _print_compact_debug_snapshot(
    *,
    flux0: float,
    wc0: float,
    A: float,
    w1_0: float,
    w2_0: float,
    j_0: float,
    zeta_0: float,
    H1_0: np.ndarray,
    H2_eff_0: np.ndarray,
    H2_0: np.ndarray,
    nlevels_qubit: int,
    nlevels_coupler: int,
) -> None:
    """Print a short first-flux comparison snapshot."""
    H_comp_h = computational_subspace_block(
        H2_0,
        nlevels_qubit,
        nlevels_coupler,
        hermitianize=True,
    )
    fro = np.linalg.norm(H2_eff_0 - H1_0, ord="fro")

    e1 = np.linalg.eigvalsh(np.real(H1_0))
    e2_eff = np.linalg.eigvalsh(np.real(H2_eff_0))
    e2_full = np.linalg.eigvalsh(np.real(H2_0))
    e2_comp = np.linalg.eigvalsh(np.real(H_comp_h))
    wc_flux0 = float(coupler_frequency(wc0, A, flux0))

    print(
        "[plot_compare_model1_model2_vs_flux] model2 computational 4x4 block at phi0:\n"
        f"{np.array2string(np.round(H_comp_h, 6), separator=', ')}",
        flush=True,
    )
    print(
        "[plot_compare_model1_model2_vs_flux] snapshot "
        f"phi0={flux0:.6g}: w1={w1_0:.6g} GHz, w2={w2_0:.6g} GHz, "
        f"J={j_0:.6g} GHz, zeta={zeta_0:.6g} GHz, wc={wc_flux0:.6g} GHz, "
        f"||H2_dressed_eff-H_eff||_F={fro:.6g}",
        flush=True,
    )
    print(
        "[plot_compare_model1_model2_vs_flux] "
        f"E_model1={np.array2string(np.round(e1, 6), separator=', ')} "
        f"E_model2_dressed_comp={np.array2string(np.round(e2_eff, 6), separator=', ')} "
        f"E_model2_comp={np.array2string(np.round(e2_comp, 6), separator=', ')} "
        f"E_model2_full_low4={np.array2string(np.round(e2_full[:4], 6), separator=', ')}",
        flush=True,
    )


def plot_compare_model1_model2_vs_flux(
    flux_values: np.ndarray,
    *,
    outfile: str = "model1_vs_model2_energy_vs_flux.pdf",
    subtract_ground: bool = True,
    title: str | None = None,
    verbose: bool = False,
    n_candidate_states: int = 16,
    wc0: float = 5.0,
    A: float = 0.0,
    **ham_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay tracked levels: model-1 ``H_eff`` vs a dressed model-2 computational effective ``4x4``.

    At each flux, this diagonalizes full model-2 ``H`` and overlap-matches dressed states to
    ``|00>``, ``|01>``, ``|10>``, ``|11>``. A Löwdin-orthonormalized effective ``4x4`` is formed
    from those dressed eigenpairs, then ``(w1, w2, J, zeta)`` are extracted and fed to model 1.
    """
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    m1 = _import_model1_heff()

    dim_heff = 4
    nq = int(ham_kwargs["nlevels_qubit"])
    nc = int(ham_kwargs["nlevels_coupler"])
    dim_three = nq * nc * nq
    n_track = min(dim_heff, dim_three)
    if verbose:
        print(
            "[plot_compare_model1_model2_vs_flux] "
            f"n_track={n_track} "
            f"(model1 dim={dim_heff} [H_eff is 4x4], "
            f"model2 dim={dim_three} [= {nq}x{nc}x{nq}])",
            f"alpha_1={ham_kwargs['alpha_1']}, alpha_c={ham_kwargs['alpha_c']}, alpha_2={ham_kwargs['alpha_2']}",
            flush=True,
        )

    H2 = three_mode_hamiltonian_stack_vs_flux(
        flux_values,
        wc0=wc0,
        A=A,
        ham_kwargs=ham_kwargs,
    )

    comp_idx = computational_state_indices(nq, nc)
    d_full = H2.shape[1]
    n_cand = max(4, min(int(n_candidate_states), d_full))
    H2_eff = np.empty((flux_values.shape[0], 4, 4), dtype=complex)
    for k in range(flux_values.shape[0]):
        evals_full, evecs_full = np.linalg.eigh(H2[k])
        overlap = np.abs(evecs_full[comp_idx, :n_cand]) ** 2
        row_ind, col_ind = linear_sum_assignment(-overlap)

        evals_comp = np.empty(4, dtype=float)
        comp_components = np.empty((4, 4), dtype=complex)
        for t in range(4):
            r = int(row_ind[t])
            c = int(col_ind[t])
            evals_comp[r] = float(evals_full[c])
            comp_components[:, r] = evecs_full[comp_idx, c]

        gram = comp_components.conj().T @ comp_components
        gram_evals, gram_vecs = np.linalg.eigh(gram)
        gram_evals = np.clip(gram_evals, 1e-15, None)
        gram_inv_sqrt = gram_vecs @ np.diag(1.0 / np.sqrt(gram_evals)) @ gram_vecs.conj().T
        dressed_basis = comp_components @ gram_inv_sqrt
        heff2 = dressed_basis @ np.diag(evals_comp) @ dressed_basis.conj().T
        H2_eff[k] = 0.5 * (heff2 + heff2.conj().T)

    evals2 = track_energy_levels_stack(H2_eff, n_track)

    d00 = np.real(H2_eff[:, 0, 0])
    d01 = np.real(H2_eff[:, 1, 1])
    d10 = np.real(H2_eff[:, 2, 2])
    d11 = np.real(H2_eff[:, 3, 3])
    zeta = d11 - d10 - d01 + d00
    w1f = d10 - d00 + 0.5 * zeta
    w2f = d01 - d00 + 0.5 * zeta
    jf = 0.5 * np.real(H2_eff[:, 1, 2])

    max_j_imag = float(np.max(np.abs(np.imag(H2_eff[:, 1, 2]))))
    if verbose:
        print(
            "[plot_compare_model1_model2_vs_flux] "
            f"derived per-flux from full-H2 dressed states/energies; max imag(H01,10)={max_j_imag:.3e}",
            flush=True,
        )

    H1_raw = np.asarray(m1.heff(w1f, w2f, jf, zeta), dtype=complex)
    H1 = heff_spin_to_lab_hamiltonian(H1_raw, w1f, w2f)
    evals1 = track_energy_levels_stack(H1, n_track)

    if verbose and flux_values.size > 0:
        _print_compact_debug_snapshot(
            flux0=float(flux_values[0]),
            wc0=wc0,
            A=A,
            w1_0=float(w1f[0]),
            w2_0=float(w2f[0]),
            j_0=float(jf[0]),
            zeta_0=float(zeta[0]),
            H1_0=H1[0],
            H2_eff_0=H2_eff[0],
            H2_0=H2[0],
            nlevels_qubit=nq,
            nlevels_coupler=nc,
        )

    if subtract_ground:
        evals1 = evals1 - evals1[:, :1]
        evals2 = evals2 - evals2[:, :1]

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    if n_track <= 10:
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, n_track, endpoint=False))
    else:
        cmap = plt.get_cmap("tab20")
        colors = cmap(np.linspace(0, 1, min(n_track, 20), endpoint=False))

    for i in range(n_track):
        c = colors[i % len(colors)]
        c_rgb = np.asarray(c[:3], dtype=float)
        c_dash = tuple(np.clip(0.55 * c_rgb, 0.0, 1.0))
        ax.plot(
            flux_values,
            evals1[:, i],
            color=c,
            linestyle="-",
            linewidth=1.8,
            alpha=0.9,
            zorder=2,
            label=rf"model 1 $E_{{{i}}}$",
        )
        ax.plot(
            flux_values,
            evals2[:, i],
            color=c_dash,
            linestyle="--",
            linewidth=1.4,
            alpha=1.0,
            zorder=3,
            label=rf"model 2 $E_{{{i}}}$",
        )

    ax.set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
    ax.set_ylabel(r"Energy (GHz, rel. ground)" if subtract_ground else "Energy (GHz)")
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
