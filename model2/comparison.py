"""Model-1 bridge utilities and model-1 vs model-2 comparisons."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from toolkit.helpers import I2, pz
from toolkit.spectrum import track_energy_levels_stack

from model2.analysis import model1_exchange_and_zz_from_eigenvalues
from model2.core import computational_state_indices, coupler_frequency, three_mode_hamiltonian

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
    w_O: float,
    w_A: float,
    H1_0: np.ndarray,
    H2_0: np.ndarray,
    nlevels_qubit: int,
    nlevels_coupler: int,
) -> None:
    """Print a short first-flux comparison snapshot."""
    idx = computational_state_indices(nlevels_qubit, nlevels_coupler)
    H_comp = H2_0[np.ix_(idx, idx)]
    H_comp_h = 0.5 * (H_comp + H_comp.conj().T)
    fro = np.linalg.norm(H_comp - H1_0, ord="fro")

    e1 = np.linalg.eigvalsh(H1_0.real)
    e2_full = np.linalg.eigvalsh(H2_0.real)
    e2_comp = np.linalg.eigvalsh(H_comp_h.real)
    wc_flux0 = float(coupler_frequency(wc0, A, flux0))
    wq_flux0 = float(w_O + w_A * np.cos(2 * np.pi * flux0))

    print(
        "[plot_compare_model1_model2_vs_flux] snapshot "
        f"phi0={flux0:.6g}: wq={wq_flux0:.6g} GHz, wc={wc_flux0:.6g} GHz, "
        f"||H_comp-H_eff||_F={fro:.6g}",
        flush=True,
    )
    print(
        "[plot_compare_model1_model2_vs_flux] "
        f"E_model1={np.array2string(np.round(e1, 6), separator=', ')} "
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
    """Overlay tracked low-energy levels: model-1 ``H_eff`` vs model-2 three-mode.

    When ``verbose=True``, prints compact setup details and a first-flux mismatch snapshot.
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
            f"n_levels requested={n_levels} -> n_track={n_track} "
            f"(model1 dim={dim_heff} [H_eff is 4x4], "
            f"model2 dim={dim_three} [= {nq}x{nc}x{nq}])",
            f"alpha_1={ham_kwargs['alpha_1']}, alpha_c={ham_kwargs['alpha_c']}, alpha_2={ham_kwargs['alpha_2']}",
            flush=True,
        )

    w1f = m1.w_vs_flux(w_O, w_A, flux_values)
    w2f = m1.w_vs_flux(w_O, w_A, flux_values)

    jf_seed = m1.J_vs_flux(J_O, J_A, flux_values)
    zf_seed = m1.zeta_vs_flux(zeta_O, zeta_A, flux_values)
    H1_seed_raw = np.asarray(m1.heff(w1f, w2f, jf_seed, zf_seed), dtype=complex)
    H1_seed = heff_spin_to_lab_hamiltonian(H1_seed_raw, w1f, w2f)
    j_abs_from_evals, zeta_from_evals = model1_exchange_and_zz_from_eigenvalues(H1_seed, w1f, w2f)
    cosf = np.cos(2 * np.pi * flux_values)
    j_sign = -1.0 if J_O < 0 else 1.0
    J_O = float(np.mean(j_sign * j_abs_from_evals - J_A * cosf))
    zeta_O = float(np.mean(zeta_from_evals - zeta_A * cosf))
    if verbose:
        print(
            "[plot_compare_model1_model2_vs_flux] "
            f"derived from model1 energies: J_O={J_O:.9g}, zeta_O={zeta_O:.9g}",
            flush=True,
        )

    jf = m1.J_vs_flux(J_O, J_A, flux_values)
    zf = m1.zeta_vs_flux(zeta_O, zeta_A, flux_values)
    H1_raw = np.asarray(m1.heff(w1f, w2f, jf, zf), dtype=complex)
    H1 = heff_spin_to_lab_hamiltonian(H1_raw, w1f, w2f)
    evals1 = track_energy_levels_stack(H1, n_track)

    wc_arr = coupler_frequency(wc0, A, flux_values)
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

    if verbose and flux_values.size > 0:
        _print_compact_debug_snapshot(
            flux0=float(flux_values[0]),
            wc0=wc0,
            A=A,
            w_O=w_O,
            w_A=w_A,
            H1_0=H1[0],
            H2_0=H2[0],
            nlevels_qubit=nq,
            nlevels_coupler=nc,
        )

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
