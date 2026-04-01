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
    """Print diagnostics comparing model-1 ``H_eff`` and model-2 computational subspace."""
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
    wc = float(coupler_frequency(wc0, A, phi))
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
        f"2) Hilbert space: H_eff is 4x4; three-mode is {H2.shape[0]}x{H2.shape[0]} "
        f"({nq}x{nc}x{nq}). Comparing lowest four eigenvalues of the full three-mode H "
        "mixes coupler excitations with the qubit manifold; the 4x4 block is only the "
        "bare |n_1,0_c,n_2> corner.\n"
    )
    print(f"3) At flux phi={phi}: w_c(three-mode)={wc:.6g} GHz (wc0={wc0}, A={A}).")
    print(
        f"   heff scalar freqs: w1=w2={float(np.asarray(w1_heff).reshape(())):.6g} GHz "
        f"(w_O={w_O}, w_A={w_A}); J={float(np.asarray(Jv).reshape(())):.6g}, "
        f"zeta={float(np.asarray(zv).reshape(())):.6g}.\n"
    )
    print("4) H_eff -> lab w*n convention (real part, GHz):")
    print(np.array2string(np.round(H1.real, 6), prefix="   "))
    print("\n5) Three-mode H on computational {|00>,|01>,|10>,|11>} subspace (n_c=0), GHz:")
    print(np.array2string(np.round(H_comp_h.real, 6), prefix="   "))
    print(f"\n6) Frobenius ||H_comp - H_eff(lab)||_F = {fro:.6g} GHz (same basis |q1 q2> order).\n")
    print(f"7) Energy levels at phi={phi} (GHz, ascending):")
    print("   model 1 - H_eff lab frame (all 4):")
    print(f"      E = {np.array2string(np.round(e1, 6), separator=', ')}")
    print(f"      rel. ground = {np.array2string(np.round(e1 - e1[0], 6), separator=', ')}")
    print("   model 2 - full three-mode spectrum:")
    print(f"      E = {np.array2string(np.round(e2_full, 6), separator=', ')}")
    print(f"      rel. ground = {np.array2string(np.round(e2_full - e2_full[0], 6), separator=', ')}")
    print("   model 2 - {|00>,|01>,|10>,|11>} block only (n_c=0):")
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
    """Overlay tracked low-energy levels: model-1 ``H_eff`` vs model-2 three-mode."""
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
