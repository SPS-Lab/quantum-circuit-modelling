"""Model-1 bridge utilities and model-1 vs model-2 comparisons."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Mapping, Unpack

import matplotlib.pyplot as plt
import numpy as np

from toolkit.helpers import I2, pz
from toolkit.spectrum import track_energy_levels_stack

from model2.core import computational_subspace_block, coupler_frequency, three_mode_hamiltonian_stack_vs_flux
from model2.effective import (
    build_dressed_effective_computational_stack,
    extract_model1_parameters_from_4x4_stack,
)
from model2.hamiltonian_types import ThreeModeHamiltonianCommonKwargs

# Repo root (parent of model2/) so `toolkit` / `model1` resolve when run from model2/.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_Model1FluxParamSpec = float | np.ndarray | Callable[[np.ndarray], np.ndarray]


def _resolve_flux_param(
    flux_values: np.ndarray,
    spec: _Model1FluxParamSpec,
    *,
    name: str,
) -> np.ndarray:
    """Resolve a scalar / 1D array / callable flux parameter to shape ``(n_flux,)``."""
    values = spec(flux_values) if callable(spec) else spec
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(flux_values.shape[0], float(arr), dtype=float)
    arr = arr.ravel()
    if arr.size != flux_values.size:
        raise ValueError(
            f"model1 parameter {name!r} must be scalar or length n_flux={flux_values.size}, "
            f"got shape {arr.shape}"
        )
    return arr.astype(float, copy=False)


def _resolve_model1_params(
    flux_values: np.ndarray,
    model1_params: Mapping[str, _Model1FluxParamSpec],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve user-provided model-1 flux parameters (w1, w2, J, zeta)."""
    allowed_aliases = {
        "w1": ("w1", "w_1"),
        "w2": ("w2", "w_2"),
        "J": ("J", "j"),
        "zeta": ("zeta",),
    }
    allowed_keys = {k for aliases in allowed_aliases.values() for k in aliases}
    unknown = sorted(set(model1_params.keys()) - allowed_keys)
    if unknown:
        raise ValueError(
            "Unknown keys in model1_params: "
            f"{unknown}. Allowed keys: {sorted(allowed_keys)}"
        )

    resolved: dict[str, np.ndarray] = {}
    for canonical, aliases in allowed_aliases.items():
        matched = [alias for alias in aliases if alias in model1_params]
        if len(matched) == 0:
            raise ValueError(
                f"Missing model1 parameter {canonical!r} in model1_params. "
                f"Accepted key(s): {aliases}"
            )
        if len(matched) > 1:
            raise ValueError(
                f"Multiple aliases provided for {canonical!r}: {matched}. "
                "Use only one."
            )
        resolved[canonical] = _resolve_flux_param(
            flux_values,
            model1_params[matched[0]],
            name=canonical,
        )
    return resolved["w1"], resolved["w2"], resolved["J"], resolved["zeta"]


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
        "[plot_compare_model1_model2_vs_flux] full model2 at phi0:\n"
        f"{np.round(H2_0, 6)}",
        flush=True,
    )
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
    plot_eff2_levels: bool = True,
    n_model2_levels: int | None = None,
    dressed_selection_mode: str = "continuous",
    model1_params: Mapping[str, _Model1FluxParamSpec] | None = None,
    **ham_kwargs: Unpack[ThreeModeHamiltonianCommonKwargs],
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay tracked levels: model-1 ``H_eff`` vs a dressed model-2 computational effective ``4x4``.

    At each flux, this diagonalizes full model-2 ``H`` and overlap-matches dressed states to
    ``|00>``, ``|01>``, ``|10>``, ``|11>``. A Löwdin-orthonormalized effective ``4x4`` is formed
    from those dressed eigenpairs, then ``(w1, w2, J, zeta)`` are extracted and fed to model 1.
    To compare against an independently specified effective model-1 sweep, pass ``model1_params``
    with keys ``w1, w2, J, zeta`` (alias keys ``w_1, w_2, j`` also accepted), each as a scalar,
    1D array of length ``n_flux``, or callable ``f(flux_values) -> array``.
    Optionally, additional model-2-only levels can be plotted from the full three-mode Hamiltonian.
    ``dressed_selection_mode`` controls dressed-state selection:
    - ``"continuous"``: continue labels by overlap with previous flux step (stable along sweep)
    - ``"bare"``: match independently at each flux to bare computational states
    """
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    m1 = _import_model1_heff()
    dressed_selection_mode = str(dressed_selection_mode).strip().lower()
    if dressed_selection_mode not in {"continuous", "bare"}:
        raise ValueError(
            "dressed_selection_mode must be one of {'continuous', 'bare'}, "
            f"got {dressed_selection_mode!r}"
        )

    dim_heff = 4
    nq = int(ham_kwargs["nlevels_qubit"])
    nc = int(ham_kwargs["nlevels_coupler"])
    dim_three = nq * nc * nq
    n_track = min(dim_heff, dim_three)
    n_model2_plot = dim_three if n_model2_levels is None else int(n_model2_levels)
    n_model2_plot = max(n_track, min(n_model2_plot, dim_three))
    if verbose:
        print(
            "[plot_compare_model1_model2_vs_flux] "
            f"{n_track=} {n_model2_plot=} "
            f"(model1 {dim_heff=} [H_eff is {dim_heff}x{dim_heff}], "
            f"model2 {dim_three=} [= {nq}x{nc}x{nq}])",
            f"{dressed_selection_mode=}",
            f"alpha_1={ham_kwargs['alpha_1']}, alpha_c={ham_kwargs['alpha_c']}, alpha_2={ham_kwargs['alpha_2']}",
            flush=True,
        )

    H2 = three_mode_hamiltonian_stack_vs_flux(
        flux_values,
        wc0=wc0,
        A=A,
        ham_kwargs=ham_kwargs,
    )

    
    H2_eff = build_dressed_effective_computational_stack(
        H2,
        nlevels_qubit=nq,
        nlevels_coupler=nc,
        n_candidate_states=n_candidate_states,
        selection_mode=dressed_selection_mode,
    )

    
    if plot_eff2_levels:
        evals2 = track_energy_levels_stack(H2_eff, n_track)
    else:
        evals2 = track_energy_levels_stack(H2, n_track)
    evals2_full_plot = track_energy_levels_stack(H2, n_model2_plot)

    if model1_params is None:
        params = extract_model1_parameters_from_4x4_stack(H2_eff)
        w1f = params["w1"]
        w2f = params["w2"]
        jf = params["J"]
        zeta = params["zeta"]

        max_j_imag = float(np.max(np.abs(np.imag(H2_eff[:, 1, 2]))))
        if verbose:
            print(
                "[plot_compare_model1_model2_vs_flux] "
                f"derived model1 params per flux from model2 dressed states; "
                f"max imag(H01,10)={max_j_imag:.3e}",
                flush=True,
            )
    else:
        w1f, w2f, jf, zeta = _resolve_model1_params(flux_values, model1_params)
        if verbose:
            print(
                "[plot_compare_model1_model2_vs_flux] "
                "using user-provided model1_params (independent of model2 per flux)",
                flush=True,
            )

    H1_raw = np.asarray(m1.heff(w1f, w2f, jf, zeta), dtype=complex)
    H1 = heff_spin_to_lab_hamiltonian(H1_raw, w1f, w2f)
    evals1 = track_energy_levels_stack(H1, n_track)

    if verbose and flux_values.size > 0:
        flux_ind = 0
        print(
            "[plot_compare_model1_model2_vs_flux] snapshot at first flux point "
            f"phi0={flux_values[flux_ind]:.6g}:",
        )
        print(            "[plot_compare_model1_model2_vs_flux] model1 H_eff:\n"
            f"{np.round(H1[flux_ind], 6)}",
            flush=True,
        )
        print(
            "[plot_compare_model1_model2_vs_flux] model2 dressed effective H at phi0:\n"
            f"{np.round(H2_eff[flux_ind], 6)}",
            flush=True,
        )
        #_print_compact_debug_snapshot(
        #    flux0=float(flux_values[flux_ind]),
        #    wc0=wc0,
        #    A=A,
        #    w1_0=float(w1f[flux_ind]),
        #    w2_0=float(w2f[flux_ind]),
        #    j_0=float(jf[flux_ind]),
        #    zeta_0=float(zeta[flux_ind]),
        #    H1_0=H1[flux_ind],
        #    H2_eff_0=H2_eff[flux_ind],
        #    H2_0=H2[flux_ind],
        #    nlevels_qubit=nq,
        #    nlevels_coupler=nc,
        #)

    if subtract_ground:
        evals1 = evals1 - evals1[:, :1]
        evals2 = evals2 - evals2[:, :1]
        evals2_full_plot = evals2_full_plot - evals2_full_plot[:, :1]

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    n_colors = n_model2_plot
    if n_colors <= 10:
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, n_colors, endpoint=False))
    else:
        cmap = plt.get_cmap("tab20")
        colors = cmap(np.linspace(0, 1, min(n_colors, 20), endpoint=False))

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
    for i in range(n_track, n_model2_plot):
        c = colors[i % len(colors)]
        ax.plot(
            flux_values,
            evals2_full_plot[:, i],
            color=c,
            linestyle=":",
            linewidth=1.1,
            alpha=0.65,
            zorder=1,
            label=(rf"model 2 extra $E_{{{i}}}$"),
        )

    ax.set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
    ax.set_ylabel(r"Energy (GHz, rel. ground)" if subtract_ground else "Energy (GHz)")
    ax.set_title(
        title
        or rf"Lowest {n_track} levels: $H_\mathrm{{eff}}$ (solid) vs three-mode{'-eff' if plot_eff2_levels else '-full'} (dashed); "
        rf"model-2 shown up to level {n_model2_plot - 1}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize="small")
    fig.tight_layout()
    plt.savefig(outfile, format="pdf")
    plt.close(fig)
    return evals1, evals2
