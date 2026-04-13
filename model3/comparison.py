"""Regime-of-validity comparison: model1/model2 against scqubits reference."""

from __future__ import annotations

from collections.abc import Mapping
import sys
from pathlib import Path
from typing import TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np

# Repo root (parent of model3/) so `model1`/`model2` resolve when run from model3/.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model1.heff import heff
from model2.comparison import heff_spin_to_lab_hamiltonian
from model2.core import coupler_frequency, three_mode_hamiltonian_stack_vs_flux
from model2.effective import (
    build_dressed_effective_computational_stack,
    extract_model1_parameters_from_4x4_stack,
)
from model3.scqref import (
    transmon_oscillator_stack_vs_flux,
)
from model3.reference_params import DEFAULT_TRANSMON_KEY, load_transmon_params
from toolkit.spectrum import track_energy_levels_stack


class ScqubitsReferenceConfig(TypedDict, total=False):
    """Config for the model3 scqubits reference."""

    wc0: float
    A: float
    transmon1_params: Mapping[str, float]
    transmon2_params: Mapping[str, float]
    transmon_ng: float
    transmon_ncut: int
    transmon_dim: int
    coupler_dim: int
    g_1c: float
    g_2c: float


def _transmon_w01_alpha_from_ej_ec(
    *,
    ej: float,
    ec: float,
    ng: float,
    ncut: int,
    truncated_dim: int,
) -> tuple[float, float]:
    """Return transmon ``(w01, alpha)`` from EJ/EC via low-lying eigenvalues."""
    try:
        import scqubits as scq
    except Exception as exc:  # pragma: no cover - import guard only
        raise ImportError(
            "scqubits import failed; cannot auto-calibrate model2 from reference transmons."
        ) from exc

    t = scq.Transmon(
        EJ=float(ej),
        EC=float(ec),
        ng=float(ng),
        ncut=int(ncut),
        truncated_dim=int(truncated_dim),
    )
    evals = np.asarray(t.eigenvals(evals_count=3), dtype=float).ravel()
    if evals.size < 3:
        raise ValueError("Need at least 3 transmon levels to extract anharmonicity.")
    w01 = float(evals[1] - evals[0])
    alpha = float((evals[2] - evals[1]) - (evals[1] - evals[0]))
    return w01, alpha


def _resolved_reference_config(
    *,
    wc0: float,
    A: float,
    reference: ScqubitsReferenceConfig | None,
) -> dict[str, object]:
    """Merge user reference config with defaults."""
    cfg: dict[str, object] = {
        "wc0": float(wc0),
        "A": float(A),
        "transmon1_params": None,
        "transmon2_params": None,
        "transmon_ng": 0.0,
        "transmon_ncut": 45,
        "transmon_dim": 6,
        "coupler_dim": 8,
        "g_1c": 0.08,
        "g_2c": 0.08,
    }
    if reference is not None:
        cfg.update(dict(reference))
    cfg["wc0"] = float(cfg["wc0"])
    cfg["A"] = float(cfg["A"])
    cfg["transmon_ng"] = float(cfg["transmon_ng"])
    cfg["transmon_ncut"] = int(cfg["transmon_ncut"])
    cfg["transmon_dim"] = int(cfg["transmon_dim"])
    cfg["coupler_dim"] = int(cfg["coupler_dim"])
    cfg["g_1c"] = float(cfg["g_1c"])
    cfg["g_2c"] = float(cfg["g_2c"])
    return cfg


def _relative_energies(H_stack: np.ndarray, *, n_track: int = 4) -> np.ndarray:
    evals = track_energy_levels_stack(np.asarray(H_stack, dtype=complex), int(n_track))
    return evals - evals[:, :1]


def _per_flux_rmse(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    diff = np.asarray(pred, dtype=float)[:, 1:] - np.asarray(ref, dtype=float)[:, 1:]
    return np.sqrt(np.mean(diff * diff, axis=1))


def _summarize_masked(err: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return float("nan"), float("nan")
    e = np.asarray(err, dtype=float)[mask]
    return float(np.sqrt(np.mean(e * e))), float(np.max(np.abs(e)))


def _fit_single_harmonic(flux_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fit ``x0 + a cos(2πφ) + b sin(2πφ)`` and return predicted values."""
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    theta = 2.0 * np.pi * flux_values
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    beta, *_ = np.linalg.lstsq(X, values, rcond=None)
    return X @ beta


def compare_model1_model2_against_scqubits(
    flux_values: np.ndarray,
    *,
    wc0: float,
    A: float,
    reference: ScqubitsReferenceConfig | None = None,
    model2_nlevels_qubit: int = 3,
    model2_nlevels_coupler: int = 3,
    model2_transmon_ncut: int = 60,
    model2_transmon_dim: int = 12,
    n_candidate_states: int = 16,
    model1_mode: str = "cosine-fit",
    idle_ratio: float = 6.0,
    near_ratio: float = 2.0,
    outfile: str = "model1_model2_vs_scqubits_regime_map.pdf",
    title: str | None = None,
) -> dict[str, object]:
    """Compare model-1/model-2 against a Transmon+Oscillator scqubits reference.

    Model-2 parameters are always calibrated from the reference transmon EJ/EC:
    ``w_1,w_2,alpha_1,alpha_2`` from transmon spectra, ``alpha_c=0``,
    and couplings ``g_1c,g_2c`` copied from reference.
    """
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    ref = _resolved_reference_config(wc0=wc0, A=A, reference=reference)
    nq_ref = int(ref["transmon_dim"])
    nc_ref = int(ref["coupler_dim"])
    ref_wc0 = float(ref["wc0"])
    ref_A = float(ref["A"])
    if ref["transmon1_params"] is None or ref["transmon2_params"] is None:
        raise ValueError(
            "reference.transmon1_params and reference.transmon2_params are required. "
            "Load them from model3/reference_params.json near your main script and pass explicitly."
        )
    p1 = cast(Mapping[str, float], ref["transmon1_params"])
    p2 = cast(Mapping[str, float], ref["transmon2_params"])
    ng = float(ref["transmon_ng"])
    w1, alpha1 = _transmon_w01_alpha_from_ej_ec(
        ej=float(p1["EJ"]),
        ec=float(p1["EC"]),
        ng=ng,
        ncut=int(model2_transmon_ncut),
        truncated_dim=int(model2_transmon_dim),
    )
    w2, alpha2 = _transmon_w01_alpha_from_ej_ec(
        ej=float(p2["EJ"]),
        ec=float(p2["EC"]),
        ng=ng,
        ncut=int(model2_transmon_ncut),
        truncated_dim=int(model2_transmon_dim),
    )
    ham_kwargs_used: dict[str, float | int] = {
        "w_1": float(w1),
        "w_2": float(w2),
        "alpha_1": float(alpha1),
        "alpha_c": 0.0,
        "alpha_2": float(alpha2),
        "g_1c": float(ref["g_1c"]),
        "g_2c": float(ref["g_2c"]),
        "nlevels_qubit": max(int(model2_nlevels_qubit), 3),
        "nlevels_coupler": max(int(model2_nlevels_coupler), 3),
    }
    nq = int(ham_kwargs_used["nlevels_qubit"])
    nc = int(ham_kwargs_used["nlevels_coupler"])

    H2 = three_mode_hamiltonian_stack_vs_flux(
        flux_values,
        wc0=wc0,
        A=A,
        ham_kwargs=ham_kwargs_used,
    )
    H3 = transmon_oscillator_stack_vs_flux(
        flux_values,
        wc0=ref_wc0,
        A=ref_A,
        transmon1_params=ref["transmon1_params"],
        transmon2_params=ref["transmon2_params"],
        transmon_ng=float(ref["transmon_ng"]),
        transmon_ncut=int(ref["transmon_ncut"]),
        transmon_dim=nq_ref,
        coupler_dim=nc_ref,
        g_1c=float(ref["g_1c"]),
        g_2c=float(ref["g_2c"]),
    )

    H2_eff = build_dressed_effective_computational_stack(
        H2,
        nlevels_qubit=nq,
        nlevels_coupler=nc,
        n_candidate_states=n_candidate_states,
        selection_mode="continuous",
    )
    H3_eff = build_dressed_effective_computational_stack(
        H3,
        nlevels_qubit=nq_ref,
        nlevels_coupler=nc_ref,
        n_candidate_states=max(n_candidate_states, 20),
        selection_mode="continuous",
    )

    p2 = extract_model1_parameters_from_4x4_stack(H2_eff)
    p3 = extract_model1_parameters_from_4x4_stack(H3_eff)

    mode = str(model1_mode).strip().lower()
    if mode == "from-model2-per-flux":
        p1 = p2
    elif mode == "cosine-fit":
        p1 = {
            "w1": _fit_single_harmonic(flux_values, p2["w1"]),
            "w2": _fit_single_harmonic(flux_values, p2["w2"]),
            "J": _fit_single_harmonic(flux_values, p2["J"]),
            "zeta": _fit_single_harmonic(flux_values, p2["zeta"]),
        }
    else:
        raise ValueError(
            "model1_mode must be one of {'cosine-fit', 'from-model2-per-flux'}, "
            f"got {model1_mode!r}"
        )

    H1_raw = np.asarray(heff(p1["w1"], p1["w2"], p1["J"], p1["zeta"]), dtype=complex)
    H1 = heff_spin_to_lab_hamiltonian(H1_raw, p1["w1"], p1["w2"])

    E1_rel = _relative_energies(H1, n_track=4)
    E2_rel = _relative_energies(H2_eff, n_track=4)
    E3_rel = _relative_energies(H3_eff, n_track=4)

    err_model1 = _per_flux_rmse(E1_rel, E3_rel)
    err_model2 = _per_flux_rmse(E2_rel, E3_rel)

    wc = np.asarray(coupler_frequency(ref_wc0, ref_A, flux_values), dtype=float).ravel()
    d1 = np.abs(p3["w1"] - wc)
    d2 = np.abs(p3["w2"] - wc)
    g_scale = max(abs(float(ref["g_1c"])), abs(float(ref["g_2c"])), 1e-12)
    detuning_ratio = np.minimum(d1, d2) / g_scale
    idle_mask = detuning_ratio >= float(idle_ratio)
    near_mask = detuning_ratio <= float(near_ratio)

    m1_idle = _summarize_masked(err_model1, idle_mask)
    m1_near = _summarize_masked(err_model1, near_mask)
    m2_idle = _summarize_masked(err_model2, idle_mask)
    m2_near = _summarize_masked(err_model2, near_mask)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    axE, axErr, axParams, axRegime = axes.ravel()

    for i in (1, 2, 3):
        c = f"C{i-1}"
        axE.plot(flux_values, E3_rel[:, i], color=c, linewidth=1.8, label=rf"scqubits $E_{{{i}}}$")
        axE.plot(flux_values, E2_rel[:, i], color=c, linestyle="--", linewidth=1.4, label=rf"model2 $E_{{{i}}}$")
        axE.plot(flux_values, E1_rel[:, i], color=c, linestyle=":", linewidth=1.4, label=rf"model1 $E_{{{i}}}$")
    axE.set_ylabel("Energy (GHz, rel. ground)")
    axE.set_title("Dressed computational energies")
    axE.grid(True, alpha=0.3)
    axE.legend(loc="best", fontsize="small", ncol=3)

    axErr.plot(flux_values, err_model1, color="C3", linewidth=1.8, label="model1 vs scqubits")
    axErr.plot(flux_values, err_model2, color="C0", linewidth=1.8, label="model2 vs scqubits")
    if np.any(near_mask):
        axErr.fill_between(flux_values, 0.0, np.max(err_model1) * 1.05, where=near_mask, color="C3", alpha=0.08)
    if np.any(idle_mask):
        axErr.fill_between(flux_values, 0.0, np.max(err_model1) * 1.05, where=idle_mask, color="C0", alpha=0.05)
    axErr.set_ylabel("Per-flux RMSE (GHz)")
    axErr.set_title("Error to scqubits reference")
    axErr.grid(True, alpha=0.3)
    axErr.legend(loc="best", fontsize="small")

    axParams.plot(flux_values, p3["J"], color="C1", linewidth=1.8, label=r"scqubits $J$")
    axParams.plot(flux_values, p2["J"], color="C1", linestyle="--", linewidth=1.4, label=r"model2 $J$")
    axParams.plot(flux_values, p1["J"], color="C1", linestyle=":", linewidth=1.4, label=r"model1 $J$")
    axParams.plot(flux_values, p3["zeta"], color="C2", linewidth=1.8, label=r"scqubits $\zeta$")
    axParams.plot(flux_values, p2["zeta"], color="C2", linestyle="--", linewidth=1.4, label=r"model2 $\zeta$")
    axParams.plot(flux_values, p1["zeta"], color="C2", linestyle=":", linewidth=1.4, label=r"model1 $\zeta$")
    axParams.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    axParams.set_ylabel("Effective params (GHz)")
    axParams.set_title(r"Extracted $J$ and $\zeta$")
    axParams.grid(True, alpha=0.3)
    axParams.legend(loc="best", fontsize="small")

    axRegime.plot(flux_values, detuning_ratio, color="C4", linewidth=1.8, label=r"$\min(|\Delta_1|,|\Delta_2|)/g$")
    axRegime.axhline(float(idle_ratio), color="C0", linestyle="--", linewidth=1.2, label=f"idle threshold={idle_ratio:g}")
    axRegime.axhline(float(near_ratio), color="C3", linestyle="--", linewidth=1.2, label=f"near threshold={near_ratio:g}")
    axRegime.set_ylabel("Detuning ratio")
    axRegime.set_title("Regime classifier")
    axRegime.grid(True, alpha=0.3)
    axRegime.legend(loc="best", fontsize="small")

    axes[1, 0].set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
    axes[1, 1].set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
    fig.suptitle(
        title
        or (
            "Model validity against scqubits reference "
            + rf"(model2={nq}x{nc}x{nq}, ref={nq_ref}x{nc_ref}x{nq_ref})"
        )
    )
    fig.tight_layout()
    plt.savefig(outfile, format="pdf")
    plt.close(fig)

    summary = {
        "model1_idle_rmse": m1_idle[0],
        "model1_idle_max_abs": m1_idle[1],
        "model1_near_rmse": m1_near[0],
        "model1_near_max_abs": m1_near[1],
        "model2_idle_rmse": m2_idle[0],
        "model2_idle_max_abs": m2_idle[1],
        "model2_near_rmse": m2_near[0],
        "model2_near_max_abs": m2_near[1],
    }
    return {
        "flux": flux_values,
        "E1_rel": E1_rel,
        "E2_rel": E2_rel,
        "E3_rel": E3_rel,
        "err_model1": err_model1,
        "err_model2": err_model2,
        "detuning_ratio": detuning_ratio,
        "idle_mask": idle_mask.astype(bool),
        "near_mask": near_mask.astype(bool),
        "params_model2": p2,
        "params_model1": p1,
        "params_scqubits": p3,
        "reference_config": ref,
        "ham_kwargs_used": dict(ham_kwargs_used),
        "summary": summary,
    }


def main() -> None:
    flux = np.linspace(0.0, 1.0, 121)
    outdir = Path(__file__).resolve().parent
    transmon_params = load_transmon_params(DEFAULT_TRANSMON_KEY)
    out = compare_model1_model2_against_scqubits(
        flux,
        wc0=5.05,
        A=0.95,
        reference={
            "transmon1_params": transmon_params,
            "transmon2_params": transmon_params,
            "transmon_dim": 5,
            "coupler_dim": 6,
            "g_1c": 0.09,
            "g_2c": 0.085,
        },
        model2_nlevels_qubit=3,
        model2_nlevels_coupler=3,
        model1_mode="cosine-fit",
        outfile=str(outdir / "model1_model2_vs_scqubits_regime_map.pdf"),
    )
    print("Model2 kwargs used:")
    for key, value in out["ham_kwargs_used"].items():
        print(f"  {key}: {value}")
    print("Summary (RMSE/max_abs in GHz):")
    for key, value in out["summary"].items():
        print(f"  {key}: {value:.6e}")
    print(f"Wrote: {outdir / 'model1_model2_vs_scqubits_regime_map.pdf'}")


if __name__ == "__main__":
    main()
