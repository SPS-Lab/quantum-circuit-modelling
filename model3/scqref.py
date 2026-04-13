"""scqubits-backed reference builders for the three-mode Kerr model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Repo root (parent of model3/) so `model2` resolves when run from model3/.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model2.core import coupler_frequency, three_mode_hamiltonian_from_kwargs
from model2.hamiltonian_types import ThreeModeHamiltonianCommonKwargs

try:
    import scqubits as scq
except Exception as exc:  # pragma: no cover - import guard only
    scq = None
    _SCQ_IMPORT_ERROR = exc
else:
    _SCQ_IMPORT_ERROR = None


def _require_scqubits() -> None:
    if scq is None:
        raise ImportError(
            "scqubits import failed; install dependencies from scmodels-env.yml"
        ) from _SCQ_IMPORT_ERROR


def _kerr_from_duffing(
    w: float,
    alpha: float,
    truncated_dim: int,
    *,
    id_str: str,
):
    """Build a Kerr oscillator matching ``w*n + (alpha/2) n(n-1)``."""
    # scqubits KerrOscillator uses H = E_osc * n - K * n(n-1).
    # Matching to model2 Duffing convention gives K = -alpha/2.
    K = -0.5 * float(alpha)
    return scq.KerrOscillator(
        E_osc=float(w),
        K=float(K),
        truncated_dim=int(truncated_dim),
        id_str=id_str,
    )


def three_mode_scqubits_hamiltonian(
    w_1: float,
    w_c: float,
    w_2: float,
    alpha_1: float,
    alpha_c: float,
    alpha_2: float,
    g_1c: float,
    g_2c: float,
    nlevels_qubit: int,
    nlevels_coupler: int,
) -> np.ndarray:
    """Construct the three-mode Hamiltonian via scqubits + qutip back-end.

    The model mirrors ``model2.core.three_mode_hamiltonian``:
    - three Kerr oscillators (qubit-coupler-qubit)
    - rotating-wave exchange couplings
      ``g_1c (a1^dag ac + ac^dag a1) + g_2c (a2^dag ac + ac^dag a2)``.
    """
    _require_scqubits()
    q1 = _kerr_from_duffing(w_1, alpha_1, nlevels_qubit, id_str="q1")
    c = _kerr_from_duffing(w_c, alpha_c, nlevels_coupler, id_str="c")
    q2 = _kerr_from_duffing(w_2, alpha_2, nlevels_qubit, id_str="q2")

    hilbertspace = scq.HilbertSpace([q1, c, q2])
    hilbertspace.add_interaction(
        check_validity=False,
        g=float(g_1c),
        op1=(q1.creation_operator(), q1),
        op2=(c.annihilation_operator(), c),
        add_hc=True,
    )
    hilbertspace.add_interaction(
        check_validity=False,
        g=float(g_2c),
        op1=(q2.creation_operator(), q2),
        op2=(c.annihilation_operator(), c),
        add_hc=True,
    )

    H = hilbertspace.hamiltonian()
    return np.asarray(H.full(), dtype=complex)


def three_mode_scqubits_hamiltonian_from_kwargs(
    ham_kwargs: ThreeModeHamiltonianCommonKwargs,
    *,
    w_c: float,
    nlevels_qubit_ref: int | None = None,
    nlevels_coupler_ref: int | None = None,
) -> np.ndarray:
    """Build a scqubits Hamiltonian from model2 kwargs + optional reference truncation."""
    nq_ref = int(ham_kwargs["nlevels_qubit"]) if nlevels_qubit_ref is None else int(nlevels_qubit_ref)
    nc_ref = int(ham_kwargs["nlevels_coupler"]) if nlevels_coupler_ref is None else int(nlevels_coupler_ref)
    return three_mode_scqubits_hamiltonian(
        float(ham_kwargs["w_1"]),
        float(w_c),
        float(ham_kwargs["w_2"]),
        float(ham_kwargs["alpha_1"]),
        float(ham_kwargs["alpha_c"]),
        float(ham_kwargs["alpha_2"]),
        float(ham_kwargs["g_1c"]),
        float(ham_kwargs["g_2c"]),
        nq_ref,
        nc_ref,
    )


def three_mode_scqubits_stack_vs_flux(
    flux_values: np.ndarray,
    *,
    wc0: float,
    A: float,
    ham_kwargs: ThreeModeHamiltonianCommonKwargs,
    nlevels_qubit_ref: int | None = None,
    nlevels_coupler_ref: int | None = None,
) -> np.ndarray:
    """Return ``(n_flux, d, d)`` scqubits Hamiltonian stack for a flux sweep."""
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    wc_arr = np.asarray(coupler_frequency(wc0, A, flux_values), dtype=float).ravel()
    mats = [
        three_mode_scqubits_hamiltonian_from_kwargs(
            ham_kwargs,
            w_c=float(wc_arr[k]),
            nlevels_qubit_ref=nlevels_qubit_ref,
            nlevels_coupler_ref=nlevels_coupler_ref,
        )
        for k in range(flux_values.shape[0])
    ]
    return np.stack(mats, axis=0)


def compare_single_flux_with_model2(
    *,
    flux: float,
    wc0: float,
    A: float,
    ham_kwargs: ThreeModeHamiltonianCommonKwargs,
    nlevels_qubit_ref: int | None = None,
    nlevels_coupler_ref: int | None = None,
) -> dict[str, float]:
    """Return simple one-point diagnostics for model2 vs scqubits matrix agreement."""
    wc = float(coupler_frequency(wc0, A, float(flux)))
    H_model2 = three_mode_hamiltonian_from_kwargs(ham_kwargs, w_c=wc)
    H_scq = three_mode_scqubits_hamiltonian_from_kwargs(
        ham_kwargs,
        w_c=wc,
        nlevels_qubit_ref=nlevels_qubit_ref,
        nlevels_coupler_ref=nlevels_coupler_ref,
    )
    d = min(H_model2.shape[0], H_scq.shape[0])
    e2 = np.linalg.eigvalsh(H_model2)[: min(4, d)]
    e3 = np.linalg.eigvalsh(H_scq)[: min(4, d)]
    return {
        "fro_norm_overlap_dim": float(np.linalg.norm(H_model2[:d, :d] - H_scq[:d, :d], ord="fro")),
        "max_abs_low4_eig_diff": float(np.max(np.abs(e2 - e3))),
    }


if __name__ == "__main__":
    common: ThreeModeHamiltonianCommonKwargs = {
        "w_1": 5.0,
        "w_2": 5.2,
        "alpha_1": -0.3,
        "alpha_c": -0.25,
        "alpha_2": -0.32,
        "g_1c": 0.08,
        "g_2c": 0.075,
        "nlevels_qubit": 3,
        "nlevels_coupler": 3,
    }
    out_same = compare_single_flux_with_model2(
        flux=0.2,
        wc0=6.0,
        A=1.0,
        ham_kwargs=common,
        nlevels_qubit_ref=3,
        nlevels_coupler_ref=3,
    )
    out_ref = compare_single_flux_with_model2(
        flux=0.2,
        wc0=6.0,
        A=1.0,
        ham_kwargs=common,
        nlevels_qubit_ref=4,
        nlevels_coupler_ref=5,
    )
    print("Same truncation model2 vs scqubits:", out_same)
    print("Larger reference truncation (4/5) vs model2(3/3):", out_ref)
