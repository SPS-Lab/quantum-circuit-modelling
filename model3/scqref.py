"""scqubits reference model: two Transmons + one Oscillator coupler."""

from __future__ import annotations

from collections.abc import Mapping
import sys
from pathlib import Path

import numpy as np

# Repo root (parent of model3/) so local imports resolve when run from model3/.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model3.reference_params import DEFAULT_TRANSMON_KEY, load_transmon_params

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


def _validated_transmon_params(params: Mapping[str, float], *, name: str) -> dict[str, float]:
    params = dict(params)
    if "EJ" not in params or "EC" not in params:
        raise ValueError(f"{name} must include EJ and EC, got {params}")
    return {"EJ": float(params["EJ"]), "EC": float(params["EC"])}


def transmon_oscillator_hamiltonian(
    *,
    transmon1_params: Mapping[str, float],
    transmon2_params: Mapping[str, float],
    transmon_ng: float = 0.0,
    transmon_ncut: int = 45,
    transmon_dim: int = 6,
    coupler_E_osc: float = 6.0,
    coupler_dim: int = 8,
    g_1c: float = 0.08,
    g_2c: float = 0.08,
) -> np.ndarray:
    """Build full reference Hamiltonian for Transmon-oscillator-Transmon."""
    _require_scqubits()
    if transmon_dim < 2 or coupler_dim < 2:
        raise ValueError("transmon_dim and coupler_dim must be >= 2")
    p1 = _validated_transmon_params(transmon1_params, name="transmon1_params")
    p2 = _validated_transmon_params(transmon2_params, name="transmon2_params")

    q1 = scq.Transmon(
        EJ=p1["EJ"],
        EC=p1["EC"],
        ng=float(transmon_ng),
        ncut=int(transmon_ncut),
        truncated_dim=int(transmon_dim),
        id_str="q1",
    )
    q2 = scq.Transmon(
        EJ=p2["EJ"],
        EC=p2["EC"],
        ng=float(transmon_ng),
        ncut=int(transmon_ncut),
        truncated_dim=int(transmon_dim),
        id_str="q2",
    )
    c = scq.Oscillator(
        E_osc=float(coupler_E_osc),
        truncated_dim=int(coupler_dim),
        id_str="c",
    )

    hilbertspace = scq.HilbertSpace([q1, c, q2])
    x_c = c.creation_operator() + c.annihilation_operator()
    hilbertspace.add_interaction(
        check_validity=False,
        g=float(g_1c),
        op1=(q1.n_operator(), q1),
        op2=(x_c, c),
    )
    hilbertspace.add_interaction(
        check_validity=False,
        g=float(g_2c),
        op1=(q2.n_operator(), q2),
        op2=(x_c, c),
    )

    return np.asarray(hilbertspace.hamiltonian().full(), dtype=complex)


def transmon_oscillator_stack_vs_flux(
    flux_values: np.ndarray,
    *,
    wc0: float,
    A: float,
    transmon1_params: Mapping[str, float],
    transmon2_params: Mapping[str, float],
    transmon_ng: float = 0.0,
    transmon_ncut: int = 45,
    transmon_dim: int = 6,
    coupler_dim: int = 8,
    g_1c: float = 0.08,
    g_2c: float = 0.08,
) -> np.ndarray:
    """Return scqubits reference Hamiltonian stack across flux."""
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    wc = np.asarray(wc0 + A * np.cos(2.0 * np.pi * flux_values), dtype=float).ravel()
    mats = [
        transmon_oscillator_hamiltonian(
            transmon1_params=transmon1_params,
            transmon2_params=transmon2_params,
            transmon_ng=transmon_ng,
            transmon_ncut=transmon_ncut,
            transmon_dim=transmon_dim,
            coupler_E_osc=float(wc_k),
            coupler_dim=coupler_dim,
            g_1c=g_1c,
            g_2c=g_2c,
        )
        for wc_k in wc
    ]
    return np.stack(mats, axis=0)


if __name__ == "__main__":
    params = load_transmon_params(DEFAULT_TRANSMON_KEY)
    print("Loaded transmon params:", params)
    H = transmon_oscillator_hamiltonian(
        transmon1_params={"EJ": params["EJ"] * 0.96, "EC": params["EC"]},
        transmon2_params=params,
        coupler_E_osc=6.0,
        transmon_dim=5,
        coupler_dim=6,
        g_1c=0.09,
        g_2c=0.085,
    )
    evals = np.linalg.eigvalsh(H)
    print("Reference dims:", H.shape)
    print("Lowest 8 levels:", np.round(evals[:8], 6))
