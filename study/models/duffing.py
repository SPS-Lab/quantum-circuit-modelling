"""Three-mode Duffing model construction for study benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model0.cpb import flux_dependent_EJ
from model2.core import three_mode_hamiltonian_stack_vs_flux
from study.config import CouplerFrequencyConfig, DuffingModelConfig, SystemParams


@dataclass(frozen=True)
class DuffingCalibrationResult:
    q1_w01: float
    q2_w01: float
    q1_alpha: float
    q2_alpha: float


@dataclass(frozen=True)
class DuffingModelBuildResult:
    hamiltonian_stack: np.ndarray
    hamiltonian_kwargs: dict[str, float | int]
    calibration: DuffingCalibrationResult



def _transmon_w01_alpha(
    EJ: float,
    EC: float,
    ng: float,
    ncut: int,
    truncated_dim: int,
) -> tuple[float, float]:
    try:
        import scqubits as scq
    except Exception as exc:  # pragma: no cover - import guard only
        raise ImportError("scqubits import failed while calibrating Duffing parameters") from exc

    transmon = scq.Transmon(
        EJ=float(EJ),
        EC=float(EC),
        ng=float(ng),
        ncut=int(ncut),
        truncated_dim=int(truncated_dim),
    )
    evals = np.asarray(transmon.eigenvals(evals_count=3), dtype=float).ravel()
    if evals.size < 3:
        raise ValueError("Need at least 3 transmon levels to extract anharmonicity")
    w01 = float(evals[1] - evals[0])
    alpha = float((evals[2] - evals[1]) - (evals[1] - evals[0]))
    return w01, alpha



def build_duffing_model_stack(
    flux_values: np.ndarray,
    system_params: SystemParams,
    coupler_frequency: CouplerFrequencyConfig,
    duffing_config: DuffingModelConfig,
) -> DuffingModelBuildResult:
    """Build a three-mode Duffing Hamiltonian stack from system + study config."""
    EJ1 = float(flux_dependent_EJ(system_params.q1.EJmax, system_params.q1.flux, system_params.q1.d))
    EJ2 = float(flux_dependent_EJ(system_params.q2.EJmax, system_params.q2.flux, system_params.q2.d))

    ncut = int(duffing_config.transmon_spectral_extraction.ncut)
    trunc_dim = int(duffing_config.transmon_spectral_extraction.truncated_dim)

    w1, alpha1 = _transmon_w01_alpha(EJ1, system_params.q1.EC, system_params.q1.ng, ncut, trunc_dim)
    w2, alpha2 = _transmon_w01_alpha(EJ2, system_params.q2.EC, system_params.q2.ng, ncut, trunc_dim)

    ham_kwargs: dict[str, float | int] = {
        "w_1": float(w1),
        "w_2": float(w2),
        "alpha_1": float(alpha1),
        "alpha_c": float(duffing_config.coupler_anharmonicity),
        "alpha_2": float(alpha2),
        "g_1c": float(system_params.interactions.g_1c),
        "g_2c": float(system_params.interactions.g_2c),
        "nlevels_qubit": int(duffing_config.hilbert_truncation.nlevels_qubit),
        "nlevels_coupler": int(duffing_config.hilbert_truncation.nlevels_coupler),
    }

    H_stack = three_mode_hamiltonian_stack_vs_flux(
        np.asarray(flux_values, dtype=float),
        wc0=float(coupler_frequency.wc0),
        A=float(coupler_frequency.amplitude),
        ham_kwargs=ham_kwargs,
    )

    return DuffingModelBuildResult(
        hamiltonian_stack=H_stack,
        hamiltonian_kwargs=ham_kwargs,
        calibration=DuffingCalibrationResult(
            q1_w01=float(w1),
            q2_w01=float(w2),
            q1_alpha=float(alpha1),
            q2_alpha=float(alpha2),
        ),
    )
