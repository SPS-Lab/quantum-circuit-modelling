"""Three-mode Duffing model construction for study benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.josephson import flux_dependent_EJ
from models.sweep import resolve_static_sweep_values
from models.three_mode import three_mode_hamiltonian
from study_config import CouplerFrequencyConfig, DuffingModelConfig, SystemParams


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
    *,
    sweep_target: str = "coupler",
) -> DuffingModelBuildResult:
    """Build a three-mode Duffing Hamiltonian stack from system + study config.

    The coupler is treated as a harmonic mode and transmons are re-calibrated
    along the configured sweep target.
    """
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    q1_flux_arr, q2_flux_arr, wc_arr = resolve_static_sweep_values(
        flux_arr,
        system_params=system_params,
        coupler_frequency_config=coupler_frequency,
        sweep_target=sweep_target,
    )

    ncut = int(duffing_config.transmon_spectral_extraction.ncut)
    trunc_dim = int(duffing_config.transmon_spectral_extraction.truncated_dim)

    nlevels_q = int(duffing_config.hilbert_truncation.nlevels_qubit)
    nlevels_c = int(duffing_config.hilbert_truncation.nlevels_coupler)
    alpha_c = float(duffing_config.coupler_anharmonicity)
    g_1c = float(system_params.interactions.g_1c)
    g_2c = float(system_params.interactions.g_2c)

    mats: list[np.ndarray] = []
    w1_0 = 0.0
    w2_0 = 0.0
    alpha1_0 = 0.0
    alpha2_0 = 0.0
    for k in range(flux_arr.shape[0]):
        EJ1 = float(flux_dependent_EJ(system_params.q1.EJmax, q1_flux_arr[k], system_params.q1.d))
        EJ2 = float(flux_dependent_EJ(system_params.q2.EJmax, q2_flux_arr[k], system_params.q2.d))
        w1, alpha1 = _transmon_w01_alpha(EJ1, system_params.q1.EC, system_params.q1.ng, ncut, trunc_dim)
        w2, alpha2 = _transmon_w01_alpha(EJ2, system_params.q2.EC, system_params.q2.ng, ncut, trunc_dim)

        if k == 0:
            w1_0 = float(w1)
            w2_0 = float(w2)
            alpha1_0 = float(alpha1)
            alpha2_0 = float(alpha2)

        mats.append(
            three_mode_hamiltonian(
                w_1=float(w1),
                w_c=float(wc_arr[k]),
                w_2=float(w2),
                alpha_1=float(alpha1),
                alpha_c=alpha_c,
                alpha_2=float(alpha2),
                g_1c=g_1c,
                g_2c=g_2c,
                nlevels_qubit=nlevels_q,
                nlevels_coupler=nlevels_c,
            )
        )
    H_stack = np.stack(mats, axis=0)

    ham_kwargs: dict[str, float | int] = {
        "w_1": w1_0,
        "w_2": w2_0,
        "alpha_1": alpha1_0,
        "alpha_c": alpha_c,
        "alpha_2": alpha2_0,
        "g_1c": g_1c,
        "g_2c": g_2c,
        "nlevels_qubit": nlevels_q,
        "nlevels_coupler": nlevels_c,
    }

    return DuffingModelBuildResult(
        hamiltonian_stack=H_stack,
        hamiltonian_kwargs=ham_kwargs,
        calibration=DuffingCalibrationResult(
            q1_w01=w1_0,
            q2_w01=w2_0,
            q1_alpha=alpha1_0,
            q2_alpha=alpha2_0,
        ),
    )
