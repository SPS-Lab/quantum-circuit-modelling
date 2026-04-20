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


def _transmon_analytic_w01_alpha(EJ: np.ndarray, EC: float) -> tuple[np.ndarray, np.ndarray]:
    """Approximate transmon spectral parameters from EJ, EC.

    Uses the standard transmon expansion:
      w01 ~= sqrt(8 EJ EC) - EC
      alpha ~= -EC
    """
    EJ_arr = np.asarray(EJ, dtype=float)
    ec = float(EC)
    w01 = np.sqrt(8.0 * EJ_arr * ec) - ec
    alpha = np.full_like(EJ_arr, -ec, dtype=float)
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

    The coupler is treated as a harmonic mode. Transmon parameters are either:
    - `fixed`: calibrated once at system-configured qubit flux biases, or
    - `analytic-per-flux`: transmon approximation at each sweep point, or
    - `per-flux`: recalibrated at every sweep point.
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
    calibration_mode = str(duffing_config.calibration_mode).strip().lower()

    EJ1_arr = np.asarray(
        flux_dependent_EJ(system_params.q1.EJmax, q1_flux_arr, system_params.q1.d),
        dtype=float,
    ).ravel()
    EJ2_arr = np.asarray(
        flux_dependent_EJ(system_params.q2.EJmax, q2_flux_arr, system_params.q2.d),
        dtype=float,
    ).ravel()

    if calibration_mode == "per-flux":
        w1_arr = np.empty_like(flux_arr, dtype=float)
        w2_arr = np.empty_like(flux_arr, dtype=float)
        alpha1_arr = np.empty_like(flux_arr, dtype=float)
        alpha2_arr = np.empty_like(flux_arr, dtype=float)
        for k in range(flux_arr.shape[0]):
            EJ1 = float(EJ1_arr[k])
            EJ2 = float(EJ2_arr[k])
            w1_arr[k], alpha1_arr[k] = _transmon_w01_alpha(EJ1, system_params.q1.EC, system_params.q1.ng, ncut, trunc_dim)
            w2_arr[k], alpha2_arr[k] = _transmon_w01_alpha(EJ2, system_params.q2.EC, system_params.q2.ng, ncut, trunc_dim)
    elif calibration_mode == "analytic-per-flux":
        w1_arr, alpha1_arr = _transmon_analytic_w01_alpha(EJ1_arr, system_params.q1.EC)
        w2_arr, alpha2_arr = _transmon_analytic_w01_alpha(EJ2_arr, system_params.q2.EC)
    elif calibration_mode == "fixed":
        # Fair baseline: keep Duffing transmon calibration fixed at the system's
        # configured parking biases across the whole sweep.
        EJ1_ref = float(flux_dependent_EJ(system_params.q1.EJmax, system_params.q1.flux, system_params.q1.d))
        EJ2_ref = float(flux_dependent_EJ(system_params.q2.EJmax, system_params.q2.flux, system_params.q2.d))
        w1_ref, alpha1_ref = _transmon_w01_alpha(EJ1_ref, system_params.q1.EC, system_params.q1.ng, ncut, trunc_dim)
        w2_ref, alpha2_ref = _transmon_w01_alpha(EJ2_ref, system_params.q2.EC, system_params.q2.ng, ncut, trunc_dim)
        w1_arr = np.full_like(flux_arr, float(w1_ref), dtype=float)
        w2_arr = np.full_like(flux_arr, float(w2_ref), dtype=float)
        alpha1_arr = np.full_like(flux_arr, float(alpha1_ref), dtype=float)
        alpha2_arr = np.full_like(flux_arr, float(alpha2_ref), dtype=float)
    else:
        raise ValueError(f"Unsupported Duffing calibration_mode {duffing_config.calibration_mode!r}")

    mats: list[np.ndarray] = []
    w1_0 = float(w1_arr[0])
    w2_0 = float(w2_arr[0])
    alpha1_0 = float(alpha1_arr[0])
    alpha2_0 = float(alpha2_arr[0])
    for k in range(flux_arr.shape[0]):
        mats.append(
            three_mode_hamiltonian(
                w_1=float(w1_arr[k]),
                w_c=float(wc_arr[k]),
                w_2=float(w2_arr[k]),
                alpha_1=float(alpha1_arr[k]),
                alpha_c=alpha_c,
                alpha_2=float(alpha2_arr[k]),
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
