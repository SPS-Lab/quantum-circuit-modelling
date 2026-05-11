"""Three-mode Duffing model construction for study benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from models.dressed import (
    build_dressed_effective_computational_stack,
    extract_model1_parameters_from_4x4_stack,
)
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
    mode_parameters: dict[str, np.ndarray]


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


def _build_mode_parameter_arrays(
    flux_values: np.ndarray,
    system_params: SystemParams,
    coupler_frequency: CouplerFrequencyConfig,
    duffing_config: DuffingModelConfig,
    *,
    sweep_target: str,
) -> dict[str, np.ndarray]:
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    q1_flux_arr, q2_flux_arr, wc_arr = resolve_static_sweep_values(
        flux_arr,
        system_params=system_params,
        coupler_frequency_config=coupler_frequency,
        sweep_target=sweep_target,
    )

    ncut = int(duffing_config.transmon_spectral_extraction.ncut)
    trunc_dim = int(duffing_config.transmon_spectral_extraction.truncated_dim)
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
            w1_arr[k], alpha1_arr[k] = _transmon_w01_alpha(
                float(EJ1_arr[k]),
                system_params.q1.EC,
                system_params.q1.ng,
                ncut,
                trunc_dim,
            )
            w2_arr[k], alpha2_arr[k] = _transmon_w01_alpha(
                float(EJ2_arr[k]),
                system_params.q2.EC,
                system_params.q2.ng,
                ncut,
                trunc_dim,
            )
    elif calibration_mode == "analytic-per-flux":
        w1_arr, alpha1_arr = _transmon_analytic_w01_alpha(EJ1_arr, system_params.q1.EC)
        w2_arr, alpha2_arr = _transmon_analytic_w01_alpha(EJ2_arr, system_params.q2.EC)
    elif calibration_mode == "fixed":
        EJ1_ref = float(flux_dependent_EJ(system_params.q1.EJmax, system_params.q1.flux, system_params.q1.d))
        EJ2_ref = float(flux_dependent_EJ(system_params.q2.EJmax, system_params.q2.flux, system_params.q2.d))
        w1_ref, alpha1_ref = _transmon_w01_alpha(
            EJ1_ref,
            system_params.q1.EC,
            system_params.q1.ng,
            ncut,
            trunc_dim,
        )
        w2_ref, alpha2_ref = _transmon_w01_alpha(
            EJ2_ref,
            system_params.q2.EC,
            system_params.q2.ng,
            ncut,
            trunc_dim,
        )
        w1_arr = np.full_like(flux_arr, float(w1_ref), dtype=float)
        w2_arr = np.full_like(flux_arr, float(w2_ref), dtype=float)
        alpha1_arr = np.full_like(flux_arr, float(alpha1_ref), dtype=float)
        alpha2_arr = np.full_like(flux_arr, float(alpha2_ref), dtype=float)
    elif calibration_mode == "fitted-static":
        # Use per-flux numerical transmon extraction as the latent-parameter prior.
        w1_arr = np.empty_like(flux_arr, dtype=float)
        w2_arr = np.empty_like(flux_arr, dtype=float)
        alpha1_arr = np.empty_like(flux_arr, dtype=float)
        alpha2_arr = np.empty_like(flux_arr, dtype=float)
        for k in range(flux_arr.shape[0]):
            w1_arr[k], alpha1_arr[k] = _transmon_w01_alpha(
                float(EJ1_arr[k]),
                system_params.q1.EC,
                system_params.q1.ng,
                ncut,
                trunc_dim,
            )
            w2_arr[k], alpha2_arr[k] = _transmon_w01_alpha(
                float(EJ2_arr[k]),
                system_params.q2.EC,
                system_params.q2.ng,
                ncut,
                trunc_dim,
            )
    else:
        raise ValueError(f"Unsupported Duffing calibration_mode {duffing_config.calibration_mode!r}")

    return {
        "w1": np.asarray(w1_arr, dtype=float),
        "w2": np.asarray(w2_arr, dtype=float),
        "alpha1": np.asarray(alpha1_arr, dtype=float),
        "alpha2": np.asarray(alpha2_arr, dtype=float),
        "wc": np.asarray(wc_arr, dtype=float),
    }


def build_duffing_model_stack_from_parameters(
    mode_parameters: Mapping[str, np.ndarray],
    system_params: SystemParams,
    duffing_config: DuffingModelConfig,
) -> DuffingModelBuildResult:
    """Build a Duffing Hamiltonian stack from explicit per-point mode parameters."""
    w1_arr = np.asarray(mode_parameters["w1"], dtype=float).ravel()
    w2_arr = np.asarray(mode_parameters["w2"], dtype=float).ravel()
    alpha1_arr = np.asarray(mode_parameters["alpha1"], dtype=float).ravel()
    alpha2_arr = np.asarray(mode_parameters["alpha2"], dtype=float).ravel()
    wc_arr = np.asarray(mode_parameters["wc"], dtype=float).ravel()

    if not (w1_arr.shape == w2_arr.shape == alpha1_arr.shape == alpha2_arr.shape == wc_arr.shape):
        raise ValueError("Duffing mode parameter arrays must all share the same shape")

    nlevels_q = int(duffing_config.hilbert_truncation.nlevels_qubit)
    nlevels_c = int(duffing_config.hilbert_truncation.nlevels_coupler)
    alpha_c = float(duffing_config.coupler_anharmonicity)
    g_1c = float(system_params.interactions.g_1c)
    g_2c = float(system_params.interactions.g_2c)

    mats: list[np.ndarray] = []
    for k in range(w1_arr.shape[0]):
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
        "w_1": float(w1_arr[0]),
        "w_2": float(w2_arr[0]),
        "alpha_1": float(alpha1_arr[0]),
        "alpha_c": alpha_c,
        "alpha_2": float(alpha2_arr[0]),
        "g_1c": g_1c,
        "g_2c": g_2c,
        "nlevels_qubit": nlevels_q,
        "nlevels_coupler": nlevels_c,
    }

    return DuffingModelBuildResult(
        hamiltonian_stack=H_stack,
        hamiltonian_kwargs=ham_kwargs,
        calibration=DuffingCalibrationResult(
            q1_w01=float(w1_arr[0]),
            q2_w01=float(w2_arr[0]),
            q1_alpha=float(alpha1_arr[0]),
            q2_alpha=float(alpha2_arr[0]),
        ),
        mode_parameters={
            "w1": np.asarray(w1_arr, dtype=float),
            "w2": np.asarray(w2_arr, dtype=float),
            "alpha1": np.asarray(alpha1_arr, dtype=float),
            "alpha2": np.asarray(alpha2_arr, dtype=float),
            "wc": np.asarray(wc_arr, dtype=float),
        },
    )


def fit_duffing_mode_parameters_to_reference(
    flux_values: np.ndarray,
    reference_dressed_stack: np.ndarray,
    system_params: SystemParams,
    coupler_frequency: CouplerFrequencyConfig,
    duffing_config: DuffingModelConfig,
    *,
    sweep_target: str,
    n_candidate_states: int,
    selection_mode: str,
    regularization_weight: float = 0.10,
    max_nfev: int = 80,
) -> dict[str, np.ndarray]:
    """Fit latent Duffing parameters so static dressed observables track a reference stack."""
    from scipy.optimize import least_squares

    initial = _build_mode_parameter_arrays(
        flux_values,
        system_params,
        coupler_frequency,
        duffing_config,
        sweep_target=sweep_target,
    )
    ref_params = extract_model1_parameters_from_4x4_stack(reference_dressed_stack)
    n_q = int(duffing_config.hilbert_truncation.nlevels_qubit)
    n_c = int(duffing_config.hilbert_truncation.nlevels_coupler)

    fitted = {
        "w1": np.array(initial["w1"], copy=True, dtype=float),
        "w2": np.array(initial["w2"], copy=True, dtype=float),
        "alpha1": np.array(initial["alpha1"], copy=True, dtype=float),
        "alpha2": np.array(initial["alpha2"], copy=True, dtype=float),
        "wc": np.array(initial["wc"], copy=True, dtype=float),
    }

    for k in range(fitted["w1"].shape[0]):
        x0 = np.array(
            [
                float(initial["w1"][k]),
                float(initial["w2"][k]),
                float(initial["alpha1"][k]),
                float(initial["alpha2"][k]),
            ],
            dtype=float,
        )
        target = np.array(
            [
                float(ref_params["w1"][k]),
                float(ref_params["w2"][k]),
                float(ref_params["J"][k]),
                float(ref_params["zeta"][k]),
            ],
            dtype=float,
        )
        obs_scale = np.array(
            [
                max(abs(target[0]), 1.0),
                max(abs(target[1]), 1.0),
                max(abs(target[2]), 2e-2),
                max(abs(target[3]), 2e-2),
            ],
            dtype=float,
        )
        latent_scale = np.array(
            [
                max(abs(x0[0]), 1.0),
                max(abs(x0[1]), 1.0),
                max(abs(x0[2]), 0.25),
                max(abs(x0[3]), 0.25),
            ],
            dtype=float,
        )

        def residual(x: np.ndarray) -> np.ndarray:
            candidate = build_duffing_model_stack_from_parameters(
                {
                    "w1": np.array([float(x[0])], dtype=float),
                    "w2": np.array([float(x[1])], dtype=float),
                    "alpha1": np.array([float(x[2])], dtype=float),
                    "alpha2": np.array([float(x[3])], dtype=float),
                    "wc": np.array([float(initial["wc"][k])], dtype=float),
                },
                system_params=system_params,
                duffing_config=duffing_config,
            ).hamiltonian_stack
            dressed = build_dressed_effective_computational_stack(
                candidate,
                nlevels_qubit=n_q,
                nlevels_coupler=n_c,
                n_candidate_states=n_candidate_states,
                selection_mode=selection_mode,
            )
            params = extract_model1_parameters_from_4x4_stack(dressed)
            obs = np.array(
                [
                    float(params["w1"][0]),
                    float(params["w2"][0]),
                    float(params["J"][0]),
                    float(params["zeta"][0]),
                ],
                dtype=float,
            )
            obs_res = (obs - target) / obs_scale
            reg_res = float(regularization_weight) * (x - x0) / latent_scale
            return np.concatenate([obs_res, reg_res])

        result = least_squares(
            residual,
            x0=x0,
            bounds=(
                np.array([0.0, 0.0, -5.0, -5.0], dtype=float),
                np.array([20.0, 20.0, -1e-6, -1e-6], dtype=float),
            ),
            max_nfev=int(max_nfev),
        )
        x_best = x0 if not result.success else np.asarray(result.x, dtype=float)
        fitted["w1"][k] = float(x_best[0])
        fitted["w2"][k] = float(x_best[1])
        fitted["alpha1"][k] = float(x_best[2])
        fitted["alpha2"][k] = float(x_best[3])

    return fitted


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
    - `fitted-static`: fitted against static reference observables in a higher-level step.
    """
    if str(duffing_config.calibration_mode).strip().lower() == "fitted-static":
        raise ValueError(
            "Duffing calibration_mode 'fitted-static' requires a reference-driven "
            "static calibration step; use fit_duffing_mode_parameters_to_reference "
            "and build_duffing_model_stack_from_parameters instead."
        )

    mode_parameters = _build_mode_parameter_arrays(
        flux_values,
        system_params,
        coupler_frequency,
        duffing_config,
        sweep_target=sweep_target,
    )
    return build_duffing_model_stack_from_parameters(
        mode_parameters,
        system_params=system_params,
        duffing_config=duffing_config,
    )
