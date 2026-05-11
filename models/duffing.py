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
    q0_w01: float
    q1_w01: float
    q0_alpha: float
    q1_alpha: float


@dataclass(frozen=True)
class DuffingModelBuildResult:
    hamiltonian_stack: np.ndarray
    hamiltonian_kwargs: dict[str, float | int]
    calibration: DuffingCalibrationResult
    mode_parameters: dict[str, np.ndarray]


@dataclass(frozen=True)
class DuffingSymbolicParameterFitResult:
    coefficient_names: dict[str, np.ndarray]
    coefficients: dict[str, np.ndarray]
    fitted_parameters: dict[str, np.ndarray]


def is_reference_calibrated_duffing_mode(calibration_mode: str) -> bool:
    mode = str(calibration_mode).strip().lower()
    return mode in {"fitted-static", "symbolic-fitted-static"}


def _transmon_w01_alpha(
    *,
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


def _transmon_analytic_w01_alpha(
    *,
    EJ: np.ndarray,
    EC: float
) -> tuple[np.ndarray, np.ndarray]:
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
    *,
    system_params: SystemParams,
    coupler_frequency: CouplerFrequencyConfig,
    duffing_config: DuffingModelConfig,
    sweep_target: str,
) -> dict[str, np.ndarray]:
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    q0_flux_arr, q1_flux_arr, wc_arr = resolve_static_sweep_values(
        flux_arr,
        system_params=system_params,
        coupler_frequency_config=coupler_frequency,
        sweep_target=sweep_target,
    )

    ncut = int(duffing_config.transmon_spectral_extraction.ncut)
    trunc_dim = int(duffing_config.transmon_spectral_extraction.truncated_dim)
    calibration_mode = str(duffing_config.calibration_mode).strip().lower()

    EJ0_arr = np.asarray(
        flux_dependent_EJ(
            EJ_max=system_params.q0.EJmax,
            flux_bias=q0_flux_arr,
            d=system_params.q0.d
        ),
        dtype=float,
    ).ravel()
    EJ1_arr = np.asarray(
        flux_dependent_EJ(
            EJ_max=system_params.q1.EJmax,
            flux_bias=q1_flux_arr,
            d=system_params.q1.d
        ),
        dtype=float,
    ).ravel()

    if calibration_mode in {"per-flux", "fitted-static", "symbolic-fitted-static"}:
        w0_arr = np.empty_like(flux_arr, dtype=float)
        w1_arr = np.empty_like(flux_arr, dtype=float)
        alpha0_arr = np.empty_like(flux_arr, dtype=float)
        alpha1_arr = np.empty_like(flux_arr, dtype=float)
        for k in range(flux_arr.shape[0]):
            w0_arr[k], alpha0_arr[k] = _transmon_w01_alpha(
                EJ=float(EJ0_arr[k]),
                EC=system_params.q0.EC,
                ng=system_params.q0.ng,
                ncut=ncut,
                truncated_dim=trunc_dim,
            )
            w1_arr[k], alpha1_arr[k] = _transmon_w01_alpha(
                EJ=float(EJ1_arr[k]),
                EC=system_params.q1.EC,
                ng=system_params.q1.ng,
                ncut=ncut,
                truncated_dim=trunc_dim,
            )
    elif calibration_mode == "analytic-per-flux":
        w0_arr, alpha0_arr = _transmon_analytic_w01_alpha(
            EJ=EJ0_arr,
            EC=system_params.q0.EC
        )
        w1_arr, alpha1_arr = _transmon_analytic_w01_alpha(
            EJ=EJ1_arr,
            EC=system_params.q1.EC
        )
    elif calibration_mode == "fixed":
        EJ0_ref = float(flux_dependent_EJ(
            EJ_max=system_params.q0.EJmax,
            flux_bias=system_params.q0.flux,
            d=system_params.q0.d
        ))
        EJ1_ref = float(flux_dependent_EJ(
            EJ_max=system_params.q1.EJmax,
            flux_bias=system_params.q1.flux,
            d=system_params.q1.d)
        )
        w0_ref, alpha0_ref = _transmon_w01_alpha(
            EJ=EJ0_ref,
            EC=system_params.q0.EC,
            ng=system_params.q0.ng,
            ncut=ncut,
            truncated_dim=trunc_dim,
        )
        w1_ref, alpha1_ref = _transmon_w01_alpha(
            EJ=EJ1_ref,
            EC=system_params.q1.EC,
            ng=system_params.q1.ng,
            ncut=ncut,
            truncated_dim=trunc_dim,
        )
        w0_arr = np.full_like(flux_arr, float(w0_ref), dtype=float)
        w1_arr = np.full_like(flux_arr, float(w1_ref), dtype=float)
        alpha0_arr = np.full_like(flux_arr, float(alpha0_ref), dtype=float)
        alpha1_arr = np.full_like(flux_arr, float(alpha1_ref), dtype=float)
    else:
        raise ValueError(f"Unsupported Duffing calibration_mode {duffing_config.calibration_mode!r}")

    return {
        "w0": np.asarray(w0_arr, dtype=float),
        "w1": np.asarray(w1_arr, dtype=float),
        "alpha0": np.asarray(alpha0_arr, dtype=float),
        "alpha1": np.asarray(alpha1_arr, dtype=float),
        "wc": np.asarray(wc_arr, dtype=float),
    }


def build_duffing_model_stack_from_parameters(
    mode_parameters: Mapping[str, np.ndarray],
    system_params: SystemParams,
    duffing_config: DuffingModelConfig,
) -> DuffingModelBuildResult:
    """Build a Duffing Hamiltonian stack from explicit per-point mode parameters."""
    w0_arr = np.asarray(mode_parameters["w0"], dtype=float).ravel()
    w1_arr = np.asarray(mode_parameters["w1"], dtype=float).ravel()
    alpha0_arr = np.asarray(mode_parameters["alpha0"], dtype=float).ravel()
    alpha1_arr = np.asarray(mode_parameters["alpha1"], dtype=float).ravel()
    wc_arr = np.asarray(mode_parameters["wc"], dtype=float).ravel()

    if not (w0_arr.shape == w1_arr.shape == alpha0_arr.shape == alpha1_arr.shape == wc_arr.shape):
        raise ValueError("Duffing mode parameter arrays must all share the same shape")

    nlevels_q = int(duffing_config.hilbert_truncation.nlevels_qubit)
    nlevels_c = int(duffing_config.hilbert_truncation.nlevels_coupler)
    alpha_c = float(duffing_config.coupler_anharmonicity)
    g_0c = float(system_params.interactions.g_0c)
    g_1c = float(system_params.interactions.g_1c)

    mats: list[np.ndarray] = []
    for k in range(w0_arr.shape[0]):
        mats.append(
            three_mode_hamiltonian(
                w_0=float(w0_arr[k]),
                w_c=float(wc_arr[k]),
                w_1=float(w1_arr[k]),
                alpha_0=float(alpha0_arr[k]),
                alpha_c=alpha_c,
                alpha_1=float(alpha1_arr[k]),
                g_0c=g_0c,
                g_1c=g_1c,
                nlevels_qubit=nlevels_q,
                nlevels_coupler=nlevels_c,
            )
        )
    H_stack = np.stack(mats, axis=0)

    ham_kwargs: dict[str, float | int] = {
        "w_0": float(w0_arr[0]),
        "w_1": float(w1_arr[0]),
        "alpha_0": float(alpha0_arr[0]),
        "alpha_c": alpha_c,
        "alpha_1": float(alpha1_arr[0]),
        "g_0c": g_0c,
        "g_1c": g_1c,
        "nlevels_qubit": nlevels_q,
        "nlevels_coupler": nlevels_c,
    }

    return DuffingModelBuildResult(
        hamiltonian_stack=H_stack,
        hamiltonian_kwargs=ham_kwargs,
        calibration=DuffingCalibrationResult(
            q0_w01=float(w0_arr[0]),
            q1_w01=float(w1_arr[0]),
            q0_alpha=float(alpha0_arr[0]),
            q1_alpha=float(alpha1_arr[0]),
        ),
        mode_parameters={
            "w0": np.asarray(w0_arr, dtype=float),
            "w1": np.asarray(w1_arr, dtype=float),
            "alpha0": np.asarray(alpha0_arr, dtype=float),
            "alpha1": np.asarray(alpha1_arr, dtype=float),
            "wc": np.asarray(wc_arr, dtype=float),
        },
    )


def _select_symbolic_harmonic_count(flux_values: np.ndarray, *, max_harmonics: int = 6) -> int:
    n_points = int(np.asarray(flux_values, dtype=float).size)
    if n_points < 3:
        return 1
    return max(1, min(int(max_harmonics), (n_points - 1) // 2, n_points // 4))


def _fourier_design_matrix(flux_values: np.ndarray, *, n_harmonics: int) -> np.ndarray:
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    theta = 2.0 * np.pi * flux_arr
    columns = [np.ones_like(theta)]
    for harmonic in range(1, int(n_harmonics) + 1):
        columns.append(np.cos(float(harmonic) * theta))
        columns.append(np.sin(float(harmonic) * theta))
    return np.column_stack(columns)


def _fourier_coefficient_names(*, n_harmonics: int) -> np.ndarray:
    labels = ["c0"]
    for harmonic in range(1, int(n_harmonics) + 1):
        labels.append(f"cos{harmonic}")
        labels.append(f"sin{harmonic}")
    return np.asarray(labels, dtype=str)


def _fit_fourier_parameter_coefficients(
    flux_values: np.ndarray,
    *,
    parameter_targets: Mapping[str, np.ndarray],
    n_harmonics: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    design = _fourier_design_matrix(flux_values, n_harmonics=n_harmonics)
    coefficients: dict[str, np.ndarray] = {}
    packed: list[np.ndarray] = []
    for name in ("w0", "w1", "alpha0", "alpha1"):
        target = np.asarray(parameter_targets[name], dtype=float).ravel()
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        coeff = np.asarray(beta, dtype=float)
        coefficients[name] = coeff
        packed.append(coeff)
    return np.concatenate(packed), coefficients


def _unpack_fourier_parameter_coefficients(
    packed: np.ndarray,
    *,
    coeff_size: int,
) -> dict[str, np.ndarray]:
    vector = np.asarray(packed, dtype=float).ravel()
    if vector.size != 4 * int(coeff_size):
        raise ValueError("Packed symbolic Duffing coefficient vector has unexpected size")
    return {
        "w0": np.asarray(vector[0 * coeff_size:1 * coeff_size], dtype=float),
        "w1": np.asarray(vector[1 * coeff_size:2 * coeff_size], dtype=float),
        "alpha0": np.asarray(vector[2 * coeff_size:3 * coeff_size], dtype=float),
        "alpha1": np.asarray(vector[3 * coeff_size:4 * coeff_size], dtype=float),
    }


def _evaluate_fourier_parameter_coefficients(
    flux_values: np.ndarray,
    *,
    coefficient_map: Mapping[str, np.ndarray],
    wc_values: np.ndarray,
) -> dict[str, np.ndarray]:
    coeff_size = int(np.asarray(next(iter(coefficient_map.values())), dtype=float).size)
    n_harmonics = max(0, (coeff_size - 1) // 2)
    design = _fourier_design_matrix(flux_values, n_harmonics=n_harmonics)
    parameters = {
        "w0": np.asarray(design @ np.asarray(coefficient_map["w0"], dtype=float).ravel(), dtype=float),
        "w1": np.asarray(design @ np.asarray(coefficient_map["w1"], dtype=float).ravel(), dtype=float),
        "alpha0": np.asarray(design @ np.asarray(coefficient_map["alpha0"], dtype=float).ravel(), dtype=float),
        "alpha1": np.asarray(design @ np.asarray(coefficient_map["alpha1"], dtype=float).ravel(), dtype=float),
        "wc": np.asarray(wc_values, dtype=float).ravel(),
    }
    return parameters


def fit_duffing_mode_parameters_to_reference(
    flux_values: np.ndarray,
    *,
    reference_dressed_stack: np.ndarray,
    system_params: SystemParams,
    coupler_frequency: CouplerFrequencyConfig,
    duffing_config: DuffingModelConfig,
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
        system_params=system_params,
        coupler_frequency=coupler_frequency,
        duffing_config=duffing_config,
        sweep_target=sweep_target,
    )
    ref_params = extract_model1_parameters_from_4x4_stack(reference_dressed_stack)
    n_q = int(duffing_config.hilbert_truncation.nlevels_qubit)
    n_c = int(duffing_config.hilbert_truncation.nlevels_coupler)

    fitted = {
        "w0": np.array(initial["w0"], copy=True, dtype=float),
        "w1": np.array(initial["w1"], copy=True, dtype=float),
        "alpha0": np.array(initial["alpha0"], copy=True, dtype=float),
        "alpha1": np.array(initial["alpha1"], copy=True, dtype=float),
        "wc": np.array(initial["wc"], copy=True, dtype=float),
    }

    for k in range(fitted["w0"].shape[0]):
        x0 = np.array(
            [
                float(initial["w0"][k]),
                float(initial["w1"][k]),
                float(initial["alpha0"][k]),
                float(initial["alpha1"][k]),
            ],
            dtype=float,
        )
        target = np.array(
            [
                float(ref_params["w0"][k]),
                float(ref_params["w1"][k]),
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
                    "w0": np.array([float(x[0])], dtype=float),
                    "w1": np.array([float(x[1])], dtype=float),
                    "alpha0": np.array([float(x[2])], dtype=float),
                    "alpha1": np.array([float(x[3])], dtype=float),
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
                    float(params["w0"][0]),
                    float(params["w1"][0]),
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
        fitted["w0"][k] = float(x_best[0])
        fitted["w1"][k] = float(x_best[1])
        fitted["alpha0"][k] = float(x_best[2])
        fitted["alpha1"][k] = float(x_best[3])

    return fitted


def fit_symbolic_duffing_mode_parameters_to_reference(
    flux_values: np.ndarray,
    *,
    reference_dressed_stack: np.ndarray,
    system_params: SystemParams,
    coupler_frequency: CouplerFrequencyConfig,
    duffing_config: DuffingModelConfig,
    sweep_target: str,
    n_candidate_states: int,
    selection_mode: str,
    max_harmonics: int = 6,
    pointwise_max_nfev: int = 80,
    refinement_max_nfev: int = 30,
    regularization_weight: float = 0.02,
) -> DuffingSymbolicParameterFitResult:
    """Fit a global symbolic Duffing surrogate over flux against reference observables."""
    from scipy.optimize import least_squares

    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    initial = _build_mode_parameter_arrays(
        flux_arr,
        system_params=system_params,
        coupler_frequency=coupler_frequency,
        duffing_config=duffing_config,
        sweep_target=sweep_target,
    )
    pointwise = fit_duffing_mode_parameters_to_reference(
        flux_arr,
        reference_dressed_stack=reference_dressed_stack,
        system_params=system_params,
        coupler_frequency=coupler_frequency,
        duffing_config=duffing_config,
        sweep_target=sweep_target,
        n_candidate_states=n_candidate_states,
        selection_mode=selection_mode,
        max_nfev=pointwise_max_nfev,
    )
    ref_params = extract_model1_parameters_from_4x4_stack(reference_dressed_stack)
    n_q = int(duffing_config.hilbert_truncation.nlevels_qubit)
    n_c = int(duffing_config.hilbert_truncation.nlevels_coupler)
    n_harmonics = _select_symbolic_harmonic_count(flux_arr, max_harmonics=max_harmonics)
    coeff_names = _fourier_coefficient_names(n_harmonics=n_harmonics)
    x0, coeff_init = _fit_fourier_parameter_coefficients(
        flux_arr,
        parameter_targets=pointwise,
        n_harmonics=n_harmonics,
    )
    coeff_size = int(coeff_names.size)

    obs_scale = {
        "w0": np.maximum(np.abs(np.asarray(ref_params["w0"], dtype=float).ravel()), 1.0),
        "w1": np.maximum(np.abs(np.asarray(ref_params["w1"], dtype=float).ravel()), 1.0),
        "J": np.maximum(np.abs(np.asarray(ref_params["J"], dtype=float).ravel()), 2e-2),
        "zeta": np.maximum(np.abs(np.asarray(ref_params["zeta"], dtype=float).ravel()), 2e-2),
    }
    latent_scale = {
        "w0": np.maximum(np.abs(np.asarray(pointwise["w0"], dtype=float).ravel()), 1.0),
        "w1": np.maximum(np.abs(np.asarray(pointwise["w1"], dtype=float).ravel()), 1.0),
        "alpha0": np.maximum(np.abs(np.asarray(pointwise["alpha0"], dtype=float).ravel()), 0.25),
        "alpha1": np.maximum(np.abs(np.asarray(pointwise["alpha1"], dtype=float).ravel()), 0.25),
    }

    def residual(packed: np.ndarray) -> np.ndarray:
        coeff_map = _unpack_fourier_parameter_coefficients(packed, coeff_size=coeff_size)
        symbolic_parameters = _evaluate_fourier_parameter_coefficients(
            flux_arr,
            coefficient_map=coeff_map,
            wc_values=initial["wc"],
        )
        candidate = build_duffing_model_stack_from_parameters(
            symbolic_parameters,
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
        obs_res = np.concatenate(
            [
                (np.asarray(params["w0"], dtype=float).ravel() - np.asarray(ref_params["w0"], dtype=float).ravel()) / obs_scale["w0"],
                (np.asarray(params["w1"], dtype=float).ravel() - np.asarray(ref_params["w1"], dtype=float).ravel()) / obs_scale["w1"],
                (np.asarray(params["J"], dtype=float).ravel() - np.asarray(ref_params["J"], dtype=float).ravel()) / obs_scale["J"],
                (np.asarray(params["zeta"], dtype=float).ravel() - np.asarray(ref_params["zeta"], dtype=float).ravel()) / obs_scale["zeta"],
            ]
        )
        reg_res = float(regularization_weight) * np.concatenate(
            [
                (np.asarray(symbolic_parameters["w0"], dtype=float).ravel() - np.asarray(pointwise["w0"], dtype=float).ravel()) / latent_scale["w0"],
                (np.asarray(symbolic_parameters["w1"], dtype=float).ravel() - np.asarray(pointwise["w1"], dtype=float).ravel()) / latent_scale["w1"],
                (np.asarray(symbolic_parameters["alpha0"], dtype=float).ravel() - np.asarray(pointwise["alpha0"], dtype=float).ravel()) / latent_scale["alpha0"],
                (np.asarray(symbolic_parameters["alpha1"], dtype=float).ravel() - np.asarray(pointwise["alpha1"], dtype=float).ravel()) / latent_scale["alpha1"],
            ]
        )
        return np.concatenate([obs_res, reg_res])

    result = least_squares(
        residual,
        x0=x0,
        max_nfev=int(refinement_max_nfev),
    )
    coeff_best = coeff_init if not result.success else _unpack_fourier_parameter_coefficients(result.x, coeff_size=coeff_size)
    fitted = _evaluate_fourier_parameter_coefficients(
        flux_arr,
        coefficient_map=coeff_best,
        wc_values=initial["wc"],
    )
    coefficient_names = {
        name: np.array(coeff_names, copy=True, dtype=str)
        for name in ("w0", "w1", "alpha0", "alpha1")
    }
    coefficients = {
        name: np.asarray(values, dtype=float)
        for name, values in coeff_best.items()
    }
    return DuffingSymbolicParameterFitResult(
        coefficient_names=coefficient_names,
        coefficients=coefficients,
        fitted_parameters=fitted,
    )


def build_duffing_model_stack(
    flux_values: np.ndarray,
    *,
    system_params: SystemParams,
    coupler_frequency: CouplerFrequencyConfig,
    duffing_config: DuffingModelConfig,
    sweep_target: str = "coupler",
) -> DuffingModelBuildResult:
    """Build a three-mode Duffing Hamiltonian stack from system + study config.

    The coupler is treated as a harmonic mode. Transmon parameters are either:
    - `fixed`: calibrated once at system-configured qubit flux biases, or
    - `analytic-per-flux`: transmon approximation at each sweep point, or
    - `per-flux`: recalibrated at every sweep point.
    - `fitted-static`: fitted against static reference observables in a higher-level step.
    - `symbolic-fitted-static`: global symbolic surrogate fitted against static reference observables.
    """
    if is_reference_calibrated_duffing_mode(duffing_config.calibration_mode):
        raise ValueError(
            "Duffing calibration_mode requires a reference-driven static calibration "
            "step; use fit_duffing_mode_parameters_to_reference or "
            "fit_symbolic_duffing_mode_parameters_to_reference and then "
            "build_duffing_model_stack_from_parameters instead."
        )

    mode_parameters = _build_mode_parameter_arrays(
        flux_values,
        system_params=system_params,
        coupler_frequency=coupler_frequency,
        duffing_config=duffing_config,
        sweep_target=sweep_target,
    )
    return build_duffing_model_stack_from_parameters(
        mode_parameters,
        system_params=system_params,
        duffing_config=duffing_config,
    )
