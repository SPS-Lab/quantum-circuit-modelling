"""Three-mode Duffing model construction for study benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping

import numpy as np

from models.dressed import (
    build_dressed_effective_computational_stack,
    extract_model1_parameters_from_4x4_stack,
)
from models.josephson import flux_dependent_EJ
from models.sweep import resolve_static_sweep_values
from models.three_mode import three_mode_hamiltonian
from runtime_utils import format_elapsed_compact, log_progress
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
        "g0c": np.full_like(flux_arr, float(system_params.interactions.g_0c), dtype=float),
        "g1c": np.full_like(flux_arr, float(system_params.interactions.g_1c), dtype=float),
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
    g0c_arr = np.asarray(
        mode_parameters.get(
            "g0c",
            np.full_like(w0_arr, float(system_params.interactions.g_0c), dtype=float),
        ),
        dtype=float,
    ).ravel()
    g1c_arr = np.asarray(
        mode_parameters.get(
            "g1c",
            np.full_like(w0_arr, float(system_params.interactions.g_1c), dtype=float),
        ),
        dtype=float,
    ).ravel()

    if not (
        w0_arr.shape
        == w1_arr.shape
        == alpha0_arr.shape
        == alpha1_arr.shape
        == wc_arr.shape
        == g0c_arr.shape
        == g1c_arr.shape
    ):
        raise ValueError("Duffing mode parameter arrays must all share the same shape")

    nlevels_q = int(duffing_config.hilbert_truncation.nlevels_qubit)
    nlevels_c = int(duffing_config.hilbert_truncation.nlevels_coupler)
    alpha_c = float(duffing_config.coupler_anharmonicity)

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
                g_0c=float(g0c_arr[k]),
                g_1c=float(g1c_arr[k]),
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
        "g_0c": float(g0c_arr[0]),
        "g_1c": float(g1c_arr[0]),
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
            "g0c": np.asarray(g0c_arr, dtype=float),
            "g1c": np.asarray(g1c_arr, dtype=float),
        },
    )


def _select_symbolic_harmonic_count(flux_values: np.ndarray, *, max_harmonics: int = 5) -> int:
    n_points = int(np.asarray(flux_values, dtype=float).size)
    if n_points < 3:
        return 1
    return max(1, min(int(max_harmonics), n_points - 1, n_points // 3))


def _cosine_design_matrix(flux_values: np.ndarray, *, n_harmonics: int) -> np.ndarray:
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    theta = 2.0 * np.pi * flux_arr
    columns = [np.ones_like(theta)]
    for harmonic in range(1, int(n_harmonics) + 1):
        columns.append(np.cos(float(harmonic) * theta))
    return np.column_stack(columns)


def _cosine_coefficient_names(*, n_harmonics: int) -> np.ndarray:
    labels = ["c0"]
    for harmonic in range(1, int(n_harmonics) + 1):
        labels.append(f"cos{harmonic}")
    return np.asarray(labels, dtype=str)


def _constant_design_matrix(flux_values: np.ndarray) -> np.ndarray:
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    return np.ones((flux_arr.size, 1), dtype=float)


def _fit_parameter_coefficients_from_designs(
    *,
    parameter_targets: Mapping[str, np.ndarray],
    design_map: Mapping[str, np.ndarray],
    parameter_order: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    coefficients: dict[str, np.ndarray] = {}
    packed: list[np.ndarray] = []
    for name in parameter_order:
        design = np.asarray(design_map[name], dtype=float)
        target = np.asarray(parameter_targets[name], dtype=float).ravel()
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        coeff = np.asarray(beta, dtype=float)
        coefficients[name] = coeff
        packed.append(coeff)
    return np.concatenate(packed), coefficients


def _unpack_parameter_coefficients(
    packed: np.ndarray,
    *,
    coefficient_sizes: Mapping[str, int],
    parameter_order: tuple[str, ...],
) -> dict[str, np.ndarray]:
    vector = np.asarray(packed, dtype=float).ravel()
    expected = int(sum(int(coefficient_sizes[name]) for name in parameter_order))
    if vector.size != expected:
        raise ValueError("Packed symbolic Duffing coefficient vector has unexpected size")
    out: dict[str, np.ndarray] = {}
    offset = 0
    for name in parameter_order:
        size = int(coefficient_sizes[name])
        out[name] = np.asarray(vector[offset:offset + size], dtype=float)
        offset += size
    return out


def _evaluate_parameter_coefficients_from_designs(
    *,
    coefficient_map: Mapping[str, np.ndarray],
    design_map: Mapping[str, np.ndarray],
    parameter_order: tuple[str, ...],
    base_parameters: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    parameters = {name: np.asarray(values, dtype=float).ravel() for name, values in base_parameters.items()}
    for name in parameter_order:
        design = np.asarray(design_map[name], dtype=float)
        coeff = np.asarray(coefficient_map[name], dtype=float).ravel()
        parameters[name] = np.asarray(design @ coeff, dtype=float)
    return parameters


def _reference_calibration_designs(
    flux_values: np.ndarray,
    *,
    sweep_target: str,
    n_harmonics: int,
) -> tuple[tuple[str, ...], dict[str, np.ndarray], dict[str, np.ndarray]]:
    cosine_design = _cosine_design_matrix(flux_values, n_harmonics=n_harmonics)
    constant_design = _constant_design_matrix(flux_values)
    cosine_names = _cosine_coefficient_names(n_harmonics=n_harmonics)
    constant_names = np.asarray(["c0"], dtype=str)
    target = str(sweep_target).strip().lower()

    if target == "q1":
        design_map = {
            "w0": constant_design,
            "w1": cosine_design,
            "alpha0": constant_design,
            "alpha1": cosine_design,
            "g0c": constant_design,
            "g1c": cosine_design,
        }
        coefficient_names = {
            "w0": constant_names,
            "w1": cosine_names,
            "alpha0": constant_names,
            "alpha1": cosine_names,
            "g0c": constant_names,
            "g1c": cosine_names,
        }
    elif target == "q0":
        design_map = {
            "w0": cosine_design,
            "w1": constant_design,
            "alpha0": cosine_design,
            "alpha1": constant_design,
            "g0c": cosine_design,
            "g1c": constant_design,
        }
        coefficient_names = {
            "w0": cosine_names,
            "w1": constant_names,
            "alpha0": cosine_names,
            "alpha1": constant_names,
            "g0c": cosine_names,
            "g1c": constant_names,
        }
    elif target == "coupler":
        design_map = {
            "w0": constant_design,
            "w1": constant_design,
            "alpha0": constant_design,
            "alpha1": constant_design,
            "g0c": constant_design,
            "g1c": constant_design,
        }
        coefficient_names = {
            name: constant_names
            for name in ("w0", "w1", "alpha0", "alpha1", "g0c", "g1c")
        }
    else:
        raise ValueError(f"Unsupported sweep_target {sweep_target!r}")

    parameter_order = ("w0", "w1", "alpha0", "alpha1", "g0c", "g1c")
    return parameter_order, design_map, coefficient_names


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
    progress_label: str | None = None,
    progress_interval_s: float = 30.0,
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
        "g0c": np.array(initial["g0c"], copy=True, dtype=float),
        "g1c": np.array(initial["g1c"], copy=True, dtype=float),
    }

    fit_started = time.perf_counter()
    interval = max(float(progress_interval_s), 1.0)
    if progress_label is not None:
        log_progress(f"{progress_label}: fitting {fitted['w0'].shape[0]} flux points")

    for k in range(fitted["w0"].shape[0]):
        point_started = time.perf_counter()
        point_last_progress = point_started
        residual_calls = 0
        point_label = None if progress_label is None else f"{progress_label}: point {k + 1}/{fitted['w0'].shape[0]}"
        if point_label is not None:
            log_progress(point_label)
        x0_raw = np.array(
            [
                float(initial["w0"][k]),
                float(initial["w1"][k]),
                float(initial["alpha0"][k]),
                float(initial["alpha1"][k]),
            ],
            dtype=float,
        )
        lower_bounds = np.array([0.0, 0.0, -5.0, -5.0], dtype=float)
        upper_bounds = np.array([20.0, 20.0, -1e-6, -1e-6], dtype=float)
        # Small-ncut or aggressively truncated initial spectra can yield
        # non-physical positive anharmonicities. Project the optimizer seed
        # into the feasible box so reference-driven modes remain usable.
        x0 = np.clip(x0_raw, lower_bounds + 1e-12, upper_bounds - 1e-12)
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
            nonlocal residual_calls, point_last_progress
            residual_calls += 1
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
            now = time.perf_counter()
            if point_label is not None and (now - point_last_progress) >= interval:
                elapsed = format_elapsed_compact(now - point_started)
                log_progress(f"{point_label} still optimizing after {elapsed} (residual evals={residual_calls})")
                point_last_progress = now
            return np.concatenate([obs_res, reg_res])

        result = least_squares(
            residual,
            x0=x0,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=int(max_nfev),
        )
        x_best = x0 if not result.success else np.asarray(result.x, dtype=float)
        fitted["w0"][k] = float(x_best[0])
        fitted["w1"][k] = float(x_best[1])
        fitted["alpha0"][k] = float(x_best[2])
        fitted["alpha1"][k] = float(x_best[3])
        if point_label is not None:
            point_elapsed = format_elapsed_compact(time.perf_counter() - point_started)
            status = "converged" if result.success else "stopped"
            log_progress(
                f"{point_label} {status} in {point_elapsed} (nfev={int(getattr(result, 'nfev', residual_calls))})"
            )

    if progress_label is not None:
        total_elapsed = format_elapsed_compact(time.perf_counter() - fit_started)
        log_progress(f"{progress_label}: finished pointwise fit in {total_elapsed}")

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
    max_harmonics: int,
    pointwise_max_nfev: int,
    refinement_max_nfev: int,
    regularization_weight: float,
    progress_label: str | None = None,
    progress_interval_s: float = 30.0,
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
        progress_label=(
            None if progress_label is None else f"{progress_label}: pointwise seed fit"
        ),
        progress_interval_s=progress_interval_s,
    )
    pointwise_targets = {
        "w0": np.asarray(pointwise["w0"], dtype=float).ravel(),
        "w1": np.asarray(pointwise["w1"], dtype=float).ravel(),
        "alpha0": np.asarray(pointwise["alpha0"], dtype=float).ravel(),
        "alpha1": np.asarray(pointwise["alpha1"], dtype=float).ravel(),
        "g0c": np.asarray(initial["g0c"], dtype=float).ravel(),
        "g1c": np.asarray(initial["g1c"], dtype=float).ravel(),
    }
    ref_params = extract_model1_parameters_from_4x4_stack(reference_dressed_stack)
    n_q = int(duffing_config.hilbert_truncation.nlevels_qubit)
    n_c = int(duffing_config.hilbert_truncation.nlevels_coupler)
    # Calibrate the swept-side qubit and coupling more flexibly than the parked side.
    n_harmonics = _select_symbolic_harmonic_count(flux_arr, max_harmonics=max_harmonics)
    if progress_label is not None:
        log_progress(
            f"{progress_label}: global symbolic refinement with {n_harmonics} harmonics over {flux_arr.size} flux points"
        )
    parameter_order, design_map, coefficient_names = _reference_calibration_designs(
        flux_arr,
        sweep_target=sweep_target,
        n_harmonics=n_harmonics,
    )
    x0, coeff_init = _fit_parameter_coefficients_from_designs(
        parameter_targets=pointwise_targets,
        design_map=design_map,
        parameter_order=parameter_order,
    )
    coefficient_sizes = {name: int(np.asarray(design_map[name], dtype=float).shape[1]) for name in parameter_order}

    obs_scale = {
        "w0": np.maximum(np.abs(np.asarray(ref_params["w0"], dtype=float).ravel()), 1.0),
        "w1": np.maximum(np.abs(np.asarray(ref_params["w1"], dtype=float).ravel()), 1.0),
        "J": np.maximum(np.abs(np.asarray(ref_params["J"], dtype=float).ravel()), 2e-2),
        "zeta": np.maximum(np.abs(np.asarray(ref_params["zeta"], dtype=float).ravel()), 2e-2),
    }
    latent_scale = {
        "w0": np.maximum(np.abs(np.asarray(pointwise_targets["w0"], dtype=float).ravel()), 1.0),
        "w1": np.maximum(np.abs(np.asarray(pointwise_targets["w1"], dtype=float).ravel()), 1.0),
        "alpha0": np.maximum(np.abs(np.asarray(pointwise_targets["alpha0"], dtype=float).ravel()), 0.25),
        "alpha1": np.maximum(np.abs(np.asarray(pointwise_targets["alpha1"], dtype=float).ravel()), 0.25),
        "g0c": np.maximum(np.abs(np.asarray(pointwise_targets["g0c"], dtype=float).ravel()), 1e-3),
        "g1c": np.maximum(np.abs(np.asarray(pointwise_targets["g1c"], dtype=float).ravel()), 1e-3),
    }

    refinement_started = time.perf_counter()
    refinement_last_progress = refinement_started
    refinement_calls = 0

    def residual(packed: np.ndarray) -> np.ndarray:
        nonlocal refinement_calls, refinement_last_progress
        refinement_calls += 1
        coeff_map = _unpack_parameter_coefficients(
            packed,
            coefficient_sizes=coefficient_sizes,
            parameter_order=parameter_order,
        )
        symbolic_parameters = _evaluate_parameter_coefficients_from_designs(
            coefficient_map=coeff_map,
            design_map=design_map,
            parameter_order=parameter_order,
            base_parameters=initial,
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
                (np.asarray(symbolic_parameters["w0"], dtype=float).ravel() - np.asarray(pointwise_targets["w0"], dtype=float).ravel()) / latent_scale["w0"],
                (np.asarray(symbolic_parameters["w1"], dtype=float).ravel() - np.asarray(pointwise_targets["w1"], dtype=float).ravel()) / latent_scale["w1"],
                (np.asarray(symbolic_parameters["alpha0"], dtype=float).ravel() - np.asarray(pointwise_targets["alpha0"], dtype=float).ravel()) / latent_scale["alpha0"],
                (np.asarray(symbolic_parameters["alpha1"], dtype=float).ravel() - np.asarray(pointwise_targets["alpha1"], dtype=float).ravel()) / latent_scale["alpha1"],
                (np.asarray(symbolic_parameters["g0c"], dtype=float).ravel() - np.asarray(pointwise_targets["g0c"], dtype=float).ravel()) / latent_scale["g0c"],
                (np.asarray(symbolic_parameters["g1c"], dtype=float).ravel() - np.asarray(pointwise_targets["g1c"], dtype=float).ravel()) / latent_scale["g1c"],
            ]
        )
        now = time.perf_counter()
        if progress_label is not None and (now - refinement_last_progress) >= max(float(progress_interval_s), 1.0):
            elapsed = format_elapsed_compact(now - refinement_started)
            log_progress(f"{progress_label}: global refinement still running after {elapsed} (residual evals={refinement_calls})")
            refinement_last_progress = now
        return np.concatenate([obs_res, reg_res])

    result = least_squares(
        residual,
        x0=x0,
        max_nfev=int(refinement_max_nfev),
    )
    coeff_best = coeff_init if not result.success else _unpack_parameter_coefficients(
        result.x,
        coefficient_sizes=coefficient_sizes,
        parameter_order=parameter_order,
    )
    fitted = _evaluate_parameter_coefficients_from_designs(
        coefficient_map=coeff_best,
        design_map=design_map,
        parameter_order=parameter_order,
        base_parameters=initial,
    )
    coefficients = {
        name: np.asarray(values, dtype=float)
        for name, values in coeff_best.items()
    }
    if progress_label is not None:
        refinement_elapsed = format_elapsed_compact(time.perf_counter() - refinement_started)
        status = "converged" if result.success else "stopped"
        log_progress(
            f"{progress_label}: global symbolic refinement {status} in {refinement_elapsed} "
            f"(nfev={int(getattr(result, 'nfev', refinement_calls))})"
        )
    return DuffingSymbolicParameterFitResult(
        coefficient_names={
            name: np.array(coefficient_names[name], copy=True, dtype=str)
            for name in parameter_order
        },
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
