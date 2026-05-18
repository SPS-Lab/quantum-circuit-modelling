"""Reference-driven and scqubits-backed Duffing calibration workflows."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping

import numpy as np

from models.dressed import (
    build_dressed_effective_computational_stack,
    extract_effective_model_parameters_from_4x4_stack,
)
from models.duffing_model import (
    _assemble_fixed_bus_duffing_mode_parameters,
    _evaluate_parameter_coefficients_from_designs,
    _reference_calibration_designs,
    _select_symbolic_harmonic_count,
    build_duffing_model_stack_from_parameters,
)
from models.josephson import flux_dependent_EJ
from models.sweep import resolve_static_sweep_values
from runtime_utils import format_elapsed_compact, log_progress
from study_config import DuffingModelConfig, SystemParams


@dataclass(frozen=True)
class DuffingSymbolicParameterFitResult:
    coefficient_names: dict[str, np.ndarray]
    coefficients: dict[str, np.ndarray]


def _transmon_w01_alpha(
    *,
    EJmax: float,
    EC: float,
    d: float,
    flux: float,
    ng: float,
    ncut: int,
    truncated_dim: int,
) -> tuple[float, float]:
    try:
        import scqubits as scq
    except Exception as exc:  # pragma: no cover - import guard only
        raise ImportError("scqubits import failed while calibrating Duffing parameters") from exc

    transmon = scq.TunableTransmon(
        EJmax=float(EJmax),
        EC=float(EC),
        d=float(d),
        flux=float(flux),
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
    EC: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate transmon spectral parameters from EJ, EC."""
    EJ_arr = np.asarray(EJ, dtype=float)
    ec = float(EC)
    w01 = np.sqrt(8.0 * EJ_arr * ec) - ec
    alpha = np.full_like(EJ_arr, -ec, dtype=float)
    return w01, alpha


def _build_mode_parameter_arrays(
    flux_values: np.ndarray,
    *,
    system_params: SystemParams,
    duffing_config: DuffingModelConfig,
    sweep_target: str,
) -> dict[str, np.ndarray]:
    """Build per-flux Duffing mode arrays, including transmon spectral extraction when needed."""
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    q0_flux_arr, q1_flux_arr, wc_arr = resolve_static_sweep_values(
        flux_arr,
        system_params=system_params,
        sweep_target=sweep_target,
    )

    ncut = int(duffing_config.transmon_spectral_extraction.ncut)
    trunc_dim = int(duffing_config.transmon_spectral_extraction.truncated_dim)
    calibration_mode = str(duffing_config.calibration_mode).strip().lower()

    if calibration_mode in {"per-flux", "fitted-static", "symbolic-fitted-static"}:
        w0_arr = np.empty_like(flux_arr, dtype=float)
        w1_arr = np.empty_like(flux_arr, dtype=float)
        alpha0_arr = np.empty_like(flux_arr, dtype=float)
        alpha1_arr = np.empty_like(flux_arr, dtype=float)
        for k in range(flux_arr.shape[0]):
            w0_arr[k], alpha0_arr[k] = _transmon_w01_alpha(
                EJmax=system_params.q0.EJmax,
                EC=system_params.q0.EC,
                d=system_params.q0.d,
                flux=float(q0_flux_arr[k]),
                ng=system_params.q0.ng,
                ncut=ncut,
                truncated_dim=trunc_dim,
            )
            w1_arr[k], alpha1_arr[k] = _transmon_w01_alpha(
                EJmax=system_params.q1.EJmax,
                EC=system_params.q1.EC,
                d=system_params.q1.d,
                flux=float(q1_flux_arr[k]),
                ng=system_params.q1.ng,
                ncut=ncut,
                truncated_dim=trunc_dim,
            )
    elif calibration_mode == "analytic-per-flux":
        EJ0_arr = np.asarray(
            flux_dependent_EJ(
                EJ_max=system_params.q0.EJmax,
                flux_bias=q0_flux_arr,
                d=system_params.q0.d,
            ),
            dtype=float,
        ).ravel()
        EJ1_arr = np.asarray(
            flux_dependent_EJ(
                EJ_max=system_params.q1.EJmax,
                flux_bias=q1_flux_arr,
                d=system_params.q1.d,
            ),
            dtype=float,
        ).ravel()
        w0_arr, alpha0_arr = _transmon_analytic_w01_alpha(EJ=EJ0_arr, EC=system_params.q0.EC)
        w1_arr, alpha1_arr = _transmon_analytic_w01_alpha(EJ=EJ1_arr, EC=system_params.q1.EC)
    elif calibration_mode == "fixed":
        w0_ref, alpha0_ref = _transmon_w01_alpha(
            EJmax=system_params.q0.EJmax,
            EC=system_params.q0.EC,
            d=system_params.q0.d,
            flux=system_params.q0.flux,
            ng=system_params.q0.ng,
            ncut=ncut,
            truncated_dim=trunc_dim,
        )
        w1_ref, alpha1_ref = _transmon_w01_alpha(
            EJmax=system_params.q1.EJmax,
            EC=system_params.q1.EC,
            d=system_params.q1.d,
            flux=system_params.q1.flux,
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


def fit_duffing_mode_parameters_to_reference(
    flux_values: np.ndarray,
    *,
    reference_dressed_stack: np.ndarray,
    system_params: SystemParams,
    duffing_config: DuffingModelConfig,
    sweep_target: str,
    n_candidate_states: int,
    selection_mode: str,
    initial_mode_parameters: Mapping[str, np.ndarray] | None = None,
    regularization_weight: float = 0.10,
    max_nfev: int = 80,
    progress_label: str | None = None,
    progress_interval_s: float = 30.0,
) -> dict[str, np.ndarray]:
    """Fit latent Duffing parameters so static dressed observables track a reference stack."""
    from scipy.optimize import least_squares

    if initial_mode_parameters is None:
        initial = _build_mode_parameter_arrays(
            flux_values,
            system_params=system_params,
            duffing_config=duffing_config,
            sweep_target=sweep_target,
        )
    else:
        initial = {
            key: np.asarray(values, dtype=float).ravel()
            for key, values in initial_mode_parameters.items()
        }
    ref_params = extract_effective_model_parameters_from_4x4_stack(reference_dressed_stack)
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
        residual_calls = 0
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
            nonlocal residual_calls
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
            params = extract_effective_model_parameters_from_4x4_stack(dressed)
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

    if progress_label is not None:
        total_elapsed = format_elapsed_compact(time.perf_counter() - fit_started)
        log_progress(f"{progress_label}: finished pointwise fit in {total_elapsed}")

    return fitted


def fit_symbolic_duffing_mode_parameters_to_reference(
    flux_values: np.ndarray,
    *,
    reference_dressed_stack: np.ndarray,
    system_params: SystemParams,
    duffing_config: DuffingModelConfig,
    sweep_target: str,
    n_candidate_states: int,
    selection_mode: str,
    max_harmonics_w: int,
    max_harmonics_alpha: int,
    max_harmonics_g: int,
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
        duffing_config=duffing_config,
        sweep_target=sweep_target,
    )
    pointwise = fit_duffing_mode_parameters_to_reference(
        flux_arr,
        reference_dressed_stack=reference_dressed_stack,
        system_params=system_params,
        duffing_config=duffing_config,
        sweep_target=sweep_target,
        n_candidate_states=n_candidate_states,
        selection_mode=selection_mode,
        initial_mode_parameters=initial,
        max_nfev=pointwise_max_nfev,
        progress_label=None if progress_label is None else f"{progress_label}: pointwise seed fit",
        progress_interval_s=progress_interval_s,
    )
    pointwise_targets = {
        "w0": np.asarray(pointwise["w0"], dtype=float).ravel(),
        "w1": np.asarray(pointwise["w1"], dtype=float).ravel(),
        "alpha0": np.asarray(pointwise["alpha0"], dtype=float).ravel(),
        "alpha1": np.asarray(pointwise["alpha1"], dtype=float).ravel(),
        "wc": np.asarray(initial["wc"], dtype=float).ravel(),
        "g0c": np.asarray(initial["g0c"], dtype=float).ravel(),
        "g1c": np.asarray(initial["g1c"], dtype=float).ravel(),
    }
    ref_params = extract_effective_model_parameters_from_4x4_stack(reference_dressed_stack)
    n_q = int(duffing_config.hilbert_truncation.nlevels_qubit)
    n_c = int(duffing_config.hilbert_truncation.nlevels_coupler)
    n_harmonics_w = _select_symbolic_harmonic_count(
        flux_arr,
        max_harmonics=max_harmonics_w,
    )
    n_harmonics_alpha = _select_symbolic_harmonic_count(
        flux_arr,
        max_harmonics=max_harmonics_alpha,
    )
    n_harmonics_g = _select_symbolic_harmonic_count(
        flux_arr,
        max_harmonics=max_harmonics_g,
    )
    if progress_label is not None:
        log_progress(
            f"{progress_label}: global symbolic refinement with "
            f"w harmonics={n_harmonics_w}, alpha harmonics={n_harmonics_alpha}, "
            f"g harmonics={n_harmonics_g} "
            f"over {flux_arr.size} flux points"
        )
    parameter_order, design_map, coefficient_names = _reference_calibration_designs(
        flux_arr,
        sweep_target=sweep_target,
        n_harmonics_w=n_harmonics_w,
        n_harmonics_alpha=n_harmonics_alpha,
        n_harmonics_g=n_harmonics_g,
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
        "wc": np.maximum(np.abs(np.asarray(pointwise_targets["wc"], dtype=float).ravel()), 1.0),
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
        )
        full_mode_parameters = _assemble_fixed_bus_duffing_mode_parameters(
            symbolic_parameters,
            system_params=system_params,
        )
        candidate = build_duffing_model_stack_from_parameters(
            full_mode_parameters,
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
        params = extract_effective_model_parameters_from_4x4_stack(dressed)
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
                (np.asarray(full_mode_parameters["w0"], dtype=float).ravel() - np.asarray(pointwise_targets["w0"], dtype=float).ravel()) / latent_scale["w0"],
                (np.asarray(full_mode_parameters["w1"], dtype=float).ravel() - np.asarray(pointwise_targets["w1"], dtype=float).ravel()) / latent_scale["w1"],
                (np.asarray(full_mode_parameters["alpha0"], dtype=float).ravel() - np.asarray(pointwise_targets["alpha0"], dtype=float).ravel()) / latent_scale["alpha0"],
                (np.asarray(full_mode_parameters["alpha1"], dtype=float).ravel() - np.asarray(pointwise_targets["alpha1"], dtype=float).ravel()) / latent_scale["alpha1"],
                (np.asarray(full_mode_parameters["wc"], dtype=float).ravel() - np.asarray(pointwise_targets["wc"], dtype=float).ravel()) / latent_scale["wc"],
                (np.asarray(full_mode_parameters["g0c"], dtype=float).ravel() - np.asarray(pointwise_targets["g0c"], dtype=float).ravel()) / latent_scale["g0c"],
                (np.asarray(full_mode_parameters["g1c"], dtype=float).ravel() - np.asarray(pointwise_targets["g1c"], dtype=float).ravel()) / latent_scale["g1c"],
            ]
        )
        now = time.perf_counter()
        if progress_label is not None and (now - refinement_last_progress) >= max(float(progress_interval_s), 1.0):
            elapsed = format_elapsed_compact(now - refinement_started)
            log_progress(
                f"{progress_label}: global refinement still running after {elapsed} (residual evals={refinement_calls})"
            )
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
    )
