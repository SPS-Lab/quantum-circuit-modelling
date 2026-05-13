"""Reconstruct fitted lower-model parameters for downstream dynamic benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np

from models import evaluate_effective_parameter_fit, evaluate_symbolic_duffing_parameter_fit
from study_config import StudyConfig
from models.sweep import resolve_static_sweep_values


def effective_parameters_for_flux(
    static_result: Any,
    config: StudyConfig,
    flux_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the fitted effective model at target flux points."""
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    coeff_names = getattr(static_result, "effective_fit_coefficient_names", {})
    coeffs = getattr(static_result, "effective_fit_coefficients", {})
    required = ("w0", "w1", "J", "zeta")
    if not all(name in coeff_names and name in coeffs for name in required):
        return _interpolate_parameter_mapping(
            flux_reference=np.asarray(static_result.flux_values, dtype=float).ravel(),
            parameters_reference=getattr(static_result, "effective_parameters"),
            pulse_flux=flux_arr,
            keys=required,
        )
    fit_basis = str(
        getattr(static_result, "effective_fit_basis", "")
        or config.static_benchmark.effective_model.fit_basis
    )
    _, _, wc = resolve_static_sweep_values(
        flux_arr,
        system_params=config.system,
        coupler_frequency_config=config.static_benchmark.coupler_frequency,
        sweep_target=_sweep_target(static_result, config),
    )
    return evaluate_effective_parameter_fit(
        flux_arr,
        fit_basis=fit_basis,
        coefficient_names=coeff_names,
        coefficients=coeffs,
        coupler_frequency_values=np.asarray(wc, dtype=float),
    )


def duffing_mode_parameters_for_flux(
    static_result: Any,
    config: StudyConfig,
    flux_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate fitted Duffing parameters at target flux points when possible."""
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    sweep_target = _sweep_target(static_result, config)
    mode = str(
        getattr(static_result, "duffing_calibration_mode", "")
        or config.static_benchmark.duffing_model.calibration_mode
    ).strip().lower()
    if mode == "symbolic-fitted-static":
        symbolic_coeff_names = getattr(static_result, "duffing_symbolic_coefficient_names", {})
        symbolic_coefficients = getattr(static_result, "duffing_symbolic_coefficients", {})
        required = ("w0", "w1", "alpha0", "alpha1", "g0c", "g1c")
        if all(name in symbolic_coeff_names and name in symbolic_coefficients for name in required):
            parameters = evaluate_symbolic_duffing_parameter_fit(
                flux_arr,
                sweep_target=sweep_target,
                coefficient_names=symbolic_coeff_names,
                coefficients=symbolic_coefficients,
            )
            _, _, wc = resolve_static_sweep_values(
                flux_arr,
                system_params=config.system,
                coupler_frequency_config=config.static_benchmark.coupler_frequency,
                sweep_target=sweep_target,
            )
            parameters["wc"] = np.asarray(wc, dtype=float)
            return parameters
    return _interpolate_parameters(
        static_result,
        flux_arr,
        keys=("w0", "w1", "alpha0", "alpha1", "g0c", "g1c"),
        include_wc=True,
        config=config,
    )


def _interpolate_parameters(
    static_result: Any,
    flux_values: np.ndarray,
    *,
    keys: tuple[str, ...],
    include_wc: bool,
    config: StudyConfig,
) -> dict[str, np.ndarray]:
    x_ref = np.asarray(static_result.flux_values, dtype=float).ravel()
    x = np.asarray(flux_values, dtype=float).ravel()
    source = getattr(static_result, "duffing_mode_parameters")
    out = _interpolate_parameter_mapping(
        flux_reference=x_ref,
        parameters_reference=source,
        pulse_flux=x,
        keys=keys,
    )
    if include_wc:
        _, _, wc = resolve_static_sweep_values(
            x,
            system_params=config.system,
            coupler_frequency_config=config.static_benchmark.coupler_frequency,
            sweep_target=_sweep_target(static_result, config),
        )
        out["wc"] = np.asarray(wc, dtype=float)
    return out


def _sweep_target(static_result: Any, config: StudyConfig) -> str:
    return str(
        getattr(static_result, "sweep_target", "")
        or config.static_benchmark.flux_control.sweep_target
    )


def _interpolate_parameter_mapping(
    *,
    flux_reference: np.ndarray,
    parameters_reference: dict[str, np.ndarray],
    pulse_flux: np.ndarray,
    keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    x_ref = np.asarray(flux_reference, dtype=float).ravel()
    x = np.asarray(pulse_flux, dtype=float).ravel()
    return {
        key: np.interp(x, x_ref, np.asarray(parameters_reference[key], dtype=float).ravel())
        for key in keys
    }
