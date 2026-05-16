"""Reference-driven extraction and fitting for the effective two-level model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from models.dressed import extract_effective_model_parameters_from_4x4_stack
from models.effective_model import (
    _even_three_harmonic_design_matrix,
    _single_harmonic_design_matrix,
)


@dataclass(frozen=True)
class EffectiveParameterFitResult:
    coefficient_names: dict[str, tuple[str, ...]]
    coefficients: dict[str, np.ndarray]
    fitted_parameters: dict[str, np.ndarray]


@dataclass(frozen=True)
class EffectiveModelDerivationResult:
    extracted_parameters: dict[str, np.ndarray]
    parameter_fit: EffectiveParameterFitResult


def fit_single_harmonic_parameters(
    flux_values: np.ndarray,
    *,
    extracted_parameters: Mapping[str, np.ndarray],
) -> EffectiveParameterFitResult:
    """Fit ``x0 + a cos(2*pi*psi) + b sin(2*pi*psi)`` for each effective parameter."""
    design = _single_harmonic_design_matrix(flux_values)
    coeff_names = ("x0", "a", "b")
    coeff_name_map: dict[str, tuple[str, ...]] = {}
    coeff: dict[str, np.ndarray] = {}
    fitted: dict[str, np.ndarray] = {}
    for name in ("w0", "w1", "J", "zeta"):
        y = np.asarray(extracted_parameters[name], dtype=float).ravel()
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        coeff_name_map[name] = coeff_names
        coeff[name] = beta
        fitted[name] = design @ beta
    return EffectiveParameterFitResult(
        coefficient_names=coeff_name_map,
        coefficients=coeff,
        fitted_parameters=fitted,
    )


def _fit_even_three_harmonic_parameter(
    flux_values: np.ndarray,
    y: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    design = _even_three_harmonic_design_matrix(flux_values)
    y_arr = np.asarray(y, dtype=float).ravel()
    beta, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    return ("x0", "a1", "a2", "a3"), np.asarray(beta, dtype=float), np.asarray(design @ beta, dtype=float)


def _fit_magnitude_exchange_single_parameter(
    *,
    w0: np.ndarray,
    w1: np.ndarray,
    wc: np.ndarray,
    y: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    delta1 = np.asarray(w0, dtype=float).ravel() - np.asarray(wc, dtype=float).ravel()
    delta2 = np.asarray(w1, dtype=float).ravel() - np.asarray(wc, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    coeff_names = ("gamma", "c0", "c_r1", "c_r2", "c_prod", "c_r1_sq", "c_r2_sq")

    gamma_grid = np.geomspace(1e-3, 5.0, 600)
    best_coeffs = np.empty(len(coeff_names), dtype=float)
    best_fit = np.empty_like(y_arr)
    best_rmse = float("inf")

    for gamma in gamma_grid:
        r1 = 1.0 / np.sqrt(delta1 * delta1 + gamma * gamma)
        r2 = 1.0 / np.sqrt(delta2 * delta2 + gamma * gamma)
        design = np.column_stack(
            [
                np.ones_like(delta1),
                r1,
                r2,
                r1 * r2,
                r1 * r1,
                r2 * r2,
            ]
        )
        linear_coeffs, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
        fitted = design @ linear_coeffs
        rmse = float(np.sqrt(np.mean((fitted - y_arr) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_coeffs = np.asarray([float(gamma), *linear_coeffs.tolist()], dtype=float)
            best_fit = np.asarray(fitted, dtype=float)

    return coeff_names, best_coeffs, best_fit


def fit_magnitude_exchange_parameters(
    flux_values: np.ndarray,
    *,
    extracted_parameters: Mapping[str, np.ndarray],
    coupler_frequency_values: np.ndarray,
) -> EffectiveParameterFitResult:
    """Fit ``w0,w1`` with even harmonics and ``J,zeta`` with a magnitude-exchange surrogate."""
    coeff_name_map: dict[str, tuple[str, ...]] = {}
    coeff: dict[str, np.ndarray] = {}
    fitted: dict[str, np.ndarray] = {}

    for name in ("w0", "w1"):
        coeff_names, beta, y_fit = _fit_even_three_harmonic_parameter(
            flux_values,
            np.asarray(extracted_parameters[name], dtype=float).ravel(),
        )
        coeff_name_map[name] = coeff_names
        coeff[name] = beta
        fitted[name] = y_fit

    w0_ref = np.asarray(fitted["w0"], dtype=float).ravel()
    w1_ref = np.asarray(fitted["w1"], dtype=float).ravel()
    wc_ref = np.asarray(coupler_frequency_values, dtype=float).ravel()
    if wc_ref.shape != w0_ref.shape:
        raise ValueError(
            "coupler_frequency_values must have the same shape as extracted effective parameter arrays"
        )

    for name in ("J", "zeta"):
        coeff_names, beta, y_fit = _fit_magnitude_exchange_single_parameter(
            w0=w0_ref,
            w1=w1_ref,
            wc=wc_ref,
            y=np.asarray(extracted_parameters[name], dtype=float).ravel(),
        )
        coeff_name_map[name] = coeff_names
        coeff[name] = beta
        fitted[name] = y_fit

    return EffectiveParameterFitResult(
        coefficient_names=coeff_name_map,
        coefficients=coeff,
        fitted_parameters=fitted,
    )


def derive_effective_model_from_dressed_stack(
    flux_values: np.ndarray,
    *,
    dressed_stack: np.ndarray,
    fit_basis: str,
    coupler_frequency_values: np.ndarray | None = None,
) -> EffectiveModelDerivationResult:
    """Extract effective parameters from a dressed stack and fit compact flux laws."""
    extracted = extract_effective_model_parameters_from_4x4_stack(dressed_stack)
    if fit_basis == "single-harmonic":
        parameter_fit = fit_single_harmonic_parameters(flux_values, extracted_parameters=extracted)
    elif fit_basis == "magnitude-exchange-like":
        if coupler_frequency_values is None:
            raise ValueError("coupler_frequency_values are required for fit_basis 'magnitude-exchange-like'")
        parameter_fit = fit_magnitude_exchange_parameters(
            flux_values,
            extracted_parameters=extracted,
            coupler_frequency_values=coupler_frequency_values,
        )
    else:
        raise ValueError(f"Unsupported fit_basis {fit_basis!r}")
    return EffectiveModelDerivationResult(
        extracted_parameters=extracted,
        parameter_fit=parameter_fit,
    )
