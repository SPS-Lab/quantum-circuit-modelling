"""Effective two-level model utilities for study benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from models.dressed import extract_effective_model_parameters_from_4x4_stack
from toolkit.helpers import I2, px, py, pz


@dataclass(frozen=True)
class EffectiveParameterFitResult:
    coefficient_names: dict[str, tuple[str, ...]]
    coefficients: dict[str, np.ndarray]
    fitted_parameters: dict[str, np.ndarray]


@dataclass(frozen=True)
class EffectiveModelDerivationResult:
    extracted_parameters: dict[str, np.ndarray]
    parameter_fit: EffectiveParameterFitResult



def _coeff_for_ham(c: np.ndarray | float) -> np.ndarray | float:
    """Scalar * (4,4) stays (4,4); 1D (n,) -> (n,1,1) for batched eigensolvers."""
    c = np.asarray(c)
    if c.ndim == 0:
        return c
    if c.ndim == 1:
        return c[:, np.newaxis, np.newaxis]
    raise ValueError("w0, w1, J, zeta must be scalars or 1D arrays")



def heff(
    *,
    w0: np.ndarray | float,
    w1: np.ndarray | float,
    J: np.ndarray | float,
    zeta: np.ndarray | float) -> np.ndarray:
    """Build effective model Hamiltonian in spin convention."""
    w0c = _coeff_for_ham(w0)
    w1c = _coeff_for_ham(w1)
    jc = _coeff_for_ham(J)
    zc = _coeff_for_ham(zeta)
    return (
        0.5 * w1c * np.kron(pz, I2)
        + 0.5 * w0c * np.kron(I2, pz)
        + jc * (np.kron(px, px) + np.kron(py, py))
        + 0.25 * zc * np.kron(pz, pz)
    )



def heff_spin_to_lab_hamiltonian(
    H_eff: np.ndarray,
    *,
    w0: np.ndarray,
    w1: np.ndarray
) -> np.ndarray:
    """Convert model-1 ``(w/2) sigma_z`` convention to lab-frame ``w n``."""
    H_eff_arr = np.asarray(H_eff, dtype=complex)
    w0_arr = np.asarray(w0, dtype=float).reshape(-1)
    w1_arr = np.asarray(w1, dtype=float).reshape(-1)

    w0_b = w0_arr[:, np.newaxis, np.newaxis]
    w1_b = w1_arr[:, np.newaxis, np.newaxis]
    pz_q1 = np.kron(pz, I2)
    pz_q0 = np.kron(I2, pz)
    eye4 = np.eye(4, dtype=complex)
    return H_eff_arr + 0.5 * (w0_b + w1_b) * eye4 - w0_b * pz_q0 - w1_b * pz_q1



def _single_harmonic_design_matrix(flux_values: np.ndarray) -> np.ndarray:
    theta = 2.0 * np.pi * np.asarray(flux_values, dtype=float).ravel()
    return np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])


def _even_three_harmonic_design_matrix(flux_values: np.ndarray) -> np.ndarray:
    theta = 2.0 * np.pi * np.asarray(flux_values, dtype=float).ravel()
    return np.column_stack(
        [
            np.ones_like(theta),
            np.cos(theta),
            np.cos(2.0 * theta),
            np.cos(3.0 * theta),
        ]
    )



def fit_single_harmonic_parameters(
    flux_values: np.ndarray,
    *,
    extracted_parameters: Mapping[str, np.ndarray],
) -> EffectiveParameterFitResult:
    """Fit ``x0 + a cos(2*pi*psi) + b sin(2*pi*psi)`` for each effective parameter."""
    X = _single_harmonic_design_matrix(flux_values)
    coeff_names = ("x0", "a", "b")
    coeff_name_map: dict[str, tuple[str, ...]] = {}
    coeff: dict[str, np.ndarray] = {}
    fitted: dict[str, np.ndarray] = {}
    for name in ("w0", "w1", "J", "zeta"):
        y = np.asarray(extracted_parameters[name], dtype=float).ravel()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        coeff_name_map[name] = coeff_names
        coeff[name] = beta
        fitted[name] = X @ beta
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
    best_gamma = float("nan")
    best_coeffs = np.empty(len(coeff_names), dtype=float)
    best_fit = np.empty_like(y_arr)
    best_rmse = float("inf")

    for gamma in gamma_grid:
        r1 = 1.0 / np.sqrt(delta1 * delta1 + gamma * gamma)
        r2 = 1.0 / np.sqrt(delta2 * delta2 + gamma * gamma)
        # Keep the detuning-mediated structure but allow asymmetric and nonlinear
        # exchange responses across the swept and parked qubit branches.
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
            best_gamma = float(gamma)
            best_coeffs = np.asarray(
                [best_gamma, *linear_coeffs.tolist()],
                dtype=float,
            )
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


def evaluate_effective_parameter_fit(
    flux_values: np.ndarray,
    *,
    fit_basis: str,
    coefficient_names: Mapping[str, tuple[str, ...] | np.ndarray],
    coefficients: Mapping[str, np.ndarray],
    coupler_frequency_values: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate fitted effective-model parameters at flux points."""
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    out: dict[str, np.ndarray] = {}

    if fit_basis == "single-harmonic":
        design = _single_harmonic_design_matrix(flux_arr)
        for name in ("w0", "w1", "J", "zeta"):
            beta = np.asarray(coefficients[name], dtype=float).ravel()
            out[name] = np.asarray(design @ beta, dtype=float)
        return out

    if fit_basis == "magnitude-exchange-like":
        design_w = _even_three_harmonic_design_matrix(flux_arr)
        for name in ("w0", "w1"):
            beta = np.asarray(coefficients[name], dtype=float).ravel()
            out[name] = np.asarray(design_w @ beta, dtype=float)

        if coupler_frequency_values is None:
            raise ValueError(
                "coupler_frequency_values are required to evaluate fit_basis "
                "'magnitude-exchange-like'"
            )
        wc = np.asarray(coupler_frequency_values, dtype=float).ravel()
        if wc.shape != out["w0"].shape:
            raise ValueError("coupler_frequency_values must match the evaluated flux grid shape")

        delta1 = out["w0"] - wc
        delta2 = out["w1"] - wc
        for name in ("J", "zeta"):
            beta = np.asarray(coefficients[name], dtype=float).ravel()
            if beta.size != 7:
                raise ValueError(
                    f"Expected 7 magnitude-exchange coefficients for {name!r}, got {beta.size}"
                )
            gamma, c0, c_r1, c_r2, c_prod, c_r1_sq, c_r2_sq = [float(value) for value in beta]
            r1 = 1.0 / np.sqrt(delta1 * delta1 + gamma * gamma)
            r2 = 1.0 / np.sqrt(delta2 * delta2 + gamma * gamma)
            out[name] = np.asarray(
                c0 + c_r1 * r1 + c_r2 * r2 + c_prod * (r1 * r2) + c_r1_sq * (r1 * r1) + c_r2_sq * (r2 * r2),
                dtype=float,
            )
        return out

    raise ValueError(f"Unsupported fit_basis {fit_basis!r}")



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



def build_effective_hamiltonian_stack(parameters: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build lab-frame effective Hamiltonian stack from ``w0,w1,J,zeta`` arrays."""
    w0 = np.asarray(parameters["w0"], dtype=float).ravel()
    w1 = np.asarray(parameters["w1"], dtype=float).ravel()
    j = np.asarray(parameters["J"], dtype=float).ravel()
    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()

    H_eff_spin = np.asarray(heff(w0=w0, w1=w1, J=j, zeta=zeta), dtype=complex)
    return heff_spin_to_lab_hamiltonian(H_eff_spin, w0=w0, w1=w1)
