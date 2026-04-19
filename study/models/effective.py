"""Effective two-level model utilities for study benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from model1.heff import heff
from model2.effective import extract_model1_parameters_from_4x4_stack
from toolkit.helpers import I2, pz


@dataclass(frozen=True)
class HarmonicFitResult:
    coefficients: dict[str, np.ndarray]
    fitted_parameters: dict[str, np.ndarray]


@dataclass(frozen=True)
class EffectiveModelDerivationResult:
    extracted_parameters: dict[str, np.ndarray]
    harmonic_fit: HarmonicFitResult



def heff_spin_to_lab_hamiltonian(H_eff: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Convert model-1 ``(w/2) sigma_z`` convention to lab-frame ``w n``."""
    H_eff_arr = np.asarray(H_eff, dtype=complex)
    w1_arr = np.asarray(w1, dtype=float).reshape(-1)
    w2_arr = np.asarray(w2, dtype=float).reshape(-1)

    w1_b = w1_arr[:, np.newaxis, np.newaxis]
    w2_b = w2_arr[:, np.newaxis, np.newaxis]
    pz1 = np.kron(pz, I2)
    pz2 = np.kron(I2, pz)
    eye4 = np.eye(4, dtype=complex)
    return H_eff_arr + 0.5 * (w1_b + w2_b) * eye4 - w1_b * pz1 - w2_b * pz2



def _single_harmonic_design_matrix(flux_values: np.ndarray) -> np.ndarray:
    theta = 2.0 * np.pi * np.asarray(flux_values, dtype=float).ravel()
    return np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])



def fit_single_harmonic_parameters(
    flux_values: np.ndarray,
    extracted_parameters: Mapping[str, np.ndarray],
) -> HarmonicFitResult:
    """Fit ``x0 + a cos(2πφ) + b sin(2πφ)`` for each effective parameter."""
    X = _single_harmonic_design_matrix(flux_values)
    coeff: dict[str, np.ndarray] = {}
    fitted: dict[str, np.ndarray] = {}
    for name in ("w1", "w2", "J", "zeta"):
        y = np.asarray(extracted_parameters[name], dtype=float).ravel()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        coeff[name] = beta
        fitted[name] = X @ beta
    return HarmonicFitResult(coefficients=coeff, fitted_parameters=fitted)



def derive_effective_model_from_dressed_stack(
    flux_values: np.ndarray,
    dressed_stack: np.ndarray,
    fit_basis: str,
) -> EffectiveModelDerivationResult:
    """Extract effective parameters from a dressed stack and fit compact flux laws."""
    extracted = extract_model1_parameters_from_4x4_stack(dressed_stack)
    if fit_basis != "single-harmonic":
        raise ValueError(f"Unsupported fit_basis {fit_basis!r}")
    harmonic_fit = fit_single_harmonic_parameters(flux_values, extracted)
    return EffectiveModelDerivationResult(
        extracted_parameters=extracted,
        harmonic_fit=harmonic_fit,
    )



def build_effective_hamiltonian_stack(parameters: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build lab-frame effective Hamiltonian stack from ``w1,w2,J,zeta`` arrays."""
    w1 = np.asarray(parameters["w1"], dtype=float).ravel()
    w2 = np.asarray(parameters["w2"], dtype=float).ravel()
    j = np.asarray(parameters["J"], dtype=float).ravel()
    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()

    H_eff_spin = np.asarray(heff(w1, w2, j, zeta), dtype=complex)
    return heff_spin_to_lab_hamiltonian(H_eff_spin, w1, w2)
