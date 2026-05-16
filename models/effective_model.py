"""Effective two-level Hamiltonian definitions and model-side coefficient evaluation."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from study_config import SystemParams
from toolkit.helpers import I2, px, py, pz


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
    zeta: np.ndarray | float,
) -> np.ndarray:
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
    w1: np.ndarray,
) -> np.ndarray:
    """Convert effective ``(w/2) sigma_z`` convention to lab-frame ``w n``."""
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


def evaluate_effective_parameter_fit(
    flux_values: np.ndarray,
    *,
    system_params: SystemParams,
    fit_basis: str,
    coefficient_names: Mapping[str, tuple[str, ...] | np.ndarray],
    coefficients: Mapping[str, np.ndarray],
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

        wc = np.full_like(out["w0"], float(system_params.c.E_osc), dtype=float)

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


def build_effective_hamiltonian_stack(parameters: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build lab-frame effective Hamiltonian stack from ``w0,w1,J,zeta`` arrays."""
    w0 = np.asarray(parameters["w0"], dtype=float).ravel()
    w1 = np.asarray(parameters["w1"], dtype=float).ravel()
    j = np.asarray(parameters["J"], dtype=float).ravel()
    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()

    H_eff_spin = np.asarray(heff(w0=w0, w1=w1, J=j, zeta=zeta), dtype=complex)
    return heff_spin_to_lab_hamiltonian(H_eff_spin, w0=w0, w1=w1)
