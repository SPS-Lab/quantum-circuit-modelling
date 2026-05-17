"""Three-mode Duffing Hamiltonian definitions and model-side parameter evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from models.three_mode import three_mode_hamiltonian
from study_config import DuffingModelConfig, SystemParams


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


def is_reference_calibrated_duffing_mode(calibration_mode: str) -> bool:
    mode = str(calibration_mode).strip().lower()
    return mode in {"fitted-static", "symbolic-fitted-static"}


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


def _evaluate_parameter_coefficients_from_designs(
    *,
    coefficient_map: Mapping[str, np.ndarray],
    design_map: Mapping[str, np.ndarray],
    parameter_order: tuple[str, ...],
) -> dict[str, np.ndarray]:
    parameters: dict[str, np.ndarray] = {}
    for name in parameter_order:
        design = np.asarray(design_map[name], dtype=float)
        coeff = np.asarray(coefficient_map[name], dtype=float).ravel()
        parameters[name] = np.asarray(design @ coeff, dtype=float)
    return parameters


def _reference_calibration_designs(
    flux_values: np.ndarray,
    *,
    sweep_target: str,
    n_harmonics_w: int,
    n_harmonics_alpha: int,
    n_harmonics_g: int,
) -> tuple[tuple[str, ...], dict[str, np.ndarray], dict[str, np.ndarray]]:
    cosine_design_w = _cosine_design_matrix(flux_values, n_harmonics=n_harmonics_w)
    cosine_design_alpha = _cosine_design_matrix(flux_values, n_harmonics=n_harmonics_alpha)
    cosine_design_g = _cosine_design_matrix(flux_values, n_harmonics=n_harmonics_g)
    constant_design = _constant_design_matrix(flux_values)
    cosine_names_w = _cosine_coefficient_names(n_harmonics=n_harmonics_w)
    cosine_names_alpha = _cosine_coefficient_names(n_harmonics=n_harmonics_alpha)
    cosine_names_g = _cosine_coefficient_names(n_harmonics=n_harmonics_g)
    constant_names = np.asarray(["c0"], dtype=str)
    target = str(sweep_target).strip().lower()

    if target == "q1":
        design_map = {
            "w0": constant_design,
            "w1": cosine_design_w,
            "alpha0": constant_design,
            "alpha1": cosine_design_alpha,
            "wc": constant_design,
            "g0c": constant_design,
            "g1c": cosine_design_g,
        }
        coefficient_names = {
            "w0": constant_names,
            "w1": cosine_names_w,
            "alpha0": constant_names,
            "alpha1": cosine_names_alpha,
            "wc": constant_names,
            "g0c": constant_names,
            "g1c": cosine_names_g,
        }
    elif target == "q0":
        design_map = {
            "w0": cosine_design_w,
            "w1": constant_design,
            "alpha0": cosine_design_alpha,
            "alpha1": constant_design,
            "wc": constant_design,
            "g0c": cosine_design_g,
            "g1c": constant_design,
        }
        coefficient_names = {
            "w0": cosine_names_w,
            "w1": constant_names,
            "alpha0": cosine_names_alpha,
            "alpha1": constant_names,
            "wc": constant_names,
            "g0c": cosine_names_g,
            "g1c": constant_names,
        }
    else:
        raise ValueError(f"Unsupported sweep_target {sweep_target!r}")

    parameter_order = ("w0", "w1", "alpha0", "alpha1", "wc", "g0c", "g1c")
    return parameter_order, design_map, coefficient_names


def evaluate_symbolic_duffing_parameter_fit(
    flux_values: np.ndarray,
    *,
    sweep_target: str,
    coefficient_names: Mapping[str, np.ndarray],
    coefficients: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Evaluate the symbolic Duffing parameter curves at flux points."""
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    inferred_harmonics_w = 0
    inferred_harmonics_alpha = 0
    inferred_harmonics_g = 0
    for name, labels in coefficient_names.items():
        if name not in coefficients:
            continue
        label_arr = [str(label) for label in np.asarray(labels, dtype=str).ravel()]
        for label in label_arr:
            if label.startswith("cos"):
                try:
                    harmonic = int(label.removeprefix("cos"))
                except ValueError:
                    continue
                if name in {"w0", "w1"}:
                    inferred_harmonics_w = max(inferred_harmonics_w, harmonic)
                elif name in {"alpha0", "alpha1"}:
                    inferred_harmonics_alpha = max(inferred_harmonics_alpha, harmonic)
                elif name in {"g0c", "g1c"}:
                    inferred_harmonics_g = max(inferred_harmonics_g, harmonic)

    parameter_order, design_map, _ = _reference_calibration_designs(
        flux_arr,
        sweep_target=sweep_target,
        n_harmonics_w=inferred_harmonics_w,
        n_harmonics_alpha=inferred_harmonics_alpha,
        n_harmonics_g=inferred_harmonics_g,
    )
    available_order = tuple(
        name for name in parameter_order if name in coefficient_names and name in coefficients
    )
    return _evaluate_parameter_coefficients_from_designs(
        coefficient_map={
            name: np.asarray(coefficients[name], dtype=float).ravel()
            for name in available_order
        },
        design_map=design_map,
        parameter_order=available_order,
    )


def _assemble_fixed_bus_duffing_mode_parameters(
    symbolic_parameters: Mapping[str, np.ndarray],
    *,
    system_params: SystemParams,
) -> dict[str, np.ndarray]:
    """Attach fixed-bus mode parameters needed for a full Duffing Hamiltonian build."""
    parameters = {
        key: np.asarray(values, dtype=float).ravel()
        for key, values in symbolic_parameters.items()
    }
    if not parameters:
        raise ValueError("symbolic_parameters must not be empty")
    reference_shape = next(iter(parameters.values())).shape
    if any(np.asarray(values, dtype=float).ravel().shape != reference_shape for values in parameters.values()):
        raise ValueError("All Duffing symbolic parameter arrays must share the same shape")
    if "wc" in parameters:
        parameters["wc"] = np.asarray(parameters["wc"], dtype=float).ravel()
    else:
        parameters["wc"] = np.full(reference_shape, float(system_params.c.E_osc), dtype=float)
    return parameters


def evaluate_symbolic_duffing_mode_parameters(
    flux_values: np.ndarray,
    *,
    system_params: SystemParams,
    sweep_target: str,
    coefficient_names: Mapping[str, np.ndarray],
    coefficients: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Evaluate symbolic Duffing coefficients into full fixed-bus mode parameters."""
    symbolic_parameters = evaluate_symbolic_duffing_parameter_fit(
        flux_values,
        sweep_target=sweep_target,
        coefficient_names=coefficient_names,
        coefficients=coefficients,
    )
    return _assemble_fixed_bus_duffing_mode_parameters(
        symbolic_parameters,
        system_params=system_params,
    )


def build_duffing_model_stack_from_coefficients(
    flux_values: np.ndarray,
    *,
    system_params: SystemParams,
    duffing_config: DuffingModelConfig,
    sweep_target: str,
    coefficient_names: Mapping[str, np.ndarray],
    coefficients: Mapping[str, np.ndarray],
) -> DuffingModelBuildResult:
    """Build a Duffing Hamiltonian stack directly from symbolic coefficient tables."""
    mode_parameters = evaluate_symbolic_duffing_mode_parameters(
        flux_values,
        system_params=system_params,
        sweep_target=sweep_target,
        coefficient_names=coefficient_names,
        coefficients=coefficients,
    )
    return build_duffing_model_stack_from_parameters(
        mode_parameters,
        system_params=system_params,
        duffing_config=duffing_config,
    )


def build_duffing_model_stack_from_scratch(
    flux_values: np.ndarray,
    *,
    system_params: SystemParams,
    duffing_config: DuffingModelConfig,
    sweep_target: str = "q1",
) -> DuffingModelBuildResult:
    """Build a Duffing Hamiltonian stack without any higher-model reference input."""
    if is_reference_calibrated_duffing_mode(duffing_config.calibration_mode):
        raise ValueError(
            "Duffing calibration_mode requires a reference-driven static calibration "
            "step; use fit_duffing_mode_parameters_to_reference or "
            "fit_symbolic_duffing_mode_parameters_to_reference and then "
            "build_duffing_model_stack_from_parameters instead."
        )

    from models.duffing_calibration import _build_mode_parameter_arrays

    mode_parameters = _build_mode_parameter_arrays(
        flux_values,
        system_params=system_params,
        duffing_config=duffing_config,
        sweep_target=sweep_target,
    )
    return build_duffing_model_stack_from_parameters(
        mode_parameters,
        system_params=system_params,
        duffing_config=duffing_config,
    )
