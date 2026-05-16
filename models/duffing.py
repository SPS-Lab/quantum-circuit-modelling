"""Compatibility exports for Duffing model construction and calibration."""

from models.duffing_calibration import (
    DuffingSymbolicParameterFitResult,
    _build_mode_parameter_arrays,
    fit_duffing_mode_parameters_to_reference,
    fit_symbolic_duffing_mode_parameters_to_reference,
)
from models.duffing_model import (
    DuffingCalibrationResult,
    DuffingModelBuildResult,
    _assemble_fixed_bus_duffing_mode_parameters,
    build_duffing_model_stack_from_coefficients,
    build_duffing_model_stack_from_parameters,
    build_duffing_model_stack_from_scratch,
    evaluate_symbolic_duffing_mode_parameters,
    evaluate_symbolic_duffing_parameter_fit,
    is_reference_calibrated_duffing_mode,
)

__all__ = [
    "DuffingCalibrationResult",
    "DuffingModelBuildResult",
    "DuffingSymbolicParameterFitResult",
    "build_duffing_model_stack_from_coefficients",
    "build_duffing_model_stack_from_parameters",
    "build_duffing_model_stack_from_scratch",
    "evaluate_symbolic_duffing_mode_parameters",
    "evaluate_symbolic_duffing_parameter_fit",
    "fit_duffing_mode_parameters_to_reference",
    "fit_symbolic_duffing_mode_parameters_to_reference",
    "is_reference_calibrated_duffing_mode",
]
