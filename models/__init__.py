"""Model constructors and core utilities for the paper study."""

from models.circuit import CircuitModelBuildResult, build_circuit_model_stack
from models.dressed import (
    build_dressed_effective_computational_stack,
    build_dressed_effective_stack,
    extract_model1_parameters_from_4x4_stack,
)
from models.duffing import (
    DuffingModelBuildResult,
    build_duffing_model_stack,
    build_duffing_model_stack_from_parameters,
    fit_duffing_mode_parameters_to_reference,
)
from models.effective import (
    EffectiveModelDerivationResult,
    build_effective_hamiltonian_stack,
    derive_effective_model_from_dressed_stack,
    fit_single_harmonic_parameters,
    heff,
)
from models.josephson import flux_dependent_EJ
from models.sweep import resolve_static_sweep_values
from models.three_mode import (
    ThreeModeHamiltonianCommonKwargs,
    ThreeModeHamiltonianKwargs,
    computational_state_indices,
    computational_subspace_block,
    coupler_frequency,
    three_mode_hamiltonian,
    three_mode_hamiltonian_from_kwargs,
    three_mode_hamiltonian_stack_vs_flux,
)

__all__ = [
    "CircuitModelBuildResult",
    "DuffingModelBuildResult",
    "EffectiveModelDerivationResult",
    "build_circuit_model_stack",
    "build_duffing_model_stack",
    "build_duffing_model_stack_from_parameters",
    "build_effective_hamiltonian_stack",
    "derive_effective_model_from_dressed_stack",
    "fit_duffing_mode_parameters_to_reference",
    "fit_single_harmonic_parameters",
    "heff",
    "build_dressed_effective_stack",
    "build_dressed_effective_computational_stack",
    "extract_model1_parameters_from_4x4_stack",
    "flux_dependent_EJ",
    "resolve_static_sweep_values",
    "ThreeModeHamiltonianCommonKwargs",
    "ThreeModeHamiltonianKwargs",
    "coupler_frequency",
    "three_mode_hamiltonian",
    "three_mode_hamiltonian_from_kwargs",
    "three_mode_hamiltonian_stack_vs_flux",
    "computational_state_indices",
    "computational_subspace_block",
]
