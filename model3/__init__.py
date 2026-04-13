"""scqubits Transmon+Oscillator reference and comparison utilities."""

__all__ = [
    "DEFAULT_TRANSMON_KEY",
    "load_transmon_params",
    "transmon_oscillator_hamiltonian",
    "transmon_oscillator_stack_vs_flux",
    "compare_model1_model2_against_scqubits",
]


def __getattr__(name: str):
    if name in {"DEFAULT_TRANSMON_KEY", "load_transmon_params"}:
        from model3.reference_params import DEFAULT_TRANSMON_KEY, load_transmon_params

        return {
            "DEFAULT_TRANSMON_KEY": DEFAULT_TRANSMON_KEY,
            "load_transmon_params": load_transmon_params,
        }[name]
    if name in {"transmon_oscillator_hamiltonian", "transmon_oscillator_stack_vs_flux"}:
        from model3.scqref import (
            transmon_oscillator_hamiltonian,
            transmon_oscillator_stack_vs_flux,
        )

        return {
            "transmon_oscillator_hamiltonian": transmon_oscillator_hamiltonian,
            "transmon_oscillator_stack_vs_flux": transmon_oscillator_stack_vs_flux,
        }[name]
    if name == "compare_model1_model2_against_scqubits":
        from model3.comparison import compare_model1_model2_against_scqubits

        return compare_model1_model2_against_scqubits
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
