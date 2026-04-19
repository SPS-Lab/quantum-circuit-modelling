"""Circuit (scqubits) model construction for study benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model0.cpb import flux_dependent_EJ
from study.config import CircuitModelConfig, CouplerFrequencyConfig, SystemParams


@dataclass(frozen=True)
class CircuitModelBuildResult:
    hamiltonian_stack: np.ndarray



def _require_scqubits_module():
    try:
        import scqubits as scq
    except Exception as exc:  # pragma: no cover - import guard only
        raise ImportError("scqubits import failed while building circuit model") from exc
    return scq



def _build_circuit_hamiltonian(
    system_params: SystemParams,
    circuit_config: CircuitModelConfig,
    coupler_E_osc: float,
):
    scq = _require_scqubits_module()

    EJ1 = float(flux_dependent_EJ(system_params.q1.EJmax, system_params.q1.flux, system_params.q1.d))
    EJ2 = float(flux_dependent_EJ(system_params.q2.EJmax, system_params.q2.flux, system_params.q2.d))

    q1 = scq.Transmon(
        EJ=EJ1,
        EC=float(system_params.q1.EC),
        ng=float(system_params.q1.ng),
        ncut=int(system_params.q1.ncut),
        truncated_dim=int(circuit_config.hilbert_truncation.q1_truncated_dim),
        id_str=str(system_params.q1.id_str),
    )
    q2 = scq.Transmon(
        EJ=EJ2,
        EC=float(system_params.q2.EC),
        ng=float(system_params.q2.ng),
        ncut=int(system_params.q2.ncut),
        truncated_dim=int(circuit_config.hilbert_truncation.q2_truncated_dim),
        id_str=str(system_params.q2.id_str),
    )
    c = scq.Oscillator(
        E_osc=float(coupler_E_osc),
        truncated_dim=int(circuit_config.hilbert_truncation.c_truncated_dim),
        id_str=str(system_params.c.id_str),
    )

    hilbertspace = scq.HilbertSpace([q1, c, q2])
    x_c = c.creation_operator() + c.annihilation_operator()
    hilbertspace.add_interaction(
        check_validity=bool(circuit_config.interaction_validity_check),
        g=float(system_params.interactions.g_1c),
        op1=(q1.n_operator(), q1),
        op2=(x_c, c),
    )
    hilbertspace.add_interaction(
        check_validity=bool(circuit_config.interaction_validity_check),
        g=float(system_params.interactions.g_2c),
        op1=(q2.n_operator(), q2),
        op2=(x_c, c),
    )

    return np.asarray(hilbertspace.hamiltonian().full(), dtype=complex)



def build_circuit_model_stack(
    flux_values: np.ndarray,
    system_params: SystemParams,
    coupler_frequency: CouplerFrequencyConfig,
    circuit_config: CircuitModelConfig,
) -> CircuitModelBuildResult:
    """Build circuit-model Hamiltonians across a coupler-frequency flux sweep."""
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    wc = np.asarray(
        float(coupler_frequency.wc0) + float(coupler_frequency.amplitude) * np.cos(2.0 * np.pi * flux_arr),
        dtype=float,
    ).ravel()

    mats = [
        _build_circuit_hamiltonian(
            system_params=system_params,
            circuit_config=circuit_config,
            coupler_E_osc=float(wc_k),
        )
        for wc_k in wc
    ]
    return CircuitModelBuildResult(hamiltonian_stack=np.stack(mats, axis=0))
