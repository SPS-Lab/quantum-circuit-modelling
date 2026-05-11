"""Circuit (scqubits) model construction for study benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.sweep import resolve_static_sweep_values
from study_config import CircuitModelConfig, CouplerFrequencyConfig, SystemParams


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
    *,
    system_params: SystemParams,
    circuit_config: CircuitModelConfig,
    q1_flux: float,
    q2_flux: float,
    coupler_E_osc: float,
):
    scq = _require_scqubits_module()

    q1 = scq.TunableTransmon(
        EJmax=float(system_params.q1.EJmax),
        EC=float(system_params.q1.EC),
        d=float(system_params.q1.d),
        flux=float(q1_flux),
        ng=float(system_params.q1.ng),
        ncut=int(system_params.q1.ncut),
        truncated_dim=int(circuit_config.hilbert_truncation.q1_truncated_dim),
        id_str=str(system_params.q1.id_str),
    )
    q2 = scq.TunableTransmon(
        EJmax=float(system_params.q2.EJmax),
        EC=float(system_params.q2.EC),
        d=float(system_params.q2.d),
        flux=float(q2_flux),
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

    hilbertspace = scq.HilbertSpace([q2, c, q1])
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
    *,
    flux_values: np.ndarray,
    system_params: SystemParams,
    coupler_frequency: CouplerFrequencyConfig,
    circuit_config: CircuitModelConfig,
    sweep_target: str,
) -> CircuitModelBuildResult:
    """Build circuit-model Hamiltonians for the configured static sweep target."""
    q1_flux_arr, q2_flux_arr, wc = resolve_static_sweep_values(
        flux_values,
        system_params=system_params,
        coupler_frequency_config=coupler_frequency,
        sweep_target=sweep_target,
    )

    mats = [
        _build_circuit_hamiltonian(
            system_params=system_params,
            circuit_config=circuit_config,
            q1_flux=float(q1_flux_arr[k]),
            q2_flux=float(q2_flux_arr[k]),
            coupler_E_osc=float(wc_k),
        )
        for k, wc_k in enumerate(wc)
    ]
    return CircuitModelBuildResult(hamiltonian_stack=np.stack(mats, axis=0))
