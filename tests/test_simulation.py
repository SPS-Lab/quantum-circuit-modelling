"""Integration checks for the public circuit simulation workflow."""

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.circuit import build_circuit_model_stack
from study_config import load_study_config
from toolkit.simulation import simulate_circuit
from toolkit.solver import propagate_piecewise_constant


@pytest.fixture
def circuit_settings():
    config = load_study_config(
        system_params_path=_ROOT / "params/system_params.json",
        study_params_path=_ROOT / "params/benchmark_params.json",
    )
    circuit = config.static_benchmark.circuit_model
    circuit = replace(
        circuit,
        hilbert_truncation=replace(
            circuit.hilbert_truncation,
            q0_truncated_dim=2,
            q1_truncated_dim=3,
            c_truncated_dim=2,
        ),
    )
    return dict(system_params=config.system, circuit_config=circuit, sweep_target="q1")


@pytest.mark.parametrize("mode", ["serial", "dask-threads", "dask-processes", "client"])
def test_simulation_matches_explicit_workflow(circuit_settings, mode):
    times = np.array([0.0, 0.5, 1.0, 1.5, 1.75])
    flux = np.array([0.0, 0.2, 0.0])
    counts = np.array([1, 2, 1])
    hamiltonians = build_circuit_model_stack(
        **circuit_settings, flux_values=flux
    ).hamiltonian_stack
    indices = np.array([0, 1, 4, 5])  # |q1,c,q0>, including asymmetric qubit sizes.
    states = np.zeros((12, 4), dtype=complex)
    states[indices, np.arange(4)] = 1
    expected = propagate_piecewise_constant(
        hamiltonians, times, states, counts=counts, output_indices=indices
    )
    with ExitStack() as stack:
        client = None
        if mode == "client":
            distributed = pytest.importorskip("distributed")
            client = stack.enter_context(
                distributed.Client(
                    processes=False,
                    n_workers=1,
                    threads_per_worker=2,
                    dashboard_address=None,
                )
            )
        elif mode != "serial":
            pytest.importorskip("distributed")
        result = simulate_circuit(
            **circuit_settings,
            times_ns=times,
            flux_values=flux,
            counts=counts,
            preparation="serial" if mode == "client" else mode,
            threads=2,
            workers=2,
            dashboard_address="127.0.0.1:0",
            client=client,
        )
        if client is not None:
            assert client.status == "running"
            assert client.submit(sum, [1, 2]).result() == 3
    np.testing.assert_allclose(result.amplitudes, expected, atol=1e-13)
    np.testing.assert_array_equal(result.computational_indices, indices)
    assert result.hamiltonian_shape == (3, 12, 12)
    assert result.build_runtime_s >= 0
    assert result.preparation_runtime_s >= 0
    assert result.state_propagation_runtime_s >= 0
    assert result.propagation_runtime_s == (
        result.preparation_runtime_s + result.state_propagation_runtime_s
    )


def test_custom_states_and_output_rows(circuit_settings):
    times = np.array([0.0, 0.25, 0.5])
    flux = np.array([0.0, 0.1, 0.0])
    hamiltonians = build_circuit_model_stack(
        **circuit_settings, flux_values=flux
    ).hamiltonian_stack
    states = np.zeros((12, 1), dtype=complex)
    states[[1, 6], 0] = [1 / np.sqrt(2), 1j / np.sqrt(2)]
    rows = np.array([6, 1, 3])
    result = simulate_circuit(
        **circuit_settings,
        times_ns=times,
        flux_values=flux,
        initial_states=states,
        output_indices=rows,
    )
    expected = propagate_piecewise_constant(
        hamiltonians, times, states, output_indices=rows
    )
    np.testing.assert_allclose(result.amplitudes, expected, atol=1e-13)
