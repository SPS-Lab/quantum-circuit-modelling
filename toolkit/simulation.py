"""Build and simulate a circuit without managing Hamiltonians or unitaries."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from models.circuit import build_circuit_model_stack
from runtime_utils import log_progress
from study_config import CircuitModelConfig, SystemParams
from toolkit.solver import prepare_segments, propagate_piecewise_constant


@dataclass(frozen=True)
class CircuitSimulationResult:
    """Recorded amplitudes and wall-clock timings, without retaining dense matrices."""

    amplitudes: np.ndarray
    computational_indices: np.ndarray
    hamiltonian_shape: tuple[int, ...]
    build_runtime_s: float
    preparation_runtime_s: float
    state_propagation_runtime_s: float

    @property
    def propagation_runtime_s(self) -> float:
        """Preparation plus state propagation, matching benchmark solver timings."""
        return self.preparation_runtime_s + self.state_propagation_runtime_s


def simulate_circuit(
    *,
    system_params: SystemParams,
    circuit_config: CircuitModelConfig,
    sweep_target: str,
    flux_values: np.ndarray,
    times_ns: np.ndarray,
    counts: np.ndarray | None = None,
    initial_states: np.ndarray | None = None,
    output_indices: np.ndarray | None = None,
    preparation: Literal["serial", "dask-threads", "dask-processes"] = "serial",
    threads: int = 4,
    workers: int = 4,
    dashboard_address: str = "127.0.0.1:8787",
    client=None,
    progress_label: str | None = None,
) -> CircuitSimulationResult:
    """Build circuit Hamiltonians, prepare unitaries, and propagate state columns.

    Supply a sampled pulse, or compressed flux segments with integer ``counts``
    and the full time grid. Hamiltonians are in GHz; times are in ns.
    By default, propagate the four computational basis inputs and record their
    computational amplitudes, ordered |00>, |01>, |10>, |11>. Custom state
    columns and output rows use the full |q1,c,q0> basis. To retain all rows,
    pass ``output_indices=np.arange(dimension)``.

    Dask clients created here are closed on return; their dashboard URL is
    printed when ``progress_label`` is set. Alternatively, supply a client you
    own (with the default ``preparation='serial'``); it remains open. Timings
    exclude client startup/teardown. Preparation includes matrix transfers.
    No observables, calibration, or artifact persistence are imposed.
    """
    if preparation not in ("serial", "dask-threads", "dask-processes"):
        raise ValueError(f"Unknown preparation mode {preparation!r}")
    if threads < 1 or workers < 1:
        raise ValueError("threads and workers must be >= 1")
    if client is not None and preparation != "serial":
        raise ValueError("Supply either a client or a Dask preparation mode")
    truncation = circuit_config.hilbert_truncation
    if (
        initial_states is None
        and min(truncation.q0_truncated_dim, truncation.q1_truncated_dim) < 2
    ):
        raise ValueError("Computational inputs require at least two levels per qubit")
    stride = truncation.c_truncated_dim * truncation.q0_truncated_dim
    computational_indices = np.array([0, 1, stride, stride + 1], dtype=int)
    if output_indices is None:
        output_indices = computational_indices

    def report(message):
        if progress_label:
            log_progress(f"{progress_label}: {message}")

    report("building circuit Hamiltonians")
    started = perf_counter()
    hamiltonians = build_circuit_model_stack(
        flux_values=flux_values,
        system_params=system_params,
        circuit_config=circuit_config,
        sweep_target=sweep_target,
    ).hamiltonian_stack
    build_time = perf_counter() - started
    report(f"Build: {build_time:.3f} s; Hamiltonian stack shape={hamiltonians.shape}")
    with ExitStack() as stack:
        if preparation != "serial":
            from dask.distributed import Client

            processes = preparation == "dask-processes"
            client = stack.enter_context(
                Client(
                    processes=processes,
                    n_workers=workers if processes else 1,
                    threads_per_worker=1 if processes else threads,
                    dashboard_address=dashboard_address,
                )
            )
            host = dashboard_address.rsplit(":", 1)[0] or "127.0.0.1"
            if host in ("0.0.0.0", "[::]"):
                host = "127.0.0.1"
            port = client.scheduler_info()["services"]["dashboard"]
            report(f"Dask dashboard: http://{host}:{port}/status")
        report("preparing segment unitaries")
        started = perf_counter()
        prepared = prepare_segments(
            hamiltonians, times_ns, counts=counts, client=client
        )
        preparation_time = perf_counter() - started
        report(f"Unitary preparation: {preparation_time:.3f} s")
        started = perf_counter()
        if initial_states is None:
            initial_states = np.zeros((hamiltonians.shape[1], 4), dtype=complex)
            initial_states[computational_indices, np.arange(4)] = 1.0
        amplitudes = propagate_piecewise_constant(
            hamiltonians,
            times_ns,
            initial_states,
            counts=counts,
            prepared_segments=prepared,
            output_indices=output_indices,
            progress_label=progress_label,
        )
        propagation_time = perf_counter() - started
        report(f"State propagation: {propagation_time:.3f} s")
        report(
            f"Preparation + propagation: {preparation_time + propagation_time:.3f} s"
        )
    return CircuitSimulationResult(
        amplitudes=amplitudes,
        computational_indices=computational_indices,
        hamiltonian_shape=hamiltonians.shape,
        build_runtime_s=build_time,
        preparation_runtime_s=preparation_time,
        state_propagation_runtime_s=propagation_time,
    )
