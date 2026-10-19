"""Build and simulate a circuit-model CZ pulse using the public toolkit.

Run from the repository root:
    python scripts/cz_example.py

The example constructs a flux pulse, builds circuit Hamiltonians, propagates
the four computational basis states, and reports final conditional phase and
leakage from |11>, along with build and propagation times. Each qubit uses eight
levels; device parameters and remaining basis settings come from params/.

Pulse settings are fixed and can be overridden on the command line. Hamiltonians
are in GHz and times are in ns; the solver applies the factor of 2*pi. Results
are printed to the terminal. See docs/simulation.md for usage and optional
profiling commands.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.circuit import build_circuit_model_stack
from models.three_mode import computational_state_indices
from study_config import load_study_config
from toolkit.observables import extract_cz_observables
from toolkit.pulses import ramp_hold_ramp_flux_pulse
from toolkit.solver import propagate_piecewise_constant

QUBIT_DIM = 8


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Pulse from results/experiments/20261016_163913_cz_c6753c6.
    parser.add_argument(
        "--target-flux",
        type=float,
        default=0.23333333333333334,
        help="Pulse plateau flux in units of the flux quantum",
    )
    parser.add_argument(
        "--ramp-time-ns",
        type=float,
        default=2.0,
        help="Duration of each cosine ramp in ns",
    )
    parser.add_argument(
        "--hold-time-ns",
        type=float,
        default=237.92738645287855,
        help="Duration of the pulse plateau in ns",
    )
    parser.add_argument(
        "--dt-ns",
        type=float,
        default=0.5,
        help="Propagation time step in ns (the final step may be shorter)",
    )
    args = parser.parse_args()
    if args.ramp_time_ns <= 0 or args.dt_ns <= 0 or args.hold_time_ns < 0:
        parser.error("ramp time and dt must be positive; hold time must be nonnegative")
    if not all(np.isfinite(value) for value in vars(args).values()):
        parser.error("pulse parameters must be finite")

    config = load_study_config(
        system_params_path=_REPO_ROOT / "params/system_params.json",
        study_params_path=_REPO_ROOT / "params/benchmark_params.json",
    )
    circuit = config.static_benchmark.circuit_model
    circuit = replace(
        circuit,
        hilbert_truncation=replace(
            circuit.hilbert_truncation,
            q0_truncated_dim=QUBIT_DIM,
            q1_truncated_dim=QUBIT_DIM,
        ),
    )
    sweep_target = config.static_benchmark.flux_control.sweep_target
    idle_flux = getattr(config.system, sweep_target).flux
    times_ns, pulse_flux = ramp_hold_ramp_flux_pulse(
        ramp_time_ns=args.ramp_time_ns,
        hold_time_ns=args.hold_time_ns,
        dt_ns=args.dt_ns,
        idle_flux=idle_flux,
        target_flux=args.target_flux,
    )
    dims = dict(
        nlevels_qubit=QUBIT_DIM,
        nlevels_coupler=circuit.hilbert_truncation.c_truncated_dim,
    )
    print(
        f"Circuit CZ: N_E,q={QUBIT_DIM}, N_E,c={dims['nlevels_coupler']}, "
        f"{times_ns.size - 1} propagation steps; "
        f"{sweep_target} flux {idle_flux:g} -> {args.target_flux:g}, "
        f"ramp={args.ramp_time_ns:g} ns, hold={args.hold_time_ns:g} ns, "
        f"dt={args.dt_ns:g} ns",
        flush=True,
    )

    started = time.perf_counter()
    hamiltonians = build_circuit_model_stack(
        flux_values=pulse_flux,
        system_params=config.system,
        circuit_config=circuit,
        sweep_target=sweep_target,
    ).hamiltonian_stack
    print(
        f"Build: {time.perf_counter() - started:.3f} s; "
        f"Hamiltonian stack shape={hamiltonians.shape}",
        flush=True,
    )

    computational_indices = computational_state_indices(**dims)
    initial_states = np.zeros((hamiltonians.shape[1], 4), dtype=complex)
    initial_states[computational_indices, np.arange(4)] = 1.0

    started = time.perf_counter()
    amplitudes = propagate_piecewise_constant(
        hamiltonians,
        times_ns,
        initial_states,
        output_indices=computational_indices,
        progress_label="propagate circuit CZ",
    )
    print(f"Propagation: {time.perf_counter() - started:.3f} s", flush=True)
    observables = extract_cz_observables(amplitudes)
    print(
        f"Final conditional phase={observables.conditional_phase[-1]:.6g} rad; "
        f"leakage from |11>={observables.leakage_11[-1]:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
