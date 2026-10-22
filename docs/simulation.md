# Building and simulating a circuit

[`scripts/cz_example.py`](../scripts/cz_example.py) demonstrates a complete
circuit-model simulation using the public model and toolkit functions. After
[setting up the environment](../README.md#quick-start), run from the repo root:

```bash
python scripts/cz_example.py
```

The example prints the pulse settings, Hamiltonian dimensions, build and
propagation times, final conditional phase in radians, and leakage from initial
`|11>`. It runs a single fixed pulse and prints results to the terminal without
creating figures or benchmark artifacts.

## Simulation steps

1. Load device parameters and numerical basis settings with `load_study_config`.
2. Construct a time grid and cosine ramp-hold-ramp flux pulse with
   `toolkit.pulses.ramp_hold_ramp_flux_pulse(compress=True)`. This returns
   the full grid, a shorter list of consecutive flux segments, and integer
   `counts` counting how many time steps each segment is used.
3. Build one circuit Hamiltonian per flux segment with
   `models.circuit.build_circuit_model_stack`.
4. Prepare the four computational basis states as columns in the full Hilbert
   space, using `models.three_mode.computational_state_indices`.
5. Pass `counts` to `toolkit.solver.propagate_piecewise_constant` and retain the
   computational basis amplitudes at every time point.
6. Extract populations, leakage, and conditional phase with
   `toolkit.observables.extract_cz_observables`.

Hamiltonians are in GHz and times are in ns; the propagator includes the
required `2*pi` conversion. Full basis labels use `|q1,c,q0>`, and computational
states are ordered `|00>, |01>, |10>, |11>`. The propagator accepts arbitrary
initial statevector columns; see [architecture.md](architecture.md) for a small
two-state example and the amplitude array conventions.

During the hold, the same Hamiltonian and step length reuse one matrix
exponential. Every original time step still advances the four statevectors and
records their amplitudes. A shorter final step uses a separate exponential
when needed. For the default pulse, 484 steps use 9 Hamiltonians and 9 `expm`
calls, instead of building 485 Hamiltonians and computing 484 exponentials.

## Parameters

The script reads device parameters from `params/system_params.json` and basis
settings and the flux sweep target from `params/benchmark_params.json`. It sets
both qubit Hilbert dimensions to `QUBIT_DIM = 8`, retaining the configured
coupler dimension and charge-basis cutoffs. It uses the selected qubit's parking
flux as the pulse's initial and final flux.

Pulse defaults come from runtime experiment `20260518_050816_runtime_4c84263`:
target flux `0.23333333333333334`, each ramp `2 ns`, hold
`237.9595316591736 ns`, and time step `0.5 ns`. These settings are fixed
in the example; it does not search for a CZ pulse or recalibrate when device or
basis settings change. Command-line options override the pulse settings:

```bash
python scripts/cz_example.py --help
python scripts/cz_example.py --hold-time-ns 0 --dt-ns 2
```

The second command is a short smoke run at the same Hilbert dimensions. Its
shorter pulse and coarser time grid are for checking the workflow, not for
assessing CZ gate performance. Edit `QUBIT_DIM` to explore other truncations.
For other input states, pass your own statevector columns to the propagator
and analyze its returned amplitudes; `extract_cz_observables` expects the four
computational basis inputs used in this example.

For calibrated model comparisons, parameter sweeps, and saved plots, use the
[benchmark workflow](benchmark.md).
