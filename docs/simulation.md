# Building and simulating a circuit

[`scripts/cz_example.py`](../scripts/cz_example.py) demonstrates a complete
circuit-model simulation using the public model and toolkit functions. After
[setting up the environment](../README.md#quick-start), run from the repo root:

```bash
python scripts/cz_example.py
```

The example prints the pulse settings, Hamiltonian dimensions, build and
unitary preparation and state propagation times, their combined time, final
conditional phase in radians, and leakage from initial
`|11>`. It runs a single fixed pulse and prints results to the terminal without
creating figures or benchmark artifacts.

## Simulation steps

1. Load device parameters and numerical basis settings with `load_study_config`.
2. Construct a time grid and cosine ramp-hold-ramp flux pulse with
   `toolkit.pulses.ramp_hold_ramp_flux_pulse(compress=True)`. This returns
   the full grid, a shorter list of consecutive flux segments, and integer
   `counts` counting how many time steps each segment is used.
3. Call `toolkit.simulation.simulate_circuit` with the device settings and pulse.
   It builds the Hamiltonians, prepares segment unitaries, and propagates the
   four computational basis states. Select serial, threaded, or process
   execution with `preparation`; no Dask client setup is needed in the script.
4. Analyze `result.amplitudes` with
   `toolkit.observables.extract_cz_observables`.

```python
from toolkit.observables import extract_cz_observables
from toolkit.simulation import simulate_circuit

result = simulate_circuit(
    system_params=config.system,
    circuit_config=config.static_benchmark.circuit_model,
    sweep_target="q1",
    flux_values=pulse_flux,
    times_ns=times_ns,
    counts=counts,
)
observables = extract_cz_observables(result.amplitudes)
```

The same function is used by the CZ benchmark and its runtime benchmark.
It returns build, unitary preparation, and state propagation times.
`result.propagation_runtime_s` combines preparation and state propagation to
match the existing benchmark timing convention. Dense Hamiltonians and
unitaries are released after the call. Saved benchmark propagation timings
also include observable extraction, as they do for the other models.
For repeated runs with a persistent Dask cluster, pass `client=client` instead
of a Dask preparation mode; the caller retains ownership of that client.

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
both qubit Hilbert dimensions using `QUBIT_DIM`, retaining the configured
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
For other input states, pass `initial_states` and `output_indices` to
`simulate_circuit` and analyze `result.amplitudes`; `extract_cz_observables` expects the four
computational basis inputs used in this example.

For calibrated model comparisons, parameter sweeps, and saved plots, use the
[benchmark workflow](benchmark.md).

## Parallel unitary preparation

The example defaults to serial preparation. To prepare segment unitaries in
parallel, use a local Dask distributed client with one worker and multiple
threads in a single process:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/cz_example.py --preparation dask-threads --threads 4
```

To compare against separate worker processes, each with one execution thread:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/cz_example.py --preparation dask-processes --workers 4
```

Processes allow Python code in different tasks to execute independently of the
GIL. They also transfer Hamiltonians to workers and unitaries back to the
client, adding communication and memory costs. The preparation timing includes
these transfers. Both Dask modes provide the dashboard and submit one task per
segment.

The active dashboard URL is printed; by default it binds to
`http://127.0.0.1:8787`. It shows segment preparation tasks while the example
runs and closes when the client exits. Use `--dashboard-address 127.0.0.1:0`
to request any available port. Dask, distributed, and Bokeh are included in
`scmodels-env.yml`.

One task prepares each segment's dictionary of step length to unitary. Dask
assigns these tasks to available workers, and results are gathered in pulse
order. The four statevectors are then propagated sequentially through those
unitaries. `--preparation serial` uses the same preparation function without
Dask. All modes prepare the complete schedule before propagating and retain
the prepared matrices in memory.

For timing comparisons, keep BLAS thread settings the same between runs.
Start with one BLAS thread per task to avoid nested thread pools competing for
cores, and adjust `--threads` (threaded mode) or `--workers` (process mode)
to control simultaneous preparations and their
temporary memory use. The printed preparation time excludes Dask client
startup; state propagation time excludes exponential preparation. Compare
`Preparation + propagation` against the previous combined solver timing.
These are execution timings; initial imports and Hamiltonian building are
outside that total.

To measure strong scaling over several thread and process counts with saved
samples, plots, and provenance, use
[`run_dask_scaling_benchmark.py`](../scripts/run_dask_scaling_benchmark.py).
See [Dask strong scaling](benchmark.md#dask-strong-scaling) for timing boundaries
and configuration.
