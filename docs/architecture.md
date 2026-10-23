# Architecture

The paper-aligned workflow is organized by responsibility:

- `models`: model builders (`effective`, `duffing`, `circuit`)
- `comparison`: benchmark logic (`static`, `cz`, `leakage_flow`, `truncation`)
- `plotting`: plotting only
- `study_config.py`: typed config loading/validation
- `params`: all runtime parameters consumed by main scripts
- `toolkit/solver.py`: public statevector propagation and eigensystem utilities
- `toolkit/pulses.py`: reusable pulse schedules
- `toolkit/observables.py`: observable extraction from amplitude histories

State labels in this repo follow `|q1,c,q0>` (so `q0` is the right/LSB qubit when bit significance matters).

For a complete circuit simulation, start with
[`scripts/cz_example.py`](../scripts/cz_example.py) and the
[simulation guide](simulation.md). The example builds a pulse and Hamiltonians,
propagates initial states, and extracts CZ observables using `models` and
`toolkit`, independently of `comparison`.

The public propagator also accepts Hamiltonians constructed directly:

```python
import numpy as np
from toolkit.solver import propagate_piecewise_constant

times_ns = np.array([0.0, 0.5, 1.0])
hamiltonian_ghz = np.array([[0.0, 0.25], [0.25, 0.0]])
hamiltonians = np.stack([hamiltonian_ghz] * len(times_ns))
initial_states = np.array([[1.0], [0.0]], dtype=complex)
amplitudes = propagate_piecewise_constant(hamiltonians, times_ns, initial_states)
populations = np.abs(amplitudes[:, :, 0]) ** 2
```

Initial states are columns. The returned axes are `(time, output basis row,
initial state)`. Optional `output_indices` limits the recorded rows while
propagation always uses the full Hilbert space. Hamiltonian `k` acts over
`[times_ns[k], times_ns[k + 1])`; the final Hamiltonian is unused. The solver
converts GHz and ns with a factor of `2*pi`.

The CPU propagator also accepts a compressed schedule:

```python
from toolkit.pulses import ramp_hold_ramp_flux_pulse

times_ns, flux_values, counts = ramp_hold_ramp_flux_pulse(
    ramp_time_ns=2.0, hold_time_ns=237.9595316591736, dt_ns=0.5,
    idle_flux=0.0, target_flux=0.23333333333333334, compress=True,
)
# Build one Hamiltonian per flux_values entry, then:
# amplitudes = propagate_piecewise_constant(
#     hamiltonians, times_ns, initial_states, counts=counts,
# )
```

`counts` contains integer step counts, not times in ns. Counts sum to
`len(times_ns) - 1`; the unused final flux is omitted from the compressed
schedule. Consecutive equal flux values share a Hamiltonian, and each segment
reuses its unitary for equal step lengths. A different step length, including
a shorter final interval, gets its own exponential. Step lengths are compared
exactly, so floating-point differences in the grid are preserved. Amplitudes
are still recorded at every original time point. Omitting `compress` or
`counts` retains the original APIs.

`prepare_segment(hamiltonian_ghz, times_segment)` returns a dictionary from
exact step length to unitary. The segment time grid includes both endpoints.
`prepare_segments(hamiltonians, times_ns, counts=counts)` prepares the whole
schedule serially. Passing a Dask distributed `client` submits one task per
segment, gathers results in chronological order, and releases the futures.
Task results are not cached across preparation calls.

Both serial and Dask modes prepare all unitaries before state propagation.
Pass the returned list to `propagate_piecewise_constant` as
`prepared_segments=prepared` to avoid recomputation; without that argument,
the propagator prepares them serially itself. All prepared unitaries remain
in memory while propagating. Propagation remains sequential and uses the same
numerical operations in both modes. The [simulation guide](simulation.md#parallel-unitary-preparation)
shows how to enable threaded or process execution and its dashboard.

The CZ example, CZ benchmark, and its runtime benchmark use this compression.
Saved CZ results still contain the full sampled pulse for plotting. The GPU
propagators retain their original uncompressed interface.

Reusable pulse schedules live in `toolkit/pulses.py`. CZ analysis lives in
`toolkit.observables.extract_cz_observables`, which consumes
amplitude histories in computational order `|00>, |01>, |10>, |11>`. The CZ
benchmark and standalone [`CZ example`](../scripts/cz_example.py) share
`toolkit.simulation.simulate_circuit`. It owns Hamiltonian building, default
computational input construction, unitary preparation, propagation, and optional
Dask client lifetime. It returns amplitudes and separate build, preparation,
and state propagation timings; combined preparation and propagation preserves
the benchmark timing convention. Custom state columns, recorded basis rows,
and caller-owned Dask clients are supported. Effective and Duffing models
continue to use the lower-level propagator. Usage, pulse settings, and optional profiling are covered in
the [simulation guide](simulation.md).
