# Architecture

The paper-aligned workflow is organized by responsibility:

- `models`: model builders (`effective`, `duffing`, `circuit`)
- `comparison`: benchmark logic (`static`, `cz`, `leakage_flow`, `truncation`)
- `plotting`: plotting only
- `study_config.py`: typed config loading/validation
- `params`: all runtime parameters consumed by main scripts
- `toolkit/solver.py`: public statevector propagation and eigensystem utilities

State labels in this repo follow `|q1,c,q0>` (so `q0` is the right/LSB qubit when bit significance matters).

To simulate a Hamiltonian independently of a benchmark:

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

Reusable pulse schedules live in `toolkit/pulses.py`. CZ analysis lives in
`toolkit.observables.extract_cz_observables`, which consumes
amplitude histories in computational order `|00>, |01>, |10>, |11>`. The CZ
benchmark uses a thin adapter around the public propagator; the standalone
`scripts/profile_cz.py` calls the propagator and analyzes its output. For an overall profile with fixed qubit dimensions of 8:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m cProfile -o /tmp/cz-n8.prof scripts/profile_cz.py
```

The script includes imports, configuration, circuit Hamiltonian construction,
and propagation. It uses a fixed pulse without static fitting or hold-time
search, and writes no benchmark artifacts. For a short smoke run, append
`--hold-time-ns 0 --dt-ns 2` (this changes the pulse and is not a calibrated CZ).
