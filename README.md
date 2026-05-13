# scmodels

Code to answer the question "Which Superconducting-Qubit Model Is Good Enough?"

## Setup

```bash
micromamba create -f scmodels-env.yml
micromamba activate scmodels-env
```

And verify with:

```bash
python print_versions.py
```

## Refactored Workflow

The paper-aligned workflow is organized by responsibility:

- `models`: model builders (`effective`, `duffing`, `circuit`)
- `comparison`: benchmark logic (`static`, `cz`, `leakage_flow`, `truncation`)
- `plotting`: plotting only
- `study_config.py`: typed config loading/validation
- `params`: all runtime parameters consumed by main scripts

Legacy `model0/1/2/3` packages have been merged into the modules above.

State labels in this repo follow `|q1,c,q0>` (so `q0` is the right/LSB qubit when bit significance matters).

## Benchmark Run Artifacts

All benchmark scripts now default to creating a timestamped run directory under:

- `results/experiments/<timestamp>_<benchmark-or-experiment-name>_<commit-short>[_dirty]/`

Each run directory stores the benchmark `.h5` file, generated figure(s), a
`benchmark_run.json` metadata snapshot, a glanceable `git_head.txt`, a
`git_scope_snapshot.txt`, and copies of the input parameter JSON files used for
the run.

Useful CLI patterns:

```bash
python scripts/run_static_benchmark.py
python scripts/run_static_benchmark.py --experiment-name appendix_static_fit
python scripts/run_static_benchmark.py --plot-only
```

`--plot-only` now uses the newest matching run by default when `--results` is
not provided. You can still point any benchmark script at an explicit `.h5`
file with `--results path/to/<benchmark>_results.h5`.

Run the static benchmark:

```bash
python scripts/run_static_benchmark.py
```

This writes into the run directory:
- the static figure
- `static_results.h5`
- `static_fitted_parameters.json`
- `static_fitted_parameters_table.tex`

To replot without rerunning the benchmark:

```bash
python scripts/run_static_benchmark.py --plot-only
```

This reads:

- `params/system_params.json` (device/system parameters)
- `params/benchmark_params.json` (benchmark/model/plot settings)

The Duffing model supports calibration modes via
`static_benchmark.duffing_model.calibration_mode`:
- `fixed`: calibrate transmon Duffing parameters once at system parking biases
- `analytic-per-flux`: flux-dependent transmon approximation (no per-point numerical calibration)
- `per-flux`: recalibrate transmon Duffing parameters at every sweep point using transmon diagonalization
- `fitted-static`: fit latent Duffing parameters independently at each flux point to match circuit dressed observables
- `symbolic-fitted-static`: fit one global cosine-only symbolic surrogate for `w0`, `w1`, `alpha0`, `alpha1` over flux, refined against circuit dressed observables

The checked-in benchmark config currently uses `symbolic-fitted-static`.

The explicit static fitted-parameter artifact is meant for reusing the lower
models without rerunning the static fit. The generated LaTeX table is a plain
appendix-friendly tabular block for manual pasting into e.g.
`untracked/paper/main.tex`.

Run the CZ dynamics benchmark:

```bash
python scripts/run_cz_benchmark.py
```

The CZ benchmark:
- uses a shared flux pulse schedule for all models,
- uses a ramp-hold-ramp pulse configured under `cz_benchmark`,
- propagates all models with dense `numpy/scipy` matrix exponentials,
- builds the circuit Hamiltonians with `scqubits`,
- reports shared precompute time plus per-model build/propagation runtimes in the saved summary,
- focuses on CZ behavior/statevector from `|++>`,
- writes a figure with:
  - flux vs time,
  - conditional phase vs time for all models,
  - three computational heatmaps (effective, Duffing, circuit) where brightness is population and hue is relative phase.
- writes CZ results to an `.h5` file next to that figure.

Timing/scan settings are read from `params/benchmark_params.json` under:
- `cz_benchmark.ramp_time_ns`
- `cz_benchmark.dt_ns`
- `cz_benchmark.enable_hold_time_scan`
- `cz_benchmark.total_time_ns` or `cz_benchmark.hold_time_ns` (fixed-hold mode only; do not set either when scan is enabled)
- `cz_benchmark.scan_dt_ns`
- `cz_benchmark.scan_max_hold_ns`
- `cz_benchmark.scan_leakage_penalty`

Replot from saved CZ data only:

```bash
python scripts/run_cz_benchmark.py --plot-only
```

Reuse fitted static parameters from a previous static run:

```bash
python scripts/run_cz_benchmark.py --from-fitted results/experiments/<static-run>/
```

`--from-fitted` accepts a static run directory, `static_fitted_parameters.json`,
or `static_results.h5`.

Run the driven single-qubit RX benchmark:

```bash
python scripts/run_rx_benchmark.py
```

The RX benchmark:
- uses one shared rotating-frame/RWA microwave envelope for all models,
- drives `q0` using the Krantz et al. quadrature convention (`phase_rad=0` -> `X`, `phase_rad=pi/2` -> `Y`),
- compares the `|00> -> |01>` and `|10> -> |11>` transfer traces,
- reports leakage from `|00>`/`|10>` and spectator-state mismatch,
- writes separate population and diagnostics figures,
- and writes one `.h5` results file next to the populations figure.

Timing/drive settings are read from `params/benchmark_params.json` under:
- `rx_benchmark.drive_qubit`
- `rx_benchmark.drive_frequency`
- `rx_benchmark.drive_amplitude`
- `rx_benchmark.drive_phase_rad`
- `rx_benchmark.total_time_ns`
- `rx_benchmark.dt_ns`
- `rx_benchmark.rise_time_ns`

Replot from saved RX data only:

```bash
python scripts/run_rx_benchmark.py --plot-only
```

Run the combined leakage/flow benchmark (from `|1,0,1>`):

```bash
python scripts/run_leakage_flow_benchmark.py
```

This benchmark uses a short pulse and writes one figure with 4 heatmaps:
- top row: population+phase heatmaps for Duffing and circuit models
- bottom row: signed transition-current heatmaps for Duffing and circuit models

Transition channels follow a fixed canonical ordering rule:
`(q1 + c + q0, q1, c, q0)` (excitation-first then lexicographic), and each row is directed `|a> -> |b>` with that ordering.
Rows are aligned between Duffing and circuit by taking the union of states/transitions selected by each model.

Timing/selection settings are read from `params/benchmark_params.json` under:
- `leakage_flow_benchmark.total_time_ns`
- `leakage_flow_benchmark.ramp_time_ns`
- `leakage_flow_benchmark.dt_ns`
- `leakage_flow_benchmark.population_min_average`
- `leakage_flow_benchmark.transition_min_integrated_abs`
- `leakage_flow_benchmark.max_population_rows`
- `leakage_flow_benchmark.max_transition_rows`

Replot from saved leakage/flow data only:

```bash
python scripts/run_leakage_flow_benchmark.py --plot-only
```

Reuse fitted static parameters from a previous static run:

```bash
python scripts/run_leakage_flow_benchmark.py --from-fitted results/experiments/<static-run>/
```

Run the circuit truncation benchmark:

```bash
python scripts/run_circuit_truncation_benchmark.py
```

This reads settings from:
- `params/benchmark_params.json` under `circuit_truncation_benchmark`

This benchmark:
- aggregates static RMSE over 5 configured flux points,
- sweeps circuit charge-basis `ncut`,
- sweeps circuit dressed-space truncated dimensions `{qubit, coupler}`,
- compares each sweep point to one strict circuit reference point,
- reports `energy_rmse` together with `|dJ|` and `|dzeta|`,
- and writes results to an `.h5` file next to the configured figure.

Replot from saved circuit truncation data only:

```bash
python scripts/run_circuit_truncation_benchmark.py --plot-only
```

Run the Duffing truncation benchmark:

```bash
python scripts/run_duffing_truncation_benchmark.py
```

This reads settings from:
- `params/benchmark_params.json` under `duffing_truncation_benchmark`

This benchmark:
- aggregates static RMSE over 5 configured flux points,
- sweeps Duffing transmon calibration `ncut`,
- clips the extraction basis per point to `min(duffing_truncated_dim, 2*ncut+1)`,
- sweeps Duffing Hilbert-space truncated dimensions `{qubit, coupler}`,
- compares each sweep point to one strict circuit reference point,
- reports `energy_rmse` together with `|dJ|` and `|dzeta|`,
- and writes results to an `.h5` file next to the configured figure.

Replot from saved Duffing truncation data only:

```bash
python scripts/run_duffing_truncation_benchmark.py --plot-only
```

Run the CZ runtime benchmark (qubit truncation vs time for Duffing and circuit):

```bash
python scripts/run_runtime_benchmark.py
```

This benchmark:
- sweeps `runtime_benchmark.qubit_truncation_values`,
- applies each sweep value to Duffing `hilbert_truncation.nlevels_qubit`
  and to circuit `hilbert_truncation.q0_truncated_dim/q1_truncated_dim`,
- reuses the configured CZ benchmark pulse settings with one fixed hold time across the whole sweep,
- measures per-model CZ build and propagation time for Duffing and circuit,
- plots build and propagation runtime separately vs qubit truncation,
- stores the split runtime components plus spread across repeats in the saved `.h5`.

Runtime-benchmark settings are read from:
- `params/benchmark_params.json` under `runtime_benchmark`

For cleaner runtime results:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_runtime_benchmark.py
```

Replot from saved runtime benchmark data only:

```bash
python scripts/run_runtime_benchmark.py --plot-only
```

All scripts support `--results <path>` to override the default HDF5 path.
You can also rerender all plots from existing results:

```bash
python scripts/run_all_benchmarks.py --plot-only
```
