# Benchmark

To build and simulate a single circuit-model CZ pulse, start with
[`scripts/cz_example.py`](../scripts/cz_example.py) and the
[simulation guide](simulation.md). The benchmark scripts described below
compare models and produce saved results, figures, and provenance metadata.

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
- `params/benchmark_params.json` (benchmark/model/plot settings, including numerical basis cutoffs such as circuit-model transmon `ncut`)

For the current DiCarlo-style fixed harmonic bus workflow, the coupler frequency
is taken from `system_params.json` as `system.parameters.c.E_osc`. The static
flux sweep target is qubit-only (`q0` or `q1`); benchmark config no longer
defines a separate tunable `wc`.

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

Run the combined truncation benchmark:

```bash
python scripts/run_truncation_benchmark.py
```

This reads settings from:
- `params/benchmark_params.json` under both `circuit_truncation_benchmark` and `duffing_truncation_benchmark`

This benchmark:
- aggregates static RMSE over the full static flux sweep by default,
- optionally accepts explicit truncation-only `flux_values` lists for the circuit and Duffing truncation studies,
- runs six convergence sweeps in one figure:
  - circuit `N_Q`, `N_{E,q}`, `N_{E,c}` on the left
  - Duffing `N_Q`, `N_{E,q}`, `N_{E,c}` on the right
- compares each sweep point to one strict circuit reference point,
- reports `energy_rmse` together with `|dJ|` and `|dzeta|`,
- and writes results to an `.h5` file next to the configured figure.

Replot from saved combined truncation data only:

```bash
python scripts/run_truncation_benchmark.py --plot-only
```

The old standalone scripts still exist if you want a one-family-only run:

```bash
python scripts/run_circuit_truncation_benchmark.py
python scripts/run_duffing_truncation_benchmark.py
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

Set `runtime_benchmark.circuit_only` to `true` to time only the circuit model.
In that mode, `runtime_benchmark.duffing_calibration_mode` may be omitted.
This skips Duffing static fitting and effective and Duffing model construction
and propagation, including during the priming run.

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

All benchmark scripts support `--results <path>` to override the default HDF5 path.
You can also rerender all plots from existing results:

```bash
python scripts/run_all_benchmarks.py --plot-only
```

## Dask strong scaling

Run a fixed circuit CZ workload while varying Dask concurrency:

```bash
python scripts/run_dask_scaling_benchmark.py
```

Defaults live in [`params/dask_scaling_params.json`](../params/dask_scaling_params.json):
`N_E,q=8`, parallelism `1, 2, 4, 8`, five repeats, one full-workload warmup,
and one BLAS thread per task. The pulse matches the example's runtime-paper
pulse: target flux `0.23333333333333334`, 2 ns ramps, hold
`237.9595316591736 ns`, and dt `0.5 ns`. Device parameters, coupler dimension,
and charge cutoffs come from the usual system and benchmark parameter files.
All four computational basis inputs are propagated. The workload stays fixed
throughout the concurrency sweep, without a flux or hold-time scan.

```bash
python scripts/run_dask_scaling_benchmark.py --qubit-dim 15 --parallelism 1 2 4 8
python scripts/run_dask_scaling_benchmark.py --plot-only
```

Thread mode uses one worker with `p` threads; process mode uses `p` workers
with one thread each. BLAS thread limits are applied in the parent and every
worker, and actual worker threadpool counts are saved. The modes and concurrency
points run in a seeded shuffled order; each cluster is reused for warmups and
measured repeats. Warmup work is recomputed rather than cached. Each measured
run calls the shared `simulate_circuit` workflow, building its Hamiltonians
serially and timing that build separately.

The figure shows unitary preparation and preparation-plus-propagation runtimes
and speedups. Runtimes are medians with min/max ranges. Strong-scaling speedup
is `median(T_mode,1) / median(T_mode,p)`, using each mode's own one-worker
baseline. A separate serial reference exposes Dask overhead. Preparation
includes matrix scatter/gather; cluster startup, teardown, warmups, and
numerical validation are excluded. Startup times are saved separately.
Amdahl's law limits the combined speedup because state propagation is sequential.
The default compressed pulse has only nine segment tasks, so useful concurrency
is limited by both task count and differences in task cost.

In addition to the standard timestamped provenance directory, this benchmark
saves:

- `dask_scaling_results.h5`: every timing sample, pulse schedule, serial reference
  amplitudes, numerical differences, execution order, and effective settings.
- `dask_scaling_samples.csv`: raw serial/thread/process timing samples, including
  build-plus-propagation totals.
- `effective_settings.json` and `execution_environment.json`: resolved circuit
  and pulse settings, package versions, worker PIDs, worker counts, thread counts,
  memory limits, CPU affinity, and actual BLAS/OpenMP threadpool details.
- Timing and speedup PDFs in the standard paper and presentation styles.
- `dask_scaling_summary.csv`: one row per stage, Dask mode, and parallelism,
  plus one serial reference row per stage. It is regenerated on normal runs
  and `--plot-only`. Plots use medians with min/max ranges; speedups are ratios
  of medians.
- Source snapshots for the scaling benchmark, script, plotter, public simulation
  module, and solver, including their contents when they are still untracked.

Every warmup and measured trajectory is checked against the serial reference
outside the timed sections. Use `--dashboard-address 127.0.0.1:0` for an available
port; each active cluster's URL is printed. This benchmark runs separately from
`run_all_benchmarks.py`, which reproduces the paper's original benchmarks.
