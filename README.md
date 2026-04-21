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

Run the static benchmark:

```bash
python scripts/run_static_benchmark.py
```

This writes:
- figure: `static_benchmark.outputs.figure`
- results: same path with `.h5` suffix

To replot without rerunning the benchmark:

```bash
python scripts/run_static_benchmark.py --plot-only
```

This reads:

- `params/system_params.json` (device/system parameters)
- `params/static_benchmark_params.json` (benchmark/model/plot settings)

The Duffing model supports calibration modes via
`static_benchmark.duffing_model.calibration_mode`:
- `fixed`: calibrate transmon Duffing parameters once at system parking biases
- `analytic-per-flux` (default): flux-dependent transmon approximation (no per-point numerical calibration)
- `per-flux`: recalibrate transmon Duffing parameters at every sweep point using transmon diagonalization

Run the CZ dynamics benchmark:

```bash
python scripts/run_cz_benchmark.py
```

The CZ benchmark:
- uses a shared flux pulse schedule for all models,
- uses a ramp-hold-ramp pulse configured under `cz_benchmark`,
- propagates effective + Duffing models with `numpy/scipy`,
- propagates the circuit model with `scqubits` Hamiltonians + `qutip`,
- focuses on CZ behavior/statevector from `|++>`,
- writes a figure with:
  - flux vs time,
  - conditional phase vs time for all models,
  - three computational heatmaps (effective, Duffing, circuit) where brightness is population and hue is relative phase.
- writes CZ results to an `.h5` file next to that figure.

Timing/scan settings are read from `params/static_benchmark_params.json` under:
- `cz_benchmark.total_time_ns`
- `cz_benchmark.ramp_time_ns`
- `cz_benchmark.dt_ns`
- `cz_benchmark.enable_hold_time_scan`
- `cz_benchmark.scan_dt_ns`
- `cz_benchmark.scan_max_hold_ns`
- `cz_benchmark.scan_leakage_penalty`

Replot from saved CZ data only:

```bash
python scripts/run_cz_benchmark.py --plot-only
```

Run the combined leakage/flow benchmark (from `|1,0,1>`):

```bash
python scripts/run_leakage_flow_benchmark.py
```

This benchmark uses a short pulse and writes one figure with 4 heatmaps:
- top row: population+phase heatmaps for Duffing and circuit models
- bottom row: signed transition-current heatmaps for Duffing and circuit models

Transition channels follow a fixed canonical ordering rule:
`(q1 + c + q2, q1, c, q2)` (excitation-first then lexicographic), and each row is directed `|a> -> |b>` with that ordering.
Rows are aligned between Duffing and circuit by taking the union of states/transitions selected by each model.

Timing/selection settings are read from `params/static_benchmark_params.json` under:
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

Run the fixed-flux truncation benchmark (`J`, `zeta` vs Duffing `ncut`):

```bash
python scripts/run_truncation_benchmark.py
```

This reads truncation settings from:
- `params/static_benchmark_params.json` under `truncation_benchmark`

This benchmark:
- fixes one flux point,
- sweeps Duffing transmon calibration `ncut`,
- uses `truncation_benchmark.duffing_truncated_dim` for transmon spectral extraction,
  with per-point safety clipping to `min(duffing_truncated_dim, 2*ncut+1)`,
- uses `truncation_benchmark.lowest_excited_levels_to_plot` to control how many
  lowest excited levels are shown in the level and level-difference subplots,
- plots Duffing `J`/`\zeta` against a circuit reference shown as horizontal lines,
- and shows the lowest relative energy levels vs `ncut` (Duffing curves with circuit horizontal references).
  (computed once at large circuit `ncut`).
- and prints `Duffing - circuit` numerically at the maximum Duffing `ncut`
  for those reported excited levels, including relative difference as percent
  of circuit energy.
- writes truncation results to an `.h5` file next to the configured truncation figure.

Replot from saved truncation data only:

```bash
python scripts/run_truncation_benchmark.py --plot-only
```

All scripts support `--results <path>` to override the default HDF5 path.
You can also rerender all plots from existing results:

```bash
python scripts/run_all_benchmarks.py --plot-only
```
