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
- `comparison`: benchmark logic (`static`, `cz`, `leakage`)
- `plots`: plotting only
- `study_config.py`: typed config loading/validation
- `params`: all runtime parameters consumed by main scripts

Legacy `model0/1/2/3` packages have been merged into the modules above.

Run the static benchmark:

```bash
python scripts/run_static_benchmark.py
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
- uses a ramp-hold-ramp pulse with automatic hold-time scan toward CZ phase (`pi`),
- propagates effective + Duffing models with `numpy/scipy`,
- propagates the circuit model with `scqubits` Hamiltonians + `qutip`,
- focuses on CZ behavior/statevector from `|++>` and writes a CZ figure next to the configured static figure path.

Run the leakage benchmark (from `|11>`):

```bash
python scripts/run_leakage_benchmark.py
```

The leakage benchmark reuses the same calibrated pulse and reports/plots leakage-focused dynamics separately.

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
