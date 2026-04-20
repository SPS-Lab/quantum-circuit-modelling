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
- `comparison`: benchmark logic (`static`, CZ/leakage headers)
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

CZ/leakage benchmark headers exist but are intentionally not implemented yet:

```bash
python scripts/run_cz_benchmark.py
python scripts/run_leakage_benchmark.py
```
