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

## Refactored Study Workflow

The paper-aligned workflow is now organized by responsibility:

- `study/models`: model builders (`effective`, `duffing`, `circuit`)
- `study/comparison`: benchmark logic (`static`, CZ/leakage headers)
- `study/plots`: plotting only
- `params`: all runtime parameters consumed by main scripts

Run the static benchmark:

```bash
python scripts/run_static_benchmark.py
```

This reads:

- `params/system_params.json` (device/system parameters)
- `params/static_benchmark_params.json` (benchmark/model/plot settings)

CZ/leakage benchmark headers exist but are intentionally not implemented yet:

```bash
python scripts/run_cz_benchmark.py
python scripts/run_leakage_benchmark.py
```
