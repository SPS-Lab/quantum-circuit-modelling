# scmodels

Code to answer the question "Which Superconducting-Qubit Model Is Good Enough?"

Calibration details for the models presented in the paper is presented in [docs/calibration.md](docs/calibration.md).

## Quick Start

Setup the environment:

```bash
micromamba create -f scmodels-env.yml
micromamba activate scmodels-env
```

To reproduce all figures in the paper, run:

```bash
python scripts/run_all_benchmarks.py
```

More information on the benchmark system is provided in [docs/benchmark.md](docs/benchmark.md).
