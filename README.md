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

## Comparison scripts

Run the main comparison workflows:

```bash
python tests/test1.py
python tests/test_full_range/test_full_range.py
python tests/test_model3/test_regime_map.py
```

Focused pytest run (fast, recommended during iteration):

```bash
pytest -q tests/test_model3/test_regime_map_pytest.py
```
