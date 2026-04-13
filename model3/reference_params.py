"""Load model3 reference parameters from a repo-local JSON file."""

from __future__ import annotations

import json
from pathlib import Path


REFERENCE_PARAMS_PATH = Path(__file__).resolve().with_name("reference_params.json")
DEFAULT_TRANSMON_KEY = "ibm_manila_q1_04_2022"


def load_reference_params() -> dict:
    """Return parsed model3 reference parameter JSON."""
    with REFERENCE_PARAMS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_transmon_params(key: str = DEFAULT_TRANSMON_KEY) -> dict[str, float]:
    """Return transmon params (EJ, EC) for a named preset."""
    data = load_reference_params()
    catalog = data.get("transmon", {})
    if key not in catalog:
        raise KeyError(f"Unknown transmon key {key!r}. Available: {sorted(catalog.keys())}")
    params = dict(catalog[key])
    if "EJ" not in params or "EC" not in params:
        raise ValueError(f"Transmon params must include EJ and EC, got {params}")
    return {"EJ": float(params["EJ"]), "EC": float(params["EC"])}
