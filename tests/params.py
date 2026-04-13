import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
params_path = _ROOT / "model3" / "reference_params.json"

with params_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

print(data["transmon"]["ibm_manila_q1_04_2022"])
