"""Legacy script entrypoint migrated to the refactored study pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comparison.regime_map import compare_model1_model2_against_scqubits



def main() -> None:
    out = compare_model1_model2_against_scqubits(
        system_params_path=_ROOT / "params" / "system_params.json",
        study_params_path=_ROOT / "params" / "static_benchmark_params.json",
    )
    print("Migrated legacy test1 summary:")
    for key, value in out["summary"].items():
        print(f"  {key}: {float(value):.6e}")


if __name__ == "__main__":
    main()
