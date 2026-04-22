"""Run and summarize migrated regime comparison."""

from __future__ import annotations

from pathlib import Path
import sys

# Repo root so `comparison` resolves when executed directly.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comparison.regime_map import compare_model1_model2_against_scqubits



def main() -> None:
    outdir = Path(__file__).resolve().parent
    outfile = outdir / "model1_model2_vs_scqubits_regime_map_test.pdf"

    out = compare_model1_model2_against_scqubits(
        system_params_path=_ROOT / "params" / "system_params.json",
        study_params_path=_ROOT / "params" / "benchmark_params.json",
        outfile=outfile,
    )

    print("Summary (RMSE/max_abs in GHz):")
    for key, value in out["summary"].items():
        print(f"  {key}: {float(value):.6e}")
    print(f"Wrote: {outfile}")


if __name__ == "__main__":
    main()
