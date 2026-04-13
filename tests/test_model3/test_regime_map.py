"""Run and summarize model1/model2 vs scqubits regime-of-validity comparison."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Repo root so `model3` resolves when executed directly.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model3.comparison import compare_model1_model2_against_scqubits
from model3.reference_params import DEFAULT_TRANSMON_KEY, load_transmon_params


def main() -> None:
    ham_kwargs = {
        "w_1": 5.0,
        "w_2": 5.12,
        "alpha_1": -0.28,
        "alpha_c": -0.22,
        "alpha_2": -0.31,
        "g_1c": 0.12,
        "g_2c": 0.105,
        "nlevels_qubit": 2,
        "nlevels_coupler": 2,
    }
    flux = np.linspace(0.0, 1.0, 121)
    outdir = Path(__file__).resolve().parent
    outfile = outdir / "model1_model2_vs_scqubits_regime_map_test.pdf"
    transmon_params = load_transmon_params(DEFAULT_TRANSMON_KEY)

    out = compare_model1_model2_against_scqubits(
        flux,
        wc0=5.05,
        A=0.95,
        reference={
            "transmon1_params": transmon_params,
            "transmon2_params": transmon_params,
            "transmon_dim": 5,
            "coupler_dim": 6,
            "g_1c": 0.09,
            "g_2c": 0.085,
        },
        model1_mode="cosine-fit",
        outfile=str(outfile),
        **ham_kwargs,
    )

    print("Summary (RMSE/max_abs in GHz):")
    for key, value in out["summary"].items():
        print(f"  {key}: {float(value):.6e}")
    print(f"Wrote: {outfile}")


if __name__ == "__main__":
    main()
