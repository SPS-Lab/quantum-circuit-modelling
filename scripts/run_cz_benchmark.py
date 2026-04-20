"""Run CZ benchmark with parameters loaded from /params."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from comparison.cz import run_cz_benchmark
from plots.cz import plot_cz_benchmark
from study_config import load_study_config



def main() -> None:
    repo_root = _REPO_ROOT
    config = load_study_config(
        repo_root / "params" / "system_params.json",
        repo_root / "params" / "static_benchmark_params.json",
    )
    result = run_cz_benchmark(config)

    static_figure = repo_root / config.static_benchmark.outputs.figure
    figure_path = static_figure.with_name("model_comparison_cz_dynamics.pdf")
    title = "CZ-relevant dynamics: effective vs Duffing vs circuit"
    plot_cz_benchmark(result, figure_path, title)

    print("CZ benchmark summary:")
    for key, value in result.summary.items():
        print(f"  {key}: {value:.6e}")
    print(
        "Selected pulse: "
        f"sweep_target={result.sweep_target}, idle_flux={result.idle_flux:.6f}, target_flux={result.target_flux:.6f}"
    )
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    main()
