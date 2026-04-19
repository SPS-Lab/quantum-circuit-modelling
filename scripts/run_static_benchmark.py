"""Run paper static benchmark with parameters loaded from /params."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from study.comparison.static import run_static_benchmark
from study.config import load_study_config
from study.plots.static import plot_static_benchmark



def main() -> None:
    repo_root = _REPO_ROOT
    system_params_path = repo_root / "params" / "system_params.json"
    study_params_path = repo_root / "params" / "static_benchmark_params.json"

    config = load_study_config(system_params_path, study_params_path)
    result = run_static_benchmark(config)

    figure_path = repo_root / config.static_benchmark.outputs.figure
    title = (
        "Static benchmark across flux: effective vs Duffing vs circuit "
        f"(effective source={config.static_benchmark.effective_model.derivation_source})"
    )
    plot_static_benchmark(result, figure_path, title)

    print("Static benchmark summary (GHz):")
    for key, value in result.summary.items():
        print(f"  {key}: {value:.6e}")
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    main()
