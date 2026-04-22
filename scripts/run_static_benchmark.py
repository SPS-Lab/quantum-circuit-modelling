"""Run paper static benchmark with parameters loaded from /params."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_results_io import (
    default_results_path_for_figure,
    load_result_hdf5,
    save_result_hdf5,
)
from comparison.static import StaticBenchmarkResult, run_static_benchmark
from plotting.static import plot_static_benchmark
from runtime_utils import run_main_with_timing
from study_config import load_study_config



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to HDF5 results file (default: figure path with .h5 suffix).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmark computation and plot from an existing HDF5 results file.",
    )
    return parser.parse_args()


def _resolve_repo_relative(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def main() -> None:
    args = _parse_args()
    repo_root = _REPO_ROOT
    system_params_path = repo_root / "params" / "system_params.json"
    study_params_path = repo_root / "params" / "benchmark_params.json"

    config = load_study_config(system_params_path, study_params_path)
    figure_path = repo_root / config.static_benchmark.outputs.figure
    results_path = (
        _resolve_repo_relative(repo_root, args.results)
        if args.results is not None
        else default_results_path_for_figure(figure_path)
    )

    if args.plot_only:
        result = load_result_hdf5(
            results_path,
            StaticBenchmarkResult,
            expected_benchmark_name="static",
        )
    else:
        result = run_static_benchmark(config)
        save_result_hdf5(result, results_path, benchmark_name="static")

    title = (
        "Static benchmark across flux: effective vs Duffing vs circuit "
        f"(effective source={config.static_benchmark.effective_model.derivation_source})"
    )
    plot_static_benchmark(result, figure_path, title)

    print("Static benchmark summary (GHz):")
    for key, value in result.summary.items():
        print(f"  {key}: {value:.6e}")
    if args.plot_only:
        print(f"Loaded results: {results_path}")
    else:
        print(f"Wrote results: {results_path}")
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    run_main_with_timing(main)
