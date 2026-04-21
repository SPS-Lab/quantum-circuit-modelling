"""Run combined leakage/population + transition-flow benchmark with /params config."""

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
from comparison.leakage_flow import LeakageFlowBenchmarkResult, run_leakage_flow_benchmark
from plotting.leakage_flow import plot_leakage_flow_benchmark
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
    config = load_study_config(
        repo_root / "params" / "system_params.json",
        repo_root / "params" / "static_benchmark_params.json",
    )

    lf_cfg = config.leakage_flow_benchmark
    target_total_time_ns = float(lf_cfg.total_time_ns)
    ramp_time_ns = float(lf_cfg.ramp_time_ns)
    hold_time_ns = target_total_time_ns - 2.0 * ramp_time_ns
    dt_ns = float(lf_cfg.dt_ns)

    static_figure = repo_root / config.static_benchmark.outputs.figure
    figure_path = static_figure.with_name("model_comparison_leakage_flow.pdf")
    results_path = (
        _resolve_repo_relative(repo_root, args.results)
        if args.results is not None
        else default_results_path_for_figure(figure_path)
    )

    if args.plot_only:
        result = load_result_hdf5(
            results_path,
            LeakageFlowBenchmarkResult,
            expected_benchmark_name="leakage_flow",
        )
    else:
        result = run_leakage_flow_benchmark(
            config,
            ramp_time_ns=ramp_time_ns,
            hold_time_ns=hold_time_ns,
            dt_ns=dt_ns,
            population_min_average=float(lf_cfg.population_min_average),
            transition_min_integrated_abs=float(lf_cfg.transition_min_integrated_abs),
            max_population_rows=int(lf_cfg.max_population_rows),
            max_transition_rows=int(lf_cfg.max_transition_rows),
        )
        save_result_hdf5(result, results_path, benchmark_name="leakage_flow")

    title = (
        "Leakage/flow benchmark from |1,0,1>: "
        "population+phase states and canonical signed transitions"
    )
    plot_leakage_flow_benchmark(result, figure_path, title)

    print("Leakage/flow benchmark summary:")
    for key, value in result.summary.items():
        print(f"  {key}: {value:.6e}")
    print(
        "Selected pulse: "
        f"sweep_target={result.sweep_target}, idle_flux={result.idle_flux:.6f}, "
        f"target_flux={result.target_flux:.6f}, ramp_time_ns={result.ramp_time_ns:.3f}, "
        f"hold_time_ns={result.hold_time_ns:.3f}, dt_ns={result.dt_ns:.3f}"
    )
    if args.plot_only:
        print(f"Loaded results: {results_path}")
    else:
        print(f"Wrote results: {results_path}")
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    main()
