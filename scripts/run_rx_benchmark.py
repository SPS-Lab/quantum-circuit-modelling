"""Run RX benchmark with parameters loaded from /params."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_cli_reporting import CliReporter, build_common_truncation_lines
from benchmark_results_io import (
    load_result_hdf5,
    save_result_hdf5,
)
from benchmark_run_artifacts import prepare_benchmark_run
from comparison.rx import RxBenchmarkResult, run_rx_benchmark
from plotting.rx import plot_rx_diagnostics_benchmark, plot_rx_populations_benchmark
from study_config import load_study_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "Path to HDF5 results file. When omitted, a new timestamped experiment "
            "directory is created; with --plot-only, the newest RX run is used."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmark computation and plot from an existing HDF5 results file.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name used in the timestamped run-directory name.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root directory for timestamped experiment runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = _REPO_ROOT
    reporter = CliReporter(benchmark_name="rx", script_name=Path(__file__).name)
    config = load_study_config(
        system_params_path=repo_root / "params" / "system_params.json",
        study_params_path=repo_root / "params" / "benchmark_params.json",
    )
    rx_cfg = config.rx_benchmark

    run_paths = prepare_benchmark_run(
        repo_root=repo_root,
        benchmark_name="rx",
        figure_paths={
            "populations": repo_root / rx_cfg.outputs.populations_figure,
            "diagnostics": repo_root / rx_cfg.outputs.diagnostics_figure,
        },
        results_path_arg=args.results,
        plot_only=bool(args.plot_only),
        experiment_name=args.experiment_name,
        output_root=args.output_root,
        argv=sys.argv,
        input_files={
            "system_params": repo_root / "params" / "system_params.json",
            "benchmark_params": repo_root / "params" / "benchmark_params.json",
        },
    )
    populations_figure_path = run_paths.figure_paths["populations"]
    diagnostics_figure_path = run_paths.figure_paths["diagnostics"]
    results_path = run_paths.results_path

    if args.plot_only:
        result = load_result_hdf5(
            results_path,
            RxBenchmarkResult,
            expected_benchmark_name="rx",
        )
    else:
        result = run_rx_benchmark(
            config,
            drive_qubit=str(rx_cfg.drive_qubit),
            drive_frequency=float(rx_cfg.drive_frequency),
            drive_amplitude=float(rx_cfg.drive_amplitude),
            drive_phase_rad=float(rx_cfg.drive_phase_rad),
            total_time_ns=float(rx_cfg.total_time_ns),
            dt_ns=float(rx_cfg.dt_ns),
            rise_time_ns=float(rx_cfg.rise_time_ns),
        )
        save_result_hdf5(result, results_path, benchmark_name="rx")

    plot_rx_populations_benchmark(result, populations_figure_path)
    plot_rx_diagnostics_benchmark(result, diagnostics_figure_path)

    for line in build_common_truncation_lines(config):
        reporter.line(line)
    reporter.line("RX benchmark summary:")
    rx_summary_keys = [
        "effective_final_pop_00_to_01",
        "duffing_final_pop_00_to_01",
        "circuit_final_pop_00_to_01",
        "effective_final_pop_10_to_11",
        "duffing_final_pop_10_to_11",
        "circuit_final_pop_10_to_11",
        "effective_max_leakage_from_00",
        "duffing_max_leakage_from_00",
        "circuit_max_leakage_from_00",
        "effective_final_spectator_population_delta",
        "duffing_final_spectator_population_delta",
        "circuit_final_spectator_population_delta",
        "drive_frequency",
        "drive_amplitude",
        "drive_phase_rad",
        "total_time_ns",
        "dt_ns",
        "rise_time_ns",
    ]
    for key in rx_summary_keys:
        if key in result.summary:
            reporter.line(f"  {key}: {result.summary[key]:.6e}")
    if args.plot_only:
        reporter.line(f"Loaded results: {results_path}")
    else:
        reporter.line(f"Wrote results: {results_path}")
    reporter.line(f"Wrote populations figure: {populations_figure_path}")
    reporter.line(f"Wrote diagnostics figure: {diagnostics_figure_path}")
    if run_paths.metadata_path.exists():
        reporter.line(f"Wrote run metadata: {run_paths.metadata_path}")
    if run_paths.git_snapshot_path.exists():
        reporter.line(f"Wrote git snapshot: {run_paths.git_snapshot_path}")
    reporter.add_runtime_line()
    reporter.persist(results_path)


if __name__ == "__main__":
    main()
