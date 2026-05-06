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
    default_results_path_for_figure,
    load_result_hdf5,
    save_result_hdf5,
)
from comparison.rx import RxBenchmarkResult, run_rx_benchmark
from plotting.rx import plot_rx_diagnostics_benchmark, plot_rx_populations_benchmark
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
    reporter = CliReporter(benchmark_name="rx", script_name=Path(__file__).name)
    config = load_study_config(
        repo_root / "params" / "system_params.json",
        repo_root / "params" / "benchmark_params.json",
    )
    rx_cfg = config.rx_benchmark

    populations_figure_path = repo_root / rx_cfg.outputs.populations_figure
    diagnostics_figure_path = repo_root / rx_cfg.outputs.diagnostics_figure
    results_path = (
        _resolve_repo_relative(repo_root, args.results)
        if args.results is not None
        else default_results_path_for_figure(populations_figure_path)
    )

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
    reporter.add_runtime_line()
    reporter.persist(results_path)


if __name__ == "__main__":
    main()
