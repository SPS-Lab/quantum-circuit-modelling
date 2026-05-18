"""Run combined circuit and Duffing static truncation-convergence benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_cli_reporting import (
    CliReporter,
    build_circuit_truncation_benchmark_extra_lines,
    build_common_truncation_lines,
    build_duffing_truncation_benchmark_extra_lines,
)
from benchmark_results_io import load_result_hdf5, save_result_hdf5
from benchmark_run_artifacts import prepare_benchmark_run
from comparison.truncation import TruncationBenchmarkResult, run_truncation_benchmark
from plotting.truncation import plot_truncation_benchmark
from study_config import load_study_config

_DEFAULT_FIGURE = Path("results/model_comparison_truncation_static_metrics.pdf")


def _selected_sweeps_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    selected: list[str] = []
    if args.only_ncut:
        selected.append("ncut")
    if args.only_qubit_dim:
        selected.append("qubit")
    if args.only_coupler_dim:
        selected.append("coupler")
    if not selected:
        return ("ncut", "qubit", "coupler")
    return tuple(selected)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--only-ncut",
        action="store_true",
        help="Run or plot only the charge-basis / extraction-ncut row.",
    )
    parser.add_argument(
        "--only-qubit-dim",
        action="store_true",
        help="Run or plot only the qubit Hilbert-dimension row.",
    )
    parser.add_argument(
        "--only-coupler-dim",
        action="store_true",
        help="Run or plot only the coupler Hilbert-dimension row.",
    )
    parser.add_argument(
        "--spectrum-metric",
        action="store_true",
        help=(
            "Also compute the broader low-energy full-spectrum RMSE for each truncation point. "
            "By default only the computational-manifold RMSE is computed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = _REPO_ROOT
    reporter = CliReporter(benchmark_name="truncation", script_name=Path(__file__).name)
    selected_sweeps = _selected_sweeps_from_args(args)
    config = load_study_config(
        system_params_path=repo_root / "params" / "system_params.json",
        study_params_path=repo_root / "params" / "benchmark_params.json",
    )
    run_paths = prepare_benchmark_run(
        repo_root=repo_root,
        benchmark_name="truncation",
        figure_paths={"figure": repo_root / _DEFAULT_FIGURE},
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
    figure_path = run_paths.figure_paths["figure"]
    results_path = run_paths.results_path

    if args.plot_only:
        try:
            result = load_result_hdf5(
                results_path,
                TruncationBenchmarkResult,
                expected_benchmark_name="truncation",
            )
        except ValueError as exc:
            raise ValueError(
                f"{results_path} does not match the current combined truncation benchmark schema. "
                "Re-run without --plot-only to regenerate the results file."
            ) from exc
    else:
        result = run_truncation_benchmark(
            config,
            selected_sweeps=selected_sweeps,
            include_spectrum_energy_metric=bool(args.spectrum_metric),
        )
        save_result_hdf5(result, results_path, benchmark_name="truncation")

    plot_truncation_benchmark(result, figure_path)

    for line in build_common_truncation_lines(config):
        reporter.line(line)
    for line in build_circuit_truncation_benchmark_extra_lines(config):
        reporter.line(line)
    for line in build_duffing_truncation_benchmark_extra_lines(config):
        reporter.line(line)
    reporter.line(f"Selected truncation sweeps: {', '.join(selected_sweeps)}")
    reporter.line("Combined truncation benchmark summary:")
    for key, value in result.summary.items():
        reporter.line(f"  {key}: {value:.6e}")
    if args.plot_only:
        reporter.line(f"Loaded results: {results_path}")
    else:
        reporter.line(f"Wrote results: {results_path}")
    reporter.line(f"Wrote figure: {figure_path}")
    if run_paths.git_head_path.exists():
        reporter.line(f"Wrote git head summary: {run_paths.git_head_path}")
    if run_paths.metadata_path.exists():
        reporter.line(f"Wrote run metadata: {run_paths.metadata_path}")
    if run_paths.git_snapshot_path.exists():
        reporter.line(f"Wrote git snapshot: {run_paths.git_snapshot_path}")
    reporter.add_runtime_line()
    reporter.persist(results_path)


if __name__ == "__main__":
    main()
