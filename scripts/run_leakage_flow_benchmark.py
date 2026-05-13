"""Run combined leakage/population + transition-flow benchmark with /params config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_results_io import (
    load_result_hdf5,
    save_result_hdf5,
)
from benchmark_run_artifacts import prepare_benchmark_run
from benchmark_cli_reporting import CliReporter, build_common_truncation_lines
from comparison.leakage_flow import LeakageFlowBenchmarkResult, run_leakage_flow_benchmark
from plotting.leakage_flow import plot_leakage_flow_benchmark
from static_fitted_artifacts import (
    load_static_fitted_models_artifact,
    resolve_static_fitted_artifact_path,
)
from study_config import load_study_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "Path to HDF5 results file. When omitted, a new timestamped experiment "
            "directory is created; with --plot-only, the newest leakage-flow run is used."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmark computation and plot from an existing HDF5 results file.",
    )
    parser.add_argument(
        "--from-fitted",
        type=Path,
        default=None,
        help=(
            "Optional static fitted-parameter artifact (.json, .h5, or run directory) "
            "to reuse instead of recomputing the static benchmark."
        ),
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
    reporter = CliReporter(benchmark_name="leakage_flow", script_name=Path(__file__).name)
    config = load_study_config(
        system_params_path=repo_root / "params" / "system_params.json",
        study_params_path=repo_root / "params" / "benchmark_params.json",
    )

    lf_cfg = config.leakage_flow_benchmark
    target_total_time_ns = float(lf_cfg.total_time_ns)
    ramp_time_ns = float(lf_cfg.ramp_time_ns)
    hold_time_ns = target_total_time_ns - 2.0 * ramp_time_ns
    dt_ns = float(lf_cfg.dt_ns)

    run_paths = prepare_benchmark_run(
        repo_root=repo_root,
        benchmark_name="leakage_flow",
        figure_paths={"figure": repo_root / lf_cfg.outputs.figure},
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
    fitted_source_path = (
        None
        if args.plot_only or args.from_fitted is None
        else resolve_static_fitted_artifact_path(args.from_fitted)
    )
    precomputed_static_result = (
        None if fitted_source_path is None else load_static_fitted_models_artifact(fitted_source_path)
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
            precomputed_static_result=precomputed_static_result,
        )
        save_result_hdf5(result, results_path, benchmark_name="leakage_flow")

    title = (
        "Leakage/flow benchmark from |1,0,1>: "
        "population+phase states and canonical signed transitions"
    )
    plot_leakage_flow_benchmark(result, figure_path, title)

    for line in build_common_truncation_lines(config):
        reporter.line(line)
    reporter.line("Leakage/flow benchmark summary:")
    for key, value in result.summary.items():
        reporter.line(f"  {key}: {value:.6e}")
    reporter.line(
        "Selected pulse: "
        f"sweep_target={result.sweep_target}, idle_flux={result.idle_flux:.6f}, "
        f"target_flux={result.target_flux:.6f}, ramp_time_ns={result.ramp_time_ns:.3f}, "
        f"hold_time_ns={result.hold_time_ns:.3f}, dt_ns={result.dt_ns:.3f}"
    )
    if args.plot_only:
        reporter.line(f"Loaded results: {results_path}")
    else:
        reporter.line(f"Wrote results: {results_path}")
    if fitted_source_path is not None:
        reporter.line(f"Reused fitted static artifact: {fitted_source_path}")
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
