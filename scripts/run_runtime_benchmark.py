"""Run CZ runtime-vs-qubit-truncation benchmark with parameters loaded from /params."""

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
from comparison.runtime import RuntimeBenchmarkResult, run_runtime_benchmark
from plotting.runtime import plot_runtime_benchmark
from study_config import load_study_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "Path to HDF5 results file. When omitted, a new timestamped experiment "
            "directory is created; with --plot-only, the newest runtime run is used."
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
    reporter = CliReporter(benchmark_name="runtime", script_name=Path(__file__).name)
    config = load_study_config(
        system_params_path=repo_root / "params" / "system_params.json",
        study_params_path=repo_root / "params" / "benchmark_params.json",
    )
    runtime_cfg = config.runtime_benchmark

    run_paths = prepare_benchmark_run(
        repo_root=repo_root,
        benchmark_name="runtime",
        figure_paths={"figure": repo_root / runtime_cfg.outputs.figure},
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
        result = load_result_hdf5(
            results_path,
            RuntimeBenchmarkResult,
            expected_benchmark_name="runtime",
        )
    else:
        result = run_runtime_benchmark(
            config,
            qubit_truncation_values=list(runtime_cfg.qubit_truncation_values),
            duffing_calibration_mode=str(runtime_cfg.duffing_calibration_mode),
            repeats=int(runtime_cfg.repeats),
            hold_time_ns=runtime_cfg.hold_time_ns,
        )
        save_result_hdf5(result, results_path, benchmark_name="runtime")

    plot_runtime_benchmark(result, figure_path, title="CZ Runtime Benchmark")

    for line in build_common_truncation_lines(config):
        reporter.line(line)
    reporter.line("Runtime benchmark sweep settings:")
    reporter.line(f"  duffing_calibration_mode={result.duffing_calibration_mode}")
    reporter.line(f"  repeats={int(result.repeats)}")
    reporter.line(f"  fixed_hold_time_ns={float(result.fixed_hold_time_ns):.6f}")
    reporter.line(f"  qubit_truncation_values={[int(v) for v in result.qubit_truncation_values]}")
    reporter.line("Runtime benchmark summary:")
    for key, value in result.summary.items():
        reporter.line(f"  {key}: {value:.6e}")
    reporter.line("Per-truncation CZ build/propagation runtime (s):")
    for idx, trunc in enumerate(result.qubit_truncation_values):
        reporter.line(
            f"  qubit_trunc={int(trunc):4d}, "
            f"duf_dim={int(result.duffing_hilbert_dims[idx]):4d}, "
            f"cir_dim={int(result.circuit_hilbert_dims[idx]):4d}, "
            f"hold_ns={float(result.selected_hold_times_ns[idx]):9.6f}, "
            f"duffing_build={float(result.duffing_build_runtime_s[idx]):.6e} +/- "
            f"{float(result.duffing_build_runtime_std_s[idx]):.2e}, "
            f"duffing_prop={float(result.duffing_propagation_runtime_s[idx]):.6e} +/- "
            f"{float(result.duffing_propagation_runtime_std_s[idx]):.2e}, "
            f"circuit_build={float(result.circuit_build_runtime_s[idx]):.6e} +/- "
            f"{float(result.circuit_build_runtime_std_s[idx]):.2e}, "
            f"circuit_prop={float(result.circuit_propagation_runtime_s[idx]):.6e} +/- "
            f"{float(result.circuit_propagation_runtime_std_s[idx]):.2e}"
        )
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
