"""Run Duffing static truncation-convergence benchmark with parameters loaded from /params."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_cli_reporting import (
    CliReporter,
    build_common_truncation_lines,
    build_duffing_truncation_benchmark_extra_lines,
)
from benchmark_results_io import load_result_hdf5, save_result_hdf5
from benchmark_run_artifacts import prepare_benchmark_run
from comparison.truncation import DuffingTruncationBenchmarkResult, run_duffing_truncation_benchmark
from plotting.truncation import plot_duffing_truncation_benchmark
from study_config import load_study_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = _REPO_ROOT
    reporter = CliReporter(benchmark_name="duffing_truncation", script_name=Path(__file__).name)
    config = load_study_config(
        system_params_path=repo_root / "params" / "system_params.json",
        study_params_path=repo_root / "params" / "benchmark_params.json",
    )
    bench_cfg = config.duffing_truncation_benchmark
    run_paths = prepare_benchmark_run(
        repo_root=repo_root,
        benchmark_name="duffing_truncation",
        figure_paths={"figure": repo_root / bench_cfg.outputs.figure},
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
                DuffingTruncationBenchmarkResult,
                expected_benchmark_name="duffing_truncation",
            )
        except ValueError as exc:
            raise ValueError(
                f"{results_path} does not match the current Duffing truncation benchmark schema. "
                "Re-run without --plot-only to regenerate the results file."
            ) from exc
    else:
        result = run_duffing_truncation_benchmark(config)
        save_result_hdf5(result, results_path, benchmark_name="duffing_truncation")

    plot_duffing_truncation_benchmark(result, figure_path)

    for line in build_common_truncation_lines(config):
        reporter.line(line)
    for line in build_duffing_truncation_benchmark_extra_lines(config):
        reporter.line(line)
    reporter.line("Duffing truncation benchmark summary:")
    for key, value in result.summary.items():
        reporter.line(f"  {key}: {value:.6e}")
    reporter.line("Duffing extraction ncut sweep (RMSE in GHz):")
    for ncut, trunc_dim, total_rmse, energy_rmse, j_err, zeta_err in zip(
        result.duffing_ncut_values,
        result.duffing_ncut_effective_truncated_dim_values,
        result.duffing_ncut_total_rmse,
        result.duffing_ncut_energy_rmse,
        result.duffing_ncut_j_abs_error,
        result.duffing_ncut_zeta_abs_error,
    ):
        reporter.line(
            f"  ncut={int(ncut):4d}, trunc_dim={int(trunc_dim):3d}: total_rmse={float(total_rmse):.6e}, "
            f"energy_rmse={float(energy_rmse):.6e}, |dJ|={float(j_err):.6e}, |dzeta|={float(zeta_err):.6e}"
        )
    reporter.line("Duffing qubit Hilbert-dim sweep (coupler fixed; RMSE in GHz):")
    for qdim, total_rmse, energy_rmse, j_err, zeta_err in zip(
        result.duffing_hilbert_qubit_dim_values,
        result.duffing_hilbert_qubit_total_rmse,
        result.duffing_hilbert_qubit_energy_rmse,
        result.duffing_hilbert_qubit_j_abs_error,
        result.duffing_hilbert_qubit_zeta_abs_error,
    ):
        reporter.line(
            f"  q={int(qdim):2d}: total_rmse={float(total_rmse):.6e}, "
            f"energy_rmse={float(energy_rmse):.6e}, |dJ|={float(j_err):.6e}, |dzeta|={float(zeta_err):.6e}"
        )
    reporter.line("Duffing coupler Hilbert-dim sweep (qubit fixed; RMSE in GHz):")
    for cdim, total_rmse, energy_rmse, j_err, zeta_err in zip(
        result.duffing_hilbert_coupler_dim_values,
        result.duffing_hilbert_coupler_total_rmse,
        result.duffing_hilbert_coupler_energy_rmse,
        result.duffing_hilbert_coupler_j_abs_error,
        result.duffing_hilbert_coupler_zeta_abs_error,
    ):
        reporter.line(
            f"  c={int(cdim):2d}: total_rmse={float(total_rmse):.6e}, "
            f"energy_rmse={float(energy_rmse):.6e}, |dJ|={float(j_err):.6e}, |dzeta|={float(zeta_err):.6e}"
        )
    if args.plot_only:
        reporter.line(f"Loaded results: {results_path}")
    else:
        reporter.line(f"Wrote results: {results_path}")
    reporter.line(f"Wrote figure: {figure_path}")
    if run_paths.metadata_path.exists():
        reporter.line(f"Wrote run metadata: {run_paths.metadata_path}")
    if run_paths.git_snapshot_path.exists():
        reporter.line(f"Wrote git snapshot: {run_paths.git_snapshot_path}")
    reporter.add_runtime_line()
    reporter.persist(results_path)


if __name__ == "__main__":
    main()
