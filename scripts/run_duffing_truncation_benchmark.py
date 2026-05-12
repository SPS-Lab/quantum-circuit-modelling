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
from benchmark_results_io import default_results_path_for_figure, load_result_hdf5, save_result_hdf5
from comparison.truncation import DuffingTruncationBenchmarkResult, run_duffing_truncation_benchmark
from plotting.truncation import plot_duffing_truncation_benchmark
from study_config import load_study_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def _resolve_repo_relative(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def main() -> None:
    args = _parse_args()
    repo_root = _REPO_ROOT
    reporter = CliReporter(benchmark_name="duffing_truncation", script_name=Path(__file__).name)
    config = load_study_config(
        system_params_path=repo_root / "params" / "system_params.json",
        study_params_path=repo_root / "params" / "benchmark_params.json",
    )
    bench_cfg = config.duffing_truncation_benchmark
    figure_path = repo_root / bench_cfg.outputs.figure
    results_path = (
        _resolve_repo_relative(repo_root, args.results)
        if args.results is not None
        else default_results_path_for_figure(figure_path)
    )

    if args.plot_only:
        result = load_result_hdf5(
            results_path,
            DuffingTruncationBenchmarkResult,
            expected_benchmark_name="duffing_truncation",
        )
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
    reporter.line("Duffing Hilbert truncation sweep (q/c -> RMSE in GHz):")
    for qdim, cdim, total_rmse, energy_rmse, j_err, zeta_err in zip(
        result.duffing_hilbert_qubit_values,
        result.duffing_hilbert_coupler_values,
        result.duffing_hilbert_total_rmse,
        result.duffing_hilbert_energy_rmse,
        result.duffing_hilbert_j_abs_error,
        result.duffing_hilbert_zeta_abs_error,
    ):
        reporter.line(
            f"  {int(qdim):2d}/{int(cdim):2d}: total_rmse={float(total_rmse):.6e}, "
            f"energy_rmse={float(energy_rmse):.6e}, |dJ|={float(j_err):.6e}, |dzeta|={float(zeta_err):.6e}"
        )
    reporter.add_runtime_line()
    reporter.persist(results_path)


if __name__ == "__main__":
    main()
