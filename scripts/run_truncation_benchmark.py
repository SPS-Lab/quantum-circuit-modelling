"""Run fixed-flux Duffing truncation benchmark with parameters loaded from /params."""

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
from benchmark_cli_reporting import (
    CliReporter,
    build_common_truncation_lines,
    build_truncation_benchmark_extra_lines,
)
from comparison.truncation import TruncationBenchmarkResult, run_truncation_benchmark
from plotting.truncation import plot_truncation_benchmark
from study_config import load_study_config

_NUMERICAL_ERROR_LEVELS_TO_REPORT = (5, 6, 7, 8)


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
    reporter = CliReporter(benchmark_name="truncation", script_name=Path(__file__).name)
    config = load_study_config(
        repo_root / "params" / "system_params.json",
        repo_root / "params" / "benchmark_params.json",
    )
    trunc_cfg = config.truncation_benchmark

    configured_figure = Path(trunc_cfg.outputs.figure)
    if configured_figure.is_absolute():
        figure_path = configured_figure
    else:
        figure_path = repo_root / configured_figure
    results_path = (
        _resolve_repo_relative(repo_root, args.results)
        if args.results is not None
        else default_results_path_for_figure(figure_path)
    )

    if args.plot_only:
        result = load_result_hdf5(
            results_path,
            TruncationBenchmarkResult,
            expected_benchmark_name="truncation",
        )
    else:
        result = run_truncation_benchmark(
            config,
            duffing_ncut_values=list(trunc_cfg.duffing_ncut_values),
            fixed_flux=float(trunc_cfg.fixed_flux),
            duffing_truncated_dim=int(trunc_cfg.duffing_truncated_dim),
            lowest_excited_levels_to_report=int(trunc_cfg.lowest_excited_levels_to_plot),
            reported_excited_levels=list(_NUMERICAL_ERROR_LEVELS_TO_REPORT),
            circuit_reference_ncut=int(trunc_cfg.circuit_reference_ncut),
            duffing_calibration_mode=str(trunc_cfg.duffing_calibration_mode),
        )
        save_result_hdf5(result, results_path, benchmark_name="truncation")

    title = (
        "Fixed-flux truncation benchmark: Duffing vs circuit "
        f"(flux={result.flux:.6f}, target={result.sweep_target})"
    )
    plot_truncation_benchmark(
        result,
        figure_path,
        title,
        lowest_excited_levels_to_plot=int(trunc_cfg.lowest_excited_levels_to_plot),
    )

    for line in build_common_truncation_lines(config):
        reporter.line(line)
    for line in build_truncation_benchmark_extra_lines(config):
        reporter.line(line)
    reporter.line("Truncation benchmark summary:")
    for key, value in result.summary.items():
        reporter.line(f"  {key}: {value:.6e}")

    reporter.line("Per-ncut metrics (GHz):")
    for ncut, trunc_dim, j, zeta in zip(
        result.duffing_ncut_values,
        result.duffing_effective_truncated_dim_values,
        result.duffing_j,
        result.duffing_zeta,
    ):
        reporter.line(
            f"  ncut={int(ncut):4d}, trunc_dim={int(trunc_dim):3d}: "
            f"J={j:.6e}, zeta={zeta:.6e}"
        )
    reporter.line(
        "Circuit reference (GHz): "
        f"ncut={result.circuit_reference_ncut}, "
        f"J={result.circuit_j:.6e}, zeta={result.circuit_zeta:.6e}"
    )
    levels_reported_text = ", ".join(f"E{int(level)}" for level in result.max_ncut_reported_excited_levels)
    if not levels_reported_text:
        levels_reported_text = "none available"
    reporter.line(
        "Duffing - circuit at max Duffing ncut "
        f"(ncut={result.max_duffing_ncut}) for excited levels {levels_reported_text} (GHz):"
    )
    for level, diff, rel_pct in zip(
        result.max_ncut_reported_excited_levels,
        result.duffing_minus_circuit_at_max_ncut,
        result.duffing_minus_circuit_percent_of_circuit_at_max_ncut,
    ):
        rel_text = f"{float(rel_pct):.6f}%"
        if not (float(rel_pct) == float(rel_pct)):  # NaN check
            rel_text = "nan%"
        reporter.line(f"  E{int(level)}: {float(diff):.12e} ({rel_text} of circuit)")
    if args.plot_only:
        reporter.line(f"Loaded results: {results_path}")
    else:
        reporter.line(f"Wrote results: {results_path}")
    reporter.line(f"Wrote figure: {figure_path}")
    reporter.add_runtime_line()
    reporter.persist(results_path)


if __name__ == "__main__":
    main()
