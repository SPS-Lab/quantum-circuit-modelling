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
from benchmark_cli_reporting import CliReporter, build_common_truncation_lines
from comparison.static import StaticBenchmarkResult, run_static_benchmark
from plotting.static import plot_static_benchmark
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


def _format_fit_line(name: str, coeff_names: object, coeffs: object) -> str:
    if not hasattr(coeff_names, "__len__") or not hasattr(coeffs, "__len__"):
        raise ValueError(f"Expected sequence-like coefficient names and values for {name}")
    labels = [str(label) for label in coeff_names]
    values = [float(value) for value in coeffs]
    if len(labels) != len(values):
        raise ValueError(f"Coefficient name/value mismatch for {name}: {labels!r} vs {values!r}")
    parts = ", ".join(f"{label}={value:.6e}" for label, value in zip(labels, values))
    return f"  {name}: {parts}"


def main() -> None:
    args = _parse_args()
    repo_root = _REPO_ROOT
    reporter = CliReporter(benchmark_name="static", script_name=Path(__file__).name)
    system_params_path = repo_root / "params" / "system_params.json"
    study_params_path = repo_root / "params" / "benchmark_params.json"

    config = load_study_config(
        system_params_path=system_params_path,
        study_params_path=study_params_path
    )
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

    for line in build_common_truncation_lines(config):
        reporter.line(line)
    reporter.line("Static benchmark summary (GHz):")
    for key, value in result.summary.items():
        reporter.line(f"  {key}: {value:.6e}")
    reporter.line("Effective-model fit coefficients (GHz):")
    reporter.line(
        _format_fit_line(
            "J",
            result.effective_fit_coefficient_names["J"],
            result.effective_fit_coefficients["J"],
        )
    )
    reporter.line(
        _format_fit_line(
            "zeta",
            result.effective_fit_coefficient_names["zeta"],
            result.effective_fit_coefficients["zeta"],
        )
    )
    if result.duffing_symbolic_coefficients:
        reporter.line("Duffing symbolic calibration coefficients (GHz):")
        ordered_names = ("w0", "w1", "alpha0", "alpha1", "g0c", "g1c")
        for name in ordered_names:
            if name in result.duffing_symbolic_coefficients and name in result.duffing_symbolic_coefficient_names:
                reporter.line(
                    _format_fit_line(
                        name,
                        result.duffing_symbolic_coefficient_names[name],
                        result.duffing_symbolic_coefficients[name],
                    )
                )
    if args.plot_only:
        reporter.line(f"Loaded results: {results_path}")
    else:
        reporter.line(f"Wrote results: {results_path}")
    reporter.line(f"Wrote figure: {figure_path}")
    reporter.add_runtime_line()
    reporter.persist(results_path)


if __name__ == "__main__":
    main()
