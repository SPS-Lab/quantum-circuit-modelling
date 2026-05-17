"""Run paper static benchmark with parameters loaded from /params."""

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
from benchmark_run_artifacts import get_git_info, prepare_benchmark_run
from benchmark_cli_reporting import CliReporter, build_common_truncation_lines
from comparison.static import StaticBenchmarkResult, run_static_benchmark
from plotting.static import (
    plot_static_benchmark,
    plot_static_computational_basis_amplitudes,
    plot_static_raw_energies,
    plot_static_single_excitation_overlaps,
)
from static_fitted_artifacts import (
    build_static_fitted_latex_table,
    build_static_fitted_markdown_table,
    build_static_fitted_models_artifact,
    save_static_fitted_models_artifact,
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
            "directory is created; with --plot-only, the newest static run is used."
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
    run_paths = prepare_benchmark_run(
        repo_root=repo_root,
        benchmark_name="static",
        figure_paths={"figure": repo_root / config.static_benchmark.outputs.figure},
        results_path_arg=args.results,
        plot_only=bool(args.plot_only),
        experiment_name=args.experiment_name,
        output_root=args.output_root,
        argv=sys.argv,
        input_files={
            "system_params": system_params_path,
            "benchmark_params": study_params_path,
        },
    )
    figure_path = run_paths.figure_paths["figure"]
    raw_figure_path = figure_path.with_name(f"{figure_path.stem}_raw_energies.pdf")
    overlap_figure_path = figure_path.with_name(f"{figure_path.stem}_single_excitation_overlaps.pdf")
    basis_amplitude_figure_path = figure_path.with_name(f"{figure_path.stem}_computational_basis_amplitudes.pdf")
    results_path = run_paths.results_path

    if args.plot_only:
        try:
            result = load_result_hdf5(
                results_path,
                StaticBenchmarkResult,
                expected_benchmark_name="static",
            )
        except ValueError as exc:
            raise ValueError(
                f"{results_path} does not match the current static benchmark schema. "
                "Re-run without --plot-only to regenerate the results file."
            ) from exc
    else:
        result = run_static_benchmark(config)
        save_result_hdf5(result, results_path, benchmark_name="static")

    title = (
        "Static benchmark across flux: effective vs Duffing vs circuit "
        f"(effective source={config.static_benchmark.effective_model.derivation_source})"
    )
    plot_static_benchmark(result, figure_path, title)
    plot_static_raw_energies(result, raw_figure_path, f"{title} [raw energies]")
    plot_static_single_excitation_overlaps(result, overlap_figure_path, f"{title} [single-excitation overlaps]")
    plot_static_computational_basis_amplitudes(
        result,
        basis_amplitude_figure_path,
        f"{title} [computational basis amplitudes]",
    )
    fitted_artifact = build_static_fitted_models_artifact(result, config=config)
    fitted_json_path = run_paths.run_dir / "static_fitted_parameters.json"
    fitted_table_path = run_paths.run_dir / "static_fitted_parameters_table.tex"
    fitted_markdown_path = run_paths.run_dir / "static_fitted_parameters_table.md"
    git_info = get_git_info(repo_root)
    save_static_fitted_models_artifact(fitted_artifact, fitted_json_path)
    fitted_table_path.write_text(
        build_static_fitted_latex_table(
            fitted_artifact,
            git_info=git_info,
            experiment_folder_name=run_paths.run_dir.name,
        ),
        encoding="utf-8",
    )
    fitted_markdown_path.write_text(
        build_static_fitted_markdown_table(
            fitted_artifact,
            git_info=git_info,
            experiment_folder_name=run_paths.run_dir.name,
        ),
        encoding="utf-8",
    )

    for line in build_common_truncation_lines(config):
        reporter.line(line)
    reporter.line("Static benchmark summary (GHz):")
    for key, value in result.summary.items():
        reporter.line(f"  {key}: {value:.6e}")
    reporter.line("Static benchmark metric notes:")
    for key in (
        "effective_error_rmse",
        "duffing_error_rmse",
        "effective_computational_energy_rmse",
        "duffing_computational_energy_rmse",
        "duffing_truncation_style_energy_rmse",
        "effective_mean_abs_dJ",
        "effective_mean_abs_dzeta",
        "duffing_mean_abs_dJ",
        "duffing_mean_abs_dzeta",
    ):
        if key in result.metric_notes:
            reporter.line(f"  {key}: {result.metric_notes[key]}")
    reporter.line("Effective-model fit coefficients (GHz):")
    for name in ("w0", "w1", "J", "zeta"):
        reporter.line(
            _format_fit_line(
                name,
                result.effective_fit_coefficient_names[name],
                result.effective_fit_coefficients[name],
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
    reporter.line(f"Wrote raw-energy figure: {raw_figure_path}")
    reporter.line(f"Wrote single-excitation overlap figure: {overlap_figure_path}")
    reporter.line(f"Wrote computational-basis amplitude figure: {basis_amplitude_figure_path}")
    reporter.line(f"Wrote fitted-parameter artifact: {fitted_json_path}")
    reporter.line(f"Wrote LaTeX table: {fitted_table_path}")
    reporter.line(f"Wrote Markdown table: {fitted_markdown_path}")
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
