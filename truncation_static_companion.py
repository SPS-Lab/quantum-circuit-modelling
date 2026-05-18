"""Helpers for writing static-benchmark companion artifacts from truncation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark_results_io import load_result_hdf5, save_result_hdf5
from benchmark_run_artifacts import get_git_info
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


def _has_extra_sideplot_data(result: StaticBenchmarkResult) -> bool:
    return (
        result.circuit_full_raw_energies.size > 0
        and result.duffing_full_raw_energies.size > 0
        and result.circuit_computational_bare_overlaps.size > 0
        and result.duffing_computational_bare_overlaps.size > 0
        and result.circuit_tracked_branch_bare_amplitudes.size > 0
        and result.duffing_tracked_branch_bare_amplitudes.size > 0
    )


@dataclass(frozen=True)
class StaticCompanionPaths:
    results_path: Path
    figure_path: Path
    raw_figure_path: Path
    overlap_figure_path: Path
    basis_amplitude_figure_path: Path
    fitted_json_path: Path
    fitted_table_path: Path
    fitted_markdown_path: Path


def static_companion_paths(*, run_dir: Path, config: object) -> StaticCompanionPaths:
    static_cfg = getattr(config, "static_benchmark")
    figure_path = Path(run_dir) / Path(static_cfg.outputs.figure).name
    return StaticCompanionPaths(
        results_path=Path(run_dir) / "static_results.h5",
        figure_path=figure_path,
        raw_figure_path=figure_path.with_name(f"{figure_path.stem}_raw_energies.pdf"),
        overlap_figure_path=figure_path.with_name(f"{figure_path.stem}_single_excitation_overlaps.pdf"),
        basis_amplitude_figure_path=figure_path.with_name(f"{figure_path.stem}_computational_basis_amplitudes.pdf"),
        fitted_json_path=Path(run_dir) / "static_fitted_parameters.json",
        fitted_table_path=Path(run_dir) / "static_fitted_parameters_table.tex",
        fitted_markdown_path=Path(run_dir) / "static_fitted_parameters_table.md",
    )


def materialize_static_companion_artifacts(
    *,
    run_dir: Path,
    config: object,
    repo_root: Path,
    plot_only: bool,
    include_extra_sideplots: bool = False,
    include_truncation_style_metric: bool = False,
) -> StaticCompanionPaths | None:
    paths = static_companion_paths(run_dir=run_dir, config=config)

    if plot_only:
        if not paths.results_path.exists():
            return None
        try:
            result = load_result_hdf5(
                paths.results_path,
                StaticBenchmarkResult,
                expected_benchmark_name="static",
            )
        except ValueError as exc:
            raise ValueError(
                f"{paths.results_path} does not match the current static benchmark schema. "
                "Re-run the truncation benchmark without --plot-only to regenerate the companion static results."
            ) from exc
    else:
        result = run_static_benchmark(
            config,
            include_extra_sideplot_data=bool(include_extra_sideplots),
            include_full_spectrum_plot_data=bool(include_extra_sideplots),
            include_truncation_style_metric=bool(include_truncation_style_metric),
        )
        save_result_hdf5(result, paths.results_path, benchmark_name="static")

    title = (
        "Static benchmark across flux: effective vs Duffing vs circuit "
        f"(effective source={config.static_benchmark.effective_model.derivation_source})"
    )
    plot_static_benchmark(result, paths.figure_path, title)
    if include_extra_sideplots:
        if not _has_extra_sideplot_data(result):
            raise ValueError(
                "Extra side-plot data is not present in this companion static results file. "
                "Re-run the truncation benchmark without --plot-only and with --extra-sideplots to generate it."
            )
        plot_static_raw_energies(result, paths.raw_figure_path, f"{title} [raw energies]")
        plot_static_single_excitation_overlaps(
            result, paths.overlap_figure_path, f"{title} [single-excitation overlaps]"
        )
        plot_static_computational_basis_amplitudes(
            result,
            paths.basis_amplitude_figure_path,
            f"{title} [computational basis amplitudes]",
        )
    fitted_artifact = build_static_fitted_models_artifact(result, config=config)
    save_static_fitted_models_artifact(fitted_artifact, paths.fitted_json_path)
    paths.fitted_table_path.write_text(
        build_static_fitted_latex_table(
            fitted_artifact,
            git_info=get_git_info(repo_root),
            experiment_folder_name=Path(run_dir).name,
        ),
        encoding="utf-8",
    )
    paths.fitted_markdown_path.write_text(
        build_static_fitted_markdown_table(
            fitted_artifact,
            git_info=get_git_info(repo_root),
            experiment_folder_name=Path(run_dir).name,
        ),
        encoding="utf-8",
    )
    return paths
