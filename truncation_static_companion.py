"""Helpers for writing static-benchmark companion artifacts from truncation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark_results_io import load_result_hdf5, save_result_hdf5
from benchmark_run_artifacts import get_git_info
from comparison.static import StaticBenchmarkResult, run_static_benchmark
from plotting.static import plot_static_benchmark, plot_static_raw_energies
from static_fitted_artifacts import (
    build_static_fitted_latex_table,
    build_static_fitted_models_artifact,
    save_static_fitted_models_artifact,
)


@dataclass(frozen=True)
class StaticCompanionPaths:
    results_path: Path
    figure_path: Path
    raw_figure_path: Path
    fitted_json_path: Path
    fitted_table_path: Path


def static_companion_paths(*, run_dir: Path, config: object) -> StaticCompanionPaths:
    static_cfg = getattr(config, "static_benchmark")
    figure_path = Path(run_dir) / Path(static_cfg.outputs.figure).name
    return StaticCompanionPaths(
        results_path=Path(run_dir) / "static_results.h5",
        figure_path=figure_path,
        raw_figure_path=figure_path.with_name(f"{figure_path.stem}_raw_energies.pdf"),
        fitted_json_path=Path(run_dir) / "static_fitted_parameters.json",
        fitted_table_path=Path(run_dir) / "static_fitted_parameters_table.tex",
    )


def materialize_static_companion_artifacts(
    *,
    run_dir: Path,
    config: object,
    repo_root: Path,
    plot_only: bool,
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
        result = run_static_benchmark(config)
        save_result_hdf5(result, paths.results_path, benchmark_name="static")

    title = (
        "Static benchmark across flux: effective vs Duffing vs circuit "
        f"(effective source={config.static_benchmark.effective_model.derivation_source})"
    )
    plot_static_benchmark(result, paths.figure_path, title)
    plot_static_raw_energies(result, paths.raw_figure_path, f"{title} [raw energies]")
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
    return paths
