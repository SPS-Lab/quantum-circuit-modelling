"""Static regime-map comparison entrypoint.

This delegates to the refactored static benchmark + plotting stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comparison.static import run_static_benchmark
from plotting.static import plot_static_benchmark
from study_config import load_study_config



def _as_legacy_result(
    *,
    selection_mode: str,
    result,
) -> dict[str, Any]:
    """Map static-benchmark output to legacy regime-map keys."""
    return {
        "flux": result.flux_values,
        "E1_rel": result.effective_relative_energies,
        "E2_rel": result.duffing_relative_energies,
        "E3_rel": result.circuit_relative_energies,
        "err_model1": result.effective_error_rmse,
        "err_model2": result.duffing_error_rmse,
        "detuning_ratio": result.detuning_ratio,
        "idle_mask": result.idle_mask,
        "near_mask": result.near_mask,
        "params_model1": result.effective_parameters,
        "params_model2": result.duffing_parameters,
        "params_scqubits": result.circuit_parameters,
        "selection_mode": selection_mode,
        "summary": {
            "model1_idle_rmse": result.summary["effective_idle_rmse"],
            "model1_idle_max_abs": result.summary["effective_idle_max_abs"],
            "model1_near_rmse": result.summary["effective_near_rmse"],
            "model1_near_max_abs": result.summary["effective_near_max_abs"],
            "model2_idle_rmse": result.summary["duffing_idle_rmse"],
            "model2_idle_max_abs": result.summary["duffing_idle_max_abs"],
            "model2_near_rmse": result.summary["duffing_near_rmse"],
            "model2_near_max_abs": result.summary["duffing_near_max_abs"],
        },
    }



def compare_model1_model2_against_scqubits(
    *,
    system_params_path: str | Path,
    study_params_path: str | Path,
    title: str | None = None,
    outfile: str | Path | None = None,
) -> dict[str, Any]:
    """Run static comparison from explicit params files."""
    system_path = Path(system_params_path).resolve()
    study_path = Path(study_params_path).resolve()

    config = load_study_config(system_path, study_path)
    result = run_static_benchmark(config)

    repo_root = system_path.parents[1]
    configured_figure = Path(config.static_benchmark.outputs.figure)
    if configured_figure.is_absolute():
        figure_path = configured_figure
    else:
        figure_path = repo_root / configured_figure

    if outfile is not None:
        figure_path = Path(outfile).resolve()

    plot_title = (
        title
        or "Static benchmark across flux: effective vs Duffing vs circuit "
        f"(effective source={config.static_benchmark.effective_model.derivation_source})"
    )
    plot_static_benchmark(result, figure_path, plot_title)

    return _as_legacy_result(
        selection_mode=config.static_benchmark.dressed_subspace.selection_mode,
        result=result,
    )
