from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark_results_io import load_result_hdf5, save_result_hdf5
from comparison.fitted_reconstruction import (
    duffing_mode_parameters_for_flux,
    effective_parameters_for_flux,
)
from comparison.cz import run_cz_benchmark
from comparison.leakage_flow import run_leakage_flow_benchmark
from comparison.rx import run_rx_benchmark
from comparison.static import StaticBenchmarkResult, run_static_benchmark
import comparison.static as static_module
import models.duffing as duffing_module
import models.duffing_calibration as duffing_calibration_module
from models import (
    build_circuit_model_stack,
    build_duffing_model_stack_from_scratch,
    evaluate_symbolic_duffing_mode_parameters,
)
from models.dressed import extract_effective_model_parameters_from_4x4_stack
from plotting.cz import plot_cz_benchmark
from plotting.leakage_flow import plot_leakage_flow_benchmark
from plotting.rx import plot_rx_diagnostics_benchmark, plot_rx_populations_benchmark
from benchmark_run_artifacts import get_git_info
from static_fitted_artifacts import (
    build_static_fitted_latex_table,
    build_static_fitted_markdown_table,
    build_static_fitted_models_artifact,
    load_static_fitted_models_artifact,
    save_static_fitted_models_artifact,
)
from study_config import _flatten_run_all_benchmark_params, load_study_config
from truncation_static_companion import materialize_static_companion_artifacts
from models.effective import (
    evaluate_effective_parameter_fit,
    fit_magnitude_exchange_parameters,
    fit_single_harmonic_parameters,
)



def _add_required_study_sections(payload: dict[str, object]) -> None:
    payload.setdefault(
        "leakage_benchmark",
        {
            "total_time_ns": 8.0,
            "ramp_time_ns": 2.0,
            "dt_ns": 0.02,
            "top_destination_rows": 5,
        },
    )
    payload.setdefault(
        "state_to_state_leakage_benchmark",
        {
            "total_time_ns": 8.0,
            "ramp_time_ns": 2.0,
            "dt_ns": 0.02,
            "top_transition_rows": 6,
        },
    )


def _write_small_system_params(tmp_path: Path) -> Path:
    src = _ROOT / "params" / "system_params.json"
    payload = json.loads(src.read_text(encoding="utf-8"))

    dst = tmp_path / "system_params_small.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst



def _write_small_study_params(
    tmp_path: Path,
    *,
    sweep_target: str = "q0",
    duffing_calibration_mode: str = "analytic-per-flux",
) -> Path:
    src = _ROOT / "params" / "benchmark_params.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload = _flatten_run_all_benchmark_params(payload)
    _add_required_study_sections(payload)
    sb = payload["static_benchmark"]
    sb["flux_sweep"]["num_points"] = 9
    sb["dressed_subspace"]["n_candidate_states"] = 12
    sb["duffing_model"]["hilbert_truncation"]["nlevels_qubit"] = 3
    sb["duffing_model"]["hilbert_truncation"]["nlevels_coupler"] = 3
    sb["circuit_model"]["transmon_charge_basis"]["q0_ncut"] = 25
    sb["circuit_model"]["transmon_charge_basis"]["q1_ncut"] = 25
    sb["circuit_model"]["hilbert_truncation"]["q0_truncated_dim"] = 4
    sb["circuit_model"]["hilbert_truncation"]["q1_truncated_dim"] = 4
    sb["circuit_model"]["hilbert_truncation"]["c_truncated_dim"] = 4
    sb["flux_control"]["sweep_target"] = str(sweep_target)
    sb["duffing_model"]["calibration_mode"] = str(duffing_calibration_mode)
    rb = payload["rx_benchmark"]
    rb["drive_frequency"] = 9.733
    rb["drive_amplitude"] = 0.05
    rb["drive_phase_rad"] = 0.0
    rb["total_time_ns"] = 6.0
    rb["dt_ns"] = 0.1
    rb["rise_time_ns"] = 1.0

    dst = tmp_path / f"study_params_small_{sweep_target}_{duffing_calibration_mode}.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst



def test_load_study_config(tmp_path: Path) -> None:
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path),
    )
    assert cfg.system.q0.EJmax > 0.0
    assert cfg.static_benchmark.flux_sweep.num_points > 2
    assert cfg.static_benchmark.effective_model.derivation_source in {"duffing", "circuit"}
    assert cfg.static_benchmark.effective_model.fit_basis in {"single-harmonic", "magnitude-exchange-like"}
    assert cfg.static_benchmark.flux_control.sweep_target in {"q0", "q1"}
    assert cfg.static_benchmark.duffing_model.calibration_mode in {
        "fixed",
        "analytic-per-flux",
        "per-flux",
        "fitted-static",
        "symbolic-fitted-static",
    }
    assert cfg.static_benchmark.duffing_model.symbolic_fit is not None
    assert cfg.static_benchmark.duffing_model.symbolic_fit.max_harmonics_w >= 1
    assert cfg.static_benchmark.duffing_model.symbolic_fit.max_harmonics_alpha >= 1
    assert cfg.static_benchmark.duffing_model.symbolic_fit.max_harmonics_g >= 1
    assert cfg.static_benchmark.duffing_model.symbolic_fit.pointwise_max_nfev >= 1
    assert cfg.static_benchmark.duffing_model.symbolic_fit.refinement_max_nfev >= 1
    assert cfg.static_benchmark.duffing_model.symbolic_fit.regularization_weight >= 0.0
    assert cfg.static_benchmark.circuit_model.transmon_charge_basis.q0_ncut > 0
    assert cfg.static_benchmark.circuit_model.transmon_charge_basis.q1_ncut > 0
    assert len(cfg.circuit_truncation_benchmark.circuit_ncut_values) > 0
    assert len(cfg.circuit_truncation_benchmark.circuit_qubit_truncated_dim_values) > 0
    assert len(cfg.circuit_truncation_benchmark.circuit_coupler_truncated_dim_values) > 0
    assert cfg.circuit_truncation_benchmark.lowest_excited_levels_to_plot >= 1
    assert cfg.circuit_truncation_benchmark.circuit_reference_ncut > 0
    assert cfg.circuit_truncation_benchmark.circuit_reference_qubit_truncated_dim > 0
    assert cfg.circuit_truncation_benchmark.circuit_reference_coupler_truncated_dim > 0
    assert len(cfg.duffing_truncation_benchmark.duffing_ncut_values) > 0
    assert cfg.duffing_truncation_benchmark.duffing_truncated_dim >= 3
    assert len(cfg.duffing_truncation_benchmark.duffing_hilbert_qubit_dim_values) > 0
    assert len(cfg.duffing_truncation_benchmark.duffing_hilbert_coupler_dim_values) > 0
    assert cfg.duffing_truncation_benchmark.duffing_reference_hilbert_qubit_dim >= 2
    assert cfg.duffing_truncation_benchmark.duffing_reference_hilbert_coupler_dim >= 2
    assert cfg.duffing_truncation_benchmark.lowest_excited_levels_to_plot >= 1
    assert cfg.duffing_truncation_benchmark.circuit_reference_ncut > 0
    assert cfg.duffing_truncation_benchmark.circuit_reference_qubit_truncated_dim > 0
    assert cfg.duffing_truncation_benchmark.circuit_reference_coupler_truncated_dim > 0
    assert cfg.duffing_truncation_benchmark.duffing_calibration_mode in {
        "fixed",
        "analytic-per-flux",
        "per-flux",
        "fitted-static",
        "symbolic-fitted-static",
    }
    assert len(cfg.runtime_benchmark.qubit_truncation_values) > 0
    assert all(v >= 2 for v in cfg.runtime_benchmark.qubit_truncation_values)
    assert cfg.runtime_benchmark.duffing_calibration_mode in {
        "fixed",
        "analytic-per-flux",
        "per-flux",
        "fitted-static",
        "symbolic-fitted-static",
    }
    assert cfg.runtime_benchmark.repeats >= 1
    assert cfg.runtime_benchmark.hold_time_ns is None or cfg.runtime_benchmark.hold_time_ns >= 0.0
    assert cfg.cz_benchmark.total_time_ns is None or cfg.cz_benchmark.total_time_ns > 0.0
    assert cfg.cz_benchmark.hold_time_ns is None or cfg.cz_benchmark.hold_time_ns >= 0.0
    assert cfg.cz_benchmark.ramp_time_ns > 0.0
    assert cfg.cz_benchmark.dt_ns > 0.0
    assert cfg.cz_benchmark.scan_dt_ns > 0.0
    assert cfg.cz_benchmark.scan_max_hold_ns >= 0.0
    assert cfg.cz_benchmark.scan_leakage_penalty >= 0.0
    assert cfg.rx_benchmark.drive_qubit == "q0"
    assert cfg.rx_benchmark.drive_frequency > 0.0
    assert cfg.rx_benchmark.drive_amplitude >= 0.0
    assert cfg.rx_benchmark.total_time_ns > 0.0
    assert cfg.rx_benchmark.dt_ns > 0.0
    assert cfg.rx_benchmark.rise_time_ns > 0.0
    assert cfg.leakage_flow_benchmark.total_time_ns > 0.0
    assert cfg.leakage_flow_benchmark.ramp_time_ns > 0.0
    assert cfg.leakage_flow_benchmark.dt_ns > 0.0
    assert cfg.leakage_flow_benchmark.population_min_average >= 0.0
    assert cfg.leakage_flow_benchmark.transition_min_integrated_abs >= 0.0
    assert cfg.leakage_flow_benchmark.max_population_rows >= 1
    assert cfg.leakage_flow_benchmark.max_transition_rows >= 1



def test_effective_fit_is_compact_global_model() -> None:
    flux = np.linspace(0.0, 1.0, 41)
    non_harmonic = np.cos(2.0 * np.pi * flux) + 0.2 * np.cos(4.0 * np.pi * flux)
    extracted = {
        "w0": non_harmonic,
        "w1": non_harmonic + 0.1,
        "J": 0.05 * non_harmonic,
        "zeta": 0.01 * non_harmonic,
    }
    fit = fit_single_harmonic_parameters(flux, extracted_parameters=extracted)
    mismatch = np.max(np.abs(fit.fitted_parameters["w0"] - extracted["w0"]))
    assert mismatch > 1e-3


def test_extract_effective_model_parameters_gauge_fixes_exchange_phase() -> None:
    H = np.zeros((3, 4, 4), dtype=complex)
    for k, j in enumerate([0.05, -0.06, 0.04]):
        H[k, 1, 1] = 1.0
        H[k, 2, 2] = 1.2
        H[k, 1, 2] = 2.0 * j
        H[k, 2, 1] = 2.0 * j

    params_raw = extract_effective_model_parameters_from_4x4_stack(H, gauge_fix_exchange=False)
    params_fixed = extract_effective_model_parameters_from_4x4_stack(H, gauge_fix_exchange=True)

    assert np.any(params_raw["J"] < 0.0)
    assert np.all(params_fixed["J"] >= 0.0)
    assert np.allclose(params_fixed["w0"], params_raw["w0"])
    assert np.allclose(params_fixed["w1"], params_raw["w1"])
    assert np.allclose(params_fixed["zeta"], params_raw["zeta"])



def test_static_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    out = run_static_benchmark(cfg)

    assert out.effective_relative_energies.shape == (9, 4)
    assert out.duffing_relative_energies.shape == (9, 4)
    assert out.circuit_relative_energies.shape == (9, 4)
    assert out.circuit_tracked_branch_bare_amplitudes.ndim == 3
    assert out.duffing_tracked_branch_bare_amplitudes.ndim == 3
    assert out.circuit_tracked_branch_bare_amplitudes.shape[0] == 9
    assert out.duffing_tracked_branch_bare_amplitudes.shape[0] == 9
    assert out.circuit_tracked_branch_bare_amplitudes.shape[2] == 4
    assert out.duffing_tracked_branch_bare_amplitudes.shape[2] == 4
    assert out.circuit_bare_state_labels.shape[0] == out.circuit_tracked_branch_bare_amplitudes.shape[1]
    assert out.duffing_bare_state_labels.shape[0] == out.duffing_tracked_branch_bare_amplitudes.shape[1]
    assert out.circuit_computational_bare_amplitudes.shape == (9, 4, 4)
    assert out.duffing_computational_bare_amplitudes.shape == (9, 4, 4)
    assert np.iscomplexobj(out.circuit_computational_bare_amplitudes)
    assert np.iscomplexobj(out.duffing_computational_bare_amplitudes)
    assert np.isfinite(float(np.mean(out.effective_error_rmse)))
    assert np.isfinite(float(np.mean(out.duffing_error_rmse)))
    assert float(out.summary["computational_excited_levels_compared"]) == 3.0
    assert np.isfinite(float(out.summary["effective_computational_energy_rmse"]))
    assert np.isfinite(float(out.summary["duffing_computational_energy_rmse"]))
    assert np.isfinite(float(out.summary["duffing_truncation_style_energy_rmse"]))
    assert np.isfinite(float(out.summary["effective_mean_abs_dJ"]))
    assert np.isfinite(float(out.summary["effective_mean_abs_dzeta"]))
    assert np.isfinite(float(out.summary["duffing_mean_abs_dJ"]))
    assert np.isfinite(float(out.summary["duffing_mean_abs_dzeta"]))
    assert "duffing_truncation_style_energy_rmse" in out.metric_notes
    assert set(out.effective_fit_coefficient_names) == {"w0", "w1", "J", "zeta"}
    assert set(out.effective_fit_coefficients) == {"w0", "w1", "J", "zeta"}
    assert set(out.duffing_mode_parameters) == {"w0", "w1", "alpha0", "alpha1", "wc", "g0c", "g1c"}
    assert len(out.effective_fit_coefficient_names["w0"]) == out.effective_fit_coefficients["w0"].shape[0]
    assert len(out.effective_fit_coefficient_names["w1"]) == out.effective_fit_coefficients["w1"].shape[0]
    assert len(out.effective_fit_coefficient_names["J"]) == out.effective_fit_coefficients["J"].shape[0]
    assert len(out.effective_fit_coefficient_names["zeta"]) == out.effective_fit_coefficients["zeta"].shape[0]
    assert out.effective_fit_coefficients["w0"].shape[0] >= 3
    assert out.effective_fit_coefficients["w1"].shape[0] >= 3
    assert out.effective_fit_coefficients["J"].shape[0] >= 3
    assert out.effective_fit_coefficients["zeta"].shape[0] >= 3


def test_static_benchmark_fit_coefficients_roundtrip_through_hdf5(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    out = run_static_benchmark(cfg)
    results_path = tmp_path / "static_results.h5"
    save_result_hdf5(out, results_path, benchmark_name="static")
    loaded = load_result_hdf5(results_path, StaticBenchmarkResult, expected_benchmark_name="static")

    for key in ("w0", "w1", "J", "zeta"):
        assert np.array_equal(loaded.effective_fit_coefficient_names[key], out.effective_fit_coefficient_names[key])
        assert np.allclose(loaded.effective_fit_coefficients[key], out.effective_fit_coefficients[key])
    assert loaded.duffing_symbolic_coefficient_names.keys() == out.duffing_symbolic_coefficient_names.keys()
    assert loaded.duffing_symbolic_coefficients.keys() == out.duffing_symbolic_coefficients.keys()
    for key in loaded.duffing_symbolic_coefficient_names:
        assert np.array_equal(loaded.duffing_symbolic_coefficient_names[key], out.duffing_symbolic_coefficient_names[key])
    for key in loaded.duffing_symbolic_coefficients:
        assert np.allclose(loaded.duffing_symbolic_coefficients[key], out.duffing_symbolic_coefficients[key])
    assert loaded.metric_notes == out.metric_notes
    assert loaded.summary.keys() == out.summary.keys()


def test_static_duffing_truncation_style_rmse_uses_sorted_full_spectra(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path, duffing_calibration_mode="analytic-per-flux")
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    out = run_static_benchmark(cfg)
    circuit = build_circuit_model_stack(
        flux_values=out.flux_values,
        system_params=cfg.system,
        circuit_config=cfg.static_benchmark.circuit_model,
        sweep_target=cfg.static_benchmark.flux_control.sweep_target,
    )
    duffing = build_duffing_model_stack_from_scratch(
        flux_values=out.flux_values,
        system_params=cfg.system,
        duffing_config=cfg.static_benchmark.duffing_model,
        sweep_target=cfg.static_benchmark.flux_control.sweep_target,
    )
    n_full_track = min(10, duffing.hamiltonian_stack.shape[1], circuit.hamiltonian_stack.shape[1])
    circuit_sorted = static_module._sorted_relative_energies(
        circuit.hamiltonian_stack,
        n_track=n_full_track,
    )
    duffing_sorted = static_module._sorted_relative_energies(
        duffing.hamiltonian_stack,
        n_track=n_full_track,
    )
    n_excited = min(
        int(cfg.duffing_truncation_benchmark.lowest_excited_levels_to_plot),
        int(circuit_sorted.shape[1]) - 1,
        int(duffing_sorted.shape[1]) - 1,
    )
    expected_rmse = static_module._aggregate_rmse(
        duffing_sorted,
        circuit_sorted,
        n_excited=n_excited,
    )

    assert out.summary["duffing_truncation_style_excited_levels_compared"] == float(n_excited)
    assert out.summary["duffing_truncation_style_energy_rmse"] == pytest.approx(expected_rmse)


def test_effective_fit_reconstructs_static_grid(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    out = run_static_benchmark(cfg)
    reconstructed = effective_parameters_for_flux(out, cfg, out.flux_values)

    for key in ("w0", "w1", "J", "zeta"):
        assert np.allclose(reconstructed[key], out.effective_parameters[key])


def test_magnitude_exchange_fit_reconstructs_asymmetric_exchange_targets(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)
    flux_values = np.linspace(-0.2, 0.2, 17)
    theta = 2.0 * np.pi * flux_values
    wc = np.full_like(flux_values, float(cfg.system.c.E_osc), dtype=float)
    w0 = 4.95 + 0.22 * np.cos(theta) - 0.06 * np.cos(2.0 * theta) + 0.01 * np.cos(3.0 * theta)
    w1 = 5.35 - 0.18 * np.cos(theta) + 0.04 * np.cos(2.0 * theta) - 0.015 * np.cos(3.0 * theta)

    gamma = float(np.geomspace(1e-3, 5.0, 600)[240])
    delta1 = w0 - wc
    delta2 = w1 - wc
    r1 = 1.0 / np.sqrt(delta1 * delta1 + gamma * gamma)
    r2 = 1.0 / np.sqrt(delta2 * delta2 + gamma * gamma)
    j_target = 0.018 + 0.009 * r1 - 0.013 * r2 + 0.022 * r1 * r2 + 0.006 * r1 * r1 - 0.004 * r2 * r2
    zeta_target = -0.001 + 0.002 * r1 + 0.003 * r2 - 0.0025 * r1 * r2 + 0.0015 * r1 * r1 + 0.0008 * r2 * r2

    fit = fit_magnitude_exchange_parameters(
        flux_values,
        extracted_parameters={
            "w0": w0,
            "w1": w1,
            "J": j_target,
            "zeta": zeta_target,
        },
        coupler_frequency_values=wc,
    )
    reconstructed = evaluate_effective_parameter_fit(
        flux_values,
        system_params=cfg.system,
        fit_basis="magnitude-exchange-like",
        coefficient_names=fit.coefficient_names,
        coefficients=fit.coefficients,
    )

    for key, expected in (("w0", w0), ("w1", w1), ("J", j_target), ("zeta", zeta_target)):
        assert np.allclose(reconstructed[key], expected, atol=1e-10, rtol=1e-10)


def test_symbolic_duffing_fit_reconstructs_static_grid(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        duffing_calibration_mode="symbolic-fitted-static",
    )
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    out = run_static_benchmark(cfg)
    reconstructed = duffing_mode_parameters_for_flux(out, cfg, out.flux_values)

    for key in ("w0", "w1", "alpha0", "alpha1", "wc", "g0c", "g1c"):
        assert np.allclose(reconstructed[key], out.duffing_mode_parameters[key])


def test_static_fitted_models_artifact_roundtrip_and_latex(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        duffing_calibration_mode="symbolic-fitted-static",
    )
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    out = run_static_benchmark(cfg)
    artifact = build_static_fitted_models_artifact(out, config=cfg)

    json_path = tmp_path / "static_fitted_parameters.json"
    save_static_fitted_models_artifact(artifact, json_path)
    loaded_json = load_static_fitted_models_artifact(json_path)

    assert np.allclose(loaded_json.flux_values, artifact.flux_values)
    assert set(loaded_json.effective_parameters) == {"w0", "w1", "J", "zeta"}
    assert set(loaded_json.duffing_mode_parameters) == {"w0", "w1", "alpha0", "alpha1", "wc", "g0c", "g1c"}
    assert loaded_json.sweep_target == str(cfg.static_benchmark.flux_control.sweep_target)
    assert loaded_json.duffing_calibration_mode == "symbolic-fitted-static"

    results_path = tmp_path / "static_results.h5"
    save_result_hdf5(out, results_path, benchmark_name="static")
    loaded_h5 = load_static_fitted_models_artifact(results_path)
    assert np.allclose(loaded_h5.flux_values, artifact.flux_values)
    assert np.allclose(loaded_h5.circuit_parameters["zeta"], artifact.circuit_parameters["zeta"])

    latex = build_static_fitted_latex_table(
        loaded_json,
        git_info=get_git_info(_ROOT),
        experiment_folder_name="20260513_123550_static_2b0daa0",
    )
    assert "Effective parameter & Coefficient & Value (GHz)" in latex
    assert "Duffing parameter & Coefficient & Value (GHz)" in latex
    assert r"\begin{tabular}{lll}" in latex
    assert latex.count("% Git provenance: commit=") == 2
    assert latex.count("% Experiment folder: 20260513_123550_static_2b0daa0") == 2

    markdown = build_static_fitted_markdown_table(
        loaded_json,
        git_info=get_git_info(_ROOT),
        experiment_folder_name="20260513_123550_static_2b0daa0",
    )
    assert "## Effective fitted coefficients" in markdown
    assert "| Effective parameter | Coefficient | Value (GHz) |" in markdown
    assert "## Symbolic Duffing fitted coefficients" in markdown
    assert markdown.count("<!-- Git provenance: commit=") == 2
    assert markdown.count("<!-- Experiment folder: 20260513_123550_static_2b0daa0 -->") == 2


def test_truncation_static_companion_materializes_artifacts(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    run_dir = tmp_path / "truncation_run"
    paths = materialize_static_companion_artifacts(
        run_dir=run_dir,
        config=cfg,
        repo_root=_ROOT,
        plot_only=False,
    )
    assert paths is not None
    assert paths.results_path.exists()
    assert paths.figure_path.exists()
    assert paths.raw_figure_path.exists()
    assert paths.overlap_figure_path.exists()
    assert paths.basis_amplitude_figure_path.exists()
    assert paths.fitted_json_path.exists()
    assert paths.fitted_table_path.exists()
    assert paths.fitted_markdown_path.exists()

    loaded_paths = materialize_static_companion_artifacts(
        run_dir=run_dir,
        config=cfg,
        repo_root=_ROOT,
        plot_only=True,
    )
    assert loaded_paths == paths


def test_static_benchmark_uses_fixed_coupler_frequency_from_system_params(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path, sweep_target="q1")

    baseline = run_static_benchmark(load_study_config(system_params_path=system_path, study_params_path=study_path))

    payload = json.loads(system_path.read_text(encoding="utf-8"))
    payload["parameters"]["c"]["E_osc"] = 7.321
    system_path.write_text(json.dumps(payload), encoding="utf-8")
    shifted = run_static_benchmark(load_study_config(system_params_path=system_path, study_params_path=study_path))

    # Changing the fixed bus frequency in system params must change the benchmark outputs.
    assert not np.allclose(baseline.circuit_parameters["w0"], shifted.circuit_parameters["w0"])


def test_static_benchmark_q0_sweep_varies_spectrum_with_fixed_coupler(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path, sweep_target="q0")

    out = run_static_benchmark(load_study_config(system_params_path=system_path, study_params_path=study_path))

    assert float(np.ptp(out.circuit_relative_energies[:, 1])) > 1e-6


def test_duffing_fixed_calibration_is_not_recomputed_per_flux(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        sweep_target="q0",
        duffing_calibration_mode="fixed",
    )
    out = run_static_benchmark(load_study_config(system_params_path=system_path, study_params_path=study_path))
    assert float(np.ptp(out.duffing_relative_energies[:, 1])) < 1e-10


def test_duffing_per_flux_calibration_can_be_enabled_explicitly(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        sweep_target="q0",
        duffing_calibration_mode="per-flux",
    )
    out = run_static_benchmark(load_study_config(system_params_path=system_path, study_params_path=study_path))
    assert float(np.ptp(out.duffing_relative_energies[:, 1])) > 1e-6


def test_duffing_analytic_per_flux_calibration_varies_with_flux(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        sweep_target="q0",
        duffing_calibration_mode="analytic-per-flux",
    )
    out = run_static_benchmark(load_study_config(system_params_path=system_path, study_params_path=study_path))
    assert float(np.ptp(out.duffing_relative_energies[:, 1])) > 1e-6


def test_duffing_fitted_static_calibration_runs_and_exposes_mode_parameters(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        sweep_target="q0",
        duffing_calibration_mode="fitted-static",
    )
    out = run_static_benchmark(load_study_config(system_params_path=system_path, study_params_path=study_path))
    assert set(out.duffing_mode_parameters) == {"w0", "w1", "alpha0", "alpha1", "wc", "g0c", "g1c"}
    assert all(np.all(np.isfinite(values)) for values in out.duffing_mode_parameters.values())
    assert out.duffing_mode_parameters["w0"].shape == out.flux_values.shape
    assert out.duffing_symbolic_coefficient_names == {}
    assert out.duffing_symbolic_coefficients == {}


def test_duffing_symbolic_fitted_static_runs_and_exposes_symbolic_coefficients(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        sweep_target="q0",
        duffing_calibration_mode="symbolic-fitted-static",
    )
    out = run_static_benchmark(load_study_config(system_params_path=system_path, study_params_path=study_path))
    assert set(out.duffing_mode_parameters) == {"w0", "w1", "alpha0", "alpha1", "wc", "g0c", "g1c"}
    assert set(out.duffing_symbolic_coefficient_names) == {"w0", "w1", "alpha0", "alpha1", "wc", "g0c", "g1c"}
    assert set(out.duffing_symbolic_coefficients) == {"w0", "w1", "alpha0", "alpha1", "wc", "g0c", "g1c"}
    assert all(np.all(np.isfinite(values)) for values in out.duffing_mode_parameters.values())
    assert all(np.all(np.isfinite(values)) for values in out.duffing_symbolic_coefficients.values())
    for key, values in out.duffing_symbolic_coefficients.items():
        assert values.shape == out.duffing_symbolic_coefficient_names[key].shape
    assert out.duffing_symbolic_coefficient_names["wc"].shape == (1,)
    assert out.duffing_symbolic_coefficient_names["wc"][0] == "c0"


def test_duffing_symbolic_fitted_static_reuses_initial_mode_parameter_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        sweep_target="q0",
        duffing_calibration_mode="symbolic-fitted-static",
    )
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    original = duffing_calibration_module._build_mode_parameter_arrays
    call_count = 0

    def counting_build_mode_parameter_arrays(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        duffing_calibration_module,
        "_build_mode_parameter_arrays",
        counting_build_mode_parameter_arrays,
    )

    run_static_benchmark(cfg)

    assert call_count == 1


def test_duffing_symbolic_fitted_static_does_not_use_manual_flux_to_ej_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        sweep_target="q0",
        duffing_calibration_mode="symbolic-fitted-static",
    )
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    original = duffing_calibration_module.flux_dependent_EJ
    call_count = 0

    def counting_flux_dependent_EJ(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        duffing_calibration_module,
        "flux_dependent_EJ",
        counting_flux_dependent_EJ,
    )

    run_static_benchmark(cfg)

    assert call_count == 0


def test_symbolic_duffing_evaluator_returns_symbolic_parameters_only(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        sweep_target="q0",
        duffing_calibration_mode="symbolic-fitted-static",
    )
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)
    out = run_static_benchmark(cfg)

    symbolic_parameters = duffing_module.evaluate_symbolic_duffing_parameter_fit(
        out.flux_values,
        sweep_target=cfg.static_benchmark.flux_control.sweep_target,
        coefficient_names=out.duffing_symbolic_coefficient_names,
        coefficients=out.duffing_symbolic_coefficients,
    )
    assert set(symbolic_parameters) == {"w0", "w1", "alpha0", "alpha1", "wc", "g0c", "g1c"}

    full_parameters = evaluate_symbolic_duffing_mode_parameters(
        out.flux_values,
        system_params=cfg.system,
        sweep_target=cfg.static_benchmark.flux_control.sweep_target,
        coefficient_names=out.duffing_symbolic_coefficient_names,
        coefficients=out.duffing_symbolic_coefficients,
    )
    assert set(full_parameters) == {"w0", "w1", "alpha0", "alpha1", "wc", "g0c", "g1c"}
    assert np.allclose(full_parameters["wc"], out.duffing_mode_parameters["wc"])


def test_cz_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    out = run_cz_benchmark(
        cfg,
        ramp_time_ns=4.0,
        hold_time_ns=12.0,
        dt_ns=1.0,
        enable_hold_time_scan=False,
    )
    assert out.times_ns.shape == (21,)


def test_cz_benchmark_ignores_sampled_static_arrays_when_fit_laws_exist(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        duffing_calibration_mode="symbolic-fitted-static",
    )
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)
    static_out = run_static_benchmark(cfg)

    baseline = run_cz_benchmark(
        cfg,
        ramp_time_ns=4.0,
        hold_time_ns=12.0,
        dt_ns=1.0,
        enable_hold_time_scan=False,
        precomputed_static_result=static_out,
        precomputed_static_runtime_s=0.0,
    )

    corrupted_effective = {
        key: np.full_like(np.asarray(values, dtype=float), 123.456)
        for key, values in static_out.effective_parameters.items()
    }
    corrupted_duffing = {
        key: np.full_like(np.asarray(values, dtype=float), -78.9)
        for key, values in static_out.duffing_mode_parameters.items()
    }
    corrupted_duffing["wc"] = np.full_like(
        np.asarray(static_out.duffing_mode_parameters["wc"], dtype=float),
        6.54321,
    )
    corrupted_static = replace(
        static_out,
        effective_parameters=corrupted_effective,
        duffing_mode_parameters=corrupted_duffing,
    )

    rebuilt = run_cz_benchmark(
        cfg,
        ramp_time_ns=4.0,
        hold_time_ns=12.0,
        dt_ns=1.0,
        enable_hold_time_scan=False,
        precomputed_static_result=corrupted_static,
        precomputed_static_runtime_s=0.0,
    )

    assert np.allclose(rebuilt.effective_conditional_phase, baseline.effective_conditional_phase)
    assert np.allclose(rebuilt.duffing_conditional_phase, baseline.duffing_conditional_phase)
    assert np.allclose(rebuilt.effective_populations_11, baseline.effective_populations_11)
    assert np.allclose(rebuilt.duffing_populations_11, baseline.duffing_populations_11)


def test_rx_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    out = run_rx_benchmark(
        cfg,
        drive_qubit=str(cfg.rx_benchmark.drive_qubit),
        drive_frequency=float(cfg.rx_benchmark.drive_frequency),
        drive_amplitude=float(cfg.rx_benchmark.drive_amplitude),
        drive_phase_rad=float(cfg.rx_benchmark.drive_phase_rad),
        total_time_ns=float(cfg.rx_benchmark.total_time_ns),
        dt_ns=float(cfg.rx_benchmark.dt_ns),
        rise_time_ns=float(cfg.rx_benchmark.rise_time_ns),
    )

    assert out.times_ns.ndim == 1
    assert out.effective_computational_amplitudes.shape[1:] == (4, 4)
    assert out.duffing_computational_amplitudes.shape[1:] == (4, 4)
    assert out.circuit_computational_amplitudes.shape[1:] == (4, 4)
    assert np.all(out.effective_pop_00_to_01 >= 0.0)
    assert np.all(out.duffing_leakage_from_00 >= 0.0)
    assert np.all(out.circuit_spectator_population_delta >= 0.0)
    assert out.effective_computational_amplitudes.shape[0] == out.times_ns.size
    assert out.duffing_computational_amplitudes.shape[0] == out.times_ns.size
    assert out.circuit_computational_amplitudes.shape[0] == out.times_ns.size
    assert np.all(np.isfinite(out.effective_leakage_from_10))
    assert np.all(np.isfinite(out.duffing_leakage_from_10))
    assert np.all(np.isfinite(out.circuit_leakage_from_10))


def test_cz_plot_writes_pdf(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    out = run_cz_benchmark(
        cfg,
        ramp_time_ns=4.0,
        hold_time_ns=12.0,
        dt_ns=1.0,
        enable_hold_time_scan=False,
    )
    outfile = tmp_path / "cz_benchmark.pdf"
    plot_cz_benchmark(out, outfile, title="test")
    assert outfile.exists()


def test_leakage_flow_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    out = run_leakage_flow_benchmark(
        cfg,
        ramp_time_ns=2.0,
        hold_time_ns=4.0,
        dt_ns=1.0,
        population_min_average=1e-5,
        transition_min_integrated_abs=1e-5,
        max_population_rows=12,
        max_transition_rows=12,
    )

    assert out.times_ns.shape == (9,)
    assert out.duffing_population_state_amplitudes_11.shape[0] == out.times_ns.size
    assert out.circuit_population_state_amplitudes_11.shape[0] == out.times_ns.size
    assert out.duffing_transition_signed_currents_11.shape[0] == out.times_ns.size
    assert out.circuit_transition_signed_currents_11.shape[0] == out.times_ns.size
    assert out.population_state_labels_11.size > 0
    assert out.transition_labels_11.size > 0
    assert out.duffing_population_state_amplitudes_11.shape[1] == out.population_state_labels_11.size
    assert out.circuit_population_state_amplitudes_11.shape[1] == out.population_state_labels_11.size
    assert out.duffing_transition_signed_currents_11.shape[1] == out.transition_labels_11.size
    assert out.circuit_transition_signed_currents_11.shape[1] == out.transition_labels_11.size
    assert np.all(np.isfinite(out.duffing_leakage_11))
    assert np.all(np.isfinite(out.circuit_leakage_11))
    assert np.all(out.duffing_leakage_11 >= -1e-12)
    assert np.all(out.circuit_leakage_11 >= -1e-12)


def test_leakage_flow_plot_writes_pdf(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    out = run_leakage_flow_benchmark(
        cfg,
        ramp_time_ns=2.0,
        hold_time_ns=4.0,
        dt_ns=1.0,
        population_min_average=1e-5,
        transition_min_integrated_abs=1e-5,
        max_population_rows=12,
        max_transition_rows=12,
    )

    outfile = tmp_path / "leakage_flow_benchmark.pdf"
    plot_leakage_flow_benchmark(out, outfile, title="test")
    assert outfile.exists()


def test_rx_plots_write_pdf(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_params_path=system_path, study_params_path=study_path)

    out = run_rx_benchmark(
        cfg,
        drive_qubit=str(cfg.rx_benchmark.drive_qubit),
        drive_frequency=float(cfg.rx_benchmark.drive_frequency),
        drive_amplitude=float(cfg.rx_benchmark.drive_amplitude),
        drive_phase_rad=float(cfg.rx_benchmark.drive_phase_rad),
        total_time_ns=float(cfg.rx_benchmark.total_time_ns),
        dt_ns=float(cfg.rx_benchmark.dt_ns),
        rise_time_ns=float(cfg.rx_benchmark.rise_time_ns),
    )

    populations_outfile = tmp_path / "rx_populations_benchmark.pdf"
    diagnostics_outfile = tmp_path / "rx_diagnostics_benchmark.pdf"
    plot_rx_populations_benchmark(out, populations_outfile)
    plot_rx_diagnostics_benchmark(out, diagnostics_outfile)
    assert populations_outfile.exists()
    assert diagnostics_outfile.exists()
