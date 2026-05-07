from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark_results_io import load_result_hdf5, save_result_hdf5
from comparison.cz import run_cz_benchmark
from comparison.leakage_flow import run_leakage_flow_benchmark
from comparison.rx import run_rx_benchmark
from comparison.static import StaticBenchmarkResult, run_static_benchmark
from models.dressed import extract_model1_parameters_from_4x4_stack
from plotting.cz import plot_cz_benchmark
from plotting.leakage_flow import plot_leakage_flow_benchmark
from study_config import _flatten_run_all_benchmark_params, load_study_config
from models.effective import fit_single_harmonic_parameters



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
    payload["parameters"]["q1"]["ncut"] = 25
    payload["parameters"]["q2"]["ncut"] = 25

    dst = tmp_path / "system_params_small.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst



def _write_small_study_params(
    tmp_path: Path,
    *,
    coupler_amplitude: float = 0.0,
    sweep_target: str = "q1",
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
    sb["circuit_model"]["hilbert_truncation"]["q1_truncated_dim"] = 4
    sb["circuit_model"]["hilbert_truncation"]["q2_truncated_dim"] = 4
    sb["circuit_model"]["hilbert_truncation"]["c_truncated_dim"] = 4
    sb["coupler_frequency"]["amplitude"] = float(coupler_amplitude)
    sb["flux_control"]["sweep_target"] = str(sweep_target)
    sb["duffing_model"]["calibration_mode"] = str(duffing_calibration_mode)
    rb = payload["rx_benchmark"]
    rb["drive_frequency"] = 9.733
    rb["drive_amplitude"] = 0.05
    rb["drive_phase_rad"] = 0.0
    rb["total_time_ns"] = 6.0
    rb["dt_ns"] = 0.1
    rb["rise_time_ns"] = 1.0

    suffix = str(coupler_amplitude).replace("-", "m").replace(".", "p")
    dst = tmp_path / f"study_params_small_{sweep_target}_A{suffix}_{duffing_calibration_mode}.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst



def test_load_study_config(tmp_path: Path) -> None:
    cfg = load_study_config(
        _write_small_system_params(tmp_path),
        _write_small_study_params(tmp_path),
    )
    assert cfg.system.q1.EJmax > 0.0
    assert cfg.static_benchmark.flux_sweep.num_points > 2
    assert cfg.static_benchmark.effective_model.derivation_source in {"duffing", "circuit"}
    assert cfg.static_benchmark.flux_control.sweep_target in {"coupler", "q1", "q2"}
    assert cfg.static_benchmark.duffing_model.calibration_mode in {"fixed", "analytic-per-flux", "per-flux"}
    assert len(cfg.truncation_benchmark.duffing_ncut_values) > 0
    assert cfg.truncation_benchmark.duffing_truncated_dim >= 3
    assert cfg.truncation_benchmark.lowest_excited_levels_to_plot >= 1
    assert cfg.truncation_benchmark.circuit_reference_ncut > 0
    assert cfg.truncation_benchmark.duffing_calibration_mode in {"fixed", "analytic-per-flux", "per-flux"}
    assert cfg.cz_benchmark.total_time_ns is None or cfg.cz_benchmark.total_time_ns > 0.0
    assert cfg.cz_benchmark.hold_time_ns is None or cfg.cz_benchmark.hold_time_ns >= 0.0
    assert cfg.cz_benchmark.ramp_time_ns > 0.0
    assert cfg.cz_benchmark.dt_ns > 0.0
    assert cfg.cz_benchmark.scan_dt_ns > 0.0
    assert cfg.cz_benchmark.scan_max_hold_ns >= 0.0
    assert cfg.cz_benchmark.scan_leakage_penalty >= 0.0
    assert cfg.rx_benchmark.drive_qubit == "q1"
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
        "w1": non_harmonic,
        "w2": non_harmonic + 0.1,
        "J": 0.05 * non_harmonic,
        "zeta": 0.01 * non_harmonic,
    }
    fit = fit_single_harmonic_parameters(flux, extracted)
    mismatch = np.max(np.abs(fit.fitted_parameters["w1"] - extracted["w1"]))
    assert mismatch > 1e-3


def test_extract_model1_parameters_gauge_fixes_exchange_phase() -> None:
    H = np.zeros((3, 4, 4), dtype=complex)
    for k, j in enumerate([0.05, -0.06, 0.04]):
        H[k, 1, 1] = 1.0
        H[k, 2, 2] = 1.2
        H[k, 1, 2] = 2.0 * j
        H[k, 2, 1] = 2.0 * j

    params_raw = extract_model1_parameters_from_4x4_stack(H, gauge_fix_exchange=False)
    params_fixed = extract_model1_parameters_from_4x4_stack(H, gauge_fix_exchange=True)

    assert np.any(params_raw["J"] < 0.0)
    assert np.all(params_fixed["J"] >= 0.0)
    assert np.allclose(params_fixed["w1"], params_raw["w1"])
    assert np.allclose(params_fixed["w2"], params_raw["w2"])
    assert np.allclose(params_fixed["zeta"], params_raw["zeta"])



def test_static_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_path, study_path)

    out = run_static_benchmark(cfg)

    assert out.effective_relative_energies.shape == (9, 4)
    assert out.duffing_relative_energies.shape == (9, 4)
    assert out.circuit_relative_energies.shape == (9, 4)
    assert np.isfinite(float(np.mean(out.effective_error_rmse)))
    assert np.isfinite(float(np.mean(out.duffing_error_rmse)))
    assert set(out.effective_fit_coefficients) == {"J", "zeta"}
    assert out.effective_fit_coefficients["J"].shape == (3,)
    assert out.effective_fit_coefficients["zeta"].shape == (3,)


def test_static_benchmark_fit_coefficients_roundtrip_through_hdf5(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_path, study_path)

    out = run_static_benchmark(cfg)
    results_path = tmp_path / "static_results.h5"
    save_result_hdf5(out, results_path, benchmark_name="static")
    loaded = load_result_hdf5(results_path, StaticBenchmarkResult, expected_benchmark_name="static")

    assert np.allclose(loaded.effective_fit_coefficients["J"], out.effective_fit_coefficients["J"])
    assert np.allclose(loaded.effective_fit_coefficients["zeta"], out.effective_fit_coefficients["zeta"])



def test_static_benchmark_uses_coupler_amplitude_from_config(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path_zero = _write_small_study_params(tmp_path, coupler_amplitude=0.0, sweep_target="coupler")
    study_path_nonzero = _write_small_study_params(tmp_path, coupler_amplitude=0.8, sweep_target="coupler")

    out_zero = run_static_benchmark(load_study_config(system_path, study_path_zero))
    out_nonzero = run_static_benchmark(load_study_config(system_path, study_path_nonzero))

    # Nonzero coupler modulation must produce flux variation in the static spectrum.
    assert float(np.ptp(out_zero.circuit_relative_energies[:, 1])) < 1e-10
    assert float(np.ptp(out_nonzero.circuit_relative_energies[:, 1])) > 1e-6


def test_static_benchmark_q1_sweep_varies_spectrum_with_fixed_coupler(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path, coupler_amplitude=0.0, sweep_target="q1")

    out = run_static_benchmark(load_study_config(system_path, study_path))

    assert float(np.ptp(out.circuit_relative_energies[:, 1])) > 1e-6


def test_duffing_fixed_calibration_is_not_recomputed_per_flux(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        coupler_amplitude=0.0,
        sweep_target="q1",
        duffing_calibration_mode="fixed",
    )
    out = run_static_benchmark(load_study_config(system_path, study_path))
    assert float(np.ptp(out.duffing_relative_energies[:, 1])) < 1e-10


def test_duffing_per_flux_calibration_can_be_enabled_explicitly(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        coupler_amplitude=0.0,
        sweep_target="q1",
        duffing_calibration_mode="per-flux",
    )
    out = run_static_benchmark(load_study_config(system_path, study_path))
    assert float(np.ptp(out.duffing_relative_energies[:, 1])) > 1e-6


def test_duffing_analytic_per_flux_calibration_varies_with_flux(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(
        tmp_path,
        coupler_amplitude=0.0,
        sweep_target="q1",
        duffing_calibration_mode="analytic-per-flux",
    )
    out = run_static_benchmark(load_study_config(system_path, study_path))
    assert float(np.ptp(out.duffing_relative_energies[:, 1])) > 1e-6


def test_cz_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    pytest.importorskip("qutip")

    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_path, study_path)

    out = run_cz_benchmark(
        cfg,
        ramp_time_ns=4.0,
        hold_time_ns=12.0,
        dt_ns=1.0,
        enable_hold_time_scan=False,
    )
    assert out.times_ns.shape == (21,)


def test_rx_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_path, study_path)

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
    pytest.importorskip("qutip")

    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_path, study_path)

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
    cfg = load_study_config(system_path, study_path)

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
    cfg = load_study_config(system_path, study_path)

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
