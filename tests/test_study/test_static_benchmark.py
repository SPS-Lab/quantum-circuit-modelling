from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comparison.cz import run_cz_benchmark
from comparison.leakage import run_leakage_benchmark
from comparison.static import run_static_benchmark
from models.dressed import extract_model1_parameters_from_4x4_stack
from study_config import load_study_config
from models.effective import fit_single_harmonic_parameters



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
    src = _ROOT / "params" / "static_benchmark_params.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
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

    suffix = str(coupler_amplitude).replace("-", "m").replace(".", "p")
    dst = tmp_path / f"study_params_small_{sweep_target}_A{suffix}_{duffing_calibration_mode}.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst



def test_load_study_config() -> None:
    cfg = load_study_config(
        _ROOT / "params" / "system_params.json",
        _ROOT / "params" / "static_benchmark_params.json",
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
    assert cfg.leakage_benchmark.total_time_ns > 0.0
    assert cfg.leakage_benchmark.ramp_time_ns > 0.0
    assert cfg.leakage_benchmark.dt_ns > 0.0
    assert cfg.leakage_benchmark.top_destination_rows >= 1



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
    assert out.effective_populations_11.shape == (21, 4)
    assert out.duffing_populations_11.shape == (21, 4)
    assert out.circuit_populations_11.shape == (21, 4)
    assert np.all(np.isfinite(out.effective_conditional_phase))
    assert np.all(np.isfinite(out.duffing_conditional_phase))
    assert np.all(np.isfinite(out.circuit_conditional_phase))
    assert np.all(out.effective_leakage_11 >= -1e-12)
    assert np.all(out.duffing_leakage_11 >= -1e-12)
    assert np.all(out.circuit_leakage_11 >= -1e-12)


def test_leakage_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    pytest.importorskip("qutip")

    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_path, study_path)

    out = run_leakage_benchmark(
        cfg,
        ramp_time_ns=4.0,
        hold_time_ns=12.0,
        dt_ns=1.0,
        enable_hold_time_scan=False,
    )
    assert out.times_ns.shape == (21,)
    assert out.effective_leakage_11.shape == (21,)
    assert out.duffing_leakage_11.shape == (21,)
    assert out.circuit_leakage_11.shape == (21,)
    assert out.circuit_populations_11.shape == (21, 4)
    assert np.all(np.isfinite(out.circuit_leakage_11))
    assert np.all(out.circuit_leakage_11 >= -1e-12)
    assert np.isfinite(out.summary["duffing_fraction_of_time_integrated_leakage_to_state_011_11"])
    assert np.isfinite(out.summary["circuit_fraction_of_time_integrated_leakage_to_state_011_11"])
    assert len(out.duffing_leakage_destination_populations_11) > 0
    assert len(out.circuit_leakage_destination_populations_11) > 0
    assert "|0,1,1>" in out.duffing_leakage_destination_populations_11
    assert "|0,1,1>" in out.circuit_leakage_destination_populations_11

    duf_dest_matrix = np.column_stack([v for _, v in sorted(out.duffing_leakage_destination_populations_11.items())])
    cir_dest_matrix = np.column_stack([v for _, v in sorted(out.circuit_leakage_destination_populations_11.items())])
    assert np.allclose(np.sum(duf_dest_matrix, axis=1), out.duffing_leakage_11, atol=1e-9)
    assert np.allclose(np.sum(cir_dest_matrix, axis=1), out.circuit_leakage_11, atol=1e-9)
