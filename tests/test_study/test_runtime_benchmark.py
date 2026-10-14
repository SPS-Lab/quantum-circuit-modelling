from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comparison.runtime import run_runtime_benchmark
import comparison.cz as cz_module
import comparison.static as static_module
from benchmark_results_io import load_result_hdf5, save_result_hdf5
from comparison.runtime import RuntimeBenchmarkResult
from comparison.cz import CzCircuitOnlyResult, run_cz_benchmark
from comparison.static import CircuitFluxSweepReference, build_circuit_static_reference
from plotting.runtime import plot_runtime_benchmark
from plotting.style import benchmark_style_names, benchmark_style_outfile
from study_config import _flatten_run_all_benchmark_params, load_study_config


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
    payload = json.loads(
        (_ROOT / "params" / "system_params.json").read_text(encoding="utf-8")
    )
    dst = tmp_path / "system_runtime_small.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst


def _write_small_study_params(tmp_path: Path) -> Path:
    payload = json.loads(
        (_ROOT / "params" / "benchmark_params.json").read_text(encoding="utf-8")
    )
    payload = _flatten_run_all_benchmark_params(payload)
    _add_required_study_sections(payload)

    sb = payload["static_benchmark"]
    sb["flux_sweep"]["num_points"] = 7
    sb["dressed_subspace"]["n_candidate_states"] = 10
    sb["duffing_model"]["hilbert_truncation"]["nlevels_qubit"] = 3
    sb["duffing_model"]["hilbert_truncation"]["nlevels_coupler"] = 3
    sb["circuit_model"]["transmon_charge_basis"]["q0_ncut"] = 20
    sb["circuit_model"]["transmon_charge_basis"]["q1_ncut"] = 20
    sb["circuit_model"]["hilbert_truncation"]["q0_truncated_dim"] = 4
    sb["circuit_model"]["hilbert_truncation"]["q1_truncated_dim"] = 4
    sb["circuit_model"]["hilbert_truncation"]["c_truncated_dim"] = 4
    sb["flux_control"]["sweep_target"] = "q0"

    cz = payload["cz_benchmark"]
    cz["ramp_time_ns"] = 1.0
    cz["dt_ns"] = 0.5
    cz["enable_hold_time_scan"] = False
    cz["hold_time_ns"] = 0.5

    rb = payload["runtime_benchmark"]
    rb["qubit_truncation_values"] = [2, 3]
    rb["duffing_calibration_mode"] = "per-flux"
    rb["repeats"] = 3
    rb["hold_time_ns"] = 0.5
    rb["outputs"]["figure"] = "results/test_runtime_benchmark.pdf"

    dst = tmp_path / "study_runtime_small.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst


def _assert_benchmark_pdfs_written(outfile: Path) -> None:
    for style_name in benchmark_style_names():
        styled_outfile = benchmark_style_outfile(outfile, style_name)
        assert styled_outfile.exists()
        assert styled_outfile.stat().st_size > 0


def test_runtime_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path),
    )

    out = run_runtime_benchmark(
        cfg,
        qubit_truncation_values=list(cfg.runtime_benchmark.qubit_truncation_values),
        duffing_calibration_mode=cfg.runtime_benchmark.duffing_calibration_mode,
        repeats=cfg.runtime_benchmark.repeats,
    )

    assert out.qubit_truncation_values.shape == (2,)
    assert out.repeats == 3
    assert np.isclose(out.fixed_hold_time_ns, 0.5)
    assert out.duffing_dynamics_runtime_s.shape == (2,)
    assert out.circuit_dynamics_runtime_s.shape == (2,)
    assert np.all(np.isfinite(out.duffing_dynamics_runtime_s))
    assert np.all(np.isfinite(out.circuit_dynamics_runtime_s))
    assert np.all(out.selected_hold_times_ns >= 0.0)
    assert np.allclose(out.selected_hold_times_ns, out.fixed_hold_time_ns)
    assert np.all(out.n_time_points >= 2)
    assert np.all(out.duffing_hilbert_dims > 0)
    assert np.all(out.circuit_hilbert_dims > 0)
    assert np.array_equal(out.qubit_truncation_values, np.array([2, 3], dtype=int))
    assert np.array_equal(out.duffing_hilbert_dims, np.array([12, 27], dtype=int))
    assert np.array_equal(out.circuit_hilbert_dims, np.array([16, 36], dtype=int))


def test_runtime_plot_writes_pdf(tmp_path: Path) -> None:
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path),
    )
    out = run_runtime_benchmark(
        cfg,
        qubit_truncation_values=list(cfg.runtime_benchmark.qubit_truncation_values[:1]),
        duffing_calibration_mode=cfg.runtime_benchmark.duffing_calibration_mode,
        repeats=cfg.runtime_benchmark.repeats,
    )

    outfile = tmp_path / "runtime_benchmark.pdf"
    plot_runtime_benchmark(out, outfile)
    _assert_benchmark_pdfs_written(outfile)


@pytest.mark.parametrize("auto_hold", [False, True])
def test_runtime_circuit_only_skips_other_models(
    tmp_path: Path, monkeypatch, auto_hold: bool
) -> None:
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path),
    )
    if auto_hold:
        cfg = replace(
            cfg,
            cz_benchmark=replace(
                cfg.cz_benchmark,
                hold_time_ns=None,
                enable_hold_time_scan=True,
                scan_max_hold_ns=1.0,
            ),
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("Duffing construction should be skipped")

    monkeypatch.setattr(cz_module, "build_duffing_model_stack_from_scratch", forbidden)
    monkeypatch.setattr(
        cz_module, "build_duffing_model_stack_from_parameters", forbidden
    )
    monkeypatch.setattr(cz_module, "build_effective_hamiltonian_stack", forbidden)
    monkeypatch.setattr(
        static_module, "build_duffing_model_stack_from_scratch", forbidden
    )
    monkeypatch.setattr(
        static_module, "build_duffing_model_stack_from_parameters", forbidden
    )
    monkeypatch.setattr(
        static_module, "build_duffing_model_stack_from_coefficients", forbidden
    )
    monkeypatch.setattr(static_module, "build_effective_hamiltonian_stack", forbidden)

    out = run_runtime_benchmark(
        cfg,
        qubit_truncation_values=[2],
        duffing_calibration_mode=cfg.runtime_benchmark.duffing_calibration_mode,
        circuit_only=True,
        repeats=1,
    )

    assert np.array_equal(out.duffing_hilbert_dims, [0])
    assert np.isnan(out.duffing_build_runtime_s).all()
    assert np.isnan(out.duffing_propagation_runtime_s).all()
    assert np.isfinite(out.circuit_build_runtime_s).all()
    assert np.isfinite(out.circuit_propagation_runtime_s).all()
    outfile = tmp_path / "runtime_circuit_only.pdf"
    plot_runtime_benchmark(out, outfile)
    _assert_benchmark_pdfs_written(outfile)
    results_path = tmp_path / "runtime_circuit_only.h5"
    save_result_hdf5(out, results_path, benchmark_name="runtime")
    loaded = load_result_hdf5(
        results_path, RuntimeBenchmarkResult, expected_benchmark_name="runtime"
    )
    assert np.isnan(loaded.duffing_build_runtime_s).all()
    assert np.isfinite(loaded.circuit_build_runtime_s).all()


def test_circuit_only_helpers_return_typed_results(tmp_path: Path) -> None:
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path),
    )
    reference = build_circuit_static_reference(cfg)
    assert isinstance(reference, CircuitFluxSweepReference)
    result = run_cz_benchmark(
        cfg,
        hold_time_ns=0.5,
        enable_hold_time_scan=False,
        precomputed_static_result=reference,
        circuit_only=True,
    )
    assert isinstance(result, CzCircuitOnlyResult)
    assert not hasattr(result, "duffing_populations_11")
    assert result.circuit_hilbert_dim > 0
    assert result.circuit_populations_11.shape[0] == result.times_ns.size
    full_result = run_cz_benchmark(cfg, hold_time_ns=0.5, enable_hold_time_scan=False)
    assert np.array_equal(result.pulse_flux_values, full_result.pulse_flux_values)
    assert np.allclose(
        result.circuit_populations_11, full_result.circuit_populations_11
    )
    assert np.allclose(
        result.circuit_conditional_phase, full_result.circuit_conditional_phase
    )


def test_runtime_cli_circuit_only_without_duffing_mode(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    _write_small_system_params(tmp_path).replace(params_dir / "system_params.json")
    study_path = _write_small_study_params(tmp_path)
    payload = json.loads(study_path.read_text(encoding="utf-8"))
    runtime = payload["runtime_benchmark"]
    runtime["circuit_only"] = True
    runtime.pop("duffing_calibration_mode")
    runtime["qubit_truncation_values"] = [2]
    runtime["repeats"] = 1
    (params_dir / "benchmark_params.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    script_path = _ROOT / "scripts" / "run_runtime_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "runtime_benchmark_cli_test", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", [str(script_path), "--output-root", str(tmp_path / "runs")]
    )
    module.main()

    results_path = next((tmp_path / "runs").glob("*/runtime_results.h5"))
    result = load_result_hdf5(
        results_path, RuntimeBenchmarkResult, expected_benchmark_name="runtime"
    )
    assert result.duffing_calibration_mode == ""
    assert np.isnan(result.duffing_build_runtime_s).all()
    assert np.isfinite(result.circuit_build_runtime_s).all()
    report = capsys.readouterr().out
    assert "circuit_only=True" in report
    assert "duffing_calibration_mode=" not in report
    assert "Duffing calibration extraction:" not in report
