from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comparison.truncation import run_circuit_truncation_benchmark, run_duffing_truncation_benchmark
from plotting.truncation import plot_circuit_truncation_benchmark, plot_duffing_truncation_benchmark
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
    payload = json.loads((_ROOT / "params" / "system_params.json").read_text(encoding="utf-8"))
    payload["parameters"]["q0"]["ncut"] = 25
    payload["parameters"]["q1"]["ncut"] = 25
    dst = tmp_path / "system_small.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst



def _write_small_study_params(
    tmp_path: Path,
    *,
    duffing_calibration_mode: str = "per-flux",
    flux_num_points: int = 9,
    duffing_ncut_values: list[int] | None = None,
    truncation_flux_values: list[float] | None = None,
) -> Path:
    payload = json.loads((_ROOT / "params" / "benchmark_params.json").read_text(encoding="utf-8"))
    payload = _flatten_run_all_benchmark_params(payload)
    _add_required_study_sections(payload)
    sb = payload["static_benchmark"]
    sb["flux_sweep"]["num_points"] = int(flux_num_points)
    sb["dressed_subspace"]["n_candidate_states"] = 12
    sb["duffing_model"]["transmon_spectral_extraction"]["ncut"] = 20
    sb["duffing_model"]["transmon_spectral_extraction"]["truncated_dim"] = 10
    sb["duffing_model"]["hilbert_truncation"]["nlevels_qubit"] = 3
    sb["duffing_model"]["hilbert_truncation"]["nlevels_coupler"] = 3
    sb["circuit_model"]["hilbert_truncation"]["q0_truncated_dim"] = 4
    sb["circuit_model"]["hilbert_truncation"]["q1_truncated_dim"] = 4
    sb["circuit_model"]["hilbert_truncation"]["c_truncated_dim"] = 4
    sb["flux_control"]["sweep_target"] = "q0"
    truncation_flux_values = [0.10, 0.20, 0.30, 0.40, 0.50] if truncation_flux_values is None else list(truncation_flux_values)

    ctb = payload["circuit_truncation_benchmark"]
    ctb["flux_values"] = truncation_flux_values
    ctb["circuit_ncut_values"] = [4, 20, 35]
    ctb["circuit_qubit_truncated_dim_values"] = [3, 4]
    ctb["circuit_coupler_truncated_dim_values"] = [3, 4]
    ctb["lowest_excited_levels_to_plot"] = 2
    ctb["circuit_reference_ncut"] = 35
    ctb["circuit_reference_qubit_truncated_dim"] = 4
    ctb["circuit_reference_coupler_truncated_dim"] = 4
    ctb["outputs"]["figure"] = "results/test_circuit_truncation_benchmark.pdf"

    dtb = payload["duffing_truncation_benchmark"]
    dtb["flux_values"] = truncation_flux_values
    dtb["duffing_ncut_values"] = [3, 4, 6, 8] if duffing_ncut_values is None else list(duffing_ncut_values)
    dtb["duffing_truncated_dim"] = 12
    dtb["duffing_hilbert_qubit_dim_values"] = [2, 3]
    dtb["duffing_hilbert_coupler_dim_values"] = [2, 3]
    dtb["lowest_excited_levels_to_plot"] = 2
    dtb["circuit_reference_ncut"] = 35
    dtb["circuit_reference_qubit_truncated_dim"] = 4
    dtb["circuit_reference_coupler_truncated_dim"] = 4
    dtb["duffing_calibration_mode"] = str(duffing_calibration_mode)
    dtb["outputs"]["figure"] = "results/test_duffing_truncation_benchmark.pdf"
    dst = tmp_path / "study_small.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst


def test_circuit_truncation_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path),
    )

    out = run_circuit_truncation_benchmark(cfg)

    assert out.circuit_ncut_values.shape == (3,)
    assert out.flux_values.shape == (5,)
    assert out.circuit_ncut_effective_qubit_truncated_dim_values.shape == (3,)
    assert out.circuit_ncut_energy_rmse.shape == (3,)
    assert out.circuit_qubit_truncated_dim_values.shape == (2,)
    assert out.circuit_qubit_truncation_energy_rmse.shape == (2,)
    assert out.circuit_coupler_truncated_dim_values.shape == (2,)
    assert out.circuit_coupler_truncation_energy_rmse.shape == (2,)
    assert np.all(np.isfinite(out.circuit_ncut_energy_rmse))
    assert np.all(out.circuit_ncut_effective_qubit_truncated_dim_values <= (2 * out.circuit_ncut_values + 1))
    assert np.all(np.isfinite(out.circuit_qubit_truncation_energy_rmse))
    assert np.all(np.isfinite(out.circuit_coupler_truncation_energy_rmse))
    assert out.reference_circuit_j_values.shape == (5,)
    assert out.reference_circuit_zeta_values.shape == (5,)
    assert np.all(np.isfinite(out.reference_circuit_j_values))
    assert np.all(np.isfinite(out.reference_circuit_zeta_values))


def test_circuit_truncation_benchmark_accepts_more_than_five_flux_points(tmp_path: Path) -> None:
    flux_values = [0.10, 0.20, 0.23, 0.24, 0.25, 0.40, 0.50]
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path, truncation_flux_values=flux_values),
    )

    out = run_circuit_truncation_benchmark(cfg)

    assert out.flux_values.shape == (7,)
    assert out.reference_circuit_j_values.shape == (7,)
    assert out.reference_circuit_zeta_values.shape == (7,)


def test_duffing_truncation_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path),
    )

    out = run_duffing_truncation_benchmark(cfg)

    assert out.duffing_ncut_values.shape == (4,)
    assert out.flux_values.shape == (5,)
    assert out.duffing_ncut_effective_truncated_dim_values.shape == (4,)
    assert out.duffing_ncut_energy_rmse.shape == (4,)
    assert out.duffing_hilbert_qubit_dim_values.shape == (2,)
    assert out.duffing_hilbert_qubit_energy_rmse.shape == (2,)
    assert out.duffing_hilbert_coupler_dim_values.shape == (2,)
    assert out.duffing_hilbert_coupler_energy_rmse.shape == (2,)
    assert np.all(out.duffing_ncut_effective_truncated_dim_values <= (2 * out.duffing_ncut_values + 1))
    assert np.all(out.duffing_ncut_effective_truncated_dim_values <= out.duffing_truncated_dim)
    assert np.all(np.isfinite(out.duffing_ncut_energy_rmse))
    assert np.all(np.isfinite(out.duffing_hilbert_qubit_energy_rmse))
    assert np.all(np.isfinite(out.duffing_hilbert_coupler_energy_rmse))
    assert out.reference_circuit_j_values.shape == (5,)
    assert out.reference_circuit_zeta_values.shape == (5,)
    assert np.all(np.isfinite(out.reference_circuit_j_values))
    assert np.all(np.isfinite(out.reference_circuit_zeta_values))


def test_duffing_truncation_benchmark_accepts_more_than_five_flux_points(tmp_path: Path) -> None:
    flux_values = [0.10, 0.20, 0.23, 0.24, 0.25, 0.40, 0.50]
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path, truncation_flux_values=flux_values),
    )

    out = run_duffing_truncation_benchmark(cfg)

    assert out.flux_values.shape == (7,)
    assert out.reference_circuit_j_values.shape == (7,)
    assert out.reference_circuit_zeta_values.shape == (7,)


def test_duffing_truncation_benchmark_rejects_nonpositive_ncut(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        load_study_config(
            system_params_path=_write_small_system_params(tmp_path),
            study_params_path=_write_small_study_params(tmp_path, duffing_ncut_values=[0]),
        )


def test_circuit_truncation_plot_writes_pdf(tmp_path: Path) -> None:
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path),
    )
    out = run_circuit_truncation_benchmark(cfg)

    outfile = tmp_path / "circuit_truncation_benchmark.pdf"
    plot_circuit_truncation_benchmark(out, outfile)
    assert outfile.exists()


def test_duffing_truncation_plot_writes_pdf(tmp_path: Path) -> None:
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path),
    )
    out = run_duffing_truncation_benchmark(cfg)

    outfile = tmp_path / "duffing_truncation_benchmark.pdf"
    plot_duffing_truncation_benchmark(out, outfile)
    assert outfile.exists()


def test_duffing_truncation_benchmark_runs_with_symbolic_fitted_static(tmp_path: Path) -> None:
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(
            tmp_path,
            duffing_calibration_mode="symbolic-fitted-static",
            flux_num_points=5,
            duffing_ncut_values=[3],
        ),
    )

    out = run_duffing_truncation_benchmark(cfg)

    assert out.duffing_calibration_mode == "symbolic-fitted-static"
    assert out.duffing_ncut_values.shape == (1,)
    assert np.all(np.isfinite(out.duffing_ncut_energy_rmse))
    assert np.all(np.isfinite(out.duffing_hilbert_qubit_energy_rmse))
    assert np.all(np.isfinite(out.duffing_hilbert_coupler_energy_rmse))
