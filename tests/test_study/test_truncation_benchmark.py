from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comparison.truncation import run_truncation_benchmark
from plots.truncation import plot_truncation_benchmark
from study_config import load_study_config



def _write_small_system_params(tmp_path: Path) -> Path:
    payload = json.loads((_ROOT / "params" / "system_params.json").read_text(encoding="utf-8"))
    payload["parameters"]["q1"]["ncut"] = 25
    payload["parameters"]["q2"]["ncut"] = 25
    dst = tmp_path / "system_small.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst



def _write_small_study_params(tmp_path: Path) -> Path:
    payload = json.loads((_ROOT / "params" / "static_benchmark_params.json").read_text(encoding="utf-8"))
    sb = payload["static_benchmark"]
    sb["flux_sweep"]["num_points"] = 9
    sb["dressed_subspace"]["n_candidate_states"] = 12
    sb["duffing_model"]["transmon_spectral_extraction"]["ncut"] = 20
    sb["duffing_model"]["transmon_spectral_extraction"]["truncated_dim"] = 10
    sb["duffing_model"]["hilbert_truncation"]["nlevels_qubit"] = 3
    sb["duffing_model"]["hilbert_truncation"]["nlevels_coupler"] = 3
    sb["circuit_model"]["hilbert_truncation"]["q1_truncated_dim"] = 4
    sb["circuit_model"]["hilbert_truncation"]["q2_truncated_dim"] = 4
    sb["circuit_model"]["hilbert_truncation"]["c_truncated_dim"] = 4
    sb["flux_control"]["sweep_target"] = "q1"

    tb = payload["truncation_benchmark"]
    tb["fixed_flux"] = 0.4
    tb["duffing_ncut_values"] = [3, 4, 6, 8]
    tb["duffing_truncated_dim"] = 12
    tb["lowest_excited_levels_to_plot"] = 2
    tb["circuit_reference_ncut"] = 35
    tb["duffing_calibration_mode"] = "per-flux"
    tb["outputs"]["figure"] = "figures/regime_map/test_truncation_benchmark.pdf"
    dst = tmp_path / "study_small.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst



def test_truncation_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    cfg = load_study_config(
        _write_small_system_params(tmp_path),
        _write_small_study_params(tmp_path),
    )

    out = run_truncation_benchmark(
        cfg,
        duffing_ncut_values=list(cfg.truncation_benchmark.duffing_ncut_values),
        fixed_flux=cfg.truncation_benchmark.fixed_flux,
        duffing_truncated_dim=cfg.truncation_benchmark.duffing_truncated_dim,
        circuit_reference_ncut=cfg.truncation_benchmark.circuit_reference_ncut,
        duffing_calibration_mode=cfg.truncation_benchmark.duffing_calibration_mode,
    )

    assert out.duffing_ncut_values.shape == (4,)
    assert out.duffing_j.shape == (4,)
    assert out.duffing_zeta.shape == (4,)
    assert out.duffing_effective_truncated_dim_values.shape == (4,)
    assert out.duffing_lowest_relative_energies.shape[0] == 4
    assert out.duffing_lowest_relative_energies.shape[1] >= 2
    assert out.circuit_lowest_relative_energies.shape == (out.duffing_lowest_relative_energies.shape[1],)
    assert out.max_duffing_ncut == int(np.max(out.duffing_ncut_values))
    n_report = min(
        int(cfg.truncation_benchmark.lowest_excited_levels_to_plot),
        max(0, int(out.duffing_lowest_relative_energies.shape[1]) - 1),
    )
    assert out.max_ncut_reported_excited_levels.shape == (n_report,)
    assert out.duffing_minus_circuit_at_max_ncut.shape == (n_report,)
    assert out.duffing_minus_circuit_percent_of_circuit_at_max_ncut.shape == (n_report,)
    assert np.all(np.isfinite(out.duffing_j))
    assert np.all(np.isfinite(out.duffing_zeta))
    assert np.all(out.duffing_effective_truncated_dim_values <= (2 * out.duffing_ncut_values + 1))
    assert np.all(out.duffing_effective_truncated_dim_values <= out.duffing_truncated_dim)
    assert np.all(np.isfinite(out.duffing_lowest_relative_energies))
    assert np.all(np.isfinite(out.circuit_lowest_relative_energies))
    assert np.all(np.isfinite(out.duffing_minus_circuit_at_max_ncut))
    assert np.all(np.isfinite(out.duffing_minus_circuit_percent_of_circuit_at_max_ncut))
    assert np.allclose(out.duffing_lowest_relative_energies[:, 0], 0.0, atol=1e-12)
    assert np.isclose(out.circuit_lowest_relative_energies[0], 0.0, atol=1e-12)
    assert np.isfinite(out.circuit_j)
    assert np.isfinite(out.circuit_zeta)



def test_truncation_benchmark_rejects_nonpositive_ncut(tmp_path: Path) -> None:
    cfg = load_study_config(
        _write_small_system_params(tmp_path),
        _write_small_study_params(tmp_path),
    )

    with pytest.raises(ValueError, match="positive"):
        run_truncation_benchmark(
            cfg,
            duffing_ncut_values=[0],
            fixed_flux=0.4,
            duffing_truncated_dim=cfg.truncation_benchmark.duffing_truncated_dim,
            circuit_reference_ncut=35,
            duffing_calibration_mode="per-flux",
        )



def test_truncation_plot_writes_pdf(tmp_path: Path) -> None:
    cfg = load_study_config(
        _write_small_system_params(tmp_path),
        _write_small_study_params(tmp_path),
    )
    out = run_truncation_benchmark(
        cfg,
        duffing_ncut_values=list(cfg.truncation_benchmark.duffing_ncut_values[:3]),
        fixed_flux=cfg.truncation_benchmark.fixed_flux,
        duffing_truncated_dim=cfg.truncation_benchmark.duffing_truncated_dim,
        circuit_reference_ncut=cfg.truncation_benchmark.circuit_reference_ncut,
        duffing_calibration_mode=cfg.truncation_benchmark.duffing_calibration_mode,
    )

    outfile = tmp_path / "truncation_benchmark.pdf"
    plot_truncation_benchmark(
        out,
        outfile,
        title="test",
        lowest_excited_levels_to_plot=cfg.truncation_benchmark.lowest_excited_levels_to_plot,
    )
    assert outfile.exists()
