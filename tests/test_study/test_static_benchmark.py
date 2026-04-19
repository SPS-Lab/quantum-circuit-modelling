from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from study.comparison.cz import run_cz_benchmark
from study.comparison.leakage import run_leakage_benchmark
from study.comparison.static import run_static_benchmark
from study.config import load_study_config
from study.models.effective import fit_single_harmonic_parameters



def _write_small_system_params(tmp_path: Path) -> Path:
    src = _ROOT / "params" / "system_params.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload["parameters"]["q1"]["ncut"] = 25
    payload["parameters"]["q2"]["ncut"] = 25

    dst = tmp_path / "system_params_small.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst



def _write_small_study_params(tmp_path: Path) -> Path:
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

    dst = tmp_path / "study_params_small.json"
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



def test_cz_and_leakage_headers_raise_not_implemented(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    cfg = load_study_config(system_path, study_path)

    with pytest.raises(NotImplementedError):
        run_cz_benchmark(cfg)
    with pytest.raises(NotImplementedError):
        run_leakage_benchmark(cfg)
