"""Migrated pytest coverage for regime-map comparison entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

# Repo root so `comparison` resolves under pytest.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comparison.regime_map import compare_model1_model2_against_scqubits



def _write_small_system_params(tmp_path: Path) -> Path:
    payload = json.loads((_ROOT / "params" / "system_params.json").read_text(encoding="utf-8"))
    payload["parameters"]["q1"]["ncut"] = 35
    payload["parameters"]["q2"]["ncut"] = 35
    dst = tmp_path / "system.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst



def _write_small_study_params(tmp_path: Path, *, selection_mode: str = "continuous") -> Path:
    payload = json.loads((_ROOT / "params" / "static_benchmark_params.json").read_text(encoding="utf-8"))
    sb = payload["static_benchmark"]
    sb["flux_sweep"]["num_points"] = 17
    sb["dressed_subspace"]["selection_mode"] = selection_mode
    sb["dressed_subspace"]["n_candidate_states"] = 12
    sb["duffing_model"]["hilbert_truncation"]["nlevels_qubit"] = 3
    sb["duffing_model"]["hilbert_truncation"]["nlevels_coupler"] = 3
    sb["duffing_model"]["transmon_spectral_extraction"]["ncut"] = 45
    sb["duffing_model"]["transmon_spectral_extraction"]["truncated_dim"] = 10
    sb["circuit_model"]["hilbert_truncation"]["q1_truncated_dim"] = 4
    sb["circuit_model"]["hilbert_truncation"]["q2_truncated_dim"] = 4
    sb["circuit_model"]["hilbert_truncation"]["c_truncated_dim"] = 4
    dst = tmp_path / "study.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst



def test_migrated_regime_map_runs_and_writes_figure(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    outfile = tmp_path / "regime_map_migrated.pdf"

    out = compare_model1_model2_against_scqubits(
        system_params_path=system_path,
        study_params_path=study_path,
        outfile=outfile,
    )

    assert outfile.exists()
    assert out["E1_rel"].shape == (17, 4)
    assert out["E2_rel"].shape == (17, 4)
    assert out["E3_rel"].shape == (17, 4)
    assert np.isfinite(float(np.mean(out["err_model1"])))
    assert np.isfinite(float(np.mean(out["err_model2"])))



def test_selection_mode_round_trips_from_params(tmp_path: Path) -> None:
    for mode in ("continuous", "bare"):
        system_path = _write_small_system_params(tmp_path)
        study_path = _write_small_study_params(tmp_path, selection_mode=mode)

        out = compare_model1_model2_against_scqubits(
            system_params_path=system_path,
            study_params_path=study_path,
            outfile=tmp_path / f"regime_map_{mode}.pdf",
        )
        assert out["selection_mode"] == mode



def test_invalid_selection_mode_from_params_raises(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_payload = json.loads((_ROOT / "params" / "static_benchmark_params.json").read_text(encoding="utf-8"))
    study_payload["static_benchmark"]["dressed_subspace"]["selection_mode"] = "not-a-mode"
    study_path = tmp_path / "bad_study.json"
    study_path.write_text(json.dumps(study_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="selection_mode"):
        compare_model1_model2_against_scqubits(
            system_params_path=system_path,
            study_params_path=study_path,
            outfile=tmp_path / "bad.pdf",
        )



def test_summary_keys_present(tmp_path: Path) -> None:
    system_path = _write_small_system_params(tmp_path)
    study_path = _write_small_study_params(tmp_path)
    out = compare_model1_model2_against_scqubits(
        system_params_path=system_path,
        study_params_path=study_path,
        outfile=tmp_path / "summary.pdf",
    )
    expected = {
        "model1_idle_rmse",
        "model1_idle_max_abs",
        "model1_near_rmse",
        "model1_near_max_abs",
        "model2_idle_rmse",
        "model2_idle_max_abs",
        "model2_near_rmse",
        "model2_near_max_abs",
    }
    assert expected.issubset(set(out["summary"].keys()))
