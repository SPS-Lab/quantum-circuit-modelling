from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comparison.runtime import run_runtime_benchmark
from plotting.runtime import plot_runtime_benchmark
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
    payload["parameters"]["q0"]["ncut"] = 20
    payload["parameters"]["q1"]["ncut"] = 20
    dst = tmp_path / "system_runtime_small.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst


def _write_small_study_params(tmp_path: Path) -> Path:
    payload = json.loads((_ROOT / "params" / "benchmark_params.json").read_text(encoding="utf-8"))
    payload = _flatten_run_all_benchmark_params(payload)
    _add_required_study_sections(payload)

    sb = payload["static_benchmark"]
    sb["flux_sweep"]["num_points"] = 7
    sb["dressed_subspace"]["n_candidate_states"] = 10
    sb["duffing_model"]["hilbert_truncation"]["nlevels_qubit"] = 3
    sb["duffing_model"]["hilbert_truncation"]["nlevels_coupler"] = 3
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
    rb["ncut_values"] = [3, 4]
    rb["duffing_truncated_dim"] = 8
    rb["duffing_calibration_mode"] = "per-flux"
    rb["repeats"] = 3
    rb["hold_time_ns"] = 0.5
    rb["outputs"]["figure"] = "results/test_runtime_benchmark.pdf"

    dst = tmp_path / "study_runtime_small.json"
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst


def test_runtime_benchmark_runs_with_small_config(tmp_path: Path) -> None:
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path),
    )

    out = run_runtime_benchmark(
        cfg,
        ncut_values=list(cfg.runtime_benchmark.ncut_values),
        duffing_truncated_dim=cfg.runtime_benchmark.duffing_truncated_dim,
        duffing_calibration_mode=cfg.runtime_benchmark.duffing_calibration_mode,
        repeats=cfg.runtime_benchmark.repeats,
    )

    assert out.ncut_values.shape == (2,)
    assert out.duffing_effective_truncated_dim_values.shape == (2,)
    assert out.repeats == 3
    assert np.isclose(out.fixed_hold_time_ns, 0.5)
    assert out.duffing_dynamics_runtime_s.shape == (2,)
    assert out.circuit_dynamics_runtime_s.shape == (2,)
    assert np.all(np.isfinite(out.duffing_dynamics_runtime_s))
    assert np.all(np.isfinite(out.circuit_dynamics_runtime_s))
    assert np.all(out.duffing_effective_truncated_dim_values <= (2 * out.ncut_values + 1))
    assert np.all(out.duffing_effective_truncated_dim_values <= cfg.runtime_benchmark.duffing_truncated_dim)
    assert np.all(out.selected_hold_times_ns >= 0.0)
    assert np.allclose(out.selected_hold_times_ns, out.fixed_hold_time_ns)
    assert np.all(out.n_time_points >= 2)
    assert np.all(out.duffing_hilbert_dims > 0)
    assert np.all(out.circuit_hilbert_dims > 0)


def test_runtime_plot_writes_pdf(tmp_path: Path) -> None:
    cfg = load_study_config(
        system_params_path=_write_small_system_params(tmp_path),
        study_params_path=_write_small_study_params(tmp_path),
    )
    out = run_runtime_benchmark(
        cfg,
        ncut_values=list(cfg.runtime_benchmark.ncut_values[:1]),
        duffing_truncated_dim=cfg.runtime_benchmark.duffing_truncated_dim,
        duffing_calibration_mode=cfg.runtime_benchmark.duffing_calibration_mode,
        repeats=cfg.runtime_benchmark.repeats,
    )

    outfile = tmp_path / "runtime_benchmark.pdf"
    plot_runtime_benchmark(out, outfile, title="Runtime Test")
    assert outfile.exists()
