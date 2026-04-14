"""Focused pytest coverage for Transmon+Oscillator model3 reference."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

# Repo root so `model3` resolves under pytest.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model3.comparison import compare_model1_model2_against_scqubits
from model3.reference_params import DEFAULT_TRANSMON_KEY, load_transmon_params
from model3.scqref import transmon_oscillator_hamiltonian


def test_ibm_manila_q1_params_available() -> None:
    params = load_transmon_params(DEFAULT_TRANSMON_KEY)
    assert set(params.keys()) == {"EJ", "EC"}
    assert params["EJ"] == 11.34
    assert params["EC"] == 0.293


def test_transmon_oscillator_hamiltonian_shape() -> None:
    params = load_transmon_params(DEFAULT_TRANSMON_KEY)
    H = transmon_oscillator_hamiltonian(
        transmon1_params=params,
        transmon2_params=params,
        transmon_dim=4,
        coupler_dim=5,
        transmon_ncut=35,
        coupler_E_osc=6.0,
        g_1c=0.09,
        g_2c=0.085,
    )
    assert H.shape == (4 * 5 * 4, 4 * 5 * 4)
    assert np.allclose(H, H.conj().T, atol=1e-10)


def test_regime_map_model2_outperforms_model1(tmp_path) -> None:
    params = load_transmon_params(DEFAULT_TRANSMON_KEY)
    flux = np.linspace(0.0, 1.0, 31)
    outfile = tmp_path / "regime_map.pdf"
    out = compare_model1_model2_against_scqubits(
        flux,
        wc0=5.05,
        A=0.95,
        reference={
            "transmon1_params": params,
            "transmon2_params": params,
            "transmon_dim": 4,
            "coupler_dim": 4,
            "transmon_ncut": 35,
            "g_1c": 0.09,
            "g_2c": 0.085,
        },
        model2_nlevels_qubit=3,
        model2_nlevels_coupler=3,
        model1_mode="cosine-fit",
        outfile=str(outfile),
    )
    summary = out["summary"]
    assert summary["model2_idle_rmse"] < summary["model1_idle_rmse"]
    assert float(np.mean(out["err_model2"])) < float(np.mean(out["err_model1"]))
    assert outfile.exists()


def test_selection_mode_accepts_both_options(tmp_path) -> None:
    """Selection mode API: both supported values should run and round-trip."""
    params = load_transmon_params(DEFAULT_TRANSMON_KEY)
    flux = np.linspace(0.0, 1.0, 41)
    common_kwargs = dict(
        wc0=5.05,
        A=0.95,
        reference={
            "transmon1_params": params,
            "transmon2_params": params,
            "transmon_dim": 4,
            "coupler_dim": 4,
            "transmon_ncut": 35,
            "g_1c": 0.09,
            "g_2c": 0.085,
        },
        model2_nlevels_qubit=3,
        model2_nlevels_coupler=3,
        model1_mode="cosine-fit",
    )
    for mode in ("continuous", "bare"):
        outfile = tmp_path / f"regime_map_{mode}.pdf"
        out = compare_model1_model2_against_scqubits(
            flux,
            selection_mode=mode,
            outfile=str(outfile),
            **common_kwargs,
        )
        assert out["selection_mode"] == mode
        assert np.isfinite(float(np.mean(out["err_model2"])))
        assert float(np.mean(out["err_model2"])) < float(np.mean(out["err_model1"]))
        assert outfile.exists()


def test_selection_mode_rejects_invalid_value(tmp_path) -> None:
    """Invalid selection mode should raise a clear ValueError."""
    params = load_transmon_params(DEFAULT_TRANSMON_KEY)
    flux = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match="selection_mode must be one of"):
        compare_model1_model2_against_scqubits(
            flux,
            wc0=5.05,
            A=0.95,
            reference={
                "transmon1_params": params,
                "transmon2_params": params,
                "transmon_dim": 4,
                "coupler_dim": 4,
                "transmon_ncut": 35,
                "g_1c": 0.09,
                "g_2c": 0.085,
            },
            model2_nlevels_qubit=3,
            model2_nlevels_coupler=3,
            selection_mode="not-a-mode",
            outfile=str(tmp_path / "regime_map_invalid.pdf"),
        )
