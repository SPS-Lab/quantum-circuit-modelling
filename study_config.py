"""Configuration loading for paper-aligned model comparisons.

All benchmark and model parameters are read from JSON files under ``/params`` and
then threaded explicitly through the call stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np


SelectionMode = Literal["continuous", "bare"]
DerivationSource = Literal["duffing", "circuit"]
FitBasis = Literal["single-harmonic"]


@dataclass(frozen=True)
class TransmonSystemParams:
    EJmax: float
    EC: float
    d: float
    flux: float
    ng: float
    ncut: int
    truncated_dim: int
    id_str: str


@dataclass(frozen=True)
class OscillatorSystemParams:
    E_osc: float
    kappa_over_2pi: float
    truncated_dim: int
    id_str: str


@dataclass(frozen=True)
class InteractionSystemParams:
    g_1c: float
    g_2c: float


@dataclass(frozen=True)
class SystemParams:
    q1: TransmonSystemParams
    q2: TransmonSystemParams
    c: OscillatorSystemParams
    interactions: InteractionSystemParams


@dataclass(frozen=True)
class FluxSweepConfig:
    start: float
    stop: float
    num_points: int


@dataclass(frozen=True)
class CouplerFrequencyConfig:
    wc0: float
    amplitude: float


@dataclass(frozen=True)
class DressedSubspaceConfig:
    n_candidate_states: int
    selection_mode: SelectionMode


@dataclass(frozen=True)
class TransmonSpectralExtractionConfig:
    ncut: int
    truncated_dim: int


@dataclass(frozen=True)
class DuffingHilbertTruncationConfig:
    nlevels_qubit: int
    nlevels_coupler: int


@dataclass(frozen=True)
class DuffingModelConfig:
    transmon_spectral_extraction: TransmonSpectralExtractionConfig
    hilbert_truncation: DuffingHilbertTruncationConfig
    coupler_anharmonicity: float


@dataclass(frozen=True)
class CircuitHilbertTruncationConfig:
    q1_truncated_dim: int
    q2_truncated_dim: int
    c_truncated_dim: int


@dataclass(frozen=True)
class CircuitModelConfig:
    hilbert_truncation: CircuitHilbertTruncationConfig
    interaction_validity_check: bool


@dataclass(frozen=True)
class EffectiveModelConfig:
    derivation_source: DerivationSource
    fit_basis: FitBasis


@dataclass(frozen=True)
class RegimeThresholdsConfig:
    idle_ratio: float
    near_ratio: float


@dataclass(frozen=True)
class OutputConfig:
    figure: str


@dataclass(frozen=True)
class StaticBenchmarkConfig:
    flux_sweep: FluxSweepConfig
    coupler_frequency: CouplerFrequencyConfig
    dressed_subspace: DressedSubspaceConfig
    duffing_model: DuffingModelConfig
    circuit_model: CircuitModelConfig
    effective_model: EffectiveModelConfig
    regime_thresholds: RegimeThresholdsConfig
    outputs: OutputConfig


@dataclass(frozen=True)
class StudyConfig:
    system: SystemParams
    static_benchmark: StaticBenchmarkConfig



def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON at {path} must decode to an object")
    return payload



def _require_dict(parent: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    if key not in parent:
        raise KeyError(f"Missing required key {path}.{key}")
    value = parent[key]
    if not isinstance(value, dict):
        raise TypeError(f"{path}.{key} must be an object")
    return value



def _require_float(parent: dict[str, Any], key: str, path: str) -> float:
    if key not in parent:
        raise KeyError(f"Missing required key {path}.{key}")
    return float(parent[key])



def _require_int(parent: dict[str, Any], key: str, path: str) -> int:
    if key not in parent:
        raise KeyError(f"Missing required key {path}.{key}")
    return int(parent[key])



def _require_str(parent: dict[str, Any], key: str, path: str) -> str:
    if key not in parent:
        raise KeyError(f"Missing required key {path}.{key}")
    value = parent[key]
    if not isinstance(value, str):
        raise TypeError(f"{path}.{key} must be a string")
    return value



def _require_bool(parent: dict[str, Any], key: str, path: str) -> bool:
    if key not in parent:
        raise KeyError(f"Missing required key {path}.{key}")
    value = parent[key]
    if not isinstance(value, bool):
        raise TypeError(f"{path}.{key} must be a boolean")
    return value



def _parse_system(system_payload: dict[str, Any]) -> SystemParams:
    p = _require_dict(system_payload, "parameters", "system")

    q1p = _require_dict(p, "q1", "system.parameters")
    q2p = _require_dict(p, "q2", "system.parameters")
    cp = _require_dict(p, "c", "system.parameters")
    ip = _require_dict(p, "interactions", "system.parameters")

    q1 = TransmonSystemParams(
        EJmax=_require_float(q1p, "EJmax", "system.parameters.q1"),
        EC=_require_float(q1p, "EC", "system.parameters.q1"),
        d=_require_float(q1p, "d", "system.parameters.q1"),
        flux=_require_float(q1p, "flux", "system.parameters.q1"),
        ng=_require_float(q1p, "ng", "system.parameters.q1"),
        ncut=_require_int(q1p, "ncut", "system.parameters.q1"),
        truncated_dim=_require_int(q1p, "truncated_dim", "system.parameters.q1"),
        id_str=_require_str(q1p, "id_str", "system.parameters.q1"),
    )
    q2 = TransmonSystemParams(
        EJmax=_require_float(q2p, "EJmax", "system.parameters.q2"),
        EC=_require_float(q2p, "EC", "system.parameters.q2"),
        d=_require_float(q2p, "d", "system.parameters.q2"),
        flux=_require_float(q2p, "flux", "system.parameters.q2"),
        ng=_require_float(q2p, "ng", "system.parameters.q2"),
        ncut=_require_int(q2p, "ncut", "system.parameters.q2"),
        truncated_dim=_require_int(q2p, "truncated_dim", "system.parameters.q2"),
        id_str=_require_str(q2p, "id_str", "system.parameters.q2"),
    )
    c = OscillatorSystemParams(
        E_osc=_require_float(cp, "E_osc", "system.parameters.c"),
        kappa_over_2pi=_require_float(cp, "kappa_over_2pi", "system.parameters.c"),
        truncated_dim=_require_int(cp, "truncated_dim", "system.parameters.c"),
        id_str=_require_str(cp, "id_str", "system.parameters.c"),
    )
    interactions = InteractionSystemParams(
        g_1c=_require_float(ip, "g_1c", "system.parameters.interactions"),
        g_2c=_require_float(ip, "g_2c", "system.parameters.interactions"),
    )
    return SystemParams(q1=q1, q2=q2, c=c, interactions=interactions)



def _parse_static_benchmark(study_payload: dict[str, Any]) -> StaticBenchmarkConfig:
    sb = _require_dict(study_payload, "static_benchmark", "study")

    flux = _require_dict(sb, "flux_sweep", "study.static_benchmark")
    coupler = _require_dict(sb, "coupler_frequency", "study.static_benchmark")
    dressed = _require_dict(sb, "dressed_subspace", "study.static_benchmark")
    duffing = _require_dict(sb, "duffing_model", "study.static_benchmark")
    d_spec = _require_dict(duffing, "transmon_spectral_extraction", "study.static_benchmark.duffing_model")
    d_hilbert = _require_dict(duffing, "hilbert_truncation", "study.static_benchmark.duffing_model")
    circuit = _require_dict(sb, "circuit_model", "study.static_benchmark")
    c_hilbert = _require_dict(circuit, "hilbert_truncation", "study.static_benchmark.circuit_model")
    effective = _require_dict(sb, "effective_model", "study.static_benchmark")
    regime = _require_dict(sb, "regime_thresholds", "study.static_benchmark")
    outputs = _require_dict(sb, "outputs", "study.static_benchmark")

    selection_mode = _require_str(dressed, "selection_mode", "study.static_benchmark.dressed_subspace")
    if selection_mode not in ("continuous", "bare"):
        raise ValueError("study.static_benchmark.dressed_subspace.selection_mode must be 'continuous' or 'bare'")

    derivation_source = _require_str(effective, "derivation_source", "study.static_benchmark.effective_model")
    if derivation_source not in ("duffing", "circuit"):
        raise ValueError("study.static_benchmark.effective_model.derivation_source must be 'duffing' or 'circuit'")

    fit_basis = _require_str(effective, "fit_basis", "study.static_benchmark.effective_model")
    if fit_basis not in ("single-harmonic",):
        raise ValueError("study.static_benchmark.effective_model.fit_basis must be 'single-harmonic'")

    q1_trunc = _require_int(c_hilbert, "q1_truncated_dim", "study.static_benchmark.circuit_model.hilbert_truncation")
    q2_trunc = _require_int(c_hilbert, "q2_truncated_dim", "study.static_benchmark.circuit_model.hilbert_truncation")
    if q1_trunc != q2_trunc:
        raise ValueError(
            "study.static_benchmark.circuit_model.hilbert_truncation requires "
            "q1_truncated_dim == q2_truncated_dim for computational-subspace indexing"
        )

    return StaticBenchmarkConfig(
        flux_sweep=FluxSweepConfig(
            start=_require_float(flux, "start", "study.static_benchmark.flux_sweep"),
            stop=_require_float(flux, "stop", "study.static_benchmark.flux_sweep"),
            num_points=_require_int(flux, "num_points", "study.static_benchmark.flux_sweep"),
        ),
        coupler_frequency=CouplerFrequencyConfig(
            wc0=_require_float(coupler, "wc0", "study.static_benchmark.coupler_frequency"),
            amplitude=_require_float(coupler, "amplitude", "study.static_benchmark.coupler_frequency"),
        ),
        dressed_subspace=DressedSubspaceConfig(
            n_candidate_states=_require_int(dressed, "n_candidate_states", "study.static_benchmark.dressed_subspace"),
            selection_mode=selection_mode,
        ),
        duffing_model=DuffingModelConfig(
            transmon_spectral_extraction=TransmonSpectralExtractionConfig(
                ncut=_require_int(d_spec, "ncut", "study.static_benchmark.duffing_model.transmon_spectral_extraction"),
                truncated_dim=_require_int(
                    d_spec,
                    "truncated_dim",
                    "study.static_benchmark.duffing_model.transmon_spectral_extraction",
                ),
            ),
            hilbert_truncation=DuffingHilbertTruncationConfig(
                nlevels_qubit=_require_int(d_hilbert, "nlevels_qubit", "study.static_benchmark.duffing_model.hilbert_truncation"),
                nlevels_coupler=_require_int(
                    d_hilbert,
                    "nlevels_coupler",
                    "study.static_benchmark.duffing_model.hilbert_truncation",
                ),
            ),
            coupler_anharmonicity=_require_float(
                duffing,
                "coupler_anharmonicity",
                "study.static_benchmark.duffing_model",
            ),
        ),
        circuit_model=CircuitModelConfig(
            hilbert_truncation=CircuitHilbertTruncationConfig(
                q1_truncated_dim=q1_trunc,
                q2_truncated_dim=q2_trunc,
                c_truncated_dim=_require_int(c_hilbert, "c_truncated_dim", "study.static_benchmark.circuit_model.hilbert_truncation"),
            ),
            interaction_validity_check=_require_bool(
                circuit,
                "interaction_validity_check",
                "study.static_benchmark.circuit_model",
            ),
        ),
        effective_model=EffectiveModelConfig(
            derivation_source=derivation_source,
            fit_basis=fit_basis,
        ),
        regime_thresholds=RegimeThresholdsConfig(
            idle_ratio=_require_float(regime, "idle_ratio", "study.static_benchmark.regime_thresholds"),
            near_ratio=_require_float(regime, "near_ratio", "study.static_benchmark.regime_thresholds"),
        ),
        outputs=OutputConfig(
            figure=_require_str(outputs, "figure", "study.static_benchmark.outputs"),
        ),
    )



def load_study_config(system_params_path: Path, study_params_path: Path) -> StudyConfig:
    system_payload = _load_json(system_params_path)
    study_payload = _load_json(study_params_path)
    return StudyConfig(
        system=_parse_system(system_payload),
        static_benchmark=_parse_static_benchmark(study_payload),
    )



def build_flux_values(flux_sweep: FluxSweepConfig) -> np.ndarray:
    return np.linspace(float(flux_sweep.start), float(flux_sweep.stop), int(flux_sweep.num_points))
