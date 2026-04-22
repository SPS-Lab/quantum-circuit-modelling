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
SweepTarget = Literal["coupler", "q1", "q2"]
DuffingCalibrationMode = Literal["fixed", "analytic-per-flux", "per-flux"]


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
class FluxControlConfig:
    sweep_target: SweepTarget


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
    calibration_mode: DuffingCalibrationMode


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
    flux_control: FluxControlConfig
    coupler_frequency: CouplerFrequencyConfig
    dressed_subspace: DressedSubspaceConfig
    duffing_model: DuffingModelConfig
    circuit_model: CircuitModelConfig
    effective_model: EffectiveModelConfig
    regime_thresholds: RegimeThresholdsConfig
    outputs: OutputConfig


@dataclass(frozen=True)
class TruncationBenchmarkConfig:
    fixed_flux: float
    duffing_ncut_values: tuple[int, ...]
    duffing_truncated_dim: int
    lowest_excited_levels_to_plot: int
    circuit_reference_ncut: int
    duffing_calibration_mode: DuffingCalibrationMode
    outputs: OutputConfig


@dataclass(frozen=True)
class CzBenchmarkConfig:
    total_time_ns: float
    ramp_time_ns: float
    dt_ns: float
    enable_hold_time_scan: bool
    scan_dt_ns: float
    scan_max_hold_ns: float
    scan_leakage_penalty: float
    outputs: OutputConfig


@dataclass(frozen=True)
class LeakageFlowBenchmarkConfig:
    total_time_ns: float
    ramp_time_ns: float
    dt_ns: float
    population_min_average: float
    transition_min_integrated_abs: float
    max_population_rows: int
    max_transition_rows: int
    outputs: OutputConfig


@dataclass(frozen=True)
class LeakageBenchmarkConfig:
    total_time_ns: float
    ramp_time_ns: float
    dt_ns: float
    top_destination_rows: int


@dataclass(frozen=True)
class StateToStateLeakageBenchmarkConfig:
    total_time_ns: float
    ramp_time_ns: float
    dt_ns: float
    top_transition_rows: int


@dataclass(frozen=True)
class StudyConfig:
    system: SystemParams
    static_benchmark: StaticBenchmarkConfig
    cz_benchmark: CzBenchmarkConfig
    leakage_flow_benchmark: LeakageFlowBenchmarkConfig
    truncation_benchmark: TruncationBenchmarkConfig
    leakage_benchmark: LeakageBenchmarkConfig
    state_to_state_leakage_benchmark: StateToStateLeakageBenchmarkConfig



def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON at {path} must decode to an object")
    return payload


def _merge_external_study_sections(
    study_payload: dict[str, Any],
    *,
    study_params_path: Path,
) -> dict[str, Any]:
    moved = study_payload.get("moved_out_of_run_all")
    if not isinstance(moved, dict):
        return study_payload

    file_value = moved.get("file")
    if not isinstance(file_value, str) or not file_value.strip():
        return study_payload

    external_path = Path(file_value.strip())
    if not external_path.is_absolute():
        external_path = (study_params_path.parent / external_path).resolve()
    if not external_path.exists():
        return study_payload

    external_payload = _load_json(external_path)
    sections = moved.get("sections")
    if isinstance(sections, list):
        section_names = [str(name) for name in sections]
    else:
        section_names = [str(name) for name in external_payload.keys()]

    merged = dict(study_payload)
    for name in section_names:
        if name in merged:
            continue
        if name in external_payload:
            merged[name] = external_payload[name]
    return merged


def _deep_merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in update.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(base_value, value)
        else:
            merged[key] = value
    return merged


_RUN_ALL_BENCHMARK_CATEGORY_ORDER: tuple[str, ...] = (
    "shared_static_cz_leakage_flow_truncation",
    "shared_static_and_cz",
    "static_only",
    "truncation_only",
    "cz_only",
    "leakage_flow_only",
)


def _flatten_run_all_benchmark_params(payload: dict[str, Any]) -> dict[str, Any]:
    grouped = payload.get("run_all_benchmark_params")
    if grouped is None:
        return payload
    if not isinstance(grouped, dict):
        raise TypeError("study.run_all_benchmark_params must be an object")

    normalized = {k: v for k, v in payload.items() if k != "run_all_benchmark_params"}
    for category in _RUN_ALL_BENCHMARK_CATEGORY_ORDER:
        category_payload = grouped.get(category)
        if category_payload is None:
            continue
        if not isinstance(category_payload, dict):
            raise TypeError(f"study.run_all_benchmark_params.{category} must be an object")
        normalized = _deep_merge_dict(normalized, category_payload)

    unknown_categories = sorted(set(grouped.keys()) - set(_RUN_ALL_BENCHMARK_CATEGORY_ORDER))
    if unknown_categories:
        raise ValueError(
            "Unknown study.run_all_benchmark_params categories: "
            + ", ".join(unknown_categories)
        )
    return normalized


def _normalize_study_payload(
    raw_study_payload: dict[str, Any],
    *,
    study_params_path: Path,
) -> dict[str, Any]:
    payload = _merge_external_study_sections(raw_study_payload, study_params_path=study_params_path)
    return _flatten_run_all_benchmark_params(payload)



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


def _require_list(parent: dict[str, Any], key: str, path: str) -> list[Any]:
    if key not in parent:
        raise KeyError(f"Missing required key {path}.{key}")
    value = parent[key]
    if not isinstance(value, list):
        raise TypeError(f"{path}.{key} must be a list")
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
    flux_control = _require_dict(sb, "flux_control", "study.static_benchmark")
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

    calibration_mode = _require_str(duffing, "calibration_mode", "study.static_benchmark.duffing_model").strip().lower()
    if calibration_mode not in ("fixed", "analytic-per-flux", "per-flux"):
        raise ValueError(
            "study.static_benchmark.duffing_model.calibration_mode must be "
            "'fixed', 'analytic-per-flux', or 'per-flux'"
        )

    sweep_target_str = _require_str(flux_control, "sweep_target", "study.static_benchmark.flux_control")
    if sweep_target_str not in ("coupler", "q1", "q2"):
        raise ValueError("study.static_benchmark.flux_control.sweep_target must be 'coupler', 'q1', or 'q2'")
    sweep_target: SweepTarget = sweep_target_str

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
        flux_control=FluxControlConfig(
            sweep_target=sweep_target,
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
            calibration_mode=calibration_mode,
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


def _parse_truncation_benchmark(study_payload: dict[str, Any]) -> TruncationBenchmarkConfig:
    tb = _require_dict(study_payload, "truncation_benchmark", "study")

    ncuts_raw = _require_list(tb, "duffing_ncut_values", "study.truncation_benchmark")
    ncuts = tuple(int(v) for v in ncuts_raw)
    if len(ncuts) == 0:
        raise ValueError("study.truncation_benchmark.duffing_ncut_values must be non-empty")
    if any(v < 1 for v in ncuts):
        raise ValueError("study.truncation_benchmark.duffing_ncut_values must contain positive integers")
    trunc_dim = _require_int(tb, "duffing_truncated_dim", "study.truncation_benchmark")
    if trunc_dim < 3:
        raise ValueError("study.truncation_benchmark.duffing_truncated_dim must be >= 3")
    n_levels_to_plot = _require_int(tb, "lowest_excited_levels_to_plot", "study.truncation_benchmark")
    if n_levels_to_plot < 1:
        raise ValueError("study.truncation_benchmark.lowest_excited_levels_to_plot must be >= 1")

    mode = _require_str(tb, "duffing_calibration_mode", "study.truncation_benchmark").strip().lower()
    if mode not in ("fixed", "analytic-per-flux", "per-flux"):
        raise ValueError(
            "study.truncation_benchmark.duffing_calibration_mode must be "
            "'fixed', 'analytic-per-flux', or 'per-flux'"
        )
    outputs = _require_dict(tb, "outputs", "study.truncation_benchmark")

    return TruncationBenchmarkConfig(
        fixed_flux=_require_float(tb, "fixed_flux", "study.truncation_benchmark"),
        duffing_ncut_values=ncuts,
        duffing_truncated_dim=trunc_dim,
        lowest_excited_levels_to_plot=n_levels_to_plot,
        circuit_reference_ncut=_require_int(tb, "circuit_reference_ncut", "study.truncation_benchmark"),
        duffing_calibration_mode=mode,
        outputs=OutputConfig(figure=_require_str(outputs, "figure", "study.truncation_benchmark.outputs")),
    )


def _parse_cz_benchmark(study_payload: dict[str, Any]) -> CzBenchmarkConfig:
    cz = _require_dict(study_payload, "cz_benchmark", "study")
    outputs = _require_dict(cz, "outputs", "study.cz_benchmark")
    total_time_ns = _require_float(cz, "total_time_ns", "study.cz_benchmark")
    ramp_time_ns = _require_float(cz, "ramp_time_ns", "study.cz_benchmark")
    dt_ns = _require_float(cz, "dt_ns", "study.cz_benchmark")
    enable_hold_time_scan = _require_bool(cz, "enable_hold_time_scan", "study.cz_benchmark")
    scan_dt_ns = _require_float(cz, "scan_dt_ns", "study.cz_benchmark")
    scan_max_hold_ns = _require_float(cz, "scan_max_hold_ns", "study.cz_benchmark")
    scan_leakage_penalty = _require_float(cz, "scan_leakage_penalty", "study.cz_benchmark")

    if total_time_ns <= 0.0:
        raise ValueError("study.cz_benchmark.total_time_ns must be positive")
    if ramp_time_ns <= 0.0:
        raise ValueError("study.cz_benchmark.ramp_time_ns must be positive")
    if dt_ns <= 0.0:
        raise ValueError("study.cz_benchmark.dt_ns must be positive")
    if scan_dt_ns <= 0.0:
        raise ValueError("study.cz_benchmark.scan_dt_ns must be positive")
    if scan_max_hold_ns < 0.0:
        raise ValueError("study.cz_benchmark.scan_max_hold_ns must be >= 0")
    if scan_leakage_penalty < 0.0:
        raise ValueError("study.cz_benchmark.scan_leakage_penalty must be >= 0")
    if total_time_ns < 2.0 * ramp_time_ns:
        raise ValueError(
            "study.cz_benchmark.total_time_ns must be >= 2 * ramp_time_ns "
            "for a ramp-hold-ramp pulse"
        )

    return CzBenchmarkConfig(
        total_time_ns=float(total_time_ns),
        ramp_time_ns=float(ramp_time_ns),
        dt_ns=float(dt_ns),
        enable_hold_time_scan=bool(enable_hold_time_scan),
        scan_dt_ns=float(scan_dt_ns),
        scan_max_hold_ns=float(scan_max_hold_ns),
        scan_leakage_penalty=float(scan_leakage_penalty),
        outputs=OutputConfig(figure=_require_str(outputs, "figure", "study.cz_benchmark.outputs")),
    )


def _parse_leakage_flow_benchmark(study_payload: dict[str, Any]) -> LeakageFlowBenchmarkConfig:
    lf = _require_dict(study_payload, "leakage_flow_benchmark", "study")
    outputs = _require_dict(lf, "outputs", "study.leakage_flow_benchmark")
    total_time_ns = _require_float(lf, "total_time_ns", "study.leakage_flow_benchmark")
    ramp_time_ns = _require_float(lf, "ramp_time_ns", "study.leakage_flow_benchmark")
    dt_ns = _require_float(lf, "dt_ns", "study.leakage_flow_benchmark")
    population_min_average = _require_float(lf, "population_min_average", "study.leakage_flow_benchmark")
    transition_min_integrated_abs = _require_float(
        lf,
        "transition_min_integrated_abs",
        "study.leakage_flow_benchmark",
    )
    max_population_rows = _require_int(lf, "max_population_rows", "study.leakage_flow_benchmark")
    max_transition_rows = _require_int(lf, "max_transition_rows", "study.leakage_flow_benchmark")

    if total_time_ns <= 0.0:
        raise ValueError("study.leakage_flow_benchmark.total_time_ns must be positive")
    if ramp_time_ns <= 0.0:
        raise ValueError("study.leakage_flow_benchmark.ramp_time_ns must be positive")
    if dt_ns <= 0.0:
        raise ValueError("study.leakage_flow_benchmark.dt_ns must be positive")
    if population_min_average < 0.0:
        raise ValueError("study.leakage_flow_benchmark.population_min_average must be >= 0")
    if transition_min_integrated_abs < 0.0:
        raise ValueError("study.leakage_flow_benchmark.transition_min_integrated_abs must be >= 0")
    if max_population_rows < 1:
        raise ValueError("study.leakage_flow_benchmark.max_population_rows must be >= 1")
    if max_transition_rows < 1:
        raise ValueError("study.leakage_flow_benchmark.max_transition_rows must be >= 1")
    if total_time_ns < 2.0 * ramp_time_ns:
        raise ValueError(
            "study.leakage_flow_benchmark.total_time_ns must be >= 2 * ramp_time_ns "
            "for a ramp-hold-ramp pulse"
        )

    return LeakageFlowBenchmarkConfig(
        total_time_ns=float(total_time_ns),
        ramp_time_ns=float(ramp_time_ns),
        dt_ns=float(dt_ns),
        population_min_average=float(population_min_average),
        transition_min_integrated_abs=float(transition_min_integrated_abs),
        max_population_rows=int(max_population_rows),
        max_transition_rows=int(max_transition_rows),
        outputs=OutputConfig(figure=_require_str(outputs, "figure", "study.leakage_flow_benchmark.outputs")),
    )


def _parse_leakage_benchmark(study_payload: dict[str, Any]) -> LeakageBenchmarkConfig:
    lb = _require_dict(study_payload, "leakage_benchmark", "study")
    total_time_ns = _require_float(lb, "total_time_ns", "study.leakage_benchmark")
    ramp_time_ns = _require_float(lb, "ramp_time_ns", "study.leakage_benchmark")
    dt_ns = _require_float(lb, "dt_ns", "study.leakage_benchmark")
    top_destination_rows = _require_int(lb, "top_destination_rows", "study.leakage_benchmark")

    if total_time_ns <= 0.0:
        raise ValueError("study.leakage_benchmark.total_time_ns must be positive")
    if ramp_time_ns <= 0.0:
        raise ValueError("study.leakage_benchmark.ramp_time_ns must be positive")
    if dt_ns <= 0.0:
        raise ValueError("study.leakage_benchmark.dt_ns must be positive")
    if top_destination_rows < 1:
        raise ValueError("study.leakage_benchmark.top_destination_rows must be >= 1")
    if total_time_ns < 2.0 * ramp_time_ns:
        raise ValueError(
            "study.leakage_benchmark.total_time_ns must be >= 2 * ramp_time_ns "
            "for a ramp-hold-ramp pulse"
        )

    return LeakageBenchmarkConfig(
        total_time_ns=float(total_time_ns),
        ramp_time_ns=float(ramp_time_ns),
        dt_ns=float(dt_ns),
        top_destination_rows=int(top_destination_rows),
    )


def _parse_state_to_state_leakage_benchmark(study_payload: dict[str, Any]) -> StateToStateLeakageBenchmarkConfig:
    sb = _require_dict(study_payload, "state_to_state_leakage_benchmark", "study")
    total_time_ns = _require_float(sb, "total_time_ns", "study.state_to_state_leakage_benchmark")
    ramp_time_ns = _require_float(sb, "ramp_time_ns", "study.state_to_state_leakage_benchmark")
    dt_ns = _require_float(sb, "dt_ns", "study.state_to_state_leakage_benchmark")
    top_transition_rows = _require_int(sb, "top_transition_rows", "study.state_to_state_leakage_benchmark")

    if total_time_ns <= 0.0:
        raise ValueError("study.state_to_state_leakage_benchmark.total_time_ns must be positive")
    if ramp_time_ns <= 0.0:
        raise ValueError("study.state_to_state_leakage_benchmark.ramp_time_ns must be positive")
    if dt_ns <= 0.0:
        raise ValueError("study.state_to_state_leakage_benchmark.dt_ns must be positive")
    if top_transition_rows < 1:
        raise ValueError("study.state_to_state_leakage_benchmark.top_transition_rows must be >= 1")
    if total_time_ns < 2.0 * ramp_time_ns:
        raise ValueError(
            "study.state_to_state_leakage_benchmark.total_time_ns must be >= 2 * ramp_time_ns "
            "for a ramp-hold-ramp pulse"
        )

    return StateToStateLeakageBenchmarkConfig(
        total_time_ns=float(total_time_ns),
        ramp_time_ns=float(ramp_time_ns),
        dt_ns=float(dt_ns),
        top_transition_rows=int(top_transition_rows),
    )



def load_study_config(system_params_path: Path, study_params_path: Path) -> StudyConfig:
    system_payload = _load_json(system_params_path)
    raw_study_payload = _load_json(study_params_path)
    study_payload = _normalize_study_payload(raw_study_payload, study_params_path=study_params_path)
    static_config = _parse_static_benchmark(study_payload)
    return StudyConfig(
        system=_parse_system(system_payload),
        static_benchmark=static_config,
        cz_benchmark=_parse_cz_benchmark(study_payload),
        leakage_flow_benchmark=_parse_leakage_flow_benchmark(study_payload),
        truncation_benchmark=_parse_truncation_benchmark(study_payload),
        leakage_benchmark=_parse_leakage_benchmark(study_payload),
        state_to_state_leakage_benchmark=_parse_state_to_state_leakage_benchmark(study_payload),
    )



def build_flux_values(flux_sweep: FluxSweepConfig) -> np.ndarray:
    return np.linspace(float(flux_sweep.start), float(flux_sweep.stop), int(flux_sweep.num_points))
