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
FitBasis = Literal["single-harmonic", "magnitude-exchange-like"]
SweepTarget = Literal["coupler", "q0", "q1"]
DuffingCalibrationMode = Literal[
    "fixed",
    "analytic-per-flux",
    "per-flux",
    "fitted-static",
    "symbolic-fitted-static",
]
DriveQubit = Literal["q0"]


@dataclass(frozen=True)
class TransmonSystemParams:
    EJmax: float
    EC: float
    d: float
    flux: float
    ng: float
    ncut: int
    id_str: str


@dataclass(frozen=True)
class OscillatorSystemParams:
    E_osc: float
    kappa_over_2pi: float
    id_str: str


@dataclass(frozen=True)
class InteractionSystemParams:
    g_0c: float
    g_1c: float


@dataclass(frozen=True)
class SystemParams:
    q0: TransmonSystemParams
    q1: TransmonSystemParams
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
class SymbolicDuffingFitConfig:
    max_harmonics: int
    pointwise_max_nfev: int
    refinement_max_nfev: int
    regularization_weight: float


@dataclass(frozen=True)
class DuffingModelConfig:
    transmon_spectral_extraction: TransmonSpectralExtractionConfig
    hilbert_truncation: DuffingHilbertTruncationConfig
    coupler_anharmonicity: float
    calibration_mode: DuffingCalibrationMode
    symbolic_fit: SymbolicDuffingFitConfig | None


@dataclass(frozen=True)
class CircuitHilbertTruncationConfig:
    q0_truncated_dim: int
    q1_truncated_dim: int
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
class RxOutputConfig:
    populations_figure: str
    diagnostics_figure: str


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
class CircuitTruncationBenchmarkConfig:
    flux_values: tuple[float, ...]
    circuit_ncut_values: tuple[int, ...]
    circuit_truncation_values: tuple[tuple[int, int], ...]
    lowest_excited_levels_to_plot: int
    circuit_reference_ncut: int
    circuit_reference_qubit_truncated_dim: int
    circuit_reference_coupler_truncated_dim: int
    outputs: OutputConfig


@dataclass(frozen=True)
class DuffingTruncationBenchmarkConfig:
    flux_values: tuple[float, ...]
    duffing_ncut_values: tuple[int, ...]
    duffing_truncated_dim: int
    duffing_hilbert_truncation_values: tuple[tuple[int, int], ...]
    lowest_excited_levels_to_plot: int
    circuit_reference_ncut: int
    circuit_reference_qubit_truncated_dim: int
    circuit_reference_coupler_truncated_dim: int
    duffing_calibration_mode: DuffingCalibrationMode
    outputs: OutputConfig


@dataclass(frozen=True)
class RuntimeBenchmarkConfig:
    qubit_truncation_values: tuple[int, ...]
    duffing_calibration_mode: DuffingCalibrationMode
    repeats: int
    hold_time_ns: float | None
    outputs: OutputConfig


@dataclass(frozen=True)
class CzBenchmarkConfig:
    total_time_ns: float | None
    hold_time_ns: float | None
    ramp_time_ns: float
    dt_ns: float
    enable_hold_time_scan: bool
    scan_dt_ns: float
    scan_max_hold_ns: float
    scan_leakage_penalty: float
    outputs: OutputConfig


@dataclass(frozen=True)
class RxBenchmarkConfig:
    drive_qubit: DriveQubit
    drive_frequency: float
    drive_amplitude: float
    drive_phase_rad: float
    total_time_ns: float
    dt_ns: float
    rise_time_ns: float
    outputs: RxOutputConfig


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
    rx_benchmark: RxBenchmarkConfig
    leakage_flow_benchmark: LeakageFlowBenchmarkConfig
    circuit_truncation_benchmark: CircuitTruncationBenchmarkConfig
    duffing_truncation_benchmark: DuffingTruncationBenchmarkConfig
    runtime_benchmark: RuntimeBenchmarkConfig


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON at {path} must decode to an object")
    return payload


def _deep_merge_dict(
    *,
    base: dict[str, Any],
    update: dict[str, Any]
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in update.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(base=base_value, update=value)
        else:
            merged[key] = value
    return merged


_RUN_ALL_BENCHMARK_CATEGORY_ORDER: tuple[str, ...] = (
    "shared_static_cz_leakage_flow_truncation",
    "shared_static_and_cz",
    "shared_static_and_rx",
    "static_only",
    "truncation_only",
    "runtime_only",
    "cz_only",
    "rx_only",
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
        normalized = _deep_merge_dict(base=normalized, update=category_payload)

    unknown_categories = sorted(set(grouped.keys()) - set(_RUN_ALL_BENCHMARK_CATEGORY_ORDER))
    if unknown_categories:
        raise ValueError(
            "Unknown study.run_all_benchmark_params categories: "
            + ", ".join(unknown_categories)
        )
    return normalized


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


def _require_truncation_pair_list(
    parent: dict[str, Any],
    key: str,
    path: str,
) -> tuple[tuple[int, int], ...]:
    raw = _require_list(parent, key, path)
    pairs: list[tuple[int, int]] = []
    for idx, item in enumerate(raw):
        item_path = f"{path}.{key}[{idx}]"
        if not isinstance(item, dict):
            raise TypeError(f"{item_path} must be an object")
        if "qubit" not in item or "coupler" not in item:
            raise KeyError(f"{item_path} must contain keys 'qubit' and 'coupler'")
        qubit = int(item["qubit"])
        coupler = int(item["coupler"])
        if qubit < 1 or coupler < 1:
            raise ValueError(f"{item_path} values must be positive integers")
        pairs.append((qubit, coupler))
    if len(pairs) == 0:
        raise ValueError(f"{path}.{key} must be non-empty")
    return tuple(pairs)


def _parse_system(system_payload: dict[str, Any]) -> SystemParams:
    p = _require_dict(system_payload, "parameters", "system")

    q0p = _require_dict(p, "q0", "system.parameters")
    q1p = _require_dict(p, "q1", "system.parameters")
    cp = _require_dict(p, "c", "system.parameters")
    ip = _require_dict(p, "interactions", "system.parameters")

    q0 = TransmonSystemParams(
        EJmax=_require_float(q0p, "EJmax", "system.parameters.q0"),
        EC=_require_float(q0p, "EC", "system.parameters.q0"),
        d=_require_float(q0p, "d", "system.parameters.q0"),
        flux=_require_float(q0p, "flux", "system.parameters.q0"),
        ng=_require_float(q0p, "ng", "system.parameters.q0"),
        ncut=_require_int(q0p, "ncut", "system.parameters.q0"),
        id_str=_require_str(q0p, "id_str", "system.parameters.q0"),
    )
    q1 = TransmonSystemParams(
        EJmax=_require_float(q1p, "EJmax", "system.parameters.q1"),
        EC=_require_float(q1p, "EC", "system.parameters.q1"),
        d=_require_float(q1p, "d", "system.parameters.q1"),
        flux=_require_float(q1p, "flux", "system.parameters.q1"),
        ng=_require_float(q1p, "ng", "system.parameters.q1"),
        ncut=_require_int(q1p, "ncut", "system.parameters.q1"),
        id_str=_require_str(q1p, "id_str", "system.parameters.q1"),
    )
    c = OscillatorSystemParams(
        E_osc=_require_float(cp, "E_osc", "system.parameters.c"),
        kappa_over_2pi=_require_float(cp, "kappa_over_2pi", "system.parameters.c"),
        id_str=_require_str(cp, "id_str", "system.parameters.c"),
    )
    interactions = InteractionSystemParams(
        g_0c=_require_float(ip, "g_0c", "system.parameters.interactions"),
        g_1c=_require_float(ip, "g_1c", "system.parameters.interactions"),
    )
    return SystemParams(q0=q0, q1=q1, c=c, interactions=interactions)



def _parse_static_benchmark(study_payload: dict[str, Any]) -> StaticBenchmarkConfig:
    sb = _require_dict(study_payload, "static_benchmark", "study")

    flux = _require_dict(sb, "flux_sweep", "study.static_benchmark")
    flux_control = _require_dict(sb, "flux_control", "study.static_benchmark")
    coupler = _require_dict(sb, "coupler_frequency", "study.static_benchmark")
    dressed = _require_dict(sb, "dressed_subspace", "study.static_benchmark")
    duffing = _require_dict(sb, "duffing_model", "study.static_benchmark")
    d_spec = _require_dict(duffing, "transmon_spectral_extraction", "study.static_benchmark.duffing_model")
    d_hilbert = _require_dict(duffing, "hilbert_truncation", "study.static_benchmark.duffing_model")
    symbolic_fit_payload = duffing.get("symbolic_fit")
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
    if fit_basis not in ("single-harmonic", "magnitude-exchange-like"):
        raise ValueError(
            "study.static_benchmark.effective_model.fit_basis must be "
            "'single-harmonic' or 'magnitude-exchange-like'"
        )

    calibration_mode = _require_str(duffing, "calibration_mode", "study.static_benchmark.duffing_model").strip().lower()
    if calibration_mode not in (
        "fixed",
        "analytic-per-flux",
        "per-flux",
        "fitted-static",
        "symbolic-fitted-static",
    ):
        raise ValueError(
            "study.static_benchmark.duffing_model.calibration_mode must be "
            "'fixed', 'analytic-per-flux', 'per-flux', 'fitted-static', "
            "or 'symbolic-fitted-static'"
        )

    sweep_target_str = _require_str(flux_control, "sweep_target", "study.static_benchmark.flux_control")
    if sweep_target_str not in ("coupler", "q0", "q1"):
        raise ValueError("study.static_benchmark.flux_control.sweep_target must be 'coupler', 'q0', or 'q1'")
    sweep_target: SweepTarget = sweep_target_str

    q0_trunc = _require_int(c_hilbert, "q0_truncated_dim", "study.static_benchmark.circuit_model.hilbert_truncation")
    q1_trunc = _require_int(c_hilbert, "q1_truncated_dim", "study.static_benchmark.circuit_model.hilbert_truncation")
    if q0_trunc != q1_trunc:
        raise ValueError(
            "study.static_benchmark.circuit_model.hilbert_truncation requires "
            "q0_truncated_dim == q1_truncated_dim for computational-subspace indexing"
        )

    symbolic_fit: SymbolicDuffingFitConfig | None = None
    if symbolic_fit_payload is not None:
        if not isinstance(symbolic_fit_payload, dict):
            raise TypeError("study.static_benchmark.duffing_model.symbolic_fit must be an object")
        max_harmonics = _require_int(
            symbolic_fit_payload,
            "max_harmonics",
            "study.static_benchmark.duffing_model.symbolic_fit",
        )
        pointwise_max_nfev = _require_int(
            symbolic_fit_payload,
            "pointwise_max_nfev",
            "study.static_benchmark.duffing_model.symbolic_fit",
        )
        refinement_max_nfev = _require_int(
            symbolic_fit_payload,
            "refinement_max_nfev",
            "study.static_benchmark.duffing_model.symbolic_fit",
        )
        regularization_weight = _require_float(
            symbolic_fit_payload,
            "regularization_weight",
            "study.static_benchmark.duffing_model.symbolic_fit",
        )
        if max_harmonics < 1:
            raise ValueError("study.static_benchmark.duffing_model.symbolic_fit.max_harmonics must be >= 1")
        if pointwise_max_nfev < 1:
            raise ValueError(
                "study.static_benchmark.duffing_model.symbolic_fit.pointwise_max_nfev must be >= 1"
            )
        if refinement_max_nfev < 1:
            raise ValueError(
                "study.static_benchmark.duffing_model.symbolic_fit.refinement_max_nfev must be >= 1"
            )
        if regularization_weight < 0.0:
            raise ValueError(
                "study.static_benchmark.duffing_model.symbolic_fit.regularization_weight must be >= 0"
            )
        symbolic_fit = SymbolicDuffingFitConfig(
            max_harmonics=max_harmonics,
            pointwise_max_nfev=pointwise_max_nfev,
            refinement_max_nfev=refinement_max_nfev,
            regularization_weight=regularization_weight,
        )
    if calibration_mode == "symbolic-fitted-static" and symbolic_fit is None:
        raise KeyError(
            "Missing required key study.static_benchmark.duffing_model.symbolic_fit "
            "for calibration_mode='symbolic-fitted-static'"
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
            symbolic_fit=symbolic_fit,
        ),
        circuit_model=CircuitModelConfig(
            hilbert_truncation=CircuitHilbertTruncationConfig(
                q0_truncated_dim=q0_trunc,
                q1_truncated_dim=q1_trunc,
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


def _parse_circuit_truncation_benchmark(study_payload: dict[str, Any]) -> CircuitTruncationBenchmarkConfig:
    tb = _require_dict(study_payload, "circuit_truncation_benchmark", "study")

    flux_values_raw = _require_list(tb, "flux_values", "study.circuit_truncation_benchmark")
    flux_values = tuple(float(v) for v in flux_values_raw)
    if len(flux_values) != 5:
        raise ValueError("study.circuit_truncation_benchmark.flux_values must contain exactly 5 values")
    circuit_ncuts_raw = _require_list(tb, "circuit_ncut_values", "study.circuit_truncation_benchmark")
    circuit_ncuts = tuple(int(v) for v in circuit_ncuts_raw)
    if len(circuit_ncuts) == 0:
        raise ValueError("study.circuit_truncation_benchmark.circuit_ncut_values must be non-empty")
    if any(v < 1 for v in circuit_ncuts):
        raise ValueError("study.circuit_truncation_benchmark.circuit_ncut_values must contain positive integers")
    circuit_truncation_values = _require_truncation_pair_list(
        tb,
        "circuit_truncation_values",
        "study.circuit_truncation_benchmark",
    )
    n_levels_to_plot = _require_int(tb, "lowest_excited_levels_to_plot", "study.circuit_truncation_benchmark")
    if n_levels_to_plot < 1:
        raise ValueError("study.circuit_truncation_benchmark.lowest_excited_levels_to_plot must be >= 1")
    outputs = _require_dict(tb, "outputs", "study.circuit_truncation_benchmark")

    return CircuitTruncationBenchmarkConfig(
        flux_values=flux_values,
        circuit_ncut_values=circuit_ncuts,
        circuit_truncation_values=circuit_truncation_values,
        lowest_excited_levels_to_plot=n_levels_to_plot,
        circuit_reference_ncut=_require_int(tb, "circuit_reference_ncut", "study.circuit_truncation_benchmark"),
        circuit_reference_qubit_truncated_dim=_require_int(
            tb,
            "circuit_reference_qubit_truncated_dim",
            "study.circuit_truncation_benchmark",
        ),
        circuit_reference_coupler_truncated_dim=_require_int(
            tb,
            "circuit_reference_coupler_truncated_dim",
            "study.circuit_truncation_benchmark",
        ),
        outputs=OutputConfig(figure=_require_str(outputs, "figure", "study.circuit_truncation_benchmark.outputs")),
    )


def _parse_duffing_truncation_benchmark(study_payload: dict[str, Any]) -> DuffingTruncationBenchmarkConfig:
    tb = _require_dict(study_payload, "duffing_truncation_benchmark", "study")

    flux_values_raw = _require_list(tb, "flux_values", "study.duffing_truncation_benchmark")
    flux_values = tuple(float(v) for v in flux_values_raw)
    if len(flux_values) != 5:
        raise ValueError("study.duffing_truncation_benchmark.flux_values must contain exactly 5 values")
    ncuts_raw = _require_list(tb, "duffing_ncut_values", "study.duffing_truncation_benchmark")
    ncuts = tuple(int(v) for v in ncuts_raw)
    if len(ncuts) == 0:
        raise ValueError("study.duffing_truncation_benchmark.duffing_ncut_values must be non-empty")
    if any(v < 1 for v in ncuts):
        raise ValueError("study.duffing_truncation_benchmark.duffing_ncut_values must contain positive integers")
    trunc_dim = _require_int(tb, "duffing_truncated_dim", "study.duffing_truncation_benchmark")
    if trunc_dim < 3:
        raise ValueError("study.duffing_truncation_benchmark.duffing_truncated_dim must be >= 3")
    duffing_hilbert_truncation_values = _require_truncation_pair_list(
        tb,
        "duffing_hilbert_truncation_values",
        "study.duffing_truncation_benchmark",
    )
    n_levels_to_plot = _require_int(tb, "lowest_excited_levels_to_plot", "study.duffing_truncation_benchmark")
    if n_levels_to_plot < 1:
        raise ValueError("study.duffing_truncation_benchmark.lowest_excited_levels_to_plot must be >= 1")

    mode = _require_str(tb, "duffing_calibration_mode", "study.duffing_truncation_benchmark").strip().lower()
    if mode not in (
        "fixed",
        "analytic-per-flux",
        "per-flux",
        "fitted-static",
        "symbolic-fitted-static",
    ):
        raise ValueError(
            "study.duffing_truncation_benchmark.duffing_calibration_mode must be "
            "'fixed', 'analytic-per-flux', 'per-flux', 'fitted-static', "
            "or 'symbolic-fitted-static'"
        )
    outputs = _require_dict(tb, "outputs", "study.duffing_truncation_benchmark")

    return DuffingTruncationBenchmarkConfig(
        flux_values=flux_values,
        duffing_ncut_values=ncuts,
        duffing_truncated_dim=trunc_dim,
        duffing_hilbert_truncation_values=duffing_hilbert_truncation_values,
        lowest_excited_levels_to_plot=n_levels_to_plot,
        circuit_reference_ncut=_require_int(tb, "circuit_reference_ncut", "study.duffing_truncation_benchmark"),
        circuit_reference_qubit_truncated_dim=_require_int(
            tb,
            "circuit_reference_qubit_truncated_dim",
            "study.duffing_truncation_benchmark",
        ),
        circuit_reference_coupler_truncated_dim=_require_int(
            tb,
            "circuit_reference_coupler_truncated_dim",
            "study.duffing_truncation_benchmark",
        ),
        duffing_calibration_mode=mode,
        outputs=OutputConfig(figure=_require_str(outputs, "figure", "study.duffing_truncation_benchmark.outputs")),
    )


def _parse_cz_benchmark(study_payload: dict[str, Any]) -> CzBenchmarkConfig:
    cz = _require_dict(study_payload, "cz_benchmark", "study")
    outputs = _require_dict(cz, "outputs", "study.cz_benchmark")
    ramp_time_ns = _require_float(cz, "ramp_time_ns", "study.cz_benchmark")
    dt_ns = _require_float(cz, "dt_ns", "study.cz_benchmark")
    enable_hold_time_scan = _require_bool(cz, "enable_hold_time_scan", "study.cz_benchmark")
    scan_dt_ns = _require_float(cz, "scan_dt_ns", "study.cz_benchmark")
    scan_max_hold_ns = _require_float(cz, "scan_max_hold_ns", "study.cz_benchmark")
    scan_leakage_penalty = _require_float(cz, "scan_leakage_penalty", "study.cz_benchmark")
    has_total_time = "total_time_ns" in cz
    has_hold_time = "hold_time_ns" in cz

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

    total_time_ns: float | None = None
    hold_time_ns: float | None = None
    if has_total_time:
        total_time_ns = _require_float(cz, "total_time_ns", "study.cz_benchmark")
        if total_time_ns <= 0.0:
            raise ValueError("study.cz_benchmark.total_time_ns must be positive")
        if total_time_ns < 2.0 * ramp_time_ns:
            raise ValueError(
                "study.cz_benchmark.total_time_ns must be >= 2 * ramp_time_ns "
                "for a ramp-hold-ramp pulse"
            )
    if has_hold_time:
        hold_time_ns = _require_float(cz, "hold_time_ns", "study.cz_benchmark")
        if hold_time_ns < 0.0:
            raise ValueError("study.cz_benchmark.hold_time_ns must be >= 0")

    if has_total_time and has_hold_time:
        raise ValueError(
            "Ambiguous study.cz_benchmark configuration: provide only one of "
            "total_time_ns or hold_time_ns."
        )
    if enable_hold_time_scan and (has_total_time or has_hold_time):
        raise ValueError(
            "Ambiguous study.cz_benchmark configuration: enable_hold_time_scan=true "
            "cannot be combined with fixed-hold settings (total_time_ns or hold_time_ns)."
        )
    if (not enable_hold_time_scan) and (not has_total_time) and (not has_hold_time):
        raise ValueError(
            "study.cz_benchmark requires a fixed hold setting when "
            "enable_hold_time_scan=false: provide total_time_ns or hold_time_ns."
        )

    if (not enable_hold_time_scan) and (hold_time_ns is None) and (total_time_ns is not None):
        hold_time_ns = float(total_time_ns - 2.0 * ramp_time_ns)

    return CzBenchmarkConfig(
        total_time_ns=None if total_time_ns is None else float(total_time_ns),
        hold_time_ns=None if hold_time_ns is None else float(hold_time_ns),
        ramp_time_ns=float(ramp_time_ns),
        dt_ns=float(dt_ns),
        enable_hold_time_scan=bool(enable_hold_time_scan),
        scan_dt_ns=float(scan_dt_ns),
        scan_max_hold_ns=float(scan_max_hold_ns),
        scan_leakage_penalty=float(scan_leakage_penalty),
        outputs=OutputConfig(figure=_require_str(outputs, "figure", "study.cz_benchmark.outputs")),
    )


def _parse_runtime_benchmark(study_payload: dict[str, Any]) -> RuntimeBenchmarkConfig:
    rb = _require_dict(study_payload, "runtime_benchmark", "study")

    trunc_raw = _require_list(rb, "qubit_truncation_values", "study.runtime_benchmark")
    trunc_values = tuple(int(v) for v in trunc_raw)
    if len(trunc_values) == 0:
        raise ValueError("study.runtime_benchmark.qubit_truncation_values must be non-empty")
    if any(v < 2 for v in trunc_values):
        raise ValueError(
            "study.runtime_benchmark.qubit_truncation_values must contain integers >= 2"
        )

    calibration_mode = _require_str(rb, "duffing_calibration_mode", "study.runtime_benchmark").strip().lower()
    if calibration_mode not in (
        "fixed",
        "analytic-per-flux",
        "per-flux",
        "fitted-static",
        "symbolic-fitted-static",
    ):
        raise ValueError(
            "study.runtime_benchmark.duffing_calibration_mode must be "
            "'fixed', 'analytic-per-flux', 'per-flux', 'fitted-static', "
            "or 'symbolic-fitted-static'"
        )

    repeats = _require_int(rb, "repeats", "study.runtime_benchmark")
    if repeats < 1:
        raise ValueError("study.runtime_benchmark.repeats must be >= 1")

    hold_time_ns: float | None = None
    if "hold_time_ns" in rb:
        hold_time_ns = _require_float(rb, "hold_time_ns", "study.runtime_benchmark")
        if hold_time_ns < 0.0:
            raise ValueError("study.runtime_benchmark.hold_time_ns must be >= 0")

    outputs = _require_dict(rb, "outputs", "study.runtime_benchmark")
    return RuntimeBenchmarkConfig(
        qubit_truncation_values=trunc_values,
        duffing_calibration_mode=calibration_mode,
        repeats=repeats,
        hold_time_ns=None if hold_time_ns is None else float(hold_time_ns),
        outputs=OutputConfig(figure=_require_str(outputs, "figure", "study.runtime_benchmark.outputs")),
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


def _parse_rx_benchmark(study_payload: dict[str, Any]) -> RxBenchmarkConfig:
    rx = _require_dict(study_payload, "rx_benchmark", "study")
    outputs = _require_dict(rx, "outputs", "study.rx_benchmark")

    drive_qubit = _require_str(rx, "drive_qubit", "study.rx_benchmark")
    if drive_qubit != "q0":
        raise ValueError("study.rx_benchmark.drive_qubit must be 'q0'")
    drive_qubit_lit: DriveQubit = drive_qubit

    drive_frequency = _require_float(rx, "drive_frequency", "study.rx_benchmark")
    drive_amplitude = _require_float(rx, "drive_amplitude", "study.rx_benchmark")
    drive_phase_rad = _require_float(rx, "drive_phase_rad", "study.rx_benchmark")
    total_time_ns = _require_float(rx, "total_time_ns", "study.rx_benchmark")
    dt_ns = _require_float(rx, "dt_ns", "study.rx_benchmark")
    rise_time_ns = _require_float(rx, "rise_time_ns", "study.rx_benchmark")

    if drive_frequency <= 0.0:
        raise ValueError("study.rx_benchmark.drive_frequency must be positive")
    if drive_amplitude < 0.0:
        raise ValueError("study.rx_benchmark.drive_amplitude must be >= 0")
    if total_time_ns <= 0.0:
        raise ValueError("study.rx_benchmark.total_time_ns must be positive")
    if dt_ns <= 0.0:
        raise ValueError("study.rx_benchmark.dt_ns must be positive")
    if rise_time_ns <= 0.0:
        raise ValueError("study.rx_benchmark.rise_time_ns must be positive")
    if total_time_ns < 2.0 * rise_time_ns:
        raise ValueError("study.rx_benchmark.total_time_ns must be >= 2 * rise_time_ns")

    return RxBenchmarkConfig(
        drive_qubit=drive_qubit_lit,
        drive_frequency=float(drive_frequency),
        drive_amplitude=float(drive_amplitude),
        drive_phase_rad=float(drive_phase_rad),
        total_time_ns=float(total_time_ns),
        dt_ns=float(dt_ns),
        rise_time_ns=float(rise_time_ns),
        outputs=RxOutputConfig(
            populations_figure=_require_str(outputs, "populations_figure", "study.rx_benchmark.outputs"),
            diagnostics_figure=_require_str(outputs, "diagnostics_figure", "study.rx_benchmark.outputs"),
        ),
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



def load_study_config(
    *,
    system_params_path: Path,
    study_params_path: Path
) -> StudyConfig:
    system_payload = _load_json(system_params_path)
    raw_study_payload = _load_json(study_params_path)
    study_payload = _flatten_run_all_benchmark_params(raw_study_payload)
    static_config = _parse_static_benchmark(study_payload)
    return StudyConfig(
        system=_parse_system(system_payload),
        static_benchmark=static_config,
        cz_benchmark=_parse_cz_benchmark(study_payload),
        rx_benchmark=_parse_rx_benchmark(study_payload),
        leakage_flow_benchmark=_parse_leakage_flow_benchmark(study_payload),
        circuit_truncation_benchmark=_parse_circuit_truncation_benchmark(study_payload),
        duffing_truncation_benchmark=_parse_duffing_truncation_benchmark(study_payload),
        runtime_benchmark=_parse_runtime_benchmark(study_payload),
    )



def build_flux_values(flux_sweep: FluxSweepConfig) -> np.ndarray:
    return np.linspace(float(flux_sweep.start), float(flux_sweep.stop), int(flux_sweep.num_points))
