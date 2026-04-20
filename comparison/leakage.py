"""Leakage-focused benchmark under the same calibrated pulse as CZ."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from comparison.cz import CzBenchmarkResult, run_cz_benchmark
from study_config import StudyConfig


@dataclass(frozen=True)
class LeakageBenchmarkResult:
    times_ns: np.ndarray
    pulse_flux_values: np.ndarray
    sweep_target: str
    idle_flux: float
    target_flux: float
    ramp_time_ns: float
    hold_time_ns: float
    dt_ns: float
    effective_populations_11: np.ndarray
    duffing_populations_11: np.ndarray
    circuit_populations_11: np.ndarray
    effective_leakage_11: np.ndarray
    duffing_leakage_11: np.ndarray
    circuit_leakage_11: np.ndarray
    effective_intermediate_population_11: np.ndarray
    duffing_intermediate_population_11: np.ndarray
    circuit_intermediate_population_11: np.ndarray
    summary: dict[str, float]


def _as_leakage_result(cz_result: CzBenchmarkResult) -> LeakageBenchmarkResult:
    eff_rmse = float(np.sqrt(np.mean((cz_result.effective_leakage_11 - cz_result.circuit_leakage_11) ** 2)))
    duf_rmse = float(np.sqrt(np.mean((cz_result.duffing_leakage_11 - cz_result.circuit_leakage_11) ** 2)))

    summary = {
        "effective_max_leakage_11": float(np.max(cz_result.effective_leakage_11)),
        "duffing_max_leakage_11": float(np.max(cz_result.duffing_leakage_11)),
        "circuit_max_leakage_11": float(np.max(cz_result.circuit_leakage_11)),
        "effective_final_leakage_11": float(cz_result.effective_leakage_11[-1]),
        "duffing_final_leakage_11": float(cz_result.duffing_leakage_11[-1]),
        "circuit_final_leakage_11": float(cz_result.circuit_leakage_11[-1]),
        "effective_leakage_rmse_vs_circuit": eff_rmse,
        "duffing_leakage_rmse_vs_circuit": duf_rmse,
        "effective_max_intermediate_11": float(np.max(cz_result.effective_intermediate_population_11)),
        "duffing_max_intermediate_11": float(np.max(cz_result.duffing_intermediate_population_11)),
        "circuit_max_intermediate_11": float(np.max(cz_result.circuit_intermediate_population_11)),
        "ramp_time_ns": float(cz_result.ramp_time_ns),
        "hold_time_ns": float(cz_result.hold_time_ns),
        "dt_ns": float(cz_result.dt_ns),
    }

    return LeakageBenchmarkResult(
        times_ns=np.asarray(cz_result.times_ns, dtype=float),
        pulse_flux_values=np.asarray(cz_result.pulse_flux_values, dtype=float),
        sweep_target=str(cz_result.sweep_target),
        idle_flux=float(cz_result.idle_flux),
        target_flux=float(cz_result.target_flux),
        ramp_time_ns=float(cz_result.ramp_time_ns),
        hold_time_ns=float(cz_result.hold_time_ns),
        dt_ns=float(cz_result.dt_ns),
        effective_populations_11=np.asarray(cz_result.effective_populations_11, dtype=float),
        duffing_populations_11=np.asarray(cz_result.duffing_populations_11, dtype=float),
        circuit_populations_11=np.asarray(cz_result.circuit_populations_11, dtype=float),
        effective_leakage_11=np.asarray(cz_result.effective_leakage_11, dtype=float),
        duffing_leakage_11=np.asarray(cz_result.duffing_leakage_11, dtype=float),
        circuit_leakage_11=np.asarray(cz_result.circuit_leakage_11, dtype=float),
        effective_intermediate_population_11=np.asarray(cz_result.effective_intermediate_population_11, dtype=float),
        duffing_intermediate_population_11=np.asarray(cz_result.duffing_intermediate_population_11, dtype=float),
        circuit_intermediate_population_11=np.asarray(cz_result.circuit_intermediate_population_11, dtype=float),
        summary=summary,
    )


def run_leakage_benchmark(
    config: StudyConfig,
    *,
    ramp_time_ns: float = 8.0,
    hold_time_ns: float | None = None,
    dt_ns: float = 1.0,
    enable_hold_time_scan: bool = True,
    scan_dt_ns: float = 2.0,
    scan_max_hold_ns: float = 300.0,
    scan_leakage_penalty: float = 0.25,
) -> LeakageBenchmarkResult:
    """Run leakage benchmark from |11> under the calibrated CZ pulse."""
    cz_result = run_cz_benchmark(
        config,
        ramp_time_ns=ramp_time_ns,
        hold_time_ns=hold_time_ns,
        dt_ns=dt_ns,
        enable_hold_time_scan=enable_hold_time_scan,
        scan_dt_ns=scan_dt_ns,
        scan_max_hold_ns=scan_max_hold_ns,
        scan_leakage_penalty=scan_leakage_penalty,
    )
    return _as_leakage_result(cz_result)
