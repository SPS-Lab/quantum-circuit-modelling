"""Leakage-focused benchmark under the same calibrated pulse as CZ."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from comparison.cz import CzBenchmarkResult, TWO_PI, run_cz_benchmark
from models import build_circuit_model_stack, build_duffing_model_stack
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
    duffing_state_110_11: np.ndarray
    circuit_coupler_excited_11: np.ndarray
    circuit_state_110_11: np.ndarray
    summary: dict[str, float]
    duffing_leakage_destination_populations_11: dict[str, np.ndarray] = field(default_factory=dict)
    circuit_leakage_destination_populations_11: dict[str, np.ndarray] = field(default_factory=dict)


def _idx_qcq(n1: int, nc: int, n2: int, i: int, j: int, k: int) -> int:
    return int((i * nc + j) * n2 + k)

def _time_integral(values: np.ndarray, times_ns: np.ndarray) -> float:
    y = np.asarray(values, dtype=float).ravel()
    t = np.asarray(times_ns, dtype=float).ravel()
    if y.shape != t.shape:
        raise ValueError("values and times_ns must have the same shape")
    try:
        return float(np.trapezoid(y, x=t))
    except AttributeError:  # pragma: no cover - compatibility fallback
        return float(np.trapz(y, x=t))


def _fraction_of_time_integrated_leakage_to_state(
    *,
    state_population: np.ndarray,
    total_leakage: np.ndarray,
    times_ns: np.ndarray,
) -> tuple[float, float, float]:
    state = np.clip(np.asarray(state_population, dtype=float).ravel(), 0.0, None)
    leak = np.clip(np.asarray(total_leakage, dtype=float).ravel(), 0.0, None)
    t = np.asarray(times_ns, dtype=float).ravel()
    state_area = _time_integral(state, t)
    leak_area = _time_integral(leak, t)
    frac = 0.0 if leak_area <= 0.0 else state_area / leak_area
    return state_area, leak_area, float(frac)


def _computational_indices_from_q2c0q1(n1: int, nc: int, n2: int) -> set[int]:
    comp: set[int] = set()
    for i in (0, 1):
        for k in (0, 1):
            if i < n1 and k < n2 and 0 < nc:
                comp.add(_idx_qcq(n1, nc, n2, i, 0, k))
    return comp


def _leakage_destination_layout(n1: int, nc: int, n2: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    comp = _computational_indices_from_q2c0q1(n1, nc, n2)
    idx_list: list[int] = []
    labels: list[str] = []
    triples: list[tuple[int, int, int]] = []
    for i in range(n1):
        for j in range(nc):
            for k in range(n2):
                flat = _idx_qcq(n1, nc, n2, i, j, k)
                if flat in comp:
                    continue
                idx_list.append(flat)
                labels.append(f"|{k},{j},{i}>")
                triples.append((i, j, k))
    return np.asarray(idx_list, dtype=int), labels, np.asarray(triples, dtype=int)


def _all_leakage_destinations_from_11_for_stack(
    *,
    stack: np.ndarray,
    times_ns: np.ndarray,
    n1: int,
    nc: int,
    n2: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if n1 < 2 or n2 < 2 or nc < 1:
        raise ValueError("Need at least qubit dimensions >=2 and coupler dimension >=1 for |11> initialization")

    t = np.asarray(times_ns, dtype=float).ravel()
    H = np.asarray(stack, dtype=complex)
    if H.shape[0] != t.size:
        raise ValueError("stack time axis must match times_ns length")
    if H.shape[1] != H.shape[2]:
        raise ValueError("stack must contain square Hamiltonians")
    d = int(H.shape[1])

    destination_idx, destination_labels, destination_triples = _leakage_destination_layout(n1, nc, n2)
    destination_pop = np.zeros((t.size, destination_idx.size), dtype=float)
    coupler_excited = np.zeros(t.size, dtype=float)
    state_110 = np.zeros(t.size, dtype=float)

    coupler_excited_mask = destination_triples[:, 1] > 0 if destination_triples.size > 0 else np.zeros(0, dtype=bool)
    has_110 = nc >= 2
    idx_110 = _idx_qcq(n1, nc, n2, 0, 1, 1) if has_110 else -1

    psi = np.zeros(d, dtype=complex)
    psi[_idx_qcq(n1, nc, n2, 1, 0, 1)] = 1.0

    def _record(m: int) -> None:
        if destination_idx.size > 0:
            pop = np.abs(psi[destination_idx]) ** 2
            destination_pop[m, :] = np.asarray(pop, dtype=float)
            coupler_excited[m] = float(np.sum(pop[coupler_excited_mask]))
        if has_110:
            state_110[m] = float(np.abs(psi[idx_110]) ** 2)

    _record(0)
    for m in range(t.size - 1):
        dt = float(t[m + 1] - t[m])
        if dt <= 0.0:
            raise ValueError("times_ns must be strictly increasing")
        U = expm((-1.0j * TWO_PI * dt) * H[m])
        psi = U @ psi
        _record(m + 1)

    destinations = {
        label: np.asarray(destination_pop[:, col], dtype=float)
        for col, label in enumerate(destination_labels)
    }
    return destinations, coupler_excited, state_110


def _as_leakage_result(
    *,
    config: StudyConfig,
    cz_result: CzBenchmarkResult,
) -> LeakageBenchmarkResult:
    eff_rmse = float(np.sqrt(np.mean((cz_result.effective_leakage_11 - cz_result.circuit_leakage_11) ** 2)))
    duf_rmse = float(np.sqrt(np.mean((cz_result.duffing_leakage_11 - cz_result.circuit_leakage_11) ** 2)))

    duffing_stack = build_duffing_model_stack(
        flux_values=cz_result.pulse_flux_values,
        system_params=config.system,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        duffing_config=config.static_benchmark.duffing_model,
        sweep_target=cz_result.sweep_target,
    ).hamiltonian_stack
    duffing_destinations, _, p_duffing_110 = _all_leakage_destinations_from_11_for_stack(
        stack=duffing_stack,
        times_ns=cz_result.times_ns,
        n1=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit,
        nc=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler,
        n2=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit,
    )

    circuit_stack = build_circuit_model_stack(
        flux_values=cz_result.pulse_flux_values,
        system_params=config.system,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        circuit_config=config.static_benchmark.circuit_model,
        sweep_target=cz_result.sweep_target,
    ).hamiltonian_stack
    circuit_destinations, p_coupler_exc, p_state_110 = _all_leakage_destinations_from_11_for_stack(
        stack=circuit_stack,
        times_ns=cz_result.times_ns,
        n1=config.static_benchmark.circuit_model.hilbert_truncation.q1_truncated_dim,
        nc=config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim,
        n2=config.static_benchmark.circuit_model.hilbert_truncation.q2_truncated_dim,
    )

    duf_110_area, duf_leak_area, duf_110_fraction = _fraction_of_time_integrated_leakage_to_state(
        state_population=p_duffing_110,
        total_leakage=cz_result.duffing_leakage_11,
        times_ns=cz_result.times_ns,
    )
    cir_110_area, cir_leak_area, cir_110_fraction = _fraction_of_time_integrated_leakage_to_state(
        state_population=p_state_110,
        total_leakage=cz_result.circuit_leakage_11,
        times_ns=cz_result.times_ns,
    )

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
        "duffing_max_state_110_11": float(np.max(p_duffing_110)),
        "circuit_max_coupler_excited_11": float(np.max(p_coupler_exc)),
        "circuit_max_state_110_11": float(np.max(p_state_110)),
        "duffing_time_integrated_state_110_11_ns": duf_110_area,
        "duffing_time_integrated_total_leakage_11_ns": duf_leak_area,
        "duffing_fraction_of_time_integrated_leakage_to_state_110_11": duf_110_fraction,
        "circuit_time_integrated_state_110_11_ns": cir_110_area,
        "circuit_time_integrated_total_leakage_11_ns": cir_leak_area,
        "circuit_fraction_of_time_integrated_leakage_to_state_110_11": cir_110_fraction,
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
        duffing_state_110_11=np.asarray(p_duffing_110, dtype=float),
        circuit_coupler_excited_11=np.asarray(p_coupler_exc, dtype=float),
        circuit_state_110_11=np.asarray(p_state_110, dtype=float),
        summary=summary,
        duffing_leakage_destination_populations_11=duffing_destinations,
        circuit_leakage_destination_populations_11=circuit_destinations,
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
    return _as_leakage_result(config=config, cz_result=cz_result)
