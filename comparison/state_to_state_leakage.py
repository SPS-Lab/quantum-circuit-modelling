"""State-to-state leakage-current benchmark under the calibrated CZ pulse."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from comparison.cz import CzBenchmarkResult, TWO_PI, run_cz_benchmark
from models import build_circuit_model_stack, build_duffing_model_stack
from study_config import StudyConfig


@dataclass(frozen=True)
class StateToStateLeakageBenchmarkResult:
    times_ns: np.ndarray
    pulse_flux_values: np.ndarray
    sweep_target: str
    idle_flux: float
    target_flux: float
    ramp_time_ns: float
    hold_time_ns: float
    dt_ns: float
    effective_leakage_11: np.ndarray
    duffing_leakage_11: np.ndarray
    circuit_leakage_11: np.ndarray
    duffing_max_transition_label_11: str
    circuit_max_transition_label_11: str
    duffing_comp_to_leak_currents_11: dict[str, np.ndarray] = field(default_factory=dict)
    circuit_comp_to_leak_currents_11: dict[str, np.ndarray] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)


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


def _computational_layout(n1: int, nc: int, n2: int) -> tuple[np.ndarray, list[str]]:
    if n1 < 2 or n2 < 2 or nc < 1:
        raise ValueError("Need n1>=2, n2>=2, nc>=1 for computational-subspace indexing")
    idx: list[int] = []
    labels: list[str] = []
    for i in (0, 1):
        for k in (0, 1):
            idx.append(_idx_qcq(n1, nc, n2, i, 0, k))
            labels.append(f"|{i},0,{k}>")
    return np.asarray(idx, dtype=int), labels


def _leakage_layout(n1: int, nc: int, n2: int) -> tuple[np.ndarray, list[str]]:
    comp_idx, _ = _computational_layout(n1, nc, n2)
    comp_set = set(int(x) for x in comp_idx)
    idx: list[int] = []
    labels: list[str] = []
    for i in range(n1):
        for j in range(nc):
            for k in range(n2):
                flat = _idx_qcq(n1, nc, n2, i, j, k)
                if flat in comp_set:
                    continue
                idx.append(flat)
                labels.append(f"|{i},{j},{k}>")
    return np.asarray(idx, dtype=int), labels


def _pair_currents_comp_to_leak_from_11_for_stack(
    *,
    stack: np.ndarray,
    times_ns: np.ndarray,
    n1: int,
    nc: int,
    n2: int,
) -> tuple[dict[str, np.ndarray], float, float, str]:
    t = np.asarray(times_ns, dtype=float).ravel()
    H = np.asarray(stack, dtype=complex)
    if H.ndim != 3 or H.shape[1] != H.shape[2]:
        raise ValueError(f"stack must be shape (n_time, d, d), got {H.shape}")
    if H.shape[0] != t.size:
        raise ValueError("stack time axis must match times_ns length")

    src_idx, src_labels = _computational_layout(n1, nc, n2)
    dst_idx, dst_labels = _leakage_layout(n1, nc, n2)
    if dst_idx.size == 0:
        return {}, 0.0, 0.0, ""

    d = int(H.shape[1])
    psi = np.zeros(d, dtype=complex)
    idx_11 = _idx_qcq(n1, nc, n2, 1, 0, 1)
    psi[idx_11] = 1.0

    currents = np.zeros((t.size, src_idx.size, dst_idx.size), dtype=float)

    src = src_idx[:, None]
    dst = dst_idx[None, :]

    def _record(m: int) -> None:
        H_block = H[m][src, dst]
        psi_src = np.conj(psi[src_idx])[:, None]
        psi_dst = psi[dst_idx][None, :]
        # H is in cycles/ns, so TWO_PI converts to angular units for dP/dt currents.
        directed = 2.0 * np.imag(psi_src * (TWO_PI * H_block) * psi_dst)
        currents[m, :, :] = np.clip(np.asarray(directed, dtype=float), 0.0, None)

    _record(0)
    for m in range(t.size - 1):
        dt = float(t[m + 1] - t[m])
        if dt <= 0.0:
            raise ValueError("times_ns must be strictly increasing")
        U = expm((-1.0j * TWO_PI * dt) * H[m])
        psi = U @ psi
        _record(m + 1)

    integrated = np.zeros((src_idx.size, dst_idx.size), dtype=float)
    for i in range(src_idx.size):
        for j in range(dst_idx.size):
            integrated[i, j] = _time_integral(currents[:, i, j], t)

    traces: dict[str, np.ndarray] = {}
    for i, src_label in enumerate(src_labels):
        for j, dst_label in enumerate(dst_labels):
            traces[f"{src_label}->{dst_label}"] = np.asarray(currents[:, i, j], dtype=float)

    flat = integrated.reshape(-1)
    if flat.size == 0:
        return traces, 0.0, 0.0, ""
    best_flat = int(np.argmax(flat))
    i_best = int(best_flat // dst_idx.size)
    j_best = int(best_flat % dst_idx.size)
    best_label = f"{src_labels[i_best]}->{dst_labels[j_best]}"
    best_value = float(integrated[i_best, j_best])
    total_value = float(np.sum(integrated))
    return traces, total_value, best_value, best_label


def _as_state_to_state_result(
    *,
    config: StudyConfig,
    cz_result: CzBenchmarkResult,
) -> StateToStateLeakageBenchmarkResult:
    duffing_stack = build_duffing_model_stack(
        flux_values=cz_result.pulse_flux_values,
        system_params=config.system,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        duffing_config=config.static_benchmark.duffing_model,
        sweep_target=cz_result.sweep_target,
    ).hamiltonian_stack
    duf_traces, duf_total, duf_best, duf_best_label = _pair_currents_comp_to_leak_from_11_for_stack(
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
    cir_traces, cir_total, cir_best, cir_best_label = _pair_currents_comp_to_leak_from_11_for_stack(
        stack=circuit_stack,
        times_ns=cz_result.times_ns,
        n1=config.static_benchmark.circuit_model.hilbert_truncation.q1_truncated_dim,
        nc=config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim,
        n2=config.static_benchmark.circuit_model.hilbert_truncation.q2_truncated_dim,
    )

    summary = {
        "duffing_total_integrated_comp_to_leak_current_11": float(duf_total),
        "circuit_total_integrated_comp_to_leak_current_11": float(cir_total),
        "duffing_max_integrated_transition_current_11": float(duf_best),
        "circuit_max_integrated_transition_current_11": float(cir_best),
        "ramp_time_ns": float(cz_result.ramp_time_ns),
        "hold_time_ns": float(cz_result.hold_time_ns),
        "dt_ns": float(cz_result.dt_ns),
    }

    return StateToStateLeakageBenchmarkResult(
        times_ns=np.asarray(cz_result.times_ns, dtype=float),
        pulse_flux_values=np.asarray(cz_result.pulse_flux_values, dtype=float),
        sweep_target=str(cz_result.sweep_target),
        idle_flux=float(cz_result.idle_flux),
        target_flux=float(cz_result.target_flux),
        ramp_time_ns=float(cz_result.ramp_time_ns),
        hold_time_ns=float(cz_result.hold_time_ns),
        dt_ns=float(cz_result.dt_ns),
        effective_leakage_11=np.asarray(cz_result.effective_leakage_11, dtype=float),
        duffing_leakage_11=np.asarray(cz_result.duffing_leakage_11, dtype=float),
        circuit_leakage_11=np.asarray(cz_result.circuit_leakage_11, dtype=float),
        duffing_max_transition_label_11=str(duf_best_label),
        circuit_max_transition_label_11=str(cir_best_label),
        duffing_comp_to_leak_currents_11=duf_traces,
        circuit_comp_to_leak_currents_11=cir_traces,
        summary=summary,
    )


def run_state_to_state_leakage_benchmark(
    config: StudyConfig,
    *,
    ramp_time_ns: float = 8.0,
    hold_time_ns: float | None = None,
    dt_ns: float = 1.0,
    enable_hold_time_scan: bool = True,
    scan_dt_ns: float = 2.0,
    scan_max_hold_ns: float = 300.0,
    scan_leakage_penalty: float = 0.25,
) -> StateToStateLeakageBenchmarkResult:
    """Run state-to-state leakage-current benchmark from |11> input."""
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
    return _as_state_to_state_result(config=config, cz_result=cz_result)
