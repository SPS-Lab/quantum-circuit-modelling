"""Combined leakage/population + transition-flow benchmark under a short pulse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from comparison.cz import (
    TWO_PI,
    _idle_flux_for_target,
    _pick_target_flux_from_static,
    _ramp_hold_ramp_flux_pulse,
)
from comparison.static import run_static_benchmark
from models import (
    build_circuit_model_stack,
    build_duffing_model_stack,
    computational_state_indices,
)
from study_config import StudyConfig


@dataclass(frozen=True)
class LeakageFlowBenchmarkResult:
    times_ns: np.ndarray
    pulse_flux_values: np.ndarray
    sweep_target: str
    idle_flux: float
    target_flux: float
    ramp_time_ns: float
    hold_time_ns: float
    dt_ns: float
    population_state_labels_11: np.ndarray
    duffing_population_state_amplitudes_11: np.ndarray
    circuit_population_state_amplitudes_11: np.ndarray
    transition_labels_11: np.ndarray
    duffing_transition_signed_currents_11: np.ndarray
    circuit_transition_signed_currents_11: np.ndarray
    duffing_leakage_11: np.ndarray
    circuit_leakage_11: np.ndarray
    summary: dict[str, float]


def _idx_qcq(n1: int, nc: int, n2: int, i: int, j: int, k: int) -> int:
    return int((i * nc + j) * n2 + k)


def _time_integral(values: np.ndarray, times_ns: np.ndarray) -> float:
    y = np.asarray(values, dtype=float).ravel()
    t = np.asarray(times_ns, dtype=float).ravel()
    if y.shape != t.shape:
        raise ValueError("values and times_ns must have matching shape")
    try:
        return float(np.trapezoid(y, x=t))
    except AttributeError:  # pragma: no cover - compatibility fallback
        return float(np.trapz(y, x=t))


def _simulate_state_trajectory(
    H_stack: np.ndarray,
    times_ns: np.ndarray,
    *,
    initial_index: int,
) -> np.ndarray:
    H = np.asarray(H_stack, dtype=complex)
    t = np.asarray(times_ns, dtype=float).ravel()
    if H.ndim != 3 or H.shape[1] != H.shape[2]:
        raise ValueError(f"H_stack must be (n_time, d, d), got {H.shape}")
    if H.shape[0] != t.size:
        raise ValueError("H_stack time axis must match times_ns")

    d = int(H.shape[1])
    idx0 = int(initial_index)
    if idx0 < 0 or idx0 >= d:
        raise ValueError(f"initial_index {idx0} out of bounds for dimension {d}")

    psi = np.zeros(d, dtype=complex)
    psi[idx0] = 1.0
    out = np.zeros((t.size, d), dtype=complex)
    out[0, :] = psi

    for m in range(t.size - 1):
        dt = float(t[m + 1] - t[m])
        if dt <= 0.0:
            raise ValueError("times_ns must be strictly increasing")
        U = expm((-1.0j * TWO_PI * dt) * H[m])
        psi = U @ psi
        out[m + 1, :] = psi

    return out


def _canonical_state_order_qcq(n1: int, nc: int, n2: int) -> tuple[np.ndarray, np.ndarray]:
    triples: list[tuple[int, int, int]] = []
    for i in range(int(n1)):
        for j in range(int(nc)):
            for k in range(int(n2)):
                triples.append((i, j, k))

    triples_sorted = sorted(triples, key=lambda t: (t[0] + t[1] + t[2], t[0], t[1], t[2]))
    idx = np.array([_idx_qcq(n1, nc, n2, i, j, k) for (i, j, k) in triples_sorted], dtype=int)
    labels = np.array([f"|{i},{j},{k}>" for (i, j, k) in triples_sorted], dtype=str)
    return idx, labels


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    text = np.asarray(labels, dtype=str).ravel()
    return np.asarray([s.encode("utf-8") for s in text], dtype="S")


def _decode_labels(labels: np.ndarray) -> list[str]:
    arr = np.asarray(labels).ravel()
    out: list[str] = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(bytes(x).decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _parse_state_label(label: str) -> tuple[int, int, int]:
    text = str(label).strip()
    if not (text.startswith("|") and text.endswith(">")):
        raise ValueError(f"Invalid state label {label!r}")
    parts = text[1:-1].split(",")
    if len(parts) != 3:
        raise ValueError(f"Invalid state label {label!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _state_sort_key(label: str) -> tuple[int, int, int, int]:
    i, j, k = _parse_state_label(label)
    return (i + j + k, i, j, k)


def _transition_sort_key(label: str) -> tuple[int, int, int, int, int, int, int, int]:
    text = str(label).strip()
    if "->" not in text:
        raise ValueError(f"Invalid transition label {label!r}")
    src, dst = text.split("->", 1)
    i1, j1, k1 = _parse_state_label(src.strip())
    i2, j2, k2 = _parse_state_label(dst.strip())
    return (i1 + j1 + k1, i1, j1, k1, i2 + j2 + k2, i2, j2, k2)


def _union_state_labels(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    merged = sorted(set(_decode_labels(labels_a)) | set(_decode_labels(labels_b)), key=_state_sort_key)
    return np.asarray(merged, dtype=str)


def _union_transition_labels(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    merged = sorted(set(_decode_labels(labels_a)) | set(_decode_labels(labels_b)), key=_transition_sort_key)
    return np.asarray(merged, dtype=str)


def _align_by_union_labels(
    union_labels: np.ndarray,
    model_labels: np.ndarray,
    model_traces: np.ndarray,
) -> np.ndarray:
    labels_u = _decode_labels(union_labels)
    labels_m = _decode_labels(model_labels)
    traces = np.asarray(model_traces)
    if traces.ndim != 2:
        raise ValueError(f"model_traces must be 2D (n_time, n_rows), got {traces.shape}")
    if len(labels_m) != traces.shape[1]:
        raise ValueError("model_labels size must match model_traces second axis")

    out = np.zeros((traces.shape[0], len(labels_u)), dtype=traces.dtype)
    label_to_col = {label: idx for idx, label in enumerate(labels_m)}
    for j, label in enumerate(labels_u):
        idx = label_to_col.get(label)
        if idx is not None:
            out[:, j] = traces[:, idx]
    return out


def _select_population_states(
    psi_ordered: np.ndarray,
    labels_ordered: np.ndarray,
    times_ns: np.ndarray,
    *,
    min_average_population: float,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    psi = np.asarray(psi_ordered, dtype=complex)
    labels = np.asarray(labels_ordered, dtype=str).ravel()
    t = np.asarray(times_ns, dtype=float).ravel()

    if psi.ndim != 2 or psi.shape[1] != labels.size:
        raise ValueError("psi_ordered must have shape (n_time, n_states) matching labels")
    if psi.shape[0] != t.size:
        raise ValueError("psi_ordered time axis must match times_ns")

    pop = np.abs(psi) ** 2
    total_time = float(max(1e-15, t[-1] - t[0]))
    try:
        integrated = np.trapezoid(pop, x=t, axis=0)
    except AttributeError:  # pragma: no cover - compatibility fallback
        integrated = np.trapz(pop, x=t, axis=0)
    avg = np.asarray(integrated / total_time, dtype=float)

    min_avg = float(max(0.0, min_average_population))
    keep = np.flatnonzero(avg >= min_avg)

    if keep.size == 0:
        keep = np.array([int(np.argmax(avg))], dtype=int)

    max_rows_eff = int(max(1, max_rows))
    if keep.size > max_rows_eff:
        keep_ranked = keep[np.argsort(-avg[keep])]
        keep = np.sort(keep_ranked[:max_rows_eff])

    return labels[keep], np.asarray(psi[:, keep], dtype=complex)


def _select_signed_pair_transitions(
    H_stack: np.ndarray,
    psi_ordered: np.ndarray,
    labels_ordered: np.ndarray,
    times_ns: np.ndarray,
    *,
    min_integrated_abs: float,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    H = np.asarray(H_stack, dtype=complex)
    psi = np.asarray(psi_ordered, dtype=complex)
    labels = np.asarray(labels_ordered, dtype=str).ravel()
    t = np.asarray(times_ns, dtype=float).ravel()

    if H.ndim != 3 or H.shape[1] != H.shape[2]:
        raise ValueError(f"H_stack must be (n_time, d, d), got {H.shape}")
    if psi.ndim != 2:
        raise ValueError("psi_ordered must be 2D")
    if H.shape[0] != t.size or psi.shape[0] != t.size:
        raise ValueError("H_stack and psi_ordered time axes must match times_ns")

    d = int(H.shape[1])
    if psi.shape[1] != d:
        raise ValueError("psi_ordered state axis must match H_stack dimension")
    if labels.size != d:
        raise ValueError("labels_ordered size must match H_stack dimension")

    iu, ju = np.triu_indices(d, k=1)
    n_pairs = int(iu.size)
    integrated_abs = np.zeros(n_pairs, dtype=float)

    prev_vals: np.ndarray | None = None
    for m in range(t.size):
        psi_m = psi[m]
        directed = np.asarray(
            2.0 * np.imag(np.conjugate(psi_m)[:, np.newaxis] * (TWO_PI * H[m]) * psi_m[np.newaxis, :]),
            dtype=float,
        )
        vals = np.asarray(directed[iu, ju], dtype=float)
        if prev_vals is not None:
            dt = float(t[m] - t[m - 1])
            integrated_abs += 0.5 * (np.abs(prev_vals) + np.abs(vals)) * dt
        prev_vals = vals

    thresh = float(max(0.0, min_integrated_abs))
    keep = np.flatnonzero(integrated_abs >= thresh)
    if keep.size == 0:
        keep = np.array([int(np.argmax(integrated_abs))], dtype=int)

    max_rows_eff = int(max(1, max_rows))
    if keep.size > max_rows_eff:
        keep = keep[np.argsort(-integrated_abs[keep])[:max_rows_eff]]

    order = np.argsort(-integrated_abs[keep])
    keep = keep[order]

    traces = np.zeros((t.size, keep.size), dtype=float)
    for m in range(t.size):
        psi_m = psi[m]
        directed = np.asarray(
            2.0 * np.imag(np.conjugate(psi_m)[:, np.newaxis] * (TWO_PI * H[m]) * psi_m[np.newaxis, :]),
            dtype=float,
        )
        vals = np.asarray(directed[iu, ju], dtype=float)
        traces[m, :] = vals[keep]

    labels_out = np.array([f"{labels[iu[k]]}->{labels[ju[k]]}" for k in keep], dtype=str)
    return labels_out, traces


def _leakage_from_computational(
    psi: np.ndarray,
    computational_idx: np.ndarray,
) -> np.ndarray:
    amp = np.asarray(psi, dtype=complex)
    idx = np.asarray(computational_idx, dtype=int).ravel()
    pop_comp = np.sum(np.abs(amp[:, idx]) ** 2, axis=1)
    return np.clip(1.0 - np.asarray(pop_comp, dtype=float), 0.0, 1.0)


def run_leakage_flow_benchmark(
    config: StudyConfig,
    *,
    ramp_time_ns: float,
    hold_time_ns: float,
    dt_ns: float,
    population_min_average: float,
    transition_min_integrated_abs: float,
    max_population_rows: int,
    max_transition_rows: int,
) -> LeakageFlowBenchmarkResult:
    static_result = run_static_benchmark(config)
    sweep_target = str(config.static_benchmark.flux_control.sweep_target)
    idle_flux = _idle_flux_for_target(config, sweep_target)
    target_flux = _pick_target_flux_from_static(static_result, idle_flux=idle_flux)

    times_ns, pulse_flux = _ramp_hold_ramp_flux_pulse(
        ramp_time_ns=float(ramp_time_ns),
        hold_time_ns=float(max(0.0, hold_time_ns)),
        dt_ns=float(dt_ns),
        idle_flux=float(idle_flux),
        target_flux=float(target_flux),
    )

    duffing_stack = build_duffing_model_stack(
        flux_values=pulse_flux,
        system_params=config.system,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        duffing_config=config.static_benchmark.duffing_model,
        sweep_target=sweep_target,
    ).hamiltonian_stack

    circuit_stack = build_circuit_model_stack(
        flux_values=pulse_flux,
        system_params=config.system,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        circuit_config=config.static_benchmark.circuit_model,
        sweep_target=sweep_target,
    ).hamiltonian_stack

    n_q_duf = int(config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit)
    n_c_duf = int(config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler)
    idx_duf_init = _idx_qcq(n_q_duf, n_c_duf, n_q_duf, 1, 0, 1)
    n_q1_cir = int(config.static_benchmark.circuit_model.hilbert_truncation.q1_truncated_dim)
    n_q2_cir = int(config.static_benchmark.circuit_model.hilbert_truncation.q2_truncated_dim)
    n_c_cir = int(config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim)
    idx_cir_init = _idx_qcq(n_q1_cir, n_c_cir, n_q2_cir, 1, 0, 1)

    psi_duf = _simulate_state_trajectory(duffing_stack, times_ns, initial_index=idx_duf_init)
    psi_cir = _simulate_state_trajectory(circuit_stack, times_ns, initial_index=idx_cir_init)

    idx_duf_ord, labels_duf_ord = _canonical_state_order_qcq(n_q_duf, n_c_duf, n_q_duf)
    idx_cir_ord, labels_cir_ord = _canonical_state_order_qcq(n_q1_cir, n_c_cir, n_q2_cir)

    H_duf_ord = np.asarray(duffing_stack[:, idx_duf_ord][:, :, idx_duf_ord], dtype=complex)
    H_cir_ord = np.asarray(circuit_stack[:, idx_cir_ord][:, :, idx_cir_ord], dtype=complex)

    psi_duf_ord = np.asarray(psi_duf[:, idx_duf_ord], dtype=complex)
    psi_cir_ord = np.asarray(psi_cir[:, idx_cir_ord], dtype=complex)

    pop_labels_duf_local, pop_amp_duf_local = _select_population_states(
        psi_duf_ord,
        labels_duf_ord,
        times_ns,
        min_average_population=population_min_average,
        max_rows=max_population_rows,
    )
    pop_labels_cir_local, pop_amp_cir_local = _select_population_states(
        psi_cir_ord,
        labels_cir_ord,
        times_ns,
        min_average_population=population_min_average,
        max_rows=max_population_rows,
    )
    pop_labels_union = _union_state_labels(pop_labels_duf_local, pop_labels_cir_local)
    pop_amp_duf = _align_by_union_labels(pop_labels_union, pop_labels_duf_local, pop_amp_duf_local)
    pop_amp_cir = _align_by_union_labels(pop_labels_union, pop_labels_cir_local, pop_amp_cir_local)

    tr_labels_duf_local, tr_curr_duf_local = _select_signed_pair_transitions(
        H_duf_ord,
        psi_duf_ord,
        labels_duf_ord,
        times_ns,
        min_integrated_abs=transition_min_integrated_abs,
        max_rows=max_transition_rows,
    )
    tr_labels_cir_local, tr_curr_cir_local = _select_signed_pair_transitions(
        H_cir_ord,
        psi_cir_ord,
        labels_cir_ord,
        times_ns,
        min_integrated_abs=transition_min_integrated_abs,
        max_rows=max_transition_rows,
    )
    tr_labels_union = _union_transition_labels(tr_labels_duf_local, tr_labels_cir_local)
    tr_curr_duf = _align_by_union_labels(tr_labels_union, tr_labels_duf_local, tr_curr_duf_local)
    tr_curr_cir = _align_by_union_labels(tr_labels_union, tr_labels_cir_local, tr_curr_cir_local)

    duf_comp_idx = computational_state_indices(n_q_duf, n_c_duf)
    cir_comp_idx = computational_state_indices(n_q1_cir, n_c_cir)
    duf_leakage = _leakage_from_computational(psi_duf, duf_comp_idx)
    cir_leakage = _leakage_from_computational(psi_cir, cir_comp_idx)

    summary = {
        "duffing_max_leakage_11": float(np.max(duf_leakage)),
        "circuit_max_leakage_11": float(np.max(cir_leakage)),
        "duffing_final_leakage_11": float(duf_leakage[-1]),
        "circuit_final_leakage_11": float(cir_leakage[-1]),
        "union_population_states_shown": float(pop_labels_union.size),
        "union_transitions_shown": float(tr_labels_union.size),
        "duffing_total_abs_transition_current": float(
            sum(_time_integral(np.abs(tr_curr_duf[:, i]), times_ns) for i in range(tr_curr_duf.shape[1]))
        ),
        "circuit_total_abs_transition_current": float(
            sum(_time_integral(np.abs(tr_curr_cir[:, i]), times_ns) for i in range(tr_curr_cir.shape[1]))
        ),
        "ramp_time_ns": float(ramp_time_ns),
        "hold_time_ns": float(hold_time_ns),
        "dt_ns": float(dt_ns),
    }

    return LeakageFlowBenchmarkResult(
        times_ns=np.asarray(times_ns, dtype=float),
        pulse_flux_values=np.asarray(pulse_flux, dtype=float),
        sweep_target=str(sweep_target),
        idle_flux=float(idle_flux),
        target_flux=float(target_flux),
        ramp_time_ns=float(ramp_time_ns),
        hold_time_ns=float(hold_time_ns),
        dt_ns=float(dt_ns),
        population_state_labels_11=_encode_labels(pop_labels_union),
        duffing_population_state_amplitudes_11=np.asarray(pop_amp_duf, dtype=complex),
        circuit_population_state_amplitudes_11=np.asarray(pop_amp_cir, dtype=complex),
        transition_labels_11=_encode_labels(tr_labels_union),
        duffing_transition_signed_currents_11=np.asarray(tr_curr_duf, dtype=float),
        circuit_transition_signed_currents_11=np.asarray(tr_curr_cir, dtype=float),
        duffing_leakage_11=np.asarray(duf_leakage, dtype=float),
        circuit_leakage_11=np.asarray(cir_leakage, dtype=float),
        summary=summary,
    )
