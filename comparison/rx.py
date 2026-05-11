"""Driven single-qubit RX benchmark under a shared rotating-frame/RWA pulse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from comparison.cz import TWO_PI
from models import (
    build_circuit_model_stack,
    build_dressed_effective_computational_stack,
    build_duffing_model_stack,
    build_duffing_model_stack_from_parameters,
    build_effective_hamiltonian_stack,
    computational_state_indices,
    extract_model1_parameters_from_4x4_stack,
    fit_duffing_mode_parameters_to_reference,
    is_reference_calibrated_duffing_mode,
)
from study_config import StudyConfig
from toolkit.helpers import I2, destroy


@dataclass(frozen=True)
class RxBenchmarkResult:
    times_ns: np.ndarray
    pulse_envelope: np.ndarray
    drive_qubit: str
    drive_frequency: float
    drive_amplitude: float
    drive_phase_rad: float
    total_time_ns: float
    dt_ns: float
    rise_time_ns: float
    effective_computational_amplitudes: np.ndarray
    duffing_computational_amplitudes: np.ndarray
    circuit_computational_amplitudes: np.ndarray
    effective_pop_00_to_01: np.ndarray
    duffing_pop_00_to_01: np.ndarray
    circuit_pop_00_to_01: np.ndarray
    effective_pop_10_to_11: np.ndarray
    duffing_pop_10_to_11: np.ndarray
    circuit_pop_10_to_11: np.ndarray
    effective_leakage_from_00: np.ndarray
    duffing_leakage_from_00: np.ndarray
    circuit_leakage_from_00: np.ndarray
    effective_leakage_from_10: np.ndarray
    duffing_leakage_from_10: np.ndarray
    circuit_leakage_from_10: np.ndarray
    effective_spectator_population_delta: np.ndarray
    duffing_spectator_population_delta: np.ndarray
    circuit_spectator_population_delta: np.ndarray
    summary: dict[str, float]


def _time_grid(
    *,
    total_time_ns: float,
    dt_ns: float
) -> np.ndarray:
    total = float(total_time_ns)
    dt = float(dt_ns)
    if total <= 0.0:
        raise ValueError("total_time_ns must be positive")
    if dt <= 0.0:
        raise ValueError("dt_ns must be positive")

    grid = np.arange(0.0, total, dt, dtype=float)
    if grid.size == 0:
        grid = np.array([0.0], dtype=float)
    if abs(grid[-1] - total) > 1e-12:
        grid = np.append(grid, total)
    else:
        grid[-1] = total
    if grid.size < 2:
        grid = np.array([0.0, total], dtype=float)
    return grid


def _cosine_edge_envelope(*, total_time_ns: float, rise_time_ns: float, dt_ns: float) -> tuple[np.ndarray, np.ndarray]:
    total = float(total_time_ns)
    rise = float(rise_time_ns)
    if rise <= 0.0:
        raise ValueError("rise_time_ns must be positive")
    if total < 2.0 * rise:
        raise ValueError("total_time_ns must be >= 2 * rise_time_ns")

    times = _time_grid(total_time_ns=total, dt_ns=dt_ns)
    envelope = np.zeros_like(times, dtype=float)
    flat_stop = total - rise
    for k, t in enumerate(times):
        if t <= rise:
            u = t / rise
            envelope[k] = 0.5 * (1.0 - np.cos(np.pi * u))
        elif t >= flat_stop:
            u = (total - t) / rise
            envelope[k] = 0.5 * (1.0 - np.cos(np.pi * u))
        else:
            envelope[k] = 1.0
    return times, envelope


def _simulate_computational_basis(
    H_stack: np.ndarray,
    *,
    times_ns: np.ndarray,
    computational_indices: np.ndarray,
) -> np.ndarray:
    H = np.asarray(H_stack, dtype=complex)
    t = np.asarray(times_ns, dtype=float).ravel()
    comp_idx = np.asarray(computational_indices, dtype=int).ravel()

    n_time, d, d2 = H.shape
    if d != d2:
        raise ValueError(f"H_stack must be square, got {H.shape}")
    if n_time != t.size:
        raise ValueError(f"H_stack first axis ({n_time}) must match time size ({t.size})")
    if comp_idx.size != 4:
        raise ValueError("Need exactly 4 computational indices")

    states = np.zeros((d, 4), dtype=complex)
    for j in range(4):
        states[int(comp_idx[j]), j] = 1.0

    comp_amp = np.zeros((n_time, 4, 4), dtype=complex)
    comp_amp[0] = states[comp_idx, :]
    for k in range(n_time - 1):
        dt = float(t[k + 1] - t[k])
        if dt <= 0.0:
            raise ValueError("times_ns must be strictly increasing")
        U = expm((-1.0j * TWO_PI * dt) * H[k])
        states = U @ states
        comp_amp[k + 1] = states[comp_idx, :]
    return comp_amp


def _population_trace(comp_amp: np.ndarray, *, row: int, col: int) -> np.ndarray:
    amp = np.asarray(comp_amp, dtype=complex)
    return np.asarray(np.abs(amp[:, row, col]) ** 2, dtype=float)


def _leakage_trace(comp_amp: np.ndarray, *, col: int) -> np.ndarray:
    amp = np.asarray(comp_amp, dtype=complex)
    leakage = 1.0 - np.sum(np.abs(amp[:, :, col]) ** 2, axis=1)
    return np.clip(np.asarray(np.real(leakage), dtype=float), 0.0, 1.0)


def _effective_total_excitation_operator() -> np.ndarray:
    n = np.diag([0.0, 1.0]).astype(complex)
    return np.kron(n, I2) + np.kron(I2, n)


def _effective_q0_lowering_operator() -> np.ndarray:
    sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    return np.kron(I2, sm)


def _three_mode_q0_lowering_operator(*, nlevels_qubit: int, nlevels_coupler: int) -> np.ndarray:
    id_q = np.eye(int(nlevels_qubit), dtype=complex)
    id_c = np.eye(int(nlevels_coupler), dtype=complex)
    return np.kron(np.kron(id_q, id_c), destroy(int(nlevels_qubit)))


def _three_mode_total_excitation_operator(*, nlevels_qubit: int, nlevels_coupler: int) -> np.ndarray:
    a_q = destroy(int(nlevels_qubit))
    a_c = destroy(int(nlevels_coupler))
    adag_q = a_q.conj().T
    adag_c = a_c.conj().T
    n_q = adag_q @ a_q
    n_c = adag_c @ a_c
    id_q = np.eye(int(nlevels_qubit), dtype=complex)
    id_c = np.eye(int(nlevels_coupler), dtype=complex)
    return (
        np.kron(np.kron(id_q, id_c), n_q)
        + np.kron(np.kron(id_q, n_c), id_q)
        + np.kron(np.kron(n_q, id_c), id_q)
    )


def _build_circuit_idle_components(config: StudyConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import scqubits as scq
    except Exception as exc:  # pragma: no cover - import guard only
        raise ImportError("scqubits import failed while building circuit RX benchmark") from exc

    q0_trunc = int(config.static_benchmark.circuit_model.hilbert_truncation.q0_truncated_dim)
    q1_trunc = int(config.static_benchmark.circuit_model.hilbert_truncation.q1_truncated_dim)
    c_trunc = int(config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim)

    q0 = scq.TunableTransmon(
        EJmax=float(config.system.q0.EJmax),
        EC=float(config.system.q0.EC),
        d=float(config.system.q0.d),
        flux=float(config.system.q0.flux),
        ng=float(config.system.q0.ng),
        ncut=int(config.system.q0.ncut),
        truncated_dim=q0_trunc,
        id_str=str(config.system.q0.id_str),
    )
    q1 = scq.TunableTransmon(
        EJmax=float(config.system.q1.EJmax),
        EC=float(config.system.q1.EC),
        d=float(config.system.q1.d),
        flux=float(config.system.q1.flux),
        ng=float(config.system.q1.ng),
        ncut=int(config.system.q1.ncut),
        truncated_dim=q1_trunc,
        id_str=str(config.system.q1.id_str),
    )
    c = scq.Oscillator(
        E_osc=float(config.static_benchmark.coupler_frequency.wc0),
        truncated_dim=c_trunc,
        id_str=str(config.system.c.id_str),
    )
    def _to_matrix(operator: object) -> np.ndarray:
        return np.asarray(operator.full(), dtype=complex) if hasattr(operator, "full") else np.asarray(operator, dtype=complex)

    hilbertspace = scq.HilbertSpace([q1, c, q0])
    x_c = c.creation_operator() + c.annihilation_operator()
    hilbertspace.add_interaction(
        check_validity=bool(config.static_benchmark.circuit_model.interaction_validity_check),
        g=float(config.system.interactions.g_0c),
        op1=(q0.n_operator(), q0),
        op2=(x_c, c),
    )
    hilbertspace.add_interaction(
        check_validity=bool(config.static_benchmark.circuit_model.interaction_validity_check),
        g=float(config.system.interactions.g_1c),
        op1=(q1.n_operator(), q1),
        op2=(x_c, c),
    )

    H = _to_matrix(hilbertspace.hamiltonian())
    q0_esys = q0.eigensys(evals_count=q0_trunc)
    n_q0_local = _to_matrix(q0.n_operator(energy_esys=q0_esys))
    n_q0_lower_local = np.triu(n_q0_local, 1)
    id_c = np.eye(c_trunc, dtype=complex)
    id_q1 = np.eye(q1_trunc, dtype=complex)
    drive_lower = np.kron(np.kron(id_q1, id_c), n_q0_lower_local)

    n_q0 = np.diag(np.arange(q0_trunc, dtype=float)).astype(complex)
    n_c = np.diag(np.arange(c_trunc, dtype=float)).astype(complex)
    n_q1 = np.diag(np.arange(q1_trunc, dtype=float)).astype(complex)
    total_excitation = (
        np.kron(np.kron(id_q1, id_c), n_q0)
        + np.kron(np.kron(id_q1, n_c), np.eye(q0_trunc, dtype=complex))
        + np.kron(np.kron(n_q1, id_c), np.eye(q0_trunc, dtype=complex))
    )
    return H, drive_lower, total_excitation


def _single_point_duffing_stack(config: StudyConfig, *, flux_value: float, sweep_target: str) -> np.ndarray:
    flux_arr = np.array([float(flux_value)], dtype=float)
    if not is_reference_calibrated_duffing_mode(config.static_benchmark.duffing_model.calibration_mode):
        return build_duffing_model_stack(
            flux_values=flux_arr,
            system_params=config.system,
            coupler_frequency=config.static_benchmark.coupler_frequency,
            duffing_config=config.static_benchmark.duffing_model,
            sweep_target=sweep_target,
        ).hamiltonian_stack

    circuit_stack = build_circuit_model_stack(
        flux_values=flux_arr,
        system_params=config.system,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        circuit_config=config.static_benchmark.circuit_model,
        sweep_target=sweep_target,
    ).hamiltonian_stack
    H_circuit_eff = build_dressed_effective_computational_stack(
        circuit_stack,
        nlevels_qubit=config.static_benchmark.circuit_model.hilbert_truncation.q0_truncated_dim,
        nlevels_coupler=config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim,
        n_candidate_states=config.static_benchmark.dressed_subspace.n_candidate_states,
        selection_mode=config.static_benchmark.dressed_subspace.selection_mode,
    )
    duffing_mode_parameters = fit_duffing_mode_parameters_to_reference(
        flux_values=flux_arr,
        reference_dressed_stack=H_circuit_eff,
        system_params=config.system,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        duffing_config=config.static_benchmark.duffing_model,
        sweep_target=sweep_target,
        n_candidate_states=config.static_benchmark.dressed_subspace.n_candidate_states,
        selection_mode=config.static_benchmark.dressed_subspace.selection_mode,
    )
    return build_duffing_model_stack_from_parameters(
        duffing_mode_parameters,
        system_params=config.system,
        duffing_config=config.static_benchmark.duffing_model,
    ).hamiltonian_stack


def _single_point_effective_hamiltonian(config: StudyConfig) -> np.ndarray:
    flux_value = np.array([float(config.system.q0.flux)], dtype=float)
    sweep_target = "q0"

    duffing_stack = _single_point_duffing_stack(
        config,
        flux_value=float(flux_value[0]),
        sweep_target=sweep_target,
    )
    circuit_stack = build_circuit_model_stack(
        flux_values=flux_value,
        system_params=config.system,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        circuit_config=config.static_benchmark.circuit_model,
        sweep_target=sweep_target,
    ).hamiltonian_stack

    selection_mode = config.static_benchmark.dressed_subspace.selection_mode
    n_candidate_states = config.static_benchmark.dressed_subspace.n_candidate_states
    H_duffing_eff = build_dressed_effective_computational_stack(
        duffing_stack,
        nlevels_qubit=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit,
        nlevels_coupler=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler,
        n_candidate_states=n_candidate_states,
        selection_mode=selection_mode,
    )
    H_circuit_eff = build_dressed_effective_computational_stack(
        circuit_stack,
        nlevels_qubit=config.static_benchmark.circuit_model.hilbert_truncation.q0_truncated_dim,
        nlevels_coupler=config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim,
        n_candidate_states=n_candidate_states,
        selection_mode=selection_mode,
    )

    source = str(config.static_benchmark.effective_model.derivation_source)
    if source == "duffing":
        source_stack = H_duffing_eff
    elif source == "circuit":
        source_stack = H_circuit_eff
    else:  # pragma: no cover - config parser guards this
        raise ValueError(f"Unsupported effective derivation source {source!r}")

    params = extract_model1_parameters_from_4x4_stack(source_stack)
    effective_stack = build_effective_hamiltonian_stack(params)
    return np.asarray(effective_stack[0], dtype=complex)


def _drive_hamiltonian(*, amplitude: float, phase_rad: float, envelope: np.ndarray, lowering_operator: np.ndarray) -> np.ndarray:
    coeff_raise = np.exp(-1.0j * float(phase_rad))
    coeff_lower = np.exp(1.0j * float(phase_rad))
    drive_matrix = -0.5 * float(amplitude) * (
        coeff_raise * np.asarray(lowering_operator.conj().T, dtype=complex)
        + coeff_lower * np.asarray(lowering_operator, dtype=complex)
    )
    return np.asarray(envelope, dtype=float)[:, np.newaxis, np.newaxis] * drive_matrix[np.newaxis, :, :]


def _rotation_stack(*, drift_hamiltonian: np.ndarray, drive_frequency: float, total_excitation_operator: np.ndarray, drive_hamiltonian: np.ndarray) -> np.ndarray:
    H_rot = np.asarray(drift_hamiltonian, dtype=complex) - float(drive_frequency) * np.asarray(total_excitation_operator, dtype=complex)
    return H_rot[np.newaxis, :, :] + np.asarray(drive_hamiltonian, dtype=complex)


def run_rx_benchmark(
    config: StudyConfig,
    *,
    drive_qubit: str,
    drive_frequency: float,
    drive_amplitude: float,
    drive_phase_rad: float,
    total_time_ns: float,
    dt_ns: float,
    rise_time_ns: float,
) -> RxBenchmarkResult:
    if str(drive_qubit) != "q0":
        raise ValueError("RX benchmark currently supports drive_qubit='q0' only")

    times_ns, envelope = _cosine_edge_envelope(
        total_time_ns=float(total_time_ns),
        rise_time_ns=float(rise_time_ns),
        dt_ns=float(dt_ns),
    )

    H_effective_lab = _single_point_effective_hamiltonian(config)
    H_duffing_lab = _single_point_duffing_stack(
        config,
        flux_value=float(config.system.q0.flux),
        sweep_target="q0",
    )[0]
    H_circuit_lab, b_circuit, N_circuit = _build_circuit_idle_components(config)

    b_effective = _effective_q0_lowering_operator()
    N_effective = _effective_total_excitation_operator()
    b_duffing = _three_mode_q0_lowering_operator(
        nlevels_qubit=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit,
        nlevels_coupler=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler,
    )
    N_duffing = _three_mode_total_excitation_operator(
        nlevels_qubit=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit,
        nlevels_coupler=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler,
    )

    H_effective = _rotation_stack(
        drift_hamiltonian=H_effective_lab,
        drive_frequency=float(drive_frequency),
        total_excitation_operator=N_effective,
        drive_hamiltonian=_drive_hamiltonian(
            amplitude=float(drive_amplitude),
            phase_rad=float(drive_phase_rad),
            envelope=envelope,
            lowering_operator=b_effective,
        ),
    )
    H_duffing = _rotation_stack(
        drift_hamiltonian=H_duffing_lab,
        drive_frequency=float(drive_frequency),
        total_excitation_operator=N_duffing,
        drive_hamiltonian=_drive_hamiltonian(
            amplitude=float(drive_amplitude),
            phase_rad=float(drive_phase_rad),
            envelope=envelope,
            lowering_operator=b_duffing,
        ),
    )
    H_circuit = _rotation_stack(
        drift_hamiltonian=H_circuit_lab,
        drive_frequency=float(drive_frequency),
        total_excitation_operator=N_circuit,
        drive_hamiltonian=_drive_hamiltonian(
            amplitude=float(drive_amplitude),
            phase_rad=float(drive_phase_rad),
            envelope=envelope,
            lowering_operator=b_circuit,
        ),
    )

    idx_effective = np.array([0, 1, 2, 3], dtype=int)
    idx_duffing = computational_state_indices(
        nlevels_qubit=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit,
        nlevels_coupler=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler,
    )
    idx_circuit = computational_state_indices(
        nlevels_qubit=config.static_benchmark.circuit_model.hilbert_truncation.q0_truncated_dim,
        nlevels_coupler=config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim,
    )

    amp_effective = _simulate_computational_basis(H_effective, times_ns=times_ns, computational_indices=idx_effective)
    amp_duffing = _simulate_computational_basis(H_duffing, times_ns=times_ns, computational_indices=idx_duffing)
    amp_circuit = _simulate_computational_basis(H_circuit, times_ns=times_ns, computational_indices=idx_circuit)

    eff_00_to_01 = _population_trace(amp_effective, row=1, col=0)
    duf_00_to_01 = _population_trace(amp_duffing, row=1, col=0)
    cir_00_to_01 = _population_trace(amp_circuit, row=1, col=0)
    eff_10_to_11 = _population_trace(amp_effective, row=3, col=2)
    duf_10_to_11 = _population_trace(amp_duffing, row=3, col=2)
    cir_10_to_11 = _population_trace(amp_circuit, row=3, col=2)

    eff_leak_00 = _leakage_trace(amp_effective, col=0)
    duf_leak_00 = _leakage_trace(amp_duffing, col=0)
    cir_leak_00 = _leakage_trace(amp_circuit, col=0)
    eff_leak_10 = _leakage_trace(amp_effective, col=2)
    duf_leak_10 = _leakage_trace(amp_duffing, col=2)
    cir_leak_10 = _leakage_trace(amp_circuit, col=2)

    eff_delta = np.abs(eff_00_to_01 - eff_10_to_11)
    duf_delta = np.abs(duf_00_to_01 - duf_10_to_11)
    cir_delta = np.abs(cir_00_to_01 - cir_10_to_11)

    summary = {
        "effective_final_pop_00_to_01": float(eff_00_to_01[-1]),
        "duffing_final_pop_00_to_01": float(duf_00_to_01[-1]),
        "circuit_final_pop_00_to_01": float(cir_00_to_01[-1]),
        "effective_final_pop_10_to_11": float(eff_10_to_11[-1]),
        "duffing_final_pop_10_to_11": float(duf_10_to_11[-1]),
        "circuit_final_pop_10_to_11": float(cir_10_to_11[-1]),
        "effective_max_leakage_from_00": float(np.max(eff_leak_00)),
        "duffing_max_leakage_from_00": float(np.max(duf_leak_00)),
        "circuit_max_leakage_from_00": float(np.max(cir_leak_00)),
        "effective_max_leakage_from_10": float(np.max(eff_leak_10)),
        "duffing_max_leakage_from_10": float(np.max(duf_leak_10)),
        "circuit_max_leakage_from_10": float(np.max(cir_leak_10)),
        "effective_final_spectator_population_delta": float(eff_delta[-1]),
        "duffing_final_spectator_population_delta": float(duf_delta[-1]),
        "circuit_final_spectator_population_delta": float(cir_delta[-1]),
        "drive_frequency": float(drive_frequency),
        "drive_amplitude": float(drive_amplitude),
        "drive_phase_rad": float(drive_phase_rad),
        "total_time_ns": float(total_time_ns),
        "dt_ns": float(dt_ns),
        "rise_time_ns": float(rise_time_ns),
    }

    return RxBenchmarkResult(
        times_ns=np.asarray(times_ns, dtype=float),
        pulse_envelope=np.asarray(envelope, dtype=float),
        drive_qubit=str(drive_qubit),
        drive_frequency=float(drive_frequency),
        drive_amplitude=float(drive_amplitude),
        drive_phase_rad=float(drive_phase_rad),
        total_time_ns=float(total_time_ns),
        dt_ns=float(dt_ns),
        rise_time_ns=float(rise_time_ns),
        effective_computational_amplitudes=np.asarray(amp_effective, dtype=complex),
        duffing_computational_amplitudes=np.asarray(amp_duffing, dtype=complex),
        circuit_computational_amplitudes=np.asarray(amp_circuit, dtype=complex),
        effective_pop_00_to_01=np.asarray(eff_00_to_01, dtype=float),
        duffing_pop_00_to_01=np.asarray(duf_00_to_01, dtype=float),
        circuit_pop_00_to_01=np.asarray(cir_00_to_01, dtype=float),
        effective_pop_10_to_11=np.asarray(eff_10_to_11, dtype=float),
        duffing_pop_10_to_11=np.asarray(duf_10_to_11, dtype=float),
        circuit_pop_10_to_11=np.asarray(cir_10_to_11, dtype=float),
        effective_leakage_from_00=np.asarray(eff_leak_00, dtype=float),
        duffing_leakage_from_00=np.asarray(duf_leak_00, dtype=float),
        circuit_leakage_from_00=np.asarray(cir_leak_00, dtype=float),
        effective_leakage_from_10=np.asarray(eff_leak_10, dtype=float),
        duffing_leakage_from_10=np.asarray(duf_leak_10, dtype=float),
        circuit_leakage_from_10=np.asarray(cir_leak_10, dtype=float),
        effective_spectator_population_delta=np.asarray(eff_delta, dtype=float),
        duffing_spectator_population_delta=np.asarray(duf_delta, dtype=float),
        circuit_spectator_population_delta=np.asarray(cir_delta, dtype=float),
        summary=summary,
    )
