"""CZ-relevant dynamics benchmark for effective, Duffing, and circuit models."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from comparison.fitted_reconstruction import (
    duffing_mode_parameters_for_flux,
    effective_parameters_for_flux,
)
from comparison.static import (
    CircuitFluxSweepReference,
    build_circuit_static_reference,
    run_static_benchmark,
)
from models import (
    build_duffing_model_stack_from_parameters,
    build_duffing_model_stack_from_scratch,
    build_effective_hamiltonian_stack,
    computational_state_indices,
    is_reference_calibrated_duffing_mode,
)
from runtime_utils import log_progress, progress_heartbeat
from study_config import StudyConfig
from toolkit.observables import CzDynamicsObservables, extract_cz_observables
from toolkit.pulses import ramp_hold_ramp_flux_pulse
from toolkit.solver import propagate_piecewise_constant
from toolkit.simulation import simulate_circuit

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class CzCircuitResult:
    """Circuit pulse and dynamics shared by both CZ run modes."""

    times_ns: np.ndarray
    pulse_flux_values: np.ndarray
    sweep_target: str
    idle_flux: float
    target_flux: float
    ramp_time_ns: float
    hold_time_ns: float
    dt_ns: float
    scan_hold_times_ns: np.ndarray
    scan_phase_error_rad: np.ndarray
    scan_scores: np.ndarray
    circuit_populations_11: np.ndarray
    circuit_leakage_11: np.ndarray
    circuit_conditional_phase: np.ndarray
    circuit_computational_amplitudes: np.ndarray
    circuit_statevector_plus_plus: np.ndarray
    circuit_populations_plus_plus: np.ndarray
    circuit_intermediate_population_11: np.ndarray


@dataclass(frozen=True)
class CzCircuitOnlyResult(CzCircuitResult):
    """Circuit-only CZ run with concrete timing fields."""

    circuit_hilbert_dim: int
    circuit_model_build_runtime_s: float
    circuit_propagation_runtime_s: float
    shared_static_precompute_runtime_s: float
    shared_hold_scan_runtime_s: float

    @property
    def circuit_dynamics_runtime_s(self) -> float:
        return self.circuit_model_build_runtime_s + self.circuit_propagation_runtime_s


@dataclass(frozen=True)
class CzBenchmarkResult(CzCircuitResult):
    """Full CZ run, adding effective and Duffing observations to circuit data."""

    effective_populations_11: np.ndarray
    duffing_populations_11: np.ndarray
    effective_leakage_11: np.ndarray
    duffing_leakage_11: np.ndarray
    effective_conditional_phase: np.ndarray
    duffing_conditional_phase: np.ndarray
    effective_computational_amplitudes: np.ndarray
    duffing_computational_amplitudes: np.ndarray
    effective_statevector_plus_plus: np.ndarray
    duffing_statevector_plus_plus: np.ndarray
    effective_populations_plus_plus: np.ndarray
    duffing_populations_plus_plus: np.ndarray
    effective_intermediate_population_11: np.ndarray
    duffing_intermediate_population_11: np.ndarray
    summary: dict[str, float]


def _idle_flux_for_target(config: StudyConfig, sweep_target: str) -> float:
    if sweep_target == "q0":
        return float(config.system.q0.flux)
    if sweep_target == "q1":
        return float(config.system.q1.flux)
    raise ValueError(f"Unsupported sweep_target {sweep_target!r}")


def _pick_target_flux_from_static(
    static_result: CircuitFluxSweepReference, *, idle_flux: float
) -> float:
    flux = np.asarray(static_result.flux_values, dtype=float).ravel()
    zeta = np.abs(
        np.asarray(static_result.circuit_parameters["zeta"], dtype=float).ravel()
    )
    if flux.size < 2:
        return float(idle_flux)

    valid = np.ones(flux.shape, dtype=bool)
    if flux.size > 4:
        valid[0] = False
        valid[-1] = False

    zeta_valid = np.where(valid, zeta, -np.inf)
    max_zeta = float(np.max(zeta_valid))
    near_max_mask = valid & (zeta >= 0.98 * max_zeta)
    near_max_idx = np.flatnonzero(near_max_mask)
    if near_max_idx.size > 0:
        idx = int(
            near_max_idx[np.argmin(np.abs(flux[near_max_idx] - float(idle_flux)))]
        )
    else:
        idx = int(np.argmax(zeta_valid))

    target = float(flux[idx])
    if abs(target - float(idle_flux)) < 1e-10:
        alt = int(np.argmax(np.abs(flux - float(idle_flux))))
        target = float(flux[alt])
    return target


def _phase_error_to_cz_target(phase_rad: float) -> float:
    phase = float(phase_rad)
    candidates = [np.pi + TWO_PI * k for k in range(-4, 5)]
    return float(min(abs(phase - c) for c in candidates))


def _simulate_piecewise_constant_scipy(
    H_stack: np.ndarray,
    times_ns: np.ndarray,
    computational_indices: np.ndarray,
    *,
    counts: np.ndarray | None = None,
    monitor_indices: np.ndarray | None = None,
    progress_label: str | None = None,
    progress_interval_s: float = 30.0,
) -> CzDynamicsObservables:
    """Adapt the public propagator to the CZ computational basis and analysis."""
    comp_idx = np.asarray(computational_indices, dtype=int).ravel()
    if comp_idx.size != 4:
        raise ValueError("Need exactly 4 computational indices")
    states = np.zeros((H_stack.shape[1], 4), dtype=complex)
    states[comp_idx, np.arange(4)] = 1.0
    monitor_idx = (
        np.empty(0, dtype=int)
        if monitor_indices is None
        else np.asarray(monitor_indices, dtype=int).ravel()
    )
    amplitudes = propagate_piecewise_constant(
        H_stack,
        times_ns,
        states,
        counts=counts,
        output_indices=np.concatenate((comp_idx, monitor_idx)),
        progress_label=progress_label,
        progress_interval_s=progress_interval_s,
    )
    return extract_cz_observables(
        amplitudes[:, :4, :], monitor_amplitudes=amplitudes[:, 4:, :]
    )


def _q1c0q0_index(q1: int, q0: int, *, nlevels_qubit: int, nlevels_coupler: int) -> int:
    return int(q1) * int(nlevels_coupler) * int(nlevels_qubit) + int(q0)


def _intermediate_channel_indices(
    *, nlevels_qubit: int, nlevels_coupler: int
) -> np.ndarray:
    if int(nlevels_qubit) < 3:
        return np.zeros(0, dtype=int)
    return np.array(
        [
            _q1c0q0_index(
                0, 2, nlevels_qubit=nlevels_qubit, nlevels_coupler=nlevels_coupler
            ),
            _q1c0q0_index(
                2, 0, nlevels_qubit=nlevels_qubit, nlevels_coupler=nlevels_coupler
            ),
        ],
        dtype=int,
    )


def _evaluate_circuit_candidate(
    *,
    config: StudyConfig,
    sweep_target: str,
    idle_flux: float,
    target_flux: float,
    ramp_time_ns: float,
    hold_time_ns: float,
    dt_ns: float,
    idx_circuit: np.ndarray,
) -> tuple[float, float, float]:
    times_ns, pulse_flux, counts = ramp_hold_ramp_flux_pulse(
        ramp_time_ns=ramp_time_ns,
        hold_time_ns=hold_time_ns,
        dt_ns=dt_ns,
        idle_flux=idle_flux,
        target_flux=target_flux,
        compress=True,
    )
    simulation = simulate_circuit(
        flux_values=pulse_flux,
        times_ns=times_ns,
        counts=counts,
        system_params=config.system,
        circuit_config=config.static_benchmark.circuit_model,
        sweep_target=sweep_target,
        output_indices=idx_circuit,
    )
    obs = extract_cz_observables(simulation.amplitudes)
    phase_final = float(obs.conditional_phase[-1])
    phase_error = _phase_error_to_cz_target(phase_final)
    max_leak = float(np.max(obs.leakage_11))
    return phase_final, phase_error, max_leak


def _scan_circuit_hold_time(
    config: StudyConfig,
    static_result: CircuitFluxSweepReference,
    *,
    sweep_target: str,
    idle_flux: float,
    target_flux: float,
    ramp_time_ns: float,
    dt_ns: float,
    scan_dt_ns: float,
    scan_max_hold_ns: float,
    scan_leakage_penalty: float,
    idx_circuit: np.ndarray,
) -> tuple[float, list[float], list[float], list[float]]:
    """Select a circuit CZ hold time and retain the candidate scores."""
    scan_hold_list: list[float] = []
    scan_phase_error_list: list[float] = []
    scan_score_list: list[float] = []
    phase_ramp_only, _, _ = _evaluate_circuit_candidate(
        config=config,
        sweep_target=sweep_target,
        idle_flux=idle_flux,
        target_flux=target_flux,
        ramp_time_ns=ramp_time_ns,
        hold_time_ns=0.0,
        dt_ns=scan_dt_ns,
        idx_circuit=idx_circuit,
    )
    zeta_ref = np.interp(
        target_flux,
        np.asarray(static_result.flux_values, dtype=float).ravel(),
        np.asarray(static_result.circuit_parameters["zeta"], dtype=float).ravel(),
    )
    phase_rate = TWO_PI * max(abs(float(zeta_ref)), 1e-8)
    hold_est = max(0.0, (np.pi - float(phase_ramp_only)) / phase_rate)
    hold_est = float(min(hold_est, scan_max_hold_ns))

    candidate_values = [0.0, 0.8 * hold_est, hold_est, 1.2 * hold_est]
    if hold_est >= scan_max_hold_ns - 1e-12:
        candidate_values.append(scan_max_hold_ns)
    candidates = sorted(set(candidate_values))

    scored: list[tuple[float, float]] = []
    for candidate in candidates:
        h = float(np.clip(candidate, 0.0, scan_max_hold_ns))
        log_progress(f"cz benchmark: coarse hold scan candidate {h:.3f} ns")
        _, phase_error, max_leak = _evaluate_circuit_candidate(
            config=config,
            sweep_target=sweep_target,
            idle_flux=idle_flux,
            target_flux=target_flux,
            ramp_time_ns=ramp_time_ns,
            hold_time_ns=h,
            dt_ns=scan_dt_ns,
            idx_circuit=idx_circuit,
        )
        score = float(phase_error + scan_leakage_penalty * max_leak)
        scan_hold_list.append(h)
        scan_phase_error_list.append(phase_error)
        scan_score_list.append(score)
        scored.append((score, h))

    selected_hold = float(min(scored, key=lambda x: x[0])[1])

    # Refine around the selected hold using the final integration step size.
    refine_step = max(2.0 * dt_ns, 0.05 * max(selected_hold, 1.0))
    refine_candidates = sorted(
        {
            float(np.clip(selected_hold - refine_step, 0.0, scan_max_hold_ns)),
            float(np.clip(selected_hold, 0.0, scan_max_hold_ns)),
            float(np.clip(selected_hold + refine_step, 0.0, scan_max_hold_ns)),
        }
    )
    refined: list[tuple[float, float]] = []
    for h in refine_candidates:
        log_progress(f"cz benchmark: refined hold scan candidate {h:.3f} ns")
        _, phase_error, max_leak = _evaluate_circuit_candidate(
            config=config,
            sweep_target=sweep_target,
            idle_flux=idle_flux,
            target_flux=target_flux,
            ramp_time_ns=ramp_time_ns,
            hold_time_ns=h,
            dt_ns=dt_ns,
            idx_circuit=idx_circuit,
        )
        score = float(phase_error + scan_leakage_penalty * max_leak)
        scan_hold_list.append(h)
        scan_phase_error_list.append(phase_error)
        scan_score_list.append(score)
        refined.append((score, h))

    selected_hold = float(min(refined, key=lambda x: x[0])[1])
    return selected_hold, scan_hold_list, scan_phase_error_list, scan_score_list


def run_cz_benchmark(
    config: StudyConfig,
    *,
    ramp_time_ns: float = 8.0,
    hold_time_ns: float | None = None,
    dt_ns: float = 1.0,
    enable_hold_time_scan: bool = True,
    scan_dt_ns: float = 2.0,
    scan_max_hold_ns: float = 300.0,
    scan_leakage_penalty: float = 0.25,
    precomputed_static_result: CircuitFluxSweepReference | None = None,
    precomputed_static_runtime_s: float | None = None,
    circuit_only: bool = False,
    target_flux: float | None = None,
) -> CzBenchmarkResult | CzCircuitOnlyResult:
    """Run a CZ-relevant dynamics benchmark under a shared flux pulse schedule.

    Circuit model:
      Hamiltonians from scqubits, piecewise propagation via numpy/scipy.
    Effective + Duffing models:
      piecewise propagation via numpy/scipy.
    With circuit_only=True, return circuit dynamics and timing data.
    """
    if precomputed_static_result is None:
        static_started = time.perf_counter()
        with progress_heartbeat("cz benchmark: prepare static reference"):
            static_result = (
                run_static_benchmark(config)
                if not circuit_only
                else build_circuit_static_reference(config)
            )
        static_runtime_s = float(time.perf_counter() - static_started)
    else:
        static_result = precomputed_static_result
        static_runtime_s = (
            float(precomputed_static_runtime_s)
            if precomputed_static_runtime_s is not None
            else 0.0
        )
    sweep_target = str(config.static_benchmark.flux_control.sweep_target)
    idle_flux = _idle_flux_for_target(config, sweep_target)
    if target_flux is None:
        target_flux = _pick_target_flux_from_static(static_result, idle_flux=idle_flux)

    ramp_time_ns = float(ramp_time_ns)
    dt_ns = float(dt_ns)
    scan_dt_ns = float(scan_dt_ns)
    scan_max_hold_ns = float(max(0.0, scan_max_hold_ns))
    scan_leakage_penalty = float(max(0.0, scan_leakage_penalty))
    if hold_time_ns is not None and enable_hold_time_scan:
        raise ValueError(
            "Ambiguous CZ hold-time configuration: both hold_time_ns and "
            "enable_hold_time_scan=True were provided. Set hold_time_ns=None "
            "to scan, or disable scan to use fixed hold_time_ns."
        )

    idx_circuit = computational_state_indices(
        nlevels_qubit=config.static_benchmark.circuit_model.hilbert_truncation.q0_truncated_dim,
        nlevels_coupler=config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim,
    )
    idx_circuit_intermediate = _intermediate_channel_indices(
        nlevels_qubit=config.static_benchmark.circuit_model.hilbert_truncation.q0_truncated_dim,
        nlevels_coupler=config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim,
    )

    scan_hold_list: list[float] = []
    scan_phase_error_list: list[float] = []
    scan_score_list: list[float] = []
    hold_scan_runtime_s = 0.0

    selected_hold = 0.0 if hold_time_ns is None else float(max(0.0, hold_time_ns))
    if hold_time_ns is None and enable_hold_time_scan:
        hold_scan_started = time.perf_counter()
        (
            selected_hold,
            scan_hold_list,
            scan_phase_error_list,
            scan_score_list,
        ) = _scan_circuit_hold_time(
            config,
            static_result,
            sweep_target=sweep_target,
            idle_flux=idle_flux,
            target_flux=target_flux,
            ramp_time_ns=ramp_time_ns,
            dt_ns=dt_ns,
            scan_dt_ns=scan_dt_ns,
            scan_max_hold_ns=scan_max_hold_ns,
            scan_leakage_penalty=scan_leakage_penalty,
            idx_circuit=idx_circuit,
        )
        hold_scan_runtime_s = float(time.perf_counter() - hold_scan_started)

    times_ns, pulse_flux, counts = ramp_hold_ramp_flux_pulse(
        ramp_time_ns=ramp_time_ns,
        hold_time_ns=selected_hold,
        dt_ns=dt_ns,
        idle_flux=idle_flux,
        target_flux=target_flux,
        compress=True,
    )
    # Preserve the sampled pulse in saved results and plots, including its
    # final parking point, while building only the segment Hamiltonians.
    full_pulse_flux = np.append(np.repeat(pulse_flux, counts), idle_flux)

    if not circuit_only:
        idx_effective = np.array([0, 1, 2, 3], dtype=int)
        idx_duffing = computational_state_indices(
            nlevels_qubit=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit,
            nlevels_coupler=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler,
        )
        idx_duffing_intermediate = _intermediate_channel_indices(
            nlevels_qubit=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit,
            nlevels_coupler=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler,
        )

        effective_build_started = time.perf_counter()
        with progress_heartbeat("cz benchmark: build effective Hamiltonian"):
            effective_parameters_t = effective_parameters_for_flux(
                static_result,
                config,
                pulse_flux,
            )
            H_effective = build_effective_hamiltonian_stack(effective_parameters_t)
        effective_build_runtime_s = float(time.perf_counter() - effective_build_started)

        duffing_build_started = time.perf_counter()
        with progress_heartbeat("cz benchmark: build Duffing Hamiltonian"):
            if is_reference_calibrated_duffing_mode(
                config.static_benchmark.duffing_model.calibration_mode
            ):
                duffing_mode_parameters_t = duffing_mode_parameters_for_flux(
                    static_result,
                    config,
                    pulse_flux,
                )
                duffing_stack = build_duffing_model_stack_from_parameters(
                    duffing_mode_parameters_t,
                    system_params=config.system,
                    duffing_config=config.static_benchmark.duffing_model,
                ).hamiltonian_stack
            else:
                duffing_stack = build_duffing_model_stack_from_scratch(
                    flux_values=pulse_flux,
                    system_params=config.system,
                    duffing_config=config.static_benchmark.duffing_model,
                    sweep_target=sweep_target,
                ).hamiltonian_stack
        duffing_build_runtime_s = float(time.perf_counter() - duffing_build_started)

    if not circuit_only:
        effective_prop_started = time.perf_counter()
        obs_effective = _simulate_piecewise_constant_scipy(
            H_effective,
            times_ns,
            idx_effective,
            counts=counts,
            monitor_indices=None,
            progress_label="cz benchmark: propagate effective model",
        )
        effective_prop_runtime_s = float(time.perf_counter() - effective_prop_started)
        duffing_prop_started = time.perf_counter()
        obs_duffing = _simulate_piecewise_constant_scipy(
            duffing_stack,
            times_ns,
            idx_duffing,
            counts=counts,
            monitor_indices=idx_duffing_intermediate,
            progress_label="cz benchmark: propagate Duffing model",
        )
        duffing_prop_runtime_s = float(time.perf_counter() - duffing_prop_started)

    simulation = simulate_circuit(
        flux_values=pulse_flux,
        times_ns=times_ns,
        counts=counts,
        system_params=config.system,
        circuit_config=config.static_benchmark.circuit_model,
        sweep_target=sweep_target,
        output_indices=np.concatenate((idx_circuit, idx_circuit_intermediate)),
        progress_label="cz benchmark: circuit model",
    )
    circuit_analysis_started = time.perf_counter()
    obs_circuit = extract_cz_observables(
        simulation.amplitudes[:, :4, :],
        monitor_amplitudes=simulation.amplitudes[:, 4:, :],
    )
    circuit_build_runtime_s = simulation.build_runtime_s
    # Like the other model adapters, benchmark solver timing includes analysis.
    circuit_prop_runtime_s = simulation.propagation_runtime_s + float(
        time.perf_counter() - circuit_analysis_started
    )

    plus_plus_coeff = np.full(4, 0.5 + 0.0j, dtype=complex)
    circuit_statevector_plus_plus = np.einsum(
        "tij,j->ti",
        obs_circuit.computational_amplitudes,
        plus_plus_coeff,
    )
    circuit_pop_plus_plus = np.abs(circuit_statevector_plus_plus) ** 2

    if circuit_only:
        return CzCircuitOnlyResult(
            times_ns=np.asarray(times_ns, dtype=float),
            pulse_flux_values=full_pulse_flux,
            sweep_target=sweep_target,
            idle_flux=float(idle_flux),
            target_flux=float(target_flux),
            ramp_time_ns=float(ramp_time_ns),
            hold_time_ns=float(selected_hold),
            dt_ns=float(dt_ns),
            scan_hold_times_ns=np.asarray(scan_hold_list, dtype=float),
            scan_phase_error_rad=np.asarray(scan_phase_error_list, dtype=float),
            scan_scores=np.asarray(scan_score_list, dtype=float),
            circuit_populations_11=obs_circuit.populations_11,
            circuit_leakage_11=obs_circuit.leakage_11,
            circuit_conditional_phase=obs_circuit.conditional_phase,
            circuit_computational_amplitudes=np.asarray(
                obs_circuit.computational_amplitudes, dtype=complex
            ),
            circuit_statevector_plus_plus=np.asarray(
                circuit_statevector_plus_plus, dtype=complex
            ),
            circuit_populations_plus_plus=np.asarray(
                circuit_pop_plus_plus, dtype=float
            ),
            circuit_intermediate_population_11=obs_circuit.monitor_population_11,
            circuit_hilbert_dim=int(simulation.hamiltonian_shape[-1]),
            circuit_model_build_runtime_s=circuit_build_runtime_s,
            circuit_propagation_runtime_s=circuit_prop_runtime_s,
            shared_static_precompute_runtime_s=static_runtime_s,
            shared_hold_scan_runtime_s=float(hold_scan_runtime_s),
        )

    eff_pop_rmse = float(
        np.sqrt(
            np.mean((obs_effective.populations_11 - obs_circuit.populations_11) ** 2)
        )
    )
    duf_pop_rmse = float(
        np.sqrt(np.mean((obs_duffing.populations_11 - obs_circuit.populations_11) ** 2))
    )
    eff_phase_err = float(
        np.abs(obs_effective.conditional_phase[-1] - obs_circuit.conditional_phase[-1])
    )
    duf_phase_err = float(
        np.abs(obs_duffing.conditional_phase[-1] - obs_circuit.conditional_phase[-1])
    )

    summary = {
        "effective_final_conditional_phase_rad": float(
            obs_effective.conditional_phase[-1]
        ),
        "duffing_final_conditional_phase_rad": float(obs_duffing.conditional_phase[-1]),
        "circuit_final_conditional_phase_rad": float(obs_circuit.conditional_phase[-1]),
        "circuit_final_phase_error_to_pi_rad": _phase_error_to_cz_target(
            float(obs_circuit.conditional_phase[-1])
        ),
        "effective_final_phase_error_vs_circuit_rad": eff_phase_err,
        "duffing_final_phase_error_vs_circuit_rad": duf_phase_err,
        "effective_max_leakage_11": float(np.max(obs_effective.leakage_11)),
        "duffing_max_leakage_11": float(np.max(obs_duffing.leakage_11)),
        "circuit_max_leakage_11": float(np.max(obs_circuit.leakage_11)),
        "effective_populations_rmse_vs_circuit": eff_pop_rmse,
        "duffing_populations_rmse_vs_circuit": duf_pop_rmse,
        "ramp_time_ns": float(ramp_time_ns),
        "hold_time_ns": float(selected_hold),
        "dt_ns": float(dt_ns),
        "n_time_points": float(times_ns.size),
        "n_time_steps": float(max(times_ns.size - 1, 0)),
        "effective_hilbert_dim": float(H_effective.shape[-1]),
        "duffing_hilbert_dim": float(duffing_stack.shape[-1]),
        "circuit_hilbert_dim": float(simulation.hamiltonian_shape[-1]),
        "shared_static_precompute_runtime_s": static_runtime_s,
        "shared_hold_scan_runtime_s": float(hold_scan_runtime_s),
        "effective_model_build_runtime_s": effective_build_runtime_s,
        "duffing_model_build_runtime_s": duffing_build_runtime_s,
        "circuit_model_build_runtime_s": circuit_build_runtime_s,
        "effective_propagation_runtime_s": effective_prop_runtime_s,
        "duffing_propagation_runtime_s": duffing_prop_runtime_s,
        "circuit_propagation_runtime_s": circuit_prop_runtime_s,
        "effective_dynamics_runtime_s": float(
            effective_build_runtime_s + effective_prop_runtime_s
        ),
        "duffing_dynamics_runtime_s": float(
            duffing_build_runtime_s + duffing_prop_runtime_s
        ),
        "circuit_dynamics_runtime_s": float(
            circuit_build_runtime_s + circuit_prop_runtime_s
        ),
    }

    effective_statevector_plus_plus = np.einsum(
        "tij,j->ti", obs_effective.computational_amplitudes, plus_plus_coeff
    )
    duffing_statevector_plus_plus = np.einsum(
        "tij,j->ti", obs_duffing.computational_amplitudes, plus_plus_coeff
    )
    effective_pop_plus_plus = np.abs(effective_statevector_plus_plus) ** 2
    duffing_pop_plus_plus = np.abs(duffing_statevector_plus_plus) ** 2
    return CzBenchmarkResult(
        times_ns=np.asarray(times_ns, dtype=float),
        pulse_flux_values=full_pulse_flux,
        sweep_target=sweep_target,
        idle_flux=float(idle_flux),
        target_flux=float(target_flux),
        ramp_time_ns=float(ramp_time_ns),
        hold_time_ns=float(selected_hold),
        dt_ns=float(dt_ns),
        scan_hold_times_ns=np.asarray(scan_hold_list, dtype=float),
        scan_phase_error_rad=np.asarray(scan_phase_error_list, dtype=float),
        scan_scores=np.asarray(scan_score_list, dtype=float),
        effective_populations_11=obs_effective.populations_11,
        duffing_populations_11=obs_duffing.populations_11,
        circuit_populations_11=obs_circuit.populations_11,
        effective_leakage_11=obs_effective.leakage_11,
        duffing_leakage_11=obs_duffing.leakage_11,
        circuit_leakage_11=obs_circuit.leakage_11,
        effective_conditional_phase=obs_effective.conditional_phase,
        duffing_conditional_phase=obs_duffing.conditional_phase,
        circuit_conditional_phase=obs_circuit.conditional_phase,
        effective_computational_amplitudes=np.asarray(
            obs_effective.computational_amplitudes, dtype=complex
        ),
        duffing_computational_amplitudes=np.asarray(
            obs_duffing.computational_amplitudes, dtype=complex
        ),
        circuit_computational_amplitudes=np.asarray(
            obs_circuit.computational_amplitudes, dtype=complex
        ),
        effective_statevector_plus_plus=np.asarray(
            effective_statevector_plus_plus, dtype=complex
        ),
        duffing_statevector_plus_plus=np.asarray(
            duffing_statevector_plus_plus, dtype=complex
        ),
        circuit_statevector_plus_plus=np.asarray(
            circuit_statevector_plus_plus, dtype=complex
        ),
        effective_populations_plus_plus=np.asarray(
            effective_pop_plus_plus, dtype=float
        ),
        duffing_populations_plus_plus=np.asarray(duffing_pop_plus_plus, dtype=float),
        circuit_populations_plus_plus=np.asarray(circuit_pop_plus_plus, dtype=float),
        effective_intermediate_population_11=obs_effective.monitor_population_11,
        duffing_intermediate_population_11=obs_duffing.monitor_population_11,
        circuit_intermediate_population_11=obs_circuit.monitor_population_11,
        summary=summary,
    )
