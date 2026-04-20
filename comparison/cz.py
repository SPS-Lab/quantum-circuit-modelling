"""CZ-relevant dynamics benchmark for effective, Duffing, and circuit models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from comparison.static import run_static_benchmark
from models import (
    build_circuit_model_stack,
    build_duffing_model_stack,
    build_effective_hamiltonian_stack,
    computational_state_indices,
)
from study_config import StudyConfig


@dataclass(frozen=True)
class _DynamicsObservables:
    computational_amplitudes: np.ndarray
    populations_11: np.ndarray
    leakage_11: np.ndarray
    conditional_phase: np.ndarray


@dataclass(frozen=True)
class CzBenchmarkResult:
    times_ns: np.ndarray
    pulse_flux_values: np.ndarray
    sweep_target: str
    idle_flux: float
    target_flux: float
    effective_populations_11: np.ndarray
    duffing_populations_11: np.ndarray
    circuit_populations_11: np.ndarray
    effective_leakage_11: np.ndarray
    duffing_leakage_11: np.ndarray
    circuit_leakage_11: np.ndarray
    effective_conditional_phase: np.ndarray
    duffing_conditional_phase: np.ndarray
    circuit_conditional_phase: np.ndarray
    summary: dict[str, float]


def _idle_flux_for_target(config: StudyConfig, sweep_target: str) -> float:
    if sweep_target == "q1":
        return float(config.system.q1.flux)
    if sweep_target == "q2":
        return float(config.system.q2.flux)
    if sweep_target == "coupler":
        return 0.0
    raise ValueError(f"Unsupported sweep_target {sweep_target!r}")


def _pick_target_flux_from_static(static_result, *, idle_flux: float) -> float:
    flux = np.asarray(static_result.flux_values, dtype=float).ravel()
    zeta = np.abs(np.asarray(static_result.circuit_parameters["zeta"], dtype=float).ravel())
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
        idx = int(near_max_idx[np.argmin(np.abs(flux[near_max_idx] - float(idle_flux)))])
    else:
        idx = int(np.argmax(zeta_valid))
    target = float(flux[idx])
    if abs(target - float(idle_flux)) < 1e-10:
        alt = int(np.argmax(np.abs(flux - float(idle_flux))))
        target = float(flux[alt])
    return target


def _raised_cosine_cz_pulse(
    times_ns: np.ndarray,
    *,
    idle_flux: float,
    target_flux: float,
    ramp_fraction: float,
) -> np.ndarray:
    t = np.asarray(times_ns, dtype=float).ravel()
    if t.size < 2:
        raise ValueError("times_ns must contain at least 2 points")
    total = float(t[-1] - t[0])
    if total <= 0.0:
        raise ValueError("times_ns must be strictly increasing")

    s = (t - t[0]) / total
    r = float(np.clip(ramp_fraction, 1e-6, 0.499999))

    env = np.ones_like(s, dtype=float)
    rise_mask = s < r
    fall_mask = s > (1.0 - r)
    env[rise_mask] = 0.5 * (1.0 - np.cos(np.pi * s[rise_mask] / r))
    sf = (s[fall_mask] - (1.0 - r)) / r
    env[fall_mask] = 0.5 * (1.0 + np.cos(np.pi * sf))

    return float(idle_flux) + (float(target_flux) - float(idle_flux)) * env


def _interpolate_effective_parameters(
    flux_reference: np.ndarray,
    parameters_reference: dict[str, np.ndarray],
    pulse_flux: np.ndarray,
) -> dict[str, np.ndarray]:
    x_ref = np.asarray(flux_reference, dtype=float).ravel()
    x = np.asarray(pulse_flux, dtype=float).ravel()
    out: dict[str, np.ndarray] = {}
    for key in ("w1", "w2", "J", "zeta"):
        y_ref = np.asarray(parameters_reference[key], dtype=float).ravel()
        out[key] = np.interp(x, x_ref, y_ref)
    return out


def _extract_observables_from_amplitudes(comp_amp: np.ndarray) -> _DynamicsObservables:
    amp = np.asarray(comp_amp, dtype=complex)
    if amp.ndim != 3 or amp.shape[1:] != (4, 4):
        raise ValueError(f"comp_amp must be (n_time, 4, 4), got {amp.shape}")

    populations_11 = np.abs(amp[:, :, 3]) ** 2
    leakage_11 = 1.0 - np.sum(populations_11, axis=1)
    leakage_11 = np.clip(np.real(leakage_11), 0.0, 1.0)

    d00 = amp[:, 0, 0]
    d01 = amp[:, 1, 1]
    d10 = amp[:, 2, 2]
    d11 = amp[:, 3, 3]
    den = d01 * d10
    eps = 1e-15
    den_safe = np.where(np.abs(den) < eps, eps + 0.0j, den)
    cond_phase = np.unwrap(np.angle((d00 * d11) / den_safe))

    return _DynamicsObservables(
        computational_amplitudes=amp,
        populations_11=np.asarray(populations_11, dtype=float),
        leakage_11=np.asarray(leakage_11, dtype=float),
        conditional_phase=np.asarray(cond_phase, dtype=float),
    )


def _simulate_piecewise_constant_scipy(
    H_stack: np.ndarray,
    times_ns: np.ndarray,
    computational_indices: np.ndarray,
) -> _DynamicsObservables:
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
        U = expm((-1.0j * dt) * H[k])
        states = U @ states
        comp_amp[k + 1] = states[comp_idx, :]

    return _extract_observables_from_amplitudes(comp_amp)


def _simulate_piecewise_constant_qutip(
    H_stack: np.ndarray,
    times_ns: np.ndarray,
    computational_indices: np.ndarray,
) -> _DynamicsObservables:
    try:
        import qutip as qt
    except Exception as exc:  # pragma: no cover - import guard only
        raise ImportError("qutip import failed while simulating circuit CZ benchmark") from exc

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
        step_h = qt.Qobj(H[k])
        U = (-1.0j * dt * step_h).expm()
        states = U.full() @ states
        comp_amp[k + 1] = states[comp_idx, :]

    return _extract_observables_from_amplitudes(comp_amp)


def run_cz_benchmark(
    config: StudyConfig,
    *,
    total_time_ns: float = 40.0,
    num_time_points: int = 81,
    ramp_fraction: float = 0.25,
) -> CzBenchmarkResult:
    """Run a CZ-relevant dynamics benchmark under a shared flux pulse schedule.

    Circuit model:
      Hamiltonians from scqubits, piecewise propagation via qutip.
    Effective + Duffing models:
      piecewise propagation via numpy/scipy.
    """
    static_result = run_static_benchmark(config)
    sweep_target = str(config.static_benchmark.flux_control.sweep_target)
    idle_flux = _idle_flux_for_target(config, sweep_target)
    target_flux = _pick_target_flux_from_static(static_result, idle_flux=idle_flux)

    n_time = int(num_time_points)
    if n_time < 2:
        raise ValueError("num_time_points must be at least 2")
    times_ns = np.linspace(0.0, float(total_time_ns), n_time)
    pulse_flux = _raised_cosine_cz_pulse(
        times_ns,
        idle_flux=idle_flux,
        target_flux=target_flux,
        ramp_fraction=ramp_fraction,
    )

    effective_parameters_t = _interpolate_effective_parameters(
        static_result.flux_values,
        static_result.effective_parameters,
        pulse_flux,
    )
    H_effective = build_effective_hamiltonian_stack(effective_parameters_t)

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

    idx_effective = np.array([0, 1, 2, 3], dtype=int)
    idx_duffing = computational_state_indices(
        config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit,
        config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler,
    )
    idx_circuit = computational_state_indices(
        config.static_benchmark.circuit_model.hilbert_truncation.q1_truncated_dim,
        config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim,
    )

    obs_effective = _simulate_piecewise_constant_scipy(H_effective, times_ns, idx_effective)
    obs_duffing = _simulate_piecewise_constant_scipy(duffing_stack, times_ns, idx_duffing)
    obs_circuit = _simulate_piecewise_constant_qutip(circuit_stack, times_ns, idx_circuit)

    eff_pop_rmse = float(
        np.sqrt(np.mean((obs_effective.populations_11 - obs_circuit.populations_11) ** 2))
    )
    duf_pop_rmse = float(
        np.sqrt(np.mean((obs_duffing.populations_11 - obs_circuit.populations_11) ** 2))
    )
    eff_phase_err = float(np.abs(obs_effective.conditional_phase[-1] - obs_circuit.conditional_phase[-1]))
    duf_phase_err = float(np.abs(obs_duffing.conditional_phase[-1] - obs_circuit.conditional_phase[-1]))

    summary = {
        "effective_final_conditional_phase_rad": float(obs_effective.conditional_phase[-1]),
        "duffing_final_conditional_phase_rad": float(obs_duffing.conditional_phase[-1]),
        "circuit_final_conditional_phase_rad": float(obs_circuit.conditional_phase[-1]),
        "effective_final_phase_error_vs_circuit_rad": eff_phase_err,
        "duffing_final_phase_error_vs_circuit_rad": duf_phase_err,
        "effective_max_leakage_11": float(np.max(obs_effective.leakage_11)),
        "duffing_max_leakage_11": float(np.max(obs_duffing.leakage_11)),
        "circuit_max_leakage_11": float(np.max(obs_circuit.leakage_11)),
        "effective_populations_rmse_vs_circuit": eff_pop_rmse,
        "duffing_populations_rmse_vs_circuit": duf_pop_rmse,
    }

    return CzBenchmarkResult(
        times_ns=np.asarray(times_ns, dtype=float),
        pulse_flux_values=np.asarray(pulse_flux, dtype=float),
        sweep_target=sweep_target,
        idle_flux=float(idle_flux),
        target_flux=float(target_flux),
        effective_populations_11=obs_effective.populations_11,
        duffing_populations_11=obs_duffing.populations_11,
        circuit_populations_11=obs_circuit.populations_11,
        effective_leakage_11=obs_effective.leakage_11,
        duffing_leakage_11=obs_duffing.leakage_11,
        circuit_leakage_11=obs_circuit.leakage_11,
        effective_conditional_phase=obs_effective.conditional_phase,
        duffing_conditional_phase=obs_duffing.conditional_phase,
        circuit_conditional_phase=obs_circuit.conditional_phase,
        summary=summary,
    )
