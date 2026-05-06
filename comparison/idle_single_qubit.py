"""Idle single-qubit benchmark across circuit, Duffing, and effective models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.duffing import _transmon_analytic_w01_alpha, _transmon_w01_alpha
from models.josephson import flux_dependent_EJ
from study_config import StudyConfig, TransmonSystemParams
from toolkit.helpers import destroy

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class IdleSingleQubitBenchmarkResult:
    case_label: str
    qubit: str
    flux: float
    effective_source: str
    initial_state_label: str
    total_time_ns: float
    dt_ns: float
    times_ns: np.ndarray
    circuit_relative_energies: np.ndarray
    duffing_relative_energies: np.ndarray
    effective_relative_energies: np.ndarray
    circuit_logical_amplitudes: np.ndarray
    duffing_logical_amplitudes: np.ndarray
    effective_logical_amplitudes: np.ndarray
    circuit_bloch_x: np.ndarray
    duffing_bloch_x: np.ndarray
    effective_bloch_x: np.ndarray
    circuit_bloch_y: np.ndarray
    duffing_bloch_y: np.ndarray
    effective_bloch_y: np.ndarray
    circuit_bloch_z: np.ndarray
    duffing_bloch_z: np.ndarray
    effective_bloch_z: np.ndarray
    circuit_population_0: np.ndarray
    duffing_population_0: np.ndarray
    effective_population_0: np.ndarray
    circuit_population_1: np.ndarray
    duffing_population_1: np.ndarray
    effective_population_1: np.ndarray
    circuit_relative_phase_cycles: np.ndarray
    duffing_relative_phase_cycles: np.ndarray
    effective_relative_phase_cycles: np.ndarray
    circuit_logical_leakage: np.ndarray
    duffing_logical_leakage: np.ndarray
    effective_logical_leakage: np.ndarray
    summary: dict[str, float | int | str]


@dataclass(frozen=True)
class _SingleQubitEvolution:
    relative_energies: np.ndarray
    logical_amplitudes: np.ndarray
    population_0: np.ndarray
    population_1: np.ndarray
    bloch_x: np.ndarray
    bloch_y: np.ndarray
    bloch_z: np.ndarray
    relative_phase_cycles: np.ndarray
    logical_leakage: np.ndarray
    w01: float
    alpha: float
    dim: int


def _require_scqubits_module():
    try:
        import scqubits as scq
    except Exception as exc:  # pragma: no cover - import guard only
        raise ImportError("scqubits import failed while building isolated single-qubit models") from exc
    return scq


def _resolve_qubit(config: StudyConfig, requested: str | None) -> str:
    if requested is not None:
        name = str(requested).strip().lower()
    else:
        sweep_target = str(config.static_benchmark.flux_control.sweep_target).strip().lower()
        name = sweep_target if sweep_target in {"q1", "q2"} else "q1"
    if name not in {"q1", "q2"}:
        raise ValueError(f"qubit must be 'q1' or 'q2', got {requested!r}")
    return name


def _selected_qubit_params(config: StudyConfig, qubit: str) -> tuple[TransmonSystemParams, int]:
    if qubit == "q1":
        return (
            config.system.q1,
            int(config.static_benchmark.circuit_model.hilbert_truncation.q1_truncated_dim),
        )
    if qubit == "q2":
        return (
            config.system.q2,
            int(config.static_benchmark.circuit_model.hilbert_truncation.q2_truncated_dim),
        )
    raise ValueError(f"Unsupported qubit {qubit!r}")


def _build_toy_lc_hamiltonian(
    *,
    target_w01_ghz: float,
    truncated_dim: int,
) -> np.ndarray:
    nlevels = int(max(3, truncated_dim))
    a = destroy(nlevels)
    adag = a.conj().T
    eye = np.eye(nlevels, dtype=complex)
    omega = float(target_w01_ghz)
    return np.asarray(omega * (adag @ a + 0.5 * eye), dtype=complex)


def _time_grid(total_time_ns: float, dt_ns: float) -> np.ndarray:
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


def _as_dense_matrix(operator_like) -> np.ndarray:
    if hasattr(operator_like, "full"):
        return np.asarray(operator_like.full(), dtype=complex)
    return np.asarray(operator_like, dtype=complex)


def _build_circuit_hamiltonian(
    qubit_params: TransmonSystemParams,
    *,
    truncated_dim: int,
) -> np.ndarray:
    scq = _require_scqubits_module()
    transmon = scq.TunableTransmon(
        EJmax=float(qubit_params.EJmax),
        EC=float(qubit_params.EC),
        d=float(qubit_params.d),
        flux=float(qubit_params.flux),
        ng=float(qubit_params.ng),
        ncut=int(qubit_params.ncut),
        truncated_dim=int(truncated_dim),
        id_str=str(qubit_params.id_str),
    )
    return _as_dense_matrix(transmon.hamiltonian())


def _duffing_w01_alpha(config: StudyConfig, qubit: str, qubit_params: TransmonSystemParams) -> tuple[float, float]:
    calibration_mode = str(config.static_benchmark.duffing_model.calibration_mode).strip().lower()
    extraction = config.static_benchmark.duffing_model.transmon_spectral_extraction
    EJ = float(flux_dependent_EJ(qubit_params.EJmax, qubit_params.flux, qubit_params.d))

    if calibration_mode == "analytic-per-flux":
        w01_arr, alpha_arr = _transmon_analytic_w01_alpha(np.array([EJ], dtype=float), qubit_params.EC)
        return float(w01_arr[0]), float(alpha_arr[0])

    if calibration_mode in {"fixed", "per-flux"}:
        return _transmon_w01_alpha(
            EJ,
            qubit_params.EC,
            qubit_params.ng,
            int(extraction.ncut),
            int(extraction.truncated_dim),
        )

    raise ValueError(f"Unsupported Duffing calibration mode {calibration_mode!r} for isolated {qubit}")


def _build_duffing_hamiltonian(
    config: StudyConfig,
    qubit: str,
    qubit_params: TransmonSystemParams,
) -> tuple[np.ndarray, float, float]:
    w01, alpha = _duffing_w01_alpha(config, qubit, qubit_params)
    nlevels = int(config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit)
    a = destroy(nlevels)
    adag = a.conj().T
    H = float(w01) * (adag @ a) + (float(alpha) / 2.0) * (adag @ adag @ a @ a)
    return np.asarray(H, dtype=complex), float(w01), float(alpha)


def _build_effective_hamiltonian(w01: float, *, zero_point_offset: float = 0.0) -> np.ndarray:
    e0 = float(zero_point_offset)
    return np.diag([e0, e0 + float(w01)]).astype(complex)


def _simulate_static_single_qubit(
    H: np.ndarray,
    times_ns: np.ndarray,
    *,
    initial_state: str,
) -> _SingleQubitEvolution:
    H_arr = np.asarray(H, dtype=complex)
    t = np.asarray(times_ns, dtype=float).ravel()
    if H_arr.ndim != 2 or H_arr.shape[0] != H_arr.shape[1]:
        raise ValueError(f"H must be square 2D, got {H_arr.shape}")
    if t.ndim != 1 or t.size < 2:
        raise ValueError("times_ns must be a 1D array with at least two samples")

    evals, evecs = np.linalg.eigh(H_arr)
    evals = np.asarray(evals, dtype=float)
    relative_energies = np.asarray(evals - evals[0], dtype=float)

    ground = np.asarray(evecs[:, 0], dtype=complex)
    excited = np.asarray(evecs[:, 1], dtype=complex)
    init = str(initial_state).strip().lower()
    if init == "plus":
        psi0 = (ground + excited) / np.sqrt(2.0)
    elif init == "ground":
        psi0 = ground
    elif init == "excited":
        psi0 = excited
    else:
        raise ValueError(f"Unsupported initial_state {initial_state!r}")
    coeff0 = evecs.conj().T @ psi0
    phase = np.exp(-1.0j * TWO_PI * evals[:, np.newaxis] * t[np.newaxis, :])
    psi_t = evecs @ (coeff0[:, np.newaxis] * phase)

    amp0 = ground.conj() @ psi_t
    amp1 = excited.conj() @ psi_t
    logical_amplitudes = np.column_stack([amp0, amp1])
    pop0 = np.abs(amp0) ** 2
    pop1 = np.abs(amp1) ** 2
    leak = np.clip(1.0 - pop0 - pop1, 0.0, 1.0)

    coherence = np.conjugate(amp0) * amp1
    bloch_x = 2.0 * np.real(coherence)
    bloch_y = 2.0 * np.imag(coherence)
    bloch_z = pop1 - pop0
    relative_phase_cycles = np.unwrap(np.angle(amp1 * np.conjugate(amp0))) / TWO_PI

    alpha = 0.0
    if evals.size >= 3:
        alpha = float((evals[2] - evals[1]) - (evals[1] - evals[0]))

    return _SingleQubitEvolution(
        relative_energies=relative_energies,
        logical_amplitudes=np.asarray(logical_amplitudes, dtype=complex),
        population_0=np.asarray(pop0, dtype=float),
        population_1=np.asarray(pop1, dtype=float),
        bloch_x=np.asarray(bloch_x, dtype=float),
        bloch_y=np.asarray(bloch_y, dtype=float),
        bloch_z=np.asarray(bloch_z, dtype=float),
        relative_phase_cycles=np.asarray(relative_phase_cycles, dtype=float),
        logical_leakage=np.asarray(leak, dtype=float),
        w01=float(evals[1] - evals[0]),
        alpha=alpha,
        dim=int(H_arr.shape[0]),
    )


def run_idle_single_qubit_benchmark(
    config: StudyConfig,
    *,
    qubit: str | None = None,
    total_time_ns: float = 2.0,
    dt_ns: float = 0.001,
    toy_w01_ghz: float = 5.0,
    initial_state: str = "plus",
) -> IdleSingleQubitBenchmarkResult:
    times_ns = _time_grid(total_time_ns, dt_ns)
    effective_source = "circuit"
    if qubit is None:
        case_label = f"{float(toy_w01_ghz):g}ghz"
        selected_qubit = "toy"
        flux = 0.0
        q1_trunc = int(config.static_benchmark.circuit_model.hilbert_truncation.q1_truncated_dim)
        H_circuit = _build_toy_lc_hamiltonian(
            target_w01_ghz=float(toy_w01_ghz),
            truncated_dim=q1_trunc,
        )
        circuit = _simulate_static_single_qubit(H_circuit, times_ns, initial_state=initial_state)
        duffing_alpha = 0.0
        nlevels = int(config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit)
        a = destroy(nlevels)
        adag = a.conj().T
        omega = float(toy_w01_ghz)
        H_duffing = omega * (adag @ a + 0.5 * np.eye(nlevels, dtype=complex)) + (duffing_alpha / 2.0) * (adag @ adag @ a @ a)
        duffing = _simulate_static_single_qubit(np.asarray(H_duffing, dtype=complex), times_ns, initial_state=initial_state)
        H_effective = _build_effective_hamiltonian(float(toy_w01_ghz), zero_point_offset=0.5 * float(toy_w01_ghz))
        effective = _simulate_static_single_qubit(H_effective, times_ns, initial_state=initial_state)
        summary = {
            "case_label": case_label,
            "qubit": selected_qubit,
            "flux": flux,
            "effective_source": effective_source,
            "toy_target_w01_ghz": float(toy_w01_ghz),
            "toy_alpha_ghz": 0.0,
            "circuit_dim": int(circuit.dim),
            "duffing_dim": int(duffing.dim),
            "effective_dim": int(effective.dim),
            "circuit_w01_ghz": float(circuit.w01),
            "duffing_w01_ghz": float(duffing.w01),
            "effective_w01_ghz": float(effective.w01),
            "circuit_alpha_ghz": float(circuit.alpha),
            "duffing_alpha_ghz": float(duffing.alpha),
            "effective_alpha_ghz": float(effective.alpha),
            "duffing_minus_circuit_w01_mhz": float(1.0e3 * (duffing.w01 - circuit.w01)),
            "effective_minus_circuit_w01_mhz": float(1.0e3 * (effective.w01 - circuit.w01)),
            "max_circuit_logical_leakage": float(np.max(circuit.logical_leakage)),
            "max_duffing_logical_leakage": float(np.max(duffing.logical_leakage)),
            "max_effective_logical_leakage": float(np.max(effective.logical_leakage)),
        }
    else:
        selected_qubit = _resolve_qubit(config, qubit)
        case_label = selected_qubit
        qubit_params, circuit_truncated_dim = _selected_qubit_params(config, selected_qubit)
        flux = float(qubit_params.flux)

        H_circuit = _build_circuit_hamiltonian(qubit_params, truncated_dim=circuit_truncated_dim)
        H_duffing, duffing_w01, _ = _build_duffing_hamiltonian(config, selected_qubit, qubit_params)

        circuit = _simulate_static_single_qubit(H_circuit, times_ns, initial_state=initial_state)
        duffing = _simulate_static_single_qubit(H_duffing, times_ns, initial_state=initial_state)

        effective_source = str(config.static_benchmark.effective_model.derivation_source).strip().lower()
        if effective_source == "circuit":
            effective_w01 = circuit.w01
        elif effective_source == "duffing":
            effective_w01 = duffing_w01
        else:
            raise ValueError(f"Unsupported effective derivation source {effective_source!r}")

        H_effective = _build_effective_hamiltonian(effective_w01)
        effective = _simulate_static_single_qubit(H_effective, times_ns, initial_state=initial_state)
        summary = {
            "case_label": case_label,
            "qubit": selected_qubit,
            "flux": flux,
            "effective_source": effective_source,
            "circuit_dim": int(circuit.dim),
            "duffing_dim": int(duffing.dim),
            "effective_dim": int(effective.dim),
            "circuit_w01_ghz": float(circuit.w01),
            "duffing_w01_ghz": float(duffing.w01),
            "effective_w01_ghz": float(effective.w01),
            "circuit_alpha_ghz": float(circuit.alpha),
            "duffing_alpha_ghz": float(duffing.alpha),
            "effective_alpha_ghz": float(effective.alpha),
            "duffing_minus_circuit_w01_mhz": float(1.0e3 * (duffing.w01 - circuit.w01)),
            "effective_minus_circuit_w01_mhz": float(1.0e3 * (effective.w01 - circuit.w01)),
            "max_circuit_logical_leakage": float(np.max(circuit.logical_leakage)),
            "max_duffing_logical_leakage": float(np.max(duffing.logical_leakage)),
            "max_effective_logical_leakage": float(np.max(effective.logical_leakage)),
        }

    return IdleSingleQubitBenchmarkResult(
        case_label=case_label,
        qubit=selected_qubit,
        flux=float(flux),
        effective_source=effective_source,
        initial_state_label={
            "plus": "(|0> + |1>) / sqrt(2) in each model's lowest two eigenstates",
            "ground": "|0> in each model's lowest two eigenstates",
            "excited": "|1> in each model's lowest two eigenstates",
        }[str(initial_state).strip().lower()],
        total_time_ns=float(total_time_ns),
        dt_ns=float(dt_ns),
        times_ns=times_ns,
        circuit_relative_energies=circuit.relative_energies,
        duffing_relative_energies=duffing.relative_energies,
        effective_relative_energies=effective.relative_energies,
        circuit_logical_amplitudes=circuit.logical_amplitudes,
        duffing_logical_amplitudes=duffing.logical_amplitudes,
        effective_logical_amplitudes=effective.logical_amplitudes,
        circuit_bloch_x=circuit.bloch_x,
        duffing_bloch_x=duffing.bloch_x,
        effective_bloch_x=effective.bloch_x,
        circuit_bloch_y=circuit.bloch_y,
        duffing_bloch_y=duffing.bloch_y,
        effective_bloch_y=effective.bloch_y,
        circuit_bloch_z=circuit.bloch_z,
        duffing_bloch_z=duffing.bloch_z,
        effective_bloch_z=effective.bloch_z,
        circuit_population_0=circuit.population_0,
        duffing_population_0=duffing.population_0,
        effective_population_0=effective.population_0,
        circuit_population_1=circuit.population_1,
        duffing_population_1=duffing.population_1,
        effective_population_1=effective.population_1,
        circuit_relative_phase_cycles=circuit.relative_phase_cycles,
        duffing_relative_phase_cycles=duffing.relative_phase_cycles,
        effective_relative_phase_cycles=effective.relative_phase_cycles,
        circuit_logical_leakage=circuit.logical_leakage,
        duffing_logical_leakage=duffing.logical_leakage,
        effective_logical_leakage=effective.logical_leakage,
        summary=summary,
    )
