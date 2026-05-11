"""Static benchmark: effective vs Duffing vs circuit model across flux."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models import (
    build_circuit_model_stack,
    build_dressed_effective_computational_stack,
    build_duffing_model_stack,
    build_duffing_model_stack_from_parameters,
    build_effective_hamiltonian_stack,
    derive_effective_model_from_dressed_stack,
    extract_model1_parameters_from_4x4_stack,
    fit_duffing_mode_parameters_to_reference,
    fit_symbolic_duffing_mode_parameters_to_reference,
    resolve_static_sweep_values,
)
from study_config import StudyConfig, build_flux_values
from toolkit.spectrum import track_energy_levels_stack


@dataclass(frozen=True)
class StaticBenchmarkResult:
    flux_values: np.ndarray
    effective_relative_energies: np.ndarray
    duffing_relative_energies: np.ndarray
    circuit_relative_energies: np.ndarray
    duffing_full_relative_energies: np.ndarray
    circuit_full_relative_energies: np.ndarray
    effective_error_rmse: np.ndarray
    duffing_error_rmse: np.ndarray
    effective_parameters: dict[str, np.ndarray]
    effective_fit_coefficient_names: dict[str, np.ndarray]
    effective_fit_coefficients: dict[str, np.ndarray]
    duffing_mode_parameters: dict[str, np.ndarray]
    duffing_symbolic_coefficient_names: dict[str, np.ndarray]
    duffing_symbolic_coefficients: dict[str, np.ndarray]
    duffing_parameters: dict[str, np.ndarray]
    circuit_parameters: dict[str, np.ndarray]
    detuning_ratio: np.ndarray
    idle_mask: np.ndarray
    near_mask: np.ndarray
    summary: dict[str, float]



def _relative_energies(
    H_stack: np.ndarray,
    *,
    n_track: int,
    projector_track_single_excitation: bool = False,
) -> np.ndarray:
    blocks = ((1, 2),) if projector_track_single_excitation and int(n_track) >= 3 else None
    evals = track_energy_levels_stack(
        np.asarray(H_stack, dtype=complex),
        n_track=int(n_track),
        projector_blocks=blocks,
    )
    return np.asarray(evals - evals[:, :1], dtype=float)



def _per_flux_rmse(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    diff = np.asarray(pred, dtype=float)[:, 1:] - np.asarray(ref, dtype=float)[:, 1:]
    return np.sqrt(np.mean(diff * diff, axis=1))



def _masked_rmse_max(err: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return float("nan"), float("nan")
    values = np.asarray(err, dtype=float)[mask]
    return float(np.sqrt(np.mean(values * values))), float(np.max(np.abs(values)))



def run_static_benchmark(config: StudyConfig) -> StaticBenchmarkResult:
    flux_values = build_flux_values(config.static_benchmark.flux_sweep)

    circuit = build_circuit_model_stack(
        flux_values=flux_values,
        system_params=config.system,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        circuit_config=config.static_benchmark.circuit_model,
        sweep_target=config.static_benchmark.flux_control.sweep_target,
    )

    dressed_mode = config.static_benchmark.dressed_subspace.selection_mode
    n_cand = config.static_benchmark.dressed_subspace.n_candidate_states

    H_circuit_eff = build_dressed_effective_computational_stack(
        circuit.hamiltonian_stack,
        nlevels_qubit=config.static_benchmark.circuit_model.hilbert_truncation.q0_truncated_dim,
        nlevels_coupler=config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim,
        selection_mode=dressed_mode,
        n_candidate_states=n_cand,
    )

    duffing_mode = str(config.static_benchmark.duffing_model.calibration_mode).strip().lower()
    duffing_symbolic_coefficient_names: dict[str, np.ndarray] = {}
    duffing_symbolic_coefficients: dict[str, np.ndarray] = {}
    if duffing_mode == "fitted-static":
        duffing_mode_parameters = fit_duffing_mode_parameters_to_reference(
            flux_values=flux_values,
            reference_dressed_stack=H_circuit_eff,
            system_params=config.system,
            coupler_frequency=config.static_benchmark.coupler_frequency,
            duffing_config=config.static_benchmark.duffing_model,
            sweep_target=config.static_benchmark.flux_control.sweep_target,
            selection_mode=dressed_mode,
            n_candidate_states=n_cand,
        )
        duffing = build_duffing_model_stack_from_parameters(
            duffing_mode_parameters,
            system_params=config.system,
            duffing_config=config.static_benchmark.duffing_model,
        )
    elif duffing_mode == "symbolic-fitted-static":
        duffing_symbolic_fit = fit_symbolic_duffing_mode_parameters_to_reference(
            flux_values=flux_values,
            reference_dressed_stack=H_circuit_eff,
            system_params=config.system,
            coupler_frequency=config.static_benchmark.coupler_frequency,
            duffing_config=config.static_benchmark.duffing_model,
            sweep_target=config.static_benchmark.flux_control.sweep_target,
            selection_mode=dressed_mode,
            n_candidate_states=n_cand,
        )
        duffing_mode_parameters = duffing_symbolic_fit.fitted_parameters
        duffing_symbolic_coefficient_names = duffing_symbolic_fit.coefficient_names
        duffing_symbolic_coefficients = duffing_symbolic_fit.coefficients
        duffing = build_duffing_model_stack_from_parameters(
            duffing_mode_parameters,
            system_params=config.system,
            duffing_config=config.static_benchmark.duffing_model,
        )
    else:
        duffing = build_duffing_model_stack(
            flux_values=flux_values,
            system_params=config.system,
            coupler_frequency=config.static_benchmark.coupler_frequency,
            duffing_config=config.static_benchmark.duffing_model,
            sweep_target=config.static_benchmark.flux_control.sweep_target,
        )
        duffing_mode_parameters = duffing.mode_parameters

    H_duffing_eff = build_dressed_effective_computational_stack(
        duffing.hamiltonian_stack,
        nlevels_qubit=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit,
        nlevels_coupler=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler,
        selection_mode=dressed_mode,
        n_candidate_states=n_cand,
    )

    source = config.static_benchmark.effective_model.derivation_source
    if source == "duffing":
        source_stack = H_duffing_eff
    elif source == "circuit":
        source_stack = H_circuit_eff
    else:
        raise ValueError(f"Unsupported effective derivation source {source!r}")

    _, _, wc = resolve_static_sweep_values(
        flux_values,
        system_params=config.system,
        coupler_frequency_config=config.static_benchmark.coupler_frequency,
        sweep_target=config.static_benchmark.flux_control.sweep_target,
    )

    derivation = derive_effective_model_from_dressed_stack(
        flux_values=flux_values,
        dressed_stack=source_stack,
        fit_basis=config.static_benchmark.effective_model.fit_basis,
        coupler_frequency_values=wc,
    )

    effective_parameters = derivation.parameter_fit.fitted_parameters
    effective_fit_coefficient_names = {
        name: np.asarray(derivation.parameter_fit.coefficient_names[name], dtype=str)
        for name in ("J", "zeta")
    }
    effective_fit_coefficients = {
        name: np.asarray(derivation.parameter_fit.coefficients[name], dtype=float)
        for name in ("J", "zeta")
    }
    H_effective = build_effective_hamiltonian_stack(effective_parameters)

    n_track = int(H_effective.shape[-1])
    E_eff = _relative_energies(H_effective, n_track=n_track, projector_track_single_excitation=True)
    E_duf = _relative_energies(H_duffing_eff, n_track=n_track, projector_track_single_excitation=True)
    E_cir = _relative_energies(H_circuit_eff, n_track=n_track, projector_track_single_excitation=True)

    n_full_track = int(min(10, duffing.hamiltonian_stack.shape[1], circuit.hamiltonian_stack.shape[1]))
    E_duf_full = _relative_energies(duffing.hamiltonian_stack, n_track=n_full_track)
    E_cir_full = _relative_energies(circuit.hamiltonian_stack, n_track=n_full_track)

    err_eff = _per_flux_rmse(E_eff, E_cir)
    err_duf = _per_flux_rmse(E_duf, E_cir)

    params_duffing = extract_model1_parameters_from_4x4_stack(H_duffing_eff)
    params_circuit = extract_model1_parameters_from_4x4_stack(H_circuit_eff)

    d1 = np.abs(params_circuit["w0"] - wc)
    d2 = np.abs(params_circuit["w1"] - wc)
    g_scale = max(abs(config.system.interactions.g_0c), abs(config.system.interactions.g_1c), 1e-12)
    detuning_ratio = np.minimum(d1, d2) / g_scale

    idle_thr = float(config.static_benchmark.regime_thresholds.idle_ratio)
    near_thr = float(config.static_benchmark.regime_thresholds.near_ratio)
    idle_mask = detuning_ratio >= idle_thr
    near_mask = detuning_ratio <= near_thr

    eff_idle = _masked_rmse_max(err_eff, idle_mask)
    eff_near = _masked_rmse_max(err_eff, near_mask)
    duf_idle = _masked_rmse_max(err_duf, idle_mask)
    duf_near = _masked_rmse_max(err_duf, near_mask)

    summary = {
        "effective_idle_rmse": eff_idle[0],
        "effective_idle_max_abs": eff_idle[1],
        "effective_near_rmse": eff_near[0],
        "effective_near_max_abs": eff_near[1],
        "duffing_idle_rmse": duf_idle[0],
        "duffing_idle_max_abs": duf_idle[1],
        "duffing_near_rmse": duf_near[0],
        "duffing_near_max_abs": duf_near[1],
    }

    return StaticBenchmarkResult(
        flux_values=flux_values,
        effective_relative_energies=E_eff,
        duffing_relative_energies=E_duf,
        circuit_relative_energies=E_cir,
        duffing_full_relative_energies=E_duf_full,
        circuit_full_relative_energies=E_cir_full,
        effective_error_rmse=err_eff,
        duffing_error_rmse=err_duf,
        effective_parameters={k: np.asarray(v, dtype=float) for k, v in effective_parameters.items()},
        effective_fit_coefficient_names=effective_fit_coefficient_names,
        effective_fit_coefficients=effective_fit_coefficients,
        duffing_mode_parameters={k: np.asarray(v, dtype=float) for k, v in duffing_mode_parameters.items()},
        duffing_symbolic_coefficient_names={
            k: np.asarray(v, dtype=str) for k, v in duffing_symbolic_coefficient_names.items()
        },
        duffing_symbolic_coefficients={
            k: np.asarray(v, dtype=float) for k, v in duffing_symbolic_coefficients.items()
        },
        duffing_parameters=params_duffing,
        circuit_parameters=params_circuit,
        detuning_ratio=np.asarray(detuning_ratio, dtype=float),
        idle_mask=np.asarray(idle_mask, dtype=bool),
        near_mask=np.asarray(near_mask, dtype=bool),
        summary=summary,
    )
