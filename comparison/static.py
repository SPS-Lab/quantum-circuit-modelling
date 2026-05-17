"""Static benchmark: effective vs Duffing vs circuit model across flux."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models import (
    build_circuit_model_stack,
    build_dressed_effective_computational_stack,
    build_duffing_model_stack_from_coefficients,
    build_duffing_model_stack_from_parameters,
    build_duffing_model_stack_from_scratch,
    build_effective_hamiltonian_stack,
    canonical_state_order_qcq,
    derive_effective_model_from_dressed_stack,
    extract_effective_model_parameters_from_4x4_stack,
    fit_duffing_mode_parameters_to_reference,
    fit_symbolic_duffing_mode_parameters_to_reference,
    resolve_static_sweep_values,
    tracked_bare_state_amplitudes,
    tracked_subspace_bare_amplitudes,
    tracked_subspace_bare_overlaps,
)
from study_config import StudyConfig, build_flux_values
from runtime_utils import progress_heartbeat
from toolkit.spectrum import track_energy_levels_stack


@dataclass(frozen=True)
class StaticBenchmarkResult:
    flux_values: np.ndarray
    effective_raw_energies: np.ndarray
    effective_relative_energies: np.ndarray
    duffing_raw_energies: np.ndarray
    duffing_relative_energies: np.ndarray
    circuit_raw_energies: np.ndarray
    circuit_relative_energies: np.ndarray
    duffing_full_raw_energies: np.ndarray
    duffing_full_relative_energies: np.ndarray
    circuit_full_raw_energies: np.ndarray
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
    duffing_bare_state_labels: np.ndarray
    duffing_tracked_branch_bare_amplitudes: np.ndarray
    duffing_computational_bare_amplitudes: np.ndarray
    duffing_computational_bare_overlaps: np.ndarray
    circuit_bare_state_labels: np.ndarray
    circuit_tracked_branch_bare_amplitudes: np.ndarray
    circuit_computational_bare_amplitudes: np.ndarray
    circuit_computational_bare_overlaps: np.ndarray
    detuning_ratio: np.ndarray
    idle_mask: np.ndarray
    near_mask: np.ndarray
    summary: dict[str, float]
    metric_notes: dict[str, str]


def _raw_energies(
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
    return np.asarray(evals, dtype=float)

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

def _sorted_raw_energies(H_stack: np.ndarray, *, n_track: int) -> np.ndarray:
    evals = np.linalg.eigvalsh(np.asarray(H_stack, dtype=complex))
    rel_e = np.asarray(evals, dtype=float)
    return np.asarray(rel_e[:, : int(n_track)], dtype=float)

def _sorted_relative_energies(H_stack: np.ndarray, *, n_track: int) -> np.ndarray:
    evals = np.linalg.eigvalsh(np.asarray(H_stack, dtype=complex))
    rel_e = np.asarray(evals - evals[:, :1], dtype=float)
    return np.asarray(rel_e[:, : int(n_track)], dtype=float)


def _aggregate_rmse(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    n_excited: int | None = None,
) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    max_excited = min(int(pred_arr.shape[1]), int(ref_arr.shape[1])) - 1
    n_use = max_excited if n_excited is None else min(max(0, int(n_excited)), max_excited)
    if n_use <= 0:
        return 0.0
    diff = pred_arr[:, 1 : 1 + n_use] - ref_arr[:, 1 : 1 + n_use]
    return float(np.sqrt(np.mean(diff * diff)))



def _masked_rmse_max(err: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return float("nan"), float("nan")
    values = np.asarray(err, dtype=float)[mask]
    return float(np.sqrt(np.mean(values * values))), float(np.max(np.abs(values)))


def _mean_abs_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    diff = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    return float(np.mean(np.abs(diff)))



def run_static_benchmark(config: StudyConfig) -> StaticBenchmarkResult:
    flux_values = build_flux_values(config.static_benchmark.flux_sweep)

    with progress_heartbeat("static benchmark: build_circuit_model_stack"):
        circuit = build_circuit_model_stack(
            flux_values=flux_values,
            system_params=config.system,
            circuit_config=config.static_benchmark.circuit_model,
            sweep_target=config.static_benchmark.flux_control.sweep_target,
        )

    dressed_mode = config.static_benchmark.dressed_subspace.selection_mode
    n_cand = config.static_benchmark.dressed_subspace.n_candidate_states

    with progress_heartbeat("static benchmark: dress circuit stack"):
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
        with progress_heartbeat("static benchmark: fit duffing parameters to circuit reference"):
            duffing_mode_parameters = fit_duffing_mode_parameters_to_reference(
                flux_values=flux_values,
                reference_dressed_stack=H_circuit_eff,
                system_params=config.system,
                duffing_config=config.static_benchmark.duffing_model,
                sweep_target=config.static_benchmark.flux_control.sweep_target,
                selection_mode=dressed_mode,
                n_candidate_states=n_cand,
            )
        with progress_heartbeat("static benchmark: build_duffing_model_stack_from_parameters"):
            duffing = build_duffing_model_stack_from_parameters(
                duffing_mode_parameters,
                system_params=config.system,
                duffing_config=config.static_benchmark.duffing_model,
            )
    elif duffing_mode == "symbolic-fitted-static":
        symbolic_fit_cfg = config.static_benchmark.duffing_model.symbolic_fit
        if symbolic_fit_cfg is None:
            raise ValueError(
                "symbolic-fitted-static requires static_benchmark.duffing_model.symbolic_fit "
                "settings in the study config"
            )
        with progress_heartbeat("static benchmark: symbolic Duffing fit to circuit reference"):
            duffing_symbolic_fit = fit_symbolic_duffing_mode_parameters_to_reference(
                flux_values=flux_values,
                reference_dressed_stack=H_circuit_eff,
                system_params=config.system,
                duffing_config=config.static_benchmark.duffing_model,
                sweep_target=config.static_benchmark.flux_control.sweep_target,
                selection_mode=dressed_mode,
                n_candidate_states=n_cand,
                max_harmonics_w=int(symbolic_fit_cfg.max_harmonics_w),
                max_harmonics_alpha=int(symbolic_fit_cfg.max_harmonics_alpha),
                max_harmonics_g=int(symbolic_fit_cfg.max_harmonics_g),
                pointwise_max_nfev=int(symbolic_fit_cfg.pointwise_max_nfev),
                refinement_max_nfev=int(symbolic_fit_cfg.refinement_max_nfev),
                regularization_weight=float(symbolic_fit_cfg.regularization_weight),
                progress_label="static benchmark: symbolic Duffing fit to circuit reference",
            )
        duffing_symbolic_coefficient_names = duffing_symbolic_fit.coefficient_names
        duffing_symbolic_coefficients = duffing_symbolic_fit.coefficients
        with progress_heartbeat("static benchmark: build_duffing_model_stack_from_coefficients"):
            duffing = build_duffing_model_stack_from_coefficients(
                flux_values,
                system_params=config.system,
                duffing_config=config.static_benchmark.duffing_model,
                sweep_target=config.static_benchmark.flux_control.sweep_target,
                coefficient_names=duffing_symbolic_coefficient_names,
                coefficients=duffing_symbolic_coefficients,
            )
        duffing_mode_parameters = duffing.mode_parameters
    else:
        with progress_heartbeat("static benchmark: build_duffing_model_stack_from_scratch"):
            duffing = build_duffing_model_stack_from_scratch(
                flux_values=flux_values,
                system_params=config.system,
                duffing_config=config.static_benchmark.duffing_model,
                sweep_target=config.static_benchmark.flux_control.sweep_target,
            )
        duffing_mode_parameters = duffing.mode_parameters

    with progress_heartbeat("static benchmark: dress Duffing stack"):
        H_duffing_eff = build_dressed_effective_computational_stack(
            duffing.hamiltonian_stack,
            nlevels_qubit=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit,
            nlevels_coupler=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler,
            selection_mode=dressed_mode,
            n_candidate_states=n_cand,
        )
    circuit_q0_dim = int(config.static_benchmark.circuit_model.hilbert_truncation.q0_truncated_dim)
    circuit_q1_dim = int(config.static_benchmark.circuit_model.hilbert_truncation.q1_truncated_dim)
    circuit_c_dim = int(config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim)
    circuit_overlap_subspace_idx = np.array([0, 1, circuit_c_dim * circuit_q0_dim + 0, circuit_c_dim * circuit_q0_dim + 1], dtype=int)
    duffing_q_dim = int(config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit)
    duffing_c_dim = int(config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler)
    duffing_overlap_subspace_idx = np.array([0, 1, duffing_c_dim * duffing_q_dim + 0, duffing_c_dim * duffing_q_dim + 1], dtype=int)
    circuit_state_order_idx, circuit_state_labels = canonical_state_order_qcq(
        nlevels_q0=circuit_q0_dim,
        nlevels_coupler=circuit_c_dim,
        nlevels_q1=circuit_q1_dim,
    )
    duffing_state_order_idx, duffing_state_labels = canonical_state_order_qcq(
        nlevels_q0=duffing_q_dim,
        nlevels_coupler=duffing_c_dim,
    )
    with progress_heartbeat("static benchmark: tracked_subspace_bare_overlaps, Circuit"):
        circuit_comp_overlaps = tracked_subspace_bare_overlaps(
            circuit.hamiltonian_stack,
            subspace_indices=circuit_overlap_subspace_idx,
            selection_mode=dressed_mode,
            n_candidate_states=n_cand,
            projector_blocks=((1, 2),),
        )
    with progress_heartbeat("static benchmark: tracked_subspace_bare_amplitudes, Circuit"):
        circuit_comp_amplitudes = tracked_subspace_bare_amplitudes(
            circuit.hamiltonian_stack,
            subspace_indices=circuit_overlap_subspace_idx,
            selection_mode=dressed_mode,
            n_candidate_states=n_cand,
            projector_blocks=((1, 2),),
        )
    with progress_heartbeat("static benchmark: tracked_bare_state_amplitudes, Circuit"):
        circuit_branch_amplitudes = tracked_bare_state_amplitudes(
            circuit.hamiltonian_stack,
            tracked_state_indices=circuit_overlap_subspace_idx,
            output_state_indices=circuit_state_order_idx,
            selection_mode=dressed_mode,
            n_candidate_states=n_cand,
            projector_blocks=((1, 2),),
        )
    with progress_heartbeat("static benchmark: tracked_subspace_bare_overlaps, Duffing"):
        duffing_comp_overlaps = tracked_subspace_bare_overlaps(
            duffing.hamiltonian_stack,
            subspace_indices=duffing_overlap_subspace_idx,
            selection_mode=dressed_mode,
            n_candidate_states=n_cand,
            projector_blocks=((1, 2),),
        )
    with progress_heartbeat("static benchmark: tracked_subspace_bare_amplitudes, Duffing"):
        duffing_comp_amplitudes = tracked_subspace_bare_amplitudes(
            duffing.hamiltonian_stack,
            subspace_indices=duffing_overlap_subspace_idx,
            selection_mode=dressed_mode,
            n_candidate_states=n_cand,
            projector_blocks=((1, 2),),
        )
    with progress_heartbeat("static benchmark: tracked_bare_state_amplitudes, Duffing"):
        duffing_branch_amplitudes = tracked_bare_state_amplitudes(
            duffing.hamiltonian_stack,
            tracked_state_indices=duffing_overlap_subspace_idx,
            output_state_indices=duffing_state_order_idx,
            selection_mode=dressed_mode,
            n_candidate_states=n_cand,
            projector_blocks=((1, 2),),
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
        sweep_target=config.static_benchmark.flux_control.sweep_target,
    )

    with progress_heartbeat("static benchmark: derive effective model"):
        derivation = derive_effective_model_from_dressed_stack(
            flux_values=flux_values,
            dressed_stack=source_stack,
            fit_basis=config.static_benchmark.effective_model.fit_basis,
            coupler_frequency_values=wc,
        )

    effective_parameters = derivation.parameter_fit.fitted_parameters
    effective_fit_order = ("w0", "w1", "J", "zeta")
    effective_fit_coefficient_names = {
        name: np.asarray(derivation.parameter_fit.coefficient_names[name], dtype=str)
        for name in effective_fit_order
    }
    effective_fit_coefficients = {
        name: np.asarray(derivation.parameter_fit.coefficients[name], dtype=float)
        for name in effective_fit_order
    }
    H_effective = build_effective_hamiltonian_stack(effective_parameters)

    n_track = int(H_effective.shape[-1])
    raw_E_eff = _raw_energies(H_effective, n_track=n_track, projector_track_single_excitation=True)
    rel_E_eff = _relative_energies(H_effective, n_track=n_track, projector_track_single_excitation=True)
    raw_E_duf = _raw_energies(H_duffing_eff, n_track=n_track, projector_track_single_excitation=True)
    rel_E_duf = _relative_energies(H_duffing_eff, n_track=n_track, projector_track_single_excitation=True)
    raw_E_cir = _raw_energies(H_circuit_eff, n_track=n_track, projector_track_single_excitation=True)
    rel_E_cir = _relative_energies(H_circuit_eff, n_track=n_track, projector_track_single_excitation=True)

    n_full_track = int(min(10, duffing.hamiltonian_stack.shape[1], circuit.hamiltonian_stack.shape[1]))
    raw_E_duf_full = _raw_energies(duffing.hamiltonian_stack, n_track=n_full_track)
    rel_E_duf_full = _relative_energies(duffing.hamiltonian_stack, n_track=n_full_track)
    raw_E_cir_full = _raw_energies(circuit.hamiltonian_stack, n_track=n_full_track)
    rel_E_cir_full = _relative_energies(circuit.hamiltonian_stack, n_track=n_full_track)
    raw_E_duf_full_sorted = _sorted_raw_energies(duffing.hamiltonian_stack, n_track=n_full_track)
    rel_E_duf_full_sorted = _sorted_relative_energies(duffing.hamiltonian_stack, n_track=n_full_track)
    raw_E_cir_full_sorted = _sorted_raw_energies(circuit.hamiltonian_stack, n_track=n_full_track)
    rel_E_cir_full_sorted = _sorted_relative_energies(circuit.hamiltonian_stack, n_track=n_full_track)

    err_raw_E_eff = _per_flux_rmse(raw_E_eff, raw_E_cir)
    err_rel_E_eff = _per_flux_rmse(rel_E_eff, rel_E_cir)
    err_raw_E_duf_ = _per_flux_rmse(raw_E_duf, raw_E_cir)
    err_rel_E_duf_ = _per_flux_rmse(rel_E_duf, rel_E_cir)

    params_duffing = extract_effective_model_parameters_from_4x4_stack(H_duffing_eff)
    params_circuit = extract_effective_model_parameters_from_4x4_stack(H_circuit_eff)

    d1 = np.abs(params_circuit["w0"] - wc)
    d2 = np.abs(params_circuit["w1"] - wc)
    g_scale = max(abs(config.system.interactions.g_0c), abs(config.system.interactions.g_1c), 1e-12)
    detuning_ratio = np.minimum(d1, d2) / g_scale

    idle_thr = float(config.static_benchmark.regime_thresholds.idle_ratio)
    near_thr = float(config.static_benchmark.regime_thresholds.near_ratio)
    idle_mask = detuning_ratio >= idle_thr
    near_mask = detuning_ratio <= near_thr

    eff_idle = _masked_rmse_max(err_rel_E_eff, idle_mask)
    eff_near = _masked_rmse_max(err_rel_E_eff, near_mask)
    duf_idle = _masked_rmse_max(err_rel_E_duf_, idle_mask)
    duf_near = _masked_rmse_max(err_rel_E_duf_, near_mask)
    eff_comp_rmse = _aggregate_rmse(rel_E_eff, rel_E_cir)
    duf_comp_rmse = _aggregate_rmse(rel_E_duf, rel_E_cir)
    duf_trunc_levels = min(
        int(config.duffing_truncation_benchmark.lowest_excited_levels_to_plot),
        int(rel_E_duf_full_sorted.shape[1]) - 1,
        int(rel_E_cir_full_sorted.shape[1]) - 1,
    )
    duf_trunc_style_rmse = _aggregate_rmse(
        rel_E_duf_full_sorted,
        rel_E_cir_full_sorted,
        n_excited=duf_trunc_levels,
    )
    eff_j_abs = _mean_abs_error(effective_parameters["J"], params_circuit["J"])
    eff_zeta_abs = _mean_abs_error(effective_parameters["zeta"], params_circuit["zeta"])
    duf_j_abs = _mean_abs_error(params_duffing["J"], params_circuit["J"])
    duf_zeta_abs = _mean_abs_error(params_duffing["zeta"], params_circuit["zeta"])

    summary = {
        "flux_count": float(flux_values.size),
        "computational_excited_levels_compared": float(n_track - 1),
        "effective_computational_energy_rmse": eff_comp_rmse,
        "duffing_computational_energy_rmse": duf_comp_rmse,
        "effective_mean_abs_dJ": eff_j_abs,
        "effective_mean_abs_dzeta": eff_zeta_abs,
        "duffing_mean_abs_dJ": duf_j_abs,
        "duffing_mean_abs_dzeta": duf_zeta_abs,
        "duffing_truncation_style_excited_levels_compared": float(duf_trunc_levels),
        "duffing_truncation_style_energy_rmse": duf_trunc_style_rmse,
        "effective_idle_rmse": eff_idle[0],
        "effective_idle_max_abs": eff_idle[1],
        "effective_near_rmse": eff_near[0],
        "effective_near_max_abs": eff_near[1],
        "duffing_idle_rmse": duf_idle[0],
        "duffing_idle_max_abs": duf_idle[1],
        "duffing_near_rmse": duf_near[0],
        "duffing_near_max_abs": duf_near[1],
    }
    metric_notes = {
        "effective_error_rmse": (
            "Per-flux RMSE plotted in the static benchmark. At each flux point this is the RMS "
            "energy error over the 3 excited levels of the dressed 4x4 computational subspace, "
            "relative to the circuit model."
        ),
        "duffing_error_rmse": (
            "Per-flux RMSE plotted in the static benchmark. At each flux point this is the RMS "
            "energy error over the 3 excited levels of the dressed 4x4 computational subspace, "
            "relative to the circuit model."
        ),
        "effective_computational_energy_rmse": (
            "Flux-aggregated version of effective_error_rmse: one RMS value pooled over all static "
            "flux points and the 3 excited computational-subspace levels."
        ),
        "duffing_computational_energy_rmse": (
            "Flux-aggregated version of duffing_error_rmse: one RMS value pooled over all static "
            "flux points and the 3 excited computational-subspace levels."
        ),
        "duffing_truncation_style_energy_rmse": (
            "Closest static-benchmark analog to the Duffing truncation benchmark energy_rmse. "
            "It uses sorted eigenvalues from the full Duffing and circuit spectra on the static "
            "benchmark flux grid, matching the truncation benchmark definition, and pools over "
            "the lowest excited levels requested by duffing_truncation_benchmark."
        ),
        "effective_mean_abs_dJ": (
            "Mean absolute exchange error on the static benchmark flux grid. Same quantity type "
            "as truncation |dJ|, but evaluated on the static benchmark grid."
        ),
        "effective_mean_abs_dzeta": (
            "Mean absolute residual-ZZ error on the static benchmark flux grid. Same quantity type "
            "as truncation |dzeta|, but evaluated on the static benchmark grid."
        ),
        "duffing_mean_abs_dJ": (
            "Mean absolute exchange error on the static benchmark flux grid. Same quantity type "
            "as truncation |dJ|, but evaluated on the static benchmark grid."
        ),
        "duffing_mean_abs_dzeta": (
            "Mean absolute residual-ZZ error on the static benchmark flux grid. Same quantity type "
            "as truncation |dzeta|, but evaluated on the static benchmark grid."
        ),
    }

    return StaticBenchmarkResult(
        flux_values=flux_values,
        effective_raw_energies=raw_E_eff,
        effective_relative_energies=rel_E_eff,
        duffing_raw_energies=raw_E_duf,
        duffing_relative_energies=rel_E_duf,
        circuit_raw_energies=raw_E_cir,
        circuit_relative_energies=rel_E_cir,
        duffing_full_raw_energies=raw_E_duf_full,
        duffing_full_relative_energies=rel_E_duf_full,
        circuit_full_raw_energies=raw_E_cir_full,
        circuit_full_relative_energies=rel_E_cir_full,
        effective_error_rmse=err_rel_E_eff,
        duffing_error_rmse=err_rel_E_duf_,
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
        duffing_bare_state_labels=np.asarray(duffing_state_labels, dtype=str),
        duffing_tracked_branch_bare_amplitudes=np.asarray(duffing_branch_amplitudes, dtype=complex),
        duffing_computational_bare_amplitudes=np.asarray(duffing_comp_amplitudes, dtype=complex),
        duffing_computational_bare_overlaps=np.asarray(duffing_comp_overlaps, dtype=float),
        circuit_bare_state_labels=np.asarray(circuit_state_labels, dtype=str),
        circuit_tracked_branch_bare_amplitudes=np.asarray(circuit_branch_amplitudes, dtype=complex),
        circuit_computational_bare_amplitudes=np.asarray(circuit_comp_amplitudes, dtype=complex),
        circuit_computational_bare_overlaps=np.asarray(circuit_comp_overlaps, dtype=float),
        detuning_ratio=np.asarray(detuning_ratio, dtype=float),
        idle_mask=np.asarray(idle_mask, dtype=bool),
        near_mask=np.asarray(near_mask, dtype=bool),
        summary=summary,
        metric_notes=metric_notes,
    )
