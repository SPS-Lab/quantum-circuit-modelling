"""Static convergence benchmarks for circuit and Duffing truncations."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from models import (
    build_circuit_model_stack,
    build_dressed_effective_computational_stack,
    build_duffing_model_stack,
    build_duffing_model_stack_from_parameters,
    extract_model1_parameters_from_4x4_stack,
    fit_duffing_mode_parameters_to_reference,
    fit_symbolic_duffing_mode_parameters_to_reference,
    is_reference_calibrated_duffing_mode,
)
from runtime_utils import log_progress, progress_heartbeat
from study_config import StudyConfig, build_flux_values


@dataclass(frozen=True)
class CircuitTruncationBenchmarkResult:
    flux_values: np.ndarray
    sweep_target: str
    lowest_excited_levels_compared: int
    reference_circuit_ncut: int
    reference_circuit_qubit_truncated_dim: int
    reference_circuit_coupler_truncated_dim: int
    reference_circuit_j_values: np.ndarray
    reference_circuit_zeta_values: np.ndarray
    circuit_ncut_values: np.ndarray
    circuit_ncut_effective_qubit_truncated_dim_values: np.ndarray
    circuit_ncut_energy_rmse: np.ndarray
    circuit_ncut_j_abs_error: np.ndarray
    circuit_ncut_zeta_abs_error: np.ndarray
    circuit_qubit_truncated_dim_values: np.ndarray
    circuit_qubit_truncation_energy_rmse: np.ndarray
    circuit_qubit_truncation_j_abs_error: np.ndarray
    circuit_qubit_truncation_zeta_abs_error: np.ndarray
    circuit_coupler_truncated_dim_values: np.ndarray
    circuit_coupler_truncation_energy_rmse: np.ndarray
    circuit_coupler_truncation_j_abs_error: np.ndarray
    circuit_coupler_truncation_zeta_abs_error: np.ndarray
    summary: dict[str, float]


@dataclass(frozen=True)
class DuffingTruncationBenchmarkResult:
    flux_values: np.ndarray
    sweep_target: str
    duffing_calibration_mode: str
    duffing_truncated_dim: int
    lowest_excited_levels_compared: int
    reference_circuit_ncut: int
    reference_circuit_qubit_truncated_dim: int
    reference_circuit_coupler_truncated_dim: int
    reference_circuit_j_values: np.ndarray
    reference_circuit_zeta_values: np.ndarray
    duffing_ncut_values: np.ndarray
    duffing_ncut_effective_truncated_dim_values: np.ndarray
    duffing_ncut_energy_rmse: np.ndarray
    duffing_ncut_j_abs_error: np.ndarray
    duffing_ncut_zeta_abs_error: np.ndarray
    duffing_hilbert_qubit_dim_values: np.ndarray
    duffing_hilbert_qubit_energy_rmse: np.ndarray
    duffing_hilbert_qubit_j_abs_error: np.ndarray
    duffing_hilbert_qubit_zeta_abs_error: np.ndarray
    duffing_hilbert_coupler_dim_values: np.ndarray
    duffing_hilbert_coupler_energy_rmse: np.ndarray
    duffing_hilbert_coupler_j_abs_error: np.ndarray
    duffing_hilbert_coupler_zeta_abs_error: np.ndarray
    summary: dict[str, float]


_DUFFING_TRUNCATION_SWEEP_NAMES = frozenset({"ncut", "qubit", "coupler"})


def _rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def _circuit_config_with_dims(
    config: StudyConfig,
    *,
    qubit_truncated_dim: int,
    coupler_truncated_dim: int,
):
    return replace(
        config.static_benchmark.circuit_model,
        hilbert_truncation=replace(
            config.static_benchmark.circuit_model.hilbert_truncation,
            q0_truncated_dim=int(qubit_truncated_dim),
            q1_truncated_dim=int(qubit_truncated_dim),
            c_truncated_dim=int(coupler_truncated_dim),
        ),
    )


def _duffing_config_with_truncation(
    config: StudyConfig,
    *,
    extraction_ncut: int,
    extraction_truncated_dim: int,
    hilbert_qubit_dim: int,
    hilbert_coupler_dim: int,
    calibration_mode: str,
):
    return replace(
        config.static_benchmark.duffing_model,
        transmon_spectral_extraction=replace(
            config.static_benchmark.duffing_model.transmon_spectral_extraction,
            ncut=int(extraction_ncut),
            truncated_dim=int(extraction_truncated_dim),
        ),
        hilbert_truncation=replace(
            config.static_benchmark.duffing_model.hilbert_truncation,
            nlevels_qubit=int(hilbert_qubit_dim),
            nlevels_coupler=int(hilbert_coupler_dim),
        ),
        calibration_mode=str(calibration_mode),
    )


def _extract_circuit_metrics(
    *,
    config: StudyConfig,
    flux: float,
    circuit_ncut: int,
    qubit_truncated_dim: int,
    coupler_truncated_dim: int,
) -> tuple[float, float, np.ndarray, int]:
    qubit_trunc_eff = int(min(int(qubit_truncated_dim), 2 * int(circuit_ncut) + 1))
    if qubit_trunc_eff < 2:
        raise ValueError("Effective circuit qubit truncated_dim must be >= 2")
    system_ref = replace(
        config.system,
        q0=replace(config.system.q0, ncut=int(circuit_ncut)),
        q1=replace(config.system.q1, ncut=int(circuit_ncut)),
    )
    circuit_cfg = _circuit_config_with_dims(
        config,
        qubit_truncated_dim=qubit_trunc_eff,
        coupler_truncated_dim=int(coupler_truncated_dim),
    )
    H_cir = build_circuit_model_stack(
        flux_values=np.array([float(flux)], dtype=float),
        system_params=system_ref,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        circuit_config=circuit_cfg,
        sweep_target=config.static_benchmark.flux_control.sweep_target,
    ).hamiltonian_stack
    H_cir_eff = build_dressed_effective_computational_stack(
        H_cir,
        nlevels_qubit=qubit_trunc_eff,
        nlevels_coupler=int(coupler_truncated_dim),
        n_candidate_states=config.static_benchmark.dressed_subspace.n_candidate_states,
        selection_mode=config.static_benchmark.dressed_subspace.selection_mode,
    )
    params = extract_model1_parameters_from_4x4_stack(H_cir_eff)
    evals = np.linalg.eigvalsh(np.asarray(H_cir[0], dtype=complex))
    rel_e = np.asarray(evals - evals[0], dtype=float)
    return float(params["J"][0]), float(params["zeta"][0]), rel_e, qubit_trunc_eff


def _extract_strict_circuit_reference(
    *,
    config: StudyConfig,
    flux_values: np.ndarray,
    reference_ncut: int,
    reference_qdim: int,
    reference_cdim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_j_values = np.empty(flux_values.shape[0], dtype=float)
    ref_zeta_values = np.empty(flux_values.shape[0], dtype=float)
    ref_rel_e_values: list[np.ndarray] = []
    with progress_heartbeat("truncation benchmark: build strict circuit reference points"):
        for i, flux in enumerate(np.asarray(flux_values, dtype=float)):
            ref_j, ref_zeta, ref_rel_e, _ = _extract_circuit_metrics(
                config=config,
                flux=float(flux),
                circuit_ncut=reference_ncut,
                qubit_truncated_dim=reference_qdim,
                coupler_truncated_dim=reference_cdim,
            )
            ref_j_values[i] = float(ref_j)
            ref_zeta_values[i] = float(ref_zeta)
            ref_rel_e_values.append(np.asarray(ref_rel_e, dtype=float))
    return ref_j_values, ref_zeta_values, np.stack(ref_rel_e_values, axis=0)


def _build_reference_dressed_stack(
    *,
    config: StudyConfig,
    circuit_reference_ncut: int,
    reference_qubit_truncated_dim: int,
    reference_coupler_truncated_dim: int,
    flux_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    flux_values_arr = (
        build_flux_values(config.static_benchmark.flux_sweep)
        if flux_values is None
        else np.asarray(flux_values, dtype=float)
    )
    system_ref = replace(
        config.system,
        q0=replace(config.system.q0, ncut=int(circuit_reference_ncut)),
        q1=replace(config.system.q1, ncut=int(circuit_reference_ncut)),
    )
    circuit_cfg = _circuit_config_with_dims(
        config,
        qubit_truncated_dim=int(reference_qubit_truncated_dim),
        coupler_truncated_dim=int(reference_coupler_truncated_dim),
    )
    H_cir = build_circuit_model_stack(
        flux_values=flux_values_arr,
        system_params=system_ref,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        circuit_config=circuit_cfg,
        sweep_target=config.static_benchmark.flux_control.sweep_target,
    ).hamiltonian_stack
    H_cir_eff = build_dressed_effective_computational_stack(
        H_cir,
        nlevels_qubit=int(reference_qubit_truncated_dim),
        nlevels_coupler=int(reference_coupler_truncated_dim),
        n_candidate_states=config.static_benchmark.dressed_subspace.n_candidate_states,
        selection_mode=config.static_benchmark.dressed_subspace.selection_mode,
    )
    return np.asarray(flux_values_arr, dtype=float), np.asarray(H_cir_eff, dtype=complex)


def _extract_duffing_metrics(
    *,
    config: StudyConfig,
    flux: float,
    extraction_ncut: int,
    extraction_truncated_dim: int,
    hilbert_qubit_dim: int,
    hilbert_coupler_dim: int,
    duffing_calibration_mode: str,
    reference_flux_values: np.ndarray | None = None,
    reference_dressed_stack: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, int]:
    trunc_dim_eff = int(min(int(extraction_truncated_dim), 2 * int(extraction_ncut) + 1))
    if trunc_dim_eff < 3:
        raise ValueError("Effective Duffing transmon truncated_dim must be >= 3")
    dcfg = _duffing_config_with_truncation(
        config,
        extraction_ncut=int(extraction_ncut),
        extraction_truncated_dim=trunc_dim_eff,
        hilbert_qubit_dim=int(hilbert_qubit_dim),
        hilbert_coupler_dim=int(hilbert_coupler_dim),
        calibration_mode=str(duffing_calibration_mode),
    )
    if is_reference_calibrated_duffing_mode(duffing_calibration_mode):
        if reference_flux_values is None or reference_dressed_stack is None:
            raise ValueError("Reference-driven Duffing modes require a reference dressed stack")
        if str(duffing_calibration_mode).strip().lower() == "fitted-static":
            mode_parameters = fit_duffing_mode_parameters_to_reference(
                reference_flux_values,
                reference_dressed_stack=reference_dressed_stack,
                system_params=config.system,
                coupler_frequency=config.static_benchmark.coupler_frequency,
                duffing_config=dcfg,
                sweep_target=config.static_benchmark.flux_control.sweep_target,
                n_candidate_states=config.static_benchmark.dressed_subspace.n_candidate_states,
                selection_mode=config.static_benchmark.dressed_subspace.selection_mode,
            )
        else:
            symbolic_fit_cfg = dcfg.symbolic_fit
            if symbolic_fit_cfg is None:
                raise ValueError(
                    "symbolic-fitted-static requires static_benchmark.duffing_model.symbolic_fit settings"
                )
            mode_parameters = fit_symbolic_duffing_mode_parameters_to_reference(
                reference_flux_values,
                reference_dressed_stack=reference_dressed_stack,
                system_params=config.system,
                coupler_frequency=config.static_benchmark.coupler_frequency,
                duffing_config=dcfg,
                sweep_target=config.static_benchmark.flux_control.sweep_target,
                n_candidate_states=config.static_benchmark.dressed_subspace.n_candidate_states,
                selection_mode=config.static_benchmark.dressed_subspace.selection_mode,
                max_harmonics=int(symbolic_fit_cfg.max_harmonics),
                pointwise_max_nfev=int(symbolic_fit_cfg.pointwise_max_nfev),
                refinement_max_nfev=int(symbolic_fit_cfg.refinement_max_nfev),
                regularization_weight=float(symbolic_fit_cfg.regularization_weight),
            ).fitted_parameters
        point_parameters = {
            key: np.array(
                [
                    np.interp(
                        float(flux),
                        np.asarray(reference_flux_values, dtype=float).ravel(),
                        np.asarray(values, dtype=float).ravel(),
                    )
                ],
                dtype=float,
            )
            for key, values in mode_parameters.items()
        }
        H_duf = build_duffing_model_stack_from_parameters(
            point_parameters,
            system_params=config.system,
            duffing_config=dcfg,
        ).hamiltonian_stack
    else:
        H_duf = build_duffing_model_stack(
            flux_values=np.array([float(flux)], dtype=float),
            system_params=config.system,
            coupler_frequency=config.static_benchmark.coupler_frequency,
            duffing_config=dcfg,
            sweep_target=config.static_benchmark.flux_control.sweep_target,
        ).hamiltonian_stack
    H_duf_eff = build_dressed_effective_computational_stack(
        H_duf,
        nlevels_qubit=int(hilbert_qubit_dim),
        nlevels_coupler=int(hilbert_coupler_dim),
        n_candidate_states=config.static_benchmark.dressed_subspace.n_candidate_states,
        selection_mode=config.static_benchmark.dressed_subspace.selection_mode,
    )
    params = extract_model1_parameters_from_4x4_stack(H_duf_eff)
    evals = np.linalg.eigvalsh(np.asarray(H_duf[0], dtype=complex))
    rel_e = np.asarray(evals - evals[0], dtype=float)
    return float(params["J"][0]), float(params["zeta"][0]), rel_e, trunc_dim_eff


def _extract_circuit_metrics_over_fluxes(
    *,
    config: StudyConfig,
    flux_values: np.ndarray,
    circuit_ncut: int,
    qubit_truncated_dim: int,
    coupler_truncated_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    j_values = np.empty(flux_values.shape[0], dtype=float)
    zeta_values = np.empty(flux_values.shape[0], dtype=float)
    rel_e_values: list[np.ndarray] = []
    qdim_eff_out: int | None = None
    for i, flux in enumerate(np.asarray(flux_values, dtype=float)):
        cand_j, cand_zeta, cand_rel_e, qdim_eff = _extract_circuit_metrics(
            config=config,
            flux=float(flux),
            circuit_ncut=int(circuit_ncut),
            qubit_truncated_dim=int(qubit_truncated_dim),
            coupler_truncated_dim=int(coupler_truncated_dim),
        )
        j_values[i] = float(cand_j)
        zeta_values[i] = float(cand_zeta)
        rel_e_values.append(np.asarray(cand_rel_e, dtype=float))
        qdim_eff_out = int(qdim_eff)
    if qdim_eff_out is None:
        raise ValueError("At least one flux value is required")
    return j_values, zeta_values, np.stack(rel_e_values, axis=0), qdim_eff_out


def _extract_duffing_metrics_over_fluxes(
    *,
    config: StudyConfig,
    flux_values: np.ndarray,
    extraction_ncut: int,
    extraction_truncated_dim: int,
    hilbert_qubit_dim: int,
    hilbert_coupler_dim: int,
    duffing_calibration_mode: str,
    reference_flux_values: np.ndarray | None = None,
    reference_dressed_stack: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    j_values = np.empty(flux_values.shape[0], dtype=float)
    zeta_values = np.empty(flux_values.shape[0], dtype=float)
    rel_e_values: list[np.ndarray] = []
    trunc_dim_eff_out: int | None = None
    for i, flux in enumerate(np.asarray(flux_values, dtype=float)):
        cand_j, cand_zeta, cand_rel_e, trunc_dim_eff = _extract_duffing_metrics(
            config=config,
            flux=float(flux),
            extraction_ncut=int(extraction_ncut),
            extraction_truncated_dim=int(extraction_truncated_dim),
            hilbert_qubit_dim=int(hilbert_qubit_dim),
            hilbert_coupler_dim=int(hilbert_coupler_dim),
            duffing_calibration_mode=str(duffing_calibration_mode),
            reference_flux_values=reference_flux_values,
            reference_dressed_stack=reference_dressed_stack,
        )
        j_values[i] = float(cand_j)
        zeta_values[i] = float(cand_zeta)
        rel_e_values.append(np.asarray(cand_rel_e, dtype=float))
        trunc_dim_eff_out = int(trunc_dim_eff)
    if trunc_dim_eff_out is None:
        raise ValueError("At least one flux value is required")
    return j_values, zeta_values, np.stack(rel_e_values, axis=0), trunc_dim_eff_out


def _static_error_metrics(
    *,
    candidate_j: np.ndarray,
    candidate_zeta: np.ndarray,
    candidate_rel_e: np.ndarray,
    reference_j: np.ndarray,
    reference_zeta: np.ndarray,
    reference_rel_e: np.ndarray,
    lowest_excited_levels_to_compare: int,
) -> tuple[float, float, float]:
    candidate_rel_e_arr = np.asarray(candidate_rel_e, dtype=float)
    reference_rel_e_arr = np.asarray(reference_rel_e, dtype=float)
    n_excited = int(
        min(
            max(0, int(lowest_excited_levels_to_compare)),
            max(0, int(candidate_rel_e_arr.shape[-1]) - 1),
            max(0, int(reference_rel_e_arr.shape[-1]) - 1),
        )
    )
    if n_excited > 0:
        energy_diff = (
            candidate_rel_e_arr[..., 1 : 1 + n_excited]
            - reference_rel_e_arr[..., 1 : 1 + n_excited]
        )
        energy_rmse = _rmse(energy_diff.ravel())
    else:
        energy_diff = np.zeros((0,), dtype=float)
        energy_rmse = 0.0
    j_abs_diff = np.abs(np.asarray(candidate_j, dtype=float) - np.asarray(reference_j, dtype=float))
    zeta_abs_diff = np.abs(np.asarray(candidate_zeta, dtype=float) - np.asarray(reference_zeta, dtype=float))
    j_abs_error = float(np.mean(j_abs_diff))
    zeta_abs_error = float(np.mean(zeta_abs_diff))
    return energy_rmse, j_abs_error, zeta_abs_error


def _summary_for_sweep(
    *,
    prefix: str,
    values: np.ndarray,
    energy_rmse: np.ndarray,
    j_abs_error: np.ndarray,
    zeta_abs_error: np.ndarray,
) -> dict[str, float]:
    idx = int(np.argmin(energy_rmse))
    return {
        f"{prefix}_points": float(values.shape[0]),
        f"{prefix}_best_value_index": float(idx),
        f"{prefix}_best_energy_rmse": float(energy_rmse[idx]),
        f"{prefix}_best_j_abs_error": float(j_abs_error[idx]),
        f"{prefix}_best_zeta_abs_error": float(zeta_abs_error[idx]),
    }


def _normalize_duffing_truncation_sweeps(selected_sweeps: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if selected_sweeps is None:
        return ("ncut", "qubit", "coupler")
    normalized = tuple(str(name).strip().lower() for name in selected_sweeps if str(name).strip())
    if not normalized:
        raise ValueError("At least one Duffing truncation sweep must be selected")
    unknown = [name for name in normalized if name not in _DUFFING_TRUNCATION_SWEEP_NAMES]
    if unknown:
        raise ValueError(
            "Unsupported Duffing truncation sweep selection(s): "
            + ", ".join(sorted(set(unknown)))
        )
    deduped: list[str] = []
    for name in normalized:
        if name not in deduped:
            deduped.append(name)
    return tuple(deduped)


def run_circuit_truncation_benchmark(config: StudyConfig) -> CircuitTruncationBenchmarkResult:
    cfg = config.circuit_truncation_benchmark
    flux_values = np.asarray(cfg.flux_values, dtype=float)
    log_progress(
        "circuit truncation benchmark: starting aggregated static convergence sweeps "
        f"over {flux_values.size} flux points for sweep_target={config.static_benchmark.flux_control.sweep_target}"
    )
    ref_j_values, ref_zeta_values, ref_rel_e_values = _extract_strict_circuit_reference(
        config=config,
        flux_values=flux_values,
        reference_ncut=int(cfg.circuit_reference_ncut),
        reference_qdim=int(cfg.circuit_reference_qubit_truncated_dim),
        reference_cdim=int(cfg.circuit_reference_coupler_truncated_dim),
    )
    circuit_ncut_values = np.asarray(cfg.circuit_ncut_values, dtype=int)
    circuit_ncut_effective_qdim = np.empty(circuit_ncut_values.shape[0], dtype=int)
    circuit_ncut_energy_rmse = np.empty(circuit_ncut_values.shape[0], dtype=float)
    circuit_ncut_j_abs_error = np.empty(circuit_ncut_values.shape[0], dtype=float)
    circuit_ncut_zeta_abs_error = np.empty(circuit_ncut_values.shape[0], dtype=float)
    log_progress(f"circuit truncation benchmark: ncut sweep with {circuit_ncut_values.size} points")
    for i, ncut in enumerate(circuit_ncut_values):
        log_progress(f"circuit truncation benchmark: ncut point {i + 1}/{circuit_ncut_values.size} (ncut={int(ncut)})")
        cand_j_values, cand_zeta_values, cand_rel_e_values, qdim_eff = _extract_circuit_metrics_over_fluxes(
            config=config,
            flux_values=flux_values,
            circuit_ncut=int(ncut),
            qubit_truncated_dim=int(cfg.circuit_reference_qubit_truncated_dim),
            coupler_truncated_dim=int(cfg.circuit_reference_coupler_truncated_dim),
        )
        circuit_ncut_effective_qdim[i] = int(qdim_eff)
        (
            circuit_ncut_energy_rmse[i],
            circuit_ncut_j_abs_error[i],
            circuit_ncut_zeta_abs_error[i],
        ) = _static_error_metrics(
            candidate_j=cand_j_values,
            candidate_zeta=cand_zeta_values,
            candidate_rel_e=cand_rel_e_values,
            reference_j=ref_j_values,
            reference_zeta=ref_zeta_values,
            reference_rel_e=ref_rel_e_values,
            lowest_excited_levels_to_compare=int(cfg.lowest_excited_levels_to_plot),
        )
    circuit_qubit_dims = np.asarray(cfg.circuit_qubit_truncated_dim_values, dtype=int)
    circuit_qubit_energy_rmse = np.empty(circuit_qubit_dims.shape[0], dtype=float)
    circuit_qubit_j_abs_error = np.empty(circuit_qubit_dims.shape[0], dtype=float)
    circuit_qubit_zeta_abs_error = np.empty(circuit_qubit_dims.shape[0], dtype=float)
    log_progress(f"circuit truncation benchmark: qubit truncated-dim sweep with {circuit_qubit_dims.size} points")
    for i, qdim in enumerate(circuit_qubit_dims):
        log_progress(
            "circuit truncation benchmark: qubit truncated-dim point "
            f"{i + 1}/{circuit_qubit_dims.size} (q={int(qdim)}, c_fixed={int(cfg.circuit_reference_coupler_truncated_dim)})"
        )
        cand_j_values, cand_zeta_values, cand_rel_e_values, _ = _extract_circuit_metrics_over_fluxes(
            config=config,
            flux_values=flux_values,
            circuit_ncut=int(cfg.circuit_reference_ncut),
            qubit_truncated_dim=int(qdim),
            coupler_truncated_dim=int(cfg.circuit_reference_coupler_truncated_dim),
        )
        (
            circuit_qubit_energy_rmse[i],
            circuit_qubit_j_abs_error[i],
            circuit_qubit_zeta_abs_error[i],
        ) = _static_error_metrics(
            candidate_j=cand_j_values,
            candidate_zeta=cand_zeta_values,
            candidate_rel_e=cand_rel_e_values,
            reference_j=ref_j_values,
            reference_zeta=ref_zeta_values,
            reference_rel_e=ref_rel_e_values,
            lowest_excited_levels_to_compare=int(cfg.lowest_excited_levels_to_plot),
        )
    circuit_coupler_dims = np.asarray(cfg.circuit_coupler_truncated_dim_values, dtype=int)
    circuit_coupler_energy_rmse = np.empty(circuit_coupler_dims.shape[0], dtype=float)
    circuit_coupler_j_abs_error = np.empty(circuit_coupler_dims.shape[0], dtype=float)
    circuit_coupler_zeta_abs_error = np.empty(circuit_coupler_dims.shape[0], dtype=float)
    log_progress(
        f"circuit truncation benchmark: coupler truncated-dim sweep with {circuit_coupler_dims.size} points"
    )
    for i, cdim in enumerate(circuit_coupler_dims):
        log_progress(
            "circuit truncation benchmark: coupler truncated-dim point "
            f"{i + 1}/{circuit_coupler_dims.size} (c={int(cdim)}, q_fixed={int(cfg.circuit_reference_qubit_truncated_dim)})"
        )
        cand_j_values, cand_zeta_values, cand_rel_e_values, _ = _extract_circuit_metrics_over_fluxes(
            config=config,
            flux_values=flux_values,
            circuit_ncut=int(cfg.circuit_reference_ncut),
            qubit_truncated_dim=int(cfg.circuit_reference_qubit_truncated_dim),
            coupler_truncated_dim=int(cdim),
        )
        (
            circuit_coupler_energy_rmse[i],
            circuit_coupler_j_abs_error[i],
            circuit_coupler_zeta_abs_error[i],
        ) = _static_error_metrics(
            candidate_j=cand_j_values,
            candidate_zeta=cand_zeta_values,
            candidate_rel_e=cand_rel_e_values,
            reference_j=ref_j_values,
            reference_zeta=ref_zeta_values,
            reference_rel_e=ref_rel_e_values,
            lowest_excited_levels_to_compare=int(cfg.lowest_excited_levels_to_plot),
        )
    summary = {
        "flux_count": float(flux_values.size),
        "flux_min": float(np.min(flux_values)),
        "flux_max": float(np.max(flux_values)),
        "reference_circuit_ncut": float(cfg.circuit_reference_ncut),
        "reference_circuit_qubit_truncated_dim": float(cfg.circuit_reference_qubit_truncated_dim),
        "reference_circuit_coupler_truncated_dim": float(cfg.circuit_reference_coupler_truncated_dim),
        "reference_circuit_j_mean": float(np.mean(ref_j_values)),
        "reference_circuit_zeta_mean": float(np.mean(ref_zeta_values)),
        "lowest_excited_levels_compared": float(cfg.lowest_excited_levels_to_plot),
        "circuit_ncut_effective_qubit_truncated_dim_min": float(np.min(circuit_ncut_effective_qdim)),
        "circuit_ncut_effective_qubit_truncated_dim_max": float(np.max(circuit_ncut_effective_qdim)),
    }
    summary.update(
        _summary_for_sweep(
            prefix="circuit_ncut",
            values=circuit_ncut_values.astype(float),
            energy_rmse=circuit_ncut_energy_rmse,
            j_abs_error=circuit_ncut_j_abs_error,
            zeta_abs_error=circuit_ncut_zeta_abs_error,
        )
    )
    summary.update(
        _summary_for_sweep(
            prefix="circuit_qubit_truncation",
            values=circuit_qubit_dims.astype(float),
            energy_rmse=circuit_qubit_energy_rmse,
            j_abs_error=circuit_qubit_j_abs_error,
            zeta_abs_error=circuit_qubit_zeta_abs_error,
        )
    )
    summary.update(
        _summary_for_sweep(
            prefix="circuit_coupler_truncation",
            values=circuit_coupler_dims.astype(float),
            energy_rmse=circuit_coupler_energy_rmse,
            j_abs_error=circuit_coupler_j_abs_error,
            zeta_abs_error=circuit_coupler_zeta_abs_error,
        )
    )
    return CircuitTruncationBenchmarkResult(
        flux_values=np.asarray(flux_values, dtype=float),
        sweep_target=str(config.static_benchmark.flux_control.sweep_target),
        lowest_excited_levels_compared=int(cfg.lowest_excited_levels_to_plot),
        reference_circuit_ncut=int(cfg.circuit_reference_ncut),
        reference_circuit_qubit_truncated_dim=int(cfg.circuit_reference_qubit_truncated_dim),
        reference_circuit_coupler_truncated_dim=int(cfg.circuit_reference_coupler_truncated_dim),
        reference_circuit_j_values=np.asarray(ref_j_values, dtype=float),
        reference_circuit_zeta_values=np.asarray(ref_zeta_values, dtype=float),
        circuit_ncut_values=np.asarray(circuit_ncut_values, dtype=int),
        circuit_ncut_effective_qubit_truncated_dim_values=np.asarray(circuit_ncut_effective_qdim, dtype=int),
        circuit_ncut_energy_rmse=np.asarray(circuit_ncut_energy_rmse, dtype=float),
        circuit_ncut_j_abs_error=np.asarray(circuit_ncut_j_abs_error, dtype=float),
        circuit_ncut_zeta_abs_error=np.asarray(circuit_ncut_zeta_abs_error, dtype=float),
        circuit_qubit_truncated_dim_values=np.asarray(circuit_qubit_dims, dtype=int),
        circuit_qubit_truncation_energy_rmse=np.asarray(circuit_qubit_energy_rmse, dtype=float),
        circuit_qubit_truncation_j_abs_error=np.asarray(circuit_qubit_j_abs_error, dtype=float),
        circuit_qubit_truncation_zeta_abs_error=np.asarray(circuit_qubit_zeta_abs_error, dtype=float),
        circuit_coupler_truncated_dim_values=np.asarray(circuit_coupler_dims, dtype=int),
        circuit_coupler_truncation_energy_rmse=np.asarray(circuit_coupler_energy_rmse, dtype=float),
        circuit_coupler_truncation_j_abs_error=np.asarray(circuit_coupler_j_abs_error, dtype=float),
        circuit_coupler_truncation_zeta_abs_error=np.asarray(circuit_coupler_zeta_abs_error, dtype=float),
        summary=summary,
    )


def run_duffing_truncation_benchmark(
    config: StudyConfig,
    *,
    selected_sweeps: tuple[str, ...] | list[str] | None = None,
) -> DuffingTruncationBenchmarkResult:
    cfg = config.duffing_truncation_benchmark
    sweep_selection = _normalize_duffing_truncation_sweeps(selected_sweeps)
    run_ncut = "ncut" in sweep_selection
    run_qubit = "qubit" in sweep_selection
    run_coupler = "coupler" in sweep_selection
    flux_values = np.asarray(cfg.flux_values, dtype=float)
    log_progress(
        "duffing truncation benchmark: starting aggregated static convergence sweeps "
        f"over {flux_values.size} flux points for sweep_target={config.static_benchmark.flux_control.sweep_target}"
    )
    ref_j_values, ref_zeta_values, ref_rel_e_values = _extract_strict_circuit_reference(
        config=config,
        flux_values=flux_values,
        reference_ncut=int(cfg.circuit_reference_ncut),
        reference_qdim=int(cfg.circuit_reference_qubit_truncated_dim),
        reference_cdim=int(cfg.circuit_reference_coupler_truncated_dim),
    )
    with progress_heartbeat("duffing truncation benchmark: build strict circuit reference flux stack"):
        reference_flux_values, reference_dressed_stack = _build_reference_dressed_stack(
            config=config,
            circuit_reference_ncut=int(cfg.circuit_reference_ncut),
            reference_qubit_truncated_dim=int(cfg.circuit_reference_qubit_truncated_dim),
            reference_coupler_truncated_dim=int(cfg.circuit_reference_coupler_truncated_dim),
            flux_values=flux_values,
        )
    base_duf_qdim = int(config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit)
    base_duf_cdim = int(config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler)
    base_extract_ncut = int(config.static_benchmark.duffing_model.transmon_spectral_extraction.ncut)
    base_extract_trunc_dim = int(config.static_benchmark.duffing_model.transmon_spectral_extraction.truncated_dim)
    if run_ncut:
        duffing_ncut_values = np.asarray(cfg.duffing_ncut_values, dtype=int)
        duffing_ncut_effective_trunc_dim = np.empty(duffing_ncut_values.shape[0], dtype=int)
        duffing_ncut_energy_rmse = np.empty(duffing_ncut_values.shape[0], dtype=float)
        duffing_ncut_j_abs_error = np.empty(duffing_ncut_values.shape[0], dtype=float)
        duffing_ncut_zeta_abs_error = np.empty(duffing_ncut_values.shape[0], dtype=float)
        log_progress(f"duffing truncation benchmark: extraction ncut sweep with {duffing_ncut_values.size} points")
        for i, ncut in enumerate(duffing_ncut_values):
            log_progress(
                f"duffing truncation benchmark: extraction ncut point "
                f"{i + 1}/{duffing_ncut_values.size} (ncut={int(ncut)})"
            )
            cand_j_values, cand_zeta_values, cand_rel_e_values, trunc_dim_eff = _extract_duffing_metrics_over_fluxes(
                config=config,
                flux_values=flux_values,
                extraction_ncut=int(ncut),
                extraction_truncated_dim=int(cfg.duffing_truncated_dim),
                hilbert_qubit_dim=base_duf_qdim,
                hilbert_coupler_dim=base_duf_cdim,
                duffing_calibration_mode=str(cfg.duffing_calibration_mode),
                reference_flux_values=reference_flux_values,
                reference_dressed_stack=reference_dressed_stack,
            )
            duffing_ncut_effective_trunc_dim[i] = int(trunc_dim_eff)
            (
                duffing_ncut_energy_rmse[i],
                duffing_ncut_j_abs_error[i],
                duffing_ncut_zeta_abs_error[i],
            ) = _static_error_metrics(
                candidate_j=cand_j_values,
                candidate_zeta=cand_zeta_values,
                candidate_rel_e=cand_rel_e_values,
                reference_j=ref_j_values,
                reference_zeta=ref_zeta_values,
                reference_rel_e=ref_rel_e_values,
                lowest_excited_levels_to_compare=int(cfg.lowest_excited_levels_to_plot),
            )
    else:
        duffing_ncut_values = np.asarray([], dtype=int)
        duffing_ncut_effective_trunc_dim = np.asarray([], dtype=int)
        duffing_ncut_energy_rmse = np.asarray([], dtype=float)
        duffing_ncut_j_abs_error = np.asarray([], dtype=float)
        duffing_ncut_zeta_abs_error = np.asarray([], dtype=float)

    if run_qubit:
        duffing_hilbert_qubit_dims = np.asarray(cfg.duffing_hilbert_qubit_dim_values, dtype=int)
        duffing_hilbert_qubit_energy_rmse = np.empty(duffing_hilbert_qubit_dims.shape[0], dtype=float)
        duffing_hilbert_qubit_j_abs_error = np.empty(duffing_hilbert_qubit_dims.shape[0], dtype=float)
        duffing_hilbert_qubit_zeta_abs_error = np.empty(duffing_hilbert_qubit_dims.shape[0], dtype=float)
        log_progress(
            f"duffing truncation benchmark: qubit Hilbert truncation sweep with {duffing_hilbert_qubit_dims.size} points"
        )
        for i, qdim in enumerate(duffing_hilbert_qubit_dims):
            log_progress(
                "duffing truncation benchmark: qubit Hilbert point "
                f"{i + 1}/{duffing_hilbert_qubit_dims.size} (q={int(qdim)}, c_fixed={int(base_duf_cdim)})"
            )
            cand_j_values, cand_zeta_values, cand_rel_e_values, _ = _extract_duffing_metrics_over_fluxes(
                config=config,
                flux_values=flux_values,
                extraction_ncut=base_extract_ncut,
                extraction_truncated_dim=base_extract_trunc_dim,
                hilbert_qubit_dim=int(qdim),
                hilbert_coupler_dim=int(base_duf_cdim),
                duffing_calibration_mode=str(cfg.duffing_calibration_mode),
                reference_flux_values=reference_flux_values,
                reference_dressed_stack=reference_dressed_stack,
            )
            (
                duffing_hilbert_qubit_energy_rmse[i],
                duffing_hilbert_qubit_j_abs_error[i],
                duffing_hilbert_qubit_zeta_abs_error[i],
            ) = _static_error_metrics(
                candidate_j=cand_j_values,
                candidate_zeta=cand_zeta_values,
                candidate_rel_e=cand_rel_e_values,
                reference_j=ref_j_values,
                reference_zeta=ref_zeta_values,
                reference_rel_e=ref_rel_e_values,
                lowest_excited_levels_to_compare=int(cfg.lowest_excited_levels_to_plot),
            )
    else:
        duffing_hilbert_qubit_dims = np.asarray([], dtype=int)
        duffing_hilbert_qubit_energy_rmse = np.asarray([], dtype=float)
        duffing_hilbert_qubit_j_abs_error = np.asarray([], dtype=float)
        duffing_hilbert_qubit_zeta_abs_error = np.asarray([], dtype=float)

    if run_coupler:
        duffing_hilbert_coupler_dims = np.asarray(cfg.duffing_hilbert_coupler_dim_values, dtype=int)
        duffing_hilbert_coupler_energy_rmse = np.empty(duffing_hilbert_coupler_dims.shape[0], dtype=float)
        duffing_hilbert_coupler_j_abs_error = np.empty(duffing_hilbert_coupler_dims.shape[0], dtype=float)
        duffing_hilbert_coupler_zeta_abs_error = np.empty(duffing_hilbert_coupler_dims.shape[0], dtype=float)
        log_progress(
            "duffing truncation benchmark: coupler Hilbert truncation sweep "
            f"with {duffing_hilbert_coupler_dims.size} points"
        )
        for i, cdim in enumerate(duffing_hilbert_coupler_dims):
            log_progress(
                "duffing truncation benchmark: coupler Hilbert point "
                f"{i + 1}/{duffing_hilbert_coupler_dims.size} (c={int(cdim)}, q_fixed={int(base_duf_qdim)})"
            )
            cand_j_values, cand_zeta_values, cand_rel_e_values, _ = _extract_duffing_metrics_over_fluxes(
                config=config,
                flux_values=flux_values,
                extraction_ncut=base_extract_ncut,
                extraction_truncated_dim=base_extract_trunc_dim,
                hilbert_qubit_dim=int(base_duf_qdim),
                hilbert_coupler_dim=int(cdim),
                duffing_calibration_mode=str(cfg.duffing_calibration_mode),
                reference_flux_values=reference_flux_values,
                reference_dressed_stack=reference_dressed_stack,
            )
            (
                duffing_hilbert_coupler_energy_rmse[i],
                duffing_hilbert_coupler_j_abs_error[i],
                duffing_hilbert_coupler_zeta_abs_error[i],
            ) = _static_error_metrics(
                candidate_j=cand_j_values,
                candidate_zeta=cand_zeta_values,
                candidate_rel_e=cand_rel_e_values,
                reference_j=ref_j_values,
                reference_zeta=ref_zeta_values,
                reference_rel_e=ref_rel_e_values,
                lowest_excited_levels_to_compare=int(cfg.lowest_excited_levels_to_plot),
            )
    else:
        duffing_hilbert_coupler_dims = np.asarray([], dtype=int)
        duffing_hilbert_coupler_energy_rmse = np.asarray([], dtype=float)
        duffing_hilbert_coupler_j_abs_error = np.asarray([], dtype=float)
        duffing_hilbert_coupler_zeta_abs_error = np.asarray([], dtype=float)
    summary = {
        "flux_count": float(flux_values.size),
        "flux_min": float(np.min(flux_values)),
        "flux_max": float(np.max(flux_values)),
        "reference_circuit_ncut": float(cfg.circuit_reference_ncut),
        "reference_circuit_qubit_truncated_dim": float(cfg.circuit_reference_qubit_truncated_dim),
        "reference_circuit_coupler_truncated_dim": float(cfg.circuit_reference_coupler_truncated_dim),
        "reference_circuit_j_mean": float(np.mean(ref_j_values)),
        "reference_circuit_zeta_mean": float(np.mean(ref_zeta_values)),
        "lowest_excited_levels_compared": float(cfg.lowest_excited_levels_to_plot),
        "duffing_extraction_truncated_dim_configured": float(cfg.duffing_truncated_dim),
    }
    summary["duffing_selected_sweep_count"] = float(len(sweep_selection))
    summary["duffing_selected_ncut"] = float(run_ncut)
    summary["duffing_selected_qubit"] = float(run_qubit)
    summary["duffing_selected_coupler"] = float(run_coupler)
    if duffing_ncut_values.size > 0:
        summary.update(
            _summary_for_sweep(
                prefix="duffing_ncut",
                values=duffing_ncut_values.astype(float),
                energy_rmse=duffing_ncut_energy_rmse,
                j_abs_error=duffing_ncut_j_abs_error,
                zeta_abs_error=duffing_ncut_zeta_abs_error,
            )
        )
    if duffing_hilbert_qubit_dims.size > 0:
        summary.update(
            _summary_for_sweep(
                prefix="duffing_hilbert_qubit",
                values=duffing_hilbert_qubit_dims.astype(float),
                energy_rmse=duffing_hilbert_qubit_energy_rmse,
                j_abs_error=duffing_hilbert_qubit_j_abs_error,
                zeta_abs_error=duffing_hilbert_qubit_zeta_abs_error,
            )
        )
    if duffing_hilbert_coupler_dims.size > 0:
        summary.update(
            _summary_for_sweep(
                prefix="duffing_hilbert_coupler",
                values=duffing_hilbert_coupler_dims.astype(float),
                energy_rmse=duffing_hilbert_coupler_energy_rmse,
                j_abs_error=duffing_hilbert_coupler_j_abs_error,
                zeta_abs_error=duffing_hilbert_coupler_zeta_abs_error,
            )
        )
    return DuffingTruncationBenchmarkResult(
        flux_values=np.asarray(flux_values, dtype=float),
        sweep_target=str(config.static_benchmark.flux_control.sweep_target),
        duffing_calibration_mode=str(cfg.duffing_calibration_mode),
        duffing_truncated_dim=int(cfg.duffing_truncated_dim),
        lowest_excited_levels_compared=int(cfg.lowest_excited_levels_to_plot),
        reference_circuit_ncut=int(cfg.circuit_reference_ncut),
        reference_circuit_qubit_truncated_dim=int(cfg.circuit_reference_qubit_truncated_dim),
        reference_circuit_coupler_truncated_dim=int(cfg.circuit_reference_coupler_truncated_dim),
        reference_circuit_j_values=np.asarray(ref_j_values, dtype=float),
        reference_circuit_zeta_values=np.asarray(ref_zeta_values, dtype=float),
        duffing_ncut_values=np.asarray(duffing_ncut_values, dtype=int),
        duffing_ncut_effective_truncated_dim_values=np.asarray(duffing_ncut_effective_trunc_dim, dtype=int),
        duffing_ncut_energy_rmse=np.asarray(duffing_ncut_energy_rmse, dtype=float),
        duffing_ncut_j_abs_error=np.asarray(duffing_ncut_j_abs_error, dtype=float),
        duffing_ncut_zeta_abs_error=np.asarray(duffing_ncut_zeta_abs_error, dtype=float),
        duffing_hilbert_qubit_dim_values=np.asarray(duffing_hilbert_qubit_dims, dtype=int),
        duffing_hilbert_qubit_energy_rmse=np.asarray(duffing_hilbert_qubit_energy_rmse, dtype=float),
        duffing_hilbert_qubit_j_abs_error=np.asarray(duffing_hilbert_qubit_j_abs_error, dtype=float),
        duffing_hilbert_qubit_zeta_abs_error=np.asarray(duffing_hilbert_qubit_zeta_abs_error, dtype=float),
        duffing_hilbert_coupler_dim_values=np.asarray(duffing_hilbert_coupler_dims, dtype=int),
        duffing_hilbert_coupler_energy_rmse=np.asarray(duffing_hilbert_coupler_energy_rmse, dtype=float),
        duffing_hilbert_coupler_j_abs_error=np.asarray(duffing_hilbert_coupler_j_abs_error, dtype=float),
        duffing_hilbert_coupler_zeta_abs_error=np.asarray(duffing_hilbert_coupler_zeta_abs_error, dtype=float),
        summary=summary,
    )
