"""Fixed-flux truncation benchmark for static Duffing-vs-circuit metrics."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from models import (
    build_circuit_model_stack,
    build_dressed_effective_computational_stack,
    build_duffing_model_stack,
    extract_model1_parameters_from_4x4_stack,
)
from study_config import StudyConfig


@dataclass(frozen=True)
class TruncationBenchmarkResult:
    flux: float
    sweep_target: str
    duffing_calibration_mode: str
    duffing_truncated_dim: int
    duffing_ncut_values: np.ndarray
    duffing_effective_truncated_dim_values: np.ndarray
    duffing_j: np.ndarray
    duffing_zeta: np.ndarray
    duffing_lowest_relative_energies: np.ndarray
    circuit_lowest_relative_energies: np.ndarray
    max_duffing_ncut: int
    max_ncut_reported_excited_levels: np.ndarray
    duffing_minus_circuit_at_max_ncut: np.ndarray
    duffing_minus_circuit_percent_of_circuit_at_max_ncut: np.ndarray
    circuit_reference_ncut: int
    circuit_j: float
    circuit_zeta: float
    summary: dict[str, float]


def _default_flux(config: StudyConfig) -> float:
    sweep = config.static_benchmark.flux_sweep
    return 0.5 * (float(sweep.start) + float(sweep.stop))


def _extract_duffing_metrics(
    *,
    config: StudyConfig,
    flux: float,
    duffing_ncut: int,
    duffing_truncated_dim: int,
    duffing_calibration_mode: str,
) -> tuple[float, float, np.ndarray, int]:
    dcfg = config.static_benchmark.duffing_model
    ncut = int(duffing_ncut)
    trunc_dim_cfg = int(duffing_truncated_dim)
    # scqubits requires truncated_dim <= (2*ncut+1) for the charge-basis diagonalization.
    trunc_dim_eff = int(min(trunc_dim_cfg, 2 * ncut + 1))
    if trunc_dim_eff < 3:
        raise ValueError("Effective Duffing transmon truncated_dim must be >= 3")

    dcfg_for_ncut = replace(
        dcfg,
        transmon_spectral_extraction=replace(
            dcfg.transmon_spectral_extraction,
            ncut=ncut,
            truncated_dim=trunc_dim_eff,
        ),
        calibration_mode=str(duffing_calibration_mode),
    )
    H_duf = build_duffing_model_stack(
        flux_values=np.array([float(flux)], dtype=float),
        system_params=config.system,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        duffing_config=dcfg_for_ncut,
        sweep_target=config.static_benchmark.flux_control.sweep_target,
    ).hamiltonian_stack

    H_duf_eff = build_dressed_effective_computational_stack(
        H_duf,
        nlevels_qubit=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_qubit,
        nlevels_coupler=config.static_benchmark.duffing_model.hilbert_truncation.nlevels_coupler,
        n_candidate_states=config.static_benchmark.dressed_subspace.n_candidate_states,
        selection_mode=config.static_benchmark.dressed_subspace.selection_mode,
    )
    params = extract_model1_parameters_from_4x4_stack(H_duf_eff)
    evals = np.linalg.eigvalsh(np.asarray(H_duf[0], dtype=complex))
    rel_e = np.asarray(evals - evals[0], dtype=float)
    return float(params["J"][0]), float(params["zeta"][0]), rel_e, trunc_dim_eff


def _extract_circuit_metrics(
    *,
    config: StudyConfig,
    flux: float,
    circuit_reference_ncut: int,
) -> tuple[float, float, np.ndarray]:
    ncut_ref = int(circuit_reference_ncut)
    system_ref = replace(
        config.system,
        q0=replace(config.system.q0, ncut=ncut_ref),
        q1=replace(config.system.q1, ncut=ncut_ref),
    )

    H_cir = build_circuit_model_stack(
        flux_values=np.array([float(flux)], dtype=float),
        system_params=system_ref,
        coupler_frequency=config.static_benchmark.coupler_frequency,
        circuit_config=config.static_benchmark.circuit_model,
        sweep_target=config.static_benchmark.flux_control.sweep_target,
    ).hamiltonian_stack

    H_cir_eff = build_dressed_effective_computational_stack(
        H_cir,
        nlevels_qubit=config.static_benchmark.circuit_model.hilbert_truncation.q0_truncated_dim,
        nlevels_coupler=config.static_benchmark.circuit_model.hilbert_truncation.c_truncated_dim,
        n_candidate_states=config.static_benchmark.dressed_subspace.n_candidate_states,
        selection_mode=config.static_benchmark.dressed_subspace.selection_mode,
    )
    params = extract_model1_parameters_from_4x4_stack(H_cir_eff)
    evals = np.linalg.eigvalsh(np.asarray(H_cir[0], dtype=complex))
    rel_e = np.asarray(evals - evals[0], dtype=float)
    return float(params["J"][0]), float(params["zeta"][0]), rel_e


def run_truncation_benchmark(
    config: StudyConfig,
    *,
    duffing_ncut_values: list[int] | np.ndarray,
    fixed_flux: float | None = None,
    duffing_truncated_dim: int | None = None,
    lowest_excited_levels_to_report: int | None = None,
    reported_excited_levels: list[int] | np.ndarray | None = None,
    circuit_reference_ncut: int = 120,
    duffing_calibration_mode: str = "per-flux",
) -> TruncationBenchmarkResult:
    """Compare Duffing ``J, zeta`` convergence vs a fixed circuit reference.

    Notes
    -----
    - Uses one fixed flux point.
    - Sweeps Duffing transmon calibration ``ncut``.
    - Uses a single circuit point evaluated at a large reference ``ncut``;
      the circuit values are shown as horizontal reference lines in plotting.
    """
    flux = float(_default_flux(config) if fixed_flux is None else fixed_flux)

    ncuts = np.asarray(duffing_ncut_values, dtype=int).ravel()
    if ncuts.size == 0:
        raise ValueError("duffing_ncut_values must be non-empty")
    if np.any(ncuts < 1):
        raise ValueError("duffing_ncut_values must be positive integers")
    trunc_dim_cfg = int(
        config.static_benchmark.duffing_model.transmon_spectral_extraction.truncated_dim
        if duffing_truncated_dim is None
        else duffing_truncated_dim
    )
    if trunc_dim_cfg < 3:
        raise ValueError("duffing_truncated_dim must be >= 3")
    if reported_excited_levels is None:
        n_report_cfg = int(
            config.truncation_benchmark.lowest_excited_levels_to_plot
            if lowest_excited_levels_to_report is None
            else lowest_excited_levels_to_report
        )
        if n_report_cfg < 1:
            raise ValueError("lowest_excited_levels_to_report must be >= 1")
    else:
        raw_levels = np.asarray(reported_excited_levels, dtype=int).ravel()
        if raw_levels.size == 0:
            raise ValueError("reported_excited_levels must be non-empty")
        if np.any(raw_levels < 1):
            raise ValueError("reported_excited_levels must contain positive integers")
        ordered_unique_levels = list(dict.fromkeys(int(level) for level in raw_levels))

    j_vals = np.empty(ncuts.shape[0], dtype=float)
    zeta_vals = np.empty(ncuts.shape[0], dtype=float)
    trunc_dims_used = np.empty(ncuts.shape[0], dtype=int)
    rel_levels_duffing: list[np.ndarray] = []
    for k, ncut in enumerate(ncuts):
        j_vals[k], zeta_vals[k], rel_e, trunc_dims_used[k] = _extract_duffing_metrics(
            config=config,
            flux=flux,
            duffing_ncut=int(ncut),
            duffing_truncated_dim=trunc_dim_cfg,
            duffing_calibration_mode=duffing_calibration_mode,
        )
        rel_levels_duffing.append(np.asarray(rel_e, dtype=float))

    circuit_j, circuit_zeta, rel_levels_circuit = _extract_circuit_metrics(
        config=config,
        flux=flux,
        circuit_reference_ncut=int(circuit_reference_ncut),
    )
    n_low = int(
        min(
            rel_levels_circuit.shape[0],
            min(levels.shape[0] for levels in rel_levels_duffing),
        )
    )
    duffing_low = np.stack([levels[:n_low] for levels in rel_levels_duffing], axis=0)
    circuit_low = np.asarray(rel_levels_circuit[:n_low], dtype=float)
    max_ncut = int(np.max(ncuts))
    max_ncut_idx = int(np.flatnonzero(ncuts == max_ncut)[-1])
    max_excited_available = int(max(0, n_low - 1))
    if reported_excited_levels is None:
        n_report = int(min(n_report_cfg, max_excited_available))
        reported_levels = np.arange(1, 1 + n_report, dtype=int)
    else:
        reported_levels = np.asarray(
            [level for level in ordered_unique_levels if level <= max_excited_available],
            dtype=int,
        )
        n_report = int(reported_levels.size)
    max_ncut_diff = (
        np.asarray(duffing_low[max_ncut_idx, reported_levels] - circuit_low[reported_levels], dtype=float)
        if n_report > 0
        else np.zeros(0, dtype=float)
    )
    if n_report > 0:
        circuit_ref_levels = np.asarray(circuit_low[reported_levels], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            max_ncut_diff_percent = 100.0 * (max_ncut_diff / circuit_ref_levels)
        # If a circuit reference level is effectively zero, percent is undefined.
        near_zero = np.abs(circuit_ref_levels) < 1e-15
        max_ncut_diff_percent = np.where(near_zero, np.nan, max_ncut_diff_percent)
        max_ncut_diff_percent = np.asarray(max_ncut_diff_percent, dtype=float)
    else:
        max_ncut_diff_percent = np.zeros(0, dtype=float)

    err_j = j_vals - circuit_j
    err_zeta = zeta_vals - circuit_zeta
    summary = {
        "flux": flux,
        "duffing_truncated_dim_configured": float(trunc_dim_cfg),
        "duffing_truncated_dim_used_min": float(np.min(trunc_dims_used)),
        "duffing_truncated_dim_used_max": float(np.max(trunc_dims_used)),
        "circuit_reference_ncut": float(circuit_reference_ncut),
        "duffing_j_rmse_vs_circuit": float(np.sqrt(np.mean(err_j * err_j))),
        "duffing_j_max_abs_vs_circuit": float(np.max(np.abs(err_j))),
        "duffing_zeta_rmse_vs_circuit": float(np.sqrt(np.mean(err_zeta * err_zeta))),
        "duffing_zeta_max_abs_vs_circuit": float(np.max(np.abs(err_zeta))),
        "lowest_levels_count": float(n_low),
        "max_duffing_ncut": float(max_ncut),
        "reported_excited_levels_count": float(n_report),
    }
    for level, diff in zip(reported_levels, max_ncut_diff):
        summary[f"duffing_minus_circuit_E{int(level)}_at_max_ncut"] = float(diff)
    for level, rel_pct in zip(reported_levels, max_ncut_diff_percent):
        summary[f"duffing_minus_circuit_E{int(level)}_percent_of_circuit_at_max_ncut"] = float(rel_pct)

    return TruncationBenchmarkResult(
        flux=flux,
        sweep_target=str(config.static_benchmark.flux_control.sweep_target),
        duffing_calibration_mode=str(duffing_calibration_mode),
        duffing_truncated_dim=trunc_dim_cfg,
        duffing_ncut_values=np.asarray(ncuts, dtype=int),
        duffing_effective_truncated_dim_values=np.asarray(trunc_dims_used, dtype=int),
        duffing_j=np.asarray(j_vals, dtype=float),
        duffing_zeta=np.asarray(zeta_vals, dtype=float),
        duffing_lowest_relative_energies=np.asarray(duffing_low, dtype=float),
        circuit_lowest_relative_energies=np.asarray(circuit_low, dtype=float),
        max_duffing_ncut=max_ncut,
        max_ncut_reported_excited_levels=np.asarray(reported_levels, dtype=int),
        duffing_minus_circuit_at_max_ncut=np.asarray(max_ncut_diff, dtype=float),
        duffing_minus_circuit_percent_of_circuit_at_max_ncut=np.asarray(max_ncut_diff_percent, dtype=float),
        circuit_reference_ncut=int(circuit_reference_ncut),
        circuit_j=float(circuit_j),
        circuit_zeta=float(circuit_zeta),
        summary=summary,
    )
