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
        q1=replace(config.system.q1, ncut=ncut_ref),
        q2=replace(config.system.q2, ncut=ncut_ref),
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
        nlevels_qubit=config.static_benchmark.circuit_model.hilbert_truncation.q1_truncated_dim,
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
    }

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
        circuit_reference_ncut=int(circuit_reference_ncut),
        circuit_j=float(circuit_j),
        circuit_zeta=float(circuit_zeta),
        summary=summary,
    )
