"""Runtime benchmark for CZ dynamics versus transmon charge cutoff ``ncut``."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from comparison.cz import run_cz_benchmark
from study_config import StudyConfig


@dataclass(frozen=True)
class RuntimeBenchmarkResult:
    sweep_target: str
    duffing_calibration_mode: str
    ramp_time_ns: float
    dt_ns: float
    fixed_hold_time_ns: float
    ncut_values: np.ndarray
    duffing_effective_truncated_dim_values: np.ndarray
    repeats: int
    selected_hold_times_ns: np.ndarray
    n_time_points: np.ndarray
    duffing_hilbert_dims: np.ndarray
    circuit_hilbert_dims: np.ndarray
    duffing_build_runtime_s: np.ndarray
    duffing_build_runtime_std_s: np.ndarray
    duffing_propagation_runtime_s: np.ndarray
    duffing_propagation_runtime_std_s: np.ndarray
    duffing_dynamics_runtime_s: np.ndarray
    duffing_dynamics_runtime_std_s: np.ndarray
    circuit_build_runtime_s: np.ndarray
    circuit_build_runtime_std_s: np.ndarray
    circuit_propagation_runtime_s: np.ndarray
    circuit_propagation_runtime_std_s: np.ndarray
    circuit_dynamics_runtime_s: np.ndarray
    circuit_dynamics_runtime_std_s: np.ndarray
    shared_static_precompute_runtime_s: np.ndarray
    shared_static_precompute_runtime_std_s: np.ndarray
    shared_hold_scan_runtime_s: np.ndarray
    shared_hold_scan_runtime_std_s: np.ndarray
    summary: dict[str, float]


def _config_with_ncut(
    config: StudyConfig,
    *,
    ncut: int,
    duffing_truncated_dim: int,
    duffing_calibration_mode: str,
) -> tuple[StudyConfig, int]:
    ncut_int = int(ncut)
    eff_trunc_dim = int(min(int(duffing_truncated_dim), 2 * ncut_int + 1))
    if eff_trunc_dim < 3:
        raise ValueError("Effective Duffing truncated dimension must be >= 3")

    system_cfg = replace(
        config.system,
        q0=replace(config.system.q0, ncut=ncut_int),
        q1=replace(config.system.q1, ncut=ncut_int),
    )
    static_cfg = replace(
        config.static_benchmark,
        duffing_model=replace(
            config.static_benchmark.duffing_model,
            transmon_spectral_extraction=replace(
                config.static_benchmark.duffing_model.transmon_spectral_extraction,
                ncut=ncut_int,
                truncated_dim=eff_trunc_dim,
            ),
            calibration_mode=str(duffing_calibration_mode),
        ),
    )
    return replace(config, system=system_cfg, static_benchmark=static_cfg), eff_trunc_dim


def _aggregate_samples(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"samples must be 2D, got {arr.shape}")
    median = np.median(arr, axis=1)
    std = np.std(arr, axis=1)
    return np.asarray(median, dtype=float), np.asarray(std, dtype=float)


def _resolve_fixed_hold_time_ns(
    config: StudyConfig,
    *,
    hold_time_ns: float | None,
) -> float:
    if hold_time_ns is not None:
        return float(hold_time_ns)

    cz_cfg = config.cz_benchmark
    if cz_cfg.hold_time_ns is not None:
        return float(cz_cfg.hold_time_ns)

    reference = run_cz_benchmark(
        config,
        ramp_time_ns=float(cz_cfg.ramp_time_ns),
        hold_time_ns=None,
        dt_ns=float(cz_cfg.dt_ns),
        enable_hold_time_scan=bool(cz_cfg.enable_hold_time_scan),
        scan_dt_ns=float(cz_cfg.scan_dt_ns),
        scan_max_hold_ns=float(cz_cfg.scan_max_hold_ns),
        scan_leakage_penalty=float(cz_cfg.scan_leakage_penalty),
    )
    return float(reference.summary["hold_time_ns"])


def run_runtime_benchmark(
    config: StudyConfig,
    *,
    ncut_values: list[int] | np.ndarray,
    duffing_truncated_dim: int,
    duffing_calibration_mode: str,
    repeats: int = 1,
    hold_time_ns: float | None = None,
) -> RuntimeBenchmarkResult:
    """Benchmark CZ runtime for Duffing and circuit models across ``ncut``."""
    ncuts = np.asarray(ncut_values, dtype=int).ravel()
    if ncuts.size == 0:
        raise ValueError("ncut_values must be non-empty")
    if np.any(ncuts < 1):
        raise ValueError("ncut_values must contain positive integers")

    repeats_int = int(repeats)
    if repeats_int < 1:
        raise ValueError("repeats must be >= 1")

    n_ncut = int(ncuts.size)
    duffing_trunc_dims_used = np.empty(n_ncut, dtype=int)

    hold_samples = np.empty((n_ncut, repeats_int), dtype=float)
    n_time_point_samples = np.empty((n_ncut, repeats_int), dtype=float)
    duffing_dim_samples = np.empty((n_ncut, repeats_int), dtype=float)
    circuit_dim_samples = np.empty((n_ncut, repeats_int), dtype=float)
    duffing_build_samples = np.empty((n_ncut, repeats_int), dtype=float)
    duffing_prop_samples = np.empty((n_ncut, repeats_int), dtype=float)
    duffing_dyn_samples = np.empty((n_ncut, repeats_int), dtype=float)
    circuit_build_samples = np.empty((n_ncut, repeats_int), dtype=float)
    circuit_prop_samples = np.empty((n_ncut, repeats_int), dtype=float)
    circuit_dyn_samples = np.empty((n_ncut, repeats_int), dtype=float)
    shared_static_samples = np.empty((n_ncut, repeats_int), dtype=float)
    shared_hold_scan_samples = np.empty((n_ncut, repeats_int), dtype=float)

    cz_cfg = config.cz_benchmark
    fixed_hold_time_ns = _resolve_fixed_hold_time_ns(config, hold_time_ns=hold_time_ns)
    for i, ncut in enumerate(ncuts):
        sweep_cfg, eff_trunc_dim = _config_with_ncut(
            config,
            ncut=int(ncut),
            duffing_truncated_dim=int(duffing_truncated_dim),
            duffing_calibration_mode=str(duffing_calibration_mode),
        )
        duffing_trunc_dims_used[i] = eff_trunc_dim
        for j in range(repeats_int):
            result = run_cz_benchmark(
                sweep_cfg,
                ramp_time_ns=float(cz_cfg.ramp_time_ns),
                hold_time_ns=float(fixed_hold_time_ns),
                dt_ns=float(cz_cfg.dt_ns),
                enable_hold_time_scan=False,
                scan_dt_ns=float(cz_cfg.scan_dt_ns),
                scan_max_hold_ns=float(cz_cfg.scan_max_hold_ns),
                scan_leakage_penalty=float(cz_cfg.scan_leakage_penalty),
            )
            summary = result.summary
            hold_samples[i, j] = float(summary["hold_time_ns"])
            n_time_point_samples[i, j] = float(summary["n_time_points"])
            duffing_dim_samples[i, j] = float(summary["duffing_hilbert_dim"])
            circuit_dim_samples[i, j] = float(summary["circuit_hilbert_dim"])
            duffing_build_samples[i, j] = float(summary["duffing_model_build_runtime_s"])
            duffing_prop_samples[i, j] = float(summary["duffing_propagation_runtime_s"])
            duffing_dyn_samples[i, j] = float(summary["duffing_dynamics_runtime_s"])
            circuit_build_samples[i, j] = float(summary["circuit_model_build_runtime_s"])
            circuit_prop_samples[i, j] = float(summary["circuit_propagation_runtime_s"])
            circuit_dyn_samples[i, j] = float(summary["circuit_dynamics_runtime_s"])
            shared_static_samples[i, j] = float(summary["shared_static_precompute_runtime_s"])
            shared_hold_scan_samples[i, j] = float(summary["shared_hold_scan_runtime_s"])

    hold_ns, hold_std_ns = _aggregate_samples(hold_samples)
    n_time_points, _ = _aggregate_samples(n_time_point_samples)
    duffing_dims, _ = _aggregate_samples(duffing_dim_samples)
    circuit_dims, _ = _aggregate_samples(circuit_dim_samples)
    duffing_build_runtime_s, duffing_build_runtime_std_s = _aggregate_samples(duffing_build_samples)
    duffing_propagation_runtime_s, duffing_propagation_runtime_std_s = _aggregate_samples(duffing_prop_samples)
    duffing_dynamics_runtime_s, duffing_dynamics_runtime_std_s = _aggregate_samples(duffing_dyn_samples)
    circuit_build_runtime_s, circuit_build_runtime_std_s = _aggregate_samples(circuit_build_samples)
    circuit_propagation_runtime_s, circuit_propagation_runtime_std_s = _aggregate_samples(circuit_prop_samples)
    circuit_dynamics_runtime_s, circuit_dynamics_runtime_std_s = _aggregate_samples(circuit_dyn_samples)
    shared_static_runtime_s, shared_static_runtime_std_s = _aggregate_samples(shared_static_samples)
    shared_hold_scan_runtime_s, shared_hold_scan_runtime_std_s = _aggregate_samples(shared_hold_scan_samples)

    summary = {
        "repeats": float(repeats_int),
        "ncut_min": float(np.min(ncuts)),
        "ncut_max": float(np.max(ncuts)),
        "ncut_count": float(n_ncut),
        "duffing_truncated_dim_configured": float(duffing_truncated_dim),
        "duffing_truncated_dim_used_min": float(np.min(duffing_trunc_dims_used)),
        "duffing_truncated_dim_used_max": float(np.max(duffing_trunc_dims_used)),
        "fixed_hold_time_ns": float(fixed_hold_time_ns),
        "selected_hold_time_ns_median_min": float(np.min(hold_ns)),
        "selected_hold_time_ns_median_max": float(np.max(hold_ns)),
        "selected_hold_time_ns_std_max": float(np.max(hold_std_ns)),
        "duffing_build_runtime_s_median_min": float(np.min(duffing_build_runtime_s)),
        "duffing_build_runtime_s_median_max": float(np.max(duffing_build_runtime_s)),
        "duffing_propagation_runtime_s_median_min": float(np.min(duffing_propagation_runtime_s)),
        "duffing_propagation_runtime_s_median_max": float(np.max(duffing_propagation_runtime_s)),
        "duffing_dynamics_runtime_s_median_min": float(np.min(duffing_dynamics_runtime_s)),
        "duffing_dynamics_runtime_s_median_max": float(np.max(duffing_dynamics_runtime_s)),
        "circuit_build_runtime_s_median_min": float(np.min(circuit_build_runtime_s)),
        "circuit_build_runtime_s_median_max": float(np.max(circuit_build_runtime_s)),
        "circuit_propagation_runtime_s_median_min": float(np.min(circuit_propagation_runtime_s)),
        "circuit_propagation_runtime_s_median_max": float(np.max(circuit_propagation_runtime_s)),
        "circuit_dynamics_runtime_s_median_min": float(np.min(circuit_dynamics_runtime_s)),
        "circuit_dynamics_runtime_s_median_max": float(np.max(circuit_dynamics_runtime_s)),
        "shared_static_precompute_runtime_s_median_min": float(np.min(shared_static_runtime_s)),
        "shared_static_precompute_runtime_s_median_max": float(np.max(shared_static_runtime_s)),
    }

    return RuntimeBenchmarkResult(
        sweep_target=str(config.static_benchmark.flux_control.sweep_target),
        duffing_calibration_mode=str(duffing_calibration_mode),
        ramp_time_ns=float(cz_cfg.ramp_time_ns),
        dt_ns=float(cz_cfg.dt_ns),
        fixed_hold_time_ns=float(fixed_hold_time_ns),
        ncut_values=np.asarray(ncuts, dtype=int),
        duffing_effective_truncated_dim_values=np.asarray(duffing_trunc_dims_used, dtype=int),
        repeats=repeats_int,
        selected_hold_times_ns=np.asarray(hold_ns, dtype=float),
        n_time_points=np.asarray(np.rint(n_time_points), dtype=int),
        duffing_hilbert_dims=np.asarray(np.rint(duffing_dims), dtype=int),
        circuit_hilbert_dims=np.asarray(np.rint(circuit_dims), dtype=int),
        duffing_build_runtime_s=np.asarray(duffing_build_runtime_s, dtype=float),
        duffing_build_runtime_std_s=np.asarray(duffing_build_runtime_std_s, dtype=float),
        duffing_propagation_runtime_s=np.asarray(duffing_propagation_runtime_s, dtype=float),
        duffing_propagation_runtime_std_s=np.asarray(duffing_propagation_runtime_std_s, dtype=float),
        duffing_dynamics_runtime_s=np.asarray(duffing_dynamics_runtime_s, dtype=float),
        duffing_dynamics_runtime_std_s=np.asarray(duffing_dynamics_runtime_std_s, dtype=float),
        circuit_build_runtime_s=np.asarray(circuit_build_runtime_s, dtype=float),
        circuit_build_runtime_std_s=np.asarray(circuit_build_runtime_std_s, dtype=float),
        circuit_propagation_runtime_s=np.asarray(circuit_propagation_runtime_s, dtype=float),
        circuit_propagation_runtime_std_s=np.asarray(circuit_propagation_runtime_std_s, dtype=float),
        circuit_dynamics_runtime_s=np.asarray(circuit_dynamics_runtime_s, dtype=float),
        circuit_dynamics_runtime_std_s=np.asarray(circuit_dynamics_runtime_std_s, dtype=float),
        shared_static_precompute_runtime_s=np.asarray(shared_static_runtime_s, dtype=float),
        shared_static_precompute_runtime_std_s=np.asarray(shared_static_runtime_std_s, dtype=float),
        shared_hold_scan_runtime_s=np.asarray(shared_hold_scan_runtime_s, dtype=float),
        shared_hold_scan_runtime_std_s=np.asarray(shared_hold_scan_runtime_std_s, dtype=float),
        summary=summary,
    )
