"""Runtime benchmark for CZ dynamics versus propagated qubit truncation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np

from comparison.cz import run_cz_benchmark
from comparison.static import run_static_benchmark
from study_config import StudyConfig


@dataclass(frozen=True)
class RuntimeBenchmarkResult:
    sweep_target: str
    duffing_calibration_mode: str
    ramp_time_ns: float
    dt_ns: float
    fixed_hold_time_ns: float
    qubit_truncation_values: np.ndarray
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


def _config_with_qubit_truncation(
    config: StudyConfig,
    *,
    qubit_truncation: int,
    duffing_calibration_mode: str,
) -> StudyConfig:
    trunc_int = int(qubit_truncation)
    max_circuit_trunc = min(
        2 * int(config.system.q0.ncut) + 1,
        2 * int(config.system.q1.ncut) + 1,
    )
    if trunc_int > max_circuit_trunc:
        raise ValueError(
            "qubit_truncation exceeds the circuit transmon basis available from "
            f"system ncut; got {trunc_int}, max supported is {max_circuit_trunc}"
        )
    static_cfg = replace(
        config.static_benchmark,
        duffing_model=replace(
            config.static_benchmark.duffing_model,
            hilbert_truncation=replace(
                config.static_benchmark.duffing_model.hilbert_truncation,
                nlevels_qubit=trunc_int,
            ),
            calibration_mode=str(duffing_calibration_mode),
        ),
        circuit_model=replace(
            config.static_benchmark.circuit_model,
            hilbert_truncation=replace(
                config.static_benchmark.circuit_model.hilbert_truncation,
                q0_truncated_dim=trunc_int,
                q1_truncated_dim=trunc_int,
            ),
        ),
    )
    return replace(config, static_benchmark=static_cfg)


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
    qubit_truncation_values: list[int] | np.ndarray,
    duffing_calibration_mode: str,
    repeats: int = 1,
    hold_time_ns: float | None = None,
) -> RuntimeBenchmarkResult:
    """Benchmark CZ runtime for Duffing and circuit models across qubit truncation."""
    trunc_values = np.asarray(qubit_truncation_values, dtype=int).ravel()
    if trunc_values.size == 0:
        raise ValueError("qubit_truncation_values must be non-empty")
    if np.any(trunc_values < 2):
        raise ValueError("qubit_truncation_values must contain integers >= 2")

    repeats_int = int(repeats)
    if repeats_int < 1:
        raise ValueError("repeats must be >= 1")

    n_trunc = int(trunc_values.size)

    hold_samples = np.empty((n_trunc, repeats_int), dtype=float)
    n_time_point_samples = np.empty((n_trunc, repeats_int), dtype=float)
    duffing_dim_samples = np.empty((n_trunc, repeats_int), dtype=float)
    circuit_dim_samples = np.empty((n_trunc, repeats_int), dtype=float)
    duffing_build_samples = np.empty((n_trunc, repeats_int), dtype=float)
    duffing_prop_samples = np.empty((n_trunc, repeats_int), dtype=float)
    duffing_dyn_samples = np.empty((n_trunc, repeats_int), dtype=float)
    circuit_build_samples = np.empty((n_trunc, repeats_int), dtype=float)
    circuit_prop_samples = np.empty((n_trunc, repeats_int), dtype=float)
    circuit_dyn_samples = np.empty((n_trunc, repeats_int), dtype=float)
    shared_static_samples = np.empty((n_trunc, repeats_int), dtype=float)
    shared_hold_scan_samples = np.empty((n_trunc, repeats_int), dtype=float)

    cz_cfg = config.cz_benchmark
    print(f"--- Resolving fixed hold time ---")
    fixed_hold_time_ns = _resolve_fixed_hold_time_ns(config, hold_time_ns=hold_time_ns)
    print(f"--- Resolving fixed hold time Done---")
    sweep_configs: list[StudyConfig] = []
    static_results: list[object] = []
    static_runtimes_s = np.empty(n_trunc, dtype=float)
    for i, qubit_truncation in enumerate(trunc_values):
        sweep_cfg = _config_with_qubit_truncation(
            config,
            qubit_truncation=int(qubit_truncation),
            duffing_calibration_mode=str(duffing_calibration_mode),
        )
        sweep_configs.append(sweep_cfg)
        print(f"--- run_static_benchmark for truncation={i} Starting ---")
        static_started = time.perf_counter()
        static_results.append(run_static_benchmark(sweep_cfg))
        static_runtimes_s[i] = float(time.perf_counter() - static_started)
        print(f"--- run_static_benchmark for truncation={i} Finished in {static_runtimes_s[i]}s ---")

    if sweep_configs:
        # Prime lazy imports and BLAS/Qutip solver setup outside the measured sweep
        # so the first truncation value does not inherit all cold-start overhead.
        print(f"--- Propagation outside measurement ---")
        run_cz_benchmark(
            sweep_configs[0],
            ramp_time_ns=float(cz_cfg.ramp_time_ns),
            hold_time_ns=float(fixed_hold_time_ns),
            dt_ns=float(cz_cfg.dt_ns),
            enable_hold_time_scan=False,
            scan_dt_ns=float(cz_cfg.scan_dt_ns),
            scan_max_hold_ns=float(cz_cfg.scan_max_hold_ns),
            scan_leakage_penalty=float(cz_cfg.scan_leakage_penalty),
            precomputed_static_result=static_results[0],
            precomputed_static_runtime_s=float(static_runtimes_s[0]),
        )

    for j in range(repeats_int):
        print(f"--- Repeat {j=} ---")
        order = range(n_trunc) if (j % 2 == 0) else range(n_trunc - 1, -1, -1)
        for i in order:
            sweep_cfg = sweep_configs[i]
            result = run_cz_benchmark(
                sweep_cfg,
                ramp_time_ns=float(cz_cfg.ramp_time_ns),
                hold_time_ns=float(fixed_hold_time_ns),
                dt_ns=float(cz_cfg.dt_ns),
                enable_hold_time_scan=False,
                scan_dt_ns=float(cz_cfg.scan_dt_ns),
                scan_max_hold_ns=float(cz_cfg.scan_max_hold_ns),
                scan_leakage_penalty=float(cz_cfg.scan_leakage_penalty),
                precomputed_static_result=static_results[i],
                precomputed_static_runtime_s=float(static_runtimes_s[i]),
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
        "qubit_truncation_min": float(np.min(trunc_values)),
        "qubit_truncation_max": float(np.max(trunc_values)),
        "qubit_truncation_count": float(n_trunc),
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
        qubit_truncation_values=np.asarray(trunc_values, dtype=int),
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
