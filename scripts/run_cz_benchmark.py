"""Run CZ benchmark with parameters loaded from /params."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_results_io import (
    default_results_path_for_figure,
    load_result_hdf5,
    save_result_hdf5,
)
from benchmark_cli_reporting import CliReporter, build_common_truncation_lines
from comparison.cz import CzBenchmarkResult, run_cz_benchmark
from plotting.cz import plot_cz_benchmark
from study_config import load_study_config



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to HDF5 results file (default: figure path with .h5 suffix).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmark computation and plot from an existing HDF5 results file.",
    )
    return parser.parse_args()


def _resolve_repo_relative(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def main() -> None:
    args = _parse_args()
    repo_root = _REPO_ROOT
    reporter = CliReporter(benchmark_name="cz", script_name=Path(__file__).name)
    config = load_study_config(
        repo_root / "params" / "system_params.json",
        repo_root / "params" / "benchmark_params.json",
    )
    cz_cfg = config.cz_benchmark
    ramp_time_ns = float(cz_cfg.ramp_time_ns)
    dt_ns = float(cz_cfg.dt_ns)
    hold_time_ns = None if cz_cfg.hold_time_ns is None else float(cz_cfg.hold_time_ns)
    enable_hold_time_scan = bool(cz_cfg.enable_hold_time_scan)

    figure_path = repo_root / cz_cfg.outputs.figure
    results_path = (
        _resolve_repo_relative(repo_root, args.results)
        if args.results is not None
        else default_results_path_for_figure(figure_path)
    )

    if args.plot_only:
        result = load_result_hdf5(
            results_path,
            CzBenchmarkResult,
            expected_benchmark_name="cz",
        )
    else:
        result = run_cz_benchmark(
            config,
            ramp_time_ns=ramp_time_ns,
            hold_time_ns=hold_time_ns,
            dt_ns=dt_ns,
            enable_hold_time_scan=enable_hold_time_scan,
            scan_dt_ns=float(cz_cfg.scan_dt_ns),
            scan_max_hold_ns=float(cz_cfg.scan_max_hold_ns),
            scan_leakage_penalty=float(cz_cfg.scan_leakage_penalty),
        )
        save_result_hdf5(result, results_path, benchmark_name="cz")

    title = "CZ Benchmark: Flux And CPhase"
    plot_cz_benchmark(result, figure_path, title)

    for line in build_common_truncation_lines(config):
        reporter.line(line)
    reporter.line("CZ benchmark summary:")
    cz_summary_keys = [
        "effective_final_conditional_phase_rad",
        "duffing_final_conditional_phase_rad",
        "circuit_final_conditional_phase_rad",
        "circuit_final_phase_error_to_pi_rad",
        "effective_final_phase_error_vs_circuit_rad",
        "duffing_final_phase_error_vs_circuit_rad",
        "effective_populations_rmse_vs_circuit",
        "duffing_populations_rmse_vs_circuit",
        "ramp_time_ns",
        "hold_time_ns",
        "dt_ns",
        "n_time_points",
        "effective_hilbert_dim",
        "duffing_hilbert_dim",
        "circuit_hilbert_dim",
        "shared_static_precompute_runtime_s",
        "shared_hold_scan_runtime_s",
        "effective_model_build_runtime_s",
        "duffing_model_build_runtime_s",
        "circuit_model_build_runtime_s",
        "effective_propagation_runtime_s",
        "duffing_propagation_runtime_s",
        "circuit_propagation_runtime_s",
        "effective_dynamics_runtime_s",
        "duffing_dynamics_runtime_s",
        "circuit_dynamics_runtime_s",
    ]
    for key in cz_summary_keys:
        if key in result.summary:
            reporter.line(f"  {key}: {result.summary[key]:.6e}")
    reporter.line(
        "Selected pulse: "
        f"sweep_target={result.sweep_target}, idle_flux={result.idle_flux:.6f}, "
        f"target_flux={result.target_flux:.6f}, ramp_time_ns={result.ramp_time_ns:.3f}, "
        f"hold_time_ns={result.hold_time_ns:.3f}, dt_ns={result.dt_ns:.3f}"
    )
    if result.scan_hold_times_ns.size > 0:
        reporter.line("Hold scan (ns, phase_err_to_pi_rad, score):")
        for h, err, score in zip(result.scan_hold_times_ns, result.scan_phase_error_rad, result.scan_scores):
            reporter.line(f"  {h:.6f}, {err:.6e}, {score:.6e}")
    if args.plot_only:
        reporter.line(f"Loaded results: {results_path}")
    else:
        reporter.line(f"Wrote results: {results_path}")
    reporter.line(f"Wrote figure: {figure_path}")
    reporter.add_runtime_line()
    reporter.persist(results_path)


if __name__ == "__main__":
    main()
